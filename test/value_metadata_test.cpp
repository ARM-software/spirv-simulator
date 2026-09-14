#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class ComputeBuiltinProvenanceTests : public ::SPIRVSimulator::SPIRVSimulator,
                                      public Test
{
  protected:
    struct ExpectedRange
    {
        uint64_t min;
        uint64_t max;
        uint8_t  workgroup_mask;
        uint8_t  local_mask;
    };

    static constexpr uint32_t kBuiltinVariableId = 1;
    static constexpr uint32_t kResultId = 2;

    void SetUp() override
    {
        simulation_data_ = &input_data_;
        simulation_results_ = &simulation_results_storage_;
        persistent_data_ = nullptr;
        memory_flag_tracker_ = nullptr;
        verbose_ = false;
        flags_ = 0;

        input_data_.has_compute_num_workgroups = true;
        input_data_.compute_num_workgroups = { 5, 7, 11 };
        input_data_.base_workgroups = { 2, 3, 4 };
        active_compute_local_size_ = { 4, 6, 8, true };

        types_[CommonTypes::u32] = ::SPIRVSimulator::Type::Int(32, false);
        types_[CommonTypes::uvec3] =
            ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 3);
        values_.resize(kResultId + 1);
        value_meta_.resize(kResultId + 1);
    }

    bool ProduceBuiltin(spv::BuiltIn builtin,
                        uint32_t type_id,
                        std::vector<uint32_t> component_path = {})
    {
        values_[kResultId] = {};
        value_meta_[kResultId] = {};
        decorators_[kBuiltinVariableId] = {
            ::SPIRVSimulator::DecorationInfo{
                .kind = spv::Decoration::DecorationBuiltIn,
                .literals = { static_cast<uint32_t>(builtin) },
            },
        };

        ::SPIRVSimulator::PointerV pointer = {};
        pointer.base_result_id = kBuiltinVariableId;
        pointer.storage_class = spv::StorageClass::StorageClassInput;
        pointer.idx_path = std::move(component_path);
        return TrySetComputeBuiltinValueAndMetadata(kResultId, pointer, type_id);
    }

    void ExpectRange(const ::SPIRVSimulator::DenseValueRangeInfo& range,
                     const ExpectedRange& expected) const
    {
        EXPECT_TRUE(range.valid);
        EXPECT_TRUE(range.thread_dependent);
        EXPECT_TRUE(range.dense_range);
        EXPECT_EQ(range.min, expected.min);
        EXPECT_EQ(range.max, expected.max);
        EXPECT_EQ(range.stride, 1u);
        EXPECT_EQ(range.invocation_linearization.workgroup_id_component_mask,
                  expected.workgroup_mask);
        EXPECT_EQ(range.invocation_linearization.local_invocation_id_component_mask,
                  expected.local_mask);
    }

    void ExpectVectorAndScalarComponentRanges(
        spv::BuiltIn builtin,
        const std::array<ExpectedRange, 3>& expected_ranges)
    {
        ASSERT_TRUE(ProduceBuiltin(builtin, CommonTypes::uvec3));
        const ::SPIRVSimulator::ValueMetadata& vector_meta = value_meta_[kResultId];
        EXPECT_NE(vector_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
        EXPECT_EQ(vector_meta.invocation_dependence,
                  ::SPIRVSimulator::InvocationDependence::Varying);
        ASSERT_EQ(vector_meta.component_invocation_dependence.size(),
                  expected_ranges.size());
        EXPECT_TRUE(vector_meta.subobject_invocation_dependence.empty());
        ASSERT_EQ(vector_meta.component_ranges.size(), expected_ranges.size());
        for (uint32_t component = 0; component < expected_ranges.size(); ++component)
        {
            EXPECT_EQ(vector_meta.component_invocation_dependence[component],
                      ::SPIRVSimulator::InvocationDependence::Varying);
            ExpectRange(vector_meta.component_ranges[component],
                        expected_ranges[component]);
        }
        EXPECT_EQ(
            vector_meta.value_range.invocation_linearization.workgroup_id_component_mask,
            0u);
        EXPECT_EQ(
            vector_meta.value_range.invocation_linearization.local_invocation_id_component_mask,
            0u);

        for (uint32_t component = 0; component < expected_ranges.size(); ++component)
        {
            ASSERT_TRUE(ProduceBuiltin(builtin, CommonTypes::u32, { component }));
            const ::SPIRVSimulator::ValueMetadata& scalar_meta = value_meta_[kResultId];
            EXPECT_NE(scalar_meta.flags & SPS_FLAG_THREAD_SPECIFIC,
                      0u);
            EXPECT_EQ(scalar_meta.invocation_dependence,
                      ::SPIRVSimulator::InvocationDependence::Varying);
            ExpectRange(scalar_meta.value_range, expected_ranges[component]);
            EXPECT_TRUE(scalar_meta.component_ranges.empty());
            EXPECT_TRUE(scalar_meta.component_invocation_dependence.empty());
            EXPECT_TRUE(scalar_meta.subobject_invocation_dependence.empty());
        }
    }

    ::SPIRVSimulator::SimulationData input_data_;
    ::SPIRVSimulator::SimulationResults simulation_results_storage_;
};

TEST_F(ComputeBuiltinProvenanceTests,
       WorkgroupIdSetsOnlyWorkgroupComponentProvenance)
{
    ExpectVectorAndScalarComponentRanges(
        spv::BuiltIn::BuiltInWorkgroupId,
        {
            ExpectedRange{ 2, 6, ::SPIRVSimulator::kInvocationComponentX, 0 },
            ExpectedRange{ 3, 9, ::SPIRVSimulator::kInvocationComponentY, 0 },
            ExpectedRange{ 4, 14, ::SPIRVSimulator::kInvocationComponentZ, 0 },
        });
}

TEST_F(ComputeBuiltinProvenanceTests,
       LocalInvocationIdSetsOnlyLocalComponentProvenance)
{
    ExpectVectorAndScalarComponentRanges(
        spv::BuiltIn::BuiltInLocalInvocationId,
        {
            ExpectedRange{ 0, 3, 0, ::SPIRVSimulator::kInvocationComponentX },
            ExpectedRange{ 0, 5, 0, ::SPIRVSimulator::kInvocationComponentY },
            ExpectedRange{ 0, 7, 0, ::SPIRVSimulator::kInvocationComponentZ },
        });
}

TEST_F(ComputeBuiltinProvenanceTests,
       GlobalInvocationIdSetsBothComponentProvenances)
{
    ExpectVectorAndScalarComponentRanges(
        spv::BuiltIn::BuiltInGlobalInvocationId,
        {
            ExpectedRange{ 8,
                           27,
                           ::SPIRVSimulator::kInvocationComponentX,
                           ::SPIRVSimulator::kInvocationComponentX },
            ExpectedRange{ 18,
                           59,
                           ::SPIRVSimulator::kInvocationComponentY,
                           ::SPIRVSimulator::kInvocationComponentY },
            ExpectedRange{ 32,
                           119,
                           ::SPIRVSimulator::kInvocationComponentZ,
                           ::SPIRVSimulator::kInvocationComponentZ },
        });
}

TEST_F(ComputeBuiltinProvenanceTests,
       LocalInvocationIndexUsesAllLocalComponents)
{
    ASSERT_TRUE(ProduceBuiltin(spv::BuiltIn::BuiltInLocalInvocationIndex,
                               CommonTypes::u32));
    ExpectRange(
        value_meta_[kResultId].value_range,
        ExpectedRange{ 0,
                       191,
                       0,
                       ::SPIRVSimulator::kInvocationComponentX |
                           ::SPIRVSimulator::kInvocationComponentY |
                           ::SPIRVSimulator::kInvocationComponentZ });
    EXPECT_EQ(value_meta_[kResultId].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_TRUE(value_meta_[kResultId].component_invocation_dependence.empty());
    EXPECT_TRUE(value_meta_[kResultId].subobject_invocation_dependence.empty());
}

TEST_F(ComputeBuiltinProvenanceTests,
       NumWorkgroupsProducesInvariantVectorAndScalarComponents)
{
    constexpr std::array<uint64_t, 3> expected_values = { 5, 7, 11 };

    ASSERT_TRUE(ProduceBuiltin(spv::BuiltIn::BuiltInNumWorkgroups,
                               CommonTypes::uvec3));
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(
        values_[kResultId]));
    const std::shared_ptr<::SPIRVSimulator::VectorV>& vector_value =
        std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kResultId]);
    ASSERT_NE(vector_value, nullptr);
    ASSERT_EQ(vector_value->elems.size(), expected_values.size());
    for (size_t component = 0; component < expected_values.size(); ++component)
    {
        ASSERT_TRUE(std::holds_alternative<uint64_t>(vector_value->elems[component]));
        EXPECT_EQ(std::get<uint64_t>(vector_value->elems[component]),
                  expected_values[component]);
    }

    const ::SPIRVSimulator::ValueMetadata& vector_metadata = value_meta_[kResultId];
    EXPECT_EQ(vector_metadata.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(vector_metadata.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
    EXPECT_FALSE(vector_metadata.value_range.valid);
    EXPECT_TRUE(vector_metadata.component_ranges.empty());
    ASSERT_EQ(vector_metadata.component_invocation_dependence.size(),
              expected_values.size());
    EXPECT_TRUE(vector_metadata.subobject_invocation_dependence.empty());
    for (uint32_t component = 0; component < expected_values.size(); ++component)
    {
        EXPECT_EQ(vector_metadata.component_invocation_dependence[component],
                  ::SPIRVSimulator::InvocationDependence::Invariant);
    }

    for (uint32_t component = 0; component < expected_values.size(); ++component)
    {
        ASSERT_TRUE(ProduceBuiltin(spv::BuiltIn::BuiltInNumWorkgroups,
                                   CommonTypes::u32,
                                   { component }));
        ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kResultId]));
        EXPECT_EQ(std::get<uint64_t>(values_[kResultId]), expected_values[component]);

        const ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[kResultId];
        EXPECT_EQ(metadata.invocation_dependence,
                  ::SPIRVSimulator::InvocationDependence::Invariant);
        EXPECT_EQ(metadata.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
        EXPECT_FALSE(metadata.value_range.valid);
        EXPECT_TRUE(metadata.component_invocation_dependence.empty());
        EXPECT_TRUE(metadata.subobject_invocation_dependence.empty());
    }
}

TEST_F(ComputeBuiltinProvenanceTests,
       BuiltinFamiliesRequireOnlyTheirOwnDispatchInputs)
{
    input_data_.has_compute_num_workgroups = false;

    EXPECT_FALSE(ProduceBuiltin(spv::BuiltIn::BuiltInNumWorkgroups,
                                CommonTypes::u32,
                                { 0 }));
    EXPECT_FALSE(ProduceBuiltin(spv::BuiltIn::BuiltInWorkgroupId,
                                CommonTypes::uvec3));
    EXPECT_FALSE(ProduceBuiltin(spv::BuiltIn::BuiltInGlobalInvocationId,
                                CommonTypes::uvec3));
    EXPECT_TRUE(ProduceBuiltin(spv::BuiltIn::BuiltInLocalInvocationId,
                               CommonTypes::uvec3));
    EXPECT_TRUE(ProduceBuiltin(spv::BuiltIn::BuiltInLocalInvocationIndex,
                               CommonTypes::u32));
}

TEST_F(ComputeBuiltinProvenanceTests,
       OpLoadKeepsComputeBuiltinProducerAuthoritative)
{
    decorators_[kBuiltinVariableId] = {
        ::SPIRVSimulator::DecorationInfo{
            .kind = spv::Decoration::DecorationBuiltIn,
            .literals = { static_cast<uint32_t>(spv::BuiltIn::BuiltInNumWorkgroups) },
        },
    };
    values_[kBuiltinVariableId] = ::SPIRVSimulator::PointerV{
        1,
        0,
        CommonTypes::uvec3,
        kBuiltinVariableId,
        spv::StorageClass::StorageClassInput,
        {},
    };
    value_meta_[kBuiltinVariableId].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;

    const std::vector<uint32_t> words{
        0,
        CommonTypes::uvec3,
        kResultId,
        kBuiltinVariableId,
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpLoad,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };

    Op_Load(instruction);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<::SPIRVSimulator::InvocationDependence>(
                  3, ::SPIRVSimulator::InvocationDependence::Invariant)));
    EXPECT_EQ(result_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
}

class ExternalLoadInvocationDependenceTests
    : public ::SPIRVSimulator::SPIRVSimulator,
      public Test
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kPointerId = 300;
    static constexpr uint32_t kResultId  = 301;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;
        values_.resize(kResultId + 1);
        value_meta_.resize(kResultId + 1);
        values_[kPointerId] = ::SPIRVSimulator::PointerV{
            1,
            0,
            CommonTypes::u64,
            0,
            spv::StorageClass::StorageClassPhysicalStorageBuffer,
            {},
        };
    }

    std::optional<::SPIRVSimulator::Value> ReadPointer(
        const ::SPIRVSimulator::PointerV&) override
    {
        return loaded_value_;
    }

    void ExecuteLoad(uint32_t type_id)
    {
        const std::vector<uint32_t> words{
            0,
            type_id,
            kResultId,
            kPointerId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpLoad,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        Op_Load(instruction);
    }

    ::SPIRVSimulator::Value loaded_value_ = uint64_t{ 17 };
};

TEST_F(ExternalLoadInvocationDependenceTests,
       ScalarPropagationPreservesTheTriStateBoundary)
{
    struct TestCase
    {
        const char* name;
        Dependence pointer_dependence;
        uint64_t pointer_flags;
        Dependence expected_dependence;
        bool expected_thread_specific;
    };

    constexpr std::array test_cases{
        TestCase{ "Varying", Dependence::Varying, 0, Dependence::Varying, true },
        TestCase{ "UnknownLegacy",
                  Dependence::Unknown,
                  SPS_FLAG_THREAD_SPECIFIC,
                  Dependence::Unknown,
                  true },
        TestCase{ "Unknown", Dependence::Unknown, 0, Dependence::Unknown, false },
        TestCase{ "Invariant", Dependence::Invariant, 0, Dependence::Unknown, false },
    };

    for (const TestCase& test_case : test_cases)
    {
        SCOPED_TRACE(test_case.name);
        ::SPIRVSimulator::ValueMetadata& pointer_meta = value_meta_[kPointerId];
        pointer_meta = {};
        pointer_meta.invocation_dependence = test_case.pointer_dependence;
        pointer_meta.flags = test_case.pointer_flags;

        ExecuteLoad(CommonTypes::u64);

        const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
        EXPECT_EQ(result_meta.invocation_dependence, test_case.expected_dependence);
        EXPECT_EQ((result_meta.flags & SPS_FLAG_THREAD_SPECIFIC) != 0,
                  test_case.expected_thread_specific);
    }
}

TEST_F(ExternalLoadInvocationDependenceTests,
       VectorLoadPublishesOnlyTopLevelDependence)
{
    loaded_value_ = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 11 }, uint64_t{ 22 } });

    const ::SPIRVSimulator::DenseValueRangeInfo source_range{
        true, true, true, 0, 1, 1
    };
    ::SPIRVSimulator::ValueMetadata& pointer_meta = value_meta_[kPointerId];
    pointer_meta.invocation_dependence = Dependence::Varying;
    pointer_meta.component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };
    pointer_meta.subobject_invocation_dependence = {
        ::SPIRVSimulator::SubobjectInvocationDependenceEntry{
            { 0 }, Dependence::Invariant },
        ::SPIRVSimulator::SubobjectInvocationDependenceEntry{
            { 1 }, Dependence::Varying },
    };
    pointer_meta.flags = SPS_FLAG_IS_ARBITRARY;
    pointer_meta.value_range = source_range;
    pointer_meta.component_ranges = { source_range, source_range };
    pointer_meta.subobject_scalar_ranges = {
        ::SPIRVSimulator::SubobjectScalarRangeEntry{ { 0 }, source_range },
    };
    pointer_meta.address_range_valid = true;
    pointer_meta.address_range_min = 0x1000;
    pointer_meta.address_range_max = 0x1010;
    pointer_meta.address_range_stride = 8;
    pointer_meta.address_element_size = 8;

    ExecuteLoad(CommonTypes::uvec2);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_NE(result_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
    EXPECT_EQ(result_meta.flags & SPS_FLAG_IS_ARBITRARY, 0u);
    EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    EXPECT_FALSE(result_meta.value_range.valid);
    EXPECT_TRUE(result_meta.component_ranges.empty());
    EXPECT_TRUE(result_meta.subobject_scalar_ranges.empty());
    EXPECT_FALSE(result_meta.address_range_valid);
}

class CopyInvocationDependenceTests : public SPIRVSimulatorMockBase,
                                      public TestWithParam<spv::Op>
{
  protected:
    struct PreparedCopy
    {
        std::vector<uint32_t> words;
        uint32_t              result_id;
        uint32_t              source_id;
    };

    PreparedCopy PrepareCopy(const ::SPIRVSimulator::Value& result_value,
                             const ::SPIRVSimulator::Value& source_value,
                             CommonTypes                    type)
    {
        TestParameters parameters =
            TestParametersBuilder()
                .set_opcode(GetParam())
                .set_operand_at(0, result_value, type)
                .set_operand_at(1, source_value, type)
                .build();

        std::vector<uint32_t> words = prepare_submission(parameters);
        const uint32_t result_id = words[2];
        const uint32_t source_id = words[3];
        return {
            .words = std::move(words),
            .result_id = result_id,
            .source_id = source_id,
        };
    }

    void ExecuteCopy(const PreparedCopy& copy)
    {
        EXPECT_CALL(*this, SetValue(copy.result_id, _, true))
            .WillOnce([this](uint32_t                       id,
                             const ::SPIRVSimulator::Value& value,
                             bool                           clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });

        ::SPIRVSimulator::Instruction instruction{
            .opcode = GetParam(),
            .word_count = static_cast<uint16_t>(copy.words.size()),
            .words = copy.words,
        };
        EXPECT_TRUE(ExecuteInstruction(instruction));
    }
};

TEST_P(CopyInvocationDependenceTests, CopiesScalarInvocationDependence)
{
    constexpr std::array dependence_states{
        ::SPIRVSimulator::InvocationDependence::Invariant,
        ::SPIRVSimulator::InvocationDependence::Varying,
        ::SPIRVSimulator::InvocationDependence::Unknown,
    };

    for (const auto dependence : dependence_states)
    {
        const PreparedCopy copy =
            PrepareCopy(uint64_t{ 0 }, uint64_t{ 17 }, CommonTypes::u64);
        value_meta_[copy.source_id].invocation_dependence = dependence;

        ExecuteCopy(copy);

        const ::SPIRVSimulator::ValueMetadata& result_meta =
            value_meta_[copy.result_id];
        EXPECT_EQ(result_meta.invocation_dependence, dependence);
        EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
        EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    }
}

TEST_P(CopyInvocationDependenceTests,
       CopiesAggregateInvocationDependenceAndOwnsStorage)
{
    auto result_vector = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 0, 0, 0 });
    auto source_vector = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 3, 5, 7 });
    const PreparedCopy copy =
        PrepareCopy(result_vector, source_vector, CommonTypes::uvec3);

    ::SPIRVSimulator::ValueMetadata& source_meta = value_meta_[copy.source_id];
    source_meta.invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;
    source_meta.component_invocation_dependence = {
        ::SPIRVSimulator::InvocationDependence::Invariant,
        ::SPIRVSimulator::InvocationDependence::Varying,
        ::SPIRVSimulator::InvocationDependence::Unknown,
    };
    source_meta.subobject_invocation_dependence = {
        { { 1, 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
        { { 1, 1 }, ::SPIRVSimulator::InvocationDependence::Varying },
        { { 2, 0 }, ::SPIRVSimulator::InvocationDependence::Unknown },
    };

    ExecuteCopy(copy);

    const ::SPIRVSimulator::ValueMetadata& result_meta =
        value_meta_[copy.result_id];
    EXPECT_EQ(result_meta.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              source_meta.component_invocation_dependence);
    ASSERT_EQ(result_meta.subobject_invocation_dependence.size(), 3u);
    EXPECT_EQ(result_meta.subobject_invocation_dependence[0].path,
              (std::vector<uint32_t>{ 1, 0 }));
    EXPECT_EQ(result_meta.subobject_invocation_dependence[0].dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(result_meta.subobject_invocation_dependence[1].path,
              (std::vector<uint32_t>{ 1, 1 }));
    EXPECT_EQ(result_meta.subobject_invocation_dependence[1].dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_EQ(result_meta.subobject_invocation_dependence[2].path,
              (std::vector<uint32_t>{ 2, 0 }));
    EXPECT_EQ(result_meta.subobject_invocation_dependence[2].dependence,
              ::SPIRVSimulator::InvocationDependence::Unknown);

    source_meta.component_invocation_dependence[0] =
        ::SPIRVSimulator::InvocationDependence::Unknown;
    source_meta.subobject_invocation_dependence[0].path[0] = 9;
    source_meta.subobject_invocation_dependence[0].dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;

    EXPECT_EQ(result_meta.component_invocation_dependence[0],
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(result_meta.subobject_invocation_dependence[0].path,
              (std::vector<uint32_t>{ 1, 0 }));
    EXPECT_EQ(result_meta.subobject_invocation_dependence[0].dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
}

TEST_P(CopyInvocationDependenceTests, EmptySourceClearsStaleTargetMetadata)
{
    auto result_vector = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 0, 0, 0 });
    auto source_vector = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 3, 5, 7 });
    const PreparedCopy copy =
        PrepareCopy(result_vector, source_vector, CommonTypes::uvec3);

    ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[copy.result_id];
    result_meta.invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;
    result_meta.component_invocation_dependence = {
        ::SPIRVSimulator::InvocationDependence::Varying,
    };
    result_meta.subobject_invocation_dependence = {
        { { 0 }, ::SPIRVSimulator::InvocationDependence::Varying },
    };

    ExecuteCopy(copy);

    EXPECT_EQ(result_meta.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Unknown);
    EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

INSTANTIATE_TEST_SUITE_P(
    CopyOpcodes,
    CopyInvocationDependenceTests,
    Values(spv::Op::OpCopyObject, spv::Op::OpCopyLogical),
    [](const TestParamInfo<spv::Op>& info) {
        return info.param == spv::Op::OpCopyObject ? "CopyObject"
                                                   : "CopyLogical";
    });

class BitcastInvocationDependenceTests : public SPIRVSimulatorMockBase,
                                         public Test
{
  protected:
    static constexpr uint32_t kU32Vec2TypeId         = 3100;
    static constexpr uint32_t kOperandPointerTypeId  = 3101;
    static constexpr uint32_t kResultPointerTypeId   = 3102;
    static constexpr uint32_t kResultId              = 3103;
    static constexpr uint32_t kOperandId             = 3104;

    void SetUp() override
    {
        values_.resize(kOperandId + 1);
        value_meta_.resize(kOperandId + 1);

        simulation_data_    = &simulation_data;
        simulation_results_ = &simulation_results;

        u32vec2_type = ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 2);
        operand_pointer_type = ::SPIRVSimulator::Type::Pointer(
            spv::StorageClassPhysicalStorageBuffer, CommonTypes::u32);
        result_pointer_type = ::SPIRVSimulator::Type::Pointer(
            spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    }

    const std::byte* RemapPhysicalToHostPointer(uint64_t) const override
    {
        return backing.data();
    }

    std::vector<::SPIRVSimulator::DataSourceBits>
    FindDataSourcesFromResultID(uint32_t, uint32_t*) override
    {
        return {};
    }

    void ExecuteBitcast(const ::SPIRVSimulator::Type& operand_type)
    {
        EXPECT_CALL(*this, GetValue(kOperandId))
            .WillOnce(ReturnRef(values_[kOperandId]));
        EXPECT_CALL(*this, GetTypeByResultId(kOperandId))
            .WillOnce(ReturnRef(operand_type));
        EXPECT_CALL(*this, GetTypeByTypeId(kResultPointerTypeId))
            .WillOnce(ReturnRef(result_pointer_type));
        EXPECT_CALL(*this, SetValue(kResultId, _, true))
            .WillOnce([this](uint32_t id,
                             const ::SPIRVSimulator::Value& value,
                             bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(
                    id, value, clear_meta);
            });
        EXPECT_CALL(*this,
                    TransferFlags(kResultId, TypedEq<uint32_t>(kOperandId)))
            .WillOnce([this](uint32_t target, uint32_t source) {
                ::SPIRVSimulator::SPIRVSimulator::TransferFlags(
                    target, source);
            });

        const std::array<uint32_t, 4> words{
            static_cast<uint32_t>(spv::Op::OpBitcast),
            kResultPointerTypeId,
            kResultId,
            kOperandId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpBitcast,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        Op_Bitcast(instruction);
    }

    ::SPIRVSimulator::SimulationData    simulation_data;
    ::SPIRVSimulator::SimulationResults simulation_results;
    ::SPIRVSimulator::Type              u32vec2_type;
    ::SPIRVSimulator::Type              operand_pointer_type;
    ::SPIRVSimulator::Type              result_pointer_type;
    std::array<std::byte, 8>            backing{};
};

TEST_F(BitcastInvocationDependenceTests,
       NumericVectorToPointerPropagatesWholeValueDependence)
{
    values_[kOperandId] = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 0x1000, 0 });

    ::SPIRVSimulator::ValueMetadata& operand_meta = value_meta_[kOperandId];
    operand_meta.invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;
    operand_meta.component_invocation_dependence = {
        ::SPIRVSimulator::InvocationDependence::Invariant,
        ::SPIRVSimulator::InvocationDependence::Varying,
    };
    operand_meta.subobject_invocation_dependence = {
        { { 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
    };

    ExecuteBitcast(u32vec2_type);

    const ::SPIRVSimulator::ValueMetadata& result_meta =
        value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    EXPECT_NE(result_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
    EXPECT_NE(result_meta.flags & SPS_FLAG_IS_PBUFFER_PTR, 0u);
}

TEST_F(BitcastInvocationDependenceTests,
       PointerToPointerEarlyReturnPropagatesMetadata)
{
    values_[kOperandId] = ::SPIRVSimulator::PointerV{
        0x1000,
        0,
        kOperandPointerTypeId,
        kOperandId,
        spv::StorageClassPhysicalStorageBuffer,
        {},
    };

    ::SPIRVSimulator::ValueMetadata& operand_meta = value_meta_[kOperandId];
    operand_meta.flags = SPS_FLAG_IS_ARBITRARY;
    operand_meta.invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;
    operand_meta.component_invocation_dependence = {
        ::SPIRVSimulator::InvocationDependence::Invariant,
    };

    ExecuteBitcast(operand_pointer_type);

    const ::SPIRVSimulator::ValueMetadata& result_meta =
        value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    EXPECT_NE(result_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
    EXPECT_NE(result_meta.flags & SPS_FLAG_IS_ARBITRARY, 0u);
}

class BitcastInvocationDependenceShapeTests
    : public ::SPIRVSimulator::SPIRVSimulator,
      public Test
{
  protected:
    static constexpr uint32_t kU32Vec2TypeId = 3200;
    static constexpr uint32_t kI32Vec2TypeId = 3201;
    static constexpr uint32_t kU16TypeId     = 3202;
    static constexpr uint32_t kU16Vec4TypeId = 3203;
    static constexpr uint32_t kResultId      = 1;
    static constexpr uint32_t kOperandId     = 2;

    void SetUp() override
    {
        types_[CommonTypes::u64] =
            ::SPIRVSimulator::Type::Int(64, false);
        types_[CommonTypes::u32] =
            ::SPIRVSimulator::Type::Int(32, false);
        types_[CommonTypes::i32] =
            ::SPIRVSimulator::Type::Int(32, true);
        types_[kU16TypeId] =
            ::SPIRVSimulator::Type::Int(16, false);
        types_[kU32Vec2TypeId] =
            ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 2);
        types_[kI32Vec2TypeId] =
            ::SPIRVSimulator::Type::Vector(CommonTypes::i32, 2);
        types_[kU16Vec4TypeId] =
            ::SPIRVSimulator::Type::Vector(kU16TypeId, 4);
        value_meta_.resize(kOperandId + 1);
    }
};

TEST_F(BitcastInvocationDependenceShapeTests,
       RebuildsComponentsForSameSplitAndMergedShapes)
{
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    ::SPIRVSimulator::ValueMetadata& operand_meta = value_meta_[kOperandId];
    ::SPIRVSimulator::ValueMetadata& result_meta  = value_meta_[kResultId];

    operand_meta.invocation_dependence = Dependence::Varying;
    operand_meta.component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };
    result_meta.subobject_invocation_dependence = {
        { { 0 }, Dependence::Varying },
    };

    PropagateBitcastInvocationDependence(
        kResultId,
        kOperandId,
        GetTypeByTypeId(kI32Vec2TypeId),
        GetTypeByTypeId(kU32Vec2TypeId));

    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{
                  Dependence::Invariant,
                  Dependence::Varying,
              }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());

    operand_meta = {};
    operand_meta.invocation_dependence = Dependence::Varying;
    PropagateBitcastInvocationDependence(
        kResultId,
        kOperandId,
        GetTypeByTypeId(kU32Vec2TypeId),
        GetTypeByTypeId(CommonTypes::u64));

    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{
                  Dependence::Varying,
                  Dependence::Varying,
              }));

    operand_meta = {};
    operand_meta.invocation_dependence = Dependence::Varying;
    operand_meta.component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Unknown,
        Dependence::Varying,
        Dependence::Invariant,
    };
    PropagateBitcastInvocationDependence(
        kResultId,
        kOperandId,
        GetTypeByTypeId(kU32Vec2TypeId),
        GetTypeByTypeId(kU16Vec4TypeId));

    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{
                  Dependence::Unknown,
                  Dependence::Varying,
              }));
}

class ConvertUToPtrInvocationDependenceTests : public SPIRVSimulatorMockBase,
                                               public Test
{
  protected:
    static constexpr uint32_t kPointerTypeId = 3000;
    static constexpr uint32_t kResultId      = 3001;
    static constexpr uint32_t kIntegerId     = 3002;

    void SetUp() override
    {
        values_.resize(kIntegerId + 1);
        value_meta_.resize(kIntegerId + 1);
        values_[kIntegerId] = uint64_t{ 0x1000 };

        simulation_data_    = &simulation_data;
        simulation_results_ = &simulation_results;

        pointer_type = ::SPIRVSimulator::Type::Pointer(
            spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);

        EXPECT_CALL(*this, GetTypeByTypeId(kPointerTypeId))
            .Times(3)
            .WillRepeatedly(ReturnRef(pointer_type));
        EXPECT_CALL(*this, GetValue(kIntegerId))
            .Times(3)
            .WillRepeatedly(ReturnRef(values_[kIntegerId]));
        EXPECT_CALL(*this, SetValue(kResultId, _, true))
            .Times(3)
            .WillRepeatedly(
                [this](uint32_t id,
                       const ::SPIRVSimulator::Value& value,
                       bool clear_meta) {
                    ::SPIRVSimulator::SPIRVSimulator::SetValue(
                        id, value, clear_meta);
                });
        EXPECT_CALL(*this,
                    TransferFlags(kResultId, TypedEq<uint32_t>(kIntegerId)))
            .Times(3);
    }

    const std::byte* RemapPhysicalToHostPointer(uint64_t) const override
    {
        return backing.data();
    }

    std::vector<::SPIRVSimulator::DataSourceBits>
    FindDataSourcesFromResultID(uint32_t, uint32_t*) override
    {
        return {};
    }

    void ExecuteConvert()
    {
        const std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpConvertUToPtr),
            kPointerTypeId,
            kResultId,
            kIntegerId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpConvertUToPtr,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };

        Op_ConvertUToPtr(instruction);
    }

    ::SPIRVSimulator::SimulationData    simulation_data;
    ::SPIRVSimulator::SimulationResults simulation_results;
    ::SPIRVSimulator::Type              pointer_type;
    std::array<std::byte, 8>            backing{};
};

TEST_F(ConvertUToPtrInvocationDependenceTests,
       CopiesScalarInvocationDependence)
{
    constexpr std::array dependence_states{
        ::SPIRVSimulator::InvocationDependence::Invariant,
        ::SPIRVSimulator::InvocationDependence::Varying,
        ::SPIRVSimulator::InvocationDependence::Unknown,
    };

    for (const auto dependence : dependence_states)
    {
        value_meta_[kIntegerId].invocation_dependence = dependence;

        ExecuteConvert();

        const ::SPIRVSimulator::ValueMetadata& result_meta =
            value_meta_[kResultId];
        EXPECT_EQ(result_meta.invocation_dependence, dependence);
        EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
        EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    }
}

class UConvertInvocationDependenceTests : public SPIRVSimulatorMockBase,
                                          public Test
{
  protected:
    static constexpr uint32_t kU64VectorTypeId = 3011;
    static constexpr uint32_t kResultId        = 3012;
    static constexpr uint32_t kOperandId       = 3013;

    void SetUp() override
    {
        values_.resize(kOperandId + 1);
        value_meta_.resize(kOperandId + 1);
    }

    void ExecuteConvert(uint32_t result_type_id)
    {
        const std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpUConvert),
            result_type_id,
            kResultId,
            kOperandId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpUConvert,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };

        Op_UConvert(instruction);
    }
};

TEST_F(UConvertInvocationDependenceTests,
       CopiesScalarInvocationDependenceAcrossWidths)
{
    values_[kOperandId] = uint64_t{ 7 };
    value_meta_[kOperandId].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;

    EXPECT_CALL(*this, GetValue(kOperandId))
        .WillOnce(ReturnRef(values_[kOperandId]));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::u64))
        .WillOnce(ReturnRef(types_.at(CommonTypes::u64)));
    EXPECT_CALL(*this, GetTypeByResultId(kOperandId))
        .WillOnce(ReturnRef(types_.at(CommonTypes::u32)));
    EXPECT_CALL(*this, SetValue(kResultId, _, true))
        .WillOnce([this](uint32_t                       id,
                         const ::SPIRVSimulator::Value& value,
                         bool                           clear_meta) {
            ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
        });
    EXPECT_CALL(*this,
                TransferFlags(kResultId, TypedEq<uint32_t>(kOperandId)));

    ExecuteConvert(CommonTypes::u64);

    EXPECT_EQ(value_meta_[kResultId].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_TRUE(
        value_meta_[kResultId].component_invocation_dependence.empty());
}

TEST_F(UConvertInvocationDependenceTests,
       CopiesVectorComponentInvocationDependence)
{
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    const ::SPIRVSimulator::Type u32_vector_type =
        ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 2);
    const ::SPIRVSimulator::Type u64_vector_type =
        ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 2);
    values_[kOperandId] = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 7, 11 });
    value_meta_[kOperandId].invocation_dependence = Dependence::Varying;
    value_meta_[kOperandId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };

    EXPECT_CALL(*this, GetValue(kOperandId))
        .WillOnce(ReturnRef(values_[kOperandId]));
    EXPECT_CALL(*this, GetTypeByTypeId(kU64VectorTypeId))
        .WillOnce(ReturnRef(u64_vector_type));
    EXPECT_CALL(*this, GetTypeByResultId(kOperandId))
        .WillOnce(ReturnRef(u32_vector_type));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::u64))
        .WillOnce(ReturnRef(types_.at(CommonTypes::u64)));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::u32))
        .WillOnce(ReturnRef(types_.at(CommonTypes::u32)));
    EXPECT_CALL(*this, SetValue(kResultId, _, true))
        .WillOnce([this](uint32_t                       id,
                         const ::SPIRVSimulator::Value& value,
                         bool                           clear_meta) {
            ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
        });
    EXPECT_CALL(*this,
                TransferFlags(kResultId, TypedEq<uint32_t>(kOperandId)));

    ExecuteConvert(kU64VectorTypeId);

    EXPECT_EQ(value_meta_[kResultId].invocation_dependence,
              Dependence::Varying);
    EXPECT_EQ(value_meta_[kResultId].component_invocation_dependence,
              (std::vector<Dependence>{
                  Dependence::Invariant,
                  Dependence::Varying,
              }));
}

// verify that OpCompositeConstruct produces the compact vector
// representation and the generic aggregate scalar-leaf representation.
class CompositeMetadataTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static ::SPIRVSimulator::DenseValueRangeInfo MakeRange(
        uint64_t min,
        uint64_t max,
        uint8_t workgroup_id_component_mask,
        uint8_t local_invocation_id_component_mask)
    {
        ::SPIRVSimulator::DenseValueRangeInfo range = {
            true, true, true, min, max, 1
        };
        range.invocation_linearization.workgroup_id_component_mask =
            workgroup_id_component_mask;
        range.invocation_linearization.local_invocation_id_component_mask =
            local_invocation_id_component_mask;
        return range;
    }

    static void ExpectRange(
        const ::SPIRVSimulator::DenseValueRangeInfo& actual,
        const ::SPIRVSimulator::DenseValueRangeInfo& expected)
    {
        EXPECT_EQ(actual.valid, expected.valid);
        EXPECT_EQ(actual.thread_dependent, expected.thread_dependent);
        EXPECT_EQ(actual.dense_range, expected.dense_range);
        EXPECT_EQ(actual.min, expected.min);
        EXPECT_EQ(actual.max, expected.max);
        EXPECT_EQ(actual.stride, expected.stride);
        EXPECT_EQ(
            actual.invocation_linearization.workgroup_id_component_mask,
            expected.invocation_linearization.workgroup_id_component_mask);
        EXPECT_EQ(
            actual.invocation_linearization.local_invocation_id_component_mask,
            expected.invocation_linearization.local_invocation_id_component_mask);
    }

    void SetValueWithBaseImplementation(uint32_t                       result_id,
                                        const ::SPIRVSimulator::Value& value,
                                        bool                           clear_meta)
    {
        ::SPIRVSimulator::SPIRVSimulator::SetValue(result_id, value, clear_meta);
    }

    void ExpectSetValueReset(uint32_t result_id)
    {
        EXPECT_CALL(*this, SetValue(result_id, _, true))
            .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                SetValueWithBaseImplementation(id, value, clear_meta);
            });
    }

    static void ExpectSubobjectInvocationDependence(
        const std::vector<::SPIRVSimulator::SubobjectInvocationDependenceEntry>& actual,
        std::initializer_list<::SPIRVSimulator::SubobjectInvocationDependenceEntry> expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        size_t index = 0;
        for (const auto& expected_entry : expected)
        {
            EXPECT_EQ(actual[index].path, expected_entry.path);
            EXPECT_EQ(actual[index].dependence, expected_entry.dependence);
            ++index;
        }
    }
};


TEST_F(CompositeMetadataTests, VectorConstructPreservesComponentRanges)
{
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2 }),
                CommonTypes::uvec2)
            .set_operands_range(1, CommonTypes::u64, std::initializer_list<uint64_t>{ 1, 2 })
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction instruction{ .opcode     = parameters.opcode,
                                               .word_count = static_cast<uint16_t>(words.size()),
                                               .words      = words };

    const uint32_t result_id = words[2];
    const uint32_t first_component_id = words[3];
    const uint32_t second_component_id = words[4];
    const auto first_range =
        MakeRange(0,
                  7,
                  ::SPIRVSimulator::kInvocationComponentX,
                  ::SPIRVSimulator::kInvocationComponentY);
    const auto second_range =
        MakeRange(8,
                  15,
                  ::SPIRVSimulator::kInvocationComponentZ,
                  ::SPIRVSimulator::kInvocationComponentX |
                      ::SPIRVSimulator::kInvocationComponentZ);
    value_meta_[first_component_id].value_range = first_range;
    value_meta_[second_component_id].value_range = second_range;

    EXPECT_CALL(*this, SetValue(result_id, _, true))
        .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
            SetValueWithBaseImplementation(id, value, clear_meta);
        });

    EXPECT_TRUE(ExecuteInstruction(instruction));

    const auto& ranges = value_meta_[result_id].component_ranges;
    ASSERT_EQ(ranges.size(), 2u);
    ExpectRange(ranges[0], first_range);
    ExpectRange(ranges[1], second_range);
}

TEST_F(CompositeMetadataTests, StructConstructPreservesSubobjectScalarRanges)
{
    auto offset = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 4 });
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::AggregateV>(
                    std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 2 }, offset }),
                ::SPIRVSimulator::Type::Struct(0))
            .set_operand_at(1, uint64_t{ 2 }, CommonTypes::u64)
            .set_operand_at(2, offset, CommonTypes::uvec2)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction instruction{ .opcode     = parameters.opcode,
                                               .word_count = static_cast<uint16_t>(words.size()),
                                               .words      = words };

    const uint32_t result_id = words[2];
    const uint32_t scalar_id = words[3];
    const uint32_t vector_id = words[4];
    const auto scalar_range =
        MakeRange(0,
                  7,
                  ::SPIRVSimulator::kInvocationComponentY,
                  ::SPIRVSimulator::kInvocationComponentX);
    const auto first_vector_range =
        MakeRange(0,
                  15,
                  ::SPIRVSimulator::kInvocationComponentX,
                  ::SPIRVSimulator::kInvocationComponentY);
    const auto second_vector_range =
        MakeRange(0,
                  31,
                  ::SPIRVSimulator::kInvocationComponentZ,
                  ::SPIRVSimulator::kInvocationComponentZ);
    value_meta_[scalar_id].value_range = scalar_range;
    value_meta_[vector_id].component_ranges = {
        first_vector_range,
        second_vector_range,
    };

    EXPECT_CALL(*this, SetValue(result_id, _, true))
        .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
            SetValueWithBaseImplementation(id, value, clear_meta);
        });

    EXPECT_TRUE(ExecuteInstruction(instruction));

    const auto& ranges = value_meta_[result_id].subobject_scalar_ranges;
    ASSERT_EQ(ranges.size(), 3u);
    EXPECT_EQ(ranges[0].path, std::vector<uint32_t>{ 0 });
    ExpectRange(ranges[0].value_range, scalar_range);
    EXPECT_EQ(ranges[1].path, (std::vector<uint32_t>{ 1, 0 }));
    ExpectRange(ranges[1].value_range, first_vector_range);
    EXPECT_EQ(ranges[2].path, (std::vector<uint32_t>{ 1, 1 }));
    ExpectRange(ranges[2].value_range, second_vector_range);
}

TEST_F(CompositeMetadataTests, VectorConstructFlattensInvocationDependence)
{
    auto source_vector =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 3 });
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2, 3 }),
                CommonTypes::uvec3)
            .set_operand_at(1, uint64_t{ 1 }, CommonTypes::u64)
            .set_operand_at(2, source_vector, CommonTypes::uvec2)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = parameters.opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    const uint32_t result_id = words[2];
    value_meta_[words[3]].invocation_dependence = Dependence::Invariant;
    value_meta_[words[4]].invocation_dependence = Dependence::Varying;
    value_meta_[words[4]].component_invocation_dependence = {
        Dependence::Unknown,
        Dependence::Varying,
    };

    ExpectSetValueReset(result_id);
    ASSERT_TRUE(ExecuteInstruction(instruction));

    const auto& result_meta = value_meta_[result_id];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Unknown, Dependence::Varying }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

TEST_F(CompositeMetadataTests, MatrixConstructBuildsColumnAndLeafInvocationDependence)
{
    auto first_column =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 });
    auto second_column =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 });
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::MatrixV>(
                    std::initializer_list<::SPIRVSimulator::Value>{ first_column, second_column }),
                CommonTypes::mat2)
            .set_operand_at(1, first_column, CommonTypes::vec2)
            .set_operand_at(2, second_column, CommonTypes::vec2)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = parameters.opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    const uint32_t result_id = words[2];
    // An invariant column can fill missing row proofs. A varying column cannot,
    // so its missing second row must remain Unknown.
    value_meta_[words[3]].invocation_dependence = Dependence::Invariant;
    value_meta_[words[4]].invocation_dependence = Dependence::Varying;
    value_meta_[words[4]].component_invocation_dependence = { Dependence::Invariant };

    ExpectSetValueReset(result_id);
    ASSERT_TRUE(ExecuteInstruction(instruction));

    const auto& result_meta = value_meta_[result_id];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Varying }));
    ExpectSubobjectInvocationDependence(
        result_meta.subobject_invocation_dependence,
        {
            { { 0, 0 }, Dependence::Invariant },
            { { 0, 1 }, Dependence::Invariant },
            { { 1, 0 }, Dependence::Invariant },
            { { 1, 1 }, Dependence::Unknown },
        });
}

TEST_F(CompositeMetadataTests, StructConstructRebasesMemberInvocationDependence)
{
    auto vector_member =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 3 });
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::AggregateV>(
                    std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 1 }, vector_member }),
                ::SPIRVSimulator::Type::Struct(0))
            .set_operand_at(1, uint64_t{ 1 }, CommonTypes::u64)
            .set_operand_at(2, vector_member, CommonTypes::uvec2)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = parameters.opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    const uint32_t result_id = words[2];
    value_meta_[words[3]].invocation_dependence = Dependence::Invariant;
    value_meta_[words[4]].invocation_dependence = Dependence::Varying;
    value_meta_[words[4]].component_invocation_dependence = {
        Dependence::Unknown,
        Dependence::Varying,
    };

    ExpectSetValueReset(result_id);
    ASSERT_TRUE(ExecuteInstruction(instruction));

    const auto& result_meta = value_meta_[result_id];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Varying }));
    ExpectSubobjectInvocationDependence(
        result_meta.subobject_invocation_dependence,
        {
            { { 0 }, Dependence::Invariant },
            { { 1, 0 }, Dependence::Unknown },
            { { 1, 1 }, Dependence::Varying },
        });
}

TEST_F(CompositeMetadataTests, ArrayConstructBuildsElementInvocationDependence)
{
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::AggregateV>(
                    std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 1 }, uint64_t{ 2 } }),
                ::SPIRVSimulator::Type::Array(CommonTypes::u64, 2))
            .set_operand_at(1, uint64_t{ 1 }, CommonTypes::u64)
            .set_operand_at(2, uint64_t{ 2 }, CommonTypes::u64)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = parameters.opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    const uint32_t result_id = words[2];
    value_meta_[words[3]].invocation_dependence = Dependence::Unknown;
    value_meta_[words[4]].invocation_dependence = Dependence::Invariant;

    ExpectSetValueReset(result_id);
    ASSERT_TRUE(ExecuteInstruction(instruction));

    const auto& result_meta = value_meta_[result_id];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Unknown);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Unknown, Dependence::Invariant }));
    ExpectSubobjectInvocationDependence(
        result_meta.subobject_invocation_dependence,
        {
            { { 0 }, Dependence::Unknown },
            { { 1 }, Dependence::Invariant },
        });
}

TEST_F(CompositeMetadataTests, AggregateConstantNullBuildsInvariantInvocationDependence)
{
    constexpr uint32_t kStructTypeId = 8000;
    constexpr uint32_t kResultId     = 8001;

    values_.resize(kResultId + 1);
    value_meta_.resize(kResultId + 1);
    types_[kStructTypeId]          = ::SPIRVSimulator::Type::Struct(kStructTypeId);
    struct_members_[kStructTypeId] = { CommonTypes::u64, CommonTypes::uvec2 };
    EXPECT_CALL(*this, GetTypeByTypeId(kStructTypeId)).WillRepeatedly(ReturnRef(types_[kStructTypeId]));
    ExpectSetValueReset(kResultId);

    const std::vector<uint32_t> words{
        static_cast<uint32_t>(spv::Op::OpConstantNull),
        kStructTypeId,
        kResultId,
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = spv::Op::OpConstantNull,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };
    ASSERT_TRUE(ExecuteInstruction(instruction));

    const auto& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Invariant);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Invariant }));
    ExpectSubobjectInvocationDependence(
        result_meta.subobject_invocation_dependence,
        {
            { { 0 }, Dependence::Invariant },
            { { 1, 0 }, Dependence::Invariant },
            { { 1, 1 }, Dependence::Invariant },
        });
}

TEST_F(CompositeMetadataTests,
       CompositeInsertPreservesUnchangedAndInsertedComponentProvenance)
{
    auto source_vector = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 1, 2 });
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeInsert)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::VectorV>(
                    std::initializer_list<uint64_t>{ 1, 9 }),
                CommonTypes::uvec2)
            .set_operand_at(1, uint64_t{ 9 }, CommonTypes::u64)
            .set_operand_at(2, source_vector, CommonTypes::uvec2)
            .set_operand_at(3, uint64_t{ 1 }, CommonTypes::literal)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = parameters.opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };

    const uint32_t result_id = words[2];
    const uint32_t object_id = words[3];
    const uint32_t composite_id = words[4];
    const auto unchanged_range =
        MakeRange(0,
                  7,
                  ::SPIRVSimulator::kInvocationComponentX,
                  ::SPIRVSimulator::kInvocationComponentX);
    const auto replaced_range =
        MakeRange(8,
                  15,
                  ::SPIRVSimulator::kInvocationComponentY,
                  ::SPIRVSimulator::kInvocationComponentY);
    const auto inserted_range =
        MakeRange(16,
                  31,
                  ::SPIRVSimulator::kInvocationComponentZ,
                  ::SPIRVSimulator::kInvocationComponentX |
                      ::SPIRVSimulator::kInvocationComponentZ);
    value_meta_[object_id].value_range = inserted_range;
    value_meta_[composite_id].component_ranges = {
        unchanged_range,
        replaced_range,
    };

    EXPECT_CALL(*this, SetValue(result_id, _, true))
        .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
            SetValueWithBaseImplementation(id, value, clear_meta);
        });

    EXPECT_TRUE(ExecuteInstruction(instruction));

    const auto& ranges = value_meta_[result_id].component_ranges;
    ASSERT_EQ(ranges.size(), 2u);
    ExpectRange(ranges[0], unchanged_range);
    ExpectRange(ranges[1], inserted_range);
    EXPECT_TRUE(value_meta_[result_id].subobject_scalar_ranges.empty());
}

class CompositeInsertInvocationDependenceTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kResultId    = 8000;
    static constexpr uint32_t kObjectId    = 8001;
    static constexpr uint32_t kCompositeId = 8002;
    static constexpr uint32_t kMaximumId   = kCompositeId;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;
        values_.resize(kMaximumId + 1);
        value_meta_.resize(kMaximumId + 1);

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_.at(id);
        });
        EXPECT_CALL(*this, GetTypeByResultId(_))
            .WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
                return types_.at(value_type_ids_.at(id));
            });
        EXPECT_CALL(*this, GetTypeByTypeId(_))
            .WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& { return types_.at(id); });
        EXPECT_CALL(*this, SetValue(kResultId, _, true))
            .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, TransferFlags(kResultId, TypedEq<uint32_t>(kObjectId)));
        EXPECT_CALL(*this, TransferFlags(kResultId, TypedEq<uint32_t>(kCompositeId)));
    }

    uint32_t GetTypeID(uint32_t result_id) const override { return value_type_ids_.at(result_id); }

    void SetTypedValue(uint32_t id, uint32_t type_id, ::SPIRVSimulator::Value value)
    {
        values_[id]         = std::move(value);
        value_type_ids_[id] = type_id;
    }

    void ExecuteCompositeInsert(const std::vector<uint32_t>& indexes)
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpCompositeInsert),
            value_type_ids_.at(kResultId),
            kResultId,
            kObjectId,
            kCompositeId,
        };
        words.insert(words.end(), indexes.begin(), indexes.end());

        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = spv::Op::OpCompositeInsert,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        ASSERT_TRUE(ExecuteInstruction(instruction));
    }

    ::SPIRVSimulator::UnorderedMap<uint32_t, uint32_t> value_type_ids_;
};

TEST_F(CompositeInsertInvocationDependenceTests, VectorDirectInsertChangesInvariantToVarying)
{
    SetTypedValue(kResultId, CommonTypes::uvec2, std::monostate{});
    SetTypedValue(kObjectId, CommonTypes::u64, uint64_t{ 9 });
    SetTypedValue(kCompositeId,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2 }));

    value_meta_[kCompositeId].invocation_dependence = Dependence::Invariant;
    value_meta_[kCompositeId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
    };
    value_meta_[kObjectId].invocation_dependence = Dependence::Varying;

    ExecuteCompositeInsert({ 1 });

    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kResultId]);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->elems.size(), 2u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[0]), 1u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[1]), 9u);
    EXPECT_EQ(value_meta_[kResultId].invocation_dependence, Dependence::Varying);
    EXPECT_EQ(value_meta_[kResultId].component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Varying }));
    EXPECT_TRUE(value_meta_[kResultId].subobject_invocation_dependence.empty());
}

TEST_F(CompositeInsertInvocationDependenceTests, VectorDirectInsertChangesVaryingToInvariant)
{
    SetTypedValue(kResultId, CommonTypes::uvec2, std::monostate{});
    SetTypedValue(kObjectId, CommonTypes::u64, uint64_t{ 9 });
    SetTypedValue(kCompositeId,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2 }));

    value_meta_[kCompositeId].invocation_dependence = Dependence::Varying;
    value_meta_[kCompositeId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };
    value_meta_[kObjectId].invocation_dependence = Dependence::Invariant;

    ExecuteCompositeInsert({ 1 });

    EXPECT_EQ(value_meta_[kResultId].invocation_dependence, Dependence::Invariant);
    EXPECT_EQ(value_meta_[kResultId].component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Invariant }));
    EXPECT_TRUE(value_meta_[kResultId].subobject_invocation_dependence.empty());
}

TEST_F(CompositeInsertInvocationDependenceTests, StructDirectInsertPreservesSiblingAndRemovesStaleTargetMetadata)
{
    constexpr uint32_t struct_type_id = 8100;
    types_[struct_type_id]             = ::SPIRVSimulator::Type::Struct(struct_type_id);
    struct_members_[struct_type_id]    = { CommonTypes::u64, CommonTypes::u64 };

    SetTypedValue(kResultId, struct_type_id, std::monostate{});
    SetTypedValue(kObjectId, CommonTypes::u64, uint64_t{ 9 });
    SetTypedValue(kCompositeId,
                  struct_type_id,
                  std::make_shared<::SPIRVSimulator::AggregateV>(
                      std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 1 }, uint64_t{ 2 } }));

    value_meta_[kCompositeId].invocation_dependence = Dependence::Varying;
    value_meta_[kCompositeId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };
    value_meta_[kCompositeId].subobject_invocation_dependence = {
        { { 0 }, Dependence::Invariant },
        { { 1 }, Dependence::Varying },
        { { 1, 0 }, Dependence::Varying },
    };
    value_meta_[kObjectId].invocation_dependence = Dependence::Invariant;

    ExecuteCompositeInsert({ 1 });

    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(values_[kResultId]);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->elems.size(), 2u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[0]), 1u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[1]), 9u);
    EXPECT_EQ(value_meta_[kResultId].invocation_dependence, Dependence::Invariant);
    EXPECT_EQ(value_meta_[kResultId].component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Invariant }));
    ASSERT_EQ(value_meta_[kResultId].subobject_invocation_dependence.size(), 2u);
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[0].path, std::vector<uint32_t>{ 0 });
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[0].dependence, Dependence::Invariant);
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[1].path, std::vector<uint32_t>{ 1 });
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[1].dependence, Dependence::Invariant);
}

TEST_F(CompositeInsertInvocationDependenceTests, NestedMatrixInsertRebuildsAffectedColumn)
{
    SetTypedValue(kResultId, CommonTypes::mat2, std::monostate{});
    SetTypedValue(kObjectId, CommonTypes::f64, 9.0);
    SetTypedValue(kCompositeId,
                  CommonTypes::mat2,
                  std::make_shared<::SPIRVSimulator::MatrixV>(
                      std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2));

    value_meta_[kCompositeId].invocation_dependence = Dependence::Invariant;
    value_meta_[kCompositeId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
    };
    value_meta_[kCompositeId].subobject_invocation_dependence = {
        { { 0, 0 }, Dependence::Invariant },
        { { 0, 1 }, Dependence::Invariant },
        { { 1, 0 }, Dependence::Invariant },
        { { 1, 1 }, Dependence::Invariant },
    };
    value_meta_[kObjectId].invocation_dependence = Dependence::Varying;

    ExecuteCompositeInsert({ 1, 0 });

    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::MatrixV>>(values_[kResultId]);
    ASSERT_NE(result, nullptr);
    const auto& second_column = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(result->cols[1]);
    ASSERT_NE(second_column, nullptr);
    EXPECT_DOUBLE_EQ(std::get<double>(second_column->elems[0]), 9.0);
    EXPECT_DOUBLE_EQ(std::get<double>(second_column->elems[1]), 4.0);
    EXPECT_EQ(value_meta_[kResultId].invocation_dependence, Dependence::Varying);
    EXPECT_EQ(value_meta_[kResultId].component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Varying }));
}

TEST_F(CompositeInsertInvocationDependenceTests, MatrixSubtreeInsertUsesCoarseSummaryAndRemovesOldLeaves)
{
    SetTypedValue(kResultId, CommonTypes::mat2, std::monostate{});
    SetTypedValue(kObjectId,
                  CommonTypes::vec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 9.0, 10.0 }));
    SetTypedValue(kCompositeId,
                  CommonTypes::mat2,
                  std::make_shared<::SPIRVSimulator::MatrixV>(
                      std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2));

    value_meta_[kCompositeId].invocation_dependence = Dependence::Invariant;
    value_meta_[kCompositeId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
    };
    value_meta_[kCompositeId].subobject_invocation_dependence = {
        { { 0, 0 }, Dependence::Invariant },
        { { 0, 1 }, Dependence::Invariant },
        { { 1, 0 }, Dependence::Invariant },
        { { 1, 1 }, Dependence::Invariant },
    };
    value_meta_[kObjectId].invocation_dependence = Dependence::Varying;

    ExecuteCompositeInsert({ 1 });

    EXPECT_EQ(value_meta_[kResultId].invocation_dependence, Dependence::Varying);
    EXPECT_EQ(value_meta_[kResultId].component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Varying }));
    ASSERT_EQ(value_meta_[kResultId].subobject_invocation_dependence.size(), 2u);
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[0].path,
              (std::vector<uint32_t>{ 0, 0 }));
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[1].path,
              (std::vector<uint32_t>{ 0, 1 }));
}

TEST_F(CompositeInsertInvocationDependenceTests, MissingUnchangedLeafKeepsNestedColumnUnknown)
{
    SetTypedValue(kResultId, CommonTypes::mat2, std::monostate{});
    SetTypedValue(kObjectId, CommonTypes::f64, 9.0);
    SetTypedValue(kCompositeId,
                  CommonTypes::mat2,
                  std::make_shared<::SPIRVSimulator::MatrixV>(
                      std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2));

    value_meta_[kCompositeId].invocation_dependence = Dependence::Unknown;
    value_meta_[kCompositeId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Unknown,
    };
    value_meta_[kCompositeId].subobject_invocation_dependence = {
        { { 0, 0 }, Dependence::Invariant },
        { { 0, 1 }, Dependence::Invariant },
        { { 1, 0 }, Dependence::Varying },
    };
    value_meta_[kObjectId].invocation_dependence = Dependence::Invariant;

    ExecuteCompositeInsert({ 1, 0 });

    EXPECT_EQ(value_meta_[kResultId].invocation_dependence, Dependence::Unknown);
    EXPECT_EQ(value_meta_[kResultId].component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Unknown }));
    ASSERT_EQ(value_meta_[kResultId].subobject_invocation_dependence.size(), 3u);
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[2].path,
              (std::vector<uint32_t>{ 1, 0 }));
    EXPECT_EQ(value_meta_[kResultId].subobject_invocation_dependence[2].dependence, Dependence::Invariant);
}

TEST_F(CompositeMetadataTests, VectorShufflePreservesSelectedComponentProvenance)
{
    auto first_vector = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 1, 2 });
    auto second_vector = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 3, 4 });
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpVectorShuffle)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::VectorV>(
                    std::initializer_list<uint64_t>{ 4, 2 }),
                CommonTypes::uvec2)
            .set_operand_at(1, first_vector, CommonTypes::uvec2)
            .set_operand_at(2, second_vector, CommonTypes::uvec2)
            .set_operand_at(3, uint64_t{ 3 }, CommonTypes::literal)
            .set_operand_at(4, uint64_t{ 1 }, CommonTypes::literal)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = parameters.opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };

    const uint32_t result_id = words[2];
    const uint32_t first_vector_id = words[3];
    const uint32_t second_vector_id = words[4];
    const auto first_selected_range =
        MakeRange(8,
                  15,
                  ::SPIRVSimulator::kInvocationComponentY,
                  ::SPIRVSimulator::kInvocationComponentY);
    const auto second_selected_range =
        MakeRange(24,
                  31,
                  ::SPIRVSimulator::kInvocationComponentX |
                      ::SPIRVSimulator::kInvocationComponentZ,
                  ::SPIRVSimulator::kInvocationComponentX |
                      ::SPIRVSimulator::kInvocationComponentY);
    value_meta_[first_vector_id].component_ranges = {
        MakeRange(0,
                  7,
                  ::SPIRVSimulator::kInvocationComponentX,
                  ::SPIRVSimulator::kInvocationComponentX),
        first_selected_range,
    };
    value_meta_[second_vector_id].component_ranges = {
        MakeRange(16,
                  23,
                  ::SPIRVSimulator::kInvocationComponentZ,
                  ::SPIRVSimulator::kInvocationComponentZ),
        second_selected_range,
    };

    EXPECT_CALL(*this, SetValue(result_id, _, true))
        .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
            SetValueWithBaseImplementation(id, value, clear_meta);
        });

    EXPECT_TRUE(ExecuteInstruction(instruction));

    const auto& ranges = value_meta_[result_id].component_ranges;
    ASSERT_EQ(ranges.size(), 2u);
    ExpectRange(ranges[0], second_selected_range);
    ExpectRange(ranges[1], first_selected_range);
}

class VectorShuffleInvocationDependenceTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kResultId = 8100;
    static constexpr uint32_t kVec1Id   = 8101;
    static constexpr uint32_t kVec2Id   = 8102;

    void SetUp() override
    {
        values_.resize(kVec2Id + 1);
        value_meta_.resize(kVec2Id + 1);
        values_[kVec1Id] =
            std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2 });
        values_[kVec2Id] =
            std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 4 });

        EXPECT_CALL(*this, GetValue(kVec1Id)).WillRepeatedly(ReturnRef(values_[kVec1Id]));
        EXPECT_CALL(*this, GetValue(kVec2Id)).WillRepeatedly(ReturnRef(values_[kVec2Id]));
        EXPECT_CALL(*this, GetTypeByResultId(kVec1Id)).WillRepeatedly(ReturnRef(types_[CommonTypes::uvec2]));
        EXPECT_CALL(*this, TransferFlags(TypedEq<uint32_t>(kResultId), TypedEq<uint32_t>(kVec1Id)));
        EXPECT_CALL(*this, TransferFlags(TypedEq<uint32_t>(kResultId), TypedEq<uint32_t>(kVec2Id)));
    }

    void ExecuteVectorShuffle(uint32_t result_type_id, std::initializer_list<uint32_t> selectors)
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpVectorShuffle),
            result_type_id,
            kResultId,
            kVec1Id,
            kVec2Id,
        };
        words.insert(words.end(), selectors.begin(), selectors.end());

        EXPECT_CALL(*this, SetValue(kResultId, _, true))
            .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });

        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = spv::Op::OpVectorShuffle,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        ASSERT_TRUE(ExecuteInstruction(instruction));
    }
};

TEST_F(VectorShuffleInvocationDependenceTests, ReordersAndRepeatsComponentsAcrossSources)
{
    value_meta_[kVec1Id].invocation_dependence = Dependence::Varying;
    value_meta_[kVec1Id].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };
    value_meta_[kVec2Id].invocation_dependence = Dependence::Unknown;
    value_meta_[kVec2Id].component_invocation_dependence = {
        Dependence::Unknown,
        Dependence::Invariant,
    };
    value_meta_[kResultId].subobject_invocation_dependence = {
        { { 9 }, Dependence::Varying },
    };

    ExecuteVectorShuffle(CommonTypes::uvec3, { 3, 1, 3 });

    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kResultId]);
    ASSERT_EQ(result->elems.size(), 3u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[0]), 4u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[1]), 2u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[2]), 4u);

    const auto& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Varying, Dependence::Invariant }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    EXPECT_TRUE(result_meta.component_ranges.empty());
}

TEST_F(VectorShuffleInvocationDependenceTests, UndefSelectorContributesUnknown)
{
    value_meta_[kVec1Id].invocation_dependence = Dependence::Invariant;
    value_meta_[kVec1Id].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
    };

    ExecuteVectorShuffle(CommonTypes::uvec2, { 0, 0xFFFFFFFFu });

    const auto& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Unknown);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Unknown }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

TEST_F(VectorShuffleInvocationDependenceTests, MissingComponentProofUsesConservativeFallback)
{
    value_meta_[kVec1Id].invocation_dependence = Dependence::Invariant;
    value_meta_[kVec2Id].invocation_dependence = Dependence::Varying;

    ExecuteVectorShuffle(CommonTypes::uvec2, { 1, 2 });

    const auto& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Unknown);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Unknown }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

TEST_F(VectorShuffleInvocationDependenceTests, ExactComponentProofOverridesCoarseVaryingSummary)
{
    value_meta_[kVec1Id].invocation_dependence = Dependence::Varying;
    value_meta_[kVec1Id].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };

    ExecuteVectorShuffle(CommonTypes::uvec2, { 0, 0 });

    const auto& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Invariant);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Invariant }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

namespace
{

// Shared semantic cases used by UT-CM-03 and UT-CM-04. Each case describes
// the same subtree independently of whether it is reached by extract or load.
enum class SubtreeProjectionShape
{
    StructMemberVector,
    StructMemberArray,
    NestedStructSubtree,
};

struct ExpectedValueMetadataSpec
{
    ::SPIRVSimulator::DenseValueRangeInfo value_range;
    std::vector<::SPIRVSimulator::DenseValueRangeInfo> component_ranges;
    std::vector<::SPIRVSimulator::SubobjectScalarRangeEntry> subobject_scalar_ranges;
};

struct ExpectedInvocationDependenceMetadataSpec
{
    ::SPIRVSimulator::InvocationDependence invocation_dependence = ::SPIRVSimulator::InvocationDependence::Unknown;
    std::vector<::SPIRVSimulator::InvocationDependence>               component_invocation_dependence;
    std::vector<::SPIRVSimulator::SubobjectInvocationDependenceEntry> subobject_invocation_dependence;
};

struct SubtreeProjectionTestCase
{
    const char*                                                       name;
    SubtreeProjectionShape                                            shape;
    std::vector<uint32_t>                                             selected_path;
    std::vector<::SPIRVSimulator::SubobjectScalarRangeEntry>          source_subobject_ranges;
    ExpectedValueMetadataSpec                                         expected_extracted_metadata;
    std::vector<uint32_t>                                             followup_scalar_path;
    ::SPIRVSimulator::DenseValueRangeInfo                             expected_scalar_range;
    uint64_t                                                          expected_scalar_value;
    std::vector<::SPIRVSimulator::SubobjectInvocationDependenceEntry> source_subobject_invocation_dependence;
    ::SPIRVSimulator::InvocationDependence                            source_invocation_dependence;
    std::vector<::SPIRVSimulator::InvocationDependence>               source_component_invocation_dependence;
    ExpectedInvocationDependenceMetadataSpec                          expected_extracted_invocation_dependence;
    ::SPIRVSimulator::InvocationDependence                            expected_scalar_invocation_dependence;
};

::SPIRVSimulator::DenseValueRangeInfo MakeProjectionRange(bool     thread_dependent,
                                                          bool     dense_range,
                                                          uint64_t min,
                                                          uint64_t max,
                                                          uint64_t stride,
                                                          uint8_t  workgroup_id_component_mask        = 0,
                                                          uint8_t  local_invocation_id_component_mask = 0)
{
    ::SPIRVSimulator::DenseValueRangeInfo range = {
        true, thread_dependent, dense_range, min, max, stride,
    };
    range.invocation_linearization.workgroup_id_component_mask        = workgroup_id_component_mask;
    range.invocation_linearization.local_invocation_id_component_mask = local_invocation_id_component_mask;
    return range;
}

SubtreeProjectionTestCase MakeStructMemberVectorCase()
{
    const auto range_a = MakeProjectionRange(
        true, true, 10, 19, 1, ::SPIRVSimulator::kInvocationComponentX, ::SPIRVSimulator::kInvocationComponentY);
    const auto range_b = MakeProjectionRange(
        true, true, 30, 38, 2, ::SPIRVSimulator::kInvocationComponentY, ::SPIRVSimulator::kInvocationComponentZ);
    const auto sibling_range = MakeProjectionRange(false, true, 100, 100, 1);

    return SubtreeProjectionTestCase{
        "StructMemberVector",
        SubtreeProjectionShape::StructMemberVector,
        { 0 },
        {
            { { 0, 0 }, range_a },
            { { 0, 1 }, range_b },
            { { 1 }, sibling_range },
        },
        {
            {},
            { range_a, range_b },
            {},
        },
        { 1 },
        range_b,
        22,
        {
            { { 0, 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
            { { 0, 1 }, ::SPIRVSimulator::InvocationDependence::Varying },
            { { 1 }, ::SPIRVSimulator::InvocationDependence::Invariant },
        },
        ::SPIRVSimulator::InvocationDependence::Varying,
        {
            ::SPIRVSimulator::InvocationDependence::Varying,
            ::SPIRVSimulator::InvocationDependence::Invariant,
        },
        {
            ::SPIRVSimulator::InvocationDependence::Varying,
            {
                ::SPIRVSimulator::InvocationDependence::Invariant,
                ::SPIRVSimulator::InvocationDependence::Varying,
            },
            {},
        },
        ::SPIRVSimulator::InvocationDependence::Varying,
    };
}

SubtreeProjectionTestCase MakeStructMemberArrayCase()
{
    const auto range_a = MakeProjectionRange(
        true, true, 40, 49, 1, ::SPIRVSimulator::kInvocationComponentY, ::SPIRVSimulator::kInvocationComponentX);
    const auto range_b = MakeProjectionRange(
        true, false, 60, 72, 4, ::SPIRVSimulator::kInvocationComponentZ, ::SPIRVSimulator::kInvocationComponentY);
    const auto sibling_range = MakeProjectionRange(false, true, 110, 110, 1);

    return SubtreeProjectionTestCase{
        "StructMemberArray",
        SubtreeProjectionShape::StructMemberArray,
        { 1 },
        {
            { { 1, 0 }, range_a },
            { { 1, 1 }, range_b },
            { { 0 }, sibling_range },
        },
        {
            {},
            {},
            {
                { { 0 }, range_a },
                { { 1 }, range_b },
            },
        },
        { 0 },
        range_a,
        31,
        {
            { { 1, 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
            { { 1, 1 }, ::SPIRVSimulator::InvocationDependence::Unknown },
            { { 0 }, ::SPIRVSimulator::InvocationDependence::Varying },
        },
        ::SPIRVSimulator::InvocationDependence::Varying,
        {
            ::SPIRVSimulator::InvocationDependence::Varying,
            ::SPIRVSimulator::InvocationDependence::Unknown,
        },
        {
            ::SPIRVSimulator::InvocationDependence::Unknown,
            {
                ::SPIRVSimulator::InvocationDependence::Invariant,
                ::SPIRVSimulator::InvocationDependence::Unknown,
            },
            {
                { { 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
                { { 1 }, ::SPIRVSimulator::InvocationDependence::Unknown },
            },
        },
        ::SPIRVSimulator::InvocationDependence::Invariant,
    };
}

SubtreeProjectionTestCase MakeNestedStructSubtreeCase()
{
    const auto range_a =
        MakeProjectionRange(true,
                            true,
                            80,
                            95,
                            1,
                            ::SPIRVSimulator::kInvocationComponentX | ::SPIRVSimulator::kInvocationComponentY,
                            ::SPIRVSimulator::kInvocationComponentZ);
    const auto range_b = MakeProjectionRange(false, false, 120, 132, 4);
    const auto sibling_range =
        MakeProjectionRange(true,
                            true,
                            200,
                            209,
                            1,
                            ::SPIRVSimulator::kInvocationComponentY,
                            ::SPIRVSimulator::kInvocationComponentX | ::SPIRVSimulator::kInvocationComponentZ);

    return SubtreeProjectionTestCase{
        "NestedStructSubtree",
        SubtreeProjectionShape::NestedStructSubtree,
        { 1 },
        {
            { { 1, 2, 0 }, range_a },
            { { 1, 2, 1 }, range_b },
            { { 3, 0 }, sibling_range },
        },
        {
            {},
            {},
            {
                { { 2, 0 }, range_a },
                { { 2, 1 }, range_b },
            },
        },
        { 2, 1 },
        range_b,
        42,
        {
            { { 1, 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
            { { 1, 1 }, ::SPIRVSimulator::InvocationDependence::Unknown },
            { { 1, 2, 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
            { { 1, 2, 1 }, ::SPIRVSimulator::InvocationDependence::Varying },
            { { 3, 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
        },
        ::SPIRVSimulator::InvocationDependence::Varying,
        {
            ::SPIRVSimulator::InvocationDependence::Invariant,
            ::SPIRVSimulator::InvocationDependence::Varying,
            ::SPIRVSimulator::InvocationDependence::Invariant,
            ::SPIRVSimulator::InvocationDependence::Invariant,
        },
        {
            ::SPIRVSimulator::InvocationDependence::Varying,
            {
                ::SPIRVSimulator::InvocationDependence::Invariant,
                ::SPIRVSimulator::InvocationDependence::Unknown,
                ::SPIRVSimulator::InvocationDependence::Varying,
            },
            {
                { { 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
                { { 1 }, ::SPIRVSimulator::InvocationDependence::Unknown },
                { { 2, 0 }, ::SPIRVSimulator::InvocationDependence::Invariant },
                { { 2, 1 }, ::SPIRVSimulator::InvocationDependence::Varying },
            },
        },
        ::SPIRVSimulator::InvocationDependence::Varying,
    };
}

const SubtreeProjectionTestCase kStructMemberVectorCase  = MakeStructMemberVectorCase();
const SubtreeProjectionTestCase kStructMemberArrayCase   = MakeStructMemberArrayCase();
const SubtreeProjectionTestCase kNestedStructSubtreeCase = MakeNestedStructSubtreeCase();

struct SubtreeProjectionValueSetup
{
    ::SPIRVSimulator::Value source_value;
    ::SPIRVSimulator::Value extracted_value;
    ::SPIRVSimulator::Type  extracted_type;
    uint32_t                extracted_type_id;
};

SubtreeProjectionValueSetup MakeSubtreeProjectionValueSetup(SubtreeProjectionShape shape)
{
    switch (shape)
    {
        case SubtreeProjectionShape::StructMemberVector:
        {
            auto selected = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 11, 22 });
            auto source   = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ selected, uint64_t{ 99 } });
            return SubtreeProjectionValueSetup{
                source,
                selected,
                ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 2),
                CommonTypes::uvec2,
            };
        }
        case SubtreeProjectionShape::StructMemberArray:
        {
            auto selected = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 31 }, uint64_t{ 32 } });
            auto source = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 98 }, selected });
            return SubtreeProjectionValueSetup{
                source,
                selected,
                ::SPIRVSimulator::Type::Array(CommonTypes::u64, 2),
                7010,
            };
        }
        case SubtreeProjectionShape::NestedStructSubtree:
        {
            auto nested_vector = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 41, 42 });
            auto selected =
                std::make_shared<::SPIRVSimulator::AggregateV>(std::initializer_list<::SPIRVSimulator::Value>{
                    uint64_t{ 5 },
                    uint64_t{ 6 },
                    nested_vector,
                });
            auto sibling = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 97 } });
            auto source = std::make_shared<::SPIRVSimulator::AggregateV>(std::initializer_list<::SPIRVSimulator::Value>{
                uint64_t{ 4 },
                selected,
                uint64_t{ 7 },
                sibling,
            });
            return SubtreeProjectionValueSetup{
                source,
                selected,
                ::SPIRVSimulator::Type::Struct(7011),
                7011,
            };
        }
    }

    return {};
}

ExpectedInvocationDependenceMetadataSpec
MakeUniformInvocationDependenceMetadata(SubtreeProjectionShape shape, ::SPIRVSimulator::InvocationDependence dependence)
{
    ExpectedInvocationDependenceMetadataSpec expected;
    expected.invocation_dependence = dependence;

    switch (shape)
    {
        case SubtreeProjectionShape::StructMemberVector:
            expected.component_invocation_dependence.assign(2, dependence);
            break;
        case SubtreeProjectionShape::StructMemberArray:
            expected.component_invocation_dependence.assign(2, dependence);
            expected.subobject_invocation_dependence = {
                { { 0 }, dependence },
                { { 1 }, dependence },
            };
            break;
        case SubtreeProjectionShape::NestedStructSubtree:
            expected.component_invocation_dependence.assign(3, dependence);
            expected.subobject_invocation_dependence = {
                { { 0 }, dependence },
                { { 1 }, dependence },
                { { 2, 0 }, dependence },
                { { 2, 1 }, dependence },
            };
            break;
    }

    return expected;
}

void ExpectProjectionRange(const ::SPIRVSimulator::DenseValueRangeInfo& actual,
                           const ::SPIRVSimulator::DenseValueRangeInfo& expected)
{
    EXPECT_EQ(actual.valid, expected.valid);
    EXPECT_EQ(actual.thread_dependent, expected.thread_dependent);
    EXPECT_EQ(actual.dense_range, expected.dense_range);
    EXPECT_EQ(actual.min, expected.min);
    EXPECT_EQ(actual.max, expected.max);
    EXPECT_EQ(actual.stride, expected.stride);
    EXPECT_EQ(actual.invocation_linearization.workgroup_id_component_mask,
              expected.invocation_linearization.workgroup_id_component_mask);
    EXPECT_EQ(actual.invocation_linearization.local_invocation_id_component_mask,
              expected.invocation_linearization.local_invocation_id_component_mask);
}

void ExpectProjectionMetadata(const ::SPIRVSimulator::ValueMetadata& actual, const ExpectedValueMetadataSpec& expected)
{
    ExpectProjectionRange(actual.value_range, expected.value_range);

    ASSERT_EQ(actual.component_ranges.size(), expected.component_ranges.size());
    for (size_t i = 0; i < expected.component_ranges.size(); ++i)
    {
        ExpectProjectionRange(actual.component_ranges[i], expected.component_ranges[i]);
    }

    ASSERT_EQ(actual.subobject_scalar_ranges.size(), expected.subobject_scalar_ranges.size());
    for (size_t i = 0; i < expected.subobject_scalar_ranges.size(); ++i)
    {
        EXPECT_EQ(actual.subobject_scalar_ranges[i].path, expected.subobject_scalar_ranges[i].path);
        ExpectProjectionRange(actual.subobject_scalar_ranges[i].value_range,
                              expected.subobject_scalar_ranges[i].value_range);
    }

    EXPECT_FALSE(actual.address_range_valid);
}

void ExpectInvocationDependenceMetadata(const ::SPIRVSimulator::ValueMetadata&          actual,
                                        const ExpectedInvocationDependenceMetadataSpec& expected)
{
    EXPECT_EQ(actual.invocation_dependence, expected.invocation_dependence);
    EXPECT_EQ(actual.component_invocation_dependence, expected.component_invocation_dependence);

    ASSERT_EQ(actual.subobject_invocation_dependence.size(), expected.subobject_invocation_dependence.size());
    for (size_t i = 0; i < expected.subobject_invocation_dependence.size(); ++i)
    {
        EXPECT_EQ(actual.subobject_invocation_dependence[i].path, expected.subobject_invocation_dependence[i].path);
        EXPECT_EQ(actual.subobject_invocation_dependence[i].dependence,
                  expected.subobject_invocation_dependence[i].dependence);
    }
}

void SeedStaleProjectionMetadata(::SPIRVSimulator::ValueMetadata& metadata)
{
    const auto stale_range                   = MakeProjectionRange(true, true, 900, 999, 1);
    metadata.value_range                     = stale_range;
    metadata.component_ranges                = { stale_range };
    metadata.subobject_scalar_ranges         = { { { 9 }, stale_range } };
    metadata.address_range_valid             = true;
    metadata.address_range_min               = 900;
    metadata.address_range_max               = 999;
    metadata.invocation_dependence           = ::SPIRVSimulator::InvocationDependence::Varying;
    metadata.component_invocation_dependence = {
        ::SPIRVSimulator::InvocationDependence::Varying,
    };
    metadata.subobject_invocation_dependence = {
        { { 9 }, ::SPIRVSimulator::InvocationDependence::Varying },
    };
}

void ExpectProjectedConcreteValue(SubtreeProjectionShape shape, const ::SPIRVSimulator::Value& value)
{
    switch (shape)
    {
        case SubtreeProjectionShape::StructMemberVector:
        {
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(value));
            const auto& vector = *std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(value);
            ASSERT_EQ(vector.elems.size(), 2u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[0]), 11u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[1]), 22u);
            break;
        }
        case SubtreeProjectionShape::StructMemberArray:
        {
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value));
            const auto& array = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value);
            ASSERT_EQ(array.elems.size(), 2u);
            EXPECT_EQ(std::get<uint64_t>(array.elems[0]), 31u);
            EXPECT_EQ(std::get<uint64_t>(array.elems[1]), 32u);
            break;
        }
        case SubtreeProjectionShape::NestedStructSubtree:
        {
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value));
            const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value);
            ASSERT_EQ(aggregate.elems.size(), 3u);
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(aggregate.elems[2]));
            const auto& vector = *std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(aggregate.elems[2]);
            ASSERT_EQ(vector.elems.size(), 2u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[0]), 41u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[1]), 42u);
            break;
        }
    }
}

} // namespace

// execute aggregate-valued OpCompositeExtract followed by a scalar
// extract, proving that descendant metadata is rebased and remains consumable.
class CompositeExtractSubtreeMetadataTests : public SPIRVSimulatorMockBase,
                                             public TestWithParam<SubtreeProjectionTestCase>
{
  protected:
    static constexpr uint32_t kSourceId          = 7000;
    static constexpr uint32_t kExtractedResultId = 7001;
    static constexpr uint32_t kScalarResultId    = 7002;
    static constexpr uint32_t kMaximumTestId     = 7002;

    void SetUp() override
    {
        // The mock-only default constructor does not initialize optional
        // runtime services used by production simulator construction.
        memory_flag_tracker_ = nullptr;
        verbose_             = false;

        if (values_.size() <= kMaximumTestId)
        {
            values_.resize(kMaximumTestId + 1);
        }
        if (value_meta_.size() <= kMaximumTestId)
        {
            value_meta_.resize(kMaximumTestId + 1);
        }

        const SubtreeProjectionValueSetup value_setup = MakeSubtreeProjectionValueSetup(GetParam().shape);
        values_[kSourceId]                            = value_setup.source_value;
        extracted_result_type_                        = value_setup.extracted_type;
        extracted_result_type_id_                     = value_setup.extracted_type_id;
        types_[extracted_result_type_id_]             = extracted_result_type_;
        values_[2]                                    = uint64_t{ 2 };
        if (GetParam().shape == SubtreeProjectionShape::NestedStructSubtree)
        {
            struct_members_[extracted_result_type_id_] = {
                CommonTypes::u64,
                CommonTypes::u64,
                CommonTypes::uvec2,
            };
        }

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_[id];
        });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .Times(2)
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByResultId(kExtractedResultId)).WillRepeatedly(ReturnRef(extracted_result_type_));
        EXPECT_CALL(*this, GetTypeByResultId(kScalarResultId)).WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, GetTypeByTypeId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(id);
        });
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>())).Times(2);
    }

    uint32_t GetTypeID(uint32_t result_id) const override
    {
        if (result_id == kExtractedResultId)
        {
            return extracted_result_type_id_;
        }
        if (result_id == kScalarResultId)
        {
            return CommonTypes::u64;
        }
        return ::SPIRVSimulator::SPIRVSimulator::GetTypeID(result_id);
    }

    bool ExecuteCompositeExtract(uint32_t                     result_type_id,
                                 uint32_t                     result_id,
                                 uint32_t                     composite_id,
                                 const std::vector<uint32_t>& indexes)
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpCompositeExtract),
            result_type_id,
            result_id,
            composite_id,
        };
        words.insert(words.end(), indexes.begin(), indexes.end());

        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = spv::Op::OpCompositeExtract,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        return ExecuteInstruction(instruction);
    }

    ::SPIRVSimulator::Type extracted_result_type_;
    uint32_t               extracted_result_type_id_ = 0;
};

TEST_P(CompositeExtractSubtreeMetadataTests, RebasesDescendantMetadata)
{
    value_meta_[kSourceId].subobject_scalar_ranges         = GetParam().source_subobject_ranges;
    value_meta_[kSourceId].invocation_dependence           = GetParam().source_invocation_dependence;
    value_meta_[kSourceId].component_invocation_dependence = GetParam().source_component_invocation_dependence;
    value_meta_[kSourceId].subobject_invocation_dependence = GetParam().source_subobject_invocation_dependence;
    SeedStaleProjectionMetadata(value_meta_[kExtractedResultId]);
    SeedStaleProjectionMetadata(value_meta_[kScalarResultId]);

    ASSERT_TRUE(
        ExecuteCompositeExtract(extracted_result_type_id_, kExtractedResultId, kSourceId, GetParam().selected_path));

    ExpectProjectedConcreteValue(GetParam().shape, values_[kExtractedResultId]);
    ExpectProjectionMetadata(value_meta_[kExtractedResultId], GetParam().expected_extracted_metadata);
    ExpectInvocationDependenceMetadata(value_meta_[kExtractedResultId],
                                       GetParam().expected_extracted_invocation_dependence);

    ASSERT_TRUE(ExecuteCompositeExtract(
        CommonTypes::u64, kScalarResultId, kExtractedResultId, GetParam().followup_scalar_path));

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kScalarResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kScalarResultId]), GetParam().expected_scalar_value);

    ExpectedValueMetadataSpec expected_scalar_metadata;
    expected_scalar_metadata.value_range = GetParam().expected_scalar_range;
    ExpectProjectionMetadata(value_meta_[kScalarResultId], expected_scalar_metadata);
    EXPECT_EQ(value_meta_[kScalarResultId].invocation_dependence, GetParam().expected_scalar_invocation_dependence);
    EXPECT_TRUE(value_meta_[kScalarResultId].component_invocation_dependence.empty());
    EXPECT_TRUE(value_meta_[kScalarResultId].subobject_invocation_dependence.empty());
}

TEST_P(CompositeExtractSubtreeMetadataTests, ExpandsInvariantSourceSummary)
{
    ::SPIRVSimulator::ValueMetadata& source_meta = value_meta_[kSourceId];
    source_meta.invocation_dependence            = ::SPIRVSimulator::InvocationDependence::Invariant;
    SeedStaleProjectionMetadata(value_meta_[kExtractedResultId]);
    SeedStaleProjectionMetadata(value_meta_[kScalarResultId]);

    ASSERT_TRUE(
        ExecuteCompositeExtract(extracted_result_type_id_, kExtractedResultId, kSourceId, GetParam().selected_path));

    const ExpectedInvocationDependenceMetadataSpec expected =
        MakeUniformInvocationDependenceMetadata(GetParam().shape, ::SPIRVSimulator::InvocationDependence::Invariant);
    ExpectInvocationDependenceMetadata(value_meta_[kExtractedResultId], expected);

    ASSERT_TRUE(ExecuteCompositeExtract(
        CommonTypes::u64, kScalarResultId, kExtractedResultId, GetParam().followup_scalar_path));
    EXPECT_EQ(value_meta_[kScalarResultId].invocation_dependence, ::SPIRVSimulator::InvocationDependence::Invariant);
}

TEST_P(CompositeExtractSubtreeMetadataTests, KeepsCoarseVaryingSummaryAtWholeValue)
{
    ::SPIRVSimulator::ValueMetadata& source_meta = value_meta_[kSourceId];
    source_meta.component_invocation_dependence.assign(GetParam().selected_path[0] + 1,
                                                       ::SPIRVSimulator::InvocationDependence::Unknown);
    source_meta.component_invocation_dependence[GetParam().selected_path[0]] =
        ::SPIRVSimulator::InvocationDependence::Varying;
    SeedStaleProjectionMetadata(value_meta_[kExtractedResultId]);
    SeedStaleProjectionMetadata(value_meta_[kScalarResultId]);

    ASSERT_TRUE(
        ExecuteCompositeExtract(extracted_result_type_id_, kExtractedResultId, kSourceId, GetParam().selected_path));

    ExpectedInvocationDependenceMetadataSpec expected =
        MakeUniformInvocationDependenceMetadata(GetParam().shape, ::SPIRVSimulator::InvocationDependence::Unknown);
    expected.invocation_dependence = ::SPIRVSimulator::InvocationDependence::Varying;
    ExpectInvocationDependenceMetadata(value_meta_[kExtractedResultId], expected);

    ASSERT_TRUE(ExecuteCompositeExtract(
        CommonTypes::u64, kScalarResultId, kExtractedResultId, GetParam().followup_scalar_path));
    EXPECT_EQ(value_meta_[kScalarResultId].invocation_dependence, ::SPIRVSimulator::InvocationDependence::Unknown);
}

INSTANTIATE_TEST_SUITE_P(SubtreeShapes,
                         CompositeExtractSubtreeMetadataTests,
                         Values(kStructMemberVectorCase, kStructMemberArrayCase, kNestedStructSubtreeCase),
                         [](const TestParamInfo<SubtreeProjectionTestCase>& info) {
                             return std::string(info.param.name);
                         });

namespace
{

struct StoredLoadProjectionTestCase
{
    const char* name;
    spv::StorageClass storage_class;
    const SubtreeProjectionTestCase* projection;
};

const StoredLoadProjectionTestCase kFunctionVectorMemberCase{
    "FunctionVectorMember",
    spv::StorageClass::StorageClassFunction,
    &kStructMemberVectorCase,
};

const StoredLoadProjectionTestCase kPrivateArrayMemberCase{
    "PrivateArrayMember",
    spv::StorageClass::StorageClassPrivate,
    &kStructMemberArrayCase,
};

const StoredLoadProjectionTestCase kWorkgroupNestedStructCase{
    "WorkgroupNestedStruct",
    spv::StorageClass::StorageClassWorkgroup,
    &kNestedStructSubtreeCase,
};

} // namespace

// restore an aggregate-valued descendant from an ancestor stored
// snapshot, then extract a scalar to verify the projected metadata end to end.
class StoredAggregateLoadProjectionTests : public SPIRVSimulatorMockBase,
                                           public TestWithParam<StoredLoadProjectionTestCase>
{
  public:
    MOCK_METHOD((std::pair<std::byte*, uint64_t>),
                ResolvePointerV,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (override));

  protected:
    static constexpr uint32_t kWholeObjectTypeId = 7100;
    static constexpr uint32_t kAncestorPointerId = 7101;
    static constexpr uint32_t kLoadPointerId = 7102;
    static constexpr uint32_t kStoredValueId = 7103;
    static constexpr uint32_t kLoadResultId = 7104;
    static constexpr uint32_t kScalarResultId = 7105;
    static constexpr uint32_t kMaximumTestId = 7105;

    void SetUp() override
    {
        // Keep optional external tracking and diagnostic paths disabled; this
        // suite only exercises internal stored-value metadata restoration.
        memory_flag_tracker_ = nullptr;
        verbose_ = false;

        if (values_.size() <= kMaximumTestId)
        {
            values_.resize(kMaximumTestId + 1);
        }
        if (value_meta_.size() <= kMaximumTestId)
        {
            value_meta_.resize(kMaximumTestId + 1);
        }

        value_setup_ = MakeSubtreeProjectionValueSetup(GetParam().projection->shape);
        types_[value_setup_.extracted_type_id] = value_setup_.extracted_type;
        values_[2] = uint64_t{ 2 };
        if (GetParam().projection->shape == SubtreeProjectionShape::NestedStructSubtree)
        {
            struct_members_[value_setup_.extracted_type_id] = {
                CommonTypes::u64,
                CommonTypes::u64,
                CommonTypes::uvec2,
            };
        }

        const ::SPIRVSimulator::PointerV ancestor_pointer = MakePointer({});
        const ::SPIRVSimulator::PointerV load_pointer =
            MakePointer(GetParam().projection->selected_path);
        values_[kAncestorPointerId] = ancestor_pointer;
        values_[kLoadPointerId] = load_pointer;

        ::SPIRVSimulator::StoredValueMetadataSnapshot stored_metadata;
        const auto whole_object_sentinel =
            MakeProjectionRange(true, true, 800, 899, 1);
        stored_metadata.value_range = whole_object_sentinel;
        stored_metadata.component_ranges = { whole_object_sentinel };
        stored_metadata.subobject_scalar_ranges =
            GetParam().projection->source_subobject_ranges;
        stored_metadata.invocation_dependence =
            GetParam().projection->source_invocation_dependence;
        stored_metadata.component_invocation_dependence =
            GetParam().projection->source_component_invocation_dependence;
        stored_metadata.subobject_invocation_dependence =
            GetParam().projection->source_subobject_invocation_dependence;
        stored_metadata.address_range_valid = true;
        stored_metadata.address_range_min = 800;
        stored_metadata.address_range_max = 899;
        stored_metadata.address_range_stride = 1;
        stored_metadata.address_element_size = 8;

        values_stored_[kAncestorPointerId] = ::SPIRVSimulator::StoredValueRecord{
            kStoredValueId,
            ancestor_pointer,
            stored_metadata,
        };

        EXPECT_CALL(*this, GetValue(_))
            .WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
                return values_[id];
            });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .Times(2)
            .WillRepeatedly([this](uint32_t id,
                                   const ::SPIRVSimulator::Value& value,
                                   bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByResultId(kLoadResultId))
            .WillRepeatedly(ReturnRef(value_setup_.extracted_type));
        EXPECT_CALL(*this, GetTypeByResultId(kScalarResultId))
            .WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, GetTypeByTypeId(_))
            .WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
                return types_.at(id);
            });
        EXPECT_CALL(*this, ResolvePointerV(_))
            .Times(1)
            .WillOnce(Return(std::pair<std::byte*, uint64_t>{ backing_.data(), 0 }));
        EXPECT_CALL(*this, ReadPointer(_))
            .Times(1)
            .WillOnce(Return(std::optional<::SPIRVSimulator::Value>{ value_setup_.extracted_value }));
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>()))
            .Times(1);
    }

    ::SPIRVSimulator::PointerV MakePointer(std::vector<uint32_t> idx_path) const
    {
        return ::SPIRVSimulator::PointerV{
            1,
            0,
            kWholeObjectTypeId,
            kAncestorPointerId,
            static_cast<uint32_t>(GetParam().storage_class),
            std::move(idx_path),
        };
    }

    uint32_t GetTypeID(uint32_t result_id) const override
    {
        if (result_id == kLoadResultId)
        {
            return value_setup_.extracted_type_id;
        }
        if (result_id == kScalarResultId)
        {
            return CommonTypes::u64;
        }
        return SPIRVSimulatorMockBase::GetTypeID(result_id);
    }

    bool ExecuteLoad()
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpLoad),
            value_setup_.extracted_type_id,
            kLoadResultId,
            kLoadPointerId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpLoad,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        return ExecuteInstruction(instruction);
    }

    bool ExecuteScalarExtract()
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpCompositeExtract),
            CommonTypes::u64,
            kScalarResultId,
            kLoadResultId,
        };
        words.insert(words.end(),
                     GetParam().projection->followup_scalar_path.begin(),
                     GetParam().projection->followup_scalar_path.end());

        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpCompositeExtract,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        return ExecuteInstruction(instruction);
    }

    std::array<std::byte, 1> backing_{};
    SubtreeProjectionValueSetup value_setup_;
};

TEST_P(StoredAggregateLoadProjectionTests, ProjectsAncestorSnapshotIntoLoadedSubobject)
{
    SeedStaleProjectionMetadata(value_meta_[kLoadResultId]);
    SeedStaleProjectionMetadata(value_meta_[kScalarResultId]);

    ASSERT_TRUE(ExecuteLoad());

    ExpectProjectedConcreteValue(GetParam().projection->shape,
                                 values_[kLoadResultId]);
    ExpectProjectionMetadata(value_meta_[kLoadResultId],
                             GetParam().projection->expected_extracted_metadata);
    ExpectInvocationDependenceMetadata(
        value_meta_[kLoadResultId],
        GetParam().projection->expected_extracted_invocation_dependence);
    EXPECT_TRUE(values_stored_.contains(kAncestorPointerId));
    EXPECT_FALSE(values_stored_.contains(kLoadPointerId));

    ASSERT_TRUE(ExecuteScalarExtract());

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kScalarResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kScalarResultId]),
              GetParam().projection->expected_scalar_value);

    ExpectedValueMetadataSpec expected_scalar_metadata;
    expected_scalar_metadata.value_range =
        GetParam().projection->expected_scalar_range;
    ExpectProjectionMetadata(value_meta_[kScalarResultId], expected_scalar_metadata);
    EXPECT_EQ(value_meta_[kScalarResultId].invocation_dependence,
              GetParam().projection->expected_scalar_invocation_dependence);
    EXPECT_TRUE(value_meta_[kScalarResultId].component_invocation_dependence.empty());
    EXPECT_TRUE(value_meta_[kScalarResultId].subobject_invocation_dependence.empty());
}

INSTANTIATE_TEST_SUITE_P(
    InternalStorageClasses,
    StoredAggregateLoadProjectionTests,
    Values(kFunctionVectorMemberCase,
           kPrivateArrayMemberCase,
           kWorkgroupNestedStructCase),
    [](const TestParamInfo<StoredLoadProjectionTestCase>& info) {
        return std::string(info.param.name);
    });

// Helper-level coverage for ancestor/descendant overlap invalidation in both
// stored-value maps. These tests do not execute OpStore or OpLoad.
class StoredValueInvalidationTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    static ::SPIRVSimulator::PointerV MakePointer(uint64_t pointer_handle,
                                                  std::vector<uint32_t> idx_path)
    {
        return ::SPIRVSimulator::PointerV{
            pointer_handle,
            0,
            100,
            200,
            spv::StorageClass::StorageClassFunction,
            std::move(idx_path),
        };
    }

    static ::SPIRVSimulator::PointerLocationKey MakeLocationKey(
        const ::SPIRVSimulator::PointerV& pointer,
        uint64_t byte_offset)
    {
        return ::SPIRVSimulator::PointerLocationKey{
            pointer.storage_class,
            pointer.base_result_id,
            pointer.pointer_handle,
            byte_offset,
            pointer.idx_path,
        };
    }

    void AddStoredValue(uint32_t pointer_id,
                        uint32_t stored_result_id,
                        const ::SPIRVSimulator::PointerV& pointer,
                        uint64_t byte_offset)
    {
        ::SPIRVSimulator::StoredValueRecord record{ stored_result_id, pointer, {} };
        values_stored_[pointer_id] = record;
        values_stored_by_memory_location_[MakeLocationKey(pointer, byte_offset)] = record;
    }

    void Invalidate(const ::SPIRVSimulator::PointerV& pointer)
    {
        InvalidateOverlappingStoredValues(pointer);
    }

    bool HasPointerRecord(uint32_t pointer_id) const
    {
        return values_stored_.contains(pointer_id);
    }

    bool HasLocationRecord(const ::SPIRVSimulator::PointerV& pointer,
                           uint64_t byte_offset) const
    {
        return values_stored_by_memory_location_.contains(
            MakeLocationKey(pointer, byte_offset));
    }
};

TEST_F(StoredValueInvalidationTests, WholeObjectTargetInvalidatesSubobjectRecordsInBothMaps)
{
    const auto whole_object = MakePointer(1, {});
    const auto first_member = MakePointer(1, { 0 });
    const auto nested_member = MakePointer(1, { 1, 0 });
    const auto unrelated_member = MakePointer(2, { 0 });

    AddStoredValue(10, 100, first_member, 4);
    AddStoredValue(11, 101, nested_member, 8);
    AddStoredValue(12, 102, unrelated_member, 4);

    Invalidate(whole_object);

    EXPECT_FALSE(HasPointerRecord(10));
    EXPECT_FALSE(HasLocationRecord(first_member, 4));
    EXPECT_FALSE(HasPointerRecord(11));
    EXPECT_FALSE(HasLocationRecord(nested_member, 8));
    EXPECT_TRUE(HasPointerRecord(12));
    EXPECT_TRUE(HasLocationRecord(unrelated_member, 4));
}

TEST_F(StoredValueInvalidationTests, SubobjectTargetInvalidatesAncestorAndPreservesSiblingInBothMaps)
{
    const auto whole_object = MakePointer(1, {});
    const auto first_member = MakePointer(1, { 0 });
    const auto first_member_leaf = MakePointer(1, { 0, 1 });
    const auto sibling_member = MakePointer(1, { 1 });

    AddStoredValue(20, 200, whole_object, 0);
    AddStoredValue(21, 201, first_member_leaf, 4);
    AddStoredValue(22, 202, sibling_member, 8);

    Invalidate(first_member);

    EXPECT_FALSE(HasPointerRecord(20));
    EXPECT_FALSE(HasLocationRecord(whole_object, 0));
    EXPECT_FALSE(HasPointerRecord(21));
    EXPECT_FALSE(HasLocationRecord(first_member_leaf, 4));
    EXPECT_TRUE(HasPointerRecord(22));
    EXPECT_TRUE(HasLocationRecord(sibling_member, 8));
}


namespace
{

enum class StoredTarget
{
    WholeObject,
    FirstMember,
    SecondMember,
};

::SPIRVSimulator::DenseValueRangeInfo MakeRange(
    uint64_t min,
    uint64_t max,
    uint8_t workgroup_id_component_mask = 0,
    uint8_t local_invocation_id_component_mask = 0)
{
    ::SPIRVSimulator::DenseValueRangeInfo range = {
        true, true, true, min, max, 1
    };
    range.invocation_linearization.workgroup_id_component_mask =
        workgroup_id_component_mask;
    range.invocation_linearization.local_invocation_id_component_mask =
        local_invocation_id_component_mask;
    return range;
}

} // namespace

// shared integration environment for store/load ordering.
// The fixture mocks concrete Pair memory, executes the real OpStore and OpLoad
// paths, and never pre-populates either stored-value map.
class StoreLoadMetadataTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD((std::pair<std::byte*, uint64_t>),
                ResolvePointerV,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(bool,
                WritePointer,
                (const ::SPIRVSimulator::PointerV& pointer,
                 const ::SPIRVSimulator::Value& value),
                (override));
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (override));
    MOCK_METHOD((std::vector<::SPIRVSimulator::DataSourceBits>),
                FindDataSourcesFromResultID,
                (uint32_t result_id, uint32_t* property_flags),
                (override));

  protected:
    static constexpr uint32_t kPairTypeId = 6000;
    static constexpr uint32_t kWholePointerId = 6001;
    static constexpr uint32_t kFirstMemberPointerId = 6002;
    static constexpr uint32_t kSecondMemberPointerId = 6003;
    static constexpr uint32_t kStoreValueBaseId = 6010;
    static constexpr uint32_t kLoadResultBaseId = 6020;
    static constexpr uint32_t kMaximumTestId = 6030;

    void SetUp() override
    {
        // Direct opcode execution has no Run() call stack and no external
        // tracker, so keep those optional production paths disabled.
        memory_flag_tracker_ = nullptr;
        verbose_ = false;

        if (values_.size() <= kMaximumTestId)
        {
            values_.resize(kMaximumTestId + 1);
        }
        if (value_meta_.size() <= kMaximumTestId)
        {
            value_meta_.resize(kMaximumTestId + 1);
        }

        types_[kPairTypeId] = ::SPIRVSimulator::Type::Struct(kPairTypeId);

        values_[kWholePointerId] = MakePointer({});
        values_[kFirstMemberPointerId] = MakePointer({ 0 });
        values_[kSecondMemberPointerId] = MakePointer({ 1 });

        EXPECT_CALL(*this, GetValue(_))
            .WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
                return values_[id];
            });
        EXPECT_CALL(*this, SetValue(_, _, _))
            .WillRepeatedly([this](uint32_t id,
                                   const ::SPIRVSimulator::Value& value,
                                   bool clear_meta) {
                ++set_value_count_;
                SetValueWithBaseImplementation(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByResultId(_))
            .WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, ResolvePointerV(_))
            .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
                return std::pair<std::byte*, uint64_t>{ backing_.data(), GetByteOffset(pointer) };
            });
        EXPECT_CALL(*this, WritePointer(_, _))
            .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer,
                                   const ::SPIRVSimulator::Value& value) -> bool {
                ++write_count_;
                WriteConcreteValue(pointer, value);
                return true;
            });
        EXPECT_CALL(*this, ReadPointer(_))
            .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer)
                                -> std::optional<::SPIRVSimulator::Value> {
                ++read_count_;
                return ReadConcreteValue(pointer);
            });
        EXPECT_CALL(*this, FindDataSourcesFromResultID(_, _))
            .WillRepeatedly([](uint32_t, uint32_t* property_flags) {
                if (property_flags != nullptr)
                {
                    *property_flags = 0;
                }
                return std::vector<::SPIRVSimulator::DataSourceBits>{};
            });
    }

    void SetValueWithBaseImplementation(uint32_t id,
                                        const ::SPIRVSimulator::Value& value,
                                        bool clear_meta)
    {
        ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
    }

    ::SPIRVSimulator::PointerV MakePointer(std::vector<uint32_t> idx_path) const
    {
        return ::SPIRVSimulator::PointerV{
            1,
            0,
            kPairTypeId,
            kWholePointerId,
            storage_class_,
            std::move(idx_path),
        };
    }

    void SetPointerStorageClass(uint32_t storage_class)
    {
        storage_class_ = storage_class;
        values_[kWholePointerId] = MakePointer({});
        values_[kFirstMemberPointerId] = MakePointer({ 0 });
        values_[kSecondMemberPointerId] = MakePointer({ 1 });
    }

    static uint64_t GetByteOffset(const ::SPIRVSimulator::PointerV& pointer)
    {
        if (pointer.idx_path == std::vector<uint32_t>{ 1 })
        {
            return sizeof(uint64_t);
        }
        return 0;
    }

    static uint32_t GetPointerId(StoredTarget target)
    {
        switch (target)
        {
            case StoredTarget::WholeObject:
                return kWholePointerId;
            case StoredTarget::FirstMember:
                return kFirstMemberPointerId;
            case StoredTarget::SecondMember:
                return kSecondMemberPointerId;
        }
        return 0;
    }

    void WriteConcreteValue(const ::SPIRVSimulator::PointerV& pointer,
                            const ::SPIRVSimulator::Value& value)
    {
        if (pointer.idx_path.empty())
        {
            const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value);
            pair_values_[0] = std::get<uint64_t>(aggregate.elems[0]);
            pair_values_[1] = std::get<uint64_t>(aggregate.elems[1]);
            return;
        }

        pair_values_[pointer.idx_path[0]] = std::get<uint64_t>(value);
    }

    ::SPIRVSimulator::Value ReadConcreteValue(const ::SPIRVSimulator::PointerV& pointer) const
    {
        if (pointer.idx_path.empty())
        {
            return std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ pair_values_[0], pair_values_[1] });
        }

        return pair_values_[pointer.idx_path[0]];
    }

    void ExecuteStore(StoredTarget target, uint32_t value_id)
    {
        std::vector<uint32_t> words = { 0, GetPointerId(target), value_id };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpStore,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        Op_Store(instruction);
    }

    uint32_t ExecuteLoad(StoredTarget target, uint32_t type_id)
    {
        const uint32_t result_id = next_load_result_id_++;
        std::vector<uint32_t> words = {
            0,
            type_id,
            result_id,
            GetPointerId(target),
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpLoad,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        Op_Load(instruction);
        return result_id;
    }

    void StoreScalar(StoredTarget target,
                     uint64_t value,
                     const ::SPIRVSimulator::DenseValueRangeInfo& range,
                     ::SPIRVSimulator::InvocationDependence dependence =
                         ::SPIRVSimulator::InvocationDependence::Unknown)
    {
        const uint32_t value_id = next_store_value_id_++;
        values_[value_id] = value;
        value_meta_[value_id] = {};
        value_meta_[value_id].value_range = range;
        value_meta_[value_id].invocation_dependence = dependence;
        ExecuteStore(target, value_id);
    }

    void StorePair(uint64_t first_value,
                   uint64_t second_value,
                   const ::SPIRVSimulator::DenseValueRangeInfo& first_range,
                   const ::SPIRVSimulator::DenseValueRangeInfo& second_range,
                   bool include_top_level_sentinels = false,
                   ::SPIRVSimulator::InvocationDependence first_dependence =
                       ::SPIRVSimulator::InvocationDependence::Unknown,
                   ::SPIRVSimulator::InvocationDependence second_dependence =
                       ::SPIRVSimulator::InvocationDependence::Unknown)
    {
        const uint32_t value_id = next_store_value_id_++;
        values_[value_id] = std::make_shared<::SPIRVSimulator::AggregateV>(
            std::initializer_list<::SPIRVSimulator::Value>{ first_value, second_value });

        ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[value_id];
        metadata = {};
        metadata.subobject_scalar_ranges = {
            ::SPIRVSimulator::SubobjectScalarRangeEntry{ { 0 }, first_range },
            ::SPIRVSimulator::SubobjectScalarRangeEntry{ { 1 }, second_range },
        };
        metadata.component_invocation_dependence = { first_dependence, second_dependence };
        metadata.subobject_invocation_dependence = {
            ::SPIRVSimulator::SubobjectInvocationDependenceEntry{ { 0 }, first_dependence },
            ::SPIRVSimulator::SubobjectInvocationDependenceEntry{ { 1 }, second_dependence },
        };
        if (first_dependence == ::SPIRVSimulator::InvocationDependence::Varying ||
            second_dependence == ::SPIRVSimulator::InvocationDependence::Varying)
        {
            metadata.invocation_dependence = ::SPIRVSimulator::InvocationDependence::Varying;
        }
        else if (first_dependence == ::SPIRVSimulator::InvocationDependence::Invariant &&
                 second_dependence == ::SPIRVSimulator::InvocationDependence::Invariant)
        {
            metadata.invocation_dependence = ::SPIRVSimulator::InvocationDependence::Invariant;
        }
        if (include_top_level_sentinels)
        {
            metadata.value_range = MakeRange(900, 909);
            metadata.component_ranges = { first_range, second_range };
        }

        ExecuteStore(StoredTarget::WholeObject, value_id);
    }

    uint32_t LoadScalar(StoredTarget target)
    {
        return ExecuteLoad(target, CommonTypes::u64);
    }

    uint32_t LoadPair()
    {
        return ExecuteLoad(StoredTarget::WholeObject, kPairTypeId);
    }

    static void ExpectRange(const ::SPIRVSimulator::DenseValueRangeInfo& actual,
                            const ::SPIRVSimulator::DenseValueRangeInfo& expected)
    {
        EXPECT_EQ(actual.valid, expected.valid);
        EXPECT_EQ(actual.thread_dependent, expected.thread_dependent);
        EXPECT_EQ(actual.dense_range, expected.dense_range);
        EXPECT_EQ(actual.min, expected.min);
        EXPECT_EQ(actual.max, expected.max);
        EXPECT_EQ(actual.stride, expected.stride);
        EXPECT_EQ(
            actual.invocation_linearization.workgroup_id_component_mask,
            expected.invocation_linearization.workgroup_id_component_mask);
        EXPECT_EQ(
            actual.invocation_linearization.local_invocation_id_component_mask,
            expected.invocation_linearization.local_invocation_id_component_mask);
    }

    void ExpectScalarResult(uint32_t result_id,
                            uint64_t expected_value,
                            const ::SPIRVSimulator::DenseValueRangeInfo& expected_range) const
    {
        ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[result_id]));
        EXPECT_EQ(std::get<uint64_t>(values_[result_id]), expected_value);

        const ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[result_id];
        ExpectRange(metadata.value_range, expected_range);
        EXPECT_TRUE(metadata.component_ranges.empty());
        EXPECT_TRUE(metadata.subobject_scalar_ranges.empty());
        EXPECT_FALSE(metadata.address_range_valid);
    }

    void ExpectPairResultWithoutMetadata(uint32_t result_id,
                                         uint64_t expected_first,
                                         uint64_t expected_second) const
    {
        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::AggregateV>>(
            values_[result_id]));
        const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(
            values_[result_id]);
        ASSERT_EQ(aggregate.elems.size(), 2u);
        EXPECT_EQ(std::get<uint64_t>(aggregate.elems[0]), expected_first);
        EXPECT_EQ(std::get<uint64_t>(aggregate.elems[1]), expected_second);

        const ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[result_id];
        ExpectRange(metadata.value_range, {});
        EXPECT_TRUE(metadata.component_ranges.empty());
        EXPECT_TRUE(metadata.subobject_scalar_ranges.empty());
        EXPECT_FALSE(metadata.address_range_valid);
    }

    void ExpectOperationCounts(size_t expected_stores, size_t expected_loads) const
    {
        EXPECT_EQ(write_count_, expected_stores);
        EXPECT_EQ(read_count_, expected_loads);
        EXPECT_EQ(set_value_count_, expected_loads);
    }

    std::array<std::byte, 16> backing_{};
    std::array<uint64_t, 2> pair_values_{};
    uint32_t storage_class_ = spv::StorageClass::StorageClassFunction;
    uint32_t next_store_value_id_ = kStoreValueBaseId;
    uint32_t next_load_result_id_ = kLoadResultBaseId;
    size_t write_count_ = 0;
    size_t read_count_ = 0;
    size_t set_value_count_ = 0;
};

// a later whole-object store must replace stale metadata from an
// earlier descendant store, and the scalar load must project the new leaf range.
TEST_F(StoreLoadMetadataTests, WholeObjectStoreReplacesDescendantMetadata)
{
    StoreScalar(StoredTarget::FirstMember, 11, MakeRange(10, 19));
    StorePair(101, 202, MakeRange(100, 109), MakeRange(200, 209));

    const uint32_t result_id = LoadScalar(StoredTarget::FirstMember);

    ExpectScalarResult(result_id, 101, MakeRange(100, 109));
    ExpectOperationCounts(2, 1);
}

// under the current conservative policy, a descendant store drops
// the old ancestor snapshot but remains available to an exact descendant load.
TEST_F(StoreLoadMetadataTests, DescendantStoreDropsAncestorSnapshot)
{
    StorePair(11, 22, MakeRange(10, 19), MakeRange(20, 29), true);
    StoreScalar(StoredTarget::FirstMember, 31, MakeRange(30, 39));

    const uint32_t whole_result_id = LoadPair();
    const uint32_t member_result_id = LoadScalar(StoredTarget::FirstMember);

    ExpectPairResultWithoutMetadata(whole_result_id, 31, 22);
    ExpectScalarResult(member_result_id, 31, MakeRange(30, 39));
    ExpectOperationCounts(2, 2);
}

// stores to non-overlapping sibling paths must retain independent
// reaching metadata for their subsequent exact loads.
TEST_F(StoreLoadMetadataTests, SiblingStoresRestoreIndependentMetadata)
{
    StoreScalar(StoredTarget::FirstMember, 41, MakeRange(40, 49));
    StoreScalar(StoredTarget::SecondMember, 52, MakeRange(50, 59));

    const uint32_t first_result_id = LoadScalar(StoredTarget::FirstMember);
    const uint32_t second_result_id = LoadScalar(StoredTarget::SecondMember);

    ExpectScalarResult(first_result_id, 41, MakeRange(40, 49));
    ExpectScalarResult(second_result_id, 52, MakeRange(50, 59));
    ExpectOperationCounts(2, 2);
}

TEST_F(StoreLoadMetadataTests, ScalarStoreLoadPreservesInvocationProvenance)
{
    const auto range = MakeRange(
        70,
        79,
        ::SPIRVSimulator::kInvocationComponentY,
        ::SPIRVSimulator::kInvocationComponentX |
            ::SPIRVSimulator::kInvocationComponentZ);

    StoreScalar(StoredTarget::FirstMember, 77, range);
    const uint32_t result_id = LoadScalar(StoredTarget::FirstMember);

    ExpectScalarResult(result_id, 77, range);
    ExpectOperationCounts(1, 1);
}

TEST_F(StoreLoadMetadataTests,
       WholeObjectStoreProjectionPreservesInvocationProvenance)
{
    const auto first_range = MakeRange(
        100,
        109,
        ::SPIRVSimulator::kInvocationComponentX,
        ::SPIRVSimulator::kInvocationComponentY);
    const auto second_range = MakeRange(
        200,
        209,
        ::SPIRVSimulator::kInvocationComponentZ,
        ::SPIRVSimulator::kInvocationComponentX |
            ::SPIRVSimulator::kInvocationComponentZ);

    StorePair(101, 202, first_range, second_range);
    const uint32_t first_result_id = LoadScalar(StoredTarget::FirstMember);
    const uint32_t second_result_id = LoadScalar(StoredTarget::SecondMember);

    ExpectScalarResult(first_result_id, 101, first_range);
    ExpectScalarResult(second_result_id, 202, second_range);
    ExpectOperationCounts(1, 2);
}

TEST_F(StoreLoadMetadataTests, FunctionAndPrivateLoadsRestoreInvocationDependence)
{
    StoreScalar(StoredTarget::FirstMember,
                61,
                MakeRange(60, 69),
                ::SPIRVSimulator::InvocationDependence::Varying);
    const uint32_t function_result_id = LoadScalar(StoredTarget::FirstMember);
    EXPECT_EQ(value_meta_[function_result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);

    SetPointerStorageClass(spv::StorageClass::StorageClassPrivate);
    StoreScalar(StoredTarget::FirstMember,
                71,
                MakeRange(70, 79),
                ::SPIRVSimulator::InvocationDependence::Invariant);
    const uint32_t private_result_id = LoadScalar(StoredTarget::FirstMember);
    EXPECT_EQ(value_meta_[private_result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
}

TEST_F(StoreLoadMetadataTests, LoadRestoresStoredInvariantFromRepresentativeReachingStore)
{
    StoreScalar(StoredTarget::FirstMember,
                81,
                MakeRange(80, 89),
                ::SPIRVSimulator::InvocationDependence::Invariant);

    // E.2 external-load propagation must not override the authoritative
    // reaching-store metadata used by internal memory.
    value_meta_[kFirstMemberPointerId].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;

    const uint32_t result_id = LoadScalar(StoredTarget::FirstMember);

    ExpectScalarResult(result_id, 81, MakeRange(80, 89));
    EXPECT_EQ(value_meta_[result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
}

TEST_F(StoreLoadMetadataTests, AggregateLoadAndProjectionRestoreInvocationDependence)
{
    StorePair(91,
              102,
              MakeRange(90, 99),
              MakeRange(100, 109),
              false,
              ::SPIRVSimulator::InvocationDependence::Invariant,
              ::SPIRVSimulator::InvocationDependence::Varying);

    const uint32_t aggregate_result_id = LoadPair();
    const ::SPIRVSimulator::ValueMetadata& aggregate_meta = value_meta_[aggregate_result_id];
    EXPECT_EQ(aggregate_meta.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_EQ(aggregate_meta.component_invocation_dependence,
              (std::vector<::SPIRVSimulator::InvocationDependence>{
                  ::SPIRVSimulator::InvocationDependence::Invariant,
                  ::SPIRVSimulator::InvocationDependence::Varying }));
    ASSERT_EQ(aggregate_meta.subobject_invocation_dependence.size(), 2u);
    EXPECT_EQ(aggregate_meta.subobject_invocation_dependence[0].dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(aggregate_meta.subobject_invocation_dependence[1].dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);

    const uint32_t invariant_member_result_id = LoadScalar(StoredTarget::FirstMember);
    const uint32_t varying_member_result_id = LoadScalar(StoredTarget::SecondMember);
    EXPECT_EQ(value_meta_[invariant_member_result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(value_meta_[varying_member_result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
}

TEST_F(StoreLoadMetadataTests, WorkgroupAndTaskPayloadLoadsRestoreInvocationDependence)
{
    SetPointerStorageClass(spv::StorageClass::StorageClassWorkgroup);
    StoreScalar(StoredTarget::FirstMember,
                111,
                MakeRange(110, 119),
                ::SPIRVSimulator::InvocationDependence::Varying);

    const uint32_t workgroup_result_id = LoadScalar(StoredTarget::FirstMember);

    ExpectScalarResult(workgroup_result_id, 111, MakeRange(110, 119));
    EXPECT_EQ(value_meta_[workgroup_result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);

    SetPointerStorageClass(spv::StorageClass::StorageClassTaskPayloadWorkgroupEXT);
    StoreScalar(StoredTarget::FirstMember,
                121,
                MakeRange(120, 129),
                ::SPIRVSimulator::InvocationDependence::Invariant);
    const uint32_t task_payload_result_id = LoadScalar(StoredTarget::FirstMember);
    EXPECT_EQ(value_meta_[task_payload_result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
}

// Exercises dynamic re-execution of the same aggregate instruction and result ID.
// Each test changes only source metadata between executions and verifies
// that SetValue(clear_meta=true) prevents the first execution's dependence
// state from leaking into the second.
class AggregateInvocationDependenceLifecycleTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kResultId  = 9000;
    static constexpr uint32_t kSource0Id = 9001;
    static constexpr uint32_t kSource1Id = 9002;
    static constexpr uint32_t kSource2Id = 9003;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;
        values_.resize(kSource2Id + 1);
        value_meta_.resize(kSource2Id + 1);

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_.at(id);
        });
        EXPECT_CALL(*this, GetTypeByResultId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(value_type_ids_.at(id));
        });
        EXPECT_CALL(*this, GetTypeByTypeId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(id);
        });
        EXPECT_CALL(*this, SetValue(kResultId, _, true))
            .Times(2)
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, ExtractFlags(_, _)).Times(AnyNumber());
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>())).Times(AnyNumber());
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint64_t>())).Times(AnyNumber());
    }

    uint32_t GetTypeID(uint32_t result_id) const override { return value_type_ids_.at(result_id); }

    void SetTypedValue(uint32_t id, uint32_t type_id, ::SPIRVSimulator::Value value)
    {
        values_[id]         = std::move(value);
        value_type_ids_[id] = type_id;
    }

    void Execute(const ::SPIRVSimulator::Instruction& instruction) { ASSERT_TRUE(ExecuteInstruction(instruction)); }

    ::SPIRVSimulator::UnorderedMap<uint32_t, uint32_t> value_type_ids_;
};

// Re-executes a struct construct after removing one source vector component
// proof, verifying that the old leaf and Varying summary are not retained.
TEST_F(AggregateInvocationDependenceLifecycleTests, RepeatedConstructDoesNotRetainStaleMetadata)
{
    constexpr uint32_t kStructTypeId = 9010;
    types_[kStructTypeId]            = ::SPIRVSimulator::Type::Struct(kStructTypeId);
    struct_members_[kStructTypeId]   = { CommonTypes::u64, CommonTypes::uvec2 };

    SetTypedValue(kResultId, kStructTypeId, std::monostate{});
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 1 });
    SetTypedValue(kSource1Id,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 3 }));

    const std::vector<uint32_t> words{
        static_cast<uint32_t>(spv::Op::OpCompositeConstruct), kStructTypeId, kResultId, kSource0Id, kSource1Id,
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = spv::Op::OpCompositeConstruct,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Invariant;
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Varying;
    value_meta_[kSource1Id].component_invocation_dependence = {
        Dependence::Varying,
        Dependence::Invariant,
    };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {
                                               { { 0 }, Dependence::Invariant },
                                               { { 1, 0 }, Dependence::Varying },
                                               { { 1, 1 }, Dependence::Invariant },
                                           },
                                       });

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Unknown;
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Unknown;
    value_meta_[kSource1Id].component_invocation_dependence = { Dependence::Invariant };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Unknown,
                                           { Dependence::Unknown, Dependence::Unknown },
                                           {
                                               { { 0 }, Dependence::Unknown },
                                               { { 1, 0 }, Dependence::Invariant },
                                           },
                                       });
}

// Re-executes an aggregate-to-vector extract with incomplete descendant proof,
// verifying that old component values and injected stale subobjects are reset.
TEST_F(AggregateInvocationDependenceLifecycleTests, RepeatedExtractDoesNotRetainStaleMetadata)
{
    constexpr uint32_t kStructTypeId = 9020;
    types_[kStructTypeId]            = ::SPIRVSimulator::Type::Struct(kStructTypeId);
    struct_members_[kStructTypeId]   = { CommonTypes::uvec2, CommonTypes::u64 };

    auto selected_vector = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2 });
    SetTypedValue(kResultId, CommonTypes::uvec2, std::monostate{});
    SetTypedValue(kSource0Id,
                  kStructTypeId,
                  std::make_shared<::SPIRVSimulator::AggregateV>(
                      std::initializer_list<::SPIRVSimulator::Value>{ selected_vector, uint64_t{ 3 } }));

    const std::vector<uint32_t> words{
        static_cast<uint32_t>(spv::Op::OpCompositeExtract), CommonTypes::uvec2, kResultId, kSource0Id, 0,
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = spv::Op::OpCompositeExtract,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Varying;
    value_meta_[kSource0Id].component_invocation_dependence = {
        Dependence::Varying,
        Dependence::Invariant,
    };
    value_meta_[kSource0Id].subobject_invocation_dependence = {
        { { 0, 0 }, Dependence::Varying },
        { { 0, 1 }, Dependence::Invariant },
        { { 1 }, Dependence::Invariant },
    };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Varying, Dependence::Invariant },
                                           {},
                                       });

    value_meta_[kResultId].subobject_invocation_dependence = {
        { { 9 }, Dependence::Varying },
    };
    value_meta_[kSource0Id].invocation_dependence           = Dependence::Unknown;
    value_meta_[kSource0Id].component_invocation_dependence = {
        Dependence::Unknown,
        Dependence::Invariant,
    };
    value_meta_[kSource0Id].subobject_invocation_dependence = {
        { { 0, 0 }, Dependence::Invariant },
        { { 1 }, Dependence::Invariant },
    };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Unknown,
                                           { Dependence::Invariant, Dependence::Unknown },
                                           {},
                                       });
}

// Re-executes a nested matrix insert after dropping an unchanged sibling leaf,
// verifying that the old complete subtree cannot make the new result precise.
TEST_F(AggregateInvocationDependenceLifecycleTests, RepeatedInsertDoesNotRetainStaleMetadata)
{
    SetTypedValue(kResultId, CommonTypes::mat2, std::monostate{});
    SetTypedValue(kSource0Id, CommonTypes::f64, 9.0);
    SetTypedValue(kSource1Id,
                  CommonTypes::mat2,
                  std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2));

    const std::vector<uint32_t> words{
        static_cast<uint32_t>(spv::Op::OpCompositeInsert), CommonTypes::mat2, kResultId, kSource0Id, kSource1Id, 1, 0,
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = spv::Op::OpCompositeInsert,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Varying;
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Invariant;
    value_meta_[kSource1Id].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
    };
    value_meta_[kSource1Id].subobject_invocation_dependence = {
        { { 0, 0 }, Dependence::Invariant },
        { { 0, 1 }, Dependence::Invariant },
        { { 1, 0 }, Dependence::Invariant },
        { { 1, 1 }, Dependence::Invariant },
    };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {
                                               { { 0, 0 }, Dependence::Invariant },
                                               { { 0, 1 }, Dependence::Invariant },
                                               { { 1, 1 }, Dependence::Invariant },
                                               { { 1, 0 }, Dependence::Varying },
                                           },
                                       });

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Invariant;
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Unknown;
    value_meta_[kSource1Id].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Unknown,
    };
    value_meta_[kSource1Id].subobject_invocation_dependence = {
        { { 0, 0 }, Dependence::Invariant },
        { { 0, 1 }, Dependence::Invariant },
        { { 1, 0 }, Dependence::Varying },
    };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Unknown,
                                           { Dependence::Invariant, Dependence::Unknown },
                                           {
                                               { { 0, 0 }, Dependence::Invariant },
                                               { { 0, 1 }, Dependence::Invariant },
                                               { { 1, 0 }, Dependence::Invariant },
                                           },
                                       });
}

// Re-executes the same shuffle with weaker source proof, verifying that its
// rebuilt direct components and empty vector subobject shape contain no stale state.
TEST_F(AggregateInvocationDependenceLifecycleTests, RepeatedShuffleDoesNotRetainStaleMetadata)
{
    SetTypedValue(kResultId, CommonTypes::uvec2, std::monostate{});
    SetTypedValue(kSource0Id,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2 }));
    SetTypedValue(kSource1Id,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 4 }));

    const std::vector<uint32_t> words{
        static_cast<uint32_t>(spv::Op::OpVectorShuffle), CommonTypes::uvec2, kResultId, kSource0Id, kSource1Id, 0, 3,
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode     = spv::Op::OpVectorShuffle,
        .word_count = static_cast<uint16_t>(words.size()),
        .words      = words,
    };

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Varying;
    value_meta_[kSource0Id].component_invocation_dependence = {
        Dependence::Varying,
        Dependence::Invariant,
    };
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Unknown;
    value_meta_[kSource1Id].component_invocation_dependence = {
        Dependence::Unknown,
        Dependence::Invariant,
    };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Varying, Dependence::Invariant },
                                           {},
                                       });

    value_meta_[kResultId].subobject_invocation_dependence = {
        { { 9 }, Dependence::Varying },
    };
    value_meta_[kSource0Id].invocation_dependence = Dependence::Invariant;
    value_meta_[kSource0Id].component_invocation_dependence.clear();
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Varying;
    value_meta_[kSource1Id].component_invocation_dependence = { Dependence::Varying };
    Execute(instruction);

    ExpectInvocationDependenceMetadata(value_meta_[kResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Unknown,
                                           { Dependence::Invariant, Dependence::Unknown },
                                           {},
                                       });
}

// Exercises invocation-dependence propagation across consecutive aggregate
// opcode handlers. Intermediate metadata must be produced by the preceding
// instruction rather than manually supplied to each consumer.
class AggregateInvocationDependenceOpcodeChainTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kSource0Id            = 9200;
    static constexpr uint32_t kSource1Id            = 9201;
    static constexpr uint32_t kSource2Id            = 9202;
    static constexpr uint32_t kConstructResultId    = 9203;
    static constexpr uint32_t kIntermediateResultId = 9204;
    static constexpr uint32_t kFinalResultId        = 9205;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;
        values_.resize(kFinalResultId + 1);
        value_meta_.resize(kFinalResultId + 1);

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_.at(id);
        });
        EXPECT_CALL(*this, GetTypeByResultId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(value_type_ids_.at(id));
        });
        EXPECT_CALL(*this, GetTypeByTypeId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(id);
        });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, ExtractFlags(_, _)).Times(AnyNumber());
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>())).Times(AnyNumber());
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint64_t>())).Times(AnyNumber());
    }

    uint32_t GetTypeID(uint32_t result_id) const override { return value_type_ids_.at(result_id); }

    void SetTypedValue(uint32_t id, uint32_t type_id, ::SPIRVSimulator::Value value)
    {
        values_[id]         = std::move(value);
        value_type_ids_[id] = type_id;
    }

    void Execute(spv::Op opcode, std::initializer_list<uint32_t> operands)
    {
        std::vector<uint32_t> words{ static_cast<uint32_t>(opcode) };
        words.insert(words.end(), operands.begin(), operands.end());
        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = opcode,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        ASSERT_TRUE(ExecuteInstruction(instruction));
    }

    ::SPIRVSimulator::UnorderedMap<uint32_t, uint32_t> value_type_ids_;
};

// Constructs a mixed-dependence vector and extracts its invariant component,
// verifying that Extract uses the producer's exact direct-component proof.
TEST_F(AggregateInvocationDependenceOpcodeChainTests, ConstructThenExtractDirectComponentPreservesDependence)
{
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 11 });
    SetTypedValue(kSource1Id, CommonTypes::u64, uint64_t{ 22 });
    SetTypedValue(kConstructResultId, CommonTypes::uvec2, std::monostate{});
    SetTypedValue(kFinalResultId, CommonTypes::u64, std::monostate{});

    value_meta_[kSource0Id].invocation_dependence = Dependence::Invariant;
    value_meta_[kSource1Id].invocation_dependence = Dependence::Varying;

    Execute(spv::Op::OpCompositeConstruct, { CommonTypes::uvec2, kConstructResultId, kSource0Id, kSource1Id });

    ExpectInvocationDependenceMetadata(value_meta_[kConstructResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {},
                                       });

    Execute(spv::Op::OpCompositeExtract, { CommonTypes::u64, kFinalResultId, kConstructResultId, 0 });

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kFinalResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kFinalResultId]), 11u);
    ExpectInvocationDependenceMetadata(value_meta_[kFinalResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Invariant,
                                           {},
                                           {},
                                       });
}

// Constructs a matrix and extracts a mixed-dependence column, verifying that
// the producer's {column,row} leaves are rebased to vector components.
TEST_F(AggregateInvocationDependenceOpcodeChainTests, ConstructThenExtractNestedAggregateRebasesLeafPaths)
{
    auto first_column  = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 });
    auto second_column = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 });
    SetTypedValue(kSource0Id, CommonTypes::vec2, first_column);
    SetTypedValue(kSource1Id, CommonTypes::vec2, second_column);
    SetTypedValue(kConstructResultId, CommonTypes::mat2, std::monostate{});
    SetTypedValue(kFinalResultId, CommonTypes::vec2, std::monostate{});

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Invariant;
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Varying;
    value_meta_[kSource1Id].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
    };

    Execute(spv::Op::OpCompositeConstruct, { CommonTypes::mat2, kConstructResultId, kSource0Id, kSource1Id });

    ExpectInvocationDependenceMetadata(value_meta_[kConstructResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {
                                               { { 0, 0 }, Dependence::Invariant },
                                               { { 0, 1 }, Dependence::Invariant },
                                               { { 1, 0 }, Dependence::Invariant },
                                               { { 1, 1 }, Dependence::Varying },
                                           },
                                       });

    Execute(spv::Op::OpCompositeExtract, { CommonTypes::vec2, kFinalResultId, kConstructResultId, 1 });

    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kFinalResultId]);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->elems.size(), 2u);
    EXPECT_DOUBLE_EQ(std::get<double>(result->elems[0]), 3.0);
    EXPECT_DOUBLE_EQ(std::get<double>(result->elems[1]), 4.0);
    ExpectInvocationDependenceMetadata(value_meta_[kFinalResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {},
                                       });
}

// Constructs a matrix, replaces one nested scalar, and extracts that column,
// verifying prefix erasure and both affected summaries across the full chain.
TEST_F(AggregateInvocationDependenceOpcodeChainTests, ConstructThenInsertThenExtractReflectsReplacement)
{
    auto first_column  = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 });
    auto second_column = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 });
    SetTypedValue(kSource0Id, CommonTypes::vec2, first_column);
    SetTypedValue(kSource1Id, CommonTypes::vec2, second_column);
    SetTypedValue(kSource2Id, CommonTypes::f64, 9.0);
    SetTypedValue(kConstructResultId, CommonTypes::mat2, std::monostate{});
    SetTypedValue(kIntermediateResultId, CommonTypes::mat2, std::monostate{});
    SetTypedValue(kFinalResultId, CommonTypes::vec2, std::monostate{});

    value_meta_[kSource0Id].invocation_dependence           = Dependence::Invariant;
    value_meta_[kSource1Id].invocation_dependence           = Dependence::Varying;
    value_meta_[kSource1Id].component_invocation_dependence = {
        Dependence::Varying,
        Dependence::Invariant,
    };
    value_meta_[kSource2Id].invocation_dependence = Dependence::Invariant;

    Execute(spv::Op::OpCompositeConstruct, { CommonTypes::mat2, kConstructResultId, kSource0Id, kSource1Id });

    ExpectInvocationDependenceMetadata(value_meta_[kConstructResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {
                                               { { 0, 0 }, Dependence::Invariant },
                                               { { 0, 1 }, Dependence::Invariant },
                                               { { 1, 0 }, Dependence::Varying },
                                               { { 1, 1 }, Dependence::Invariant },
                                           },
                                       });

    Execute(spv::Op::OpCompositeInsert,
            { CommonTypes::mat2, kIntermediateResultId, kSource2Id, kConstructResultId, 1, 0 });

    ExpectInvocationDependenceMetadata(value_meta_[kIntermediateResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Invariant,
                                           { Dependence::Invariant, Dependence::Invariant },
                                           {
                                               { { 0, 0 }, Dependence::Invariant },
                                               { { 0, 1 }, Dependence::Invariant },
                                               { { 1, 1 }, Dependence::Invariant },
                                               { { 1, 0 }, Dependence::Invariant },
                                           },
                                       });

    Execute(spv::Op::OpCompositeExtract, { CommonTypes::vec2, kFinalResultId, kIntermediateResultId, 1 });

    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kFinalResultId]);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->elems.size(), 2u);
    EXPECT_DOUBLE_EQ(std::get<double>(result->elems[0]), 9.0);
    EXPECT_DOUBLE_EQ(std::get<double>(result->elems[1]), 4.0);
    ExpectInvocationDependenceMetadata(value_meta_[kFinalResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Invariant,
                                           { Dependence::Invariant, Dependence::Invariant },
                                           {},
                                       });
}

// Constructs a vector, shuffles and repeats its components, then extracts one
// component, verifying selector mapping and final projection across handlers.
TEST_F(AggregateInvocationDependenceOpcodeChainTests, ConstructThenShuffleThenExtractUsesSelectedComponent)
{
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 7 });
    SetTypedValue(kSource1Id, CommonTypes::u64, uint64_t{ 8 });
    SetTypedValue(kConstructResultId, CommonTypes::uvec2, std::monostate{});
    SetTypedValue(kIntermediateResultId, CommonTypes::uvec3, std::monostate{});
    SetTypedValue(kFinalResultId, CommonTypes::u64, std::monostate{});

    value_meta_[kSource0Id].invocation_dependence = Dependence::Invariant;
    value_meta_[kSource1Id].invocation_dependence = Dependence::Varying;

    Execute(spv::Op::OpCompositeConstruct, { CommonTypes::uvec2, kConstructResultId, kSource0Id, kSource1Id });

    ExpectInvocationDependenceMetadata(value_meta_[kConstructResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {},
                                       });

    Execute(spv::Op::OpVectorShuffle,
            { CommonTypes::uvec3, kIntermediateResultId, kConstructResultId, kConstructResultId, 1, 0, 1 });

    ExpectInvocationDependenceMetadata(value_meta_[kIntermediateResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Varying, Dependence::Invariant, Dependence::Varying },
                                           {},
                                       });

    Execute(spv::Op::OpCompositeExtract, { CommonTypes::u64, kFinalResultId, kIntermediateResultId, 1 });

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kFinalResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kFinalResultId]), 7u);
    ExpectInvocationDependenceMetadata(value_meta_[kFinalResultId],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Invariant,
                                           {},
                                           {},
                                       });
}

// Stores and reloads a value with both legacy metadata and invocation
// dependence, verifying that StoredValueMetadataSnapshot restores both.
TEST_F(StoreLoadMetadataTests, StoreLoadRestoresRangeAndInvocationDependence)
{
    const auto range =
        MakeProjectionRange(true,
                            true,
                            70,
                            79,
                            1,
                            ::SPIRVSimulator::kInvocationComponentY,
                            ::SPIRVSimulator::kInvocationComponentX | ::SPIRVSimulator::kInvocationComponentZ);
    const uint32_t value_id = next_store_value_id_++;
    values_[value_id]       = uint64_t{ 77 };

    ::SPIRVSimulator::ValueMetadata& source_meta = value_meta_[value_id];
    source_meta                                  = {};
    source_meta.flags                            = SPS_FLAG_THREAD_SPECIFIC;
    source_meta.value_range                      = range;
    source_meta.address_range_valid              = true;
    source_meta.address_range_min                = 0x1000;
    source_meta.address_range_max                = 0x103f;
    source_meta.address_range_stride             = 8;
    source_meta.address_element_size             = 8;
    source_meta.invocation_dependence            = ::SPIRVSimulator::InvocationDependence::Varying;

    ExecuteStore(StoredTarget::FirstMember, value_id);
    const uint32_t result_id = LoadScalar(StoredTarget::FirstMember);

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[result_id]));
    EXPECT_EQ(std::get<uint64_t>(values_[result_id]), 77u);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[result_id];
    EXPECT_EQ(result_meta.flags, source_meta.flags);
    ExpectProjectionRange(result_meta.value_range, range);
    EXPECT_TRUE(result_meta.component_ranges.empty());
    EXPECT_TRUE(result_meta.subobject_scalar_ranges.empty());
    EXPECT_TRUE(result_meta.address_range_valid);
    EXPECT_EQ(result_meta.address_range_min, source_meta.address_range_min);
    EXPECT_EQ(result_meta.address_range_max, source_meta.address_range_max);
    EXPECT_EQ(result_meta.address_range_stride, source_meta.address_range_stride);
    EXPECT_EQ(result_meta.address_element_size, source_meta.address_element_size);
    ExpectInvocationDependenceMetadata(
        result_meta,
        ExpectedInvocationDependenceMetadataSpec{
            ::SPIRVSimulator::InvocationDependence::Varying,
            {},
            {},
        });
    ExpectOperationCounts(1, 1);
}

TEST_F(StoreLoadMetadataTests, LoadWithoutReachingStoreKeepsDependenceUnknown)
{
    value_meta_[kLoadResultBaseId].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Invariant;

    const uint32_t result_id = LoadScalar(StoredTarget::FirstMember);

    EXPECT_EQ(value_meta_[result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Unknown);
    EXPECT_TRUE(value_meta_[result_id].component_invocation_dependence.empty());
    EXPECT_TRUE(value_meta_[result_id].subobject_invocation_dependence.empty());
}

// Exercises the current boundary between supported aggregate dependence
// propagation and the deferred memory/control-flow propagation rules. The
// fixture forwards existing metadata operations so their behavior remains
// observable alongside the new dependence assertions.
class InvocationDependenceStageBoundaryTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kSource0Id         = 9400;
    static constexpr uint32_t kSource1Id         = 9401;
    static constexpr uint32_t kConditionId       = 9402;
    static constexpr uint32_t kAggregateResultId = 9403;
    static constexpr uint32_t kResult0Id         = 9404;
    static constexpr uint32_t kResult1Id         = 9405;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;
        values_.resize(kResult1Id + 1);
        value_meta_.resize(kResult1Id + 1);

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_.at(id);
        });
        EXPECT_CALL(*this, GetTypeByResultId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(value_type_ids_.at(id));
        });
        EXPECT_CALL(*this, GetTypeByTypeId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(id);
        });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, ExtractFlags(_, _)).WillRepeatedly([this](uint32_t id, uint64_t& flags) {
            ::SPIRVSimulator::SPIRVSimulator::ExtractFlags(id, flags);
        });
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>()))
            .WillRepeatedly([this](uint32_t target, uint32_t source) {
                ::SPIRVSimulator::SPIRVSimulator::TransferFlags(target, source);
            });
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint64_t>()))
            .WillRepeatedly([this](uint32_t target, uint64_t flags) {
                ::SPIRVSimulator::SPIRVSimulator::TransferFlags(target, flags);
            });
    }

    uint32_t GetTypeID(uint32_t result_id) const override { return value_type_ids_.at(result_id); }

    void SetTypedValue(uint32_t id, uint32_t type_id, ::SPIRVSimulator::Value value)
    {
        values_[id]         = std::move(value);
        value_type_ids_[id] = type_id;
    }

    void Execute(spv::Op opcode, std::initializer_list<uint32_t> operands)
    {
        std::vector<uint32_t> words{ static_cast<uint32_t>(opcode) };
        words.insert(words.end(), operands.begin(), operands.end());
        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = opcode,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        ASSERT_TRUE(ExecuteInstruction(instruction));
    }

    static void SetAddressMetadata(::SPIRVSimulator::ValueMetadata& metadata, uint64_t base)
    {
        metadata.address_range_valid  = true;
        metadata.address_range_min    = base;
        metadata.address_range_max    = base + 0x3f;
        metadata.address_range_stride = 8;
        metadata.address_element_size = 8;
    }

    static void ExpectAddressMetadata(const ::SPIRVSimulator::ValueMetadata& actual,
                                      const ::SPIRVSimulator::ValueMetadata& expected)
    {
        EXPECT_EQ(actual.address_range_valid, expected.address_range_valid);
        EXPECT_EQ(actual.address_range_min, expected.address_range_min);
        EXPECT_EQ(actual.address_range_max, expected.address_range_max);
        EXPECT_EQ(actual.address_range_stride, expected.address_range_stride);
        EXPECT_EQ(actual.address_element_size, expected.address_element_size);
    }

    ::SPIRVSimulator::UnorderedMap<uint32_t, uint32_t> value_type_ids_;
};

// Constructs a mixed vector and extracts both components, verifying that its
// Varying whole-value summary does not obscure exact component dependence.
TEST_F(InvocationDependenceStageBoundaryTests, MixedAggregateExtractKeepsExactComponentDependence)
{
    const auto invariant_range = MakeProjectionRange(false, true, 11, 11, 1);
    const auto varying_range   = MakeProjectionRange(
        true, true, 20, 29, 1, ::SPIRVSimulator::kInvocationComponentX, ::SPIRVSimulator::kInvocationComponentY);

    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 11 });
    SetTypedValue(kSource1Id, CommonTypes::u64, uint64_t{ 22 });
    SetTypedValue(kAggregateResultId, CommonTypes::uvec2, std::monostate{});
    SetTypedValue(kResult0Id, CommonTypes::u64, std::monostate{});
    SetTypedValue(kResult1Id, CommonTypes::u64, std::monostate{});

    value_meta_[kSource0Id].value_range           = invariant_range;
    value_meta_[kSource0Id].invocation_dependence = Dependence::Invariant;
    value_meta_[kSource1Id].flags                 = SPS_FLAG_THREAD_SPECIFIC;
    value_meta_[kSource1Id].value_range           = varying_range;
    value_meta_[kSource1Id].invocation_dependence = Dependence::Varying;

    Execute(spv::Op::OpCompositeConstruct, { CommonTypes::uvec2, kAggregateResultId, kSource0Id, kSource1Id });

    const ::SPIRVSimulator::ValueMetadata& aggregate_meta = value_meta_[kAggregateResultId];
    EXPECT_EQ(aggregate_meta.flags, SPS_FLAG_THREAD_SPECIFIC);
    ASSERT_EQ(aggregate_meta.component_ranges.size(), 2u);
    ExpectProjectionRange(aggregate_meta.component_ranges[0], invariant_range);
    ExpectProjectionRange(aggregate_meta.component_ranges[1], varying_range);
    ExpectInvocationDependenceMetadata(aggregate_meta,
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           { Dependence::Invariant, Dependence::Varying },
                                           {},
                                       });

    Execute(spv::Op::OpCompositeExtract, { CommonTypes::u64, kResult0Id, kAggregateResultId, 0 });
    Execute(spv::Op::OpCompositeExtract, { CommonTypes::u64, kResult1Id, kAggregateResultId, 1 });

    EXPECT_EQ(value_meta_[kResult0Id].flags, aggregate_meta.flags);
    EXPECT_EQ(value_meta_[kResult1Id].flags, aggregate_meta.flags);
    ExpectProjectionRange(value_meta_[kResult0Id].value_range, invariant_range);
    ExpectProjectionRange(value_meta_[kResult1Id].value_range, varying_range);
    ExpectInvocationDependenceMetadata(value_meta_[kResult0Id],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Invariant,
                                           {},
                                           {},
                                       });
    ExpectInvocationDependenceMetadata(value_meta_[kResult1Id],
                                       ExpectedInvocationDependenceMetadataSpec{
                                           Dependence::Varying,
                                           {},
                                           {},
                                       });
}

// Selects a Phi incoming vector with range, address, flags, and dependence
// metadata, verifying that only the currently supported metadata is copied.
TEST_F(InvocationDependenceStageBoundaryTests, PhiDoesNotPropagateInvocationDependence)
{
    constexpr uint32_t kSelectedBlock = 9500;
    constexpr uint32_t kOtherBlock    = 9501;
    const auto         first_range    = MakeProjectionRange(
        true, true, 30, 39, 1, ::SPIRVSimulator::kInvocationComponentX, ::SPIRVSimulator::kInvocationComponentZ);
    const auto second_range = MakeProjectionRange(false, true, 40, 40, 1);

    SetTypedValue(kSource0Id,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 31, 40 }));
    SetTypedValue(kSource1Id,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 51, 60 }));
    SetTypedValue(kResult0Id, CommonTypes::uvec2, std::monostate{});

    ::SPIRVSimulator::ValueMetadata& source_meta = value_meta_[kSource0Id];
    source_meta.flags                            = SPS_FLAG_THREAD_SPECIFIC;
    source_meta.component_ranges                 = { first_range, second_range };
    source_meta.invocation_dependence            = Dependence::Varying;
    source_meta.component_invocation_dependence  = { Dependence::Varying, Dependence::Invariant };
    SetAddressMetadata(source_meta, 0x2000);
    prev_block_id_ = kSelectedBlock;

    Execute(spv::Op::OpPhi, { CommonTypes::uvec2, kResult0Id, kSource0Id, kSelectedBlock, kSource1Id, kOtherBlock });

    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kResult0Id]);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->elems.size(), 2u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[0]), 31u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[1]), 40u);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResult0Id];
    EXPECT_EQ(result_meta.flags, source_meta.flags);
    ASSERT_EQ(result_meta.component_ranges.size(), source_meta.component_ranges.size());
    ExpectProjectionRange(result_meta.component_ranges[0], first_range);
    ExpectProjectionRange(result_meta.component_ranges[1], second_range);
    ExpectAddressMetadata(result_meta, source_meta);
    ExpectInvocationDependenceMetadata(result_meta, ExpectedInvocationDependenceMetadataSpec{});
}

// Selects a scalar operand with range, address, flags, and dependence metadata,
// verifying that the selected non-flag metadata is preserved without treating
// the representative branch choice as an invocation-dependence proof.
TEST_F(InvocationDependenceStageBoundaryTests, SelectDoesNotPropagateInvocationDependence)
{
    const auto selected_range = MakeProjectionRange(
        true, true, 50, 59, 1, ::SPIRVSimulator::kInvocationComponentY, ::SPIRVSimulator::kInvocationComponentZ);
    const auto other_range = MakeProjectionRange(false, true, 70, 70, 1);

    SetTypedValue(kConditionId, CommonTypes::boolean, uint64_t{ 1 });
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 55 });
    SetTypedValue(kSource1Id, CommonTypes::u64, uint64_t{ 70 });
    SetTypedValue(kResult0Id, CommonTypes::u64, std::monostate{});

    value_meta_[kConditionId].invocation_dependence = Dependence::Varying;
    ::SPIRVSimulator::ValueMetadata& selected_meta  = value_meta_[kSource0Id];
    selected_meta.flags                             = SPS_FLAG_THREAD_SPECIFIC;
    selected_meta.value_range                       = selected_range;
    selected_meta.invocation_dependence             = Dependence::Varying;
    SetAddressMetadata(selected_meta, 0x3000);
    value_meta_[kSource1Id].value_range           = other_range;
    value_meta_[kSource1Id].invocation_dependence = Dependence::Invariant;
    SetAddressMetadata(value_meta_[kSource1Id], 0x4000);

    Execute(spv::Op::OpSelect, { CommonTypes::u64, kResult0Id, kConditionId, kSource0Id, kSource1Id });

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kResult0Id]));
    EXPECT_EQ(std::get<uint64_t>(values_[kResult0Id]), 55u);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResult0Id];
    EXPECT_EQ(result_meta.flags, selected_meta.flags);
    ExpectProjectionRange(result_meta.value_range, selected_range);
    ExpectAddressMetadata(result_meta, selected_meta);
    ExpectInvocationDependenceMetadata(result_meta, ExpectedInvocationDependenceMetadataSpec{});
}
