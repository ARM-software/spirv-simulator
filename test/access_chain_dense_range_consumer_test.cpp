#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "spirv-tools/libspirv.hpp"
#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

// Verify failure handling while dense AccessChain ranges are consumed to collect
// nested PhysicalStorageBuffer pointers from read-side pointer tables.
class AccessChainReadSidePointerConsumerTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (override));

  protected:
    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        simulation_results_  = &simulation_results_storage_;
        verbose_             = false;
    }

    void ExecuteAccessChain(const ::SPIRVSimulator::Instruction& instruction)
    {
        Op_AccessChain(instruction);
    }

    size_t QueuedPointerPairCount() const
    {
        return pointers_to_physical_address_pointers_.size();
    }

    bool ReadSidePointerCoverageIncomplete() const
    {
        return simulation_results_storage_.read_side_pointer_coverage_incomplete;
    }

    static ::SPIRVSimulator::PointerV MakePointer(uint32_t index)
    {
        return ::SPIRVSimulator::PointerV{
            1,
            0,
            1,
            1,
            spv::StorageClass::StorageClassFunction,
            { index },
        };
    }

    ::SPIRVSimulator::SimulationResults simulation_results_storage_{};
};

TEST_F(AccessChainReadSidePointerConsumerTests, DenseQueueReadFailureDoesNotCommitPartialPairs)
{
    constexpr uint32_t result_id = 1000;

    value_meta_.resize(result_id + 1);
    value_meta_[result_id] = {
        .address_range_valid = true,
        .address_range_min = 100,
        .address_range_max = 116,
        .address_range_stride = 8,
        .address_element_size = 8,
    };
    const ::SPIRVSimulator::PointerV representative_pointer = MakePointer(0);
    const ::SPIRVSimulator::PointerV loaded_pointer         = MakePointer(7);
    const ::SPIRVSimulator::DenseValueRangeInfo dynamic_index_range{
        true, true, true, 0, 2, 1
    };
    const DenseAccessChainIndexSelection dynamic_index_selection{
        0,
        dynamic_index_range,
    };

    InSequence sequence;
    EXPECT_CALL(*this, ReadPointer(_))
        .WillOnce([loaded_pointer](const ::SPIRVSimulator::PointerV& pointer)
                      -> std::optional<::SPIRVSimulator::Value> {
            EXPECT_EQ(pointer.idx_path, std::vector<uint32_t>{ 0 });
            return ::SPIRVSimulator::Value{ loaded_pointer };
        });
    EXPECT_CALL(*this, ReadPointer(_))
        .WillOnce([](const ::SPIRVSimulator::PointerV& pointer)
                      -> std::optional<::SPIRVSimulator::Value> {
            EXPECT_EQ(pointer.idx_path, std::vector<uint32_t>{ 1 });
            return std::nullopt;
        });

    EXPECT_FALSE(TryQueueDenseReadSidePbufferPointers(
        result_id, representative_pointer, dynamic_index_selection));
    EXPECT_EQ(QueuedPointerPairCount(), 0u);
}

TEST_F(AccessChainReadSidePointerConsumerTests,
       PaddedPointerTableQueuesEveryLogicalSlot)
{
    constexpr uint32_t result_id = 1010;

    value_meta_.resize(result_id + 1);
    value_meta_[result_id] = {
        .address_range_valid = true,
        .address_range_min = 100,
        .address_range_max = 132,
        .address_range_stride = 16,
        .address_element_size = 8,
    };
    const ::SPIRVSimulator::PointerV representative_pointer = MakePointer(0);
    const ::SPIRVSimulator::PointerV loaded_pointer         = MakePointer(7);
    const ::SPIRVSimulator::DenseValueRangeInfo dynamic_index_range{
        true, true, true, 0, 2, 1
    };
    const DenseAccessChainIndexSelection dynamic_index_selection{
        0,
        dynamic_index_range,
    };

    InSequence sequence;
    for (uint32_t expected_index = 0; expected_index < 3; ++expected_index)
    {
        EXPECT_CALL(*this, ReadPointer(_))
            .WillOnce([expected_index, loaded_pointer](const ::SPIRVSimulator::PointerV& pointer)
                          -> std::optional<::SPIRVSimulator::Value> {
                EXPECT_EQ(pointer.idx_path, std::vector<uint32_t>{ expected_index });
                return ::SPIRVSimulator::Value{ loaded_pointer };
            });
    }

    EXPECT_TRUE(TryQueueDenseReadSidePbufferPointers(
        result_id, representative_pointer, dynamic_index_selection));
    EXPECT_EQ(QueuedPointerPairCount(), 3u);
}

TEST_F(AccessChainReadSidePointerConsumerTests,
       DynamicIndexSelectionPreservesFixedPrefixAndSuffix)
{
    constexpr uint32_t result_id = 1020;

    value_meta_.resize(result_id + 1);
    value_meta_[result_id] = {
        .address_range_valid = true,
        .address_range_min = 100,
        .address_range_max = 132,
        .address_range_stride = 16,
        .address_element_size = 8,
    };
    const ::SPIRVSimulator::PointerV representative_pointer{
        1,
        0,
        1,
        1,
        spv::StorageClass::StorageClassFunction,
        { 4, 0, 1 },
    };
    const ::SPIRVSimulator::PointerV loaded_pointer = MakePointer(7);
    const ::SPIRVSimulator::DenseValueRangeInfo dynamic_index_range{
        true, true, true, 1, 3, 1
    };
    const DenseAccessChainIndexSelection dynamic_index_selection{
        1,
        dynamic_index_range,
    };

    InSequence sequence;
    for (uint32_t expected_index = 1; expected_index <= 3; ++expected_index)
    {
        EXPECT_CALL(*this, ReadPointer(_))
            .WillOnce([expected_index, loaded_pointer](const ::SPIRVSimulator::PointerV& pointer)
                          -> std::optional<::SPIRVSimulator::Value> {
                EXPECT_EQ(pointer.idx_path,
                          (std::vector<uint32_t>{ 4, expected_index, 1 }));
                return ::SPIRVSimulator::Value{ loaded_pointer };
            });
    }

    EXPECT_TRUE(TryQueueDenseReadSidePbufferPointers(
        result_id, representative_pointer, dynamic_index_selection));
    EXPECT_EQ(QueuedPointerPairCount(), 3u);
}

TEST_F(AccessChainReadSidePointerConsumerTests, ThreadSpecificFallbackReadFailureReportsIncompleteCoverage)
{
    constexpr uint32_t base_type_id           = 1100;
    constexpr uint32_t result_type_id         = 1101;
    constexpr uint32_t nested_pointer_type_id = 1102;
    constexpr uint32_t base_id                = 1103;
    constexpr uint32_t result_id              = 1104;
    constexpr uint32_t final_index_id         = 1105;

    const ::SPIRVSimulator::Type base_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClass::StorageClassFunction, CommonTypes::u64);
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClass::StorageClassFunction, nested_pointer_type_id);
    const ::SPIRVSimulator::Type nested_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClass::StorageClassPhysicalStorageBuffer, CommonTypes::u64);

    ::SPIRVSimulator::Value base_value = ::SPIRVSimulator::PointerV{
        1,
        0,
        base_type_id,
        base_id,
        spv::StorageClass::StorageClassFunction,
        {},
    };
    ::SPIRVSimulator::Value final_index_value = uint64_t{ 0 };

    value_meta_.resize(final_index_id + 1);
    value_meta_[final_index_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;

    EXPECT_CALL(*this, GetValue(base_id)).WillOnce(ReturnRef(base_value));
    EXPECT_CALL(*this, GetValue(final_index_id)).WillOnce(ReturnRef(final_index_value));
    EXPECT_CALL(*this, GetTypeByResultId(base_id)).WillOnce(ReturnRef(base_type));
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(nested_pointer_type_id))
        .WillOnce(ReturnRef(nested_pointer_type));
    EXPECT_CALL(*this, SetValue(result_id, _, true)).Times(1);
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(base_id))).Times(1);
    EXPECT_CALL(*this, ReadPointer(_)).WillOnce(Return(std::nullopt));

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };

    ExecuteAccessChain(instruction);

    EXPECT_TRUE(ReadSidePointerCoverageIncomplete());
    EXPECT_EQ(QueuedPointerPairCount(), 0u);
}

// Shader integration cases use the real simulator rather than the opcode mocks above.
// Build: cmake --build build --target access_chain_dense_range_consumer_test -j 4
// Run: ctest --test-dir build --output-on-failure -R '^AccessChainShaderTests\.'
// CTest isolates each case, so a simulator assertion does not hide other failures.
namespace
{
class AccessChainShaderTests : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        simulation_data_.bindings[0][0] = backing_.data();
        simulation_data_.physical_address_buffers[4096] = { sizeof(backing_), backing_.data() };
        simulation_data_.has_compute_num_workgroups = true;
        simulation_data_.compute_num_workgroups = { 1, 1, 1 };
    }

    void RunShader(const char* shader_name)
    {
        const std::string path = std::string(TEST_SHADER_DIR) + "/access_chain/" + shader_name + ".spvasm";
        std::ifstream file(path);
        ASSERT_TRUE(file.is_open()) << path;
        const std::string source{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };

        std::ostringstream diagnostics;
        spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
        tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
            diagnostics << position.line << ": " << message << '\n';
        });
        std::vector<uint32_t> words;
        ASSERT_TRUE(tools.Assemble(source, &words)) << diagnostics.str();
        ASSERT_TRUE(tools.Validate(words)) << diagnostics.str();

        SPIRVSimulator::SPIRVSimulator simulator(
            words, &memory_tracker_, &simulation_data_, &simulation_results_);
        simulator.Run();
    }

    void ExpectExactWriteCoverage(std::initializer_list<uint64_t> byte_offsets)
    {
        // Verify that the shader reached its store and completed compressed
        // writeback without requiring an ordinary full dispatch.
        ASSERT_EQ(backing_[0], 1u);
        ASSERT_FALSE(simulation_results_.full_dispatch_needed)
            << "Expected compressed write coverage instead of full-dispatch fallback";

        // Every invocation's complete uint write must be classified.
        const uint64_t base_address = reinterpret_cast<uint64_t>(backing_.data());
        for (uint64_t offset : byte_offsets)
        {
            SCOPED_TRACE("write byte offset " + std::to_string(offset));
            const uint64_t address = base_address + offset;
            const auto view = memory_tracker_.queryUniformDerivedClassView(
                address,
                0,
                1,
                sizeof(uint32_t));
            ASSERT_TRUE(view.has_value()) << "Missing write coverage and no full-dispatch fallback";
            EXPECT_EQ(view->elementCount(), 1u);
            EXPECT_EQ(view->elementSize(), sizeof(uint32_t));
            EXPECT_NE(memory_tracker_.query(*view) & SPS_FLAG_UNIFORM_DERIVED_RANGE, 0u);
        }
    }

    void ExpectFullDispatchFallback()
    {
        ASSERT_EQ(backing_[0], 1u);
        EXPECT_TRUE(simulation_results_.full_dispatch_needed);
    }

    void BindPointerTable()
    {
        simulation_data_.bindings[0][0] = pointer_table_.data();
        simulation_data_.physical_address_buffers.clear();
        simulation_data_.physical_address_buffers[4096] = { sizeof(first_payload_), &first_payload_ };
        simulation_data_.physical_address_buffers[8192] = { sizeof(second_payload_), &second_payload_ };
    }

    void ExpectPointerSlotCoverage(std::initializer_list<uint64_t> byte_offsets)
    {
        std::set<uint64_t> observed_offsets;
        for (const auto& pointer : simulation_results_.physical_address_data)
        {
            for (const auto& source : pointer.bit_components)
            {
                if (source.source_ptr == pointer_table_.data())
                {
                    EXPECT_EQ(source.bitcount, 64u);
                    observed_offsets.insert(source.byte_offset);
                }
            }
        }

        // Require the representative slot to have been visited. If enumeration
        // cannot cover both slots, the public incomplete-coverage flag must say so.
        EXPECT_EQ(observed_offsets.count(0), 1u);
        if (!simulation_results_.read_side_pointer_coverage_incomplete)
        {
            EXPECT_EQ(observed_offsets, std::set<uint64_t>(byte_offsets));
        }
    }

    alignas(16) std::array<uint32_t, 32> backing_{};
    // Packed slots use elements 0 and 1; stride-16 slots use elements 0 and 2.
    std::array<uint64_t, 4> pointer_table_{ 4096, 8192, 8192, 0 };
    uint32_t first_payload_ = 1;
    uint32_t second_payload_ = 2;
    SPIRVSimulator::MemoryFlagTracker memory_tracker_;
    SPIRVSimulator::SimulationData simulation_data_;
    SPIRVSimulator::SimulationResults simulation_results_;
};

TEST_F(AccessChainShaderTests, MaskedIndexCoversBothWritesOrRequestsFullDispatch)
{
    ASSERT_NO_FATAL_FAILURE(RunShader("masked_index"));
    ExpectFullDispatchFallback();
}

TEST_F(AccessChainShaderTests, VaryingBaseAndIndexCoverBothRowsOrRequestFullDispatch)
{
    ASSERT_NO_FATAL_FAILURE(RunShader("varying_base"));
    ExpectFullDispatchFallback();
}

TEST_F(AccessChainShaderTests, StridedPointerTableEnumeratesSlotsOrReportsIncompleteCoverage)
{
    BindPointerTable();
    ASSERT_NO_FATAL_FAILURE(RunShader("strided_pointer_table"));
    ExpectPointerSlotCoverage({ 0, 16 });
}

TEST_F(AccessChainShaderTests, NonFinalVaryingIndexEnumeratesSlotsOrReportsIncompleteCoverage)
{
    BindPointerTable();
    ASSERT_NO_FATAL_FAILURE(RunShader("record_pointer_table"));
    ExpectPointerSlotCoverage({ 0, 8 });
}

// Controls differ from the regressions by one operation or layout choice.
TEST_F(AccessChainShaderTests, DirectIndexCoversBothWrites)
{
    ASSERT_NO_FATAL_FAILURE(RunShader("direct_index"));
    ExpectExactWriteCoverage({ 0, 4 });
}

TEST_F(AccessChainShaderTests, FixedBaseCoversOneRow)
{
    ASSERT_NO_FATAL_FAILURE(RunShader("fixed_base"));
    ExpectFullDispatchFallback();
}

TEST_F(AccessChainShaderTests, PackedPointerTableEnumeratesSlots)
{
    BindPointerTable();
    ASSERT_NO_FATAL_FAILURE(RunShader("packed_pointer_table"));
    ExpectPointerSlotCoverage({ 0, 8 });
}
} // namespace
