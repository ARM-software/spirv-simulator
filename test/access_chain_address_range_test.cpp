#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class AccessChainAddressRangeTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD((std::pair<std::byte*, uint64_t>),
                ResolvePointerV,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(size_t, GetBitsizeOfType, (uint32_t type_id), (const, override));
    MOCK_METHOD(uint64_t, GetArrayLength, (uint32_t length_id), (const, override));
    MOCK_METHOD(uint32_t,
                GetTargetPointerType,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));

  protected:
    void SetUp() override
    {
        simulation_data_ = &simulation_data_storage_;
        persistent_data_ = &persistent_data_storage_;
        simulation_data_storage_.physical_address_buffers.emplace(
            0x1000, std::make_pair(backing_.size(), static_cast<void*>(backing_.data())));
    }

    void SetDynamicIndexRange(uint32_t result_id, uint64_t min, uint64_t max, uint64_t stride = 1)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
        value_meta_[result_id].value_range =
            ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, min, max, stride };
        value_meta_[result_id].invocation_dependence =
            ::SPIRVSimulator::InvocationDependence::Varying;
    }

    void SetBaseInvariant(uint32_t result_id)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
        value_meta_[result_id].invocation_dependence =
            ::SPIRVSimulator::InvocationDependence::Invariant;
    }

    void SetFixedUnsignedIndex(uint32_t result_id, uint64_t value, uint64_t flags = 0)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
        value_meta_[result_id].invocation_dependence =
            ::SPIRVSimulator::InvocationDependence::Invariant;
        value_meta_[result_id].flags = flags;

        EXPECT_CALL(*this, GetTypeByResultId(result_id))
            .WillOnce(ReturnRef(types_.at(CommonTypes::u64)));
        EXPECT_CALL(*this, GetValue(result_id))
            .WillOnce(ReturnRefOfCopy(::SPIRVSimulator::Value{ value }));
    }

    void SetRuntimeArrayLength(const ::SPIRVSimulator::PointerV& pointer, size_t offset, size_t length)
    {
        simulation_data_storage_.rt_array_lengths[pointer.pointer_handle][offset] = length;
    }

    const ::SPIRVSimulator::ValueMetadata& GetMetadata(uint32_t result_id) const
    {
        return value_meta_[result_id];
    }

    void ExpectPointerDependence(
        uint32_t result_id,
        ::SPIRVSimulator::InvocationDependence expected_dependence,
        bool expect_legacy_thread_specific) const
    {
        const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
        EXPECT_EQ(metadata.invocation_dependence, expected_dependence);
        EXPECT_EQ((metadata.flags & SPS_FLAG_THREAD_SPECIFIC) != 0,
                  expect_legacy_thread_specific);
    }

    void ExpectResetAddressRangeMetadata(
        uint32_t result_id,
        ::SPIRVSimulator::InvocationDependence expected_dependence,
        bool expect_legacy_thread_specific) const
    {
        const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
        EXPECT_FALSE(metadata.address_range_valid);
        EXPECT_EQ(metadata.address_range_min, 0u);
        EXPECT_EQ(metadata.address_range_max, 0u);
        EXPECT_EQ(metadata.address_range_stride, 1u);
        EXPECT_EQ(metadata.address_element_size, 0u);
        ExpectPointerDependence(
            result_id, expected_dependence, expect_legacy_thread_specific);
    }

    void SetAddressRange(uint32_t pointer_id,
                         uint64_t min,
                         uint64_t max,
                         uint64_t stride,
                         uint64_t element_size)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), pointer_id + 1));
        ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[pointer_id];
        metadata.address_range_valid  = true;
        metadata.address_range_min    = min;
        metadata.address_range_max    = max;
        metadata.address_range_stride = stride;
        metadata.address_element_size = element_size;
    }

    void SetDefiningOpcode(uint32_t result_id, spv::Op opcode)
    {
        instructions_.push_back(::SPIRVSimulator::Instruction{
            .opcode = opcode,
            .word_count = 0,
            .words = {},
        });
        SetInstructionIndexForResultId(result_id, instructions_.size() - 1);
    }

    std::array<std::byte, 64> backing_{};
    ::SPIRVSimulator::SimulationData simulation_data_storage_{};
    ::SPIRVSimulator::InternalPersistentData persistent_data_storage_{};
};

#ifdef DEBUG_BUILD
TEST(InternalPersistentDataDiagnostics, ReportsEachInvocationDependenceFallbackOpcodeOnlyOnce)
{
    ::SPIRVSimulator::InternalPersistentData persistent_data;

    EXPECT_TRUE(persistent_data.MarkInvocationDependenceFallbackOpcodeReported(
        spv::Op::OpShiftRightLogical));
    EXPECT_FALSE(persistent_data.MarkInvocationDependenceFallbackOpcodeReported(
        spv::Op::OpShiftRightLogical));
    EXPECT_TRUE(persistent_data.MarkInvocationDependenceFallbackOpcodeReported(
        spv::Op::OpShiftLeftLogical));
}
#endif

TEST_F(AccessChainAddressRangeTests, UnknownIndexWithLegacyFlagFallsBackWithoutBecomingDynamic)
{
    constexpr uint32_t result_type_id = 5400;
    constexpr uint32_t result_id      = 5401;
    constexpr uint32_t base_id        = 5402;
    constexpr uint32_t index_id       = 5403;
    constexpr uint32_t base_pointer_type_id = 5404;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id)).Times(0);
    EXPECT_CALL(*this, GetTypeByResultId(index_id)).Times(0);
    EXPECT_CALL(*this, GetValue(index_id)).Times(0);
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    value_meta_.resize(index_id + 1);
    value_meta_[index_id].flags = SPS_FLAG_THREAD_SPECIFIC;
    SetDefiningOpcode(index_id, spv::Op::OpShiftLeftLogical);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    testing::internal::CaptureStdout();
    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    const std::string diagnostic = testing::internal::GetCapturedStdout();

#ifdef DEBUG_BUILD
    EXPECT_THAT(diagnostic, HasSubstr("AccessChain index %5403"));
    EXPECT_THAT(diagnostic, HasSubstr("defining opcode: OpShiftLeftLogical"));
    EXPECT_THAT(diagnostic, HasSubstr("using conservative fallback"));
#else
    EXPECT_TRUE(diagnostic.empty());
#endif
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

TEST_F(AccessChainAddressRangeTests, UnknownIndexWithThreadDependentRangeFallsBackWithoutBecomingDynamic)
{
    constexpr uint32_t result_type_id = 5410;
    constexpr uint32_t result_id      = 5411;
    constexpr uint32_t base_id        = 5412;
    constexpr uint32_t index_id       = 5413;
    constexpr uint32_t base_pointer_type_id = 5414;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id)).Times(0);
    EXPECT_CALL(*this, GetTypeByResultId(index_id)).Times(0);
    EXPECT_CALL(*this, GetValue(index_id)).Times(0);
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    value_meta_.resize(index_id + 1);
    value_meta_[index_id].value_range =
        ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, 0, 2, 1 };

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

TEST_F(AccessChainAddressRangeTests, InvariantFixedIndexIgnoresLegacyThreadSpecificFlag)
{
    constexpr uint32_t result_type_id       = 5420;
    constexpr uint32_t result_id            = 5421;
    constexpr uint32_t base_id              = 5422;
    constexpr uint32_t fixed_index_id       = 5423;
    constexpr uint32_t dynamic_index_id     = 5424;
    constexpr uint32_t outer_array_type_id  = 5425;
    constexpr uint32_t inner_array_type_id  = 5426;
    constexpr uint32_t outer_length_id      = 5427;
    constexpr uint32_t inner_length_id      = 5428;
    constexpr uint32_t base_pointer_type_id = 5429;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type outer_array_type =
        ::SPIRVSimulator::Type::Array(inner_array_type_id, outer_length_id);
    const ::SPIRVSimulator::Type inner_array_type =
        ::SPIRVSimulator::Type::Array(CommonTypes::u64, inner_length_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, outer_array_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(outer_array_type_id))
        .WillRepeatedly(ReturnRef(outer_array_type));
    EXPECT_CALL(*this, GetTypeByTypeId(inner_array_type_id))
        .WillRepeatedly(ReturnRef(inner_array_type));
    EXPECT_CALL(*this, GetArrayLength(outer_length_id)).WillOnce(Return(1));
    EXPECT_CALL(*this, GetArrayLength(inner_length_id)).WillOnce(Return(3));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .Times(3)
        .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(),
                                                    pointer.idx_path[1] * 8 };
        });

    SetBaseInvariant(base_id);
    SetFixedUnsignedIndex(fixed_index_id, 0, SPS_FLAG_THREAD_SPECIFIC);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);

    std::vector<uint32_t> words = {
        0, result_type_id, result_id, base_id, fixed_index_id, dynamic_index_id
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0, 0 },
    };

    ASSERT_TRUE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    const uint64_t base_address = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_TRUE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min, base_address);
    EXPECT_EQ(metadata.address_range_max, base_address + 16);
    EXPECT_EQ(metadata.address_range_stride, 8u);
    EXPECT_EQ(metadata.address_element_size, 8u);
    ExpectPointerDependence(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

TEST_F(AccessChainAddressRangeTests, UnknownBeforeVaryingStillProducesVaryingResultDependence)
{
    constexpr uint32_t result_type_id = 5430;
    constexpr uint32_t result_id      = 5431;
    constexpr uint32_t base_id        = 5432;
    constexpr uint32_t unknown_index_id = 5433;
    constexpr uint32_t dynamic_index_id = 5434;
    constexpr uint32_t base_pointer_type_id = 5435;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id)).Times(0);
    EXPECT_CALL(*this, GetTypeByResultId(unknown_index_id)).Times(0);
    EXPECT_CALL(*this, GetValue(unknown_index_id)).Times(0);
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);
    SetAddressRange(result_id, 0x1000, 0x1010, 8, 8);

    std::vector<uint32_t> words = {
        0, result_type_id, result_id, base_id, unknown_index_id, dynamic_index_id
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0, 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

TEST_F(AccessChainAddressRangeTests, AllInvariantIndexesProduceInvariantResultWithoutAddressRange)
{
    constexpr uint32_t result_type_id = 5440;
    constexpr uint32_t result_id      = 5441;
    constexpr uint32_t base_id        = 5442;
    constexpr uint32_t fixed_index_id = 5443;
    constexpr uint32_t base_pointer_type_id = 5444;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id)).Times(0);
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetFixedUnsignedIndex(fixed_index_id, 0);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, fixed_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Invariant, false);
}

TEST_F(AccessChainAddressRangeTests, UnsupportedInvariantFixedIndexLeavesResultUnknown)
{
    constexpr uint32_t result_type_id = 5450;
    constexpr uint32_t result_id      = 5451;
    constexpr uint32_t base_id        = 5452;
    constexpr uint32_t fixed_index_id = 5453;
    constexpr uint32_t base_pointer_type_id = 5454;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id)).Times(0);
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetFixedUnsignedIndex(
        fixed_index_id, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1);
    SetAddressRange(result_id, 0x1000, 0x1010, 8, 8);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, fixed_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Unknown, false);
}

TEST_F(AccessChainAddressRangeTests, BaseDependenceUnknownIsTemporarilyAccepted)
{
    constexpr uint32_t result_type_id       = 5460;
    constexpr uint32_t result_id            = 5461;
    constexpr uint32_t base_id              = 5462;
    constexpr uint32_t dynamic_index_id     = 5463;
    constexpr uint32_t container_type_id    = 5464;
    constexpr uint32_t array_length_id      = 5465;
    constexpr uint32_t base_pointer_type_id = 5466;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::Array(CommonTypes::u64, array_length_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id))
        .WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(3));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .Times(3)
        .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(),
                                                    pointer.idx_path.back() * 8 };
        });

    value_meta_.resize(dynamic_index_id + 1);
    ASSERT_EQ(value_meta_[base_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Unknown);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    ASSERT_TRUE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    const uint64_t base_address = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_TRUE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min, base_address);
    EXPECT_EQ(metadata.address_range_max, base_address + 16);
    EXPECT_EQ(metadata.address_range_stride, 8u);
    EXPECT_EQ(metadata.address_element_size, 8u);
    ExpectPointerDependence(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

struct InvalidDynamicIndexRangeCase
{
    const char*                                  name;
    ::SPIRVSimulator::DenseValueRangeInfo value_range;
};

class DynamicIndexRangeValidationTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<InvalidDynamicIndexRangeCase>
{};

TEST_P(DynamicIndexRangeValidationTests, RejectsInvalidRange)
{
    constexpr uint32_t result_type_id   = 5470;
    constexpr uint32_t result_id        = 5471;
    constexpr uint32_t base_id          = 5472;
    constexpr uint32_t dynamic_index_id = 5473;
    constexpr uint32_t base_pointer_type_id = 5474;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id)).Times(0);
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    value_meta_.resize(dynamic_index_id + 1);
    value_meta_[dynamic_index_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;
    value_meta_[dynamic_index_id].value_range = GetParam().value_range;
    SetAddressRange(result_id, 0x1000, 0x1010, 8, 8);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidDynamicIndexRanges,
    DynamicIndexRangeValidationTests,
    Values(InvalidDynamicIndexRangeCase{
               "Invalid", { false, true, true, 0, 2, 1 } },
           InvalidDynamicIndexRangeCase{
               "NotThreadDependent", { true, false, true, 0, 2, 1 } },
           InvalidDynamicIndexRangeCase{
               "NonDense", { true, true, false, 0, 2, 1 } },
           InvalidDynamicIndexRangeCase{
               "ZeroStride", { true, true, true, 0, 2, 0 } },
           InvalidDynamicIndexRangeCase{
               "MaxLessThanMin", { true, true, true, 2, 1, 1 } },
           InvalidDynamicIndexRangeCase{
               "UnreachableMax", { true, true, true, 0, 3, 2 } },
           InvalidDynamicIndexRangeCase{
               "MinExceedsUint32",
               { true,
                 true,
                 true,
                 static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1,
                 static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1,
                 1 } },
           InvalidDynamicIndexRangeCase{
               "MaxExceedsUint32",
               { true,
                 true,
                 true,
                 std::numeric_limits<uint32_t>::max(),
                 static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1,
                 1 } }),
    [](const TestParamInfo<InvalidDynamicIndexRangeCase>& info) {
        return std::string(info.param.name);
    });

struct InheritedPathMappingCase
{
    const char* name;
    spv::Op     opcode;
};

class InheritedPathMappingTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<InheritedPathMappingCase>
{};

TEST_P(InheritedPathMappingTests, MapsDynamicOperandToCorrectFullPath)
{
    constexpr uint32_t result_type_id         = 5480;
    constexpr uint32_t result_id              = 5481;
    constexpr uint32_t base_id                = 5482;
    constexpr uint32_t dynamic_index_id       = 5483;
    constexpr uint32_t field_index_id         = 5484;
    constexpr uint32_t base_pointer_type_id   = 5485;
    constexpr uint32_t root_struct_type_id    = 5486;
    constexpr uint32_t array_type_id          = 5487;
    constexpr uint32_t element_struct_type_id = 5488;
    constexpr uint32_t array_length_id        = 5489;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, root_struct_type_id);
    const ::SPIRVSimulator::Type root_struct_type =
        ::SPIRVSimulator::Type::Struct(root_struct_type_id);
    const ::SPIRVSimulator::Type array_type =
        ::SPIRVSimulator::Type::Array(element_struct_type_id, array_length_id);
    const ::SPIRVSimulator::Type element_struct_type =
        ::SPIRVSimulator::Type::Struct(element_struct_type_id);

    struct_members_[root_struct_type_id]    = { array_type_id };
    struct_members_[element_struct_type_id] = { CommonTypes::u64, CommonTypes::u64 };

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(root_struct_type_id))
        .WillRepeatedly(ReturnRef(root_struct_type));
    EXPECT_CALL(*this, GetTypeByTypeId(array_type_id)).WillRepeatedly(ReturnRef(array_type));
    EXPECT_CALL(*this, GetTypeByTypeId(element_struct_type_id))
        .WillRepeatedly(ReturnRef(element_struct_type));
    EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(4));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));

    {
        InSequence sequence;
        EXPECT_CALL(*this,
                    ResolvePointerV(Truly([](const ::SPIRVSimulator::PointerV& pointer) {
                        return pointer.idx_path == std::vector<uint32_t>{ 0, 1, 1 };
                    })))
            .WillOnce(Return(std::make_pair(backing_.data(), uint64_t{ 24 })));
        EXPECT_CALL(*this,
                    ResolvePointerV(Truly([](const ::SPIRVSimulator::PointerV& pointer) {
                        return pointer.idx_path == std::vector<uint32_t>{ 0, 3, 1 };
                    })))
            .WillOnce(Return(std::make_pair(backing_.data(), uint64_t{ 56 })));
        EXPECT_CALL(*this,
                    ResolvePointerV(Truly([](const ::SPIRVSimulator::PointerV& pointer) {
                        return pointer.idx_path == std::vector<uint32_t>{ 0, 2, 1 };
                    })))
            .WillOnce(Return(std::make_pair(backing_.data(), uint64_t{ 40 })));
    }

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 1, 3);
    SetFixedUnsignedIndex(field_index_id, 1);

    std::vector<uint32_t> words = {
        0, result_type_id, result_id, base_id, dynamic_index_id, field_index_id
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = GetParam().opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0, 0, 1 },
    };

    DenseAccessChainIndexSelection dynamic_index_selection;
    ASSERT_TRUE(TrySetDenseAccessChainAddressRange(
        instruction, result_pointer, &dynamic_index_selection));

    EXPECT_EQ(dynamic_index_selection.dynamic_index_path_position, 1u);
    EXPECT_EQ(dynamic_index_selection.dynamic_index_range.min, 1u);
    EXPECT_EQ(dynamic_index_selection.dynamic_index_range.max, 3u);
    EXPECT_EQ(dynamic_index_selection.dynamic_index_range.stride, 1u);

    const uint64_t base_address = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_TRUE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min, base_address + 24);
    EXPECT_EQ(metadata.address_range_max, base_address + 56);
    EXPECT_EQ(metadata.address_range_stride, 16u);
    EXPECT_EQ(metadata.address_element_size, 8u);
    ExpectPointerDependence(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    AccessChainOpcodes,
    InheritedPathMappingTests,
    Values(InheritedPathMappingCase{ "AccessChain", spv::Op::OpAccessChain },
           InheritedPathMappingCase{ "InBoundsAccessChain", spv::Op::OpInBoundsAccessChain }),
    [](const TestParamInfo<InheritedPathMappingCase>& info) {
        return std::string(info.param.name);
    });

enum class FixedPrefixBoundsFailure
{
    StructMemberOutOfBounds,
    ArrayIndexOutOfBounds,
    VectorComponentOutOfBounds,
    RuntimeArrayLengthMissing,
    RuntimeArrayIndexOutOfBounds,
};

struct FixedPrefixBoundsCase
{
    const char*              name;
    FixedPrefixBoundsFailure failure;
};

class FixedPrefixBoundsTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<FixedPrefixBoundsCase>
{};

TEST_P(FixedPrefixBoundsTests, RejectsInvalidFixedPrefix)
{
    constexpr uint32_t result_type_id           = 5500;
    constexpr uint32_t result_id                = 5501;
    constexpr uint32_t base_id                  = 5502;
    constexpr uint32_t fixed_index_id           = 5503;
    constexpr uint32_t dynamic_index_id         = 5504;
    constexpr uint32_t base_pointer_type_id     = 5505;
    constexpr uint32_t prefix_container_type_id = 5506;
    constexpr uint32_t dynamic_container_type_id = 5507;
    constexpr uint32_t prefix_length_id          = 5508;
    constexpr uint32_t dynamic_length_id         = 5509;

    const FixedPrefixBoundsCase& test_case = GetParam();
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, prefix_container_type_id);
    const ::SPIRVSimulator::Type dynamic_container_type =
        ::SPIRVSimulator::Type::Array(CommonTypes::u64, dynamic_length_id);

    ::SPIRVSimulator::Type prefix_container_type;
    uint64_t               fixed_index = 0;
    switch (test_case.failure)
    {
        case FixedPrefixBoundsFailure::StructMemberOutOfBounds:
            prefix_container_type =
                ::SPIRVSimulator::Type::Struct(prefix_container_type_id);
            struct_members_[prefix_container_type_id] = { dynamic_container_type_id };
            fixed_index = 1;
            break;
        case FixedPrefixBoundsFailure::ArrayIndexOutOfBounds:
            prefix_container_type =
                ::SPIRVSimulator::Type::Array(dynamic_container_type_id, prefix_length_id);
            EXPECT_CALL(*this, GetArrayLength(prefix_length_id)).WillOnce(Return(1));
            fixed_index = 1;
            break;
        case FixedPrefixBoundsFailure::VectorComponentOutOfBounds:
            prefix_container_type =
                ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 1);
            fixed_index = 1;
            break;
        case FixedPrefixBoundsFailure::RuntimeArrayLengthMissing:
            prefix_container_type =
                ::SPIRVSimulator::Type::RuntimeArray(dynamic_container_type_id);
            fixed_index = 0;
            break;
        case FixedPrefixBoundsFailure::RuntimeArrayIndexOutOfBounds:
            prefix_container_type =
                ::SPIRVSimulator::Type::RuntimeArray(dynamic_container_type_id);
            fixed_index = 2;
            break;
    }

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(prefix_container_type_id))
        .WillRepeatedly(ReturnRef(prefix_container_type));
    EXPECT_CALL(*this, GetTypeByTypeId(dynamic_container_type_id)).Times(0);
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetFixedUnsignedIndex(fixed_index_id, fixed_index);
    SetDynamicIndexRange(dynamic_index_id, 0, 1);

    std::vector<uint32_t> words = {
        0, result_type_id, result_id, base_id, fixed_index_id, dynamic_index_id
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { static_cast<uint32_t>(fixed_index), 0 },
    };
    if (test_case.failure == FixedPrefixBoundsFailure::RuntimeArrayIndexOutOfBounds)
    {
        SetRuntimeArrayLength(result_pointer, 0, 2);
    }

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidFixedPrefixes,
    FixedPrefixBoundsTests,
    Values(FixedPrefixBoundsCase{
               "StructMemberOutOfBounds", FixedPrefixBoundsFailure::StructMemberOutOfBounds },
           FixedPrefixBoundsCase{
               "ArrayIndexOutOfBounds", FixedPrefixBoundsFailure::ArrayIndexOutOfBounds },
           FixedPrefixBoundsCase{
               "VectorComponentOutOfBounds",
               FixedPrefixBoundsFailure::VectorComponentOutOfBounds },
           FixedPrefixBoundsCase{
               "RuntimeArrayLengthMissing",
               FixedPrefixBoundsFailure::RuntimeArrayLengthMissing },
           FixedPrefixBoundsCase{
               "RuntimeArrayIndexOutOfBounds",
               FixedPrefixBoundsFailure::RuntimeArrayIndexOutOfBounds }),
    [](const TestParamInfo<FixedPrefixBoundsCase>& info) {
        return std::string(info.param.name);
    });

enum class DynamicContainerBoundsFailure
{
    VectorRangeOutOfBounds,
    RuntimeArrayExplicitLengthOutOfBounds,
    NonPhysicalRuntimeArrayLengthMissing,
    UnsupportedStructContainer,
};

struct DynamicContainerBoundsCase
{
    const char*                   name;
    DynamicContainerBoundsFailure failure;
};

class DynamicContainerBoundsTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<DynamicContainerBoundsCase>
{};

TEST_P(DynamicContainerBoundsTests, RejectsInvalidDynamicContainer)
{
    constexpr uint32_t result_type_id       = 5520;
    constexpr uint32_t result_id            = 5521;
    constexpr uint32_t base_id              = 5522;
    constexpr uint32_t dynamic_index_id     = 5523;
    constexpr uint32_t base_pointer_type_id = 5524;
    constexpr uint32_t container_type_id    = 5525;

    const DynamicContainerBoundsCase& test_case = GetParam();
    uint32_t storage_class = spv::StorageClassPhysicalStorageBuffer;
    uint64_t dynamic_max   = 2;

    ::SPIRVSimulator::Type container_type;
    switch (test_case.failure)
    {
        case DynamicContainerBoundsFailure::VectorRangeOutOfBounds:
            container_type = ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 2);
            break;
        case DynamicContainerBoundsFailure::RuntimeArrayExplicitLengthOutOfBounds:
            container_type = ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);
            break;
        case DynamicContainerBoundsFailure::NonPhysicalRuntimeArrayLengthMissing:
            storage_class = spv::StorageClassFunction;
            dynamic_max   = 1;
            container_type = ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);
            break;
        case DynamicContainerBoundsFailure::UnsupportedStructContainer:
            dynamic_max = 1;
            container_type = ::SPIRVSimulator::Type::Struct(container_type_id);
            struct_members_[container_type_id] = { CommonTypes::u64, CommonTypes::u64 };
            break;
    }

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Pointer(storage_class, CommonTypes::u64);
    const ::SPIRVSimulator::Type base_pointer_type =
        ::SPIRVSimulator::Type::Pointer(storage_class, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id))
        .WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, dynamic_max);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = storage_class,
        .idx_path = { 0 },
    };
    if (test_case.failure ==
        DynamicContainerBoundsFailure::RuntimeArrayExplicitLengthOutOfBounds)
    {
        SetRuntimeArrayLength(result_pointer, 0, 2);
    }

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidDynamicContainers,
    DynamicContainerBoundsTests,
    Values(DynamicContainerBoundsCase{
               "VectorRangeOutOfBounds",
               DynamicContainerBoundsFailure::VectorRangeOutOfBounds },
           DynamicContainerBoundsCase{
               "RuntimeArrayExplicitLengthOutOfBounds",
               DynamicContainerBoundsFailure::RuntimeArrayExplicitLengthOutOfBounds },
           DynamicContainerBoundsCase{
               "NonPhysicalRuntimeArrayLengthMissing",
               DynamicContainerBoundsFailure::NonPhysicalRuntimeArrayLengthMissing },
           DynamicContainerBoundsCase{
               "UnsupportedStructContainer",
               DynamicContainerBoundsFailure::UnsupportedStructContainer }),
    [](const TestParamInfo<DynamicContainerBoundsCase>& info) {
        return std::string(info.param.name);
    });

enum class FixedSuffixBoundsFailure
{
    StructMemberOutOfBounds,
    ArrayIndexOutOfBounds,
    VectorComponentOutOfBounds,
    RuntimeArrayUnsupported,
};

struct FixedSuffixBoundsCase
{
    const char*              name;
    FixedSuffixBoundsFailure failure;
};

class FixedSuffixBoundsTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<FixedSuffixBoundsCase>
{};

TEST_P(FixedSuffixBoundsTests, RejectsInvalidFixedSuffix)
{
    constexpr uint32_t result_type_id       = 5540;
    constexpr uint32_t result_id            = 5541;
    constexpr uint32_t base_id              = 5542;
    constexpr uint32_t dynamic_index_id     = 5543;
    constexpr uint32_t fixed_index_id       = 5544;
    constexpr uint32_t base_pointer_type_id = 5545;
    constexpr uint32_t outer_array_type_id  = 5546;
    constexpr uint32_t suffix_container_type_id = 5547;
    constexpr uint32_t outer_length_id          = 5548;
    constexpr uint32_t suffix_length_id         = 5549;

    const FixedSuffixBoundsCase& test_case = GetParam();
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, outer_array_type_id);
    const ::SPIRVSimulator::Type outer_array_type =
        ::SPIRVSimulator::Type::Array(suffix_container_type_id, outer_length_id);

    ::SPIRVSimulator::Type suffix_container_type;
    uint64_t               fixed_index = 2;
    switch (test_case.failure)
    {
        case FixedSuffixBoundsFailure::StructMemberOutOfBounds:
            suffix_container_type =
                ::SPIRVSimulator::Type::Struct(suffix_container_type_id);
            struct_members_[suffix_container_type_id] = { CommonTypes::u64, CommonTypes::u64 };
            break;
        case FixedSuffixBoundsFailure::ArrayIndexOutOfBounds:
            suffix_container_type =
                ::SPIRVSimulator::Type::Array(CommonTypes::u64, suffix_length_id);
            EXPECT_CALL(*this, GetArrayLength(suffix_length_id)).WillOnce(Return(2));
            break;
        case FixedSuffixBoundsFailure::VectorComponentOutOfBounds:
            suffix_container_type = ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 2);
            break;
        case FixedSuffixBoundsFailure::RuntimeArrayUnsupported:
            suffix_container_type =
                ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);
            fixed_index = 0;
            break;
    }

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(outer_array_type_id))
        .WillRepeatedly(ReturnRef(outer_array_type));
    EXPECT_CALL(*this, GetTypeByTypeId(suffix_container_type_id))
        .WillRepeatedly(ReturnRef(suffix_container_type));
    EXPECT_CALL(*this, GetArrayLength(outer_length_id)).WillOnce(Return(2));
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 1);
    SetFixedUnsignedIndex(fixed_index_id, fixed_index);

    std::vector<uint32_t> words = {
        0, result_type_id, result_id, base_id, dynamic_index_id, fixed_index_id
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0, static_cast<uint32_t>(fixed_index) },
    };
    if (test_case.failure == FixedSuffixBoundsFailure::RuntimeArrayUnsupported)
    {
        SetRuntimeArrayLength(result_pointer, 0, 1);
    }

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidFixedSuffixes,
    FixedSuffixBoundsTests,
    Values(FixedSuffixBoundsCase{
               "StructMemberOutOfBounds", FixedSuffixBoundsFailure::StructMemberOutOfBounds },
           FixedSuffixBoundsCase{
               "ArrayIndexOutOfBounds", FixedSuffixBoundsFailure::ArrayIndexOutOfBounds },
           FixedSuffixBoundsCase{
               "VectorComponentOutOfBounds",
               FixedSuffixBoundsFailure::VectorComponentOutOfBounds },
           FixedSuffixBoundsCase{
               "RuntimeArrayUnsupported",
               FixedSuffixBoundsFailure::RuntimeArrayUnsupported }),
    [](const TestParamInfo<FixedSuffixBoundsCase>& info) {
        return std::string(info.param.name);
    });

enum class InvalidWholeElementBitSize
{
    IntZero,
    IntNonByteAligned,
    PhysicalPointerZero,
    PhysicalPointerNonByteAligned,
    VectorComponentZero,
    VectorComponentNonByteAligned,
    WholeVectorZero,
    WholeVectorNonByteAligned,
};

struct InvalidWholeElementBitSizeCase
{
    const char*                name;
    InvalidWholeElementBitSize invalid_size;
};

class InvalidWholeElementBitSizeTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<InvalidWholeElementBitSizeCase>
{};

TEST_P(InvalidWholeElementBitSizeTests, RejectsInvalidBitSize)
{
    constexpr uint32_t result_type_id           = 5560;
    constexpr uint32_t result_id                = 5561;
    constexpr uint32_t base_id                  = 5562;
    constexpr uint32_t dynamic_index_id         = 5563;
    constexpr uint32_t base_pointer_type_id     = 5564;
    constexpr uint32_t container_type_id        = 5565;
    constexpr uint32_t result_pointee_type_id   = 5566;
    constexpr uint32_t vector_component_type_id = 5567;
    constexpr uint32_t array_length_id          = 5568;

    const InvalidWholeElementBitSizeCase& test_case = GetParam();
    bool   is_vector          = false;
    bool   invalid_component = false;
    size_t result_bit_size    = 0;
    size_t component_bit_size = 32;

    ::SPIRVSimulator::Type result_pointee_type;
    ::SPIRVSimulator::Type vector_component_type =
        ::SPIRVSimulator::Type::Int(32, false);
    switch (test_case.invalid_size)
    {
        case InvalidWholeElementBitSize::IntZero:
            result_pointee_type = ::SPIRVSimulator::Type::Int(32, false);
            result_bit_size = 0;
            break;
        case InvalidWholeElementBitSize::IntNonByteAligned:
            result_pointee_type = ::SPIRVSimulator::Type::Int(32, false);
            result_bit_size = 7;
            break;
        case InvalidWholeElementBitSize::PhysicalPointerZero:
            result_pointee_type = ::SPIRVSimulator::Type::Pointer(
                spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
            result_bit_size = 0;
            break;
        case InvalidWholeElementBitSize::PhysicalPointerNonByteAligned:
            result_pointee_type = ::SPIRVSimulator::Type::Pointer(
                spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
            result_bit_size = 7;
            break;
        case InvalidWholeElementBitSize::VectorComponentZero:
            is_vector = true;
            invalid_component = true;
            component_bit_size = 0;
            result_pointee_type =
                ::SPIRVSimulator::Type::Vector(vector_component_type_id, 2);
            break;
        case InvalidWholeElementBitSize::VectorComponentNonByteAligned:
            is_vector = true;
            invalid_component = true;
            component_bit_size = 7;
            result_pointee_type =
                ::SPIRVSimulator::Type::Vector(vector_component_type_id, 2);
            break;
        case InvalidWholeElementBitSize::WholeVectorZero:
            is_vector = true;
            result_bit_size = 0;
            result_pointee_type =
                ::SPIRVSimulator::Type::Vector(vector_component_type_id, 2);
            break;
        case InvalidWholeElementBitSize::WholeVectorNonByteAligned:
            is_vector = true;
            result_bit_size = 63;
            result_pointee_type =
                ::SPIRVSimulator::Type::Vector(vector_component_type_id, 2);
            break;
    }

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, result_pointee_type_id);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::Array(result_pointee_type_id, array_length_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id))
        .WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetTypeByTypeId(result_pointee_type_id))
        .WillRepeatedly(ReturnRef(result_pointee_type));
    EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(3));
    if (is_vector)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(vector_component_type_id))
            .WillRepeatedly(ReturnRef(vector_component_type));
        EXPECT_CALL(*this, GetBitsizeOfType(vector_component_type_id))
            .WillOnce(Return(component_bit_size));
        if (invalid_component)
        {
            EXPECT_CALL(*this, GetBitsizeOfType(result_pointee_type_id)).Times(0);
        }
        else
        {
            EXPECT_CALL(*this, GetBitsizeOfType(result_pointee_type_id))
                .WillOnce(Return(result_bit_size));
        }
    }
    else
    {
        EXPECT_CALL(*this, GetBitsizeOfType(result_pointee_type_id))
            .WillOnce(Return(result_bit_size));
    }
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidWholeElementBitSizes,
    InvalidWholeElementBitSizeTests,
    Values(InvalidWholeElementBitSizeCase{
               "IntZeroBitSize", InvalidWholeElementBitSize::IntZero },
           InvalidWholeElementBitSizeCase{
               "IntNonByteAlignedBitSize", InvalidWholeElementBitSize::IntNonByteAligned },
           InvalidWholeElementBitSizeCase{
               "PhysicalPointerZeroBitSize",
               InvalidWholeElementBitSize::PhysicalPointerZero },
           InvalidWholeElementBitSizeCase{
               "PhysicalPointerNonByteAlignedBitSize",
               InvalidWholeElementBitSize::PhysicalPointerNonByteAligned },
           InvalidWholeElementBitSizeCase{
               "VectorComponentZeroBitSize",
               InvalidWholeElementBitSize::VectorComponentZero },
           InvalidWholeElementBitSizeCase{
               "VectorComponentNonByteAlignedBitSize",
               InvalidWholeElementBitSize::VectorComponentNonByteAligned },
           InvalidWholeElementBitSizeCase{
               "WholeVectorZeroBitSize", InvalidWholeElementBitSize::WholeVectorZero },
           InvalidWholeElementBitSizeCase{
               "WholeVectorNonByteAlignedBitSize",
               InvalidWholeElementBitSize::WholeVectorNonByteAligned }),
    [](const TestParamInfo<InvalidWholeElementBitSizeCase>& info) {
        return std::string(info.param.name);
    });

enum class FinalIndexContainerKind
{
    RuntimeArray,
    FixedArray,
    Vector,
};

enum class FinalIndexResultKind
{
    U32,
    U64,
    F32,
    PhysicalPointer,
    UVec2,
    FVec2,
};

struct FinalDynamicIndexCase
{
    const char*             name;
    FinalIndexContainerKind container_kind;
    FinalIndexResultKind    result_kind;
    uint64_t                index_min;
    uint64_t                index_max;
    uint64_t                index_stride;
    uint64_t                byte_step;
    uint64_t                element_size;
};

// Verifies that a supported final dynamic index publishes the exact dense range
// across accepted scalar or pointer, array or vector, and single or multi-element cases.
class FinalDynamicIndexAddressRangeTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<FinalDynamicIndexCase>
{};

TEST_P(FinalDynamicIndexAddressRangeTests, ProducesDenseRange)
{
    constexpr uint32_t result_type_id           = 5000;
    constexpr uint32_t result_id                = 5001;
    constexpr uint32_t base_id                  = 5002;
    constexpr uint32_t final_index_id           = 5003;
    constexpr uint32_t container_type_id        = 5004;
    constexpr uint32_t physical_pointer_type_id = 5005;
    constexpr uint32_t array_length_id           = 5006;
    constexpr uint32_t base_pointer_type_id      = 5007;
    constexpr uint32_t uvec2_type_id             = 5008;
    constexpr uint32_t fvec2_type_id             = 5009;

    const FinalDynamicIndexCase& test_case = GetParam();

    uint32_t result_pointee_type_id = CommonTypes::u64;
    switch (test_case.result_kind)
    {
        case FinalIndexResultKind::U32:
            result_pointee_type_id = CommonTypes::u32;
            break;
        case FinalIndexResultKind::U64:
            result_pointee_type_id = CommonTypes::u64;
            break;
        case FinalIndexResultKind::F32:
            result_pointee_type_id = CommonTypes::f32;
            break;
        case FinalIndexResultKind::PhysicalPointer:
            result_pointee_type_id = physical_pointer_type_id;
            break;
        case FinalIndexResultKind::UVec2:
            result_pointee_type_id = uvec2_type_id;
            break;
        case FinalIndexResultKind::FVec2:
            result_pointee_type_id = fvec2_type_id;
            break;
    }

    const ::SPIRVSimulator::Type physical_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    if (test_case.result_kind == FinalIndexResultKind::PhysicalPointer)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(physical_pointer_type_id))
            .WillRepeatedly(ReturnRef(physical_pointer_type));
    }

    const ::SPIRVSimulator::Type uvec2_type =
        ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 2);
    if (test_case.result_kind == FinalIndexResultKind::UVec2)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(uvec2_type_id))
            .WillRepeatedly(ReturnRef(uvec2_type));
        EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u32)).WillOnce(Return(32));
    }

    const ::SPIRVSimulator::Type fvec2_type =
        ::SPIRVSimulator::Type::Vector(CommonTypes::f32, 2);
    if (test_case.result_kind == FinalIndexResultKind::FVec2)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(fvec2_type_id))
            .WillRepeatedly(ReturnRef(fvec2_type));
        EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::f32)).WillOnce(Return(32));
    }

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, result_pointee_type_id);

    ::SPIRVSimulator::Type container_type;
    switch (test_case.container_kind)
    {
        case FinalIndexContainerKind::RuntimeArray:
            container_type = ::SPIRVSimulator::Type::RuntimeArray(result_pointee_type_id);
            break;
        case FinalIndexContainerKind::FixedArray:
            container_type =
                ::SPIRVSimulator::Type::Array(result_pointee_type_id, array_length_id);
            break;
        case FinalIndexContainerKind::Vector:
            container_type = ::SPIRVSimulator::Type::Vector(result_pointee_type_id, 4);
            break;
    }

    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    if (test_case.container_kind == FinalIndexContainerKind::FixedArray)
    {
        EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(4));
    }
    EXPECT_CALL(*this, GetBitsizeOfType(result_pointee_type_id))
        .WillOnce(Return(test_case.element_size * 8));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .WillRepeatedly([this, &test_case](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{
                backing_.data(), pointer.idx_path.back() * test_case.byte_step
            };
        });

    SetDynamicIndexRange(final_index_id,
                       test_case.index_min,
                       test_case.index_max,
                       test_case.index_stride);
    SetBaseInvariant(base_id);
    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };
    if (test_case.container_kind == FinalIndexContainerKind::RuntimeArray)
    {
        SetRuntimeArrayLength(result_pointer, 0, test_case.index_max + 1);
    }

    EXPECT_TRUE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));

    const uint64_t base_address = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_TRUE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min,
              base_address + test_case.index_min * test_case.byte_step);
    EXPECT_EQ(metadata.address_range_max,
              base_address + test_case.index_max * test_case.byte_step);
    EXPECT_EQ(metadata.address_range_stride, test_case.element_size);
    EXPECT_EQ(metadata.address_element_size, test_case.element_size);
    ExpectPointerDependence(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    SupportedFinalIndexCases,
    FinalDynamicIndexAddressRangeTests,
    Values(FinalDynamicIndexCase{ "U64RuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::U64,
                                  0, 2, 1, 8, 8 },
           FinalDynamicIndexCase{ "U32FixedArrayInBounds",
                                  FinalIndexContainerKind::FixedArray,
                                  FinalIndexResultKind::U32,
                                  1, 3, 1, 4, 4 },
           FinalDynamicIndexCase{ "PhysicalPointerRuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::PhysicalPointer,
                                  0, 2, 1, 8, 8 },
           FinalDynamicIndexCase{ "UVec2RuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::UVec2,
                                  0, 2, 1, 8, 8 },
           FinalDynamicIndexCase{ "F32RuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::F32,
                                  0, 2, 1, 4, 4 },
           FinalDynamicIndexCase{ "FVec2RuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::FVec2,
                                  0, 2, 1, 8, 8 },
           FinalDynamicIndexCase{ "U64VectorInBounds",
                                  FinalIndexContainerKind::Vector,
                                  FinalIndexResultKind::U64,
                                  1, 3, 1, 8, 8 },
           FinalDynamicIndexCase{ "U64RuntimeArraySingleElementTightlyPacked",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::U64,
                                  0, 0, 1, 8, 8 },
           FinalDynamicIndexCase{ "U64RuntimeArraySingleElementPadded",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::U64,
                                  1, 1, 1, 16, 8 },
           FinalDynamicIndexCase{ "PhysicalPointerRuntimeArraySingleElementPadded",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::PhysicalPointer,
                                  2, 2, 1, 16, 8 },
           FinalDynamicIndexCase{ "U64VectorSingleComponent",
                                  FinalIndexContainerKind::Vector,
                                  FinalIndexResultKind::U64,
                                  2, 2, 1, 8, 8 }),
    [](const TestParamInfo<FinalDynamicIndexCase>& info) { return std::string(info.param.name); });

TEST_F(AccessChainAddressRangeTests, RejectsDynamicIndexOutsideFixedArrayBounds)
{
    constexpr uint32_t result_type_id       = 5010;
    constexpr uint32_t result_id            = 5011;
    constexpr uint32_t base_id              = 5012;
    constexpr uint32_t dynamic_index_id     = 5013;
    constexpr uint32_t container_type_id    = 5014;
    constexpr uint32_t array_length_id      = 5015;
    constexpr uint32_t base_pointer_type_id = 5016;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::Array(CommonTypes::u64, array_length_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(3));
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 3);
    SetAddressRange(result_id, 0x1000, 0x1010, 8, 8);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));

    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

TEST_F(AccessChainAddressRangeTests, RejectsLastElementThatExtendsPastPhysicalBacking)
{
    constexpr uint32_t result_type_id       = 5035;
    constexpr uint32_t result_id            = 5036;
    constexpr uint32_t base_id              = 5037;
    constexpr uint32_t dynamic_index_id     = 5038;
    constexpr uint32_t container_type_id    = 5039;
    constexpr uint32_t array_length_id      = 5040;
    constexpr uint32_t base_pointer_type_id = 5041;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::Array(CommonTypes::u64, array_length_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(3));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .Times(3)
        .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(),
                                                    pointer.idx_path.back() * 8 };
        });

    simulation_data_storage_.physical_address_buffers.clear();
    simulation_data_storage_.physical_address_buffers.emplace(
        0x1000, std::make_pair(20, static_cast<void*>(backing_.data())));

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);
    SetAddressRange(result_id, 0x1000, 0x1010, 8, 8);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));

    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

struct NonFinalDynamicIndexCase
{
    const char* name;
    uint32_t    field_index;
    uint64_t    index_min;
    uint64_t    index_max;
};

class NonFinalDynamicIndexAddressRangeTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<NonFinalDynamicIndexCase>
{};

TEST_P(NonFinalDynamicIndexAddressRangeTests, ProducesStridedWholeVectorFieldRange)
{
    constexpr uint32_t result_type_id           = 5020;
    constexpr uint32_t result_id                = 5021;
    constexpr uint32_t base_id                  = 5022;
    constexpr uint32_t block_member_index_id    = 5023;
    constexpr uint32_t dynamic_index_id         = 5024;
    constexpr uint32_t field_index_id           = 5025;
    constexpr uint32_t base_pointer_type_id     = 5026;
    constexpr uint32_t block_type_id            = 5027;
    constexpr uint32_t runtime_array_type_id    = 5028;
    constexpr uint32_t interleave_struct_type_id = 5029;
    constexpr uint32_t uvec2_type_id            = 5030;

    const NonFinalDynamicIndexCase& test_case = GetParam();
    const uint32_t field_index = test_case.field_index;

    const ::SPIRVSimulator::Type fixed_index_type =
        ::SPIRVSimulator::Type::Int(32, false);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, block_type_id);
    const ::SPIRVSimulator::Type block_type =
        ::SPIRVSimulator::Type::Struct(block_type_id);
    const ::SPIRVSimulator::Type runtime_array_type =
        ::SPIRVSimulator::Type::RuntimeArray(interleave_struct_type_id);
    const ::SPIRVSimulator::Type interleave_struct_type =
        ::SPIRVSimulator::Type::Struct(interleave_struct_type_id);
    const ::SPIRVSimulator::Type uvec2_type =
        ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 2);
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, uvec2_type_id);

    struct_members_[block_type_id] = { runtime_array_type_id };
    struct_members_[interleave_struct_type_id] = { uvec2_type_id, uvec2_type_id };
    decorators_[block_type_id].push_back(
        { spv::Decoration::DecorationBlock, {} });
    struct_decorators_[block_type_id][0].push_back(
        { spv::Decoration::DecorationOffset, { 0 } });

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(block_type_id)).WillRepeatedly(ReturnRef(block_type));
    EXPECT_CALL(*this, GetTypeByTypeId(runtime_array_type_id))
        .WillRepeatedly(ReturnRef(runtime_array_type));
    EXPECT_CALL(*this, GetTypeByTypeId(interleave_struct_type_id))
        .WillRepeatedly(ReturnRef(interleave_struct_type));
    EXPECT_CALL(*this, GetTypeByTypeId(uvec2_type_id)).WillRepeatedly(ReturnRef(uvec2_type));
    EXPECT_CALL(*this, GetTypeByResultId(block_member_index_id))
        .WillOnce(ReturnRef(fixed_index_type));
    EXPECT_CALL(*this, GetTypeByResultId(field_index_id))
        .WillOnce(ReturnRef(fixed_index_type));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u32)).WillOnce(Return(32));
    EXPECT_CALL(*this, GetBitsizeOfType(uvec2_type_id)).WillOnce(Return(64));
    EXPECT_CALL(*this, GetTargetPointerType(_)).WillOnce(Return(block_type_id));

    ::SPIRVSimulator::Value block_member_index_value = uint64_t{ 0 };
    ::SPIRVSimulator::Value field_index_value = static_cast<uint64_t>(field_index);
    EXPECT_CALL(*this, GetValue(block_member_index_id))
        .WillOnce(ReturnRef(block_member_index_value));
    EXPECT_CALL(*this, GetValue(field_index_id)).WillOnce(ReturnRef(field_index_value));

    EXPECT_CALL(*this, ResolvePointerV(_))
        .Times(test_case.index_min == test_case.index_max ? 2 : 3)
        .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{
                backing_.data(), pointer.idx_path[1] * 16 + pointer.idx_path[2] * 8
            };
        });

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, test_case.index_min, test_case.index_max);
    value_meta_.resize(field_index_id + 1);
    value_meta_[block_member_index_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Invariant;
    value_meta_[field_index_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Invariant;

    std::vector<uint32_t> words = { 0,
                                    result_type_id,
                                    result_id,
                                    base_id,
                                    block_member_index_id,
                                    dynamic_index_id,
                                    field_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0, 0, field_index },
    };

    ASSERT_TRUE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));

    const uint64_t base_address = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_TRUE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min,
              base_address + test_case.index_min * 16 + field_index * 8);
    EXPECT_EQ(metadata.address_range_max,
              base_address + test_case.index_max * 16 + field_index * 8);
    EXPECT_EQ(metadata.address_range_stride,
              test_case.index_min == test_case.index_max ? 8u : 16u);
    EXPECT_EQ(metadata.address_element_size, 8u);
    ExpectPointerDependence(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InterleaveFields,
    NonFinalDynamicIndexAddressRangeTests,
    Values(NonFinalDynamicIndexCase{ "ColorMultiElement", 0, 0, 2 },
           NonFinalDynamicIndexCase{ "AddressMultiElement", 1, 0, 2 },
           NonFinalDynamicIndexCase{ "AddressSingleElement", 1, 1, 1 }),
    [](const TestParamInfo<NonFinalDynamicIndexCase>& info) {
        return std::string(info.param.name);
    });

enum class AffineAddressProofFailure
{
    MinResolveNull,
    MaxResolveNull,
    MinMaxDifferentBackingBases,
    MaxAddressBeforeMin,
    NextResolveNull,
    NextResolveDifferentBackingBase,
    NextAddressNotAfterMin,
    ByteStrideSmallerThanElement,
    NonAffineResolvedMaximum,
    MinAddressAddOverflow,
    MaxAddressAddOverflow,
    NextAddressAddOverflow,
    AddressSpanMultiplyOverflow,
    ExpectedMaxAddressAddOverflow,
};

struct AffineAddressProofFailureCase
{
    const char*               name;
    AffineAddressProofFailure failure;
};

class AffineAddressProofFailureTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<AffineAddressProofFailureCase>
{};

TEST_P(AffineAddressProofFailureTests, RejectsInvalidResolvedSequence)
{
    constexpr uint32_t result_type_id       = 5580;
    constexpr uint32_t result_id            = 5581;
    constexpr uint32_t base_id              = 5582;
    constexpr uint32_t dynamic_index_id     = 5583;
    constexpr uint32_t base_pointer_type_id = 5584;
    constexpr uint32_t container_type_id    = 5585;
    constexpr uint32_t array_length_id      = 5586;

    using ResolvedPointer = std::pair<std::byte*, uint64_t>;

    const AffineAddressProofFailureCase& test_case = GetParam();
    const uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
    std::byte* const high_base = ::SPIRVSimulator::bit_cast<std::byte*>(max_u64 - 32);
    std::byte* const very_high_base =
        ::SPIRVSimulator::bit_cast<std::byte*>(max_u64 - 100);
    std::byte* const low_base = ::SPIRVSimulator::bit_cast<std::byte*>(uint64_t{ 1 });

    ResolvedPointer min_resolved{ backing_.data(), 0 };
    ResolvedPointer max_resolved{ backing_.data(), 16 };
    ResolvedPointer next_resolved{ backing_.data(), 8 };
    bool            expect_next = true;

    switch (test_case.failure)
    {
        case AffineAddressProofFailure::MinResolveNull:
            min_resolved = { nullptr, 0 };
            expect_next = false;
            break;
        case AffineAddressProofFailure::MaxResolveNull:
            max_resolved = { nullptr, 0 };
            expect_next = false;
            break;
        case AffineAddressProofFailure::MinMaxDifferentBackingBases:
            max_resolved = { backing_.data() + 1, 16 };
            expect_next = false;
            break;
        case AffineAddressProofFailure::MaxAddressBeforeMin:
            min_resolved = { backing_.data(), 16 };
            max_resolved = { backing_.data(), 8 };
            expect_next = false;
            break;
        case AffineAddressProofFailure::NextResolveNull:
            next_resolved = { nullptr, 0 };
            break;
        case AffineAddressProofFailure::NextResolveDifferentBackingBase:
            next_resolved = { backing_.data() + 1, 8 };
            break;
        case AffineAddressProofFailure::NextAddressNotAfterMin:
            next_resolved = { backing_.data(), 0 };
            break;
        case AffineAddressProofFailure::ByteStrideSmallerThanElement:
            max_resolved  = { backing_.data(), 8 };
            next_resolved = { backing_.data(), 4 };
            break;
        case AffineAddressProofFailure::NonAffineResolvedMaximum:
            max_resolved = { backing_.data(), 24 };
            break;
        case AffineAddressProofFailure::MinAddressAddOverflow:
            min_resolved = { ::SPIRVSimulator::bit_cast<std::byte*>(max_u64 - 3), 4 };
            max_resolved = { min_resolved.first, 4 };
            expect_next = false;
            break;
        case AffineAddressProofFailure::MaxAddressAddOverflow:
            min_resolved = { high_base, 0 };
            max_resolved = { high_base, 64 };
            expect_next = false;
            break;
        case AffineAddressProofFailure::NextAddressAddOverflow:
            min_resolved  = { high_base, 0 };
            max_resolved  = { high_base, 16 };
            next_resolved = { high_base, 64 };
            break;
        case AffineAddressProofFailure::AddressSpanMultiplyOverflow:
            min_resolved  = { low_base, 0 };
            max_resolved  = { low_base, 1 };
            next_resolved = { low_base, uint64_t{ 1 } << 63 };
            break;
        case AffineAddressProofFailure::ExpectedMaxAddressAddOverflow:
            min_resolved  = { very_high_base, 0 };
            max_resolved  = { very_high_base, 10 };
            next_resolved = { very_high_base, 60 };
            break;
    }

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::Array(CommonTypes::u64, array_length_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id))
        .WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(3));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));

    {
        InSequence sequence;
        EXPECT_CALL(*this,
                    ResolvePointerV(Truly([](const ::SPIRVSimulator::PointerV& pointer) {
                        return pointer.idx_path == std::vector<uint32_t>{ 0 };
                    })))
            .WillOnce(Return(min_resolved));
        EXPECT_CALL(*this,
                    ResolvePointerV(Truly([](const ::SPIRVSimulator::PointerV& pointer) {
                        return pointer.idx_path == std::vector<uint32_t>{ 2 };
                    })))
            .WillOnce(Return(max_resolved));
        if (expect_next)
        {
            EXPECT_CALL(*this,
                        ResolvePointerV(Truly([](const ::SPIRVSimulator::PointerV& pointer) {
                            return pointer.idx_path == std::vector<uint32_t>{ 1 };
                        })))
                .WillOnce(Return(next_resolved));
        }
    }

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);
    if (test_case.failure == AffineAddressProofFailure::NonAffineResolvedMaximum)
    {
        SetAddressRange(result_id, 0x1000, 0x1010, 8, 8);
    }

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidAffineAddressSequences,
    AffineAddressProofFailureTests,
    Values(AffineAddressProofFailureCase{
               "MinResolveNull", AffineAddressProofFailure::MinResolveNull },
           AffineAddressProofFailureCase{
               "MaxResolveNull", AffineAddressProofFailure::MaxResolveNull },
           AffineAddressProofFailureCase{
               "MinMaxDifferentBackingBases",
               AffineAddressProofFailure::MinMaxDifferentBackingBases },
           AffineAddressProofFailureCase{
               "MaxAddressBeforeMin", AffineAddressProofFailure::MaxAddressBeforeMin },
           AffineAddressProofFailureCase{
               "NextResolveNull", AffineAddressProofFailure::NextResolveNull },
           AffineAddressProofFailureCase{
               "NextResolveDifferentBackingBase",
               AffineAddressProofFailure::NextResolveDifferentBackingBase },
           AffineAddressProofFailureCase{
               "NextAddressNotAfterMin",
               AffineAddressProofFailure::NextAddressNotAfterMin },
           AffineAddressProofFailureCase{
               "ByteStrideSmallerThanElement",
               AffineAddressProofFailure::ByteStrideSmallerThanElement },
           AffineAddressProofFailureCase{
               "NonAffineResolvedMaximum",
               AffineAddressProofFailure::NonAffineResolvedMaximum },
           AffineAddressProofFailureCase{
               "MinAddressAddOverflow", AffineAddressProofFailure::MinAddressAddOverflow },
           AffineAddressProofFailureCase{
               "MaxAddressAddOverflow", AffineAddressProofFailure::MaxAddressAddOverflow },
           AffineAddressProofFailureCase{
               "NextAddressAddOverflow", AffineAddressProofFailure::NextAddressAddOverflow },
           AffineAddressProofFailureCase{
               "AddressSpanMultiplyOverflow",
               AffineAddressProofFailure::AddressSpanMultiplyOverflow },
           AffineAddressProofFailureCase{
               "ExpectedMaxAddressAddOverflow",
               AffineAddressProofFailure::ExpectedMaxAddressAddOverflow }),
    [](const TestParamInfo<AffineAddressProofFailureCase>& info) {
        return std::string(info.param.name);
    });

enum class PhysicalBackingProofFailure
{
    SimulationDataMissing,
    NoMatchingAllocation,
    NullAllocationData,
    ZeroAllocationSize,
    MinStartOutsideAllocation,
    MaxStartOutsideAllocation,
    AmbiguousMatchingAllocations,
};

struct PhysicalBackingProofFailureCase
{
    const char*                 name;
    PhysicalBackingProofFailure failure;
};

class PhysicalBackingProofFailureTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<PhysicalBackingProofFailureCase>
{};

TEST_P(PhysicalBackingProofFailureTests, RejectsInvalidPhysicalBacking)
{
    constexpr uint32_t result_type_id       = 5600;
    constexpr uint32_t result_id            = 5601;
    constexpr uint32_t base_id              = 5602;
    constexpr uint32_t dynamic_index_id     = 5603;
    constexpr uint32_t base_pointer_type_id = 5604;
    constexpr uint32_t container_type_id    = 5605;
    constexpr uint32_t array_length_id      = 5606;

    const PhysicalBackingProofFailureCase& test_case = GetParam();
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::Array(CommonTypes::u64, array_length_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id))
        .WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetArrayLength(array_length_id)).WillOnce(Return(3));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .Times(3)
        .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(),
                                                    pointer.idx_path.back() * 8 };
        });

    simulation_data_storage_.physical_address_buffers.clear();
    switch (test_case.failure)
    {
        case PhysicalBackingProofFailure::SimulationDataMissing:
            simulation_data_ = nullptr;
            break;
        case PhysicalBackingProofFailure::NoMatchingAllocation:
            simulation_data_storage_.physical_address_buffers.emplace(
                0x1000,
                std::make_pair(backing_.size() - 32,
                               static_cast<void*>(backing_.data() + 32)));
            break;
        case PhysicalBackingProofFailure::NullAllocationData:
            simulation_data_storage_.physical_address_buffers.emplace(
                0x1000, std::make_pair(backing_.size(), nullptr));
            break;
        case PhysicalBackingProofFailure::ZeroAllocationSize:
            simulation_data_storage_.physical_address_buffers.emplace(
                0x1000, std::make_pair(size_t{ 0 }, static_cast<void*>(backing_.data())));
            break;
        case PhysicalBackingProofFailure::MinStartOutsideAllocation:
            simulation_data_storage_.physical_address_buffers.emplace(
                0x1000,
                std::make_pair(backing_.size() - 8,
                               static_cast<void*>(backing_.data() + 8)));
            break;
        case PhysicalBackingProofFailure::MaxStartOutsideAllocation:
            simulation_data_storage_.physical_address_buffers.emplace(
                0x1000, std::make_pair(size_t{ 16 }, static_cast<void*>(backing_.data())));
            break;
        case PhysicalBackingProofFailure::AmbiguousMatchingAllocations:
            simulation_data_storage_.physical_address_buffers.emplace(
                0x1000,
                std::make_pair(backing_.size(), static_cast<void*>(backing_.data())));
            simulation_data_storage_.physical_address_buffers.emplace(
                0x2000,
                std::make_pair(backing_.size(), static_cast<void*>(backing_.data())));
            break;
    }

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, dynamic_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidPhysicalBackings,
    PhysicalBackingProofFailureTests,
    Values(PhysicalBackingProofFailureCase{
               "SimulationDataMissing", PhysicalBackingProofFailure::SimulationDataMissing },
           PhysicalBackingProofFailureCase{
               "NoMatchingAllocation", PhysicalBackingProofFailure::NoMatchingAllocation },
           PhysicalBackingProofFailureCase{
               "NullAllocationData", PhysicalBackingProofFailure::NullAllocationData },
           PhysicalBackingProofFailureCase{
               "ZeroAllocationSize", PhysicalBackingProofFailure::ZeroAllocationSize },
           PhysicalBackingProofFailureCase{
               "MinStartOutsideAllocation",
               PhysicalBackingProofFailure::MinStartOutsideAllocation },
           PhysicalBackingProofFailureCase{
               "MaxStartOutsideAllocation",
               PhysicalBackingProofFailure::MaxStartOutsideAllocation },
           PhysicalBackingProofFailureCase{
               "AmbiguousMatchingAllocations",
               PhysicalBackingProofFailure::AmbiguousMatchingAllocations }),
    [](const TestParamInfo<PhysicalBackingProofFailureCase>& info) {
        return std::string(info.param.name);
    });

enum class DeferredRuntimeArrayShapeFailure
{
    EmptyDynamicContainerPath,
    ParentNotStruct,
    ParentMissingBlockDecoration,
    RuntimeArrayNotLastMember,
    ParentMemberTypeMismatch,
};

struct DeferredRuntimeArrayShapeFailureCase
{
    const char*                      name;
    DeferredRuntimeArrayShapeFailure failure;
};

class DeferredRuntimeArrayShapeFailureTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<DeferredRuntimeArrayShapeFailureCase>
{};

TEST_P(DeferredRuntimeArrayShapeFailureTests, RejectsInvalidDeferredShape)
{
    constexpr uint32_t result_type_id          = 5620;
    constexpr uint32_t result_id               = 5621;
    constexpr uint32_t base_id                 = 5622;
    constexpr uint32_t fixed_member_index_id   = 5623;
    constexpr uint32_t dynamic_index_id        = 5624;
    constexpr uint32_t base_pointer_type_id    = 5625;
    constexpr uint32_t root_struct_type_id     = 5626;
    constexpr uint32_t runtime_array_type_id   = 5627;
    constexpr uint32_t target_parent_type_id   = 5628;
    constexpr uint32_t other_runtime_type_id   = 5629;
    constexpr uint32_t target_array_length_id  = 5630;

    const DeferredRuntimeArrayShapeFailureCase& test_case = GetParam();
    const bool empty_container_path =
        test_case.failure == DeferredRuntimeArrayShapeFailure::EmptyDynamicContainerPath;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type root_struct_type =
        ::SPIRVSimulator::Type::Struct(root_struct_type_id);
    const ::SPIRVSimulator::Type runtime_array_type =
        ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);

    uint32_t resolved_parent_type_id = root_struct_type_id;
    ::SPIRVSimulator::Type resolved_parent_type;
    bool resolved_parent_has_separate_type = false;

    if (!empty_container_path)
    {
        struct_members_[root_struct_type_id] = { runtime_array_type_id };
        struct_decorators_[root_struct_type_id][0].push_back(
            { spv::Decoration::DecorationOffset, { 0 } });
    }

    switch (test_case.failure)
    {
        case DeferredRuntimeArrayShapeFailure::EmptyDynamicContainerPath:
            break;
        case DeferredRuntimeArrayShapeFailure::ParentNotStruct:
            resolved_parent_type_id = target_parent_type_id;
            resolved_parent_type =
                ::SPIRVSimulator::Type::Array(CommonTypes::u64, target_array_length_id);
            resolved_parent_has_separate_type = true;
            break;
        case DeferredRuntimeArrayShapeFailure::ParentMissingBlockDecoration:
            break;
        case DeferredRuntimeArrayShapeFailure::RuntimeArrayNotLastMember:
            struct_members_[root_struct_type_id] = { runtime_array_type_id, CommonTypes::u64 };
            decorators_[root_struct_type_id].push_back(
                { spv::Decoration::DecorationBlock, {} });
            break;
        case DeferredRuntimeArrayShapeFailure::ParentMemberTypeMismatch:
            resolved_parent_type_id = target_parent_type_id;
            resolved_parent_type =
                ::SPIRVSimulator::Type::Struct(target_parent_type_id);
            resolved_parent_has_separate_type = true;
            struct_members_[target_parent_type_id] = { other_runtime_type_id };
            decorators_[target_parent_type_id].push_back(
                { spv::Decoration::DecorationBlock, {} });
            break;
    }

    const uint32_t base_pointee_type_id =
        empty_container_path ? runtime_array_type_id : root_struct_type_id;
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, base_pointee_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(runtime_array_type_id))
        .WillRepeatedly(ReturnRef(runtime_array_type));
    if (!empty_container_path)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(root_struct_type_id))
            .WillRepeatedly(ReturnRef(root_struct_type));
    }
    if (resolved_parent_has_separate_type)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(resolved_parent_type_id))
            .WillRepeatedly(ReturnRef(resolved_parent_type));
    }
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);
    if (empty_container_path)
    {
        EXPECT_CALL(*this, GetTargetPointerType(_)).Times(0);
    }
    else
    {
        EXPECT_CALL(*this, GetTargetPointerType(_))
            .WillOnce(Return(resolved_parent_type_id));
    }

    SetBaseInvariant(base_id);
    SetDynamicIndexRange(dynamic_index_id, 0, 2);
    if (!empty_container_path)
    {
        SetFixedUnsignedIndex(fixed_member_index_id, 0);
    }

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id };
    if (!empty_container_path)
    {
        words.push_back(fixed_member_index_id);
    }
    words.push_back(dynamic_index_id);
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };

    std::vector<uint32_t> result_path;
    if (!empty_container_path)
    {
        result_path.push_back(0);
    }
    result_path.push_back(0);
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = result_path,
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidDeferredRuntimeArrayShapes,
    DeferredRuntimeArrayShapeFailureTests,
    Values(DeferredRuntimeArrayShapeFailureCase{
               "EmptyDynamicContainerPath",
               DeferredRuntimeArrayShapeFailure::EmptyDynamicContainerPath },
           DeferredRuntimeArrayShapeFailureCase{
               "ParentNotStruct", DeferredRuntimeArrayShapeFailure::ParentNotStruct },
           DeferredRuntimeArrayShapeFailureCase{
               "ParentMissingBlockDecoration",
               DeferredRuntimeArrayShapeFailure::ParentMissingBlockDecoration },
           DeferredRuntimeArrayShapeFailureCase{
               "RuntimeArrayNotLastMember",
               DeferredRuntimeArrayShapeFailure::RuntimeArrayNotLastMember },
           DeferredRuntimeArrayShapeFailureCase{
               "ParentMemberTypeMismatch",
               DeferredRuntimeArrayShapeFailure::ParentMemberTypeMismatch }),
    [](const TestParamInfo<DeferredRuntimeArrayShapeFailureCase>& info) {
        return std::string(info.param.name);
    });

enum class UnsupportedAccessChainCapability
{
    MultipleDynamicIndexes,
    BaseHasAddressRange,
    BaseDependenceVarying,
    UniformDescriptorArray,
    UniformFixedDescriptorArray,
    UniformConstantDescriptorArray,
    StorageBufferDescriptorArray,
};

struct UnsupportedAccessChainCapabilityCase
{
    const char*                            name;
    UnsupportedAccessChainCapability       capability;
};

// Verifies that unsupported dynamic-index placement, pre-existing base
// contributions, and descriptor-array indirection never publish a partial range.
class UnsupportedAccessChainAddressRangeCapabilityTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<UnsupportedAccessChainCapabilityCase>
{};

TEST_P(UnsupportedAccessChainAddressRangeCapabilityTests, RejectsAddressRange)
{
    constexpr uint32_t result_type_id           = 5070;
    constexpr uint32_t result_id                = 5071;
    constexpr uint32_t base_id                  = 5072;
    constexpr uint32_t first_index_id           = 5073;
    constexpr uint32_t final_index_id           = 5074;
    constexpr uint32_t container_type_id        = 5075;
    constexpr uint32_t base_pointer_type_id     = 5076;
    constexpr uint32_t container_length_id      = 5077;

    const UnsupportedAccessChainCapabilityCase& test_case = GetParam();

    uint32_t storage_class = spv::StorageClassPhysicalStorageBuffer;
    std::vector<uint32_t> index_ids;
    SetBaseInvariant(base_id);
    switch (test_case.capability)
    {
        case UnsupportedAccessChainCapability::MultipleDynamicIndexes:
            SetDynamicIndexRange(first_index_id, 0, 2);
            SetDynamicIndexRange(final_index_id, 0, 2);
            index_ids = { first_index_id, final_index_id };
            break;
        case UnsupportedAccessChainCapability::BaseHasAddressRange:
            SetDynamicIndexRange(final_index_id, 0, 2);
            SetAddressRange(base_id, 0x1000, 0x1010, 8, 8);
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::BaseDependenceVarying:
            SetDynamicIndexRange(final_index_id, 0, 2);
            value_meta_[base_id].invocation_dependence =
                ::SPIRVSimulator::InvocationDependence::Varying;
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::UniformDescriptorArray:
            storage_class = spv::StorageClassUniform;
            SetDynamicIndexRange(final_index_id, 0, 2);
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::UniformFixedDescriptorArray:
            storage_class = spv::StorageClassUniform;
            SetDynamicIndexRange(final_index_id, 0, 2);
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::UniformConstantDescriptorArray:
            storage_class = spv::StorageClassUniformConstant;
            SetDynamicIndexRange(final_index_id, 0, 2);
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::StorageBufferDescriptorArray:
            storage_class = spv::StorageClassStorageBuffer;
            SetDynamicIndexRange(final_index_id, 0, 2);
            index_ids = { final_index_id };
            break;
    }

    const bool is_fixed_descriptor_array =
        test_case.capability == UnsupportedAccessChainCapability::UniformFixedDescriptorArray;
    const bool is_descriptor_array =
        test_case.capability == UnsupportedAccessChainCapability::UniformDescriptorArray ||
        is_fixed_descriptor_array ||
        test_case.capability == UnsupportedAccessChainCapability::UniformConstantDescriptorArray ||
        test_case.capability == UnsupportedAccessChainCapability::StorageBufferDescriptorArray;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Pointer(storage_class, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        is_fixed_descriptor_array
            ? ::SPIRVSimulator::Type::Array(CommonTypes::u64, container_length_id)
            : ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);
    const ::SPIRVSimulator::Type base_pointer_type =
        ::SPIRVSimulator::Type::Pointer(storage_class, container_type_id);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));

    if (is_descriptor_array)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
            .WillRepeatedly(ReturnRef(base_pointer_type));
        EXPECT_CALL(*this, GetTypeByTypeId(container_type_id))
            .WillRepeatedly(ReturnRef(container_type));
    }
    if (is_fixed_descriptor_array)
    {
        EXPECT_CALL(*this, GetArrayLength(container_length_id)).Times(0);
    }
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id };
    words.insert(words.end(), index_ids.begin(), index_ids.end());
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .base_type_id = base_pointer_type_id,
        .storage_class = storage_class,
        .idx_path = std::vector<uint32_t>(index_ids.size(), 0),
    };

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedAccessChainCases,
    UnsupportedAccessChainAddressRangeCapabilityTests,
    Values(UnsupportedAccessChainCapabilityCase{
               "MultipleDynamicIndexes",
               UnsupportedAccessChainCapability::MultipleDynamicIndexes },
           UnsupportedAccessChainCapabilityCase{
               "BaseHasAddressRange",
               UnsupportedAccessChainCapability::BaseHasAddressRange },
           UnsupportedAccessChainCapabilityCase{
               "BaseDependenceVarying",
               UnsupportedAccessChainCapability::BaseDependenceVarying },
           UnsupportedAccessChainCapabilityCase{
               "UniformDescriptorArray",
               UnsupportedAccessChainCapability::UniformDescriptorArray },
           UnsupportedAccessChainCapabilityCase{
               "UniformFixedDescriptorArray",
               UnsupportedAccessChainCapability::UniformFixedDescriptorArray },
           UnsupportedAccessChainCapabilityCase{
               "UniformConstantDescriptorArray",
               UnsupportedAccessChainCapability::UniformConstantDescriptorArray },
           UnsupportedAccessChainCapabilityCase{
               "StorageBufferDescriptorArray",
               UnsupportedAccessChainCapability::StorageBufferDescriptorArray }),
    [](const TestParamInfo<UnsupportedAccessChainCapabilityCase>& info) {
        return std::string(info.param.name);
    });

enum class UnsupportedResultPointeeKind
{
    Bool,
    BoolVector,
    Struct,
    Matrix,
    FixedArray,
    RuntimeArray,
};

struct UnsupportedResultPointeeCase
{
    const char*                      name;
    UnsupportedResultPointeeKind     kind;
};

// Verifies the current result-pointee capability boundary while keeping the
// final dynamic index, container, storage class, and dense input range valid.
class UnsupportedResultPointeeAddressRangeTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<UnsupportedResultPointeeCase>
{};

TEST_P(UnsupportedResultPointeeAddressRangeTests, RejectsAddressRange)
{
    constexpr uint32_t result_type_id           = 5100;
    constexpr uint32_t result_id                = 5101;
    constexpr uint32_t base_id                  = 5102;
    constexpr uint32_t final_index_id           = 5103;
    constexpr uint32_t container_type_id        = 5104;
    constexpr uint32_t result_pointee_type_id   = 5105;
    constexpr uint32_t array_length_id           = 5106;
    constexpr uint32_t base_pointer_type_id      = 5107;

    const UnsupportedResultPointeeCase& test_case = GetParam();

    ::SPIRVSimulator::Type result_pointee_type;
    switch (test_case.kind)
    {
        case UnsupportedResultPointeeKind::Bool:
            result_pointee_type = ::SPIRVSimulator::Type::BoolT();
            break;
        case UnsupportedResultPointeeKind::BoolVector:
            result_pointee_type =
                ::SPIRVSimulator::Type::Vector(CommonTypes::boolean, 2);
            break;
        case UnsupportedResultPointeeKind::Struct:
            result_pointee_type = ::SPIRVSimulator::Type::Struct(result_pointee_type_id);
            break;
        case UnsupportedResultPointeeKind::Matrix:
            result_pointee_type = ::SPIRVSimulator::Type::Matrix(CommonTypes::uvec2, 2);
            break;
        case UnsupportedResultPointeeKind::FixedArray:
            result_pointee_type =
                ::SPIRVSimulator::Type::Array(CommonTypes::u32, array_length_id);
            break;
        case UnsupportedResultPointeeKind::RuntimeArray:
            result_pointee_type =
                ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u32);
            break;
    }

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, result_pointee_type_id);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::RuntimeArray(result_pointee_type_id);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetTypeByTypeId(result_pointee_type_id))
        .WillRepeatedly(ReturnRef(result_pointee_type));
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetDynamicIndexRange(final_index_id, 0, 2);
    SetBaseInvariant(base_id);
    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };
    SetRuntimeArrayLength(result_pointer, 0, 3);

    EXPECT_FALSE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    ExpectResetAddressRangeMetadata(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedResultPointeeCases,
    UnsupportedResultPointeeAddressRangeTests,
    Values(UnsupportedResultPointeeCase{ "BoolResult", UnsupportedResultPointeeKind::Bool },
           UnsupportedResultPointeeCase{
               "BoolComponentVectorResult", UnsupportedResultPointeeKind::BoolVector },
           UnsupportedResultPointeeCase{ "StructResult", UnsupportedResultPointeeKind::Struct },
           UnsupportedResultPointeeCase{ "MatrixResult", UnsupportedResultPointeeKind::Matrix },
           UnsupportedResultPointeeCase{ "FixedArrayResult",
                                         UnsupportedResultPointeeKind::FixedArray },
           UnsupportedResultPointeeCase{
               "RuntimeArrayResult", UnsupportedResultPointeeKind::RuntimeArray }),
    [](const TestParamInfo<UnsupportedResultPointeeCase>& info) {
        return std::string(info.param.name);
    });

struct StridedFinalLayoutCase
{
    const char* name;
    uint64_t    index_min;
    uint64_t    index_max;
    uint64_t    index_stride;
    uint64_t    container_byte_step;
};

// Verifies that padded elements and sparse logical indexes publish the exact
// non-overlapping affine byte stride.
class StridedFinalLayoutTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<StridedFinalLayoutCase>
{};

TEST_P(StridedFinalLayoutTests, ProducesStridedAddressRange)
{
    constexpr uint32_t result_type_id    = 5300;
    constexpr uint32_t result_id         = 5301;
    constexpr uint32_t base_id           = 5302;
    constexpr uint32_t final_index_id    = 5303;
    constexpr uint32_t container_type_id = 5304;
    constexpr uint32_t base_pointer_type_id = 5305;

    const StridedFinalLayoutCase& test_case = GetParam();
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);
    const ::SPIRVSimulator::Type base_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, container_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(base_pointer_type_id))
        .WillRepeatedly(ReturnRef(base_pointer_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .Times(3)
        .WillRepeatedly([this, &test_case](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{
                backing_.data(), pointer.idx_path.back() * test_case.container_byte_step
            };
        });

    SetDynamicIndexRange(final_index_id,
                       test_case.index_min,
                       test_case.index_max,
                       test_case.index_stride);
    SetBaseInvariant(base_id);
    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .pointer_handle = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data()),
        .base_type_id = base_pointer_type_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };
    SetRuntimeArrayLength(result_pointer, 0, test_case.index_max + 1);

    EXPECT_TRUE(TrySetDenseAccessChainAddressRange(instruction, result_pointer));
    const uint64_t base_address = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_TRUE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min,
              base_address + test_case.index_min * test_case.container_byte_step);
    EXPECT_EQ(metadata.address_range_max,
              base_address + test_case.index_max * test_case.container_byte_step);
    EXPECT_EQ(metadata.address_range_stride,
              test_case.index_stride * test_case.container_byte_step);
    EXPECT_EQ(metadata.address_element_size, 8u);
    ExpectPointerDependence(
        result_id, ::SPIRVSimulator::InvocationDependence::Varying, true);
}

INSTANTIATE_TEST_SUITE_P(
    SupportedStridedFinalLayoutCases,
    StridedFinalLayoutTests,
    Values(StridedFinalLayoutCase{ "PaddedScalarElements", 0, 2, 1, 16 },
           StridedFinalLayoutCase{ "SparseSelectedIndexes", 0, 4, 2, 8 }),
    [](const TestParamInfo<StridedFinalLayoutCase>& info) {
        return std::string(info.param.name);
    });
