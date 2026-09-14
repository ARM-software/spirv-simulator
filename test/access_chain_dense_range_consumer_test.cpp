#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class DenseAddressRangeStoreConsumerTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD((std::pair<std::byte*, uint64_t>),
                ResolvePointerV,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(bool,
                WritePointer,
                (const ::SPIRVSimulator::PointerV& pointer, const ::SPIRVSimulator::Value& value),
                (override));
    MOCK_METHOD(void, OverrideFlagsPointee, (uint32_t pointer_id, uint32_t result_id), (override));
    MOCK_METHOD((std::vector<::SPIRVSimulator::DataSourceBits>),
                FindDataSourcesFromResultID,
                (uint32_t result_id, uint32_t* property_flags),
                (override));
    MOCK_METHOD(bool,
                PointeeValueIsDescriptorBuffer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(bool, ValueHoldsPbufferPtr, (uint32_t result_id), (const, override));
    MOCK_METHOD(bool, ValueIsArbitrary, (uint32_t result_id), (const, override));

  protected:
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

    bool TryGetStoreRange(const ::SPIRVSimulator::PointerV& pointer,
                          uint32_t pointer_id,
                          uint32_t result_id,
                          uint64_t& dst_start,
                          uint64_t& byte_size,
                          uint64_t& element_size)
    {
        return TryGetDenseStoreRange(
            pointer, pointer_id, result_id, dst_start, byte_size, element_size);
    }

    void Store(const ::SPIRVSimulator::Instruction& instruction)
    {
        Op_Store(instruction);
    }

    void SetRuntimeState(::SPIRVSimulator::MemoryFlagTracker* memory_flag_tracker,
                         ::SPIRVSimulator::SimulationResults* simulation_results)
    {
        memory_flag_tracker_ = memory_flag_tracker;
        simulation_results_  = simulation_results;
    }

    std::array<std::byte, 64> backing_{};
};

// Verifies that OpStore includes the final element when consuming valid dense
// address-range metadata and records the full span without requesting fallback.
TEST_F(DenseAddressRangeStoreConsumerTests, DenseAddressRangeIsConsumedByStore)
{
    constexpr uint32_t pointer_id = 5050;
    constexpr uint32_t result_id  = 5051;

    ::SPIRVSimulator::MemoryFlagTracker memory_flag_tracker;
    ::SPIRVSimulator::SimulationResults simulation_results;
    SetRuntimeState(&memory_flag_tracker, &simulation_results);

    const ::SPIRVSimulator::PointerV pointer{
        .base_result_id = pointer_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };
    ::SPIRVSimulator::Value pointer_value = pointer;
    ::SPIRVSimulator::Value stored_value  = uint64_t{ 0x1234 };

    const uint64_t range_start = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    SetAddressRange(pointer_id, range_start, range_start + 16, 8, 8);
    value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
    value_meta_[result_id].flags = SPS_FLAG_THREAD_SPECIFIC;

    EXPECT_CALL(*this, ResolvePointerV(_))
        .WillRepeatedly([this](const ::SPIRVSimulator::PointerV&) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(), 0 };
        });

    uint64_t dense_dst_start    = 0;
    uint64_t dense_byte_size    = 0;
    uint64_t dense_element_size = 0;
    ASSERT_TRUE(TryGetStoreRange(pointer,
                                 pointer_id,
                                 result_id,
                                 dense_dst_start,
                                 dense_byte_size,
                                 dense_element_size));
    EXPECT_EQ(dense_dst_start, range_start);
    EXPECT_EQ(dense_byte_size, 24u);
    EXPECT_EQ(dense_element_size, 8u);

    EXPECT_CALL(*this, GetValue(pointer_id)).WillRepeatedly(ReturnRef(pointer_value));
    EXPECT_CALL(*this, GetValue(result_id)).WillRepeatedly(ReturnRef(stored_value));
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(types_.at(CommonTypes::u64)));
    EXPECT_CALL(*this, PointeeValueIsDescriptorBuffer(_)).WillOnce(Return(false));
    EXPECT_CALL(*this, WritePointer(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*this, OverrideFlagsPointee(pointer_id, result_id)).Times(1);
    EXPECT_CALL(*this, FindDataSourcesFromResultID(result_id, _))
        .WillOnce([](uint32_t, uint32_t* property_flags) {
            *property_flags = 0;
            return std::vector<::SPIRVSimulator::DataSourceBits>{};
        });
    EXPECT_CALL(*this, ValueHoldsPbufferPtr(result_id)).WillOnce(Return(false));
    EXPECT_CALL(*this, ValueIsArbitrary(result_id)).WillOnce(Return(false));

    std::vector<uint32_t> words = { 0, pointer_id, result_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpStore,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    Store(instruction);

    const std::optional<::SPIRVSimulator::MemoryFlagTracker::UniformDerivedRange> tracked_range =
        memory_flag_tracker.queryUniformDerivedRange(range_start);
    ASSERT_TRUE(tracked_range.has_value());
    EXPECT_EQ(tracked_range->start, range_start);
    EXPECT_EQ(tracked_range->end, range_start + 24);
    EXPECT_EQ(tracked_range->element_size, 8u);
    EXPECT_EQ(tracked_range->flags,
              SPS_FLAG_THREAD_SPECIFIC | SPS_FLAG_UNIFORM_DERIVED_RANGE);
    EXPECT_FALSE(simulation_results.full_dispatch_needed);
}

TEST_F(DenseAddressRangeStoreConsumerTests, VaryingStoreWithoutLegacyFlagRequestsFullDispatch)
{
    constexpr uint32_t pointer_id = 5060;
    constexpr uint32_t result_id  = 5061;

    ::SPIRVSimulator::MemoryFlagTracker memory_flag_tracker;
    ::SPIRVSimulator::SimulationResults simulation_results;
    SetRuntimeState(&memory_flag_tracker, &simulation_results);

    const ::SPIRVSimulator::PointerV pointer{
        .base_result_id = pointer_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };
    ::SPIRVSimulator::Value pointer_value = pointer;
    ::SPIRVSimulator::Value stored_value  = uint64_t{ 0x1234 };

    value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
    value_meta_[pointer_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;
    ASSERT_EQ(value_meta_[pointer_id].flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
    ASSERT_FALSE(value_meta_[pointer_id].address_range_valid);

    EXPECT_CALL(*this, GetValue(pointer_id)).WillRepeatedly(ReturnRef(pointer_value));
    EXPECT_CALL(*this, GetValue(result_id)).WillRepeatedly(ReturnRef(stored_value));
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(types_.at(CommonTypes::u64)));
    EXPECT_CALL(*this, PointeeValueIsDescriptorBuffer(_)).WillOnce(Return(false));
    EXPECT_CALL(*this, WritePointer(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*this, OverrideFlagsPointee(pointer_id, result_id)).Times(1);
    EXPECT_CALL(*this, FindDataSourcesFromResultID(result_id, _))
        .WillOnce([](uint32_t, uint32_t* property_flags) {
            *property_flags = 0;
            return std::vector<::SPIRVSimulator::DataSourceBits>{};
        });
    EXPECT_CALL(*this, ResolvePointerV(_))
        .WillOnce([this](const ::SPIRVSimulator::PointerV&) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(), 0 };
        });
    EXPECT_CALL(*this, ValueHoldsPbufferPtr(result_id)).WillOnce(Return(false));
    EXPECT_CALL(*this, ValueIsArbitrary(result_id)).WillOnce(Return(false));

    const std::vector<uint32_t> words = { 0, pointer_id, result_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpStore,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    Store(instruction);

    EXPECT_TRUE(simulation_results.full_dispatch_needed);
}

TEST_F(DenseAddressRangeStoreConsumerTests,
       FloatVectorWriteDoesNotEnterHandleDenseRangeTracking)
{
    constexpr uint32_t pointer_id = 5070;
    constexpr uint32_t result_id  = 5071;

    ::SPIRVSimulator::MemoryFlagTracker memory_flag_tracker;
    ::SPIRVSimulator::SimulationResults simulation_results;
    SetRuntimeState(&memory_flag_tracker, &simulation_results);

    const ::SPIRVSimulator::PointerV pointer{
        .base_result_id = pointer_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };
    ::SPIRVSimulator::Value pointer_value = pointer;
    ::SPIRVSimulator::Value stored_value =
        std::make_shared<::SPIRVSimulator::VectorV>(
            std::initializer_list<double>{ 1.0, 2.0 });
    const ::SPIRVSimulator::Type stored_type =
        ::SPIRVSimulator::Type::Vector(CommonTypes::f32, 2);

    const uint64_t range_start =
        ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    SetAddressRange(pointer_id, range_start, range_start + 16, 8, 8);
    value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
    value_meta_[pointer_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;

    EXPECT_CALL(*this, GetValue(pointer_id)).WillRepeatedly(ReturnRef(pointer_value));
    EXPECT_CALL(*this, GetValue(result_id)).WillRepeatedly(ReturnRef(stored_value));
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(stored_type));
    EXPECT_CALL(*this, PointeeValueIsDescriptorBuffer(_)).WillOnce(Return(false));
    EXPECT_CALL(*this, WritePointer(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*this, OverrideFlagsPointee(pointer_id, result_id)).Times(1);
    EXPECT_CALL(*this, FindDataSourcesFromResultID(result_id, _))
        .WillOnce([](uint32_t, uint32_t* property_flags) {
            *property_flags = SPS_FLAG_IS_FLOAT_SOURCE;
            return std::vector<::SPIRVSimulator::DataSourceBits>{};
        });
    // The normal reaching-store bookkeeping still resolves the representative
    // destination once; dense handle-range consumption must not add another
    // resolution or publish a UniformDerivedRange for this float vector.
    EXPECT_CALL(*this, ResolvePointerV(_))
        .WillOnce([this](const ::SPIRVSimulator::PointerV&) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(), 0 };
        });
    EXPECT_CALL(*this, ValueHoldsPbufferPtr(result_id)).WillOnce(Return(false));
    EXPECT_CALL(*this, ValueIsArbitrary(result_id)).WillOnce(Return(false));

    const std::vector<uint32_t> words = { 0, pointer_id, result_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpStore,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    Store(instruction);

    EXPECT_FALSE(memory_flag_tracker.queryUniformDerivedRange(range_start).has_value());
    EXPECT_FALSE(simulation_results.full_dispatch_needed);
}

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
