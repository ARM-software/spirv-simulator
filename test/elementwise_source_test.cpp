#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

class ExternalLoadElementwiseSourceProducerTests : public ::SPIRVSimulator::SPIRVSimulator, public ::testing::Test
{
  protected:
    static constexpr uint32_t kPointerId      = 100;
    static constexpr uint32_t kFirstResultId  = 101;
    static constexpr uint32_t kSecondResultId = 102;
    static constexpr uint32_t kMaximumTestId  = 102;

    void SetUp() override
    {
        simulation_data_     = &simulation_data_storage_;
        simulation_results_  = &simulation_results_storage_;
        persistent_data_     = nullptr;
        memory_flag_tracker_ = nullptr;
        verbose_             = false;

        values_.resize(kMaximumTestId + 1);
        value_meta_.resize(kMaximumTestId + 1);
        SetPointerStorageClass(spv::StorageClassPhysicalStorageBuffer);
    }

    std::optional<::SPIRVSimulator::Value> ReadPointer(const ::SPIRVSimulator::PointerV&) override
    {
        ++read_count_;
        return representative_value_;
    }

    std::pair<std::byte*, uint64_t> ResolvePointerV(const ::SPIRVSimulator::PointerV&) const override
    {
        return { nullptr, 0 };
    }

    void SetPointerStorageClass(uint32_t storage_class)
    {
        values_[kPointerId] = ::SPIRVSimulator::PointerV{
            0, 0, CommonTypes::u64, 0, storage_class, {},
        };
    }

    static ::SPIRVSimulator::StridedElementSpan MakeSlots()
    {
        return ::SPIRVSimulator::StridedElementSpan{
            true, 0x1000, 16, 8, sizeof(uint64_t),
        };
    }

    void ExecuteLoad(uint32_t result_id)
    {
        std::vector<uint32_t> words = {
            0,
            CommonTypes::u64,
            result_id,
            kPointerId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = spv::Op::OpLoad,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        Op_Load(instruction);
    }

    const ::SPIRVSimulator::ValueMetadata& Metadata(uint32_t result_id) const { return value_meta_[result_id]; }

    const std::vector<std::weak_ptr<::SPIRVSimulator::ElementwiseSourceState>>& Registry() const
    {
        return active_elementwise_sources_;
    }

    ::SPIRVSimulator::SimulationData    simulation_data_storage_;
    ::SPIRVSimulator::SimulationResults simulation_results_storage_;
    uint64_t                            representative_value_ = 0x12345678;
    uint32_t                            read_count_           = 0;
};

TEST_F(ExternalLoadElementwiseSourceProducerTests, PhysicalStorageLoadPublishesLazyElementwiseSource)
{
    const ::SPIRVSimulator::StridedElementSpan slots = MakeSlots();
    value_meta_[kPointerId].strided_element_span     = slots;

    ExecuteLoad(kFirstResultId);

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kFirstResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kFirstResultId]), representative_value_);
    EXPECT_EQ(read_count_, 1u);

    const ::SPIRVSimulator::ValueMetadata& result_meta = Metadata(kFirstResultId);
    ASSERT_NE(result_meta.elementwise_relation.source, nullptr);
    EXPECT_EQ(result_meta.elementwise_relation.source->slots, slots);
    EXPECT_EQ(result_meta.elementwise_relation.source->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::LazyHostShadow);
    EXPECT_TRUE(result_meta.elementwise_relation.source->packed_snapshot.empty());
    EXPECT_FALSE(result_meta.strided_element_span.valid);
    EXPECT_EQ(result_meta.invocation_dependence, ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_NE(result_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);

    ASSERT_EQ(Registry().size(), 1u);
    EXPECT_EQ(Registry()[0].lock(), result_meta.elementwise_relation.source);
}

TEST_F(ExternalLoadElementwiseSourceProducerTests, PhysicalStorageLoadWithoutSpanDoesNotPublishSource)
{
    value_meta_[kFirstResultId].elementwise_relation.source =
        std::make_shared<::SPIRVSimulator::ElementwiseSourceState>(MakeSlots());
    value_meta_[kFirstResultId].invocation_dependence = ::SPIRVSimulator::InvocationDependence::Invariant;
    value_meta_[kFirstResultId].flags                 = SPS_FLAG_THREAD_SPECIFIC;

    ExecuteLoad(kFirstResultId);

    const ::SPIRVSimulator::ValueMetadata& result_meta = Metadata(kFirstResultId);
    EXPECT_EQ(result_meta.elementwise_relation.source, nullptr);
    EXPECT_EQ(result_meta.invocation_dependence, ::SPIRVSimulator::InvocationDependence::Unknown);
    EXPECT_EQ(result_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
    EXPECT_TRUE(Registry().empty());
}

TEST_F(ExternalLoadElementwiseSourceProducerTests, SupportedExternalHostShadowLoadsPublishSource)
{
    constexpr std::array<uint32_t, 4> storage_classes{
        spv::StorageClassStorageBuffer,
        spv::StorageClassPushConstant,
        spv::StorageClassUniformConstant,
        spv::StorageClassUniform,
    };

    const ::SPIRVSimulator::StridedElementSpan slots = MakeSlots();
    for (uint32_t storage_class : storage_classes)
    {
        SCOPED_TRACE(storage_class);
        SetPointerStorageClass(storage_class);
        value_meta_[kPointerId].strided_element_span = slots;

        ExecuteLoad(kFirstResultId);

        const ::SPIRVSimulator::ValueMetadata& result_meta = Metadata(kFirstResultId);
        ASSERT_NE(result_meta.elementwise_relation.source, nullptr);
        EXPECT_EQ(result_meta.elementwise_relation.source->slots, slots);
        EXPECT_FALSE(result_meta.strided_element_span.valid);
        EXPECT_EQ(result_meta.invocation_dependence,
                  ::SPIRVSimulator::InvocationDependence::Varying);
        EXPECT_NE(result_meta.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
    }
}

TEST_F(ExternalLoadElementwiseSourceProducerTests, SeparateLoadsFromSameSpanUseDistinctState)
{
    const ::SPIRVSimulator::StridedElementSpan slots = MakeSlots();
    value_meta_[kPointerId].strided_element_span     = slots;

    ExecuteLoad(kFirstResultId);
    ExecuteLoad(kSecondResultId);

    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState>& first_source =
        Metadata(kFirstResultId).elementwise_relation.source;
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState>& second_source =
        Metadata(kSecondResultId).elementwise_relation.source;
    ASSERT_NE(first_source, nullptr);
    ASSERT_NE(second_source, nullptr);
    EXPECT_EQ(first_source->slots, slots);
    EXPECT_EQ(second_source->slots, slots);
    EXPECT_NE(first_source, second_source);

    ASSERT_EQ(Registry().size(), 2u);
    EXPECT_EQ(Registry()[0].lock(), first_source);
    EXPECT_EQ(Registry()[1].lock(), second_source);
}

class ElementwisePhysicalAddressResultTests : public ::SPIRVSimulator::SPIRVSimulator,
                                              public ::testing::Test
{
  protected:
    static constexpr uint32_t kOperandId = 1;

    void SetUp() override
    {
        simulation_results_ = &simulation_results_storage_;
        memory_flag_tracker_ = nullptr;
        value_meta_.resize(kOperandId + 1);
    }

    static ::SPIRVSimulator::StridedElementSpan MakeSlots()
    {
        return {
            true,
            0x1000,
            16,
            8,
            sizeof(uint64_t),
        };
    }

    std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> SetSource()
    {
        auto source = std::make_shared<::SPIRVSimulator::ElementwiseSourceState>(MakeSlots());
        value_meta_[kOperandId].elementwise_relation.source = source;
        return source;
    }

    CompressedOperationResult Publish(uint64_t expected_element_size)
    {
        return TryPublishElementwisePhysicalAddressResult(kOperandId, expected_element_size);
    }

    void Handle(uint64_t expected_element_size)
    {
        HandleElementwisePhysicalAddressResult(kOperandId, expected_element_size);
    }

    ::SPIRVSimulator::SimulationResults simulation_results_storage_;
};

TEST_F(ElementwisePhysicalAddressResultTests, PublishesCompleteResultWithoutClassView)
{
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> source = SetSource();

    EXPECT_EQ(Publish(sizeof(uint64_t)), CompressedOperationResult::Complete);

    ASSERT_EQ(simulation_results_storage_.physical_address_data.size(), 1u);
    const ::SPIRVSimulator::PhysicalAddressData& result =
        simulation_results_storage_.physical_address_data.front();
    EXPECT_TRUE(result.HasValidPhysicalAddressRange());
    EXPECT_EQ(result.physical_address_range->elementwise_source, source);
    EXPECT_FALSE(simulation_results_storage_.read_side_pointer_coverage_incomplete);
}

TEST_F(ElementwisePhysicalAddressResultTests, UnavailableSourceRequiresCoverageFallback)
{
    SetSource()->kind = ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable;

    Handle(sizeof(uint64_t));

    EXPECT_TRUE(simulation_results_storage_.physical_address_data.empty());
    EXPECT_TRUE(simulation_results_storage_.read_side_pointer_coverage_incomplete);
}

TEST_F(ElementwisePhysicalAddressResultTests, MissingVaryingSourceRequiresCoverageFallback)
{
    value_meta_[kOperandId].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;

    Handle(sizeof(uint64_t));

    EXPECT_TRUE(simulation_results_storage_.physical_address_data.empty());
    EXPECT_TRUE(simulation_results_storage_.read_side_pointer_coverage_incomplete);
}

class ElementwiseSourceRegistryTests : public ::SPIRVSimulator::SPIRVSimulator, public ::testing::Test
{
  protected:
    void SetUp() override { value_meta_.resize(64); }

    static ::SPIRVSimulator::StridedElementSpan MakeSlots(uint64_t first_address)
    {
        return ::SPIRVSimulator::StridedElementSpan{
            true, first_address, 8, 4, sizeof(uint64_t),
        };
    }

    bool CreateSource(uint32_t result_id)
    {
        return TryCreateLazyElementwiseSource(result_id, MakeSlots(0x1000 + static_cast<uint64_t>(result_id) * 0x100));
    }

    void ForceCompaction()
    {
        next_elementwise_source_compaction_size_ = active_elementwise_sources_.size();
        MaybeCompactElementwiseSourceRegistry();
    }

    size_t RegistrySize() const { return active_elementwise_sources_.size(); }

    size_t CompactionThreshold() const { return next_elementwise_source_compaction_size_; }
};

TEST_F(ElementwiseSourceRegistryTests, RegistryDoesNotOwnState)
{
    ASSERT_TRUE(CreateSource(1));
    ASSERT_EQ(RegistrySize(), 1u);

    const std::weak_ptr<::SPIRVSimulator::ElementwiseSourceState> weak_source =
        active_elementwise_sources_[0];
    ASSERT_FALSE(weak_source.expired());

    value_meta_[1].elementwise_relation.source = {};
    EXPECT_TRUE(weak_source.expired());

    ForceCompaction();
    EXPECT_EQ(RegistrySize(), 0u);
    EXPECT_EQ(CompactionThreshold(), 64u);
}

TEST_F(ElementwiseSourceRegistryTests, CompactionRemovesExpiredAndUnavailableButKeepsLazyAndPacked)
{
    ASSERT_TRUE(CreateSource(1));
    ASSERT_TRUE(CreateSource(2));
    ASSERT_TRUE(CreateSource(3));
    ASSERT_TRUE(CreateSource(4));

    value_meta_[1].elementwise_relation.source       = {};
    value_meta_[2].elementwise_relation.source->kind = ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable;
    value_meta_[4].elementwise_relation.source->kind = ::SPIRVSimulator::ElementwiseSourceStorageKind::PackedSnapshot;

    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> lazy_source =
        value_meta_[3].elementwise_relation.source;
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> packed_source =
        value_meta_[4].elementwise_relation.source;

    ForceCompaction();

    ASSERT_EQ(RegistrySize(), 2u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), lazy_source);
    EXPECT_EQ(active_elementwise_sources_[1].lock(), packed_source);
    EXPECT_EQ(CompactionThreshold(), 64u);
}

TEST_F(ElementwiseSourceRegistryTests, CompactionThresholdDoublesLargeLiveSet)
{
    constexpr uint32_t kLiveSourceCount = 40;
    for (uint32_t result_id = 0; result_id < kLiveSourceCount; ++result_id)
    {
        ASSERT_TRUE(CreateSource(result_id));
    }

    ForceCompaction();

    EXPECT_EQ(RegistrySize(), kLiveSourceCount);
    EXPECT_EQ(CompactionThreshold(), 2u * kLiveSourceCount);
}

class ExternalWriteElementwiseSourcePreparationTests : public ::SPIRVSimulator::SPIRVSimulator,
                                                       public ::testing::Test
{
  protected:
    static constexpr uint32_t kFirstSourceId  = 1;
    static constexpr uint32_t kSecondSourceId = 2;
    static constexpr uint32_t kPointerId      = 10;
    static constexpr uint32_t kStoredValueId  = 11;
    static constexpr uint32_t kMaximumTestId  = 11;
    static constexpr uint32_t kPointerTypeId  = 100;
    static constexpr uint32_t kFloatVectorTypeId = 101;
    static constexpr uint32_t kStoredPointerTypeId = 102;
    static constexpr uint32_t kPointerDestinationTypeId = 103;
    static constexpr uint32_t kLegacyStorageBufferTypeId = 104;
    static constexpr uint32_t kLegacyStorageBufferPointerTypeId = 105;

    void SetUp() override
    {
        simulation_data_     = &simulation_data_storage_;
        simulation_results_  = &simulation_results_storage_;
        persistent_data_     = nullptr;
        memory_flag_tracker_ = nullptr;
        verbose_             = false;

        values_.resize(kMaximumTestId + 1);
        value_meta_.resize(kMaximumTestId + 1);
        types_[CommonTypes::u64] = ::SPIRVSimulator::Type::Int(64, false);
        types_[CommonTypes::f64] = ::SPIRVSimulator::Type::Float(64);
        types_[kPointerTypeId] = ::SPIRVSimulator::Type::Pointer(
            spv::StorageClassPhysicalStorageBuffer,
            CommonTypes::u64);
        values_[kPointerId] = ::SPIRVSimulator::PointerV{
            Address(0),
            0,
            kPointerTypeId,
            kPointerId,
            spv::StorageClassPhysicalStorageBuffer,
            {},
        };
        values_[kStoredValueId] = uint64_t{ 0xfeedfacecafebeef };
    }

    bool WritePointer(const ::SPIRVSimulator::PointerV&, const ::SPIRVSimulator::Value& value) override
    {
        ++write_count_;
        if (before_write_)
        {
            before_write_();
        }

        uint64_t stored_value = 0;
        if (std::holds_alternative<uint64_t>(value))
        {
            stored_value = std::get<uint64_t>(value);
        }
        else if (std::holds_alternative<::SPIRVSimulator::PointerV>(value))
        {
            stored_value = std::get<::SPIRVSimulator::PointerV>(value).pointer_handle;
        }
        else
        {
            WriteValue(backing_.data() + resolved_offset_, stored_type_id_, value);
            return true;
        }
        std::memcpy(backing_.data() + resolved_offset_, &stored_value, sizeof(stored_value));
        return true;
    }

    uint32_t GetTypeID(uint32_t result_id) const override
    {
        if (result_id == kStoredValueId)
        {
            return stored_type_id_;
        }
        return ::SPIRVSimulator::SPIRVSimulator::GetTypeID(result_id);
    }

    std::pair<std::byte*, uint64_t> ResolvePointerV(const ::SPIRVSimulator::PointerV&) const override
    {
        return { backing_.data(), resolved_offset_ };
    }

    std::vector<::SPIRVSimulator::DataSourceBits> FindDataSourcesFromResultID(
        uint32_t,
        uint32_t* property_flags) override
    {
        if (property_flags != nullptr)
        {
            *property_flags = 0;
        }
        return {};
    }

    uint64_t Address(size_t offset) const
    {
        return ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data() + offset);
    }

    void WriteBacking(size_t offset, uint64_t value)
    {
        std::memcpy(backing_.data() + offset, &value, sizeof(value));
    }

    uint64_t ReadBacking(size_t offset) const
    {
        uint64_t value = 0;
        std::memcpy(&value, backing_.data() + offset, sizeof(value));
        return value;
    }

    static uint64_t ReadSnapshotElement(
        const ::SPIRVSimulator::ElementwiseSourceState& source,
        size_t                                           element_index)
    {
        uint64_t value = 0;
        std::memcpy(&value,
                    source.packed_snapshot.data() + element_index * sizeof(value),
                    sizeof(value));
        return value;
    }

    std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> CreateSource(
        uint32_t                                       result_id,
        const ::SPIRVSimulator::StridedElementSpan& slots)
    {
        if (!TryCreateLazyElementwiseSource(result_id, slots))
        {
            return nullptr;
        }
        return value_meta_[result_id].elementwise_relation.source;
    }

    void Prepare(const ::SPIRVSimulator::StridedElementSpan& write_span)
    {
        ::SPIRVSimulator::ExternalWriteFootprint footprint;
        footprint.kind = ::SPIRVSimulator::ExternalWriteFootprint::Kind::ExactSpan;
        footprint.span = write_span;
        PrepareElementwiseSourcesBeforeExternalWrite(footprint);
    }

    void ExecuteStore(const ::SPIRVSimulator::StridedElementSpan& destination_slots)
    {
        value_meta_[kPointerId].strided_element_span = destination_slots;
        const std::vector<uint32_t> words = { 0, kPointerId, kStoredValueId };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = spv::Op::OpStore,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        Op_Store(instruction);
    }

    void SetUniformDestinationDecoration(spv::Decoration decoration)
    {
        types_[kLegacyStorageBufferTypeId] =
            ::SPIRVSimulator::Type::Struct(kLegacyStorageBufferTypeId);
        types_[kLegacyStorageBufferPointerTypeId] =
            ::SPIRVSimulator::Type::Pointer(
                spv::StorageClassUniform,
                kLegacyStorageBufferTypeId);
        decorators_[kLegacyStorageBufferTypeId].clear();
        decorators_[kLegacyStorageBufferTypeId].push_back({ decoration, {} });
        values_[kPointerId] = ::SPIRVSimulator::PointerV{
            Address(0),
            0,
            kLegacyStorageBufferPointerTypeId,
            kPointerId,
            spv::StorageClassUniform,
            {},
        };
    }

    mutable std::array<std::byte, 128> backing_{};
    ::SPIRVSimulator::SimulationData    simulation_data_storage_;
    ::SPIRVSimulator::SimulationResults simulation_results_storage_;
    std::function<void()>               before_write_;
    size_t                              resolved_offset_ = 0;
    uint32_t                            write_count_      = 0;
    uint32_t                            stored_type_id_   = CommonTypes::u64;
};

TEST_F(ExternalWriteElementwiseSourcePreparationTests, ExactSameSlotStoreMaterializesBeforeWrite)
{
    constexpr uint64_t kLowValue         = 0x1111111111111111;
    constexpr uint64_t kMiddleValue      = 0x2222222222222222;
    constexpr uint64_t kHighValue        = 0x3333333333333333;
    constexpr size_t   kLowOffset        = 8;
    constexpr size_t   kMiddleOffset     = 24;
    constexpr size_t   kHighOffset       = 40;
    constexpr size_t   kElementByteCount = 3 * sizeof(uint64_t);

    WriteBacking(kLowOffset, kLowValue);
    WriteBacking(kMiddleOffset, kMiddleValue);
    WriteBacking(kHighOffset, kHighValue);

    const ::SPIRVSimulator::StridedElementSpan slots{
        true,
        Address(kHighOffset),
        -16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, slots);
    ASSERT_NE(storage, nullptr);

    resolved_offset_ = kHighOffset;
    before_write_ = [storage, kElementByteCount, kHighValue, kMiddleValue, kLowValue]() {
        EXPECT_EQ(storage->kind,
                  ::SPIRVSimulator::ElementwiseSourceStorageKind::PackedSnapshot);
        ASSERT_EQ(storage->packed_snapshot.size(), kElementByteCount);
        EXPECT_EQ(ReadSnapshotElement(*storage, 0), kHighValue);
        EXPECT_EQ(ReadSnapshotElement(*storage, 1), kMiddleValue);
        EXPECT_EQ(ReadSnapshotElement(*storage, 2), kLowValue);
    };

    ExecuteStore(slots);

    EXPECT_EQ(write_count_, 1u);
    EXPECT_EQ(ReadBacking(kHighOffset), std::get<uint64_t>(values_[kStoredValueId]));
    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::PackedSnapshot);
    ASSERT_EQ(storage->packed_snapshot.size(), kElementByteCount);
    EXPECT_EQ(ReadSnapshotElement(*storage, 0), kHighValue);
    EXPECT_EQ(ReadSnapshotElement(*storage, 1), kMiddleValue);
    EXPECT_EQ(ReadSnapshotElement(*storage, 2), kLowValue);
    ASSERT_EQ(active_elementwise_sources_.size(), 1u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), storage);
    // The store has no applicable compressed value mode, while its destination
    // span proves multi-invocation coverage. The representative write remains,
    // but complete write coverage requires fallback.
    EXPECT_TRUE(simulation_results_storage_.full_dispatch_needed);
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests, InterleavedDisjointWriteKeepsLazySource)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan write_slots{
        true,
        Address(8),
        16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    Prepare(write_slots);

    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::LazyHostShadow);
    EXPECT_TRUE(storage->packed_snapshot.empty());
    ASSERT_EQ(active_elementwise_sources_.size(), 1u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), storage);
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests, SingletonSourceInStridedGapKeepsLazySource)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(8),
        0,
        1,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan write_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    Prepare(write_slots);

    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::LazyHostShadow);
    ASSERT_EQ(active_elementwise_sources_.size(), 1u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), storage);
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests, SingletonWriteInStridedGapKeepsLazySource)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan write_slots{
        true,
        Address(8),
        0,
        1,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    Prepare(write_slots);

    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::LazyHostShadow);
    ASSERT_EQ(active_elementwise_sources_.size(), 1u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), storage);
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests, EqualStrideUnequalElementSizesCanRemainDisjoint)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        3,
        12,
    };
    const ::SPIRVSimulator::StridedElementSpan write_slots{
        true,
        Address(12),
        16,
        3,
        4,
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    Prepare(write_slots);

    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::LazyHostShadow);
    ASSERT_EQ(active_elementwise_sources_.size(), 1u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), storage);
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests, EqualStrideLargeUnequalElementsOverlap)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        3,
        20,
    };
    const ::SPIRVSimulator::StridedElementSpan write_slots{
        true,
        Address(12),
        16,
        3,
        4,
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    Prepare(write_slots);

    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable);
    EXPECT_TRUE(active_elementwise_sources_.empty());
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests, UnequalStrideArithmeticJumpCanProveDisjoint)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        4,
        4,
    };
    const ::SPIRVSimulator::StridedElementSpan write_slots{
        true,
        Address(8),
        18,
        3,
        4,
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    Prepare(write_slots);

    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::LazyHostShadow);
    ASSERT_EQ(active_elementwise_sources_.size(), 1u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), storage);
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests, UnequalStrideArithmeticJumpFindsLaterOverlap)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        4,
        4,
    };
    const ::SPIRVSimulator::StridedElementSpan write_slots{
        true,
        Address(8),
        20,
        3,
        4,
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    Prepare(write_slots);

    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable);
    EXPECT_TRUE(active_elementwise_sources_.empty());
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests,
       NonExactOverlapInvalidatesUnrelatedSourceButDoesNotBlockCompressedConsumer)
{
    constexpr uint64_t kFirstExactValue  = 0x1111222233334444;
    constexpr uint64_t kSecondExactValue = 0x5555666677778888;
    constexpr uint64_t kThirdExactValue  = 0x9999aaaabbbbcccc;

    const ::SPIRVSimulator::StridedElementSpan overlapping_slots{
        true,
        Address(16),
        16,
        3,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan exact_slots{
        true,
        Address(32),
        16,
        3,
        sizeof(uint64_t),
    };

    WriteBacking(32, kFirstExactValue);
    WriteBacking(48, kSecondExactValue);
    WriteBacking(64, kThirdExactValue);

    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> overlapping_storage =
        CreateSource(kFirstSourceId, overlapping_slots);
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> exact_storage =
        CreateSource(kSecondSourceId, exact_slots);
    ASSERT_NE(overlapping_storage, nullptr);
    ASSERT_NE(exact_storage, nullptr);
    value_meta_[kStoredValueId].elementwise_relation.source = exact_storage;

    overlapping_storage->kind =
        ::SPIRVSimulator::ElementwiseSourceStorageKind::PackedSnapshot;
    overlapping_storage->packed_snapshot.assign(3 * sizeof(uint64_t),
                                                 std::byte{ 0x5a });

    ExecuteStore(exact_slots);

    EXPECT_EQ(write_count_, 0u);
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(overlapping_storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable);
    EXPECT_TRUE(overlapping_storage->packed_snapshot.empty());
    ASSERT_EQ(exact_storage->packed_snapshot.size(), 3 * sizeof(uint64_t));
    EXPECT_EQ(ReadSnapshotElement(*exact_storage, 0), kFirstExactValue);
    EXPECT_EQ(ReadSnapshotElement(*exact_storage, 1), kSecondExactValue);
    EXPECT_EQ(ReadSnapshotElement(*exact_storage, 2), kThirdExactValue);
    ASSERT_EQ(active_elementwise_sources_.size(), 1u);
    EXPECT_EQ(active_elementwise_sources_[0].lock(), exact_storage);
}

TEST_F(ExternalWriteElementwiseSourcePreparationTests,
       UnavailableSourceRequestsFullDispatchOnlyWhenLaterConsumed)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan overlapping_write_slots{
        true,
        Address(16),
        16,
        3,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan later_destination_slots{
        true,
        Address(64),
        16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kFirstSourceId, source_slots);
    ASSERT_NE(storage, nullptr);

    // Keep the first store independently compressible as a fixed broadcast so
    // this test continues to isolate deferred consumption of the unavailable
    // provenance rather than the general multi-invocation store fallback.
    value_meta_[kStoredValueId].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Invariant;
    resolved_offset_ = 16;
    ExecuteStore(overlapping_write_slots);

    EXPECT_EQ(write_count_, 0u);
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable);

    // Model ordinary SSA metadata transport after the source became
    // unavailable. The later store is the first compressed consumer of that
    // provenance, so this is where full dispatch becomes necessary.
    value_meta_[kStoredValueId].elementwise_relation.source = storage;
    resolved_offset_ = 64;
    ExecuteStore(later_destination_slots);

    EXPECT_EQ(write_count_, 1u);
    EXPECT_TRUE(simulation_results_storage_.full_dispatch_needed);
}

class CompressedElementwiseExternalStoreTests : public ExternalWriteElementwiseSourcePreparationTests
{};

TEST_F(CompressedElementwiseExternalStoreTests, LazySourceWritesAllStridedDestinationSlots)
{
    constexpr uint64_t kLowSourceValue    = 0x1111111111111111;
    constexpr uint64_t kMiddleSourceValue = 0x2222222222222222;
    constexpr uint64_t kHighSourceValue   = 0x3333333333333333;
    constexpr uint64_t kPaddingValue      = 0xaaaaaaaaaaaaaaaa;

    WriteBacking(0, kLowSourceValue);
    WriteBacking(8, kMiddleSourceValue);
    WriteBacking(16, kHighSourceValue);
    WriteBacking(72, kPaddingValue);
    WriteBacking(88, kPaddingValue);
    WriteBacking(104, kPaddingValue);

    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(16),
        -8,
        3,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(64),
        16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kStoredValueId, source_slots);
    ASSERT_NE(storage, nullptr);

    resolved_offset_ = 64;
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 0u);
    EXPECT_EQ(ReadBacking(64), kHighSourceValue);
    EXPECT_EQ(ReadBacking(80), kMiddleSourceValue);
    EXPECT_EQ(ReadBacking(96), kLowSourceValue);
    EXPECT_EQ(ReadBacking(72), kPaddingValue);
    EXPECT_EQ(ReadBacking(88), kPaddingValue);
    EXPECT_EQ(ReadBacking(104), kPaddingValue);
    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::LazyHostShadow);
    EXPECT_TRUE(storage->packed_snapshot.empty());
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
}

TEST_F(CompressedElementwiseExternalStoreTests, PackedSnapshotIsUsedInsteadOfCurrentSourceBytes)
{
    constexpr std::array<uint64_t, 3> kLoadTimeValues{
        0x1111222233334444,
        0x5555666677778888,
        0x9999aaaabbbbcccc,
    };
    constexpr std::array<uint64_t, 3> kCurrentSourceValues{
        0xaaaabbbbccccdddd,
        0x1111333355557777,
        0x2468ace013579bdf,
    };

    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        8,
        3,
        sizeof(uint64_t),
    };
    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(64),
        16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kStoredValueId, source_slots);
    ASSERT_NE(storage, nullptr);

    storage->packed_snapshot.resize(sizeof(kLoadTimeValues));
    std::memcpy(storage->packed_snapshot.data(),
                kLoadTimeValues.data(),
                sizeof(kLoadTimeValues));
    storage->kind = ::SPIRVSimulator::ElementwiseSourceStorageKind::PackedSnapshot;

    WriteBacking(0, kCurrentSourceValues[0]);
    WriteBacking(8, kCurrentSourceValues[1]);
    WriteBacking(16, kCurrentSourceValues[2]);

    resolved_offset_ = 64;
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 0u);
    EXPECT_EQ(ReadBacking(64), kLoadTimeValues[0]);
    EXPECT_EQ(ReadBacking(80), kLoadTimeValues[1]);
    EXPECT_EQ(ReadBacking(96), kLoadTimeValues[2]);
    EXPECT_EQ(ReadBacking(0), kCurrentSourceValues[0]);
    EXPECT_EQ(ReadBacking(8), kCurrentSourceValues[1]);
    EXPECT_EQ(ReadBacking(16), kCurrentSourceValues[2]);
    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::PackedSnapshot);
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
}

TEST_F(CompressedElementwiseExternalStoreTests,
       MissingDestinationSpanDoesNotConsumeUnavailableSource)
{
    const ::SPIRVSimulator::StridedElementSpan source_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> storage =
        CreateSource(kStoredValueId, source_slots);
    ASSERT_NE(storage, nullptr);
    storage->kind = ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable;
    active_elementwise_sources_.clear();

    resolved_offset_ = 64;
    ExecuteStore({});

    EXPECT_EQ(write_count_, 1u);
    // Missing destination geometry keeps the elementwise mode inapplicable,
    // but the source relation still proves that the object covers multiple
    // invocations, so the representative write requires coverage fallback.
    EXPECT_TRUE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(storage->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::Unavailable);
}

TEST_F(CompressedElementwiseExternalStoreTests,
       InvalidSourceSlotsRequireFullDispatchForStridedDestination)
{
    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(64),
        16,
        3,
        sizeof(uint64_t),
    };
    value_meta_[kStoredValueId].elementwise_relation.source =
        std::make_shared<::SPIRVSimulator::ElementwiseSourceState>(
            ::SPIRVSimulator::StridedElementSpan{});

    resolved_offset_ = 64;
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 1u);
    EXPECT_TRUE(simulation_results_storage_.full_dispatch_needed);
}

class CompressedFixedValueExternalStoreTests : public ExternalWriteElementwiseSourcePreparationTests
{
  protected:
    void MarkStoredValueInvariant()
    {
        value_meta_[kStoredValueId].invocation_dependence =
            ::SPIRVSimulator::InvocationDependence::Invariant;
    }
};

TEST_F(CompressedFixedValueExternalStoreTests,
       InvariantScalarBroadcastsToStorageBuffersButNotUniformBlocks)
{
    constexpr uint64_t kStoredValue  = 0x1122334455667788;
    constexpr uint64_t kPaddingValue = 0xa5a5a5a5a5a5a5a5;

    values_[kStoredValueId] = kStoredValue;
    MarkStoredValueInvariant();
    WriteBacking(8, kPaddingValue);
    WriteBacking(24, kPaddingValue);
    WriteBacking(40, kPaddingValue);

    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    resolved_offset_ = 0;
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 0u);
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(ReadBacking(0), kStoredValue);
    EXPECT_EQ(ReadBacking(16), kStoredValue);
    EXPECT_EQ(ReadBacking(32), kStoredValue);
    EXPECT_EQ(ReadBacking(8), kPaddingValue);
    EXPECT_EQ(ReadBacking(24), kPaddingValue);
    EXPECT_EQ(ReadBacking(40), kPaddingValue);

    SetUniformDestinationDecoration(spv::DecorationBufferBlock);
    backing_.fill(std::byte{});
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 0u);
    EXPECT_EQ(ReadBacking(0), kStoredValue);
    EXPECT_EQ(ReadBacking(16), kStoredValue);
    EXPECT_EQ(ReadBacking(32), kStoredValue);

    SetUniformDestinationDecoration(spv::DecorationBlock);
    backing_.fill(std::byte{});
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 1u);
    EXPECT_EQ(ReadBacking(0), kStoredValue);
    EXPECT_EQ(ReadBacking(16), 0u);
    EXPECT_EQ(ReadBacking(32), 0u);
}

TEST_F(CompressedFixedValueExternalStoreTests, InvariantFloatVectorBroadcastsWholeValueToEverySlot)
{
    constexpr std::array<double, 2> kStoredValue{ 1.25, -2.5 };
    constexpr uint64_t              kPaddingValue = 0xa5a5a5a5a5a5a5a5;

    types_[kFloatVectorTypeId] = ::SPIRVSimulator::Type::Vector(CommonTypes::f64, 2);
    stored_type_id_ = kFloatVectorTypeId;
    values_[kStoredValueId] =
        std::make_shared<::SPIRVSimulator::VectorV>(
            std::initializer_list<double>{ kStoredValue[0], kStoredValue[1] });
    MarkStoredValueInvariant();
    WriteBacking(16, kPaddingValue);
    WriteBacking(48, kPaddingValue);
    WriteBacking(80, kPaddingValue);

    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(0),
        32,
        3,
        sizeof(kStoredValue),
    };
    resolved_offset_ = 0;
    ExecuteStore(destination_slots);

    const auto read_vector = [this](size_t offset) {
        std::array<double, 2> value{};
        std::memcpy(value.data(), backing_.data() + offset, sizeof(value));
        return value;
    };
    EXPECT_EQ(write_count_, 0u);
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(read_vector(0), kStoredValue);
    EXPECT_EQ(read_vector(32), kStoredValue);
    EXPECT_EQ(read_vector(64), kStoredValue);
    EXPECT_EQ(ReadBacking(16), kPaddingValue);
    EXPECT_EQ(ReadBacking(48), kPaddingValue);
    EXPECT_EQ(ReadBacking(80), kPaddingValue);
}

class InvalidFixedValueFlagTests
    : public CompressedFixedValueExternalStoreTests,
      public ::testing::WithParamInterface<uint64_t>
{};

TEST_P(InvalidFixedValueFlagTests, RequiresFullDispatch)
{
    constexpr uint64_t kStoredValue = 0x1122334455667788;
    constexpr uint64_t kOldValue    = 0xa5a5a5a5a5a5a5a5;

    values_[kStoredValueId] = kStoredValue;
    MarkStoredValueInvariant();
    value_meta_[kStoredValueId].flags = GetParam();
    WriteBacking(0, kOldValue);
    WriteBacking(16, kOldValue);
    WriteBacking(32, kOldValue);

    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    resolved_offset_ = 0;
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 1u);
    EXPECT_TRUE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(ReadBacking(0), kStoredValue);
    EXPECT_EQ(ReadBacking(16), kOldValue);
    EXPECT_EQ(ReadBacking(32), kOldValue);
}

INSTANTIATE_TEST_SUITE_P(InvalidFixedValueFlags,
                         InvalidFixedValueFlagTests,
                         ::testing::Values(SPS_FLAG_IS_ARBITRARY,
                                           SPS_FLAG_UNINITIALIZED));

TEST_F(CompressedFixedValueExternalStoreTests, InvariantValueWithoutDestinationSpanIsNotApplicable)
{
    constexpr uint64_t kStoredValue = 0x1122334455667788;

    values_[kStoredValueId] = kStoredValue;
    MarkStoredValueInvariant();
    resolved_offset_ = 64;
    ExecuteStore({});

    EXPECT_EQ(write_count_, 1u);
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(ReadBacking(64), kStoredValue);
}

TEST_F(CompressedFixedValueExternalStoreTests, UnsupportedInvariantPointerRequiresFullDispatch)
{
    constexpr uint64_t kStoredPointer = 0x1122334455667788;
    constexpr uint64_t kOldValue      = 0xa5a5a5a5a5a5a5a5;

    types_[kStoredPointerTypeId] = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer,
        CommonTypes::u64);
    types_[kPointerDestinationTypeId] = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer,
        kStoredPointerTypeId);
    std::get<::SPIRVSimulator::PointerV>(values_[kPointerId]).base_type_id =
        kPointerDestinationTypeId;
    stored_type_id_ = kStoredPointerTypeId;
    values_[kStoredValueId] = ::SPIRVSimulator::PointerV{
        kStoredPointer,
        0,
        kStoredPointerTypeId,
        kStoredValueId,
        spv::StorageClassPhysicalStorageBuffer,
        {},
    };
    MarkStoredValueInvariant();
    WriteBacking(0, kOldValue);
    WriteBacking(16, kOldValue);
    WriteBacking(32, kOldValue);

    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    resolved_offset_ = 0;
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 1u);
    EXPECT_TRUE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(ReadBacking(0), kStoredPointer);
    EXPECT_EQ(ReadBacking(16), kOldValue);
    EXPECT_EQ(ReadBacking(32), kOldValue);
}

TEST_F(CompressedFixedValueExternalStoreTests,
       FixedBroadcastSnapshotsOverlappingLazySourceBeforeScatter)
{
    constexpr uint64_t kFirstOldValue  = 0x1111111111111111;
    constexpr uint64_t kSecondOldValue = 0x2222222222222222;
    constexpr uint64_t kThirdOldValue  = 0x3333333333333333;
    constexpr uint64_t kStoredValue    = 0xfeedfacecafebeef;

    WriteBacking(0, kFirstOldValue);
    WriteBacking(16, kSecondOldValue);
    WriteBacking(32, kThirdOldValue);
    const ::SPIRVSimulator::StridedElementSpan destination_slots{
        true,
        Address(0),
        16,
        3,
        sizeof(uint64_t),
    };
    const std::shared_ptr<::SPIRVSimulator::ElementwiseSourceState> source =
        CreateSource(kFirstSourceId, destination_slots);
    ASSERT_NE(source, nullptr);

    values_[kStoredValueId] = kStoredValue;
    MarkStoredValueInvariant();
    resolved_offset_ = 0;
    ExecuteStore(destination_slots);

    EXPECT_EQ(write_count_, 0u);
    EXPECT_FALSE(simulation_results_storage_.full_dispatch_needed);
    EXPECT_EQ(source->kind,
              ::SPIRVSimulator::ElementwiseSourceStorageKind::PackedSnapshot);
    ASSERT_EQ(source->packed_snapshot.size(), 3 * sizeof(uint64_t));
    EXPECT_EQ(ReadSnapshotElement(*source, 0), kFirstOldValue);
    EXPECT_EQ(ReadSnapshotElement(*source, 1), kSecondOldValue);
    EXPECT_EQ(ReadSnapshotElement(*source, 2), kThirdOldValue);
    EXPECT_EQ(ReadBacking(0), kStoredValue);
    EXPECT_EQ(ReadBacking(16), kStoredValue);
    EXPECT_EQ(ReadBacking(32), kStoredValue);
}

enum class InternalElementwiseTarget
{
    WholeObject,
    FirstMember,
    SecondMember,
};

class InternalElementwiseSourceTransportTests : public SPIRVSimulatorMockBase, public ::testing::Test
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
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (override));
    MOCK_METHOD((std::vector<::SPIRVSimulator::DataSourceBits>),
                FindDataSourcesFromResultID,
                (uint32_t result_id, uint32_t* property_flags),
                (override));

  protected:
    static constexpr uint32_t kPairTypeId            = 200;
    static constexpr uint32_t kWholePointerId        = 201;
    static constexpr uint32_t kFirstMemberPointerId  = 202;
    static constexpr uint32_t kSecondMemberPointerId = 203;
    static constexpr uint32_t kSourcePairId          = 210;
    static constexpr uint32_t kReplacementPairId     = 211;
    static constexpr uint32_t kReplacementScalarId   = 212;
    static constexpr uint32_t kWholeLoadResultId     = 220;
    static constexpr uint32_t kMemberLoadResultId    = 221;
    static constexpr uint32_t kMaximumTestId         = 221;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;

        values_.resize(kMaximumTestId + 1);
        value_meta_.resize(kMaximumTestId + 1);
        types_[kPairTypeId] = ::SPIRVSimulator::Type::Struct(kPairTypeId);
        SetPointerStorageClass(spv::StorageClassFunction);

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_[id];
        });
        EXPECT_CALL(*this, SetValue(_, _, _))
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByResultId(_)).WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, ResolvePointerV(_)).WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{
                backing_.data(),
                GetByteOffset(pointer),
            };
        });
        EXPECT_CALL(*this, WritePointer(_, _))
            .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer, const ::SPIRVSimulator::Value& value) {
                WriteConcreteValue(pointer, value);
                return true;
            });
        EXPECT_CALL(*this, ReadPointer(_))
            .WillRepeatedly(
                [this](const ::SPIRVSimulator::PointerV& pointer) -> std::optional<::SPIRVSimulator::Value> {
                    return ReadConcreteValue(pointer);
                });
        EXPECT_CALL(*this, FindDataSourcesFromResultID(_, _)).WillRepeatedly([](uint32_t, uint32_t* property_flags) {
            if (property_flags != nullptr)
            {
                *property_flags = 0;
            }
            return std::vector<::SPIRVSimulator::DataSourceBits>{};
        });
    }

    ::SPIRVSimulator::PointerV MakePointer(std::vector<uint32_t> path) const
    {
        return ::SPIRVSimulator::PointerV{
            1, 0, kPairTypeId, kWholePointerId, storage_class_, std::move(path),
        };
    }

    void SetPointerStorageClass(uint32_t storage_class)
    {
        storage_class_                  = storage_class;
        values_[kWholePointerId]        = MakePointer({});
        values_[kFirstMemberPointerId]  = MakePointer({ 0 });
        values_[kSecondMemberPointerId] = MakePointer({ 1 });
    }

    static uint64_t GetByteOffset(const ::SPIRVSimulator::PointerV& pointer)
    {
        if (!pointer.idx_path.empty())
        {
            return pointer.idx_path[0] * sizeof(uint64_t);
        }
        return 0;
    }

    static uint32_t GetPointerId(InternalElementwiseTarget target)
    {
        switch (target)
        {
            case InternalElementwiseTarget::WholeObject:
                return kWholePointerId;
            case InternalElementwiseTarget::FirstMember:
                return kFirstMemberPointerId;
            case InternalElementwiseTarget::SecondMember:
                return kSecondMemberPointerId;
        }
        return 0;
    }

    void WriteConcreteValue(const ::SPIRVSimulator::PointerV& pointer, const ::SPIRVSimulator::Value& value)
    {
        if (pointer.idx_path.empty())
        {
            const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value);
            pair_values_[0]       = std::get<uint64_t>(aggregate.elems[0]);
            pair_values_[1]       = std::get<uint64_t>(aggregate.elems[1]);
            return;
        }

        pair_values_[pointer.idx_path[0]] = std::get<uint64_t>(value);
    }

    ::SPIRVSimulator::Value ReadConcreteValue(const ::SPIRVSimulator::PointerV& pointer) const
    {
        if (pointer.idx_path.empty())
        {
            return std::make_shared<::SPIRVSimulator::AggregateV>(std::initializer_list<::SPIRVSimulator::Value>{
                pair_values_[0],
                pair_values_[1],
            });
        }

        return pair_values_[pointer.idx_path[0]];
    }

    void SetPairValue(uint32_t value_id, uint64_t first, uint64_t second)
    {
        values_[value_id] = std::make_shared<::SPIRVSimulator::AggregateV>(
            std::initializer_list<::SPIRVSimulator::Value>{ first, second });
        value_meta_[value_id] = {};
    }

    void SetScalarValue(uint32_t value_id, uint64_t value)
    {
        values_[value_id]     = value;
        value_meta_[value_id] = {};
    }

    static ::SPIRVSimulator::StridedElementSpan MakeSlots()
    {
        return ::SPIRVSimulator::StridedElementSpan{
            true, 0x4000, 16, 8, 2 * sizeof(uint64_t),
        };
    }

    std::weak_ptr<::SPIRVSimulator::ElementwiseSourceState> CreateSource(uint32_t value_id)
    {
        EXPECT_TRUE(TryCreateLazyElementwiseSource(value_id, MakeSlots()));
        return value_meta_[value_id].elementwise_relation.source;
    }

    void ExecuteStore(InternalElementwiseTarget target, uint32_t value_id)
    {
        std::vector<uint32_t>               words = { 0, GetPointerId(target), value_id };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = spv::Op::OpStore,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        Op_Store(instruction);
    }

    void ExecuteLoad(InternalElementwiseTarget target, uint32_t result_id, uint32_t type_id)
    {
        std::vector<uint32_t> words = {
            0,
            type_id,
            result_id,
            GetPointerId(target),
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = spv::Op::OpLoad,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        Op_Load(instruction);
    }

    void ExpectWholeValue(uint32_t result_id, uint64_t expected_first, uint64_t expected_second) const
    {
        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::AggregateV>>(values_[result_id]));
        const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(values_[result_id]);
        ASSERT_EQ(aggregate.elems.size(), 2u);
        EXPECT_EQ(std::get<uint64_t>(aggregate.elems[0]), expected_first);
        EXPECT_EQ(std::get<uint64_t>(aggregate.elems[1]), expected_second);
    }

    void ExpectNoSource(uint32_t result_id) const
    {
        EXPECT_EQ(value_meta_[result_id].elementwise_relation.source, nullptr);
    }

    void RunWholeObjectRestore(uint32_t storage_class)
    {
        SetPointerStorageClass(storage_class);
        SetPairValue(kSourcePairId, 11, 22);
        const std::weak_ptr<::SPIRVSimulator::ElementwiseSourceState> source_storage =
            CreateSource(kSourcePairId);
        const auto* source_identity = source_storage.lock().get();

        ExecuteStore(InternalElementwiseTarget::WholeObject, kSourcePairId);
        value_meta_[kSourcePairId].elementwise_relation = {};
        ASSERT_FALSE(source_storage.expired());

        ExecuteLoad(InternalElementwiseTarget::WholeObject, kWholeLoadResultId, kPairTypeId);

        ExpectWholeValue(kWholeLoadResultId, 11, 22);
        const auto& restored = value_meta_[kWholeLoadResultId].elementwise_relation.source;
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(restored->slots, MakeSlots());
        EXPECT_EQ(restored.get(), source_identity);
        EXPECT_EQ(active_elementwise_sources_.size(), 1u);
    }

    std::array<std::byte, 16> backing_{};
    std::array<uint64_t, 2>   pair_values_{};
    uint32_t storage_class_ = spv::StorageClassFunction;
};

TEST_F(InternalElementwiseSourceTransportTests, FunctionWholeObjectStoreLoadRestoresExactSource)
{
    RunWholeObjectRestore(spv::StorageClassFunction);
}

TEST_F(InternalElementwiseSourceTransportTests, PrivateWholeObjectStoreLoadRestoresExactSource)
{
    RunWholeObjectRestore(spv::StorageClassPrivate);
}

TEST_F(InternalElementwiseSourceTransportTests, WholeObjectOverwriteWithoutSourceDropsOldSource)
{
    SetPairValue(kSourcePairId, 11, 22);
    const std::weak_ptr<::SPIRVSimulator::ElementwiseSourceState> source_storage = CreateSource(kSourcePairId);
    ExecuteStore(InternalElementwiseTarget::WholeObject, kSourcePairId);
    value_meta_[kSourcePairId].elementwise_relation = {};
    ASSERT_FALSE(source_storage.expired());

    SetPairValue(kReplacementPairId, 33, 44);
    ExecuteStore(InternalElementwiseTarget::WholeObject, kReplacementPairId);
    EXPECT_TRUE(source_storage.expired());

    ExecuteLoad(InternalElementwiseTarget::WholeObject, kWholeLoadResultId, kPairTypeId);
    ExpectWholeValue(kWholeLoadResultId, 33, 44);
    ExpectNoSource(kWholeLoadResultId);
}

TEST_F(InternalElementwiseSourceTransportTests, PartialStoreInvalidatesWholeObjectSource)
{
    SetPairValue(kSourcePairId, 11, 22);
    const std::weak_ptr<::SPIRVSimulator::ElementwiseSourceState> source_storage = CreateSource(kSourcePairId);
    ExecuteStore(InternalElementwiseTarget::WholeObject, kSourcePairId);
    value_meta_[kSourcePairId].elementwise_relation = {};

    SetScalarValue(kReplacementScalarId, 99);
    ExecuteStore(InternalElementwiseTarget::FirstMember, kReplacementScalarId);
    EXPECT_TRUE(source_storage.expired());

    ExecuteLoad(InternalElementwiseTarget::WholeObject, kWholeLoadResultId, kPairTypeId);
    ExpectWholeValue(kWholeLoadResultId, 99, 22);
    ExpectNoSource(kWholeLoadResultId);
}

TEST_F(InternalElementwiseSourceTransportTests, SubobjectLoadDoesNotProjectWholeObjectSource)
{
    SetPairValue(kSourcePairId, 11, 22);
    const std::weak_ptr<::SPIRVSimulator::ElementwiseSourceState> source_storage = CreateSource(kSourcePairId);
    ExecuteStore(InternalElementwiseTarget::WholeObject, kSourcePairId);
    value_meta_[kSourcePairId].elementwise_relation = {};

    ExecuteLoad(InternalElementwiseTarget::FirstMember, kMemberLoadResultId, CommonTypes::u64);

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kMemberLoadResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kMemberLoadResultId]), 11u);
    ExpectNoSource(kMemberLoadResultId);
    EXPECT_FALSE(source_storage.expired());
}

TEST_F(InternalElementwiseSourceTransportTests, WorkgroupWholeObjectLoadDoesNotRestoreSource)
{
    SetPointerStorageClass(spv::StorageClassWorkgroup);
    SetPairValue(kSourcePairId, 11, 22);
    CreateSource(kSourcePairId);
    ExecuteStore(InternalElementwiseTarget::WholeObject, kSourcePairId);
    ExecuteLoad(InternalElementwiseTarget::WholeObject, kWholeLoadResultId, kPairTypeId);
    ExpectNoSource(kWholeLoadResultId);
}

TEST_F(InternalElementwiseSourceTransportTests, TaskPayloadWholeObjectLoadDoesNotRestoreSource)
{
    SetPointerStorageClass(spv::StorageClassTaskPayloadWorkgroupEXT);
    SetPairValue(kSourcePairId, 11, 22);
    CreateSource(kSourcePairId);
    ExecuteStore(InternalElementwiseTarget::WholeObject, kSourcePairId);
    ExecuteLoad(InternalElementwiseTarget::WholeObject, kWholeLoadResultId, kPairTypeId);
    ExpectNoSource(kWholeLoadResultId);
}

class DirectElementwiseSourceTransportTests : public SPIRVSimulatorMockBase, public ::testing::Test
{
  protected:
    static constexpr uint32_t kSourceId = 300;
    static constexpr uint32_t kResultId = 301;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;

        values_.resize(kResultId + 1);
        value_meta_.resize(kResultId + 1);
        values_[kSourceId] = uint64_t{ 27 };

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_[id];
        });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByTypeId(_)).WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, GetTypeByResultId(_)).WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>()))
            .WillRepeatedly([this](uint32_t target, uint32_t source) {
                ::SPIRVSimulator::SPIRVSimulator::TransferFlags(target, source);
            });

        ::SPIRVSimulator::DenseValueRangeInfo& source_range = value_meta_[kSourceId].value_range;
        source_range.valid                                  = true;
        source_range.dense_range                            = true;
        source_range.min                                    = 20;
        source_range.max                                    = 35;
        source_range.stride                                 = 1;

        ASSERT_TRUE(TryCreateLazyElementwiseSource(kSourceId, MakeSlots()));
    }

    static ::SPIRVSimulator::StridedElementSpan MakeSlots()
    {
        return ::SPIRVSimulator::StridedElementSpan{
            true, 0x6000, -16, 8, sizeof(uint64_t),
        };
    }

    void ExecuteCopy(spv::Op opcode)
    {
        std::vector<uint32_t> words = {
            static_cast<uint32_t>(opcode),
            CommonTypes::u64,
            kResultId,
            kSourceId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode     = opcode,
            .word_count = static_cast<uint16_t>(words.size()),
            .words      = words,
        };
        ASSERT_TRUE(ExecuteInstruction(instruction));
    }

    void ExpectExactCopy(spv::Op opcode)
    {
        const auto source_state = value_meta_[kSourceId].elementwise_relation.source;
        ASSERT_NE(source_state, nullptr);
        ASSERT_EQ(active_elementwise_sources_.size(), 1u);

        ExecuteCopy(opcode);

        ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kResultId]));
        EXPECT_EQ(std::get<uint64_t>(values_[kResultId]), 27u);
        const auto& result_source = value_meta_[kResultId].elementwise_relation.source;
        ASSERT_NE(result_source, nullptr);
        EXPECT_EQ(result_source->slots, MakeSlots());
        EXPECT_EQ(result_source, source_state);
        EXPECT_EQ(active_elementwise_sources_.size(), 1u);

        const auto& source_range = value_meta_[kSourceId].value_range;
        const auto& result_range = value_meta_[kResultId].value_range;
        EXPECT_EQ(result_range.valid, source_range.valid);
        EXPECT_EQ(result_range.dense_range, source_range.dense_range);
        EXPECT_EQ(result_range.min, source_range.min);
        EXPECT_EQ(result_range.max, source_range.max);
        EXPECT_EQ(result_range.stride, source_range.stride);
    }
};

TEST_F(DirectElementwiseSourceTransportTests, CopyObjectSharesExactElementwiseSource)
{
    ExpectExactCopy(spv::Op::OpCopyObject);
}

TEST_F(DirectElementwiseSourceTransportTests, CopyLogicalSharesExactElementwiseSource)
{
    ExpectExactCopy(spv::Op::OpCopyLogical);
}

class UnsupportedElementwiseSourceTransportTests : public SPIRVSimulatorMockBase, public ::testing::Test
{
  protected:
    static constexpr uint32_t kSource0Id   = 400;
    static constexpr uint32_t kSource1Id   = 401;
    static constexpr uint32_t kConditionId = 402;
    static constexpr uint32_t kResultId    = 403;

    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        verbose_             = false;

        values_.resize(kResultId + 1);
        value_meta_.resize(kResultId + 1);

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

    static ::SPIRVSimulator::StridedElementSpan MakeSlots(uint64_t address)
    {
        return ::SPIRVSimulator::StridedElementSpan{
            true, address, 8, 4, sizeof(uint64_t),
        };
    }

    void SetTypedValue(uint32_t id, uint32_t type_id, ::SPIRVSimulator::Value value)
    {
        values_[id]         = std::move(value);
        value_type_ids_[id] = type_id;
    }

    void SetSource(uint32_t id, uint64_t address)
    {
        value_meta_[id].elementwise_relation.source =
            std::make_shared<::SPIRVSimulator::ElementwiseSourceState>(MakeSlots(address));
    }

    void SetStaleResultSource()
    {
        SetSource(kResultId, 0xf000);
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

    void ExpectNoResultRelation() const
    {
        EXPECT_EQ(value_meta_[kResultId].elementwise_relation.source, nullptr);
    }

    ::SPIRVSimulator::UnorderedMap<uint32_t, uint32_t> value_type_ids_;
};

TEST_F(UnsupportedElementwiseSourceTransportTests, PhiDoesNotPropagateElementwiseSource)
{
    constexpr uint32_t kSelectedBlock = 500;
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 11 });
    SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});
    SetSource(kSource0Id, 0x7000);
    SetStaleResultSource();
    prev_block_id_ = kSelectedBlock;

    Execute(spv::Op::OpPhi, { CommonTypes::u64, kResultId, kSource0Id, kSelectedBlock });

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kResultId]), 11u);
    ExpectNoResultRelation();
}

TEST_F(UnsupportedElementwiseSourceTransportTests, SelectDoesNotPropagateElementwiseSource)
{
    SetTypedValue(kConditionId, CommonTypes::boolean, uint64_t{ 1 });
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 11 });
    SetTypedValue(kSource1Id, CommonTypes::u64, uint64_t{ 22 });
    SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});
    SetSource(kSource0Id, 0x7000);
    SetStaleResultSource();

    Execute(spv::Op::OpSelect, { CommonTypes::u64, kResultId, kConditionId, kSource0Id, kSource1Id });

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kResultId]), 11u);
    ExpectNoResultRelation();
}

TEST_F(UnsupportedElementwiseSourceTransportTests, CompositeExtractDoesNotProjectElementwiseSource)
{
    SetTypedValue(kSource0Id,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 11, 22 }));
    SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});
    SetSource(kSource0Id, 0x7000);
    SetStaleResultSource();

    Execute(spv::Op::OpCompositeExtract, { CommonTypes::u64, kResultId, kSource0Id, 1 });

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kResultId]), 22u);
    ExpectNoResultRelation();
}

TEST_F(UnsupportedElementwiseSourceTransportTests, CompositeInsertDoesNotMergeElementwiseSource)
{
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 99 });
    SetTypedValue(kSource1Id,
                  CommonTypes::uvec2,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 11, 22 }));
    SetTypedValue(kResultId, CommonTypes::uvec2, std::monostate{});
    SetSource(kSource1Id, 0x7000);
    SetStaleResultSource();

    Execute(spv::Op::OpCompositeInsert, { CommonTypes::uvec2, kResultId, kSource0Id, kSource1Id, 0 });

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kResultId]));
    const auto& result = *std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(values_[kResultId]);
    ASSERT_EQ(result.elems.size(), 2u);
    EXPECT_EQ(std::get<uint64_t>(result.elems[0]), 99u);
    EXPECT_EQ(std::get<uint64_t>(result.elems[1]), 22u);
    ExpectNoResultRelation();
}

TEST_F(UnsupportedElementwiseSourceTransportTests, IntegerArithmeticDoesNotPropagateElementwiseSource)
{
    SetTypedValue(kSource0Id, CommonTypes::u64, uint64_t{ 10 });
    SetTypedValue(kSource1Id, CommonTypes::u64, uint64_t{ 20 });
    SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});
    SetSource(kSource0Id, 0x7000);
    SetStaleResultSource();

    Execute(spv::Op::OpIAdd, { CommonTypes::u64, kResultId, kSource0Id, kSource1Id });

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kResultId]), 30u);
    ExpectNoResultRelation();
}
