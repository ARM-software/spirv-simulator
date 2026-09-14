#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "spirv-tools/libspirv.hpp"
#include "spirv_simulator.hpp"

namespace
{
constexpr uint32_t kFirstScalarId = 10;
constexpr uint32_t kSecondScalarId = 11;
constexpr uint32_t kFalseId = 12;
constexpr uint32_t kTrueId = 13;
constexpr uint32_t kCompositeId = 14;
constexpr uint32_t kSumId = 15;
constexpr uint32_t kUndefId = 16;
constexpr uint32_t kPlainScalarId = 17;
constexpr uint32_t kPlainFalseId = 18;
constexpr uint32_t kPlainTrueId = 19;
constexpr uint32_t kNullId = 22;
constexpr uint32_t kUndefSumId = 24;
constexpr uint64_t kUncertainFlags = SPS_FLAG_IS_ARBITRARY | SPS_FLAG_UNINITIALIZED;

std::vector<uint32_t> AssembleSpecializationModule()
{
    static constexpr const char* module_text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %20 "main"
OpExecutionMode %20 LocalSize 1 1 1
OpDecorate %10 SpecId 0
OpDecorate %11 SpecId 1
OpDecorate %12 SpecId 2
OpDecorate %13 SpecId 3
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpTypeBool
%5 = OpTypeVector %3 2
%10 = OpSpecConstant %3 10
%11 = OpSpecConstant %3 20
%12 = OpSpecConstantFalse %4
%13 = OpSpecConstantTrue %4
%14 = OpSpecConstantComposite %5 %10 %11
%15 = OpSpecConstantOp %3 IAdd %10 %11
%16 = OpUndef %3
%17 = OpConstant %3 7
%18 = OpConstantFalse %4
%19 = OpConstantTrue %4
%22 = OpConstantNull %3
%24 = OpSpecConstantOp %3 IAdd %10 %16
%20 = OpFunction %1 None %2
%21 = OpLabel
OpReturn
OpFunctionEnd
)";

    spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_0);
    std::vector<uint32_t> program_words;
    if (!tools.Assemble(module_text, &program_words, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS))
    {
        throw std::runtime_error("Failed to assemble specialization constant test module");
    }
    return program_words;
}

class SpecializationConstantTestSimulator : public ::SPIRVSimulator::SPIRVSimulator
{
  public:
    using SPIRVSimulator::SPIRVSimulator;

    const ::SPIRVSimulator::Value& GetResultValue(uint32_t result_id) const
    {
        return GetValue(result_id);
    }

    uint64_t GetResultFlags(uint32_t result_id) const
    {
        return GetFlags(result_id);
    }

    ::SPIRVSimulator::InvocationDependence GetResultInvocationDependence(uint32_t result_id) const
    {
        return value_meta_[result_id].invocation_dependence;
    }

    const ::SPIRVSimulator::ValueMetadata& GetResultMetadata(uint32_t result_id) const
    {
        return value_meta_[result_id];
    }

    bool GetFixedNonNegativeIntegerScalar(uint32_t result_id, uint64_t& out_value) const
    {
        return TryGetFixedNonNegativeIntegerScalar(result_id, out_value);
    }

    void AddResultFlags(uint32_t result_id, uint64_t flags)
    {
        SetFlags(result_id, flags);
    }

    void SetSingletonRange(uint32_t result_id, uint64_t value)
    {
        ::SPIRVSimulator::DenseValueRangeInfo& range = value_meta_[result_id].value_range;
        range.valid = true;
        range.min = value;
        range.max = value;
    }
};

void ExpectExactUint(const SpecializationConstantTestSimulator& simulator,
                     uint32_t                                   result_id,
                     uint64_t                                   expected_value)
{
    const ::SPIRVSimulator::Value& value = simulator.GetResultValue(result_id);
    ASSERT_TRUE(std::holds_alternative<uint64_t>(value));
    EXPECT_EQ(std::get<uint64_t>(value), expected_value);
    EXPECT_EQ(simulator.GetResultFlags(result_id) & kUncertainFlags, 0U);
}

void ExpectExactUintVector(const SpecializationConstantTestSimulator& simulator,
                           uint32_t                                   result_id,
                           uint64_t                                   expected_first,
                           uint64_t                                   expected_second)
{
    const ::SPIRVSimulator::Value& value = simulator.GetResultValue(result_id);
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(value));

    const auto& vector = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(value);
    ASSERT_NE(vector, nullptr);
    ASSERT_EQ(vector->elems.size(), 2U);
    ASSERT_TRUE(std::holds_alternative<uint64_t>(vector->elems[0]));
    ASSERT_TRUE(std::holds_alternative<uint64_t>(vector->elems[1]));
    EXPECT_EQ(std::get<uint64_t>(vector->elems[0]), expected_first);
    EXPECT_EQ(std::get<uint64_t>(vector->elems[1]), expected_second);
    EXPECT_EQ(simulator.GetResultFlags(result_id) & kUncertainFlags, 0U);
}

class SpecializationConstantTest : public ::testing::Test
{
  protected:
    ::SPIRVSimulator::SimulationData simulation_data_;
    ::SPIRVSimulator::SimulationResults simulation_results_;
};

#ifdef NDEBUG
#define EXPECT_SPECIALIZATION_FAILURE(statement) EXPECT_THROW(statement, std::runtime_error)
#else
#define EXPECT_SPECIALIZATION_FAILURE(statement) EXPECT_DEATH(statement, ".*")
#endif

TEST_F(SpecializationConstantTest, DefaultScalarValuesAreExact)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUint(simulator, kFirstScalarId, 10);
    ExpectExactUint(simulator, kSecondScalarId, 20);
}

TEST_F(SpecializationConstantTest, ExternalScalarValuesAreExact)
{
    struct ScalarOverrides
    {
        uint32_t first;
        uint32_t second;
    };

    const ScalarOverrides overrides = { 100, 200 };
    simulation_data_.specialization_constants = &overrides;
    simulation_data_.specialization_constant_offsets = {
        { 0, offsetof(ScalarOverrides, first) },
        { 1, offsetof(ScalarOverrides, second) },
    };

    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUint(simulator, kFirstScalarId, 100);
    ExpectExactUint(simulator, kSecondScalarId, 200);
}

TEST_F(SpecializationConstantTest, OverrideEntryWithoutDataFails)
{
    simulation_data_.specialization_constant_offsets = { { 0, 0 } };
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();

    auto construct_simulator = [&]() {
        SpecializationConstantTestSimulator simulator(
            program_words, nullptr, &simulation_data_, &simulation_results_);
    };
    EXPECT_SPECIALIZATION_FAILURE(construct_simulator());
}

TEST_F(SpecializationConstantTest, BooleanDefaultsAreExact)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUint(simulator, kFalseId, 0);
    ExpectExactUint(simulator, kTrueId, 1);
}

TEST_F(SpecializationConstantTest, BooleanExternalOverridesAreExact)
{
    struct BoolOverrides
    {
        uint32_t false_value;
        uint32_t true_value;
    };

    const BoolOverrides overrides = { 1, 0 };
    simulation_data_.specialization_constants = &overrides;
    simulation_data_.specialization_constant_offsets = {
        { 2, offsetof(BoolOverrides, false_value) },
        { 3, offsetof(BoolOverrides, true_value) },
    };

    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUint(simulator, kFalseId, 1);
    ExpectExactUint(simulator, kTrueId, 0);
}

TEST_F(SpecializationConstantTest, DefaultCompositeIsExact)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUintVector(simulator, kCompositeId, 10, 20);
}

TEST_F(SpecializationConstantTest, MixedCompositeIsExact)
{
    const uint32_t first_override = 100;
    simulation_data_.specialization_constants = &first_override;
    simulation_data_.specialization_constant_offsets = { { 0, 0 } };

    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUintVector(simulator, kCompositeId, 100, 20);
}

TEST_F(SpecializationConstantTest, SpecConstantOpUsesDefaultOperands)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUint(simulator, kSumId, 30);
}

TEST_F(SpecializationConstantTest, SpecConstantOpUsesExternalAndDefaultOperands)
{
    const uint32_t first_override = 100;
    simulation_data_.specialization_constants = &first_override;
    simulation_data_.specialization_constant_offsets = { { 0, 0 } };

    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectExactUint(simulator, kSumId, 120);
}

TEST_F(SpecializationConstantTest, ExactConstantsAreInvocationInvariant)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_EQ(simulator.GetResultInvocationDependence(kFirstScalarId),
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kFalseId),
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kTrueId),
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kSumId),
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kPlainScalarId),
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kPlainFalseId),
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kPlainTrueId),
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kNullId),
              ::SPIRVSimulator::InvocationDependence::Invariant);

    const ::SPIRVSimulator::ValueMetadata& composite_meta = simulator.GetResultMetadata(kCompositeId);
    EXPECT_EQ(composite_meta.invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_EQ(composite_meta.component_invocation_dependence,
              (std::vector<::SPIRVSimulator::InvocationDependence>{
                  ::SPIRVSimulator::InvocationDependence::Invariant,
                  ::SPIRVSimulator::InvocationDependence::Invariant,
              }));
    EXPECT_TRUE(composite_meta.subobject_invocation_dependence.empty());
}

TEST_F(SpecializationConstantTest, UndefDependentConstantsRemainUnknown)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_EQ(simulator.GetResultInvocationDependence(kUndefId),
              ::SPIRVSimulator::InvocationDependence::Unknown);
    EXPECT_EQ(simulator.GetResultInvocationDependence(kUndefSumId),
              ::SPIRVSimulator::InvocationDependence::Unknown);
}

TEST_F(SpecializationConstantTest, DefaultSpecializationIsFixedNonNegativeScalar)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    uint64_t fixed_value = 0;
    EXPECT_TRUE(simulator.GetFixedNonNegativeIntegerScalar(kFirstScalarId, fixed_value));
    EXPECT_EQ(fixed_value, 10U);
    EXPECT_TRUE(simulator.GetFixedNonNegativeIntegerScalar(kNullId, fixed_value));
    EXPECT_EQ(fixed_value, 0U);
}

TEST_F(SpecializationConstantTest, ExternalSpecializationIsFixedNonNegativeScalar)
{
    const uint32_t first_override = 0;
    simulation_data_.specialization_constants = &first_override;
    simulation_data_.specialization_constant_offsets = { { 0, 0 } };

    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    uint64_t fixed_value = 0;
    EXPECT_TRUE(simulator.GetFixedNonNegativeIntegerScalar(kFirstScalarId, fixed_value));
    EXPECT_EQ(fixed_value, 0U);
}

TEST_F(SpecializationConstantTest, SpecConstantOpResultIsFixedNonNegativeScalar)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    uint64_t fixed_value = 0;
    EXPECT_TRUE(simulator.GetFixedNonNegativeIntegerScalar(kSumId, fixed_value));
    EXPECT_EQ(fixed_value, 30U);
}

TEST_F(SpecializationConstantTest, InexactSpecializationIsNotFixedNonNegativeScalar)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);
    simulator.AddResultFlags(kFirstScalarId, kUncertainFlags);

    uint64_t fixed_value = 0;
    EXPECT_FALSE(simulator.GetFixedNonNegativeIntegerScalar(kFirstScalarId, fixed_value));
}

TEST_F(SpecializationConstantTest, UnknownValueIsNotFixedScalar)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    uint64_t fixed_value = 0;
    EXPECT_FALSE(simulator.GetFixedNonNegativeIntegerScalar(kUndefId, fixed_value));
}

TEST_F(SpecializationConstantTest, SingletonRangeIsNotReclassifiedAsFixedScalar)
{
    const std::vector<uint32_t> program_words = AssembleSpecializationModule();
    SpecializationConstantTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);
    simulator.SetSingletonRange(kFirstScalarId, 10);

    uint64_t fixed_value = 0;
    EXPECT_FALSE(simulator.GetFixedNonNegativeIntegerScalar(kFirstScalarId, fixed_value));
}

#undef EXPECT_SPECIALIZATION_FAILURE
}
