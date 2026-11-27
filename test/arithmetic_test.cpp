#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <cstdint>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class ArithmeticsTests : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

TEST_P(ArithmeticsTests, ParametrizedArithmeticOperation)
{
    const auto& parameters = GetParam();

    std::vector<uint32_t>         words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(_, _, true)).WillOnce(SaveArg<1>(&captured_value));

    this->ExecuteInstruction(inst);

    std::cout << "Captured value was: " << captured_value << std::endl;
    std::cout << "Expected value was: " << parameters.operands.at(0) << std::endl;

    EXPECT_EQ(captured_value, parameters.operands.at(0));
}

std::vector<TestParameters> test_cases = {
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operands_range(0, CommonTypes::i64, std::initializer_list<SPIRVSimulator::Value>{ -1, 1 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<SPIRVSimulator::Value>{ -1.0, 1.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_range(0, CommonTypes::i64, std::initializer_list<SPIRVSimulator::Value>{ 3, 1, 2 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<SPIRVSimulator::Value>{ 3.0, 1.0, 2.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_range(
            0,
            CommonTypes::u64,
            std::initializer_list<SPIRVSimulator::Value>{ std::numeric_limits<uint64_t>::max(), uint64_t(1) })
        .set_operand_at(2, 2, CommonTypes::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_range(0, CommonTypes::i64, std::initializer_list<SPIRVSimulator::Value>{ -1, 1, 2 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<SPIRVSimulator::Value>{ -1.0, 1.0, 2.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_range(0, CommonTypes::i64, std::initializer_list<SPIRVSimulator::Value>{ 4, 2, 2 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<SPIRVSimulator::Value>{ 4.0, 2.0, 2.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operands_range(0, CommonTypes::i64, std::initializer_list<SPIRVSimulator::Value>{ -2, -5, 2 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operands_range(
            0, CommonTypes::u64, std::initializer_list<SPIRVSimulator::Value>{ uint64_t(2), uint64_t(5), uint64_t(2) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<SPIRVSimulator::Value>{ 1.0 / 3.0, 1.0, 3.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUMod)
        .set_operands_range(
            0, CommonTypes::u64, std::initializer_list<SPIRVSimulator::Value>{ uint64_t(1), uint64_t(13), uint64_t(6) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpShiftRightArithmetic)
        .set_operands_range(0,
                            CommonTypes::u64,
                            std::initializer_list<SPIRVSimulator::Value>{
                                uint64_t(0xffffffffffffffff), uint64_t(0x8000000000000000), uint64_t(63) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpShiftRightArithmetic)
        .set_operands_range(0,
                            CommonTypes::uvec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(
                                    std::initializer_list<uint64_t>{ 0xffffffffffffffff, 0xffffffffffffffff }),
                                std::make_shared<SPIRVSimulator::VectorV>(
                                    std::initializer_list<uint64_t>{ 0x8000000000000000, 0x8000000000000000 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 63, 63 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitwiseXor)
        .set_operands_range(0,
                            CommonTypes::u64,
                            std::initializer_list<SPIRVSimulator::Value>{ uint64_t(0x0000000000000000),
                                                                          uint64_t(0x8000000000000000),
                                                                          uint64_t(0x8000000000000000) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitwiseXor)
        .set_operands_range(0,
                            CommonTypes::uvec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(
                                    std::initializer_list<uint64_t>{ 0x8000000000000000, 0xffffffffffffffff }),
                                std::make_shared<SPIRVSimulator::VectorV>(
                                    std::initializer_list<uint64_t>{ 0x8000000000000000, 0x0000000000000000 }),
                                std::make_shared<SPIRVSimulator::VectorV>(
                                    std::initializer_list<uint64_t>{ 0x0000000000000000, 0xffffffffffffffff }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_range(0,
                            CommonTypes::ivec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 4, 4 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpINotEqual)
        .set_operand_at(0, uint64_t(1), CommonTypes::boolean)
        .set_operands_range(
            1, CommonTypes::u64, std::initializer_list<SPIRVSimulator::Value>{ uint64_t(13), uint64_t(6) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpINotEqual)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0, 1 }), CommonTypes::bvec2)
        .set_operands_range(1,
                            CommonTypes::ivec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 4 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operand_at(0, uint64_t(1), CommonTypes::u64)
        .set_operand_at(1, double(1.68), CommonTypes::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0, 1 }), CommonTypes::uvec2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -0.68, 1.12 }),
                        CommonTypes::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operand_at(0, int64_t(1), CommonTypes::i64)
        .set_operand_at(1, double(1.68), CommonTypes::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 1 }), CommonTypes::ivec2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -0.68, 1.12 }),
                        CommonTypes::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_range(0,
                            CommonTypes::ivec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_range(0,
                            CommonTypes::vec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 4.0, 4.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_range(0,
                            CommonTypes::vec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_range(0,
                            CommonTypes::ivec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -2, -2 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_range(0,
                            CommonTypes::vec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -2.0, -2.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVectorTimesScalar)
        .set_operands_range(
            0,
            CommonTypes::vec2,
            std::initializer_list<SPIRVSimulator::Value>{
                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -2.0, -2.0 }),
                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }) })
        .set_operand_at(2, 2.0, CommonTypes::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpMatrixTimesVector)
        .set_operands_at(std::initializer_list<uint32_t>{ 0, 2 },
                         CommonTypes::vec2,
                         std::initializer_list<SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 5.0, 11.0 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }) })
        .set_operand_at(
            1,
            std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
            CommonTypes::mat2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpMatrixTimesMatrix)
        .set_operands_range(
            0,
            CommonTypes::mat2,
            std::initializer_list<SPIRVSimulator::Value>{
                std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 7.0, 10.0, 15.0, 22.0 }, 2),
                std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpDot)
        .set_operand_at(0, 8.0, CommonTypes::f64)
        .set_operands_range(1,
                            CommonTypes::vec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }) })
        .build()
};

INSTANTIATE_TEST_SUITE_P(Arithmetics, ArithmeticsTests, ValuesIn(test_cases));

class AtomicArithmeticTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV&),
                (override));
    MOCK_METHOD(bool,
                WritePointer,
                (const ::SPIRVSimulator::PointerV&, const ::SPIRVSimulator::Value&),
                (override));
    MOCK_METHOD(void,
                TransferFlagsFromPointee,
                (uint32_t, const ::SPIRVSimulator::PointerV&),
                (override));
    MOCK_METHOD(void, TransferFlagsToPointee, (uint32_t, uint32_t), (override));
};

TEST_F(AtomicArithmeticTests, AtomicAndReturnsOriginalAndStoresBitwiseResult)
{
    constexpr uint32_t result_id  = 100;
    constexpr uint32_t pointer_id = 101;
    constexpr uint32_t value_id   = 102;

    ::SPIRVSimulator::PointerV pointer{};
    ::SPIRVSimulator::Value    pointer_value = pointer;
    ::SPIRVSimulator::Value    value         = uint64_t(0x0f0f);
    ::SPIRVSimulator::Value    pointee_value = uint64_t(0xff00);
    ::SPIRVSimulator::Value    stored_value  = uint64_t(0x0f00);

    EXPECT_CALL(*this, GetValue(pointer_id)).WillRepeatedly(ReturnRef(pointer_value));
    EXPECT_CALL(*this, GetValue(value_id)).WillRepeatedly(ReturnRef(value));
    EXPECT_CALL(*this, ReadPointer(pointer)).WillOnce(Return(pointee_value));
    EXPECT_CALL(*this, SetValue(result_id, pointee_value, true));
    EXPECT_CALL(*this, TransferFlagsFromPointee(result_id, pointer));
    EXPECT_CALL(*this, WritePointer(pointer, stored_value)).WillOnce(Return(true));
    EXPECT_CALL(*this, TransferFlagsToPointee(pointer_id, value_id));

    std::vector<uint32_t> words = { 0, CommonTypes::u32, result_id, pointer_id, 0, 0, value_id };
    ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAtomicAnd, .word_count = static_cast<uint16_t>(words.size()), .words = words
    };

    EXPECT_TRUE(this->ExecuteInstruction(instruction));
}

TEST_F(AtomicArithmeticTests, AtomicSMaxInterpretsUnsignedTypeAsSigned)
{
    constexpr uint32_t result_id  = 100;
    constexpr uint32_t pointer_id = 101;
    constexpr uint32_t value_id   = 102;

    ::SPIRVSimulator::PointerV pointer{};
    ::SPIRVSimulator::Value    pointer_value = pointer;
    ::SPIRVSimulator::Value    value         = uint64_t(1);
    ::SPIRVSimulator::Value    pointee_value = uint64_t(0xffffffff);

    EXPECT_CALL(*this, GetValue(pointer_id)).WillRepeatedly(ReturnRef(pointer_value));
    EXPECT_CALL(*this, GetValue(value_id)).WillRepeatedly(ReturnRef(value));
    EXPECT_CALL(*this, ReadPointer(pointer)).WillOnce(Return(pointee_value));
    EXPECT_CALL(*this, SetValue(result_id, pointee_value, true));
    EXPECT_CALL(*this, TransferFlagsFromPointee(result_id, pointer));
    EXPECT_CALL(*this, WritePointer(pointer, value)).WillOnce(Return(true));
    EXPECT_CALL(*this, TransferFlagsToPointee(pointer_id, value_id));

    std::vector<uint32_t> words = { 0, CommonTypes::u32, result_id, pointer_id, 0, 0, value_id };
    ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAtomicSMax, .word_count = static_cast<uint16_t>(words.size()), .words = words
    };

    EXPECT_TRUE(this->ExecuteInstruction(instruction));
}

uint8_t TestWorkgroupAxisMask(const ::SPIRVSimulator::InvocationLinearizationInfo& affine)
{
    return affine.valid ? ::SPIRVSimulator::GetInvocationAxisMask(affine.workgroup_coefficients) : 0;
}

uint8_t TestLocalAxisMask(const ::SPIRVSimulator::InvocationLinearizationInfo& affine)
{
    return affine.valid ? ::SPIRVSimulator::GetInvocationAxisMask(affine.local_coefficients) : 0;
}

class RangePropagationTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    void SetUp() override
    {
        // These tests exercise numeric/range fallback without a concrete
        // dispatch domain. Affine-to-range reconstruction is covered by the
        // derived AffineToRangePropagationTests fixture.
        simulation_data_ = nullptr;
    }

    void SetRange(uint32_t result_id,
                  uint64_t min,
                  uint64_t max,
                  uint64_t stride = 1,
                  bool dense_range = true,
                  uint8_t workgroup_mask = 0,
                  uint8_t local_mask = 0)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
        ::SPIRVSimulator::ValueMetadata& meta = value_meta_[result_id];
        meta = {};
        meta.value_range.valid = true;
        meta.value_range.thread_dependent = true;
        meta.value_range.dense_range = dense_range;
        meta.value_range.min = min;
        meta.value_range.max = max;
        meta.value_range.stride = stride;
        meta.value_range.invocation_linearization = MakeTestInvocationAffine(
            static_cast<int64_t>(min), static_cast<int64_t>(stride), workgroup_mask, local_mask);
    }

    void ExpectRange(uint32_t result_id,
                     uint64_t min,
                     uint64_t max,
                     uint64_t stride,
                     bool dense_range,
                     uint8_t workgroup_mask = 0,
                     uint8_t local_mask = 0) const
    {
        const ::SPIRVSimulator::DenseValueRangeInfo& range =
            value_meta_[result_id].value_range;
        EXPECT_TRUE(range.valid);
        EXPECT_TRUE(range.thread_dependent);
        EXPECT_EQ(range.dense_range, dense_range);
        EXPECT_EQ(range.min, min);
        EXPECT_EQ(range.max, max);
        EXPECT_EQ(range.stride, stride);
        EXPECT_EQ(TestWorkgroupAxisMask(range.invocation_linearization), workgroup_mask);
        EXPECT_EQ(TestLocalAxisMask(range.invocation_linearization), local_mask);
    }

    void ExpectAffine(uint32_t result_id,
                      int64_t origin,
                      const std::array<int64_t, 3>& workgroup_coefficients,
                      const std::array<int64_t, 3>& local_coefficients) const
    {
        const ::SPIRVSimulator::InvocationLinearizationInfo& affine =
            value_meta_[result_id].value_range.invocation_linearization;
        ASSERT_TRUE(affine.valid);
        EXPECT_EQ(affine.value_at_dispatch_origin, origin);
        EXPECT_EQ(affine.workgroup_coefficients, workgroup_coefficients);
        EXPECT_EQ(affine.local_coefficients, local_coefficients);
    }

    void SetDefinitionAndDependence(
        uint32_t                                result_id,
        spv::Op                                 opcode,
        ::SPIRVSimulator::InvocationDependence dependence)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
        value_meta_[result_id].invocation_dependence = dependence;
        instructions_.push_back(::SPIRVSimulator::Instruction{ .opcode = opcode });
        SetInstructionIndexForResultId(result_id, instructions_.size() - 1);
    }

    void SetConstantLikeDefinition(uint32_t result_id)
    {
        SetDefinitionAndDependence(
            result_id,
            spv::Op::OpConstant,
            ::SPIRVSimulator::InvocationDependence::Invariant);
    }

    bool IsFixedNonNegativeIntegerScalar(uint32_t result_id,
                                         uint64_t& out_value) const
    {
        return TryGetFixedNonNegativeIntegerScalar(result_id, out_value);
    }

    void PropagateMul(uint32_t result_id, uint32_t lhs_id, uint32_t rhs_id)
    {
        const size_t required_size = std::max({ result_id, lhs_id, rhs_id }) + 1;
        value_meta_.resize(std::max(value_meta_.size(), required_size));
        PropagateBinaryRangeMul(result_id, lhs_id, rhs_id);
    }

    void PropagateAdd(uint32_t result_id, uint32_t lhs_id, uint32_t rhs_id)
    {
        const size_t required_size = std::max({ result_id, lhs_id, rhs_id }) + 1;
        value_meta_.resize(std::max(value_meta_.size(), required_size));
        PropagateBinaryRangeAdd(result_id, lhs_id, rhs_id);
    }

    void PropagateSub(uint32_t result_id, uint32_t lhs_id, uint32_t rhs_id)
    {
        const size_t required_size = std::max({ result_id, lhs_id, rhs_id }) + 1;
        value_meta_.resize(std::max(value_meta_.size(), required_size));
        PropagateBinaryRangeSub(result_id, lhs_id, rhs_id);
    }

    const ::SPIRVSimulator::ValueMetadata& GetMetadata(uint32_t result_id) const
    {
        return value_meta_[result_id];
    }
};

TEST_F(RangePropagationTests, InvariantCopyResultsAreFixedScalars)
{
    constexpr uint32_t copy_object_id = 180;
    constexpr uint32_t copy_logical_id = 181;

    const ::SPIRVSimulator::Type integer_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value copy_object_value = uint64_t{ 7 };
    ::SPIRVSimulator::Value copy_logical_value = uint64_t{ 11 };

    SetDefinitionAndDependence(
        copy_object_id,
        spv::Op::OpCopyObject,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    SetDefinitionAndDependence(
        copy_logical_id,
        spv::Op::OpCopyLogical,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_CALL(*this, GetTypeByResultId(copy_object_id))
        .WillOnce(ReturnRef(integer_type));
    EXPECT_CALL(*this, GetTypeByResultId(copy_logical_id))
        .WillOnce(ReturnRef(integer_type));
    EXPECT_CALL(*this, GetValue(copy_object_id))
        .WillOnce(ReturnRef(copy_object_value));
    EXPECT_CALL(*this, GetValue(copy_logical_id))
        .WillOnce(ReturnRef(copy_logical_value));

    uint64_t fixed_value = 0;
    EXPECT_TRUE(IsFixedNonNegativeIntegerScalar(copy_object_id, fixed_value));
    EXPECT_EQ(fixed_value, 7u);
    EXPECT_TRUE(IsFixedNonNegativeIntegerScalar(copy_logical_id, fixed_value));
    EXPECT_EQ(fixed_value, 11u);
}

TEST_F(RangePropagationTests, UnknownAndVaryingScalarsAreNotFixed)
{
    constexpr uint32_t unknown_id = 182;
    constexpr uint32_t varying_id = 183;

    SetDefinitionAndDependence(
        unknown_id,
        spv::Op::OpCopyObject,
        ::SPIRVSimulator::InvocationDependence::Unknown);
    SetDefinitionAndDependence(
        varying_id,
        spv::Op::OpCopyObject,
        ::SPIRVSimulator::InvocationDependence::Varying);

    uint64_t fixed_value = 0;
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(unknown_id, fixed_value));
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(varying_id, fixed_value));
}

TEST_F(RangePropagationTests, RangeAndInexactFlagsPreventFixedScalarProof)
{
    constexpr uint32_t range_id = 184;
    constexpr uint32_t arbitrary_id = 185;
    constexpr uint32_t uninitialized_id = 186;

    SetDefinitionAndDependence(
        range_id,
        spv::Op::OpConstant,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    value_meta_[range_id].value_range.valid = true;
    SetDefinitionAndDependence(
        arbitrary_id,
        spv::Op::OpConstant,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    value_meta_[arbitrary_id].flags = SPS_FLAG_IS_ARBITRARY;
    SetDefinitionAndDependence(
        uninitialized_id,
        spv::Op::OpConstant,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    value_meta_[uninitialized_id].flags = SPS_FLAG_UNINITIALIZED;

    uint64_t fixed_value = 0;
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(range_id, fixed_value));
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(arbitrary_id, fixed_value));
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(uninitialized_id, fixed_value));
}

TEST_F(RangePropagationTests, ZeroIsFixedButNegativeAndNonIntegerValuesAreNotFixed)
{
    constexpr uint32_t zero_id = 187;
    constexpr uint32_t negative_id = 188;
    constexpr uint32_t float_id = 189;
    constexpr uint32_t signed_zero_id = 190;

    const ::SPIRVSimulator::Type unsigned_integer_type =
        ::SPIRVSimulator::Type::Int(32, false);
    const ::SPIRVSimulator::Type signed_integer_type =
        ::SPIRVSimulator::Type::Int(32, true);
    const ::SPIRVSimulator::Type float_type =
        ::SPIRVSimulator::Type::Float(32);
    ::SPIRVSimulator::Value zero_value = uint64_t{ 0 };
    ::SPIRVSimulator::Value signed_zero_value = int64_t{ 0 };
    ::SPIRVSimulator::Value negative_value = int64_t{ -1 };
    SetDefinitionAndDependence(
        zero_id,
        spv::Op::OpConstant,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    SetDefinitionAndDependence(
        negative_id,
        spv::Op::OpConstant,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    SetDefinitionAndDependence(
        float_id,
        spv::Op::OpConstant,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    SetDefinitionAndDependence(
        signed_zero_id,
        spv::Op::OpConstant,
        ::SPIRVSimulator::InvocationDependence::Invariant);
    EXPECT_CALL(*this, GetTypeByResultId(zero_id))
        .WillOnce(ReturnRef(unsigned_integer_type));
    EXPECT_CALL(*this, GetTypeByResultId(negative_id))
        .WillOnce(ReturnRef(signed_integer_type));
    EXPECT_CALL(*this, GetTypeByResultId(float_id))
        .WillOnce(ReturnRef(float_type));
    EXPECT_CALL(*this, GetTypeByResultId(signed_zero_id))
        .WillOnce(ReturnRef(signed_integer_type));
    EXPECT_CALL(*this, GetValue(zero_id)).WillOnce(ReturnRef(zero_value));
    EXPECT_CALL(*this, GetValue(signed_zero_id))
        .WillOnce(ReturnRef(signed_zero_value));
    EXPECT_CALL(*this, GetValue(negative_id))
        .WillOnce(ReturnRef(negative_value));

    uint64_t fixed_value = 0;
    EXPECT_TRUE(IsFixedNonNegativeIntegerScalar(zero_id, fixed_value));
    EXPECT_EQ(fixed_value, 0u);
    EXPECT_TRUE(IsFixedNonNegativeIntegerScalar(signed_zero_id, fixed_value));
    EXPECT_EQ(fixed_value, 0u);
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(negative_id, fixed_value));
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(float_id, fixed_value));
}

TEST_F(RangePropagationTests, VectorMultiplySkipsScalarRangePropagation)
{
    constexpr uint32_t result_id = 200;
    constexpr uint32_t lhs_id    = 201;
    constexpr uint32_t rhs_id    = 202;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 3);

    SetRange(lhs_id, 0, 63);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateMul(result_id, lhs_id, rhs_id);

    EXPECT_FALSE(GetMetadata(result_id).value_range.valid);
}

TEST_F(RangePropagationTests, ScalarMultiplyPropagatesRange)
{
    constexpr uint32_t result_id = 210;
    constexpr uint32_t lhs_id    = 211;
    constexpr uint32_t rhs_id    = 212;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value      rhs_value   = uint64_t{ 4 };

    SetRange(lhs_id,
             0,
             15,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                0,
                60,
                4,
                true,
                ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 0, { 0, 4, 0 }, { 0, 4, 0 });
}

TEST_F(RangePropagationTests, FixedScalarMultiplyPropagatesRange)
{
    constexpr uint32_t result_id = 220;
    constexpr uint32_t lhs_id    = 221;
    constexpr uint32_t rhs_id    = 222;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value lhs_value = uint64_t{ 4 };

    SetConstantLikeDefinition(lhs_id);
    SetRange(rhs_id,
             0,
             15,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(lhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(lhs_id)).Times(2).WillRepeatedly(ReturnRef(lhs_value));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                0,
                60,
                4,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentX);
    ExpectAffine(result_id, 0, { 4, 0, 0 }, { 4, 0, 0 });
}

TEST_F(RangePropagationTests, RangeMultiplyFixedZeroCanonicalizesDenseSingleton)
{
    constexpr uint32_t result_id = 223;
    constexpr uint32_t lhs_id    = 224;
    constexpr uint32_t rhs_id    = 225;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 0 };

    SetRange(lhs_id,
             2,
             8,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                0,
                0,
                1,
                true);
    ExpectAffine(result_id, 0, { 0, 0, 0 }, { 0, 0, 0 });
}

TEST_F(RangePropagationTests, FixedZeroMultiplyRangeCanonicalizesDenseSingleton)
{
    constexpr uint32_t result_id = 226;
    constexpr uint32_t lhs_id    = 227;
    constexpr uint32_t rhs_id    = 228;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value lhs_value = uint64_t{ 0 };

    SetConstantLikeDefinition(lhs_id);
    SetRange(rhs_id,
             2,
             8,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentZ);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(lhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(lhs_id)).Times(2).WillRepeatedly(ReturnRef(lhs_value));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                0,
                0,
                1,
                true);
    ExpectAffine(result_id, 0, { 0, 0, 0 }, { 0, 0, 0 });
}

TEST_F(RangePropagationTests, SingletonRangeMultiplyCanonicalizesDenseResult)
{
    constexpr uint32_t result_id = 230;
    constexpr uint32_t lhs_id    = 231;
    constexpr uint32_t rhs_id    = 232;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 3 };

    SetRange(lhs_id,
             7,
             7,
             9,
             false,
             ::SPIRVSimulator::kInvocationComponentZ,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                21,
                21,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentZ,
                ::SPIRVSimulator::kInvocationComponentY);
}

TEST_F(RangePropagationTests, TwoNonSingletonRangesKeepCheckedEnvelopeNonDense)
{
    constexpr uint32_t result_id = 240;
    constexpr uint32_t lhs_id    = 241;
    constexpr uint32_t rhs_id    = 242;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(lhs_id,
             1,
             3,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    SetRange(rhs_id,
             4,
             6,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id, 4, 18, 1, false);
    EXPECT_FALSE(value_meta_[result_id].value_range.invocation_linearization.valid);
}

TEST_F(RangePropagationTests, ResultTypeOverflowSkipsMultiplyRangePropagation)
{
    constexpr uint32_t result_id = 250;
    constexpr uint32_t lhs_id    = 251;
    constexpr uint32_t rhs_id    = 252;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 2 };

    SetRange(lhs_id,
             0,
             uint64_t{ 0xffffffff },
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).WillOnce(ReturnRef(rhs_value));

    PropagateMul(result_id, lhs_id, rhs_id);

    EXPECT_FALSE(GetMetadata(result_id).value_range.valid);
}

TEST_F(RangePropagationTests, PixelIndexLinearizationCombinesMultiplyAndAdd)
{
    constexpr uint32_t y_id = 260;
    constexpr uint32_t width_id = 261;
    constexpr uint32_t row_offset_id = 262;
    constexpr uint32_t x_id = 263;
    constexpr uint32_t result_id = 264;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value width_value = uint64_t{ 640 };

    SetRange(y_id,
             0,
             479,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(width_id);
    SetRange(x_id,
             0,
             639,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    EXPECT_CALL(*this, GetTypeByResultId(row_offset_id))
        .WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(width_id))
        .Times(2)
        .WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(width_id)).Times(2).WillRepeatedly(ReturnRef(width_value));
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));

    PropagateMul(row_offset_id, y_id, width_id);
    ExpectRange(row_offset_id,
                0,
                306560,
                640,
                true,
                ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentY);

    PropagateAdd(result_id, row_offset_id, x_id);
    ExpectRange(result_id,
                0,
                307199,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 0, { 1, 640, 0 }, { 1, 640, 0 });
}

TEST_F(RangePropagationTests, PixelIndexLinearizationAcceptsReversedAddOperands)
{
    constexpr uint32_t result_id = 270;
    constexpr uint32_t x_id = 271;
    constexpr uint32_t row_offset_id = 272;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(x_id,
             0,
             639,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    SetRange(row_offset_id,
             0,
             306560,
             640,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, x_id, row_offset_id);

    ExpectRange(result_id,
                0,
                307199,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
}

TEST_F(RangePropagationTests, SingletonRangeAddCanonicalizesDenseTranslation)
{
    constexpr uint32_t result_id = 280;
    constexpr uint32_t lhs_id = 281;
    constexpr uint32_t rhs_id = 282;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 5 };

    SetRange(lhs_id,
             7,
             7,
             9,
             false,
             ::SPIRVSimulator::kInvocationComponentZ,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id))
        .Times(2)
        .WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateAdd(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                12,
                12,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentZ,
                ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 12, { 0, 0, 9 }, { 0, 9, 0 });
}

TEST_F(RangePropagationTests, RangePlusFixedZeroPreservesDenseTranslation)
{
    constexpr uint32_t result_id = 283;
    constexpr uint32_t lhs_id    = 284;
    constexpr uint32_t rhs_id    = 285;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 0 };

    SetRange(lhs_id,
             4,
             10,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateAdd(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                4,
                10,
                2,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentY);
}

TEST_F(RangePropagationTests, FixedZeroPlusRangePreservesDenseTranslation)
{
    constexpr uint32_t result_id = 286;
    constexpr uint32_t lhs_id    = 287;
    constexpr uint32_t rhs_id    = 288;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value lhs_value = uint64_t{ 0 };

    SetConstantLikeDefinition(lhs_id);
    SetRange(rhs_id,
             4,
             10,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentZ);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(lhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(lhs_id)).Times(2).WillRepeatedly(ReturnRef(lhs_value));

    PropagateAdd(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                4,
                10,
                2,
                true,
                ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentZ);
}

TEST_F(RangePropagationTests, PixelIndexLinearizationRejectsRowHoles)
{
    constexpr uint32_t result_id = 290;
    constexpr uint32_t row_offset_id = 291;
    constexpr uint32_t x_id = 292;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(row_offset_id,
             0,
             306560,
             640,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    SetRange(x_id,
             0,
             638,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, row_offset_id, x_id);

    ExpectRange(result_id,
                0,
                307198,
                1,
                false,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 0, { 1, 640, 0 }, { 1, 640, 0 });
}

TEST_F(RangePropagationTests, PixelIndexLinearizationRejectsOverlappingAxes)
{
    constexpr uint32_t result_id = 300;
    constexpr uint32_t row_offset_id = 301;
    constexpr uint32_t x_id = 302;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(row_offset_id,
             0,
             306560,
             640,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    SetRange(x_id,
             0,
             639,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, row_offset_id, x_id);

    ExpectRange(result_id,
                0,
                307199,
                1,
                false,
                ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 0, { 0, 641, 0 }, { 0, 641, 0 });
}

TEST_F(RangePropagationTests, RangeMinusFixedScalarPreservesDenseTranslation)
{
    constexpr uint32_t result_id = 310;
    constexpr uint32_t lhs_id = 311;
    constexpr uint32_t rhs_id = 312;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 3 };

    SetRange(lhs_id,
             10,
             20,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id))
        .Times(2)
        .WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                7,
                17,
                2,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 7, { 2, 0, 0 }, { 0, 2, 0 });
}

TEST_F(RangePropagationTests, RangeMinusFixedZeroPreservesDenseTranslation)
{
    constexpr uint32_t result_id = 313;
    constexpr uint32_t lhs_id    = 314;
    constexpr uint32_t rhs_id    = 315;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 0 };

    SetRange(lhs_id,
             4,
             10,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentY);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                4,
                10,
                2,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentY);
}

TEST_F(RangePropagationTests, FixedZeroMinusNonZeroRangeSkipsPropagation)
{
    constexpr uint32_t result_id = 316;
    constexpr uint32_t lhs_id    = 317;
    constexpr uint32_t rhs_id    = 318;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value lhs_value = uint64_t{ 0 };

    SetConstantLikeDefinition(lhs_id);
    SetRange(rhs_id,
             3,
             9,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentZ);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(lhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(lhs_id)).Times(2).WillRepeatedly(ReturnRef(lhs_value));

    PropagateSub(result_id, lhs_id, rhs_id);

    EXPECT_FALSE(GetMetadata(result_id).value_range.valid);
}

TEST_F(RangePropagationTests, FixedScalarMinusRangePreservesDenseReflection)
{
    constexpr uint32_t result_id = 320;
    constexpr uint32_t lhs_id = 321;
    constexpr uint32_t rhs_id = 322;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value lhs_value = uint64_t{ 20 };

    SetConstantLikeDefinition(lhs_id);
    SetRange(rhs_id,
             3,
             9,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentZ);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(lhs_id))
        .Times(2)
        .WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(lhs_id)).Times(2).WillRepeatedly(ReturnRef(lhs_value));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                11,
                17,
                2,
                true,
                ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentZ);
    ExpectAffine(result_id, 17, { 0, -2, 0 }, { 0, 0, -2 });
}

TEST_F(RangePropagationTests, IndependentRangesNormalizeToDenseSubtraction)
{
    constexpr uint32_t result_id = 330;
    constexpr uint32_t lhs_id = 331;
    constexpr uint32_t rhs_id = 332;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(lhs_id,
             639,
             307199,
             640,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    SetRange(rhs_id,
             0,
             639,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                0,
                307199,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
}

TEST_F(RangePropagationTests, SubtractionNormalizationRejectsRowHoles)
{
    constexpr uint32_t result_id = 340;
    constexpr uint32_t lhs_id = 341;
    constexpr uint32_t rhs_id = 342;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(lhs_id,
             638,
             307198,
             640,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    SetRange(rhs_id,
             0,
             638,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                0,
                307198,
                1,
                false,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 638, { -1, 640, 0 }, { -1, 640, 0 });
}

TEST_F(RangePropagationTests, SubtractionNormalizationRejectsOverlappingAxes)
{
    constexpr uint32_t result_id = 350;
    constexpr uint32_t lhs_id = 351;
    constexpr uint32_t rhs_id = 352;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(lhs_id,
             639,
             307199,
             640,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    SetRange(rhs_id,
             0,
             639,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentY);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                0,
                307199,
                1,
                false,
                ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 639, { 0, 639, 0 }, { 0, 639, 0 });
}

TEST_F(RangePropagationTests, SubtractionUnderflowSkipsRangePropagation)
{
    constexpr uint32_t result_id = 360;
    constexpr uint32_t lhs_id = 361;
    constexpr uint32_t rhs_id = 362;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 20 };

    SetRange(lhs_id,
             0,
             10,
             1,
             true,
             ::SPIRVSimulator::kInvocationComponentX,
             ::SPIRVSimulator::kInvocationComponentX);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id))
        .Times(2)
        .WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateSub(result_id, lhs_id, rhs_id);

    EXPECT_FALSE(GetMetadata(result_id).value_range.valid);
}

TEST_F(RangePropagationTests, AffineOverflowDoesNotInvalidateNumericRange)
{
    constexpr uint32_t result_id = 370;
    constexpr uint32_t lhs_id = 371;
    constexpr uint32_t rhs_id = 372;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(lhs_id, 0, 0, 1, true, ::SPIRVSimulator::kInvocationComponentX);
    SetRange(rhs_id, 0, 0, 1, true, ::SPIRVSimulator::kInvocationComponentX);
    value_meta_[lhs_id].value_range.invocation_linearization.workgroup_coefficients[0] =
        std::numeric_limits<int64_t>::max();
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, lhs_id, rhs_id);

    ExpectRange(result_id, 0, 0, 1, true);
    EXPECT_FALSE(value_meta_[result_id].value_range.invocation_linearization.valid);
}

TEST_F(RangePropagationTests, FixedScalarPlusRangePropagatesCompleteAffine)
{
    constexpr uint32_t result_id = 380;
    constexpr uint32_t lhs_id = 381;
    constexpr uint32_t rhs_id = 382;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value lhs_value = uint64_t{ 5 };

    SetConstantLikeDefinition(lhs_id);
    SetRange(rhs_id,
             4,
             10,
             2,
             true,
             ::SPIRVSimulator::kInvocationComponentY,
             ::SPIRVSimulator::kInvocationComponentZ);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(lhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(lhs_id)).Times(2).WillRepeatedly(ReturnRef(lhs_value));

    PropagateAdd(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                9,
                15,
                2,
                true,
                ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentZ);
    ExpectAffine(result_id, 9, { 0, 2, 0 }, { 0, 0, 2 });
}

TEST_F(RangePropagationTests, SingletonRangesNormalizeToInvariantMultiplyAffine)
{
    constexpr uint32_t result_id = 390;
    constexpr uint32_t lhs_id = 391;
    constexpr uint32_t rhs_id = 392;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(lhs_id, 7, 7, 1, true);
    SetRange(rhs_id, 3, 3, 1, true);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id, 21, 21, 1, true);
    ExpectAffine(result_id, 21, { 0, 0, 0 }, { 0, 0, 0 });
}

TEST_F(RangePropagationTests, DenseNumericRangeDoesNotRequireAffineOperandProof)
{
    constexpr uint32_t result_id = 400;
    constexpr uint32_t lhs_id = 401;
    constexpr uint32_t rhs_id = 402;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 5 };

    SetRange(lhs_id, 4, 10, 2, true);
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).WillOnce(ReturnRef(rhs_value));

    PropagateAdd(result_id, lhs_id, rhs_id);

    ExpectRange(result_id, 9, 15, 2, true);
    EXPECT_FALSE(value_meta_[result_id].value_range.invocation_linearization.valid);
}

TEST_F(RangePropagationTests, DenseNumericSubtractionDoesNotRequireAffineOperandProof)
{
    constexpr uint32_t result_id = 403;
    constexpr uint32_t lhs_id = 404;
    constexpr uint32_t rhs_id = 405;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value lhs_value = uint64_t{ 20 };

    SetConstantLikeDefinition(lhs_id);
    SetRange(rhs_id, 3, 9, 2, true);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(lhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(lhs_id)).Times(2).WillRepeatedly(ReturnRef(lhs_value));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id, 11, 17, 2, true);
    EXPECT_FALSE(value_meta_[result_id].value_range.invocation_linearization.valid);
}

TEST_F(RangePropagationTests, AffineSubOverflowDoesNotInvalidateNumericRange)
{
    constexpr uint32_t result_id = 410;
    constexpr uint32_t lhs_id = 411;
    constexpr uint32_t rhs_id = 412;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetRange(lhs_id, 0, 0, 1, true, ::SPIRVSimulator::kInvocationComponentX);
    SetRange(rhs_id, 0, 0, 1, true, ::SPIRVSimulator::kInvocationComponentX);
    value_meta_[lhs_id].value_range.invocation_linearization.workgroup_coefficients[0] =
        std::numeric_limits<int64_t>::min();
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, lhs_id, rhs_id);

    ExpectRange(result_id, 0, 0, 1, true);
    EXPECT_FALSE(value_meta_[result_id].value_range.invocation_linearization.valid);
}

TEST_F(RangePropagationTests, AffineMulOverflowDoesNotInvalidateNumericRange)
{
    constexpr uint32_t result_id = 420;
    constexpr uint32_t lhs_id = 421;
    constexpr uint32_t rhs_id = 422;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value rhs_value = uint64_t{ 2 };

    SetRange(lhs_id, 0, 0, 1, true, ::SPIRVSimulator::kInvocationComponentX);
    value_meta_[lhs_id].value_range.invocation_linearization.workgroup_coefficients[0] =
        std::numeric_limits<int64_t>::max();
    SetConstantLikeDefinition(rhs_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(rhs_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(rhs_id)).Times(2).WillRepeatedly(ReturnRef(rhs_value));

    PropagateMul(result_id, lhs_id, rhs_id);

    ExpectRange(result_id, 0, 0, 1, true);
    EXPECT_FALSE(value_meta_[result_id].value_range.invocation_linearization.valid);
}

TEST_F(RangePropagationTests, SingletonOutsideSignedDomainKeepsNumericRange)
{
    constexpr uint32_t result_id = 430;
    constexpr uint32_t lhs_id = 431;
    constexpr uint32_t rhs_id = 432;
    constexpr uint64_t outside_signed_domain =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + uint64_t{ 1 };

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(64, false);

    SetRange(lhs_id, outside_signed_domain, outside_signed_domain, 1, true);
    SetRange(rhs_id, 0, 0, 1, true);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, lhs_id, rhs_id);

    ExpectRange(result_id,
                outside_signed_domain,
                outside_signed_domain,
                1,
                true);
    EXPECT_FALSE(value_meta_[result_id].value_range.invocation_linearization.valid);
}

class AffineToRangePropagationTests : public RangePropagationTests
{
  protected:
    void SetUp() override
    {
        SetDispatchDomain({ 2, 3, 1 }, { 4, 5, 1, true });
    }

    void SetDispatchDomain(
        const ::SPIRVSimulator::ComputeDispatchDimensions& num_workgroups,
        const ::SPIRVSimulator::ComputeLocalSize&          local_size)
    {
        simulation_data_storage_ = {};
        simulation_data_storage_.has_compute_num_workgroups = true;
        simulation_data_storage_.compute_num_workgroups = num_workgroups;
        simulation_data_storage_.compute_local_size = local_size;
        simulation_data_ = &simulation_data_storage_;
        active_compute_local_size_ = local_size;
    }

    void SetAffineRange(
        uint32_t                              result_id,
        uint64_t                              min,
        uint64_t                              max,
        uint64_t                              stride,
        bool                                  dense_range,
        int64_t                               origin,
        const std::array<int64_t, 3>&         workgroup_coefficients,
        const std::array<int64_t, 3>&         local_coefficients)
    {
        SetRange(result_id, min, max, stride, dense_range);
        ::SPIRVSimulator::InvocationLinearizationInfo& affine =
            value_meta_[result_id].value_range.invocation_linearization;
        affine = {};
        affine.valid = true;
        affine.value_at_dispatch_origin = origin;
        affine.workgroup_coefficients = workgroup_coefficients;
        affine.local_coefficients = local_coefficients;
    }

    ::SPIRVSimulator::SimulationData simulation_data_storage_;
};

TEST_F(AffineToRangePropagationTests, PixelIndexMinusItselfProducesDenseZero)
{
    constexpr uint32_t result_id = 440;
    constexpr uint32_t pixel_index_id = 441;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(pixel_index_id,
                   0,
                   119,
                   1,
                   true,
                   0,
                   { 4, 40, 0 },
                   { 1, 8, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, pixel_index_id, pixel_index_id);

    ExpectRange(result_id, 0, 0, 1, true);
    ExpectAffine(result_id, 0, { 0, 0, 0 }, { 0, 0, 0 });
}

TEST_F(AffineToRangePropagationTests, LinearMinusRowBaseCancelsCorrelatedAxis)
{
    constexpr uint32_t result_id = 450;
    constexpr uint32_t linear_id = 451;
    constexpr uint32_t row_base_id = 452;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(linear_id, 0, 119, 1, true, 0, { 4, 40, 0 }, { 1, 8, 0 });
    SetAffineRange(row_base_id, 0, 112, 8, true, 0, { 0, 40, 0 }, { 0, 8, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, linear_id, row_base_id);

    ExpectRange(result_id,
                0,
                7,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentX);
    ExpectAffine(result_id, 0, { 4, 0, 0 }, { 1, 0, 0 });
}

TEST_F(AffineToRangePropagationTests, GlobalXMinusGroupBaseProducesRepeatedLocalXRange)
{
    constexpr uint32_t result_id = 460;
    constexpr uint32_t global_x_id = 461;
    constexpr uint32_t group_base_id = 462;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(global_x_id, 0, 7, 1, true, 0, { 4, 0, 0 }, { 1, 0, 0 });
    SetAffineRange(group_base_id, 0, 4, 4, true, 0, { 4, 0, 0 }, { 0, 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, global_x_id, group_base_id);

    ExpectRange(result_id,
                0,
                3,
                1,
                true,
                0,
                ::SPIRVSimulator::kInvocationComponentX);
    ExpectAffine(result_id, 0, { 0, 0, 0 }, { 1, 0, 0 });
}

TEST_F(AffineToRangePropagationTests, LinearPlusRowBasePublishesExactNonDenseRange)
{
    constexpr uint32_t result_id = 470;
    constexpr uint32_t linear_id = 471;
    constexpr uint32_t row_base_id = 472;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(linear_id, 0, 119, 1, true, 0, { 4, 40, 0 }, { 1, 8, 0 });
    SetAffineRange(row_base_id, 0, 112, 8, true, 0, { 0, 40, 0 }, { 0, 8, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, linear_id, row_base_id);

    ExpectRange(result_id,
                0,
                231,
                1,
                false,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 0, { 4, 80, 0 }, { 1, 16, 0 });
}

TEST_F(AffineToRangePropagationTests, SameAxisAddProducesDenseStrideTwoRange)
{
    constexpr uint32_t result_id = 480;
    constexpr uint32_t global_x_id = 481;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(global_x_id, 0, 7, 1, true, 0, { 4, 0, 0 }, { 1, 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, global_x_id, global_x_id);

    ExpectRange(result_id,
                0,
                14,
                2,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentX);
    ExpectAffine(result_id, 0, { 8, 0, 0 }, { 2, 0, 0 });
}

TEST_F(AffineToRangePropagationTests, NegativeCorrelatedResultDoesNotPublishRange)
{
    constexpr uint32_t result_id = 490;
    constexpr uint32_t group_base_id = 491;
    constexpr uint32_t global_x_id = 492;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(group_base_id, 0, 4, 4, true, 0, { 4, 0, 0 }, { 0, 0, 0 });
    SetAffineRange(global_x_id, 0, 7, 1, true, 0, { 4, 0, 0 }, { 1, 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, group_base_id, global_x_id);

    EXPECT_FALSE(GetMetadata(result_id).value_range.valid);
}

TEST_F(AffineToRangePropagationTests, ResultTypeOverflowDoesNotPublishRange)
{
    constexpr uint32_t result_id = 500;
    constexpr uint32_t operand_id = 501;
    constexpr uint64_t u32_max = std::numeric_limits<uint32_t>::max();

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetDispatchDomain({ 1, 1, 1 }, { 2, 1, 1, true });
    SetAffineRange(operand_id,
                   0,
                   u32_max,
                   u32_max,
                   true,
                   0,
                   { 0, 0, 0 },
                   { static_cast<int64_t>(u32_max), 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, operand_id, operand_id);

    EXPECT_FALSE(GetMetadata(result_id).value_range.valid);
}

TEST_F(AffineToRangePropagationTests, SignedBoundsOverflowFallsBackToUnsignedNumericRange)
{
    constexpr uint32_t result_id = 510;
    constexpr uint32_t operand_id = 511;
    constexpr uint64_t coefficient =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2;
    constexpr uint64_t operand_max = coefficient * 2;
    constexpr uint64_t result_max = operand_max * 2;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(64, false);

    SetDispatchDomain({ 1, 1, 1 }, { 3, 1, 1, true });
    SetAffineRange(operand_id,
                   0,
                   operand_max,
                   coefficient,
                   true,
                   0,
                   { 0, 0, 0 },
                   { static_cast<int64_t>(coefficient), 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, operand_id, operand_id);

    ExpectRange(result_id,
                0,
                result_max,
                1,
                false,
                0,
                ::SPIRVSimulator::kInvocationComponentX);
    ExpectAffine(result_id,
                 0,
                 { 0, 0, 0 },
                 { static_cast<int64_t>(coefficient * 2), 0, 0 });
}

TEST_F(AffineToRangePropagationTests, IndependentAxesAddUsesAffineFirst)
{
    constexpr uint32_t result_id = 520;
    constexpr uint32_t row_base_id = 521;
    constexpr uint32_t global_x_id = 522;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(row_base_id, 0, 112, 8, true, 0, { 0, 40, 0 }, { 0, 8, 0 });
    SetAffineRange(global_x_id, 0, 7, 1, true, 0, { 4, 0, 0 }, { 1, 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateAdd(result_id, row_base_id, global_x_id);

    ExpectRange(result_id,
                0,
                119,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 0, { 4, 40, 0 }, { 1, 8, 0 });
}

TEST_F(AffineToRangePropagationTests, FixedTranslationUsesAffineFirst)
{
    constexpr uint32_t result_id = 530;
    constexpr uint32_t global_x_id = 531;
    constexpr uint32_t fixed_id = 532;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value fixed_value = uint64_t{ 5 };

    SetAffineRange(global_x_id, 0, 7, 1, true, 0, { 4, 0, 0 }, { 1, 0, 0 });
    SetConstantLikeDefinition(fixed_id);
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(fixed_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(fixed_id)).Times(2).WillRepeatedly(ReturnRef(fixed_value));

    PropagateAdd(result_id, global_x_id, fixed_id);

    ExpectRange(result_id,
                5,
                12,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentX);
    ExpectAffine(result_id, 5, { 4, 0, 0 }, { 1, 0, 0 });
}

TEST_F(AffineToRangePropagationTests, FixedReflectionUsesAffineFirst)
{
    constexpr uint32_t result_id = 540;
    constexpr uint32_t fixed_id = 541;
    constexpr uint32_t global_x_id = 542;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);
    ::SPIRVSimulator::Value fixed_value = uint64_t{ 7 };

    SetConstantLikeDefinition(fixed_id);
    SetAffineRange(global_x_id, 0, 7, 1, true, 0, { 4, 0, 0 }, { 1, 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(fixed_id)).Times(2).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetValue(fixed_id)).Times(2).WillRepeatedly(ReturnRef(fixed_value));

    PropagateSub(result_id, fixed_id, global_x_id);

    ExpectRange(result_id,
                0,
                7,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX,
                ::SPIRVSimulator::kInvocationComponentX);
    ExpectAffine(result_id, 7, { -4, 0, 0 }, { -1, 0, 0 });
}

TEST_F(AffineToRangePropagationTests, IndependentSubtractionUsesAffineFirst)
{
    constexpr uint32_t result_id = 550;
    constexpr uint32_t row_end_id = 551;
    constexpr uint32_t global_x_id = 552;

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Int(32, false);

    SetAffineRange(row_end_id, 7, 119, 8, true, 7, { 0, 40, 0 }, { 0, 8, 0 });
    SetAffineRange(global_x_id, 0, 7, 1, true, 0, { 4, 0, 0 }, { 1, 0, 0 });
    EXPECT_CALL(*this, GetTypeByResultId(result_id)).WillOnce(ReturnRef(result_type));

    PropagateSub(result_id, row_end_id, global_x_id);

    ExpectRange(result_id,
                0,
                119,
                1,
                true,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY,
                ::SPIRVSimulator::kInvocationComponentX |
                    ::SPIRVSimulator::kInvocationComponentY);
    ExpectAffine(result_id, 7, { -4, 40, 0 }, { -1, 8, 0 });
}

class ElementwiseIntegerInvocationDependenceTests : public SPIRVSimulatorMockBase, public TestWithParam<spv::Op>
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kLhsId        = 8000;
    static constexpr uint32_t kRhsId        = 8001;
    static constexpr uint32_t kResultId     = 8002;
    static constexpr uint32_t kNextResultId = 8003;

    void SetUp() override
    {
        simulation_data_     = nullptr;
        memory_flag_tracker_ = nullptr;
        verbose_             = false;
        values_.resize(kNextResultId + 1);
        value_meta_.resize(kNextResultId + 1);

        EXPECT_CALL(*this, GetValue(_)).WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
            return values_.at(id);
        });
        EXPECT_CALL(*this, GetTypeByResultId(_)).WillRepeatedly([this](uint32_t id) -> const ::SPIRVSimulator::Type& {
            return types_.at(value_type_ids_.at(id));
        });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>())).Times(AnyNumber());
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint64_t>())).Times(AnyNumber());
        EXPECT_CALL(*this, HasFlags(_, _)).WillRepeatedly(Return(false));
    }

    void SetDispatchDomain(
        const ::SPIRVSimulator::ComputeDispatchDimensions& num_workgroups,
        const ::SPIRVSimulator::ComputeLocalSize&          local_size)
    {
        simulation_data_storage_ = {};
        simulation_data_storage_.has_compute_num_workgroups = true;
        simulation_data_storage_.compute_num_workgroups = num_workgroups;
        simulation_data_ = &simulation_data_storage_;
        active_compute_local_size_ = local_size;
    }

    void SetTypedValue(uint32_t id, uint32_t type_id, ::SPIRVSimulator::Value value)
    {
        values_[id]         = std::move(value);
        value_type_ids_[id] = type_id;
    }

    void ExecuteBinary(uint32_t result_id, uint32_t lhs_id, uint32_t rhs_id, uint32_t type_id)
    {
        const std::vector<uint32_t> words{ static_cast<uint32_t>(GetParam()), type_id, result_id, lhs_id, rhs_id };
        const ::SPIRVSimulator::Instruction instruction{ .opcode     = GetParam(),
                                                         .word_count = static_cast<uint16_t>(words.size()),
                                                         .words      = words };
        ASSERT_TRUE(ExecuteInstruction(instruction));
    }

    bool IsFixedNonNegativeIntegerScalar(uint32_t result_id, uint64_t& out_value) const
    {
        return TryGetFixedNonNegativeIntegerScalar(result_id, out_value);
    }

    static Dependence ExpectedCombinedDependence(Dependence lhs, Dependence rhs)
    {
        if (lhs == Dependence::Varying || rhs == Dependence::Varying)
        {
            return Dependence::Varying;
        }
        if (lhs == Dependence::Invariant && rhs == Dependence::Invariant)
        {
            return Dependence::Invariant;
        }
        return Dependence::Unknown;
    }

    uint64_t ExpectedValue(uint64_t lhs, uint64_t rhs) const
    {
        switch (GetParam())
        {
            case spv::Op::OpIAdd:
                return lhs + rhs;
            case spv::Op::OpISub:
                return lhs - rhs;
            case spv::Op::OpIMul:
                return lhs * rhs;
            default:
                return 0;
        }
    }

    ::SPIRVSimulator::UnorderedMap<uint32_t, uint32_t> value_type_ids_;
    ::SPIRVSimulator::SimulationData                   simulation_data_storage_;
};

TEST_P(ElementwiseIntegerInvocationDependenceTests, ScalarCombinesAllInvocationDependenceStates)
{
    constexpr Dependence states[]{
        Dependence::Unknown,
        Dependence::Invariant,
        Dependence::Varying,
    };

    SetTypedValue(kLhsId, CommonTypes::u64, uint64_t{ 8 });
    SetTypedValue(kRhsId, CommonTypes::u64, uint64_t{ 2 });
    SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});

    for (const Dependence lhs_dependence : states)
    {
        for (const Dependence rhs_dependence : states)
        {
            value_meta_[kLhsId]                                    = {};
            value_meta_[kRhsId]                                    = {};
            value_meta_[kLhsId].invocation_dependence              = lhs_dependence;
            value_meta_[kRhsId].invocation_dependence              = rhs_dependence;
            value_meta_[kResultId].component_invocation_dependence = { Dependence::Varying };
            value_meta_[kResultId].subobject_invocation_dependence = { { { 0 }, Dependence::Varying } };

            ExecuteBinary(kResultId, kLhsId, kRhsId, CommonTypes::u64);

            const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
            EXPECT_EQ(result_meta.invocation_dependence, ExpectedCombinedDependence(lhs_dependence, rhs_dependence));
            EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
            EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
        }
    }
}

TEST_P(ElementwiseIntegerInvocationDependenceTests, VectorCombinesInvocationDependencePerComponent)
{
    SetTypedValue(kLhsId,
                  CommonTypes::uvec3,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 8, 9, 10 }));
    SetTypedValue(kRhsId,
                  CommonTypes::uvec3,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 3, 2 }));
    SetTypedValue(kResultId, CommonTypes::uvec3, std::monostate{});

    value_meta_[kLhsId].invocation_dependence           = Dependence::Varying;
    value_meta_[kLhsId].component_invocation_dependence = {
        Dependence::Varying,
        Dependence::Invariant,
        Dependence::Unknown,
    };
    value_meta_[kRhsId].invocation_dependence           = Dependence::Varying;
    value_meta_[kRhsId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
        Dependence::Varying,
    };

    ExecuteBinary(kResultId, kLhsId, kRhsId, CommonTypes::uvec3);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Varying, Dependence::Invariant, Dependence::Varying }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

TEST_P(ElementwiseIntegerInvocationDependenceTests, MissingVectorComponentProofFallsBackConservatively)
{
    SetTypedValue(kLhsId,
                  CommonTypes::uvec3,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 8, 9, 10 }));
    SetTypedValue(kRhsId,
                  CommonTypes::uvec3,
                  std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 3, 2 }));
    SetTypedValue(kResultId, CommonTypes::uvec3, std::monostate{});

    value_meta_[kLhsId].invocation_dependence           = Dependence::Varying;
    value_meta_[kLhsId].component_invocation_dependence = { Dependence::Invariant };
    value_meta_[kRhsId].invocation_dependence           = Dependence::Invariant;

    ExecuteBinary(kResultId, kLhsId, kRhsId, CommonTypes::uvec3);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Unknown);
    EXPECT_EQ(result_meta.component_invocation_dependence,
              (std::vector<Dependence>{ Dependence::Invariant, Dependence::Unknown, Dependence::Unknown }));
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

TEST_P(ElementwiseIntegerInvocationDependenceTests, InvocationDependenceCoexistsWithRangeMetadata)
{
    SetDispatchDomain({ 2, 1, 1 }, { 1, 2, 1, true });

    SetTypedValue(kLhsId, CommonTypes::u64, uint64_t{ 4 });
    SetTypedValue(kRhsId, CommonTypes::u64, uint64_t{ 2 });
    SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});

    ::SPIRVSimulator::DenseValueRangeInfo& lhs_range                      = value_meta_[kLhsId].value_range;
    lhs_range.valid                                                       = true;
    lhs_range.thread_dependent                                            = true;
    lhs_range.dense_range                                                 = true;
    lhs_range.min                                                         = 4;
    lhs_range.max                                                         = 7;
    lhs_range.stride                                                      = 1;
    lhs_range.invocation_linearization.valid                              = true;
    lhs_range.invocation_linearization.value_at_dispatch_origin           = 4;
    lhs_range.invocation_linearization.workgroup_coefficients             = { 2, 0, 0 };
    lhs_range.invocation_linearization.local_coefficients                 = { 0, 1, 0 };
    value_meta_[kLhsId].invocation_dependence                             = Dependence::Varying;
    value_meta_[kRhsId].invocation_dependence                             = Dependence::Invariant;

    ExecuteBinary(kResultId, kLhsId, kRhsId, CommonTypes::u64);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
    ASSERT_TRUE(result_meta.value_range.valid);
    EXPECT_TRUE(result_meta.value_range.thread_dependent);
    EXPECT_TRUE(result_meta.value_range.dense_range);
    EXPECT_EQ(TestWorkgroupAxisMask(result_meta.value_range.invocation_linearization),
              ::SPIRVSimulator::kInvocationComponentX);
    EXPECT_EQ(TestLocalAxisMask(result_meta.value_range.invocation_linearization),
              ::SPIRVSimulator::kInvocationComponentY);
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Varying);

    switch (GetParam())
    {
        case spv::Op::OpIAdd:
            EXPECT_EQ(result_meta.value_range.min, 6u);
            EXPECT_EQ(result_meta.value_range.max, 9u);
            EXPECT_EQ(result_meta.value_range.stride, 1u);
            break;
        case spv::Op::OpISub:
            EXPECT_EQ(result_meta.value_range.min, 2u);
            EXPECT_EQ(result_meta.value_range.max, 5u);
            EXPECT_EQ(result_meta.value_range.stride, 1u);
            break;
        case spv::Op::OpIMul:
            EXPECT_EQ(result_meta.value_range.min, 8u);
            EXPECT_EQ(result_meta.value_range.max, 14u);
            EXPECT_EQ(result_meta.value_range.stride, 2u);
            break;
        default:
            FAIL();
    }

    uint64_t fixed_value = 0;
    EXPECT_FALSE(IsFixedNonNegativeIntegerScalar(kResultId, fixed_value));
}

TEST_P(ElementwiseIntegerInvocationDependenceTests, InvariantArithmeticChainProducesFixedScalar)
{
    SetTypedValue(kLhsId, CommonTypes::u64, uint64_t{ 8 });
    SetTypedValue(kRhsId, CommonTypes::u64, uint64_t{ 2 });
    SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});
    SetTypedValue(kNextResultId, CommonTypes::u64, std::monostate{});
    value_meta_[kLhsId].invocation_dependence = Dependence::Invariant;
    value_meta_[kRhsId].invocation_dependence = Dependence::Invariant;

    ExecuteBinary(kResultId, kLhsId, kRhsId, CommonTypes::u64);
    ASSERT_EQ(value_meta_[kResultId].invocation_dependence, Dependence::Invariant);

    ExecuteBinary(kNextResultId, kResultId, kRhsId, CommonTypes::u64);
    EXPECT_EQ(value_meta_[kNextResultId].invocation_dependence, Dependence::Invariant);

    uint64_t fixed_value = 0;
    EXPECT_TRUE(IsFixedNonNegativeIntegerScalar(kNextResultId, fixed_value));
    EXPECT_EQ(fixed_value, ExpectedValue(ExpectedValue(8, 2), 2));
}

TEST_P(ElementwiseIntegerInvocationDependenceTests, CooperativeMatrixRemainsUnknown)
{
    SetTypedValue(kLhsId,
                  CommonTypes::coopMatUA2,
                  std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 8, 9, 10, 11 }, 2));
    SetTypedValue(kRhsId,
                  CommonTypes::coopMatUB2,
                  std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 2, 2, 2, 2 }, 2));
    SetTypedValue(kResultId, CommonTypes::coopMatUAcc2, std::monostate{});
    value_meta_[kLhsId].invocation_dependence              = Dependence::Invariant;
    value_meta_[kRhsId].invocation_dependence              = Dependence::Invariant;
    value_meta_[kResultId].invocation_dependence           = Dependence::Invariant;
    value_meta_[kResultId].component_invocation_dependence = { Dependence::Invariant };
    value_meta_[kResultId].subobject_invocation_dependence = { { { 0 }, Dependence::Invariant } };

    ExecuteBinary(kResultId, kLhsId, kRhsId, CommonTypes::coopMatUAcc2);

    const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
    EXPECT_EQ(result_meta.invocation_dependence, Dependence::Unknown);
    EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
    EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
}

INSTANTIATE_TEST_SUITE_P(FullyDefinedIntegerArithmetic,
                         ElementwiseIntegerInvocationDependenceTests,
                         Values(spv::Op::OpIAdd, spv::Op::OpISub, spv::Op::OpIMul));

class ElementwiseBitwiseInvocationDependenceTests : public SPIRVSimulatorMockBase,
                                                    public TestWithParam<spv::Op>
{
  protected:
    void ExecuteBitwise(const ::SPIRVSimulator::Instruction& instruction)
    {
        switch (instruction.opcode)
        {
            case spv::Op::OpBitwiseAnd:
                Op_BitwiseAnd(instruction);
                break;
            case spv::Op::OpBitwiseOr:
                Op_BitwiseOr(instruction);
                break;
            case spv::Op::OpBitwiseXor:
                Op_BitwiseXor(instruction);
                break;
            default:
                FAIL() << "Unsupported bitwise opcode";
        }
    }
};

TEST_P(ElementwiseBitwiseInvocationDependenceTests, VaryingAndInvariantProducesVaryingResult)
{
    constexpr uint32_t lhs_id    = 8050;
    constexpr uint32_t rhs_id    = 8051;
    constexpr uint32_t result_id = 8052;
    const spv::Op opcode = GetParam();

    values_.resize(result_id + 1);
    value_meta_.resize(result_id + 1);
    values_[lhs_id] = uint64_t{ 3 };
    values_[rhs_id] = uint64_t{ 1 };
    value_meta_[lhs_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Varying;
    value_meta_[rhs_id].invocation_dependence =
        ::SPIRVSimulator::InvocationDependence::Invariant;

    EXPECT_CALL(*this, GetValue(lhs_id)).WillOnce(ReturnRef(values_[lhs_id]));
    EXPECT_CALL(*this, GetValue(rhs_id)).WillOnce(ReturnRef(values_[rhs_id]));
    EXPECT_CALL(*this, GetTypeByResultId(result_id))
        .WillOnce(ReturnRef(types_.at(CommonTypes::u64)));
    EXPECT_CALL(*this, SetValue(result_id, _, true))
        .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
            ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
        });
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(lhs_id))).Times(1);
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(rhs_id))).Times(1);

    const std::vector<uint32_t> words{
        static_cast<uint32_t>(opcode),
        CommonTypes::u64,
        result_id,
        lhs_id,
        rhs_id,
    };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = opcode,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };

    ExecuteBitwise(instruction);

    uint64_t expected_value = 0;
    switch (opcode)
    {
        case spv::Op::OpBitwiseAnd:
            expected_value = 1;
            break;
        case spv::Op::OpBitwiseOr:
            expected_value = 3;
            break;
        case spv::Op::OpBitwiseXor:
            expected_value = 2;
            break;
        default:
            FAIL() << "Unsupported bitwise opcode";
    }

    EXPECT_EQ(std::get<uint64_t>(values_[result_id]), expected_value);
    EXPECT_EQ(value_meta_[result_id].invocation_dependence,
              ::SPIRVSimulator::InvocationDependence::Varying);
    EXPECT_FALSE(value_meta_[result_id].value_range.valid);
}

INSTANTIATE_TEST_SUITE_P(BitwiseIntegerOperations,
                         ElementwiseBitwiseInvocationDependenceTests,
                         Values(spv::Op::OpBitwiseAnd,
                                spv::Op::OpBitwiseOr,
                                spv::Op::OpBitwiseXor));

class ElementwiseUnsignedDivModInvocationDependenceTests : public SPIRVSimulatorMockBase, public TestWithParam<spv::Op>
{
  protected:
    using Dependence = ::SPIRVSimulator::InvocationDependence;

    static constexpr uint32_t kDividendId = 8100;
    static constexpr uint32_t kDivisorId  = 8101;
    static constexpr uint32_t kResultId   = 8102;

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
        EXPECT_CALL(*this, SetValue(_, _, true))
            .WillRepeatedly([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>())).Times(AnyNumber());
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint64_t>())).Times(AnyNumber());
        EXPECT_CALL(*this, HasFlags(_, _)).WillRepeatedly(Return(false));
    }

    void SetTypedValue(uint32_t id, uint32_t type_id, ::SPIRVSimulator::Value value)
    {
        values_[id]         = std::move(value);
        value_type_ids_[id] = type_id;
    }

    void SetScalarOperands(uint64_t divisor)
    {
        SetTypedValue(kDividendId, CommonTypes::u64, uint64_t{ 24 });
        SetTypedValue(kDivisorId, CommonTypes::u64, divisor);
        SetTypedValue(kResultId, CommonTypes::u64, std::monostate{});
    }

    void SetVectorOperands(std::initializer_list<uint64_t> dividend, std::initializer_list<uint64_t> divisor)
    {
        SetTypedValue(kDividendId, CommonTypes::uvec3, std::make_shared<::SPIRVSimulator::VectorV>(dividend));
        SetTypedValue(kDivisorId, CommonTypes::uvec3, std::make_shared<::SPIRVSimulator::VectorV>(divisor));
        SetTypedValue(kResultId, CommonTypes::uvec3, std::monostate{});
    }

    void ExecuteBinary(uint32_t type_id)
    {
        const std::vector<uint32_t> words{
            static_cast<uint32_t>(GetParam()), type_id, kResultId, kDividendId, kDivisorId
        };
        const ::SPIRVSimulator::Instruction instruction{ .opcode     = GetParam(),
                                                         .word_count = static_cast<uint16_t>(words.size()),
                                                         .words      = words };
        ASSERT_TRUE(ExecuteInstruction(instruction));
    }

    void ExpectScalarDependence(Dependence expected) const
    {
        const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
        EXPECT_EQ(result_meta.invocation_dependence, expected);
        EXPECT_TRUE(result_meta.component_invocation_dependence.empty());
        EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    }

    void ExpectVectorDependence(Dependence                        expected_summary,
                                std::initializer_list<Dependence> expected_components) const
    {
        const ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[kResultId];
        EXPECT_EQ(result_meta.invocation_dependence, expected_summary);
        EXPECT_EQ(result_meta.component_invocation_dependence, std::vector<Dependence>(expected_components));
        EXPECT_TRUE(result_meta.subobject_invocation_dependence.empty());
    }

    ::SPIRVSimulator::UnorderedMap<uint32_t, uint32_t> value_type_ids_;
};

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, ScalarExactInvariantNonZeroDivisorCombinesDependence)
{
    SetScalarOperands(3);
    value_meta_[kDividendId].invocation_dependence = Dependence::Varying;
    value_meta_[kDivisorId].invocation_dependence  = Dependence::Invariant;

    ExecuteBinary(CommonTypes::u64);

    ExpectScalarDependence(Dependence::Varying);
}

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, ScalarZeroDivisorReplacementDoesNotProveDefinedness)
{
    SetScalarOperands(0);
    value_meta_[kDividendId].invocation_dependence = Dependence::Invariant;
    value_meta_[kDivisorId].invocation_dependence  = Dependence::Invariant;

    ExecuteBinary(CommonTypes::u64);

    ExpectScalarDependence(Dependence::Unknown);
}

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, ScalarInexactInvariantDivisorIsUnknown)
{
    SetScalarOperands(3);

    constexpr uint64_t inexact_flags[]{
        SPS_FLAG_IS_ARBITRARY,
        SPS_FLAG_UNINITIALIZED,
    };
    for (uint64_t flags : inexact_flags)
    {
        value_meta_[kDividendId]                       = {};
        value_meta_[kDivisorId]                        = {};
        value_meta_[kDividendId].invocation_dependence = Dependence::Invariant;
        value_meta_[kDivisorId].invocation_dependence  = Dependence::Invariant;
        value_meta_[kDivisorId].flags                  = flags;

        ExecuteBinary(CommonTypes::u64);

        ExpectScalarDependence(Dependence::Unknown);
    }
}

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, ScalarNonInvariantRepresentativeValueNeedsRangeProof)
{
    SetScalarOperands(3);

    constexpr Dependence non_invariant_states[]{
        Dependence::Unknown,
        Dependence::Varying,
    };
    for (Dependence divisor_dependence : non_invariant_states)
    {
        value_meta_[kDividendId]                       = {};
        value_meta_[kDivisorId]                        = {};
        value_meta_[kDividendId].invocation_dependence = Dependence::Invariant;
        value_meta_[kDivisorId].invocation_dependence  = divisor_dependence;

        ExecuteBinary(CommonTypes::u64);

        ExpectScalarDependence(Dependence::Unknown);
    }
}

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, ScalarRangeMustExcludeZero)
{
    SetScalarOperands(3);
    value_meta_[kDividendId].invocation_dependence = Dependence::Invariant;
    value_meta_[kDivisorId].invocation_dependence  = Dependence::Varying;

    ::SPIRVSimulator::DenseValueRangeInfo& divisor_range = value_meta_[kDivisorId].value_range;
    divisor_range.valid                                  = true;
    divisor_range.thread_dependent                       = true;
    divisor_range.min                                    = 0;
    divisor_range.max                                    = 3;
    divisor_range.stride                                 = 1;

    ExecuteBinary(CommonTypes::u64);
    ExpectScalarDependence(Dependence::Unknown);

    divisor_range.min = 1;
    ExecuteBinary(CommonTypes::u64);
    ExpectScalarDependence(Dependence::Varying);
}

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, VectorDefinednessIsProvedPerLane)
{
    SetVectorOperands({ 24, 24, 24 }, { 3, 0, 2 });
    value_meta_[kDividendId].invocation_dependence           = Dependence::Varying;
    value_meta_[kDividendId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
        Dependence::Invariant,
    };
    value_meta_[kDivisorId].invocation_dependence           = Dependence::Varying;
    value_meta_[kDivisorId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
        Dependence::Varying,
    };

    ExecuteBinary(CommonTypes::uvec3);

    ExpectVectorDependence(Dependence::Unknown, { Dependence::Invariant, Dependence::Unknown, Dependence::Unknown });
}

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, VectorComponentRangeCanExcludeZero)
{
    SetVectorOperands({ 24, 24, 24 }, { 3, 2, 4 });
    value_meta_[kDividendId].invocation_dependence           = Dependence::Varying;
    value_meta_[kDividendId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Invariant,
        Dependence::Varying,
    };
    value_meta_[kDivisorId].invocation_dependence           = Dependence::Varying;
    value_meta_[kDivisorId].component_invocation_dependence = {
        Dependence::Invariant,
        Dependence::Varying,
        Dependence::Invariant,
    };
    value_meta_[kDivisorId].component_ranges.resize(3);
    ::SPIRVSimulator::DenseValueRangeInfo& component_range = value_meta_[kDivisorId].component_ranges[1];
    component_range.valid                                  = true;
    component_range.thread_dependent                       = true;
    component_range.min                                    = 1;
    component_range.max                                    = 3;
    component_range.stride                                 = 1;

    ExecuteBinary(CommonTypes::uvec3);

    ExpectVectorDependence(Dependence::Varying, { Dependence::Invariant, Dependence::Varying, Dependence::Varying });
}

TEST_P(ElementwiseUnsignedDivModInvocationDependenceTests, MissingDivisorComponentProofUsesConservativeFallback)
{
    SetVectorOperands({ 24, 24, 24 }, { 3, 2, 4 });
    value_meta_[kDividendId].invocation_dependence          = Dependence::Invariant;
    value_meta_[kDivisorId].invocation_dependence           = Dependence::Invariant;
    value_meta_[kDivisorId].component_invocation_dependence = { Dependence::Invariant };

    ExecuteBinary(CommonTypes::uvec3);
    ExpectVectorDependence(Dependence::Invariant,
                           { Dependence::Invariant, Dependence::Invariant, Dependence::Invariant });

    value_meta_[kDivisorId].invocation_dependence = Dependence::Varying;
    ExecuteBinary(CommonTypes::uvec3);
    ExpectVectorDependence(Dependence::Unknown, { Dependence::Invariant, Dependence::Unknown, Dependence::Unknown });
}

INSTANTIATE_TEST_SUITE_P(UnsignedIntegerDivMod,
                         ElementwiseUnsignedDivModInvocationDependenceTests,
                         Values(spv::Op::OpUDiv, spv::Op::OpUMod));

class CoopMatrixMath : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

TEST_P(CoopMatrixMath, ParametrizedArithmeticOperation)
{
    const auto& parameters = GetParam();

    std::vector<uint32_t>         words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(_, _, true)).WillOnce(SaveArg<1>(&captured_value));

    this->ExecuteInstruction(inst);

    std::cout << "Captured value was: " << captured_value << std::endl;
    std::cout << "Expected value was: " << parameters.operands.at(0) << std::endl;

    EXPECT_EQ(captured_value, parameters.operands.at(0));
};


std::vector<TestParameters> cooperative_matrix_test{
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 1.0, 1.0, 1.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, -1.0, 1.0, 1.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 1.0, 1.0, -1.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatIB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -1, -1, -1 }, 2),
                        CommonTypes::coopMatIAcc2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 6, 8, 10, 12 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 5, 6, 7, 8 }, 2),
                        CommonTypes::coopMatUB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -2, -3, -4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 2, 4, 6, 8 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 2, 4, 6, 8 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -2, -3, -4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 4, 9, 16 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -4, -9, -16 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -2, -3, -4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -4, -9, -16 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -2, -3, -4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -1, -1, -1 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -2, -3, -4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -1, -1, -1 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, -2, -3, -4 }, 2),
                        CommonTypes::coopMatIB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 2.0, 4.0, 6.0, 8.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, -2.0, -3.0, -4.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 2.0, 4.0, 6.0, 8.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, -2.0, -3.0, -4.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 4.0, 9.0, 16.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, -4.0, -9.0, -16.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, -2.0, -3.0, -4.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, -1.0, -1.0, -1.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, -2.0, -3.0, -4.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatB2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitcast)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitcast)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertUToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertSToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, 2.0, 3.0, -4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, 2, 3, -4 }, 2),
                        CommonTypes::coopMatIA2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ -1, 2, 3, -4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ -1.0, 2.0, 3.0, -4.0 }, 2),
                        CommonTypes::coopMatA2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .build(),
};

INSTANTIATE_TEST_SUITE_P(Arithmetics, CoopMatrixMath, ValuesIn(cooperative_matrix_test));

class ArithmeticsCrashTests : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

TEST_P(ArithmeticsCrashTests, ParametrizedCrashTest)
{
    const auto& parameters = GetParam();

    std::vector<uint32_t>         words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

#ifndef NDEBUG
    EXPECT_DEATH({ this->ExecuteInstruction(inst); }, parameters.death_message);
#else
    try
    {
        this->ExecuteInstruction(inst);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_THAT(e.what(), ContainsRegex(parameters.death_message));
    }
#endif
}

std::vector<TestParameters> throw_tests{
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operand_at(0, -1.0, CommonTypes::f64)
        .set_operand_at(1, int64_t(1), CommonTypes::i64)
        .set_death_message("Invalid result type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }), CommonTypes::ivec2)
        .set_operand_at(1, int64_t(1), CommonTypes::i64)
        .set_death_message("Operand not of vector type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operand_at(0, int64_t(1), CommonTypes::i64)
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }), CommonTypes::ivec2)
        .set_death_message("Operands not of int type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }),
                        CommonTypes::vec2)
        .set_operand_at(1, 1.0, CommonTypes::f64)
        .set_death_message("Operand not of vector type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operand_at(0, 1.0, CommonTypes::f64)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }),
                        CommonTypes::vec2)
        .set_death_message("Operands not of float type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operand_at(0, 1, CommonTypes::i64)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }),
                        CommonTypes::vec2)
        .set_death_message("Invalid result type, must be vector or float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 3, 3 }), CommonTypes::ivec2)
        .set_operands_range(1, CommonTypes::i64, std::initializer_list<int64_t>{ 1, 2 })
        .set_death_message("Operands not of vector type in Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 3, 3, 3 }), CommonTypes::ivec3)
        .set_operands_range(1,
                            CommonTypes::ivec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }) })
        .set_death_message("Operands not of equal/correct length in Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_at({ 0, 2 },
                         CommonTypes::ivec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 3, 3 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }) })
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1, 1 }), CommonTypes::ivec3)
        .set_death_message("Operands not of equal/correct length in Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_at({ 0, 2 },
                         CommonTypes::ivec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 3, 3 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }) })
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), CommonTypes::vec2)
        .set_death_message("Could not find valid parameter type combination for Op_IAdd vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_at({ 0, 2 }, CommonTypes::i64, std::initializer_list<int64_t>{ 3, 2 })
        .set_operand_at(1, 1.0, CommonTypes::f64)
        .set_death_message("Could not find valid parameter type combination for Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0, 3.0, CommonTypes::f64)
        .set_operands_range(1, CommonTypes::i64, std::initializer_list<int64_t>{ 1, 2 })
        .set_death_message("Invalid result type for Op_IAdd, must be vector, int or cooperative matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0 }), CommonTypes::vec2)
        .set_operands_range(1, CommonTypes::f64, std::initializer_list<double>{ 1.0, 2.0 })
        .set_death_message("Operands not of vector type in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0, 3.0 }),
                        CommonTypes::vec3)
        .set_operands_range(1,
                            CommonTypes::vec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }) })
        .set_death_message("Operands not of equal/correct length in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_at({ 0, 2 },
                         CommonTypes::vec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }) })
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0, 1.0 }),
                        CommonTypes::vec3)
        .set_death_message("Operands not of equal/correct length in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_range(0,
                            CommonTypes::vec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }) })
        .set_operand_at(
            2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), CommonTypes::ivec2)
        .set_death_message("SPIR-V simulator: .*vector contains non-doubles in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<double>{ 3.0, 1.0 })
        .set_operand_at(2, 2, CommonTypes::i64)
        .set_death_message("SPIR-V simulator: .*Operands not of float type in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0, 3, CommonTypes::i64)
        .set_operands_range(1, CommonTypes::f64, std::initializer_list<double>{ 1.0, 2.0 })
        .set_death_message("Invalid result type for Op_FAdd, must be vector or float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), CommonTypes::ivec2)
        .set_operands_range(1, CommonTypes::i64, std::initializer_list<SPIRVSimulator::Value>{ 1, 0 })
        .set_death_message("Operands not of vector type in Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_at({ 0, 2 },
                         CommonTypes::ivec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 0 }) })
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1, 1 }), CommonTypes::ivec3)
        .set_death_message("Operands not of equal/correct length in Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1, 1 }), CommonTypes::ivec3)
        .set_operands_range(1,
                            CommonTypes::ivec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 0 }) })
        .set_death_message("Operands not of equal/correct length in Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_at({ 0, 2 },
                         CommonTypes::ivec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 0 }) })
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), CommonTypes::vec2)
        .set_death_message("Could not find valid parameter type combination for Op_ISub vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0, 1, CommonTypes::i64)
        .set_operand_at(1, 1, CommonTypes::u64)
        .set_operand_at(2, 0.0, CommonTypes::f64)
        .set_death_message("Could not find valid parameter type combination for Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0, 1, CommonTypes::f64)
        .set_operand_at(1, 1, CommonTypes::u64)
        .set_operand_at(2, 0, CommonTypes::i64)
        .set_death_message("Invalid result type for Op_ISub, must be vector or int")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), CommonTypes::vec2)
        .set_operands_range(1, CommonTypes::i64, std::initializer_list<int64_t>{ 0, 1 })
        .set_death_message("Operands set to be vector type in Op_FSub, but they are not, illegal input parameters")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0, 1.0 }),
                        CommonTypes::vec3)
        .set_operands_range(1,
                            CommonTypes::vec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 0.0, 0.0 }) })
        .set_death_message("Operands are vector type but not of equal length in Op_FSub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_at({ 0, 2 },
                         CommonTypes::vec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 0.0, 0.0 }) })
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), CommonTypes::ivec2)
        .set_death_message("Found non-floating point operand in Op_FSub vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_at({ 0, 2 }, CommonTypes::f64, std::initializer_list<double>{ 1.0, 0.0 })
        .set_operand_at(1, 1, CommonTypes::i64)
        .set_death_message("Found non-floating point operand in Op_FSub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0, 1, CommonTypes::i64)
        .set_operands_range(1, CommonTypes::f64, std::initializer_list<double>{ 1.0, 0.0 })
        .set_death_message("Invalid result type for Op_FSub, must be vector or float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_range(0,
                            CommonTypes::ivec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }) })
        .set_operand_at(2, 0, CommonTypes::i64)
        .set_death_message("Operands not of vector type in Op_IMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1, 1 }), CommonTypes::ivec3)
        .set_operands_range(1,
                            CommonTypes::ivec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }) })
        .set_death_message("Operands not of equal/correct length in Op_IMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_at({ 0, 2 },
                         CommonTypes::ivec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }) })
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), CommonTypes::vec2)
        .set_death_message("Could not find valid parameter type combination for Op_IMul vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_range(0, CommonTypes::i64, std::initializer_list<int64_t>{ 2, 2 })
        .set_operand_at(2, 1.0, f64)
        .set_death_message("Could not find valid parameter type combination for Op_IMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0, 2.0, CommonTypes::f64)
        .set_operands_range(1, CommonTypes::i64, std::initializer_list<int64_t>{ 2, 1 })
        .set_death_message("Invalid result type for Op_IMul, must be vector or integer type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_at({ 0, 2 },
                         CommonTypes::vec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }) })
        .set_operand_at(1, 2.0, CommonTypes::f64)
        .set_death_message("Operands not of vector type in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_at({ 0, 2 },
                         CommonTypes::vec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }) })
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0, 1.0 }),
                        CommonTypes::vec3)
        .set_death_message("Operands not of equal/correct length in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_at({ 0, 2 },
                         CommonTypes::vec2,
                         std::initializer_list<::SPIRVSimulator::Value>{
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }),
                             std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }) })
        .set_operand_at(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), CommonTypes::ivec2)
        .set_death_message("vector contains non-doubles in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<double>{ 2.0, 2.0 })
        .set_operand_at(2, 1, CommonTypes::i64)
        .set_death_message("Operands are not floats/doubles in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0, 2, CommonTypes::i64)
        .set_operands_range(1, CommonTypes::f64, std::initializer_list<double>{ 2.0, 1.0 })
        .set_death_message("Invalid result type for Op_FMul, must be vector or float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("CooperativeMatrixMulAddKHR takes 5 required and 1 optional operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0, 2,
                        CommonTypes::i32)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Result must be of type cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Use of result must be MatrixAccumulatorKHR")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        2,
                        CommonTypes::i32)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Matrix A must be of type cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Use of matrix A must be MatrixAKHR")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        2,
                        CommonTypes::i32)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Matrix B must be of type cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Use of matrix B must be MatrixBKHR")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(3,
                        2,
                        CommonTypes::i32)
        .set_death_message("Matrix C must be of type cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("Use of matrix C must be MatrixAccumulatorKHR")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_death_message("CooperativeMatrixMulAddKHR currently supports all-float or all-integer component type combinations")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCooperativeMatrixMulAddKHR)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatB2)
        .set_operand_at(3,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(4,
                        0x10,
                        CommonTypes::literal)
        .set_death_message("Saturating accumulation is only supported for integer cooperative matrices")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        2,
                        CommonTypes::i32)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(2,
                        2,
                        CommonTypes::i32)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        2,
                        CommonTypes::i32)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_operand_at(2,
                        2,
                        CommonTypes::i32)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("matrix component type must be same for both operands")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatUB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("Could not find valid parameter type combination for Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4, 5, 6 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("OpIAdd failed on cooperative matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        2,
                        CommonTypes::i32)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        2,
                        CommonTypes::i32)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("matrix component type must be same for both operands")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatIB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("Could not find valid parameter type combination for Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4, 5, 6 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("OpISub: matrices not the same size")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4, 5, 6 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("result matrix component type must be same as operands")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4, 5, 6 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("result matrix component type must be same as operands")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        2,
                        CommonTypes::i32)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        2,
                        CommonTypes::i32)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("matrix component type must be same for both operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatIB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("Could not find valid parameter type combination for Op_IMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4, 5, 6 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("OpIMul: matrices not the same size")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        2,
                        CommonTypes::i32)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        2,
                        CommonTypes::i32)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("matrix component type must be same for both operands")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatIB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("Could not find valid parameter type combination for Op_SDiv")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4, 5, 6 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("OpIMul: matrices not the same size")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("matrix component type must be same for both operands")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIB2)
        .set_death_message("OpUDiv: Divisor must be unsigned")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatUB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("Could not find valid parameter type combination for Op_UDiv")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 1, 1, 1 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4, 5, 6 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("OpIMul: matrices not the same size")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1, 2.0, CommonTypes::f64)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2, 2.0, CommonTypes::f64)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("matrix component type must be same for both operands and Float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("OpFAdd failed on cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1, 2.0, CommonTypes::f64)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2, 2.0, CommonTypes::f64)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("matrix component type must be same for both operands and Float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("OpFSub failed on cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1, 2.0, CommonTypes::f64)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2, 2.0, CommonTypes::f64)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("matrix component type must be same for both operands and Float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("OpFMul failed on cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1, 2.0, CommonTypes::f64)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("Op1 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2, 2.0, CommonTypes::f64)
        .set_death_message("Op2 must be Cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUB2)
        .set_death_message("matrix component type must be same for both operands and Float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op1 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::AggregateV>(),
                        CommonTypes::coopMatB2)
        .set_death_message("what\\? op2 is coopMatrix, but does not contain MatrixV")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatA2)
        .set_operand_at(2,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 2),
                        CommonTypes::coopMatB2)
        .set_death_message("OpFDiv failed on cooperative Matrix")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1, 2.0, CommonTypes::f64)
        .set_death_message("Operand set to be matrix type in OpConvertFToS, but it is not")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Operand matrix does not contain floats")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_death_message("Result matrix does not contain signed scalars")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1, 2.0, CommonTypes::f64)
        .set_death_message("Operand set to be matrix type in OpConvertFToU, but it is not")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_death_message("Operand matrix does not contain floats")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_death_message("Result matrix does not contain unsigned scalars")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertUToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1, 2, CommonTypes::u64)
        .set_death_message("Operand set to be matrix type in OpConvertUToF, but it is not")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertUToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_death_message("Operand matrix does not contain unsinged scalars")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertUToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Result matrix does not contain floats")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertSToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1, 2, CommonTypes::i64)
        .set_death_message("Operand set to be matrix type in OpConvertSToF, but it is not")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertSToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_death_message("Operand matrix does not contain singed scalars")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertSToF)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatUAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_death_message("Result matrix does not contain floats")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitcast)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::mat2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 0.0, 0.0, 0.0, 0.0 }, 2),
                        CommonTypes::coopMatAcc2)
        .set_death_message("can only bitcast coopMatrices to other coopMatrices")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitcast)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 0, 0, 0, 0 }, 2),
                        CommonTypes::coopMatIAcc2)
        .set_operand_at(1,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                            std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::AggregateV>(), 0.0 }),
                            std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<SPIRVSimulator::Value>{ 0.0, 0.0 }) }),
                        CommonTypes::coopMatAcc2)
        .set_death_message("invalid operand element type in Op_Bitcast")
        .build(),
};

INSTANTIATE_TEST_SUITE_P(Arithmetics, ArithmeticsCrashTests, ValuesIn(throw_tests));

class CooperativeMatrixConversionTests : public SPIRVSimulatorMockBase, public ::testing::Test
{
  protected:
    const uint32_t result_type_id_      = 2000;
    const uint32_t result_id_           = 2001;
    const uint32_t op1_id_              = 2002;
    const uint32_t op2_id_              = 2003;
    const uint32_t float_operand_id_    = 2004;
    const uint32_t unsigned_operand_id_ = 2005;
    const uint32_t signed_operand_id_   = 2006;
    const uint32_t cols_3_id_           = 2007;
    const uint32_t rows_3_id_           = 2008;

    ::SPIRVSimulator::Value signed_matrix_value_;
    ::SPIRVSimulator::Value unsigned_matrix_value_;
    ::SPIRVSimulator::Value float_matrix_value_;

    void SetUp() override
    {
        signed_matrix_value_ = std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<int64_t>{ 1, 2, 3, 4 }, 2);
        unsigned_matrix_value_ = std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<uint64_t>{ 1, 2, 3, 4 }, 2);
        float_matrix_value_ = std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2);

        EXPECT_CALL(*this, HasFlags(_, _)).WillRepeatedly(Return(false));
        EXPECT_CALL(*this, GetTypeByResultId(op1_id_)).WillRepeatedly(ReturnRef(types_[CommonTypes::coopMatIA2]));
        EXPECT_CALL(*this, GetTypeByResultId(op2_id_)).WillRepeatedly(ReturnRef(types_[CommonTypes::coopMatIB2]));
        EXPECT_CALL(*this, GetTypeByResultId(float_operand_id_)).WillRepeatedly(ReturnRef(types_[CommonTypes::coopMatAcc2]));
        EXPECT_CALL(*this, GetTypeByResultId(unsigned_operand_id_)).WillRepeatedly(ReturnRef(types_[CommonTypes::coopMatUAcc2]));
        EXPECT_CALL(*this, GetTypeByResultId(signed_operand_id_)).WillRepeatedly(ReturnRef(types_[CommonTypes::coopMatIAcc2]));
        EXPECT_CALL(*this, GetValue(op1_id_)).WillRepeatedly(ReturnRef(signed_matrix_value_));
        EXPECT_CALL(*this, GetValue(op2_id_)).WillRepeatedly(ReturnRef(signed_matrix_value_));
        EXPECT_CALL(*this, GetValue(float_operand_id_)).WillRepeatedly(ReturnRef(float_matrix_value_));
        EXPECT_CALL(*this, GetValue(unsigned_operand_id_)).WillRepeatedly(ReturnRef(unsigned_matrix_value_));
        EXPECT_CALL(*this, GetValue(signed_operand_id_)).WillRepeatedly(ReturnRef(signed_matrix_value_));
        EXPECT_CALL(*this, GetValue(CommonValues::coop_rows_2))
            .WillRepeatedly(ReturnRefOfCopy(::SPIRVSimulator::Value(static_cast<uint64_t>(2))));
        EXPECT_CALL(*this, GetValue(CommonValues::coop_cols_2))
            .WillRepeatedly(ReturnRefOfCopy(::SPIRVSimulator::Value(static_cast<uint64_t>(2))));
        EXPECT_CALL(*this, GetValue(cols_3_id_))
            .WillRepeatedly(ReturnRefOfCopy(::SPIRVSimulator::Value(static_cast<uint64_t>(3))));
        EXPECT_CALL(*this, GetValue(rows_3_id_))
            .WillRepeatedly(ReturnRefOfCopy(::SPIRVSimulator::Value(static_cast<uint64_t>(3))));
        EXPECT_CALL(*this, TransferFlags(::testing::A<uint32_t>(), ::testing::A<uint32_t>()))
            .Times(::testing::AnyNumber());
        EXPECT_CALL(*this, TransferFlags(::testing::A<uint32_t>(), ::testing::A<uint64_t>()))
            .Times(::testing::AnyNumber());
    }

    void TearDown() override
    {
        ::testing::Mock::VerifyAndClearExpectations(this);
    }
};


TEST_F(CooperativeMatrixConversionTests, OpConvertFToSCooperativeMatrixRejectsColumnMismatch)
{
    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::CooperativeMatrix(CommonTypes::i64,
                                                  CommonValues::coop_scope_subgroup,
                                                  CommonValues::coop_rows_2,
                                                  cols_3_id_,
                                                  CommonValues::coop_use_accumulator);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id_)).WillRepeatedly(ReturnRef(result_type));
    std::vector<uint32_t> words{ static_cast<uint32_t>(spv::Op::OpConvertFToS), result_type_id_, result_id_, float_operand_id_ };
    ::SPIRVSimulator::Instruction inst{ .opcode     = spv::Op::OpConvertFToS,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

#ifndef NDEBUG
    EXPECT_DEATH({ this->ExecuteInstruction(inst); }, "operand and result matrix size mismatch - columns");
#else
    EXPECT_THROW(this->ExecuteInstruction(inst), std::runtime_error);
#endif
}

TEST_F(CooperativeMatrixConversionTests, OpConvertFToUCooperativeMatrixRejectsColumnMismatch)
{
    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::CooperativeMatrix(CommonTypes::u64,
                                                  CommonValues::coop_scope_subgroup,
                                                  CommonValues::coop_rows_2,
                                                  cols_3_id_,
                                                  CommonValues::coop_use_accumulator);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id_)).WillRepeatedly(ReturnRef(result_type));
    std::vector<uint32_t> words{ static_cast<uint32_t>(spv::Op::OpConvertFToU), result_type_id_, result_id_, float_operand_id_ };
    ::SPIRVSimulator::Instruction inst{ .opcode     = spv::Op::OpConvertFToU,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

#ifndef NDEBUG
    EXPECT_DEATH({ this->ExecuteInstruction(inst); }, "operand and result matrix size mismatch - columns");
#else
    EXPECT_THROW(this->ExecuteInstruction(inst), std::runtime_error);
#endif
}

TEST_F(CooperativeMatrixConversionTests, OpConvertUToFCooperativeMatrixRejectsRowMismatch)
{
    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::CooperativeMatrix(CommonTypes::f64,
                                                  CommonValues::coop_scope_subgroup,
                                                  rows_3_id_,
                                                  CommonValues::coop_cols_2,
                                                  CommonValues::coop_use_accumulator);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id_)).WillRepeatedly(ReturnRef(result_type));
    std::vector<uint32_t> words{ static_cast<uint32_t>(spv::Op::OpConvertUToF), result_type_id_, result_id_, unsigned_operand_id_ };
    ::SPIRVSimulator::Instruction inst{ .opcode     = spv::Op::OpConvertUToF,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

#ifndef NDEBUG
    EXPECT_DEATH({ this->ExecuteInstruction(inst); }, "operand and result matrix size mismatch - rows");
#else
    EXPECT_THROW(this->ExecuteInstruction(inst), std::runtime_error);
#endif
}

TEST_F(CooperativeMatrixConversionTests, OpConvertSToFCooperativeMatrixRejectsRowMismatch)
{
    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::CooperativeMatrix(CommonTypes::f64,
                                                  CommonValues::coop_scope_subgroup,
                                                  rows_3_id_,
                                                  CommonValues::coop_cols_2,
                                                  CommonValues::coop_use_accumulator);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id_)).WillRepeatedly(ReturnRef(result_type));
    std::vector<uint32_t> words{ static_cast<uint32_t>(spv::Op::OpConvertSToF), result_type_id_, result_id_, signed_operand_id_ };
    ::SPIRVSimulator::Instruction inst{ .opcode     = spv::Op::OpConvertSToF,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

#ifndef NDEBUG
    EXPECT_DEATH({ this->ExecuteInstruction(inst); }, "operand and result matrix size mismatch - rows");
#else
    EXPECT_THROW(this->ExecuteInstruction(inst), std::runtime_error);
#endif
}
