#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>

#include "spirv.hpp"
#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class ArithmeticsTests : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

TEST_P(ArithmeticsTests, ParametrizedArithmeticOperation)
{
    const auto& parameters = GetParam();

    const uint32_t        result_id = NextId();
    std::vector<uint32_t> words{ parameters.opcode, parameters.operand_types[0], result_id };

    for (uint32_t op = 1; op < parameters.operands.size(); ++op)
    {
        const uint32_t op_id = NextId();
        words.push_back(op_id);
        EXPECT_CALL(*this, GetValue(op_id)).WillRepeatedly(ReturnRefOfCopy(parameters.operands[op]));
        EXPECT_CALL(*this, GetTypeByResultId(op_id)).WillRepeatedly(ReturnRef(types_[parameters.operand_types[op]]));
    }

    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(_, _)).WillOnce(SaveArg<1>(&captured_value));

    this->ExecuteInstruction(inst);

    std::cout << "Captured value was: " << captured_value << std::endl;
    std::cout << "Expected value was: " << parameters.operands[0] << std::endl;

    EXPECT_EQ(captured_value, parameters.operands[0]);
}

std::vector<TestParameters> test_cases = {
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operands_size(2)
        .set_op_n(0, -1, Type::i64)
        .set_op_n(1, int64_t(1), Type::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operands_size(2)
        .set_op_n(0, -1.0, Type::f64)
        .set_op_n(1, 1.0, Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, 3, Type::i64)
        .set_op_n(1, 1, Type::i64)
        .set_op_n(2, 2, Type::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, 3.0, Type::f64)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 2.0, Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, std::numeric_limits<uint64_t>::max(), Type::u64)
        .set_op_n(1, uint64_t(1), Type::u64)
        .set_op_n(2, 2, Type::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, -1, Type::i64)
        .set_op_n(1, 1, Type::i64)
        .set_op_n(2, 2, Type::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(0, -1.0, Type::f64)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 2.0, Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, 4, Type::i64)
        .set_op_n(1, 2, Type::i64)
        .set_op_n(2, 2, Type::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, 4.0, Type::f64)
        .set_op_n(1, 2.0, Type::f64)
        .set_op_n(2, 2.0, Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_operands_size(3)
        .set_op_n(0, -2, Type::i64)
        .set_op_n(1, -5, Type::i64)
        .set_op_n(2, 2, Type::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_operands_size(3)
        .set_op_n(0, uint64_t(2), Type::u64)
        .set_op_n(1, uint64_t(5), Type::u64)
        .set_op_n(2, uint64_t(2), Type::u64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_operands_size(3)
        .set_op_n(0, 1.0 / 3.0, Type::f64)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 3.0, Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpUMod)
        .set_operands_size(3)
        .set_op_n(0, uint64_t(1), Type::u64)
        .set_op_n(1, uint64_t(13), Type::u64)
        .set_op_n(2, uint64_t(6), Type::u64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpShiftRightArithmetic)
        .set_operands_size(3)
        .set_op_n(0, uint64_t(0xffffffffffffffff), Type::u64)
        .set_op_n(1, uint64_t(0x8000000000000000), Type::u64)
        .set_op_n(2, uint64_t(63), Type::u64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpShiftRightArithmetic)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0xffffffffffffffff, 0xffffffffffffffff }), Type::uvec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0x8000000000000000, 0x8000000000000000 }), Type::uvec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 63, 63 }), Type::uvec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitwiseXor)
        .set_operands_size(3)
        .set_op_n(0, uint64_t(0x0000000000000000), Type::u64)
        .set_op_n(1, uint64_t(0x8000000000000000), Type::u64)
        .set_op_n(2, uint64_t(0x8000000000000000), Type::u64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpBitwiseXor)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0x8000000000000000, 0xffffffffffffffff }), Type::uvec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0x8000000000000000, 0x0000000000000000 }), Type::uvec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0x0000000000000000, 0xffffffffffffffff }), Type::uvec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 4, 4 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), Type::ivec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpINotEqual)
        .set_operands_size(3)
        .set_op_n(0, uint64_t(1), Type::boolean)
        .set_op_n(1, uint64_t(13), Type::u64)
        .set_op_n(2, uint64_t(6), Type::u64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpINotEqual)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0, 1 }), Type::bvec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 4 }), Type::ivec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operands_size(2)
        .set_op_n(0, uint64_t(1), Type::u64)
        .set_op_n(1, double(1.68), Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToU)
        .set_operands_size(2)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 0, 1 }), Type::uvec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -0.68, 1.12 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operands_size(2)
        .set_op_n(0, int64_t(1), Type::i64)
        .set_op_n(1, double(1.68), Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConvertFToS)
        .set_operands_size(2)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 1 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -0.68, 1.12 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }), Type::ivec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 4.0, 4.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -2, -2 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 2, 2 }), Type::ivec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -2.0, -2.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVectorTimesScalar)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -2.0, -2.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }), Type::vec2)
        .set_op_n(2, 2.0, Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpMatrixTimesVector)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 7.0, 10.0 }), Type::vec2)
        .set_op_n(1,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 }) }),
                  Type::mat2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpMatrixTimesMatrix)
        .set_operands_size(3)
        .set_op_n(0,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 7.0, 10.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 15.0, 22.0 }) }),
                  Type::mat2)
        .set_op_n(1,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 }) }),
                  Type::mat2)
        .set_op_n(2,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 }) }),
                  Type::mat2)

        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpDot)
        .set_operands_size(3)
        .set_op_n(0, 8.0, Type::f64)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .build()
};

INSTANTIATE_TEST_SUITE_P(Arithmetics, ArithmeticsTests, ValuesIn(test_cases));

// Death tests cannot work without Debug build, as they rely on program crashing with certain message in stderr
// Also, they are slow

class ArithmeticsCrashTests : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

TEST_P(ArithmeticsCrashTests, ParametrizedCrashTest)
{
    const auto& parameters = GetParam();

    const uint32_t        result_id = NextId();
    std::vector<uint32_t> words{ parameters.opcode, parameters.operand_types[0], result_id };

    for (uint32_t op = 1; op < parameters.operands.size(); ++op)
    {
        const uint32_t op_id = NextId();
        words.push_back(op_id);
        EXPECT_CALL(*this, GetValue(op_id)).WillRepeatedly(ReturnRefOfCopy(parameters.operands[op]));
        EXPECT_CALL(*this, GetTypeByResultId(op_id)).WillRepeatedly(ReturnRef(types_[parameters.operand_types[op]]));
    }

    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

#ifndef NDEBUG
    EXPECT_DEATH({ this->ExecuteInstruction(inst); }, parameters.death_message);
#else
    try
    {
        this->ExecuteInstruction(inst);
    }
    catch (std::runtime_error e)
    {
        EXPECT_THAT(e.what(), HasSubstr(parameters.death_message));
    }
#endif
}

std::vector<TestParameters> throw_tests = {
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operands_size(2)
        .set_op_n(0, -1.0, Type::f64)
        .set_op_n(1, int64_t(1), Type::i64)
        .set_death_message("Invalid result type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operands_size(2)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }), Type::ivec2)
        .set_op_n(1, int64_t(1), Type::i64)
        .set_death_message("Operand not of vector type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_operands_size(2)
        .set_op_n(0, int64_t(1), Type::i64)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -1, -1 }), Type::ivec2)
        .set_death_message("Operands not of int type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operands_size(2)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }), Type::vec2)
        .set_op_n(1, 1.0, Type::f64)
        .set_death_message("Operand not of vector type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operands_size(2)
        .set_op_n(0, 1.0, Type::f64)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }), Type::vec2)
        .set_death_message("Operands not of float type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_operands_size(2)
        .set_op_n(0, 1, Type::i64)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ -1.0, -1.0 }), Type::vec2)
        .set_death_message("Invalid result type, must be vector or float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 3 }), Type::ivec2)
        .set_op_n(1, 1, Type::i64)
        .set_op_n(2, 2, Type::i64)
        .set_death_message("Operands not of vector type in Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 3, 3 }), Type::ivec3)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), Type::ivec2)
        .set_death_message("Operands not of equal/correct length in Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 3 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1, 1 }), Type::ivec3)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), Type::ivec2)
        .set_death_message("Operands not of equal/correct length in Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 3 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), Type::ivec2)
        .set_death_message("Could not find valid parameter type combination for Op_IAdd vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, 3, Type::i64)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 2, Type::i64)
        .set_death_message("Could not find valid parameter type combination for Op_IAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, 3.0, Type::f64)
        .set_op_n(1, 1, Type::i64)
        .set_op_n(2, 2, Type::i64)
        .set_death_message("Invalid result type for Op_IAdd, must be vector or int")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0 }), Type::vec2)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 2.0, Type::f64)
        .set_death_message("Operands not of vector type in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0, 3.0 }), Type::vec3)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .set_death_message("Operands not of equal/correct length in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0 }), Type::vec2)
        .set_op_n(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0, 1.0 }), Type::vec3)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 2.0 }), Type::vec2)
        .set_death_message("Operands not of equal/correct length in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 3.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 2, 2 }), Type::ivec2)
        .set_death_message("SPIRV simulator: vector contains non-doubles in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, 3.0, Type::f64)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 2, Type::i64)
        .set_death_message("SPIRV simulator: Operands not of float type in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, 3, Type::i64)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 2.0, Type::f64)
        .set_death_message("Invalid result type for Op_FAdd, must be vector or float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(1, 1, Type::i64)
        .set_op_n(2, 0, Type::i64)
        .set_death_message("Operands not of vector type in Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1, 1 }), Type::ivec3)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 0 }), Type::ivec2)
        .set_death_message("Operands not of equal/correct length in Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1, 1 }), Type::ivec3)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 0 }), Type::ivec2)
        .set_death_message("Operands not of equal/correct length in Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 0, 0 }), Type::ivec2)
        .set_death_message("Could not find valid parameter type combination for Op_ISub vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, 1, Type::i64)
        .set_op_n(1, 1, Type::u64)
        .set_op_n(2, 0.0, Type::f64)
        .set_death_message("Could not find valid parameter type combination for Op_ISub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, 1, Type::f64)
        .set_op_n(1, 1, Type::u64)
        .set_op_n(2, 0, Type::i64)
        .set_death_message("Invalid result type for Op_ISub, must be vector or int")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(1, 0, Type::i64)
        .set_op_n(2, 1, Type::i64)
        .set_death_message("Operands set to be vector type in Op_FSub, but they are not, illegal input parameters")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0, 1.0 }), Type::vec3)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 0.0, 0.0 }), Type::vec2)
        .set_death_message("Operands are vector type but not of equal length in Op_FSub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 0.0, 0.0 }), Type::vec2)
        .set_death_message("Found non-floating point operand in Op_FSub vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(0, 1.0, Type::f64)
        .set_op_n(1, 1, Type::i64)
        .set_op_n(2, 0.0, Type::f64)
        .set_death_message("Found non-floating point operand in Op_FSub")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(0, 1, Type::i64)
        .set_op_n(1, 1.0, Type::f64)
        .set_op_n(2, 0.0, Type::f64)
        .set_death_message("Invalid result type for Op_FSub, must be vector or float")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(2, 0, Type::i64)
        .set_death_message("Operands not of vector type in Op_IMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1, 1 }), Type::ivec3)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1 }), Type::ivec2)
        .set_death_message("Operands not of equal/correct length in Op_IMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1, 1 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1 }), Type::ivec2)
        .set_death_message("Could not find valid parameter type combination for Op_IMul vector operand")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, 2, Type::i64)
        .set_op_n(1, 2, Type::i64)
        .set_op_n(2, 1.0, f64)
        .set_death_message("Could not find valid parameter type combination for Op_IMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, 2.0, Type::f64)
        .set_op_n(1, 2, Type::i64)
        .set_op_n(2, 1, i64)
        .set_death_message("Invalid result type for Op_IMul, must be vector or integer type")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(1, 2.0, Type::f64)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_death_message("Operands not of vector type in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(
            1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0, 1.0 }), Type::vec3)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_death_message("Operands not of equal/correct length in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 1.0 }), Type::vec2)
        .set_death_message("vector contains non-doubles in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, 2.0, Type::f64)
        .set_op_n(1, 2.0, Type::f64)
        .set_op_n(2, 1, Type::i64)
        .set_death_message("Operands are not floats/doubles in Op_FMul")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, 2, Type::i64)
        .set_op_n(1, 2.0, Type::f64)
        .set_op_n(2, 1.0, Type::f64)
        .set_death_message("Invalid result type for Op_FMul, must be vector or float")
        .build(),
};

INSTANTIATE_TEST_SUITE_P(Arithmetics, ArithmeticsCrashTests, ValuesIn(throw_tests));
