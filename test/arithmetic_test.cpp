#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <memory>
#include <cstdint>
#include <vector>

#include "spirv.hpp"
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
    }
    catch (std::runtime_error e)
    {
        EXPECT_THAT(e.what(), HasSubstr(parameters.death_message));
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
        .set_death_message("SPIRV simulator: vector contains non-doubles in Op_FAdd")
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_range(0, CommonTypes::f64, std::initializer_list<double>{ 3.0, 1.0 })
        .set_operand_at(2, 2, CommonTypes::i64)
        .set_death_message("SPIRV simulator: Operands not of float type in Op_FAdd")
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

