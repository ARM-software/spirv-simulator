#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "spirv.hpp"
#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;



class ArithmeticsTest : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

TEST_P(ArithmeticsTest, ArithmeticOperations)
{
    const auto& parameters = GetParam();

    const uint32_t        result_id = NextId();
    std::vector<uint32_t> words{ parameters.opcode, parameters.operand_types[0], result_id };

    for (uint32_t op = 1; op < parameters.operands.size(); ++op)
    {
        const uint32_t op_id = NextId();
        words.push_back(op_id);
        EXPECT_CALL(*this, GetValue(op_id)).WillRepeatedly(ReturnRefOfCopy(parameters.operands[op]));
        EXPECT_CALL(*this, GetTypeByResultId(op_id)).WillRepeatedly(Return(types_[parameters.operand_types[op]]));
    }

    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(_, _)).WillOnce(SaveArg<1>(&captured_value));

    this->ExecuteInstruction(inst);

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
        .set_opcode(spv::Op::OpIAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 4, 4 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1, -1 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1, 1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 4.0, 4.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1.0, -1.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -2, -2 }), Type::ivec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1, -1 }), Type::ivec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -2.0, -2.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1.0, -1.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVectorTimesScalar)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -2.0, -2.0 }), Type::vec2)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1.0, -1.0 }), Type::vec2)
        .set_op_n(2, 2.0, Type::f64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpMatrixTimesVector)
        .set_operands_size(3)
        .set_op_n(0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 7.0, 10.0 }), Type::vec2)
        .set_op_n(1,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 3.0, 4.0 }) }),
                  Type::mat2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }), Type::vec2)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpMatrixTimesMatrix)
        .set_operands_size(3)
        .set_op_n(0,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 7.0, 10.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 15.0, 22.0 }) }),
                  Type::mat2)
        .set_op_n(1,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 3.0, 4.0 }) }),
                  Type::mat2)
        .set_op_n(2,
                  std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }),
                      std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 3.0, 4.0 }) }),
                  Type::mat2)

        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpDot)
        .set_operands_size(3)
        .set_op_n(0, 8.0, Type::f64)
        .set_op_n(1, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_op_n(2, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .build()
};

INSTANTIATE_TEST_SUITE_P(Arithmetics, ArithmeticsTest, ValuesIn(test_cases));
