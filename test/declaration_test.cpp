#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <memory>
#include <cstdint>

#include "spirv.hpp"
#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class DeclarationTests : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

TEST_P(DeclarationTests, ParametrizedDeclarationOperation)
{
    const auto& parameters = GetParam();

    std::vector<uint32_t>         words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(_, _)).WillOnce(SaveArg<1>(&captured_value));

    this->ExecuteInstruction(inst);

    EXPECT_EQ(captured_value, parameters.operands.at(0));
}

std::vector<TestParameters> test_cases{
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConstantTrue)
        .set_operand_at(0, static_cast<uint64_t>(true), CommonTypes::boolean)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConstantFalse)
        .set_operand_at(0, static_cast<uint64_t>(false), CommonTypes::boolean)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpConstant)
        .set_operand_at(0, static_cast<int64_t>(1), CommonTypes::i64)
        .set_operand_at(1, static_cast<int64_t>(1), CommonTypes::literal)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCompositeConstruct)
        .set_operand_at(
            0, std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }), CommonTypes::vec2)
        .set_operands_range(1, CommonTypes::f64, std::initializer_list<SPIRVSimulator::Value>{ 1.0, 2.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCompositeConstruct)
        .set_operands_range(0,
                            CommonTypes::vec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCompositeConstruct)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                            std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }),
                            std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 }) }),
                        CommonTypes::mat2)
        .set_operands_range(1,
                            CommonTypes::vec2,
                            std::initializer_list<SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, 4.0 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCompositeConstruct)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::AggregateV>(
                            std::initializer_list<::SPIRVSimulator::Value>{ 1.0, static_cast<int64_t>(3) }),
                        ::SPIRVSimulator::Type::Struct(0))
        .set_operand_at(1, 1.0, CommonTypes::f64)
        .set_operand_at(2, static_cast<int64_t>(3), CommonTypes::i64)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCompositeConstruct)
        .set_operand_at(0,
                        std::make_shared<SPIRVSimulator::AggregateV>(
                            std::initializer_list<::SPIRVSimulator::Value>{ 1.0, 2.0, 3.0 }),
                        ::SPIRVSimulator::Type::Array(CommonTypes::f64, 3))
        .set_operands_at(std::initializer_list<uint32_t>{ 1, 2, 3 },
                         CommonTypes::f64,
                         std::initializer_list<::SPIRVSimulator::Value>{ 1.0, 2.0, 3.0 })
        .build(),
};

INSTANTIATE_TEST_SUITE_P(Declaration, DeclarationTests, ValuesIn(test_cases));
