#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <sys/types.h>

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

    local_data = prepare_input_data(parameters);
    simulation_data_ = &local_data;

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(_, _, true)).WillOnce(SaveArg<1>(&captured_value));

    this->ExecuteInstruction(inst);

    expect_equal(parameters.operands.at(0), captured_value);
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
        .set_operands_range(1, CommonTypes::f64, std::initializer_list<double>{ 1.0, 2.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCompositeConstruct)
        .set_operands_range(0,
                            CommonTypes::vec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 2.0 }) })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpCompositeConstruct)
        .set_operand_at(
            0,
            std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
            CommonTypes::mat2)
        .set_operands_range(1,
                            CommonTypes::vec2,
                            std::initializer_list<::SPIRVSimulator::Value>{
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.0, 3.0 }),
                                std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list<double>{ 2.0, 4.0 }) })
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
                         std::initializer_list<double>{ 1.0, 2.0, 3.0 })
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constant(static_cast<uint32_t>(1))
        .set_operand_at(0,
                        static_cast<uint64_t>(1),
                        ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::u32))
        .set_operand_at(1, static_cast<uint64_t>(spv::StorageClassPushConstant), CommonTypes::storage_class)
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constant(static_cast<uint64_t>(1))
        .set_operand_at(0,
                        static_cast<uint64_t>(1),
                        ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::u64))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constant(static_cast<int32_t>(-1))
        .set_operand_at(0,
                        static_cast<int32_t>(-1),
                        ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::i32))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constant(static_cast<int64_t>(1))
        .set_operand_at(0,
                        static_cast<int64_t>(1),
                        ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::i64))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constant(1.0f)
        .set_operand_at(0, 1.0f, ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::f32))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constant(1.0)
        .set_operand_at(0, 1.0, ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::f64))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constants(std::vector{ 1.1, 2.0, 3.0 })
        .set_operand_at(0,
                        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 1.1, 2.0, 3.0 }),
                        ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::vec3))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constants(std::vector{ 1.0, 2.0, 3.0, 4.0 })
        .set_operand_at(
            0,
            std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
            ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::mat2),
            std::initializer_list<::SPIRVSimulator::DecorationInfo>{
                ::SPIRVSimulator::DecorationInfo{ .kind     = spv::DecorationMatrixStride,
                                                  .literals = { sizeof(double) * 2 } },
                ::SPIRVSimulator::DecorationInfo{ .kind = spv::DecorationRowMajor } })
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constants(std::vector{ 1.0, 3.0, 2.0, 4.0 })
        .set_operand_at(
            0,
            std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2),
            ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::mat2),
            std::initializer_list<::SPIRVSimulator::DecorationInfo>{
                ::SPIRVSimulator::DecorationInfo{ .kind     = spv::DecorationMatrixStride,
                                                  .literals = { sizeof(double) * 2 } },
                ::SPIRVSimulator::DecorationInfo{ .kind = spv::DecorationColMajor } })
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build()
};

INSTANTIATE_TEST_SUITE_P(Declaration, DeclarationTests, ValuesIn(test_cases));
