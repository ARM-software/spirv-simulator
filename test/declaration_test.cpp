#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <sys/types.h>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class DeclarationTests : public SPIRVSimulatorMockBase, public TestWithParam<TestParameters>
{};

class OpConstantNullDeclarationTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    static constexpr uint32_t kPointerTypeId       = 1001;
    static constexpr uint32_t kStructTypeId        = 1002;
    static constexpr uint32_t kArrayTypeId         = 1003;
    static constexpr uint32_t kArrayLengthId       = 1004;
    static constexpr uint32_t kDirectResultId      = 2001;
    static constexpr uint32_t kStructResultId      = 2002;
    static constexpr uint32_t kArrayResultId       = 2003;
    static constexpr uint32_t kPhysicalResultId    = 2004;
    static constexpr uint64_t kArrayLength         = 3;
    static constexpr const char* kPhysicalNullError =
        "SPIRV Simulator: OpConstantNull - PyhsicalStorageBuffer not allowed";

    void RegisterType(uint32_t type_id, const ::SPIRVSimulator::Type& type)
    {
        types_[type_id] = type;
        EXPECT_CALL(*this, GetTypeByTypeId(type_id)).WillRepeatedly(ReturnRef(types_[type_id]));
    }

    void RegisterStructType(uint32_t type_id, const std::vector<uint32_t>& members)
    {
        RegisterType(type_id, ::SPIRVSimulator::Type::Struct(type_id));
        struct_members_[type_id] = members;
    }

    void RegisterArrayType(uint32_t type_id, uint32_t element_type_id, uint32_t length_id, uint64_t length)
    {
        RegisterType(type_id, ::SPIRVSimulator::Type::Array(element_type_id, length_id));
        EXPECT_CALL(*this, GetValue(length_id)).WillRepeatedly(ReturnRefOfCopy(::SPIRVSimulator::Value(length)));
    }
};

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
        .set_operand_at(0,
                        static_cast<uint64_t>(0),
                        ::SPIRVSimulator::Type::Pointer(spv::StorageClassTaskPayloadWorkgroupEXT, CommonTypes::u32))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassTaskPayloadWorkgroupEXT),
                        CommonTypes::storage_class)
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
        .build(),
    TestParametersBuilder()
        .set_opcode(spv::Op::OpVariable)
        .add_push_constants(std::vector{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 })
        .set_operand_at(
            0,
            std::make_shared<::SPIRVSimulator::MatrixV>(
                std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 3),
            ::SPIRVSimulator::Type::Pointer(spv::StorageClassPushConstant, CommonTypes::mat2x3))
        .set_operand_at(1,
                        static_cast<uint64_t>(spv::StorageClassPushConstant),
                        CommonTypes::storage_class) // Storage class is always a raw uint
        .build()
};

INSTANTIATE_TEST_SUITE_P(Declaration, DeclarationTests, ValuesIn(test_cases));

TEST_F(OpConstantNullDeclarationTests, LogicalPointer)
{
    RegisterType(kPointerTypeId, ::SPIRVSimulator::Type::Pointer(spv::StorageClassGeneric, CommonTypes::u32));

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(kDirectResultId, _, true)).WillOnce(SaveArg<1>(&captured_value));

    std::vector<uint32_t> instruction_words = { static_cast<uint32_t>(spv::Op::OpConstantNull), kPointerTypeId, kDirectResultId };

    ::SPIRVSimulator::Instruction instr = { .opcode     = spv::Op::OpConstantNull,
                                            .word_count = 3,
                                            .words      = instruction_words };
    this->ExecuteInstruction(instr);

    EXPECT_EQ(captured_value,
              ::SPIRVSimulator::Value(::SPIRVSimulator::PointerV{
                  0, 0, kPointerTypeId, kDirectResultId, spv::StorageClassGeneric, {} }));
}

TEST_F(OpConstantNullDeclarationTests, StructContainingLogicalPointer)
{
    RegisterType(kPointerTypeId, ::SPIRVSimulator::Type::Pointer(spv::StorageClassGeneric, CommonTypes::u32));
    RegisterStructType(kStructTypeId, { kPointerTypeId });

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(kStructResultId, _, true)).WillOnce(SaveArg<1>(&captured_value));

    std::vector<uint32_t> instruction_words = { static_cast<uint32_t>(spv::Op::OpConstantNull), kStructTypeId, kStructResultId };

    ::SPIRVSimulator::Instruction instr = { .opcode     = spv::Op::OpConstantNull,
                                            .word_count = 3,
                                            .words      = instruction_words };
    this->ExecuteInstruction(instr);

    EXPECT_EQ(captured_value,
              ::SPIRVSimulator::Value(std::make_shared<::SPIRVSimulator::AggregateV>(
                  std::initializer_list<::SPIRVSimulator::Value>{ ::SPIRVSimulator::PointerV{
                      0, 0, kPointerTypeId, kStructResultId, spv::StorageClassGeneric, {} } })));
}

TEST_F(OpConstantNullDeclarationTests, ArrayOfStructsContainingLogicalPointer)
{
    RegisterType(kPointerTypeId, ::SPIRVSimulator::Type::Pointer(spv::StorageClassGeneric, CommonTypes::u32));
    RegisterStructType(kStructTypeId, { kPointerTypeId });
    RegisterArrayType(kArrayTypeId, kStructTypeId, kArrayLengthId, kArrayLength);

    ::SPIRVSimulator::Value captured_value;
    EXPECT_CALL(*this, SetValue(kArrayResultId, _, true)).WillOnce(SaveArg<1>(&captured_value));

    std::vector<uint32_t> instruction_words = { static_cast<uint32_t>(spv::Op::OpConstantNull), kArrayTypeId, kArrayResultId };

    ::SPIRVSimulator::Instruction instr = { .opcode     = spv::Op::OpConstantNull,
                                            .word_count = 3,
                                            .words      = instruction_words };
    this->ExecuteInstruction(instr);

    const auto* array = std::get_if<std::shared_ptr<::SPIRVSimulator::AggregateV>>(&captured_value);
    ASSERT_NE(array, nullptr);
    ASSERT_TRUE(*array);
    ASSERT_EQ((*array)->elems.size(), kArrayLength);

    for (const auto& element : (*array)->elems)
    {
        const auto* nested_struct = std::get_if<std::shared_ptr<::SPIRVSimulator::AggregateV>>(&element);
        ASSERT_NE(nested_struct, nullptr);
        ASSERT_TRUE(*nested_struct);
        ASSERT_EQ((*nested_struct)->elems.size(), 1u);

        const auto* pointer = std::get_if<::SPIRVSimulator::PointerV>(&(*nested_struct)->elems[0]);
        ASSERT_NE(pointer, nullptr);
        EXPECT_EQ(pointer->pointer_handle, 0u);
        EXPECT_EQ(pointer->pointee_flags, 0u);
        EXPECT_EQ(pointer->base_type_id, kPointerTypeId);
        EXPECT_EQ(pointer->base_result_id, kArrayResultId);
        EXPECT_EQ(pointer->storage_class, spv::StorageClassGeneric);
        EXPECT_TRUE(pointer->idx_path.empty());
    }
}

TEST_F(OpConstantNullDeclarationTests, CrashPhysicalStorageBufferPointers)
{
    RegisterType(kPointerTypeId,
                 ::SPIRVSimulator::Type::Pointer(spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64));

    std::vector<uint32_t> instruction_words = { static_cast<uint32_t>(spv::Op::OpConstantNull), kPointerTypeId, kPhysicalResultId };

    ::SPIRVSimulator::Instruction instr = { .opcode     = spv::Op::OpConstantNull,
                                            .word_count = 3,
                                            .words      = instruction_words };
#ifndef NDEBUG
    EXPECT_DEATH({ this->ExecuteInstruction(instr); },
                 kPhysicalNullError);
#else
    try
    {
        this->ExecuteInstruction(instr);
        FAIL() << "Expected OpConstantNull to reject PhysicalStorageBuffer pointers";
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_THAT(e.what(), HasSubstr(kPhysicalNullError));
    }
#endif
}
