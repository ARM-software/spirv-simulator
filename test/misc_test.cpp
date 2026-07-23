#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class GLSLExtInstructionTests : public SPIRVSimulatorMockBase, public ::testing::Test
{
  public:
    void ExecuteGLSLExtInstruction(uint32_t                         type_id,
                                   uint32_t                         result_id,
                                   uint32_t                         instruction_literal,
                                   const std::span<const uint32_t>& operands)
    {
        GLSLExtHandler(type_id, result_id, instruction_literal, operands);
    }
};

TEST_F(GLSLExtInstructionTests, DeterminantHandlesSquareFloatMatrices)
{
    constexpr uint32_t result_id  = 90;
    constexpr uint32_t operand_id = 91;

    std::vector<std::pair<uint32_t, ::SPIRVSimulator::Value>> matrices{
        { CommonTypes::mat2,
          std::make_shared<::SPIRVSimulator::MatrixV>(std::initializer_list<double>{ 1.0, 2.0, 3.0, 4.0 }, 2) },
        { CommonTypes::mat3,
          std::make_shared<::SPIRVSimulator::MatrixV>(
              std::initializer_list<double>{ 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0 }, 3) },
        { CommonTypes::mat4,
          std::make_shared<::SPIRVSimulator::MatrixV>(
              std::initializer_list<double>{
                  1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 5.0, 6.0, 0.0, 0.0, 3.0, 7.0, 0.0, 0.0, 0.0, 4.0 },
              4) },
    };
    const std::vector<double> expected{ -2.0, 22.0, 24.0 };

    for (size_t i = 0; i < matrices.size(); ++i)
    {
        const uint32_t          matrix_type_id = matrices[i].first;
        const auto&             matrix_type    = types_.at(matrix_type_id);
        const uint32_t          column_type_id = matrix_type.matrix.col_type_id;
        ::SPIRVSimulator::Value captured_result;

        EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::f64))
            .Times(2)
            .WillRepeatedly(ReturnRef(types_.at(CommonTypes::f64)));
        EXPECT_CALL(*this, GetTypeByTypeId(column_type_id)).WillOnce(ReturnRef(types_.at(column_type_id)));
        EXPECT_CALL(*this, GetValue(operand_id)).WillOnce(ReturnRef(matrices[i].second));
        EXPECT_CALL(*this, GetTypeByResultId(operand_id)).WillOnce(ReturnRef(matrix_type));
        EXPECT_CALL(*this, SetValue(result_id, _, true)).WillOnce(SaveArg<1>(&captured_result));
        EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_id)));

        const std::vector<uint32_t> operands{ operand_id };
        ExecuteGLSLExtInstruction(CommonTypes::f64, result_id, 33, operands);

        ASSERT_TRUE(std::holds_alternative<double>(captured_result));
        EXPECT_DOUBLE_EQ(std::get<double>(captured_result), expected[i]);
        Mock::VerifyAndClearExpectations(this);
    }
}

TEST_F(GLSLExtInstructionTests, FMinAcceptsFloatVectorElements)
{
    constexpr uint32_t result_id    = 100;
    constexpr uint32_t operand_1_id = 101;
    constexpr uint32_t operand_2_id = 102;

    ::SPIRVSimulator::Value operand_1 =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 4.0, -2.0, 8.0 });
    ::SPIRVSimulator::Value operand_2 =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<double>{ 3.0, -1.0, 7.0 });
    ::SPIRVSimulator::Value captured_result;

    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::vec3)).WillOnce(ReturnRef(types_.at(CommonTypes::vec3)));
    EXPECT_CALL(*this, GetValue(operand_1_id)).WillOnce(ReturnRef(operand_1));
    EXPECT_CALL(*this, GetValue(operand_2_id)).WillOnce(ReturnRef(operand_2));
    EXPECT_CALL(*this, SetValue(result_id, _, true)).WillOnce(SaveArg<1>(&captured_result));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_1_id)));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_2_id)));

    const std::vector<uint32_t> operands{ operand_1_id, operand_2_id };
    ExecuteGLSLExtInstruction(CommonTypes::vec3, result_id, 37, operands);

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result));
    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result);
    ASSERT_EQ(result->elems.size(), 3u);
    EXPECT_EQ(std::get<double>(result->elems[0]), 3.0);
    EXPECT_EQ(std::get<double>(result->elems[1]), -2.0);
    EXPECT_EQ(std::get<double>(result->elems[2]), 7.0);
}

TEST_F(GLSLExtInstructionTests, SMaxHandlesSignedIntegerVectors)
{
    constexpr uint32_t result_id    = 200;
    constexpr uint32_t operand_1_id = 201;
    constexpr uint32_t operand_2_id = 202;

    ::SPIRVSimulator::Value operand_1 =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -4, 2, 8 });
    ::SPIRVSimulator::Value operand_2 =
        std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<int64_t>{ -3, -1, 7 });
    ::SPIRVSimulator::Value captured_result;

    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::ivec3)).WillOnce(ReturnRef(types_.at(CommonTypes::ivec3)));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::i64)).WillOnce(ReturnRef(types_.at(CommonTypes::i64)));
    EXPECT_CALL(*this, GetValue(operand_1_id)).WillOnce(ReturnRef(operand_1));
    EXPECT_CALL(*this, GetValue(operand_2_id)).WillOnce(ReturnRef(operand_2));
    EXPECT_CALL(*this, SetValue(result_id, _, true)).WillOnce(SaveArg<1>(&captured_result));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_1_id)));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_2_id)));

    const std::vector<uint32_t> operands{ operand_1_id, operand_2_id };
    ExecuteGLSLExtInstruction(CommonTypes::ivec3, result_id, 42, operands);

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result));
    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result);
    ASSERT_EQ(result->elems.size(), 3u);
    EXPECT_EQ(std::get<int64_t>(result->elems[0]), -3);
    EXPECT_EQ(std::get<int64_t>(result->elems[1]), 2);
    EXPECT_EQ(std::get<int64_t>(result->elems[2]), 8);
}

TEST_F(GLSLExtInstructionTests, SMaxHandlesUnsignedResultType)
{
    constexpr uint32_t result_type_id = 250;
    constexpr uint32_t result_id      = 251;
    constexpr uint32_t operand_1_id   = 252;
    constexpr uint32_t operand_2_id   = 253;

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 3);
    ::SPIRVSimulator::Value operand_1 = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 0xffffffffu, 2, 0x80000000u });
    ::SPIRVSimulator::Value operand_2 = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<int64_t>{ -3, -1, 7 });
    ::SPIRVSimulator::Value captured_result;

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::u32)).WillOnce(ReturnRef(types_.at(CommonTypes::u32)));
    EXPECT_CALL(*this, GetValue(operand_1_id)).WillOnce(ReturnRef(operand_1));
    EXPECT_CALL(*this, GetValue(operand_2_id)).WillOnce(ReturnRef(operand_2));
    EXPECT_CALL(*this, SetValue(result_id, _, true)).WillOnce(SaveArg<1>(&captured_result));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_1_id)));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_2_id)));

    const std::vector<uint32_t> operands{ operand_1_id, operand_2_id };
    ExecuteGLSLExtInstruction(result_type_id, result_id, 42, operands);

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result));
    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result);
    ASSERT_EQ(result->elems.size(), 3u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[0]), 0xffffffffu);
    EXPECT_EQ(std::get<uint64_t>(result->elems[1]), 2u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[2]), 7u);
}

TEST_F(GLSLExtInstructionTests, FindSMsbInterpretsUnsignedOperandAsSigned)
{
    constexpr uint32_t result_type_id = 300;
    constexpr uint32_t result_id      = 301;
    constexpr uint32_t operand_id     = 302;

    const ::SPIRVSimulator::Type result_type  = ::SPIRVSimulator::Type::Vector(CommonTypes::i32, 3);
    const ::SPIRVSimulator::Type operand_type = ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 3);
    ::SPIRVSimulator::Value operand = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<uint64_t>{ 0xffffffffu, 0xfffffffcu, 8 });
    ::SPIRVSimulator::Value captured_result;

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(operand_id)).WillOnce(ReturnRef(operand_type));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::u32)).WillOnce(ReturnRef(types_.at(CommonTypes::u32)));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::i32)).WillOnce(ReturnRef(types_.at(CommonTypes::i32)));
    EXPECT_CALL(*this, GetValue(operand_id)).WillOnce(ReturnRef(operand));
    EXPECT_CALL(*this, SetValue(result_id, _, true)).WillOnce(SaveArg<1>(&captured_result));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_id)));

    const std::vector<uint32_t> operands{ operand_id };
    ExecuteGLSLExtInstruction(result_type_id, result_id, 74, operands);

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result));
    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result);
    ASSERT_EQ(result->elems.size(), 3u);
    EXPECT_EQ(std::get<int64_t>(result->elems[0]), -1);
    EXPECT_EQ(std::get<int64_t>(result->elems[1]), 1);
    EXPECT_EQ(std::get<int64_t>(result->elems[2]), 3);
}

TEST_F(GLSLExtInstructionTests, FindUMsbInterpretsSignedOperandAsUnsigned)
{
    constexpr uint32_t result_type_id = 400;
    constexpr uint32_t result_id      = 401;
    constexpr uint32_t operand_id     = 402;

    const ::SPIRVSimulator::Type result_type  = ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 3);
    const ::SPIRVSimulator::Type operand_type = ::SPIRVSimulator::Type::Vector(CommonTypes::i32, 3);
    ::SPIRVSimulator::Value operand = std::make_shared<::SPIRVSimulator::VectorV>(
        std::initializer_list<int64_t>{ 0, 4, -2147483648ll });
    ::SPIRVSimulator::Value captured_result;

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillOnce(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByResultId(operand_id)).WillOnce(ReturnRef(operand_type));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::i32)).WillOnce(ReturnRef(types_.at(CommonTypes::i32)));
    EXPECT_CALL(*this, GetTypeByTypeId(CommonTypes::u32)).WillOnce(ReturnRef(types_.at(CommonTypes::u32)));
    EXPECT_CALL(*this, GetValue(operand_id)).WillOnce(ReturnRef(operand));
    EXPECT_CALL(*this, SetValue(result_id, _, true)).WillOnce(SaveArg<1>(&captured_result));
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(operand_id)));

    const std::vector<uint32_t> operands{ operand_id };
    ExecuteGLSLExtInstruction(result_type_id, result_id, 75, operands);

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result));
    const auto& result = std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(captured_result);
    ASSERT_EQ(result->elems.size(), 3u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[0]), 0xffffffffu);
    EXPECT_EQ(std::get<uint64_t>(result->elems[1]), 2u);
    EXPECT_EQ(std::get<uint64_t>(result->elems[2]), 31u);
}

class MemoryBarrierTests : public SPIRVSimulatorMockBase, public ::testing::Test
{};

TEST_F(MemoryBarrierTests, ActsAsNoOpInSimulator)
{
    auto parameters = TestParametersBuilder()
                          .set_opcode(spv::Op::OpMemoryBarrier)
                          .set_operand_at(0, static_cast<uint64_t>(spv::ScopeDevice), CommonTypes::storage_class)
                          .set_operand_at(1,
                                          static_cast<uint64_t>(spv::MemorySemanticsAcquireReleaseMask),
                                          CommonTypes::storage_class)
                          .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                        .word_count = static_cast<uint16_t>(words.size()),
                                        .words      = words };

    EXPECT_CALL(*this, SetValue(_, _, _)).Times(0);
    EXPECT_CALL(*this, TransferFlags(_, ::testing::A<uint32_t>())).Times(0);
    EXPECT_CALL(*this, TransferFlags(_, ::testing::A<uint64_t>())).Times(0);

    ASSERT_TRUE(this->ExecuteInstruction(inst));
}

static std::vector<uint32_t> PackStringToWords(const std::string& literal)
{
    std::vector<uint32_t> words;
    uint32_t              word     = 0;
    uint32_t              byte_idx = 0;

    for (size_t i = 0; i <= literal.size(); ++i) // include NUL
    {
        uint8_t byte = (i < literal.size()) ? static_cast<uint8_t>(literal[i]) : 0;
        word |= static_cast<uint32_t>(byte) << (8 * byte_idx);
        byte_idx += 1;

        if (byte_idx == 4)
        {
            words.push_back(word);
            word     = 0;
            byte_idx = 0;
        }
    }

    if (byte_idx != 0)
    {
        words.push_back(word);
    }

    return words;
}

class StringTests : public SPIRVSimulatorMockBase, public ::testing::Test
{};

TEST_F(StringTests, StoresLiteralAndPrintsInOpLine)
{
    const uint32_t        string_id = 1;
    const std::string     literal   = "hello.spvasm";
    const auto            literal_words = PackStringToWords(literal);
    const uint16_t        string_word_count = static_cast<uint16_t>(2 + literal_words.size());
    std::vector<uint32_t> string_instruction_words;
    string_instruction_words.reserve(string_word_count);
    string_instruction_words.push_back(
        static_cast<uint32_t>((string_word_count << 16) | static_cast<uint32_t>(spv::Op::OpString)));
    string_instruction_words.push_back(string_id);
    string_instruction_words.insert(string_instruction_words.end(), literal_words.begin(), literal_words.end());

    ::SPIRVSimulator::Instruction string_inst{ .opcode     = spv::Op::OpString,
                                               .word_count = string_word_count,
                                               .words      = string_instruction_words };

    ASSERT_TRUE(this->ExecuteInstruction(string_inst));

    const uint16_t line_word_count = 4;
    std::vector<uint32_t> line_instruction_words{
        static_cast<uint32_t>((line_word_count << 16) | static_cast<uint32_t>(spv::Op::OpLine)),
        string_id,
        5,
        7,
    };

    ::SPIRVSimulator::Instruction line_inst{ .opcode     = spv::Op::OpLine,
                                             .word_count = line_word_count,
                                             .words      = line_instruction_words };

    std::stringstream capture;
    auto*             old_buf = std::cout.rdbuf(capture.rdbuf());
    this->PrintInstruction(line_inst);
    std::cout.rdbuf(old_buf);

    EXPECT_THAT(capture.str(), HasSubstr(literal));
    EXPECT_THAT(capture.str(), HasSubstr("5:7"));
}

TEST_F(StringTests, OpLineExecutesAsNoOp)
{
    const uint32_t        string_id = 1;
    const std::string     literal   = "path/to/file.glsl";
    const auto            literal_words = PackStringToWords(literal);
    const uint16_t        string_word_count = static_cast<uint16_t>(2 + literal_words.size());
    std::vector<uint32_t> string_instruction_words;
    string_instruction_words.reserve(string_word_count);
    string_instruction_words.push_back(
        static_cast<uint32_t>((string_word_count << 16) | static_cast<uint32_t>(spv::Op::OpString)));
    string_instruction_words.push_back(string_id);
    string_instruction_words.insert(string_instruction_words.end(), literal_words.begin(), literal_words.end());

    ::SPIRVSimulator::Instruction string_inst{ .opcode     = spv::Op::OpString,
                                               .word_count = string_word_count,
                                               .words      = string_instruction_words };

    EXPECT_CALL(*this, SetValue(_, _, _)).Times(0);
    EXPECT_CALL(*this, TransferFlags(_, ::testing::A<uint32_t>())).Times(0);
    EXPECT_CALL(*this, TransferFlags(_, ::testing::A<uint64_t>())).Times(0);

    ASSERT_TRUE(this->ExecuteInstruction(string_inst));

    const uint16_t line_word_count = 4;
    std::vector<uint32_t> line_instruction_words{
        static_cast<uint32_t>((line_word_count << 16) | static_cast<uint32_t>(spv::Op::OpLine)),
        string_id,
        12,
        34,
    };

    ::SPIRVSimulator::Instruction line_inst{ .opcode     = spv::Op::OpLine,
                                             .word_count = line_word_count,
                                             .words      = line_instruction_words };

    EXPECT_TRUE(this->ExecuteInstruction(line_inst));
}
