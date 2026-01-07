#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "spirv.hpp"
#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

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
