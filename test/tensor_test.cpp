#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class TensorCrashTests : public SPIRVSimulatorMockBase, public ::testing::Test
{
  protected:
    static constexpr uint32_t kTensorId = 200;
    static constexpr uint32_t kReadResultTypeId = 201;
    static constexpr uint32_t kReadResultId = 202;
    static constexpr uint32_t kCoordsId = 203;
    static constexpr uint32_t kRankId = 204;
    static constexpr uint32_t kCoordCountId = 205;
    static constexpr uint32_t kObjectId = 206;
    static constexpr uint32_t kQueryResultTypeId = 207;
    static constexpr uint32_t kQueryResultId = 208;
    static constexpr uint32_t kDimensionId = 209;

    void SetUp() override
    {
        if (values_.size() <= kQueryResultId)
        {
            values_.resize(kQueryResultId + 1, std::monostate{});
        }

        if (value_meta_.size() <= kQueryResultId)
        {
            value_meta_.resize(kQueryResultId + 1, { 0 });
        }

        rank_value_             = static_cast<uint64_t>(2);
        coord_count_value_      = static_cast<uint64_t>(2);
        tensor_result_type_     = MakeTensorType(CommonTypes::f32, kRankId);
        read_result_type_       = ::SPIRVSimulator::Type::Float(32);
        coords_result_type_     = ::SPIRVSimulator::Type::Array(CommonTypes::i32, kCoordCountId);
        object_result_type_     = ::SPIRVSimulator::Type::Float(32);
        query_result_type_      = ::SPIRVSimulator::Type::Int(32, false);
        query_tensor_type_      = MakeTensorType(CommonTypes::f32, kRankId);
        dimension_result_type_  = ::SPIRVSimulator::Type::Int(32, false);

        EXPECT_CALL(*this, GetValue(kRankId)).WillRepeatedly(ReturnRef(rank_value_));
        EXPECT_CALL(*this, GetValue(kCoordCountId)).WillRepeatedly(ReturnRef(coord_count_value_));

        EXPECT_CALL(*this, GetTypeByResultId(kTensorId)).WillRepeatedly(ReturnRef(tensor_result_type_));
        EXPECT_CALL(*this, GetTypeByResultId(kCoordsId)).WillRepeatedly(ReturnRef(coords_result_type_));
        EXPECT_CALL(*this, GetTypeByResultId(kObjectId)).WillRepeatedly(ReturnRef(object_result_type_));

        EXPECT_CALL(*this, GetTypeByTypeId(kReadResultTypeId)).WillRepeatedly(ReturnRef(read_result_type_));
        EXPECT_CALL(*this, GetTypeByTypeId(kQueryResultTypeId)).WillRepeatedly(ReturnRef(query_result_type_));
        EXPECT_CALL(*this, GetTypeByTypeId(kTensorId)).WillRepeatedly(ReturnRef(query_tensor_type_));
        EXPECT_CALL(*this, GetTypeByTypeId(kDimensionId)).WillRepeatedly(ReturnRef(dimension_result_type_));
    }

    static ::SPIRVSimulator::Type MakeTensorType(uint32_t element_type_id, std::optional<uint32_t> rank_id)
    {
        ::SPIRVSimulator::Type type;
        type.kind   = ::SPIRVSimulator::Type::Kind::TensorARM;
        type.tensor = { .element_type_id = element_type_id, .rank_id = rank_id, .shape_id = std::nullopt };
        return type;
    }

    ::SPIRVSimulator::Value rank_value_;
    ::SPIRVSimulator::Value coord_count_value_;
    ::SPIRVSimulator::Type  tensor_result_type_;
    ::SPIRVSimulator::Type  read_result_type_;
    ::SPIRVSimulator::Type  coords_result_type_;
    ::SPIRVSimulator::Type  object_result_type_;
    ::SPIRVSimulator::Type  query_result_type_;
    ::SPIRVSimulator::Type  query_tensor_type_;
    ::SPIRVSimulator::Type  dimension_result_type_;
};

TEST_F(TensorCrashTests, TypeTensorRejectsTooFewOperands)
{
    const std::vector<uint32_t> instruction_words = { static_cast<uint32_t>(spv::Op::OpTypeTensorARM), kTensorId };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTypeTensorARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV Simulator: OpTypeTensorARM requires at least 3 arguments");
}

TEST_F(TensorCrashTests, TensorReadRejectsNonScalarResult)
{
    read_result_type_ = ::SPIRVSimulator::Type::Vector(CommonTypes::f32, 2);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorReadARM), kReadResultTypeId, kReadResultId, kTensorId, kCoordsId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorReadARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorRead result must be scalar or array of scalars");
}

TEST_F(TensorCrashTests, TensorReadRejectsUnrankedTensor)
{
    tensor_result_type_ = MakeTensorType(CommonTypes::f32, std::nullopt);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorReadARM), kReadResultTypeId, kReadResultId, kTensorId, kCoordsId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorReadARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorRead tensor must be ranked");
}

TEST_F(TensorCrashTests, TensorReadRejectsCoordinateCountMismatch)
{
    coord_count_value_ = static_cast<uint64_t>(3);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorReadARM), kReadResultTypeId, kReadResultId, kTensorId, kCoordsId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorReadARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorRead number of coords must be equal to rank of tensor");
}

TEST_F(TensorCrashTests, TensorReadRejectsNonIntegerCoordinates)
{
    coords_result_type_ = ::SPIRVSimulator::Type::Array(CommonTypes::f32, kCoordCountId);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorReadARM), kReadResultTypeId, kReadResultId, kTensorId, kCoordsId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorReadARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorRead coords must be integer type scalars");
}

TEST_F(TensorCrashTests, TensorReadRejectsMakeElementAvailableOperand)
{
    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorReadARM), kReadResultTypeId, kReadResultId, kTensorId, kCoordsId, 0x4
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorReadARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: MakeElementAvailableARM illegal for TensorRead");
}

TEST_F(TensorCrashTests, TensorWriteRejectsUnrankedTensor)
{
    tensor_result_type_ = MakeTensorType(CommonTypes::f32, std::nullopt);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorWriteARM), kTensorId, kCoordsId, kObjectId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorWriteARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorWrite tensor must be ranked");
}

TEST_F(TensorCrashTests, TensorWriteRejectsCoordinateCountMismatch)
{
    coord_count_value_ = static_cast<uint64_t>(3);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorWriteARM), kTensorId, kCoordsId, kObjectId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorWriteARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);},
                           "SPIRV simulator: TensorWrite number of coords must be equal to rank of tensor");
}

TEST_F(TensorCrashTests, TensorWriteRejectsNonIntegerCoordinates)
{
    coords_result_type_ = ::SPIRVSimulator::Type::Array(CommonTypes::f32, kCoordCountId);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorWriteARM), kTensorId, kCoordsId, kObjectId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorWriteARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorWrite coords must be integer type scalars");
}

TEST_F(TensorCrashTests, TensorWriteRejectsNonScalarObject)
{
    object_result_type_ = ::SPIRVSimulator::Type::Vector(CommonTypes::f32, 2);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorWriteARM), kTensorId, kCoordsId, kObjectId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorWriteARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorWrite result must be scalar or array of scalars");
}

TEST_F(TensorCrashTests, TensorWriteRejectsScalarObjectWithWrongElementType)
{
    object_result_type_ = ::SPIRVSimulator::Type::Int(32, true);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorWriteARM), kTensorId, kCoordsId, kObjectId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorWriteARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorWrite object must be type contained in tensor");
}

TEST_F(TensorCrashTests, TensorWriteRejectsArrayObjectWithWrongElementType)
{
    object_result_type_ = ::SPIRVSimulator::Type::Array(CommonTypes::i32, kCoordCountId);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorWriteARM), kTensorId, kCoordsId, kObjectId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorWriteARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorWrite object must be type contained in tensor");
}

TEST_F(TensorCrashTests, TensorQuerySizeRejectsNonIntegerResultType)
{
    query_result_type_ = ::SPIRVSimulator::Type::Float(32);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorQuerySizeARM), kQueryResultTypeId, kQueryResultId, kTensorId, kDimensionId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorQuerySizeARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorQuerySize result type must be integer scalar");
}

TEST_F(TensorCrashTests, TensorQuerySizeRejectsUnrankedTensor)
{
    query_tensor_type_ = MakeTensorType(CommonTypes::f32, std::nullopt);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorQuerySizeARM), kQueryResultTypeId, kQueryResultId, kTensorId, kDimensionId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorQuerySizeARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorQuerySize tensor must be ranked");
}

TEST_F(TensorCrashTests, TensorQuerySizeRejectsNonIntegerDimension)
{
    dimension_result_type_ = ::SPIRVSimulator::Type::Float(32);

    const std::vector<uint32_t> instruction_words = {
        static_cast<uint32_t>(spv::Op::OpTensorQuerySizeARM), kQueryResultTypeId, kQueryResultId, kTensorId, kDimensionId
    };
    const auto instruction = ::SPIRVSimulator::Instruction{ .opcode     = spv::Op::OpTensorQuerySizeARM,
                                                            .word_count = static_cast<uint16_t>(instruction_words.size()),
                                                            .words      = instruction_words };

    EXPECT_DEATH( {this->ExecuteInstruction(instruction);}, "SPIRV simulator: TensorQuerySize dimension must be given as integer scalar");
}
