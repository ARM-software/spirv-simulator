#include <array>
#include <cstdint>
#include <limits>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "spirv.hpp"
#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

char opcode_to_char(spv::Op opcode)
{
    switch (opcode)
    {
        case spv::OpIAdd:
        case spv::OpFAdd:
        {
            return '+';
        }
        case spv::OpISub:
        case spv::OpFSub:
        {
            return '-';
        }
        case spv::OpIMul:
        case spv::OpFMul:
        {
            return '*';
        }
        case spv::OpSDiv:
        case spv::OpUDiv:
        case spv::OpFDiv:
        {
            return '/';
        }
        default:
            return ' ';
    }
}

class ArithmeticsMock : public SPIRVSimulatorMockBase
{
  public:
    MOCK_METHOD(void, SetValue, (uint32_t id, const ::SPIRVSimulator::Value& value), (override));
    MOCK_METHOD(::SPIRVSimulator::Value&, GetValue, (uint32_t id), (override));

    MOCK_METHOD(::SPIRVSimulator::Type, GetTypeByTypeId, (uint32_t id), (const override));
};

struct ArithmeticParams
{
    spv::Op               opcode;
    SPIRVSimulator::Value lhs;
    SPIRVSimulator::Value rhs;
    SPIRVSimulator::Value expected;
    uint32_t              expected_type_id;

    friend std::ostream& operator<<(std::ostream& os, ArithmeticParams const& p)
    {
        os << p.lhs << ' ' << opcode_to_char(p.opcode) << ' ' << p.rhs << " = " << p.expected;
        return os;
    }
};

enum Type : uint32_t
{
    boolean = 0,
    i64     = 1,
    u64     = 2,
    f64     = 3,

    ivec2 = 4,
    uvec2 = 5,
    vec2  = 6,

    num_types = 7
};

std::array<SPIRVSimulator::Type, Type::num_types> types_ = { SPIRVSimulator::Type::Bool(),
                                                             SPIRVSimulator::Type::Int(64, true),
                                                             SPIRVSimulator::Type::Int(64, false),
                                                             SPIRVSimulator::Type::Float(32),
                                                             SPIRVSimulator::Type::Vector(Type::i64, 2),
                                                             SPIRVSimulator::Type::Vector(Type::u64, 2),
                                                             SPIRVSimulator::Type::Vector(Type::f64, 2) };

SPIRVSimulator::Type get(uint32_t type_id)
{
    return types_[type_id];
}

class ArithmeticsTest : public TestWithParam<ArithmeticParams>
{
  public:
    ArithmeticsTest()
    {
        for (uint32_t i = 0; i < types_.size(); ++i)
        {
            EXPECT_CALL(mock, GetTypeByTypeId(i)).WillRepeatedly(Return(types_[i]));
        }
    }
    uint32_t NextId() { return id_counter++; }

  protected:
    uint32_t        id_counter = Type::num_types;
    ArithmeticsMock mock;
};

// clang-format off
std::vector<ArithmeticParams> scalar_cases = {
    { spv::Op::OpIAdd,
      SPIRVSimulator::Value(int64_t(1)),
      SPIRVSimulator::Value(int64_t(2)),
      SPIRVSimulator::Value(int64_t(3)),
      Type::i64 },
    { spv::Op::OpFAdd,
      SPIRVSimulator::Value(double(1.0)),
      SPIRVSimulator::Value(double(2.0)),
      SPIRVSimulator::Value(double(3.0)),
      Type::f64 },
    { spv::Op::OpISub,
      SPIRVSimulator::Value(uint64_t(1)),
      SPIRVSimulator::Value(int64_t(2)),
      SPIRVSimulator::Value(std::numeric_limits<uint64_t>::max()),
      Type::u64 },
    { spv::Op::OpISub,
      SPIRVSimulator::Value(int64_t(1)),
      SPIRVSimulator::Value(int64_t(2)),
      SPIRVSimulator::Value(int64_t(-1)),
      Type::i64 },
    { spv::Op::OpFSub,
      SPIRVSimulator::Value(1.0),
      SPIRVSimulator::Value(2.0),
      SPIRVSimulator::Value(-1.0),
      Type::f64 },
    { spv::Op::OpIMul,
      SPIRVSimulator::Value(int64_t(2)),
      SPIRVSimulator::Value(int64_t(2)),
      SPIRVSimulator::Value(int64_t(4)),
      Type::i64 },
    { spv::Op::OpFMul,
      SPIRVSimulator::Value(2.0),
      SPIRVSimulator::Value(2.0),
      SPIRVSimulator::Value(4.0),
      Type::f64 },
    { spv::Op::OpSDiv,
      SPIRVSimulator::Value(int64_t(-5)),
      SPIRVSimulator::Value(int64_t(2)),
      SPIRVSimulator::Value(int64_t(-2)),
      Type::i64 },
    { spv::Op::OpUDiv,
      SPIRVSimulator::Value(uint64_t(5)),
      SPIRVSimulator::Value(uint64_t(2)),
      SPIRVSimulator::Value(uint64_t(2)),
      Type::u64 },
};

// clang-format on

class ScalarTests : public ArithmeticsTest
{};

TEST_P(ScalarTests, ScalarOperations)
{
    const auto& parameters = GetParam();

    const uint32_t lhs_id = NextId();
    EXPECT_CALL(mock, GetValue(lhs_id)).WillRepeatedly(ReturnRefOfCopy(parameters.lhs));

    const uint32_t rhs_id = NextId();
    EXPECT_CALL(mock, GetValue(rhs_id)).WillRepeatedly(ReturnRefOfCopy(parameters.rhs));

    SPIRVSimulator::Value captured_value;
    EXPECT_CALL(mock, SetValue(_, _)).WillOnce(SaveArg<1>(&captured_value));

    const uint32_t              result_id = NextId();
    std::vector<uint32_t>       words{ parameters.opcode, parameters.expected_type_id, result_id, lhs_id, rhs_id };
    SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                      .word_count = static_cast<uint16_t>(words.size()),
                                      .words      = words };
    mock.ExecuteInstruction(inst);

    if (parameters.expected_type_id == Type::i64)
    {
        EXPECT_EQ(std::get<int64_t>(captured_value), std::get<int64_t>(parameters.expected));
    }
    else if (parameters.expected_type_id == Type::u64)
    {
        EXPECT_EQ(std::get<uint64_t>(captured_value), std::get<uint64_t>(parameters.expected));
    }
    else if (parameters.expected_type_id == Type::f64)
    {
        EXPECT_EQ(std::get<double>(captured_value), std::get<double>(parameters.expected));
    }
    else if (parameters.expected_type_id == Type::boolean)
    {
        EXPECT_EQ(std::get<uint64_t>(captured_value), std::get<uint64_t>(parameters.expected));
    }
}

INSTANTIATE_TEST_SUITE_P(Arithmetics, ScalarTests, ValuesIn(scalar_cases));

class VectorTests : public ArithmeticsTest
{};

// clang-format off
std::vector<ArithmeticParams> vector_cases = {
    { spv::Op::OpIAdd,
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2, 2 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2, 2 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 4, 4 })),
      Type::ivec2 },
    { spv::Op::OpFAdd,
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2.0, 2.0 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2.0, 2.0 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 4.0, 4.0 })),
      Type::vec2 },
    { spv::Op::OpISub,
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 1, 1 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2, 2 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ -1, -1 })),
      Type::ivec2 },
    { spv::Op::OpFSub,
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 1.0, 1.0 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2.0, 2.0 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ -1.0, -1.0 })),
      Type::vec2 },
    { spv::Op::OpIMul,
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ -1, -1 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2, 2 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ -2, -2 })),
      Type::ivec2 },
    { spv::Op::OpFMul,
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ -1.0, -1.0 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ 2.0, 2.0 })),
      SPIRVSimulator::Value(std::make_shared<SPIRVSimulator::VectorV>(std::vector<SPIRVSimulator::Value>{ -2.0, -2.0 })),
      Type::vec2 }
};

// clang-format on

TEST_P(VectorTests, VectorOperations)
{
    const auto& parameters = GetParam();

    const uint32_t lhs_id = NextId();
    EXPECT_CALL(mock, GetValue(lhs_id)).WillRepeatedly(::ReturnRefOfCopy(parameters.lhs));
    const uint32_t rhs_id = NextId();
    EXPECT_CALL(mock, GetValue(rhs_id)).WillRepeatedly(::ReturnRefOfCopy(parameters.rhs));

    SPIRVSimulator::Value captured_value;
    EXPECT_CALL(mock, SetValue(_, _)).WillOnce(SaveArg<1>(&captured_value));

    const uint32_t              result_id = NextId();
    std::vector<uint32_t>       words{ parameters.opcode, parameters.expected_type_id, result_id, lhs_id, rhs_id };
    SPIRVSimulator::Instruction instruction{ .opcode = parameters.opcode, .word_count = 5, .words = words };

    mock.ExecuteInstruction(instruction);

    const auto& result = std::get<std::shared_ptr<SPIRVSimulator::VectorV>>(captured_value);
    const std::vector<SPIRVSimulator::Value>& elems = result->elems;

    std::shared_ptr<SPIRVSimulator::VectorV> expected =
        std::get<std::shared_ptr<SPIRVSimulator::VectorV>>(parameters.expected);
    if (parameters.expected_type_id == ivec2)
    {
        for (uint32_t i = 0; i < elems.size(); ++i)
        {
            EXPECT_EQ(std::get<int64_t>(elems[i]), std::get<int64_t>(expected->elems[i]));
        }
    }
    else if (parameters.expected_type_id == uvec2)
    {
        for (uint32_t i = 0; i < elems.size(); ++i)
        {
            EXPECT_EQ(std::get<uint64_t>(elems[i]), std::get<uint64_t>(expected->elems[i]));
        }
    }
    else if (parameters.expected_type_id == vec2)
    {
        for (uint32_t i = 0; i < elems.size(); ++i)
        {
            EXPECT_EQ(std::get<double>(elems[i]), std::get<double>(expected->elems[i]));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Arithmetics, VectorTests, ValuesIn(vector_cases));