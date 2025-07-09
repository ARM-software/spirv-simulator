#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <variant>

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
        case spv::OpSNegate:
        {
            return '-';
        }
        case spv::OpIMul:
        case spv::OpFMul:
        case spv::OpVectorTimesScalar:
        case spv::OpDot:
        {
            return '*';
        }
        case spv::OpMatrixTimesVector:
        case spv::OpMatrixTimesMatrix:
        {
            return 'x';
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

enum Type : uint32_t
{
    void_t  = 0,
    boolean = 1,
    i64     = 2,
    u64     = 3,
    f64     = 4,

    ivec2 = 5,
    uvec2 = 6,
    vec2  = 7,

    mat2 = 8,

    num_types = 9
};

std::array<SPIRVSimulator::Type, Type::num_types> types_ = { SPIRVSimulator::Type(),
                                                             SPIRVSimulator::Type::Bool(),
                                                             SPIRVSimulator::Type::Int(64, true),
                                                             SPIRVSimulator::Type::Int(64, false),
                                                             SPIRVSimulator::Type::Float(32),
                                                             SPIRVSimulator::Type::Vector(Type::i64, 2),
                                                             SPIRVSimulator::Type::Vector(Type::u64, 2),
                                                             SPIRVSimulator::Type::Vector(Type::f64, 2),
                                                             SPIRVSimulator::Type::Matrix(Type::vec2, 2) };

SPIRVSimulator::Type get(uint32_t type_id)
{
    return types_[type_id];
}

class ArithmeticsMock : public SPIRVSimulatorMockBase
{
  public:
    MOCK_METHOD(void, SetValue, (uint32_t id, const ::SPIRVSimulator::Value& value), (override));
    MOCK_METHOD(::SPIRVSimulator::Value&, GetValue, (uint32_t id), (override));

    MOCK_METHOD(::SPIRVSimulator::Type, GetTypeByTypeId, (uint32_t id), (const override));
    MOCK_METHOD(::SPIRVSimulator::Type, GetTypeByResultId, (uint32_t id), (const override));
};

struct ArithmeticParams
{
    spv::Op               opcode;
    SPIRVSimulator::Value op1;
    Type                  op1_type;
    SPIRVSimulator::Value op2;
    Type                  op2_type;
    SPIRVSimulator::Value expected;
    Type                  expected_type;

    friend std::ostream& operator<<(std::ostream& os, ArithmeticParams const& p)
    {
        // In case of unary operation, op2 is empty and obsolete
        if (std::holds_alternative<std::monostate>(p.op2))
        {
            os << opcode_to_char(p.opcode) << p.op1 << " = " << p.expected;
            return os;
        }
        else
        {
            os << p.op1 << ' ' << opcode_to_char(p.opcode) << ' ' << p.op2 << " = " << p.expected;
        }
        return os;
    }
};

struct ArithmeticParamsBuilder
{
    ArithmeticParams build()
    {
        ArithmeticParams temp = params;
        params                = {};
        return temp;
    }
    ArithmeticParams params;

    ArithmeticParamsBuilder& set_opcode(spv::Op opcode)
    {
        params.opcode = opcode;
        return *this;
    }
    ArithmeticParamsBuilder& set_op1(const SPIRVSimulator::Value& v, Type t)
    {
        params.op1      = v;
        params.op1_type = t;
        return *this;
    }
    ArithmeticParamsBuilder& set_op2(const SPIRVSimulator::Value& v, Type t)
    {
        params.op2      = v;
        params.op2_type = t;
        return *this;
    }
    ArithmeticParamsBuilder& set_expected(const SPIRVSimulator::Value& v, Type t)
    {
        params.expected      = v;
        params.expected_type = t;
        return *this;
    }
};

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

TEST_P(ArithmeticsTest, ArithmeticOperations)
{
    const auto& parameters = GetParam();

    const uint32_t lhs_id = NextId();
    EXPECT_CALL(mock, GetValue(lhs_id)).WillRepeatedly(ReturnRefOfCopy(parameters.op1));
    EXPECT_CALL(mock, GetTypeByResultId(lhs_id)).WillRepeatedly(Return(types_[parameters.op1_type]));

    const uint32_t rhs_id = NextId();
    EXPECT_CALL(mock, GetValue(rhs_id)).WillRepeatedly(ReturnRefOfCopy(parameters.op2));
    EXPECT_CALL(mock, GetTypeByResultId(rhs_id)).WillRepeatedly(Return(types_[parameters.op2_type]));

    SPIRVSimulator::Value captured_value;
    EXPECT_CALL(mock, SetValue(_, _)).WillOnce(SaveArg<1>(&captured_value));

    const uint32_t              result_id = NextId();
    std::vector<uint32_t>       words{ parameters.opcode, parameters.expected_type, result_id, lhs_id, rhs_id };
    SPIRVSimulator::Instruction inst{ .opcode     = parameters.opcode,
                                      .word_count = static_cast<uint16_t>(words.size()),
                                      .words      = words };
    mock.ExecuteInstruction(inst);

    EXPECT_EQ(captured_value, parameters.expected);
}

std::vector<ArithmeticParams> test_cases = {
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpSNegate)
        .set_op1(int64_t(1), Type::i64)
        .set_op2(SPIRVSimulator::Value(), Type::i64)
        .set_expected(-1, Type::i64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFNegate)
        .set_op1(1.0, Type::f64)
        .set_op2(SPIRVSimulator::Value(), Type::f64)
        .set_expected(-1.0, Type::f64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_op1(1, Type::i64)
        .set_op2(2, Type::i64)
        .set_expected(3, Type::i64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_op1(1.0, Type::f64)
        .set_op2(2.0, Type::f64)
        .set_expected(3.0, Type::f64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_op1(uint64_t(1), Type::u64)
        .set_op2(2, Type::i64)
        .set_expected(std::numeric_limits<uint64_t>::max(), Type::u64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_op1(1, Type::i64)
        .set_op2(2, Type::i64)
        .set_expected(-1, Type::i64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_op1(1.0, Type::f64)
        .set_op2(2.0, Type::f64)
        .set_expected(-1.0, Type::f64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_op1(2, Type::i64)
        .set_op2(2, Type::i64)
        .set_expected(4, Type::i64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_op1(2.0, Type::f64)
        .set_op2(2.0, Type::f64)
        .set_expected(4.0, Type::f64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpSDiv)
        .set_op1(-5, Type::i64)
        .set_op2(2, Type::i64)
        .set_expected(-2, Type::i64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpUDiv)
        .set_op1(uint64_t(5), Type::u64)
        .set_op2(uint64_t(2), Type::u64)
        .set_expected(uint64_t(2), Type::u64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFDiv)
        .set_op1(1.0, Type::f64)
        .set_op2(3.0, Type::f64)
        .set_expected(1.0 / 3.0, Type::f64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpUMod)
        .set_op1(uint64_t(13), Type::u64)
        .set_op2(uint64_t(6), Type::u64)
        .set_expected(uint64_t(1), Type::u64)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpIAdd)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 4, 4 }), Type::ivec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpISub)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1, 1 }), Type::ivec2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1, -1 }), Type::ivec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFAdd)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 4.0, 4.0 }), Type::vec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFSub)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 1.0 }), Type::vec2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1.0, -1.0 }), Type::vec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpIMul)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1, -1 }), Type::ivec2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2, 2 }), Type::ivec2)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -2, -2 }), Type::ivec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpFMul)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1.0, -1.0 }), Type::vec2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -2.0, -2.0 }), Type::vec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpVectorTimesScalar)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -1.0, -1.0 }), Type::vec2)
        .set_op2(2.0, Type::f64)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ -2.0, -2.0 }), Type::vec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpMatrixTimesVector)
        .set_op1(std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                     std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }),
                     std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 3.0, 4.0 }) }),
                 Type::mat2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }), Type::vec2)
        .set_expected(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 7.0, 10.0 }), Type::vec2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpMatrixTimesMatrix)
        .set_op1(std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                     std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }),
                     std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 3.0, 4.0 }) }),
                 Type::mat2)
        .set_op2(std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                     std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 1.0, 2.0 }),
                     std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 3.0, 4.0 }) }),
                 Type::mat2)
        .set_expected(std::make_shared<SPIRVSimulator::MatrixV>(std::initializer_list<SPIRVSimulator::Value>{
                          std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 7.0, 10.0 }),
                          std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 15.0, 22.0 }) }),
                      Type::mat2)
        .build(),
    ArithmeticParamsBuilder()
        .set_opcode(spv::Op::OpDot)
        .set_op1(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_op2(std::make_shared<SPIRVSimulator::VectorV>(std::initializer_list{ 2.0, 2.0 }), Type::vec2)
        .set_expected(8.0, Type::f64)
        .build()
};

INSTANTIATE_TEST_SUITE_P(Arithmetics, ArithmeticsTest, ValuesIn(test_cases));
