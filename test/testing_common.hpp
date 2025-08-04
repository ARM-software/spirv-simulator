#pragma once

#include <cstdint>
#ifndef ARM_TESTING_COMMON_HPP
#define ARM_TESTING_COMMON_HPP

#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <spirv_simulator.hpp>

using namespace testing;

char opcode_to_char(spv::Op opcode);

std::ostream& operator<<(std::ostream& os, SPIRVSimulator::Value const& v);

enum Type : uint32_t
{
    void_t = 0,
    boolean,
    i64,
    u64,
    f64,

    bvec2,
    ivec2,
    uvec2,
    vec2,
    bvec3,
    ivec3,
    uvec3,
    vec3,
    ivec4,
    uvec4,
    vec4,

    mat2,
    mat2x3,
    mat2x4,
    mat3,
    mat3x2,
    mat3x4,
    mat4,
    mat4x2,
    mat4x3,

    num_types
};

struct TestParameters
{
    spv::Op                            opcode;
    std::vector<SPIRVSimulator::Value> operands;
    std::vector<Type>                  operand_types;
    std::string                        death_message;

    friend std::ostream& operator<<(std::ostream& os, TestParameters const& p)
    {
        os << opcode_to_char(p.opcode) << " ";
        for (uint32_t i = 0; i < p.operands.size() - 1; ++i)
        {
            if (!std::holds_alternative<std::monostate>(p.operands[i]))
            {
                os << p.operands[i] << " ";
            }
        }
        os << p.operands[p.operands.size() - 1];
        return os;
    }
};

struct TestParametersBuilder
{
    TestParameters build() { return params; }
    TestParameters params;

    TestParametersBuilder& set_opcode(spv::Op opcode)
    {
        params.opcode = opcode;
        return *this;
    }

    TestParametersBuilder& set_operands_size(uint32_t n)
    {
        params.operands.resize(n);
        params.operand_types.resize(n);
        return *this;
    }

    TestParametersBuilder& set_op_n(uint32_t n, const SPIRVSimulator::Value& v, Type t)
    {
        params.operands[n]      = v;
        params.operand_types[n] = t;
        return *this;
    }
    TestParametersBuilder& set_death_message(const std::string& message)
    {
        params.death_message = message;
        return *this;
    }
};

class SPIRVSimulatorMockBase : public SPIRVSimulator::SPIRVSimulator
{
  public:
    MOCK_METHOD(void, SetValue, (uint32_t id, const ::SPIRVSimulator::Value& value), (override));
    MOCK_METHOD(::SPIRVSimulator::Value&, GetValue, (uint32_t id), (override));

    MOCK_METHOD(const ::SPIRVSimulator::Type&, GetTypeByTypeId, (uint32_t id), (const override));
    MOCK_METHOD(const ::SPIRVSimulator::Type&, GetTypeByResultId, (uint32_t id), (const override));

    SPIRVSimulatorMockBase()
    {
        types_ = {
            { Type::void_t, ::SPIRVSimulator::Type() },
            { Type::boolean, ::SPIRVSimulator::Type::BoolT() },
            { Type::i64, ::SPIRVSimulator::Type::Int(64, true) },
            { Type::u64, ::SPIRVSimulator::Type::Int(64, false) },
            { Type::f64, ::SPIRVSimulator::Type::Float(64) },
            { Type::bvec2, ::SPIRVSimulator::Type::Vector(Type::boolean, 2) },
            { Type::ivec2, ::SPIRVSimulator::Type::Vector(Type::i64, 2) },
            { Type::uvec2, ::SPIRVSimulator::Type::Vector(Type::u64, 2) },
            { Type::vec2, ::SPIRVSimulator::Type::Vector(Type::f64, 2) },
            { Type::bvec3, ::SPIRVSimulator::Type::Vector(Type::boolean, 3) },
            { Type::ivec3, ::SPIRVSimulator::Type::Vector(Type::i64, 3) },
            { Type::uvec3, ::SPIRVSimulator::Type::Vector(Type::u64, 3) },
            { Type::vec3, ::SPIRVSimulator::Type::Vector(Type::f64, 3) },
            { Type::ivec4, ::SPIRVSimulator::Type::Vector(Type::i64, 4) },
            { Type::uvec4, ::SPIRVSimulator::Type::Vector(Type::u64, 4) },
            { Type::vec4, ::SPIRVSimulator::Type::Vector(Type::f64, 4) },
            { Type::mat2, ::SPIRVSimulator::Type::Matrix(Type::vec2, 2) },
            { Type::mat2x3, ::SPIRVSimulator::Type::Matrix(Type::vec2, 3) },
            { Type::mat2x4, ::SPIRVSimulator::Type::Matrix(Type::vec2, 4) },
            { Type::mat3, ::SPIRVSimulator::Type::Matrix(Type::vec3, 3) },
            { Type::mat3x2, ::SPIRVSimulator::Type::Matrix(Type::vec3, 2) },
            { Type::mat3x4, ::SPIRVSimulator::Type::Matrix(Type::vec3, 4) },
            { Type::mat4, ::SPIRVSimulator::Type::Matrix(Type::vec4, 4) },
            { Type::mat4x2, ::SPIRVSimulator::Type::Matrix(Type::vec4, 2) },
            { Type::mat4x3, ::SPIRVSimulator::Type::Matrix(Type::vec4, 3) },
        };
        for (const auto& [id, type] : types_)
        {
            EXPECT_CALL(*this, GetTypeByTypeId(id)).WillRepeatedly(ReturnRef(type));
        }
    }
    ~SPIRVSimulatorMockBase() = default;

    using SPIRVSimulator::SPIRVSimulator::ExecuteInstruction;

  protected:
    uint32_t NextId() { return id_counter++; }
    uint32_t id_counter = Type::num_types;
};

#endif