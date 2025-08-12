#pragma once

#ifndef ARM_TESTING_COMMON_HPP
#define ARM_TESTING_COMMON_HPP

#include <ostream>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "spirv.hpp"
#include <spirv_simulator.hpp>

using namespace testing;

std::string opcode_to_string(spv::Op opcode);

std::ostream& operator<<(std::ostream& os, SPIRVSimulator::Value const& v);

enum CommonTypes : uint32_t
{
    literal = 0,
    void_t,
    boolean,
    i32,
    i64,
    u32,
    u64,
    f32,
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
    spv::Op                                                                         opcode;
    std::unordered_map<uint32_t, SPIRVSimulator::Value>                             operands;
    std::unordered_map<uint32_t, std::variant<CommonTypes, ::SPIRVSimulator::Type>> operand_types;
    std::vector<std::pair<uint32_t, std::vector<SPIRVSimulator::DecorationInfo>>>   decorations;
    std::string                                                                     death_message;

    friend std::ostream& operator<<(std::ostream& os, TestParameters const& p)
    {
        os << opcode_to_string(p.opcode) << " ";
        for (uint32_t i = 0; i < p.operands.size() - 1; ++i)
        {
            if (!std::holds_alternative<std::monostate>(p.operands.at(i)))
            {
                os << p.operands.at(i) << " ";
            }
        }
        os << p.operands.at(p.operands.size() - 1);
        return os;
    }
};

template <typename T>
concept TypeT = std::is_same_v<T, CommonTypes> || std::is_same_v<T, ::SPIRVSimulator::Type>;

struct TestParametersBuilder
{
    TestParameters build() { return std::exchange(params, {}); }
    TestParameters params;

    TestParametersBuilder& set_opcode(spv::Op opcode)
    {
        params.opcode = opcode;
        return *this;
    }

    template <TypeT T>
    TestParametersBuilder& set_operand_at(uint32_t n, const SPIRVSimulator::Value& value, T type)
    {
        params.operands[n]      = value;
        params.operand_types[n] = type;
        return *this;
    }

    template <TypeT T>
    TestParametersBuilder&
    set_operands_range(uint32_t index_of_first, T type, std::initializer_list<SPIRVSimulator::Value> vals)
    {
        for (size_t i = 0; i < vals.size(); ++i)
        {
            params.operands[index_of_first + i]      = *(vals.begin() + i);
            params.operand_types[index_of_first + i] = type;
        }
        return *this;
    }

    template <TypeT T>
    TestParametersBuilder&
    set_operands_at(std::initializer_list<uint32_t> indices, T type, std::initializer_list<SPIRVSimulator::Value> vals)
    {
        assert(indices.size() == vals.size() && "indices and vals must be same length");
        for (size_t i = 0; i < indices.size(); ++i)
        {
            params.operands[*(indices.begin() + i)]      = *(vals.begin() + i);
            params.operand_types[*(indices.begin() + i)] = type;
        }
        return *this;
    }

    TestParametersBuilder& set_death_message(const std::string& message)
    {
        params.death_message = message;
        return *this;
    }

    TestParametersBuilder& set_decorations(uint32_t id, const std::vector<SPIRVSimulator::DecorationInfo>& decorations)
    {
        params.decorations.emplace_back(id, decorations);
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
            { CommonTypes::literal, ::SPIRVSimulator::Type() },
            { CommonTypes::void_t, ::SPIRVSimulator::Type() },
            { CommonTypes::boolean, ::SPIRVSimulator::Type::BoolT() },
            { CommonTypes::i32, ::SPIRVSimulator::Type::Int(32, true) },
            { CommonTypes::i64, ::SPIRVSimulator::Type::Int(64, true) },
            { CommonTypes::u32, ::SPIRVSimulator::Type::Int(32, false) },
            { CommonTypes::u64, ::SPIRVSimulator::Type::Int(64, false) },
            { CommonTypes::f64, ::SPIRVSimulator::Type::Float(32) },
            { CommonTypes::f64, ::SPIRVSimulator::Type::Float(64) },
            { CommonTypes::bvec2, ::SPIRVSimulator::Type::Vector(CommonTypes::boolean, 2) },
            { CommonTypes::ivec2, ::SPIRVSimulator::Type::Vector(CommonTypes::i64, 2) },
            { CommonTypes::uvec2, ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 2) },
            { CommonTypes::vec2, ::SPIRVSimulator::Type::Vector(CommonTypes::f64, 2) },
            { CommonTypes::bvec3, ::SPIRVSimulator::Type::Vector(CommonTypes::boolean, 3) },
            { CommonTypes::ivec3, ::SPIRVSimulator::Type::Vector(CommonTypes::i64, 3) },
            { CommonTypes::uvec3, ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 3) },
            { CommonTypes::vec3, ::SPIRVSimulator::Type::Vector(CommonTypes::f64, 3) },
            { CommonTypes::ivec4, ::SPIRVSimulator::Type::Vector(CommonTypes::i64, 4) },
            { CommonTypes::uvec4, ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 4) },
            { CommonTypes::vec4, ::SPIRVSimulator::Type::Vector(CommonTypes::f64, 4) },
            { CommonTypes::mat2, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec2, 2) },
            { CommonTypes::mat2x3, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec2, 3) },
            { CommonTypes::mat2x4, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec2, 4) },
            { CommonTypes::mat3, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec3, 3) },
            { CommonTypes::mat3x2, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec3, 2) },
            { CommonTypes::mat3x4, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec3, 4) },
            { CommonTypes::mat4, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec4, 4) },
            { CommonTypes::mat4x2, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec4, 2) },
            { CommonTypes::mat4x3, ::SPIRVSimulator::Type::Matrix(CommonTypes::vec4, 3) },
        };

        id_counter = types_.size();

        for (const auto& [id, type] : types_)
        {
            EXPECT_CALL(*this, GetTypeByTypeId(id)).WillRepeatedly(ReturnRef(type));
        }
    }
    virtual ~SPIRVSimulatorMockBase() = default;

    using SPIRVSimulator::SPIRVSimulator::ExecuteInstruction;

  protected:
    uint32_t              NextId() { return id_counter++; }
    std::vector<uint32_t> prepare_submission(const TestParameters& parameters);

  protected:
    uint32_t id_counter;
};

#endif