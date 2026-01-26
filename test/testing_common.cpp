#include "testing_common.hpp"
#include "spirv.hpp"
#include "spirv_simulator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

std::ostream& operator<<(std::ostream& os, const SPIRVSimulator::Value& value)
{
    if (const int64_t* inner_int = std::get_if<int64_t>(&value))
    {
        os << *inner_int;
    }
    else if (const uint64_t* inner_uint = std::get_if<uint64_t>(&value))
    {
        os << *inner_uint;
    }
    else if (const double* inner_double = std::get_if<double>(&value))
    {
        os << *inner_double;
    }
    else if (const std::shared_ptr<SPIRVSimulator::VectorV>* inner_vec =
                 std::get_if<std::shared_ptr<SPIRVSimulator::VectorV>>(&value))
    {
        const std::shared_ptr<SPIRVSimulator::VectorV>& inner = *inner_vec;
        const std::vector<SPIRVSimulator::Value>&       elems = inner->elems;
        os << "(";
        for (uint32_t i = 0; i < elems.size() - 1; ++i)
        {
            os << elems[i] << ",";
        }
        os << elems.back() << ")";
    }
    else if (const std::shared_ptr<SPIRVSimulator::MatrixV>* inner_mat =
                 std::get_if<std::shared_ptr<SPIRVSimulator::MatrixV>>(&value))
    {
        const std::shared_ptr<SPIRVSimulator::MatrixV>& inner   = *inner_mat;
        const std::vector<SPIRVSimulator::Value>&       columns = inner->cols;
        const size_t rows = std::get<std::shared_ptr<SPIRVSimulator::VectorV>>(columns[0])->elems.size();
        os << '(';
        for (size_t i = 0; i < rows; ++i)
        {
            os << '(';
            for (size_t j = 0; j < columns.size() - 1; ++j)
            {
                os << std::get<std::shared_ptr<SPIRVSimulator::VectorV>>(columns[j])->elems[i] << ',';
            }
            os << std::get<std::shared_ptr<SPIRVSimulator::VectorV>>(columns[columns.size() - 1])->elems[i];
            os << ")";
        }
        os << ')';
    }
    // TODO for aggregate and pointer
    else
        os << "<invalid>";
    return os;
}

// Ideally, this should be generated from SPIRV spec
std::string opcode_to_string(spv::Op opcode)
{
    switch (opcode)
    {
        case spv::OpIAdd:
        case spv::OpFAdd:
        {
            return "+";
        }
        case spv::OpISub:
        case spv::OpFSub:
        case spv::OpSNegate:
        {
            return "-";
        }
        case spv::OpIMul:
        case spv::OpFMul:
        case spv::OpVectorTimesScalar:
        case spv::OpDot:
        {
            return "*";
        }
        case spv::OpMatrixTimesVector:
        case spv::OpMatrixTimesMatrix:
        {
            return "x";
        }
        case spv::OpSDiv:
        case spv::OpUDiv:
        case spv::OpFDiv:
        {
            return "/";
        }
        case spv::OpVariable:
        {
            return "OpVariable";
        }
        case spv::OpCompositeConstruct:
        {
            return "OpCompositeConstruct";
        }
        default:
            return "";
    }
}

std::vector<uint32_t> SPIRVSimulatorMockBase::prepare_submission(const TestParameters& parameters)
{

    const size_t          result_id = NextId();
    std::vector<uint32_t> words{ static_cast<uint32_t>(parameters.opcode) };
    uint32_t              type_id = 0;
    for (uint32_t op = 0; op < parameters.operands.size(); ++op)
    {
        const auto& type_variant = parameters.operand_types.at(op);
        // Special handling for common types (raw words mostly)
        if (type_variant.index() == 0)
        {
            type_id = std::get<0>(type_variant);

            // Literals are special as they don't have id's
            const auto& value_variant = parameters.operands.at(op);

            if (type_id == CommonTypes::literal)
            {
                // Infer the characteristics from the result type?
                CommonTypes result_type = std::get<CommonTypes>(parameters.operand_types.at(0));
                uint32_t    width       = types_[result_type].scalar.width;
                bool        is_signed   = types_[result_type].scalar.is_signed;
                switch (parameters.operands.at(op).index())
                {
                    // uint64_t
                    case 1:
                    {
                        uint64_t value = std::get<1>(value_variant);
                        if (width == 64)
                        {
                            words.push_back(static_cast<uint32_t>(value));
                            words.push_back(static_cast<uint32_t>(value >> 32));
                        }
                        else
                        {
                            words.push_back(static_cast<uint32_t>(value));
                        }
                        break;
                    }
                    // int64_t
                    case 2:
                    {
                        int64_t value = std::get<2>(value_variant);

                        if (width == 64)
                        {
                            words.push_back(static_cast<uint32_t>(value));
                            words.push_back(static_cast<uint32_t>(value >> 32));
                        }
                        else
                        {
                            words.push_back(static_cast<uint32_t>(value));
                        }
                        break;
                    }
                    // double
                    case 3:
                    {
                        double value = std::get<3>(value_variant);
                        if (width == 64)
                        {
                            float halves[2];
                            std::memcpy(halves, &value, sizeof(value));
                            words.push_back(::SPIRVSimulator::bit_cast<uint32_t>(halves[0]));
                            words.push_back(::SPIRVSimulator::bit_cast<uint32_t>(halves[1]));
                        }
                        else
                        {
                            words.push_back(static_cast<uint32_t>(value));
                        }
                        break;
                    }
                    default:
                        std::cerr << "Cannot deduce literal type , must be a Scalar\n";
                }
                continue;
            }
            else if (type_id == CommonTypes::storage_class)
            {
                words.push_back(static_cast<uint32_t>(std::get<1>(value_variant)));
                continue;
            }
        }
        // Custom type, need to register or fetch already registered for this operand
        else
        {
            const ::SPIRVSimulator::Type& type = std::get<1>(type_variant);
            if (type.kind == ::SPIRVSimulator::Type::Kind::Struct)
            {
                // Try to make the new type, and return iterator to existing if possible
                auto [it, inserted] = types_.try_emplace(CommonTypes::num_types + type.structure.id,
                                                         ::SPIRVSimulator::Type::Struct(type.structure.id));
                type_id             = it->first;
            }
            else if (type.kind == ::SPIRVSimulator::Type::Kind::Array ||
                     type.kind == ::SPIRVSimulator::Type::Kind::RuntimeArray)
            {
                uint32_t key = type.array.elem_type_id;
                key ^= type.array.length_id + 0x9e3779b9u + (key << 6) + (key >> 2);
                key += CommonTypes::num_types;

                auto [it, inserted] = types_.try_emplace(key, type);
                type_id             = it->first;
            }
            else if (type.kind == ::SPIRVSimulator::Type::Kind::Pointer)
            {
                uint32_t key = type.pointer.storage_class;
                key ^= type.pointer.pointee_type_id + 0x9e3779b9u + (key << 6) + (key >> 2);
                key += CommonTypes::num_types;

                auto [it, inserted] = types_.try_emplace(key, type);
                type_id             = it->first;

                if (auto decorations_it = parameters.decorations.find(op);
                    decorations_it != parameters.decorations.end())
                {
                    decorators_[type.pointer.pointee_type_id] = decorations_it->second;
                }
            }

            EXPECT_CALL(*this, GetTypeByTypeId(type_id)).WillRepeatedly(ReturnRef(types_[type_id]));
        }
        const size_t op_id = NextId();
        if (op == 0)
        {
            words.push_back(type_id);
        }

        uint64_t dummy_flags = 0;
        words.push_back(static_cast<uint32_t>(op_id));
        EXPECT_CALL(*this, GetTypeByResultId(op_id)).WillRepeatedly(ReturnRef(types_[type_id]));
        EXPECT_CALL(*this, GetValue(op_id)).WillRepeatedly(ReturnRefOfCopy(parameters.operands.at(op)));
        EXPECT_CALL(*this, TransferFlags(::testing::A<uint32_t>(), ::testing::A<uint32_t>()))
            .Times(::testing::AnyNumber());
        EXPECT_CALL(*this, TransferFlags(::testing::A<uint32_t>(), ::testing::A<uint64_t>()))
            .Times(::testing::AnyNumber());
        EXPECT_CALL(*this, HasFlags(_, _)).WillRepeatedly(::testing::Return(false));
        EXPECT_CALL(*this, ExtractFlags(_, _)).Times(::testing::AnyNumber());
        EXPECT_CALL(*this, SetFlags(_, _)).Times(::testing::AnyNumber());
        EXPECT_CALL(*this, SetFlagsPointee(::testing::A<uint32_t>(), ::testing::A<uint64_t>()))
            .Times(::testing::AnyNumber());
        EXPECT_CALL(*this, SetFlagsPointee(::testing::A<::SPIRVSimulator::PointerV&>(), ::testing::A<uint64_t>()))
            .Times(::testing::AnyNumber());
    }

    return words;
}

::SPIRVSimulator::SimulationData SPIRVSimulatorMockBase::prepare_input_data(const TestParameters& parameters)
{
    ::SPIRVSimulator::SimulationData inputs;

    if (!parameters.push_constants_.empty())
    {
        inputs.push_constants = reinterpret_cast<const void*>(parameters.push_constants_.data());
    }

    return inputs;
}
