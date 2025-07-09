#include "testing_common.hpp"
#include "spirv_simulator.hpp"
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
        const uint32_t rows = std::get<std::shared_ptr<SPIRVSimulator::VectorV>>(columns[0])->elems.size();
        os << '(';
        for (uint32_t i = 0; i < rows; ++i)
        {
            os << '(';
            for (uint32_t j = 0; j < columns.size() - 1; ++j)
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