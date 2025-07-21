#include "testing_common.hpp"
#include "spirv_simulator.hpp"
#include <cstdint>
#include <memory>

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
        os << "(";
        for (uint32_t i = 0; i < inner->elems.size() - 1; ++i)
        {
            os << inner->elems[i] << ",";
        }
        os << inner->elems.back() << ")";
    }
    // TODO for matrix, aggregate and pointer
    else
        os << "<invalid>";
    return os;
}