#pragma once

#ifndef ARM_SPIRV_SIMULATOR_EXCEPTIONS_HPP
#define ARM_SPIRV_SIMULATOR_EXCEPTIONS_HPP


#include <exception>
#include <string>

namespace SPIRVSimulator
{

class InvalidSpirvInputError : public std::exception {
 public:
    InvalidSpirvInputError(const std::string& message) : msg_(message) {}

    const char* what() const noexcept override {
        return msg_.c_str();
    }

 private:
    std::string msg_;
};

} // namespace SPIRVSimulator

#endif