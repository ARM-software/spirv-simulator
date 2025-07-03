#pragma once

#ifndef ARM_TESTING_COMMON_HPP
#define ARM_TESTING_COMMON_HPP

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <spirv_simulator.hpp>


class SPIRVSimulatorMockBase : public SPIRVSimulator::SPIRVSimulator {
public:
  SPIRVSimulatorMockBase()
  {
    RegisterOpcodeHandlers();
  }
  ~SPIRVSimulatorMockBase() = default;

  void ExecuteSingleInstruction(const ::SPIRVSimulator::Instruction &instruction)
  {
    ExecuteInstruction(instruction);
  }
};

#endif