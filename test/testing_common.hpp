#pragma once

#ifndef ARM_TESTING_COMMON_HPP
#define ARM_TESTING_COMMON_HPP

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <spirv_simulator.hpp>
#include <util.hpp>

struct SPIRVSimulatorMockedFunctions {
  std::function<void(uint32_t, const SPIRVSimulator::Value &)> SetValueMock;
  std::function<SPIRVSimulator::Value &(uint32_t)> GetValueMock;
  std::function<SPIRVSimulator::Type(uint32_t)> GetTypeMock;
};

class SPIRVSimulatorMock : public SPIRVSimulator::SPIRVSimulator {
public:
  SPIRVSimulatorMock(const SPIRVSimulatorMockedFunctions &mocked_functions)
      : mocked_functions_(mocked_functions) {
    RegisterOpcodeHandlers();
  }

  void ExecuteSingleInstruction(const ::SPIRVSimulator::Instruction &instruction) {
    ExecuteInstruction(instruction);
  }

protected:
  SPIRVSimulatorMockedFunctions mocked_functions_;

protected:
  void DecodeHeader() override;
  void SetValue(uint32_t result_id,
                const ::SPIRVSimulator::Value &value) override;
  ::SPIRVSimulator::Value &GetValue(uint32_t result_id) override;
  ::SPIRVSimulator::Type GetType(uint32_t result_id) const override;
};

#endif