#include "testing_common.hpp"
#include "spirv_simulator.hpp"

// Just skip decoding the header for the manually encoded test instruction
void SPIRVSimulatorMock::DecodeHeader() {
  stream_ = std::span<const uint32_t>(program_words_);
}

void SPIRVSimulatorMock::SetValue(uint32_t result_id,
                                  const ::SPIRVSimulator::Value &value) {
  if (mocked_functions_.SetValueMock) {
    mocked_functions_.SetValueMock(result_id, value);
  } else {
    SPIRVSimulator::SetValue(result_id, value);
  }
}

::SPIRVSimulator::Value &SPIRVSimulatorMock::GetValue(uint32_t result_id) {
  if (mocked_functions_.GetValueMock) {
    return mocked_functions_.GetValueMock(result_id);
  } else {
    return SPIRVSimulator::GetValue(result_id);
  }
}

::SPIRVSimulator::Type SPIRVSimulatorMock::GetType(uint32_t type_id) const {
  if (mocked_functions_.GetTypeMock) {
    return mocked_functions_.GetTypeMock(type_id);
  } else {
    return SPIRVSimulator::GetType(type_id);
  }
}
