#include "spirv.hpp"
#include "spirv_simulator.hpp"
#include "testing_common.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

TEST(ArithmeticTest, IntegerAddition) {
  std::unordered_map<uint32_t, SPIRVSimulator::Value> values{
      {2, 1},
      {3, 2},
      {std::numeric_limits<uint32_t>::max(), SPIRVSimulator::Value()}};

  SPIRVSimulatorMockedFunctions mocked_functions;

  mocked_functions.GetValueMock =
      [&values](uint32_t id) -> SPIRVSimulator::Value & {
    auto result = values.find(id);
    if (result != values.end()) {
      return result->second;
    }
    return values[std::numeric_limits<uint32_t>::max()];
  };

  mocked_functions.GetTypeMock = [](uint32_t id) -> SPIRVSimulator::Type {
    SPIRVSimulator::Type type{};
    if (id == 0) {
      type.kind = SPIRVSimulator::Type::Kind::Int;
      type.scalar = {32, true};
      return type;
    };
    return type;
  };

  mocked_functions.SetValueMock = [](uint32_t id,
                                     const SPIRVSimulator::Value &value) {
    EXPECT_EQ(id , 1);
    EXPECT_EQ(std::get<int64_t>(value) , static_cast<int64_t>(3));
  };

  // This means add the values referenced by id's 2 and 3 into and store them in value referenced by id 1 of type id 0
  std::vector<uint32_t> words{spv::Op::OpIAdd, 0, 1, 2, 3};
  SPIRVSimulator::Instruction instruction{
      .opcode = spv::Op::OpIAdd, .word_count = 5, .words = words};

  SPIRVSimulatorMock mock(mocked_functions);
  mock.ExecuteSingleInstruction(instruction);
}
