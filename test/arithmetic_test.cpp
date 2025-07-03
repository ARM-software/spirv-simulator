#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

TEST(ArithmeticTest, IntegerAddition) {
  class Mock : public SPIRVSimulatorMockBase{
      public:
        MOCK_METHOD(void, SetValue,
                    (uint32_t id, const ::  SPIRVSimulator::Value &value),
                    (override));
        MOCK_METHOD(::SPIRVSimulator::Value &, GetValue, (uint32_t id),
                    (override));

        MOCK_METHOD(::SPIRVSimulator::Type, GetType, (uint32_t id),
                    (const override));
  };

  SPIRVSimulator::Type type;
  type.kind = SPIRVSimulator::Type::Kind::Int;
  type.scalar = {32, true};

  Mock mock;
  EXPECT_CALL(mock, GetType(0)).WillRepeatedly(::Return(type));

  ::SPIRVSimulator::Value lhs(1);
  ::SPIRVSimulator::Value rhs(2);
  EXPECT_CALL(mock, GetValue(2)).WillRepeatedly(::ReturnRef(lhs));
  EXPECT_CALL(mock, GetValue(3)).WillRepeatedly(::ReturnRef(rhs));

  uint32_t captured_id;
  ::SPIRVSimulator::Value captured_value;
  EXPECT_CALL(mock,SetValue(::_,::_)).WillOnce(::DoAll(::SaveArg<0>(&captured_id), ::SaveArg<1>(&captured_value)));

  std::vector<uint32_t> words{spv::Op::OpIAdd, 0, 1, 2, 3};
  SPIRVSimulator::Instruction instruction{.opcode = spv::Op::OpIAdd, .word_count = 5, .words = words};

  mock.ExecuteSingleInstruction(instruction);

  EXPECT_EQ(captured_id, 1);
  EXPECT_EQ(std::get<int64_t>(captured_value), 3);
}
