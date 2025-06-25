#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_ALL_REPORTERS

#include <memory>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>
#include "../spirv_simulator.hpp"
#include "../util.hpp"

const char* input_test_file = "../test_shaders/vulkan_type_creation.spirv";

struct SimulatorSetup
{
	SimulatorSetup(){
		sim = std::make_unique<SPIRVSimulator::SPIRVSimulator>(util::ReadFile(input_test_file), inputs, false);
		sim->Run();
	}

	SPIRVSimulator::InputData inputs{};
	std::unique_ptr<SPIRVSimulator::SPIRVSimulator> sim;
};

static SimulatorSetup setup;

TEST_CASE("1: Declaring an main function that returns \"void\" should create a \"void\" type","[single-file]")
{
	const auto & types = setup.sim->GetTypes();
	auto result = std::find_if(types.begin(),types.end(),[](const std::pair<uint32_t, SPIRVSimulator::Type>& entry){ return entry.second.kind == SPIRVSimulator::Type::Kind::Void;});
	REQUIRE(result != types.end());
	REQUIRE(result->second.scalar.is_signed == false);
	REQUIRE(result->second.scalar.width == 0);
}

TEST_CASE("2: Declaring an \"int\" variable should create an \"int\" type","[single-file]")
{
	const auto & types = setup.sim->GetTypes();
	auto result = std::find_if(types.begin(),types.end(),[](const std::pair<uint32_t, SPIRVSimulator::Type>& entry){ return entry.second.kind == SPIRVSimulator::Type::Kind::Int;});
	REQUIRE(result != types.end());
}