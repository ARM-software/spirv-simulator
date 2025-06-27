#include "testing_common.hpp"

#include <memory>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <filesystem>

std::filesystem::path input_test_file_path = std::filesystem::path(TEST_SHADER_DIR) / "vulkan_type_creation.spirv";
static SimulatorSetup setup(input_test_file_path.c_str());

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