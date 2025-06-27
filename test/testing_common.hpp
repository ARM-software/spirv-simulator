#pragma once

#ifndef ARM_TESTING_COMMON_HPP
#define ARM_TESTING_COMMON_HPP

#define CATCH_CONFIG_MAIN

#include <spirv_simulator.hpp>
#include <util.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string_view>

struct SimulatorSetup
{
	SimulatorSetup(std::string_view input_file){
		sim = std::make_unique<SPIRVSimulator::SPIRVSimulator>(util::ReadFile(input_file.data()), inputs, false);
		sim->Run();
	}

	SPIRVSimulator::InputData inputs{};
	std::unique_ptr<SPIRVSimulator::SPIRVSimulator> sim;
};

#endif