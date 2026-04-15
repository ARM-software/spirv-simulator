#include <iostream>
#include <fstream>

#include <spirv_simulator.hpp>
#include <util.hpp>

#include "framework/memory_flag_tracker.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <list_of_shader_paths>.txt\n";
        return 1;
    }

    const std::string filename = argv[1];
    std::ifstream     file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }

    std::string           line;
    std::set<std::string> unsupported_instructions;
    while (std::getline(file, line))
    {
        SPIRVSimulator::MemoryFlagTracker   mem_tracker;
        SPIRVSimulator::SimulationData      inputs;
        SPIRVSimulator::SimulationResults   outputs;
        SPIRVSimulator::SPIRVSimulator sim(util::ReadFile(line), &mem_tracker, &inputs, &outputs);
        unsupported_instructions.insert(sim.unsupported_opcodes.begin(), sim.unsupported_opcodes.end());
    }

    for (const auto& opc : unsupported_instructions)
    {
        std::cout << opc << std::endl;
    }

    file.close();

    return 0;
}
