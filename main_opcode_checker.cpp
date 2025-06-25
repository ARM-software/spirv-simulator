#include "spirv_simulator.hpp"
#include "util.hpp"


int main(int argc, char** argv){
    if(argc!=2){
        std::cerr << "Usage: " << argv[0] << " <list_of_shader_paths>.txt\n";
        return 1;
    }

    const std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }

    std::string line;
    std::set<std::string> unsupported_instructions;
    while (std::getline(file, line)) {
        SPIRVSimulator::InputData inputs;
        SPIRVSimulator::SPIRVSimulator sim(util::ReadFile(line), inputs);
        unsupported_instructions.insert(sim.unsupported_opcodes.begin(), sim.unsupported_opcodes.end());
    }

    for (const auto& opc : unsupported_instructions){
        std::cout << opc << std::endl;
    }

    file.close();

    return 0;
}
