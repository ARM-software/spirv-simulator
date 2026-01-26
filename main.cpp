#include <iostream>

#include <spirv_simulator.hpp>
#include <util.hpp>

static void usage()
{
    printf("Usage: spirv_simulator [options] <filename>\n");
    printf("Version %d.%d.%d. Command line options\n", PROJECT_VERSION_MAJOR, PROJECT_VERSION_MINOR, PROJECT_VERSION_PATCH);
    printf("-h/--help              This help\n");
    printf("-v/--verbose           Output a lot of extra information\n");
    exit(-1);
}

static std::string get_str(const char* in, int& remaining)
{
    assert(in != nullptr);
    if (remaining == 0)
    {
        usage();
    }
    remaining--;
    return in;
}

static bool match(const char* in, const char* short_form, const char* long_form, int& remaining)
{
    if ((short_form && strcmp(in, short_form) == 0) || (long_form && strcmp(in, long_form) == 0))
    {
        remaining--;
        return true;
    }
    return false;
}

int main(int argc, char** argv)
{
    int remaining = argc - 1; // zeroth is name of program
    std::string filename;
    bool verbose = false;
    for (int i = 1; i < argc; i++)
    {
        if (match(argv[i], "-h", "--help", remaining))
        {
            usage();
        }
        else if (match(argv[i], "-v", "--verbose", remaining))
        {
            verbose = true;
        }
        else
        {
            filename = get_str(argv[i], remaining);
            if (remaining > 0) { printf("Options after filename is not valid!\n\n"); usage(); }
        }
    }

    if (filename.empty()) usage();

    SPIRVSimulator::SimulationData sim_data;
    SPIRVSimulator::SimulationResults results;
    SPIRVSimulator::SPIRVSimulator sim(util::ReadFile(filename.c_str()), &sim_data, &results, nullptr, verbose);
    sim.Run();

    auto physical_address_data = results.physical_address_data;

    if (physical_address_data.size() > 0) std::cout << "Pointers to pbuffers:" << std::endl;
    for (const auto& pointer_t : physical_address_data)
    {
        std::cout << "  Found pointer with address: 0x" << std::hex << pointer_t.raw_pointer_value << std::dec
                  << " made from input bit components:" << std::endl;
        for (auto bit_component : pointer_t.bit_components)
        {
            if (bit_component.location == SPIRVSimulator::BitLocation::Constant)
            {
                std::cout << "    "
                          << "From Constant in SPIRV input words, at Byte Offset: " << bit_component.byte_offset
                          << std::endl;
            }
            else
            {
                if (bit_component.location == SPIRVSimulator::BitLocation::SpecConstant)
                {
                    std::cout << "    " << "From SpecId: " << bit_component.binding_id;
                }
                else
                {
                    std::cout << "    " << "From DescriptorSetID: " << bit_component.set_id
                              << ", Binding: " << bit_component.binding_id;
                }

                if (bit_component.location == SPIRVSimulator::BitLocation::StorageClass)
                {
                    std::cout << ", in StorageClass: " << spv::StorageClassToString(bit_component.storage_class);
                }
                std::cout << ", Byte Offset: " << bit_component.byte_offset << ", Bitsize: " << bit_component.bitcount
                          << ", to val Bit Offset: " << bit_component.val_bit_offset << std::endl;
            }
        }
    }

    return 0;
}
