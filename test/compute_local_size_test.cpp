#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "spirv-tools/libspirv.hpp"
#include "spirv_simulator.hpp"
#include "util.hpp"

namespace
{
struct LocalSizeModuleOptions
{
    std::string execution_model = "GLCompute";
    std::optional<std::array<uint32_t, 3>> local_size = std::array<uint32_t, 3>{ 1, 1, 1 };
    bool include_workgroup_size = true;
    uint32_t workgroup_component_count = 3;
    std::array<uint32_t, 3> workgroup_size = { 32, 32, 1 };
    bool include_second_workgroup_size = false;
};

struct LocalSizeIdModuleOptions
{
    std::array<uint32_t, 3> defaults = { 8, 4, 2 };
    bool second_component_is_bool = false;
    bool third_component_is_undef = false;
    std::optional<std::array<uint32_t, 3>> literal_local_size;
};

std::string BuildLocalSizeModule(const LocalSizeModuleOptions& options)
{
    static constexpr std::array<std::string_view, 3> component_names = { "x", "y", "z" };

    std::ostringstream module;
    module << "OpCapability Shader\n"
           << "OpMemoryModel Logical GLSL450\n"
           << "OpEntryPoint " << options.execution_model << " %main \"main\"\n";

    if (options.local_size.has_value())
    {
        module << "OpExecutionMode %main LocalSize " << (*options.local_size)[0] << " "
               << (*options.local_size)[1] << " " << (*options.local_size)[2] << "\n";
    }

    if (options.include_workgroup_size)
    {
        for (uint32_t component_index = 0; component_index < options.workgroup_component_count; ++component_index)
        {
            module << "OpDecorate %local_size_" << component_names[component_index] << " SpecId "
                   << component_index << "\n";
        }
        module << "OpDecorate %workgroup_size BuiltIn WorkgroupSize\n";
        if (options.include_second_workgroup_size)
        {
            module << "OpDecorate %second_workgroup_size BuiltIn WorkgroupSize\n";
        }
    }

    module << "%void = OpTypeVoid\n"
           << "%uint = OpTypeInt 32 0\n";

    if (options.include_workgroup_size)
    {
        module << "%workgroup_size_type = OpTypeVector %uint " << options.workgroup_component_count << "\n";
        for (uint32_t component_index = 0; component_index < options.workgroup_component_count; ++component_index)
        {
            module << "%local_size_" << component_names[component_index] << " = OpSpecConstant %uint "
                   << options.workgroup_size[component_index] << "\n";
        }

        module << "%workgroup_size = OpSpecConstantComposite %workgroup_size_type";
        for (uint32_t component_index = 0; component_index < options.workgroup_component_count; ++component_index)
        {
            module << " %local_size_" << component_names[component_index];
        }
        module << "\n";

        if (options.include_second_workgroup_size)
        {
            module << "%second_workgroup_size = OpSpecConstantComposite %workgroup_size_type";
            for (uint32_t component_index = 0; component_index < options.workgroup_component_count; ++component_index)
            {
                module << " %local_size_" << component_names[component_index];
            }
            module << "\n";
        }
    }

    module << "%function_type = OpTypeFunction %void\n"
           << "%main = OpFunction %void None %function_type\n"
           << "%entry = OpLabel\n"
           << "OpReturn\n"
           << "OpFunctionEnd\n";

    return module.str();
}

std::string BuildLocalSizeIdModule(const LocalSizeIdModuleOptions& options)
{
    std::ostringstream module;
    module << "OpCapability Shader\n"
           << "OpMemoryModel Logical GLSL450\n"
           << "OpEntryPoint GLCompute %main \"main\"\n";

    if (options.literal_local_size.has_value())
    {
        module << "OpExecutionMode %main LocalSize " << (*options.literal_local_size)[0] << " "
               << (*options.literal_local_size)[1] << " " << (*options.literal_local_size)[2] << "\n";
    }

    module << "OpExecutionModeId %main LocalSizeId %local_size_x %local_size_y %local_size_z\n"
           << "OpDecorate %local_size_x SpecId 10\n"
           << "OpDecorate %local_size_y SpecId 11\n";
    if (!options.third_component_is_undef)
    {
        module << "OpDecorate %local_size_z SpecId 12\n";
    }

    module << "%void = OpTypeVoid\n"
           << "%uint = OpTypeInt 32 0\n"
           << "%bool = OpTypeBool\n"
           << "%local_size_x = OpSpecConstant %uint " << options.defaults[0] << "\n";

    if (options.second_component_is_bool)
    {
        module << "%local_size_y = OpSpecConstantTrue %bool\n";
    }
    else
    {
        module << "%local_size_y = OpSpecConstant %uint " << options.defaults[1] << "\n";
    }

    if (options.third_component_is_undef)
    {
        module << "%local_size_z = OpUndef %uint\n";
    }
    else
    {
        module << "%local_size_z = OpSpecConstant %uint " << options.defaults[2] << "\n";
    }

    module << "%function_type = OpTypeFunction %void\n"
           << "%main = OpFunction %void None %function_type\n"
           << "%entry = OpLabel\n"
           << "OpReturn\n"
           << "OpFunctionEnd\n";

    return module.str();
}

std::string BuildMultiEntryPointLocalSizeModule(
    std::string_view                              second_execution_model,
    const std::optional<std::array<uint32_t, 3>>& second_local_size)
{
    std::ostringstream module;
    module << "OpCapability Shader\n"
           << "OpMemoryModel Logical GLSL450\n"
           << "OpEntryPoint GLCompute %first \"first\"\n"
           << "OpEntryPoint " << second_execution_model << " %second \"second\"\n"
           << "OpExecutionMode %first LocalSize 4 2 1\n";

    if (second_local_size.has_value())
    {
        module << "OpExecutionMode %second LocalSize " << (*second_local_size)[0] << " "
               << (*second_local_size)[1] << " " << (*second_local_size)[2] << "\n";
    }

    module << "%void = OpTypeVoid\n"
           << "%function_type = OpTypeFunction %void\n"
           << "%first = OpFunction %void None %function_type\n"
           << "%first_entry = OpLabel\n"
           << "OpReturn\n"
           << "OpFunctionEnd\n"
           << "%second = OpFunction %void None %function_type\n"
           << "%second_entry = OpLabel\n"
           << "OpReturn\n"
           << "OpFunctionEnd\n";

    return module.str();
}

std::vector<uint32_t> AssembleModule(const std::string& module_text,
                                     spv_target_env     target_env = SPV_ENV_UNIVERSAL_1_0)
{
    spvtools::SpirvTools tools(target_env);
    std::vector<uint32_t> program_words;
    if (!tools.Assemble(module_text, &program_words))
    {
        throw std::runtime_error("Failed to assemble local-size test module");
    }
    return program_words;
}

class ComputeLocalSizeTestSimulator : public ::SPIRVSimulator::SPIRVSimulator
{
  public:
    using SPIRVSimulator::SPIRVSimulator;

    ::SPIRVSimulator::ComputeLocalSize DeriveLocalSize(std::string_view entry_point_name)
    {
        for (const auto& [entry_point_id, name] : entry_points_)
        {
            if (name == entry_point_name)
            {
                DeriveActiveComputeLocalSize(entry_point_id);
                return active_compute_local_size_;
            }
        }
        throw std::runtime_error("Failed to find local-size test entry point");
    }

    void AddFlagsForDecoration(spv::Decoration decoration_kind, uint32_t literal, uint64_t flags)
    {
        for (const auto& [result_id, decoration_list] : decorators_)
        {
            for (const ::SPIRVSimulator::DecorationInfo& decoration : decoration_list)
            {
                if (decoration.kind == decoration_kind && !decoration.literals.empty() &&
                    decoration.literals[0] == literal)
                {
                    SetFlags(result_id, flags);
                    return;
                }
            }
        }

        throw std::runtime_error("Failed to find decorated local-size test value");
    }
};

void ExpectLocalSize(const ::SPIRVSimulator::ComputeLocalSize& local_size,
                     uint64_t                                  expected_x,
                     uint64_t                                  expected_y,
                     uint64_t                                  expected_z)
{
    EXPECT_TRUE(local_size.valid);
    EXPECT_EQ(local_size.x, expected_x);
    EXPECT_EQ(local_size.y, expected_y);
    EXPECT_EQ(local_size.z, expected_z);
}

class ComputeLocalSizeTest : public ::testing::Test
{
  protected:
    ::SPIRVSimulator::SimulationData simulation_data_;
    ::SPIRVSimulator::SimulationResults simulation_results_;
};

#ifdef NDEBUG
#define EXPECT_LOCAL_SIZE_FAILURE(statement) EXPECT_THROW(statement, std::runtime_error)
#else
#define EXPECT_LOCAL_SIZE_FAILURE(statement) EXPECT_DEATH(statement, ".*")
#endif

TEST_F(ComputeLocalSizeTest, WorkgroupSizeTakesPriorityOverPlaceholderLocalSize)
{
    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule({}));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 32, 32, 1);
}

TEST_F(ComputeLocalSizeTest, WorkgroupSizeUsesAllExternalSpecializationValues)
{
    struct LocalSizeSpecialization
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    };

    const LocalSizeSpecialization specialization = { 8, 4, 2 };
    simulation_data_.specialization_constants = &specialization;
    simulation_data_.specialization_constant_offsets = {
        { 0, offsetof(LocalSizeSpecialization, x) },
        { 1, offsetof(LocalSizeSpecialization, y) },
        { 2, offsetof(LocalSizeSpecialization, z) },
    };

    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule({}));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 8, 4, 2);
}

TEST_F(ComputeLocalSizeTest, WorkgroupSizeMixesExternalAndDefaultSpecializationValues)
{
    const uint32_t y_override = 7;
    simulation_data_.specialization_constants = &y_override;
    simulation_data_.specialization_constant_offsets = { { 1, 0 } };

    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule({}));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 32, 7, 1);
}

TEST_F(ComputeLocalSizeTest, InexactWorkgroupSizeFails)
{
    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule({}));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);
    simulator.AddFlagsForDecoration(spv::Decoration::DecorationBuiltIn,
                                    static_cast<uint32_t>(spv::BuiltIn::BuiltInWorkgroupSize),
                                    SPS_FLAG_IS_ARBITRARY);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, CallerOverrideTakesPriorityOverWorkgroupSize)
{
    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule({}));
    simulation_data_.compute_local_size = { 8, 4, 1, true };
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 8, 4, 1);
}

TEST_F(ComputeLocalSizeTest, FallsBackToLocalSizeWhenWorkgroupSizeIsAbsent)
{
    LocalSizeModuleOptions options;
    options.local_size = std::array<uint32_t, 3>{ 4, 2, 1 };
    options.include_workgroup_size = false;

    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule(options));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 4, 2, 1);
}

TEST_F(ComputeLocalSizeTest, LocalSizeExecutionModesAreIndexedPerEntryPoint)
{
    const std::vector<uint32_t> program_words = AssembleModule(BuildMultiEntryPointLocalSizeModule(
        "GLCompute", std::array<uint32_t, 3>{ 8, 4, 2 }));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("first"), 4, 2, 1);
    ExpectLocalSize(simulator.DeriveLocalSize("second"), 8, 4, 2);
}

TEST_F(ComputeLocalSizeTest, EntryPointWithoutLocalSizeDoesNotUseAnotherEntryPointsMode)
{
    const std::vector<uint32_t> program_words =
        AssembleModule(BuildMultiEntryPointLocalSizeModule("Vertex", std::nullopt));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    const ::SPIRVSimulator::ComputeLocalSize local_size = simulator.DeriveLocalSize("second");
    EXPECT_FALSE(local_size.valid);
}

TEST_F(ComputeLocalSizeTest, LocalSizeIdUsesDefaultSpecializationValues)
{
    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule({}), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 8, 4, 2);
}

TEST_F(ComputeLocalSizeTest, LocalSizeIdUsesAllExternalSpecializationValues)
{
    struct LocalSizeSpecialization
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    };

    const LocalSizeSpecialization specialization = { 16, 8, 4 };
    simulation_data_.specialization_constants = &specialization;
    simulation_data_.specialization_constant_offsets = {
        { 10, offsetof(LocalSizeSpecialization, x) },
        { 11, offsetof(LocalSizeSpecialization, y) },
        { 12, offsetof(LocalSizeSpecialization, z) },
    };

    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule({}), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 16, 8, 4);
}

TEST_F(ComputeLocalSizeTest, LocalSizeIdMixesExternalAndDefaultSpecializationValues)
{
    const uint32_t y_override = 7;
    simulation_data_.specialization_constants = &y_override;
    simulation_data_.specialization_constant_offsets = { { 11, 0 } };

    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule({}), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 8, 7, 2);
}

TEST_F(ComputeLocalSizeTest, InexactLocalSizeIdComponentFails)
{
    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule({}), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);
    simulator.AddFlagsForDecoration(spv::Decoration::DecorationSpecId, 10, SPS_FLAG_UNINITIALIZED);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, ZeroLocalSizeIdComponentFails)
{
    LocalSizeIdModuleOptions options;
    options.defaults = { 8, 0, 2 };

    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule(options), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, NonIntegerLocalSizeIdComponentFails)
{
    LocalSizeIdModuleOptions options;
    options.second_component_is_bool = true;

    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule(options), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, NonConstantLocalSizeIdComponentFails)
{
    LocalSizeIdModuleOptions options;
    options.third_component_is_undef = true;

    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule(options), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, ConflictingLocalSizeAndLocalSizeIdFail)
{
    LocalSizeIdModuleOptions options;
    options.literal_local_size = std::array<uint32_t, 3>{ 1, 1, 1 };

    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule(options), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, MatchingLocalSizeAndLocalSizeIdStillFail)
{
    LocalSizeIdModuleOptions options;
    options.literal_local_size = options.defaults;

    const std::vector<uint32_t> program_words =
        AssembleModule(BuildLocalSizeIdModule(options), SPV_ENV_UNIVERSAL_1_6);
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, SpecializedWorkgroupSizeFromRealShader)
{
    struct LocalSizeSpecialization
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    };

    const LocalSizeSpecialization specialization = { 32, 32, 1 };
    simulation_data_.specialization_constants = &specialization;
    simulation_data_.specialization_constant_offsets = {
        { 0, offsetof(LocalSizeSpecialization, x) },
        { 1, offsetof(LocalSizeSpecialization, y) },
        { 2, offsetof(LocalSizeSpecialization, z) },
    };
    const std::vector<uint32_t> program_words =
        util::ReadFile(std::string(TEST_SHADER_DIR) + "/vulkan_compute_bda_copying_address_interleave.spirv");
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    ExpectLocalSize(simulator.DeriveLocalSize("main"), 32, 32, 1);
}

TEST_F(ComputeLocalSizeTest, NonComputeEntryPointIgnoresWorkgroupSize)
{
    LocalSizeModuleOptions options;
    options.execution_model = "Vertex";
    options.local_size.reset();

    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule(options));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    const ::SPIRVSimulator::ComputeLocalSize local_size = simulator.DeriveLocalSize("main");
    EXPECT_FALSE(local_size.valid);
}

TEST_F(ComputeLocalSizeTest, InvalidWorkgroupSizeShapeDoesNotFallBackToLocalSize)
{
    LocalSizeModuleOptions options;
    options.local_size = std::array<uint32_t, 3>{ 4, 2, 1 };
    options.workgroup_component_count = 2;

    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule(options));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, ZeroWorkgroupSizeComponentDoesNotFallBackToLocalSize)
{
    LocalSizeModuleOptions options;
    options.local_size = std::array<uint32_t, 3>{ 4, 2, 1 };
    options.workgroup_size = { 32, 0, 1 };

    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule(options));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

TEST_F(ComputeLocalSizeTest, MultipleWorkgroupSizeObjectsFail)
{
    LocalSizeModuleOptions options;
    options.include_second_workgroup_size = true;

    const std::vector<uint32_t> program_words = AssembleModule(BuildLocalSizeModule(options));
    ComputeLocalSizeTestSimulator simulator(program_words, nullptr, &simulation_data_, &simulation_results_);

    EXPECT_LOCAL_SIZE_FAILURE(simulator.DeriveLocalSize("main"));
}

#undef EXPECT_LOCAL_SIZE_FAILURE
}
