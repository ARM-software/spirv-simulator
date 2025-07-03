cmake_minimum_required(VERSION 3.9)

# Note, this MUST NOT be compile with -ffinite-math or -ffast-math (or any optimization option that affects nan of inf representations), this will break nan operations that we need for correct SPIRV simulation.
# In general, leave the -fno-fast-math option below alone so that we keep full IEEE 754 compliance.

enable_testing()

project(spirv_simulator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(framework)
add_subdirectory(test)

add_executable(spirv_simulator ${CMAKE_SOURCE_DIR}/main.cpp)
target_compile_options(spirv_simulator PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20 -fno-fast-math)
target_link_libraries(spirv_simulator PRIVATE spirv_simulator_lib)

add_executable(spirv_opcode_support_checker ${CMAKE_SOURCE_DIR}/main_opcode_checker.cpp)
target_compile_options(spirv_opcode_support_checker PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20 -fno-fast-math)
target_link_libraries(spirv_opcode_support_checker PRIVATE spirv_simulator_lib)