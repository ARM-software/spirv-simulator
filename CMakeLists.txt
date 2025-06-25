cmake_minimum_required(VERSION 3.9)

# Note, this MUST NOT be compile with -ffinite-math or -ffast-math (or any optimization option that affects nan of inf representations), this will break nan operations that we need for correct SPIRV simulation.
# In general, leave the -fno-fast-math option below alone so that we keep full IEEE 754 compliance.

enable_testing()

project(spirv_simulator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_FILES ${CMAKE_SOURCE_DIR}/spirv_simulator.cpp ${CMAKE_SOURCE_DIR}/util.cpp)
set(HEADER_FILES ${CMAKE_SOURCE_DIR}/spirv_simulator.hpp ${CMAKE_SOURCE_DIR}/spirv.hpp ${CMAKE_SOURCE_DIR}/util.hpp)

add_executable(spirv_simulator ${SOURCE_FILES} ${HEADER_FILES} ${CMAKE_SOURCE_DIR}/main.cpp)
target_compile_options(spirv_simulator PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20 -fno-fast-math)

add_executable(spirv_opcode_support_checker ${SOURCE_FILES} ${HEADER_FILES} ${CMAKE_SOURCE_DIR}/main_opcode_checker.cpp)
target_compile_options(spirv_opcode_support_checker PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20 -fno-fast-math)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(test_type_creation)
target_link_libraries(test_type_creation PRIVATE Catch2::Catch2WithMain)
target_sources(test_type_creation PRIVATE ${CMAKE_SOURCE_DIR}/test/T_type_creation_test.cpp ${SOURCE_FILES} ${HEADER_FILES})

include(CTest)
include(Catch)
catch_discover_tests(test_type_creation REPORTER compact)

add_custom_target(run_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS test_type_creation
)

add_custom_target(always_run_tests ALL
  DEPENDS run_tests
)