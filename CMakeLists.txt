# Running with sanitizers:
# Create a _new_ build directory and run cmake with the -DSANITIZE=<option> command line option. The various options that should work are: address, thread, undefined
# You may need to run "export ASAN_OPTIONS=detect_leaks=0" before running any of the tests with the address sanitizer.

cmake_minimum_required(VERSION 3.10)

# Note, this MUST NOT be compile with -ffinite-math or -ffast-math (or any optimization option that affects nan of inf representations), this will break nan operations that we need for correct SPIRV simulation.
# In general, leave the -fno-fast-math option below alone so that we keep full IEEE 754 compliance.

project(spirv_simulator VERSION 0.0.1 DESCRIPTION "SPIRV simulator for finding memory and pointer usage information from shaders" LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(SANITIZE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(SANITIZER -fsanitize=${SANITIZE} -fstack-protector-all -fsanitize-undefined-trap-on-error)
    else()
        set(SANITIZER -fsanitize=${SANITIZE} -fstack-protector-all -fsanitize-undefined-trap-on-error -fuse-ld=gold)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_BUILD)
endif()

if(MSVC)
    set(COMPILE_OPTIONS /W4 /permissive- /fp:strict)
else()
    set(COMPILE_OPTIONS -Wall -Wextra -Wpedantic -Wshadow -fno-fast-math)
endif()

add_subdirectory(framework)
add_subdirectory(test)

add_executable(spirv_simulator ${PROJECT_SOURCE_DIR}/main.cpp)
target_compile_options(spirv_simulator PRIVATE ${SANITIZER} ${COMPILE_OPTIONS})
target_link_libraries(spirv_simulator PRIVATE ${SANITIZER} spirv_simulator_lib)
target_compile_definitions(spirv_simulator PRIVATE PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR} PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR} PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})

add_executable(spirv_opcode_support_checker ${PROJECT_SOURCE_DIR}/main_opcode_checker.cpp)
target_compile_options(spirv_opcode_support_checker PRIVATE ${SANITIZER} ${COMPILE_OPTIONS})
target_link_libraries(spirv_opcode_support_checker PRIVATE ${SANITIZER} spirv_simulator_lib)
