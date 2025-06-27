Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

set(SOURCE_FILES T_type_creation_test.cpp)
set(HEADER_FILES testing_common.hpp)

add_executable(test_type_creation)
target_sources(test_type_creation PRIVATE ${SOURCE_FILES} ${HEADER_FILES})
target_compile_definitions(test_type_creation PRIVATE TEST_SHADER_DIR="${CMAKE_SOURCE_DIR}/test_shaders")
target_link_libraries(test_type_creation PRIVATE Catch2::Catch2WithMain spirv_simulator_lib)

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