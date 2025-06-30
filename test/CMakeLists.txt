Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

set(SOURCE_FILES arithmetic_test.cpp testing_common.cpp)
set(HEADER_FILES testing_common.hpp)

add_executable(arithmetic_test)
target_sources(arithmetic_test PRIVATE ${SOURCE_FILES} ${HEADER_FILES})
target_compile_definitions(arithmetic_test PRIVATE TEST_SHADER_DIR="${CMAKE_SOURCE_DIR}/test_shaders")
target_link_libraries(arithmetic_test PRIVATE Catch2::Catch2WithMain spirv_simulator_lib)

include(CTest)
include(Catch)

catch_discover_tests(arithmetic_test REPORTER compact)

add_custom_target(run_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS arithmetic_test
)

add_custom_target(always_run_tests ALL
  DEPENDS run_tests
)