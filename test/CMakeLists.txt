include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(SOURCE_FILES arithmetic_test.cpp testing_common.cpp)
set(HEADER_FILES testing_common.hpp)

add_executable(arithmetic_test)
target_sources(arithmetic_test PRIVATE ${SOURCE_FILES} ${HEADER_FILES})
target_compile_definitions(arithmetic_test PRIVATE TEST_SHADER_DIR="${CMAKE_SOURCE_DIR}/test_shaders")
target_link_libraries(arithmetic_test PRIVATE GTest::gtest_main spirv_simulator_lib)

include(CTest)
include(GoogleTest)

gtest_discover_tests(arithmetic_test REPORTER compact)

add_custom_target(run_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS arithmetic_test
)

add_custom_target(always_run_tests ALL
  DEPENDS run_tests
)