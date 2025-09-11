include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(SOURCE_FILES testing_common.cpp )
set(HEADER_FILES testing_common.hpp)

add_executable(arithmetic_test)
target_sources(arithmetic_test PRIVATE ${SOURCE_FILES} ${HEADER_FILES} arithmetic_test.cpp)
target_compile_definitions(arithmetic_test PRIVATE TEST_SHADER_DIR="${PROJECT_SOURCE_DIR}/test_shaders")
target_link_libraries(arithmetic_test PRIVATE GTest::gtest_main GTest::gmock spirv_simulator_lib)

add_executable(declaration_test)
target_sources(declaration_test PRIVATE ${SOURCE_FILES} ${HEADER_FILES} declaration_test.cpp)
target_compile_definitions(declaration_test PRIVATE TEST_SHADER_DIR="${PROJECT_SOURCE_DIR}/test_shaders")
target_link_libraries(declaration_test PRIVATE GTest::gtest_main GTest::gmock spirv_simulator_lib)

include(CTest)
include(GoogleTest)

gtest_discover_tests(arithmetic_test REPORTER compact DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(declaration_test REPORTER compact DISCOVERY_MODE PRE_TEST)

add_custom_target(run_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS arithmetic_test declaration_test
)

file(GLOB SPIRV_FILES ${PROJECT_SOURCE_DIR}/test_shaders/*.spv ${PROJECT_SOURCE_DIR}/test_shaders/*.spirv)
foreach(X ${SPIRV_FILES})
	get_filename_component(TESTNAME ${X} NAME_WLE)
	add_test(NAME ${TESTNAME} COMMAND ${PROJECT_BINARY_DIR}/spirv_simulator ${X})
endforeach()
