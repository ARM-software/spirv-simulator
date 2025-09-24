set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/framework/spirv_simulator.cpp ${PROJECT_SOURCE_DIR}/framework/util.cpp)
set(HEADER_FILES ${PROJECT_SOURCE_DIR}/framework/spirv_simulator.hpp ${PROJECT_SOURCE_DIR}/framework/spirv.hpp ${PROJECT_SOURCE_DIR}/framework/util.hpp)

add_library(spirv_simulator_lib ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(spirv_simulator_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include)

# --- SPIRV-Tools submodule setup ------------------------------------------
option(SPIRV_TOOLS_SYNC_DEPS "Run utils/git-sync-deps inside the SPIRV-Tools submodule" ON)
set(SPIRV_TOOLS_DIR ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools CACHE PATH "Path to the SPIRV-Tools submodule root")

if(SPIRV_TOOLS_SYNC_DEPS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${SPIRV_TOOLS_DIR}/utils/git-sync-deps
    WORKING_DIRECTORY ${SPIRV_TOOLS_DIR}
    RESULT_VARIABLE _sync_res
  )
  if(NOT _sync_res EQUAL 0)
    message(FATAL_ERROR "utils/git-sync-deps failed in ${SPIRV_TOOLS_DIR}")
  endif()
endif()

set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)

add_subdirectory(${SPIRV_TOOLS_DIR} ${CMAKE_BINARY_DIR}/_spvtools EXCLUDE_FROM_ALL)

# Pick the right target name (varies with static/shared settings)
if(TARGET SPIRV-Tools)
  set(_spvtools_lib SPIRV-Tools)
elseif(TARGET SPIRV-Tools-static)
  set(_spvtools_lib SPIRV-Tools-static)
else()
  message(FATAL_ERROR "Could not find SPIRV-Tools target after add_subdirectory()")
endif()

target_compile_features(spirv_simulator_lib PUBLIC cxx_std_20)
set_property(TARGET spirv_simulator_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(spirv_simulator_lib PRIVATE ${_spvtools_lib})

# In most cases, SPIRV-Tools exports its include dir via the target.
# If you want to be explicit, uncomment:
# target_include_directories(simulator PRIVATE ${SPIRV_TOOLS_DIR}/include)