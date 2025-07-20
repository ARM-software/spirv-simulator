set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/framework/spirv_simulator.cpp ${PROJECT_SOURCE_DIR}/framework/util.cpp)
set(HEADER_FILES ${PROJECT_SOURCE_DIR}/framework/spirv_simulator.hpp ${PROJECT_SOURCE_DIR}/framework/spirv.hpp ${PROJECT_SOURCE_DIR}/framework/util.hpp)

add_library(spirv_simulator_lib ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(spirv_simulator_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
