set(SOURCE_FILES ${CMAKE_SOURCE_DIR}/framework/spirv_simulator.cpp ${CMAKE_SOURCE_DIR}/framework/util.cpp)
set(HEADER_FILES ${CMAKE_SOURCE_DIR}/framework/spirv_simulator.hpp ${CMAKE_SOURCE_DIR}/framework/spirv.hpp ${CMAKE_SOURCE_DIR}/framework/util.hpp)

add_library(spirv_simulator_lib ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(spirv_simulator_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
