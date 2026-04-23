cmake_minimum_required(VERSION 3.12)
project(distance_calculator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(distance SHARED distance_calculator.cpp)
set_target_properties(distance PROPERTIES OUTPUT_NAME "distance")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(distance PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(distance PRIVATE DISTANCE_USE_OPENMP=1)
endif()
