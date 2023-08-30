cmake_minimum_required(VERSION 3.24)
project(include)

include(ExternalProject)

option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    set(DISGRAPHICS_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    file(GLOB sources *.c *.cpp)
    add_subdirectory(examples/shading)
endif()
