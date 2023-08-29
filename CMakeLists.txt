cmake_minimum_required(VERSION 3.24)
project(include)

option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    set(disgraphics_SOURCE_DIR ${CMAKE_SOURCE_DIR})
    add_subdirectory(examples/shading)
endif()
