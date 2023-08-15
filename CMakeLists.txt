cmake_minimum_required(VERSION 3.24)
project(cpuraytracer)

include(cmake/CPM.cmake)

# glm
CPMAddPackage("gh:g-truc/glm#0.9.9.8")

# cgem
CPMAddPackage("gh:kthohr/gcem@1.17.0")
include_directories("${gcem_SOURCE_DIR}/include")

# glfw
CPMAddPackage(
        NAME glfw
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.3.2
        OPTIONS
        "GLFW_BUILD_TESTS Off"
        "GLFW_BUILD_EXAMPLES Off"
        "GLFW_BUILD_DOCS Off"
        "GLFW_INSTALL Off"
        "GLFW_USE_HYBRID_HPG On"
)

set(CMAKE_CXX_STANDARD 23)

add_executable(cpuraytracer main.cpp)

target_link_libraries(cpuraytracer PRIVATE glfw glm GL)

