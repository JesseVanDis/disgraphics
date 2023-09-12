cmake_minimum_required(VERSION 3.24)
project(disgraphics_example_shading)

include(cmake/CPM.cmake)

set(CMAKE_CXX_STANDARD 23)

# disgraphics
include_directories("${DISGRAPHICS_SOURCE_DIR}/include")

# glm
CPMAddPackage("gh:g-truc/glm#5c46b9c07008ae65cb81ab79cd677ecc1934b903") # 0.9.9.9

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
        "GLFW_USE_WAYLAND On" # for wayland, make sure you install 'extra-cmake-modules wayland-devel wayland-protocols-devel libxkbcommon-devel freeglut-devel'
)
include_directories("${glfw_SOURCE_DIR}/include")
include_directories("${glm_SOURCE_DIR}")

file(GLOB sources *.c *.cpp)
add_executable(disgraphics_example_shading ${sources})

# may want 'sudo apt install libgl1-mesa-dev' for -lGL
target_link_libraries(disgraphics_example_shading PRIVATE GL glfw glm)


