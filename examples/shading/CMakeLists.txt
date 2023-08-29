cmake_minimum_required(VERSION 3.24)
project(disgraphics_example_shading)

include(cmake/CPM.cmake)

set(CMAKE_CXX_STANDARD 23)

# disgraphics
include_directories("${disgraphics_SOURCE_DIR}/include")

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
)
include_directories("${glfw_SOURCE_DIR}/include")
include_directories("${glm_SOURCE_DIR}")

# CrossWindow ( TODO Remove glfw as it depends on GL, and use 'CrossWindow' instead )
CPMAddPackage("gh:alaingalvan/CrossWindow#5f744aa904d78fc43650f6efb01976b9561f628a")

file(GLOB sources *.c *.cpp)

add_executable(disgraphics_example_shading ${sources})

target_link_libraries(disgraphics_example_shading PRIVATE glfw glm)


