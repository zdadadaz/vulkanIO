cmake_minimum_required(VERSION 3.17)
project(VulkanImagePlayer)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(${Vulkan_INCLUDE_DIRS})

# Shader compilation
find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin /opt/homebrew/bin)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPV_DIR})

set(SHADERS
    ${SHADER_DIR}/shader.vert
    ${SHADER_DIR}/shader.frag
    ${SHADER_DIR}/RM.frag
    ${SHADER_DIR}/draw.frag
    ${SHADER_DIR}/depthDS.frag
    ${SHADER_DIR}/TNR.frag
    ${SHADER_DIR}/SNR.frag
    ${SHADER_DIR}/SNR2.frag
    ${SHADER_DIR}/TNR2.frag
    ${SHADER_DIR}/computeFresnel.frag
)

foreach(SHADER ${SHADERS})
    get_filename_component(FILENAME ${SHADER} NAME)
    add_custom_command(
        OUTPUT ${SPV_DIR}/${FILENAME}.spv
        COMMAND ${GLSL_VALIDATOR} -V ${SHADER} -o ${SPV_DIR}/${FILENAME}.spv
        DEPENDS ${SHADER}
        COMMENT "Compiling ${FILENAME}"
    )
    list(APPEND SPV_SHADERS ${SPV_DIR}/${FILENAME}.spv)
endforeach()

add_executable(VulkanImagePlayer src/main.cpp src/VulkanRenderer.cpp ${SPV_SHADERS})
target_link_libraries(VulkanImagePlayer glfw ${Vulkan_LIBRARIES})
target_include_directories(VulkanImagePlayer PRIVATE ${glfw3_INCLUDE_DIRS})

target_compile_definitions(VulkanImagePlayer PRIVATE 
    SHADER_DIR="${SPV_DIR}"
)
