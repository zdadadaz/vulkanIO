# Vulkan Image Sequence Player

This project is a Vulkan-based application designed to play sequence of raw images and demonstrate advanced rendering techniques such as Ray Marching, Temporal Anti-Aliasing (TAA), and depth processing.

## Features

- **Raw Image Playback**: Loads and displays sequences of raw images (RGBA8888).
- **Ray Marching Integration**: Includes `RM.frag` for ray marching effects.
- **Temporal Anti-Aliasing (TAA)**: Implements TAA logic in `TNR.frag` using motion vectors and history buffers.
- **Depth Processing**: Handles depth texture inputs via `depthDS.frag`.
- **Shader Pipeline**: Custom shader pipeline including `draw.frag`, `SNR.frag`, and vertex shaders.

## Prerequisites

To build and run this project, you need the following installed on your system:

- **C++ Compiler**: Support for C++17.
- **CMake**: Version 3.17 or higher.
- **Vulkan SDK**: Including the validation layers and `glslangValidator`.
- **GLFW3**: For window creation and context management.

## Building the Project

The project uses CMake for the build system.

### Standard CMake Build

1.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```

2.  Configure the project:
    ```bash
    cmake ..
    ```

3.  Build the executable:
    ```bash
    make
    ```

## Running the Application

### macOS

A convenience script `run.sh` is provided for macOS users, which sets up the necessary MoltenVK environment variables and builds/runs the project.

```bash
./run.sh
```

Ensure `run.sh` is executable:
```bash
chmod +x run.sh
```

### Manual Execution

After building, you can run the executable directly from the project root (ensure paths to assets/shaders are correct relative to execution):

```bash
./build/VulkanImagePlayer
```

## Project Structure

- `src/`: C++ source files (`main.cpp`, `VulkanRenderer.cpp`, etc.).
- `shaders/`: GLSL shader files (`.vert`, `.frag`).
- `CMakeLists.txt`: CMake build configuration.
- `run.sh`: Helper script for building and running on macOS.
