# OMM Sample

*Opacity Micro-Map sample* demostrates usage of Opacity Micro-Maps in Raytracing. Sample details:
- Opacity Micro-Map SDK is used for OMM baking. Both CPU baker and GPU baker are supported in the sample: https://github.com/NVIDIA-RTX/OMM
- Rendering and Denoising are based on NRDSample: https://github.com/NVIDIAGameWorks/NRDSample

**Note:** The DXR 1.2 API requires a preview driver, available at: https://developer.nvidia.com/downloads/shadermodel6-9-preview-driver

## Build instructions

- Install [*Cmake*](https://cmake.org/download/) 4.1+
- Install on
    - Windows: latest *WindowsSDK* (22000+), *VulkanSDK* (1.3.231+)
    - Linux: a C++17 compiler, Ninja, and windowing development packages:
        - Wayland (default): `libwayland-dev`, `wayland-protocols`, `libxkbcommon-dev` on Ubuntu
        - X11 (`-DNRIF_USE_WAYLAND=OFF`): `libx11-dev`, `libxrandr-dev`, `libxinerama-dev`, `libxcursor-dev`, `libxi-dev` on Ubuntu
- Build (variant 1) - using *Git* and *CMake* explicitly
    - Clone project and init submodules
    - Generate and build project using *CMake*
- Build (variant 2) - by running scripts:
    - Run `1-Deploy`
    - Run `2-Build`

Run the commands for your platform and target architecture from the repository root. Both deployment scripts forward additional CMake arguments, such as `-DUSE_MINIMAL_DATA=ON`. Use a fresh `_Build` directory when changing architecture or toolchain.

### Windows

Install Visual Studio 2022 C++ tools for the host and target architectures. Set `VULKAN_SDK` to the host architecture's Vulkan SDK (ShaderMake and DXC run on the host).

```bat
1-Deploy.bat -A x64
2-Build.bat
```

For ARM64, replace `x64` with `ARM64`.

### Linux

Builds use Ninja Multi-Config. The downloaded DXC requires x64 glibc 2.38+. On ARM64 hosts, configure transparent x64 emulation and install the x64 runtime libraries.

For native x64 or ARM64 builds:

```sh
sh 1-Deploy.sh
sh 2-Build.sh
```

To use native DXC instead, put `dxc` in `PATH` and add `-DSHADERMAKE_FIND_DXC=OFF` to the deployment command.

For cross-compilation, add `-DCMAKE_TOOLCHAIN_FILE=...` to the deployment command.

The window is created with Wayland by default. To build the X11 backend instead (also usable through XWayland), add `-DNRIF_USE_WAYLAND=OFF` to the deployment command.

### CMake options

- `USE_MINIMAL_DATA=ON` - download minimal resource package (90MB); run with `--scene=ShaderBalls/ShaderBalls.gltf`
- `RTXCR_INTEGRATION=OFF` - disable the optional hair and skin rendering integration and Claire assets
- `NRIF_USE_WAYLAND=OFF` - Linux only: use the X11 window backend instead of Wayland
- `SHADERMAKE_FIND_DXC=OFF` with `SHADERMAKE_DXC_PATH=C:/path/to/dxc.exe` - use a custom DXIL compiler
- `SHADERMAKE_FIND_DXC_VK=OFF` with `SHADERMAKE_DXC_VK_PATH=C:/path/to/dxc.exe` - use a custom SPIR-V compiler

## How to run

- Run `3-Run OMM sample.bat` on Windows or `sh "3-Run OMM sample.sh"` on Linux from the repository root and select the runtime options.
- Executables are in `_Bin/Release` or `_Bin/Debug`. When launching them directly, set the working directory to the repository root so they can load assets from `_Data`.

## Command Line Arguments
- If [Smart Command Line Arguments extension for Visual Studio](https://marketplace.visualstudio.com/items?itemName=MBulli.SmartCommandlineArguments) is installed, all command line arguments will be loaded into corresponding window
- `--api=D3D12`, `--api=VULKAN` for graphics API selection 
- `--width=x`, `--height=y` for setting window size
- `--scene=*path*` for scene selection
- `--help` to print all the available commands

## Minimum Requirements

Any RTX GPU:
- RTX 5000 series
- RTX 4000 series
- RTX 3000 series
- RTX 2000 series

## Usage

OMM:
- Set baker settings in the UI and press Bake OMMs
- For CPU baker it is recommended to use cache

Navigation:
- Right mouse button + W/S/A/D - move camera
- Mouse scroll - accelerate / decelerate
- F1 - toggle "gDebug" (can be useful for debugging and experiments)
- Tab - UI toggle
- Space - animation toggle

Notes:
- Check "Show all settings" in the UI to see rendering setting.

## OMM Workflow

1. Bake OMM data with Opacity Micro-Map SDK
2. Use OMM data to build OMM with corresponding API functions
3. Build BLAS using OMM
