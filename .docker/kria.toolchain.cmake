# CMake Toolchain file for cross-compiling for AMD Kria SOM (aarch64)
#
# This file configures CMake to use the aarch64 cross-compiler and the
# Kria sysroot provided within the Docker environment.

# --- Basic System Setup ---
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# --- Toolchain Definition ---
# The location of the target's sysroot inside the Docker container
set(CMAKE_SYSROOT /opt/sysroot/sysroots/aarch64-xilinx-linux)

# Specify the cross-compilers provided by the environment
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# --- Search Path Configuration ---
# Configure how CMake finds libraries, headers, and programs.
# Search for programs ONLY in the host system's paths (NEVER in the sysroot).
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers ONLY in the sysroot's paths.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# --- Additional Compiler Flags ---
# Set architecture-specific flags for ARMv8-A (which includes Cortex-A53)
set(CMAKE_C_FLAGS "-march=armv8-a" CACHE STRING "C compiler flags")
set(CMAKE_CXX_FLAGS "-march=armv8-a" CACHE STRING "C++ compiler flags")
