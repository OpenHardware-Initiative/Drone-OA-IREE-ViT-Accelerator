# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(fpga_server CXX)

# Set C++ standard for modern features like std::chrono
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This defines our target executable and the source files it needs.
# The toolchain file will ensure it's compiled with the correct compiler and flags.
add_executable(fpga_server server.cpp)

# When you add IREE back in, you will add target_link_libraries() here.
# For now, no external libraries are needed.

# Optional: Add an install rule for convenience
install(TARGETS fpga_server DESTINATION bin)