# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(fpga_iree_server CXX C)

set(CMAKE_CXX_STANDARD 17)

# Find IREE package. This works seamlessly inside your Docker container.
find_package(IREE REQUIRED CONFIG)

# --- Stage 1: Compile the MLIR model to a static library and a .vmfb file ---
set(IREE_COMPILE_ARGS
  # Model and backend settings
  "--iree-input-type=stablehlo"
  "--iree-hal-target-backends=llvm-cpu"
  # aarch64 target specification
  "--iree-llvmcpu-target-triple=aarch64-linux-gnu"
  "--iree-llvmcpu-target-cpu=cortex-a53"
  "--iree-llvmcpu-target-cpu-features=+neon,+crypto,+fp-armv8"
  "--iree-llvmcpu-link-static"
  "--iree-vm-target-index-bits=64"
)

# Define consistent names for our generated artifacts
set(MODEL_BASE_NAME "ITAViTLSTM")
set(STATIC_LIB_OBJECT_FILE "${MODEL_BASE_NAME}_static_lib.o")
set(STATIC_LIB_HEADER_FILE "${MODEL_BASE_NAME}_static_lib.h")
set(VMFB_FILE "${MODEL_BASE_NAME}.vmfb")
set(MODEL_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/output/ITAViTLSTM.mlir")

# Add the static library path to the compile args
list(APPEND IREE_COMPILE_ARGS "--iree-llvmcpu-static-library-output-path=${STATIC_LIB_OBJECT_FILE}")

# Custom command to invoke iree-compile
add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_HEADER_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  COMMAND
    ${IREE_HOST_BIN_DIR}/iree-compile
    ${IREE_COMPILE_ARGS}
    "${MODEL_FILE_PATH}"
    -o "${VMFB_FILE}"
  DEPENDS ${MODEL_FILE_PATH}
  COMMENT "Compiling ${MODEL_FILE_PATH} to a static library and VMFB"
  COMMAND_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)

# A target to drive the custom command
add_custom_target(compile_model ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}")

# --- Stage 2: Embed the .vmfb file into a C source file ---
iree_c_embed_data(
  NAME ${MODEL_BASE_NAME}_c
  IDENTIFIER ${MODEL_BASE_NAME} # Becomes the prefix for the create() function
  SRCS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  C_FILE_OUTPUT "${MODEL_BASE_NAME}_c.c"
  H_FILE_OUTPUT "${MODEL_BASE_NAME}_c.h"
  FLATTEN
  PUBLIC
)
add_dependencies(iree-c-${MODEL_BASE_NAME}_c compile_model)

# --- Stage 3: Build the final C++ server executable ---
add_executable(fpga_server
  src/server.cpp
  src/pre_processing.hpp
  src/post_processing.hpp
  # Add the generated object file from iree-compile
  "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
)

target_include_directories(fpga_server PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}" # For generated headers
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(fpga_server PRIVATE
  iree::runtime
  iree::hal
  iree::hal-local-sync
  iree::hal::loaders::static_library_loader
  iree-c-${MODEL_BASE_NAME}_c # Link the embedded bytecode
)