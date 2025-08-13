#samples/inference/CMakeLists.txt
# This file is designed to be included by the main IREE build system.

# This sample requires these components. The main IREE build will check this.
if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR NOT IREE_HAL_DRIVER_LOCAL_SYNC)
  return()
endif()

# --- 1. Define Model and Compilation Arguments ---
set(MODEL_BASE_NAME "ITAViTLSTM")

# Define the path to your model RELATIVE TO THE PROJECT ROOT, NOT this directory.
# CMAKE_CURRENT_SOURCE_DIR is samples/inference. We go up two levels.
set(MODEL_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../output/ITAViTLSTM.mlir")

set(STATIC_LIB_OBJECT_FILE "${MODEL_BASE_NAME}_static_lib.o")
set(STATIC_LIB_HEADER_FILE "${MODEL_BASE_NAME}_static_lib.h")
set(VMFB_FILE "${MODEL_BASE_NAME}.vmfb")

# Setup the exact compile args from your working command
set(_COMPILE_ARGS
  "--iree-hal-target-backends=llvm-cpu"
  "--iree-llvmcpu-target-triple=aarch64-linux-gnu"
  "--iree-llvmcpu-target-cpu=cortex-a53"
  "--iree-llvmcpu-target-cpu-features=+neon,+fp-armv8,+crypto"
  "--iree-llvmcpu-link-static"
  "--iree-llvmcpu-static-library-output-path=${STATIC_LIB_OBJECT_FILE}"
  "--iree-dispatch-creation-enable-aggressive-fusion=false"
  "--iree-input-demote-f32-to-f16"
)

# --- 2. Create the Model Artifacts using a Custom Command ---
# This command runs the HOST's iree-compile during the build.
add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_HEADER_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  COMMAND
    # IREE_HOST_BIN_DIR is provided automatically by the main build
    ${IREE_HOST_BIN_DIR}/iree-compile
    ${_COMPILE_ARGS}
    "${MODEL_FILE_PATH}"
    -o "${VMFB_FILE}"
  DEPENDS
    ${MODEL_FILE_PATH}
  COMMENT "Compiling ${MODEL_FILE_PATH} for the target"
  COMMAND_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)

# This custom target ensures the command above is actually run.
add_custom_target(compile_inference_model ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
)

# --- 3. Wrap the Generated Artifacts in Proper CMake Targets ---

# Create a STATIC library target from the .o file
add_library(${MODEL_BASE_NAME}_static_lib STATIC
  IMPORTED
  GLOBAL
)
set_target_properties(${MODEL_BASE_NAME}_static_lib PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
)
add_dependencies(${MODEL_BASE_NAME}_static_lib compile_inference_model)

# Embed the VMFB bytecode into a C source file.
iree_c_embed_data(
  NAME ${MODEL_BASE_NAME}_c
  IDENTIFIER ${MODEL_BASE_NAME}
  SRCS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  C_FILE_OUTPUT "${MODEL_BASE_NAME}_c.c"
  H_FILE_OUTPUT "${MODEL_BASE_NAME}_c.h"
  FLATTEN
  PUBLIC
)
add_dependencies(iree-c-${MODEL_BASE_NAME}_c compile_inference_model)

# --- 4. Build the Final Executable using IREE's helper ---
iree_cc_binary(
  NAME
    test_inference_with_data
  SRCS
    # The source file is in the same directory as this CMakeLists.txt
    "main.cpp"
  DEPS
    # Link against the targets we just created
    ::${MODEL_BASE_NAME}_c
    ${MODEL_BASE_NAME}_static_lib
    # Link against the core IREE libraries
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::loaders::static_library_loader
    iree::hal::drivers::local_sync
    iree::vm::bytecode::module
    iree::runtime
)

# Set the include directories for our executable
target_include_directories(test_inference_with_data PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}" # For generated headers like ITAViTLSTM_c.h
  "${CMAKE_CURRENT_SOURCE_DIR}"   # Allows #include "include/stb_image.h"
)