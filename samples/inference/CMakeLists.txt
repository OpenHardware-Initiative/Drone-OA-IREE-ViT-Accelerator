if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR NOT IREE_HAL_DRIVER_LOCAL_SYNC)
  return()
endif()

# --- 1. Define Model and Compilation Arguments ---
set(MODEL_BASE_NAME "ITAViTLSTM")
set(MODEL_FILE_PATH "${IREE_ITA_SOURCE_DIR}/output/ITAViTLSTM.mlir")
set(STATIC_LIB_OBJECT_FILE "${MODEL_BASE_NAME}_static_lib.o")
set(STATIC_LIB_HEADER_FILE "${MODEL_BASE_NAME}_static_lib.h")
set(VMFB_FILE "${MODEL_BASE_NAME}.vmfb")
set(_COMPILE_ARGS
  "--iree-input-type=stablehlo"
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
add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_HEADER_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  COMMAND
    ${IREE_HOST_BIN_DIR}/iree-compile ${_COMPILE_ARGS} "${MODEL_FILE_PATH}" -o "${VMFB_FILE}"
  DEPENDS "${MODEL_FILE_PATH}"
  COMMENT "Compiling ${MODEL_FILE_PATH} for the target"
  COMMAND_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)
# This custom target ensures the command above is actually run.
add_custom_target(compile_inference_model ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
)

# --- 3. Wrap the Generated Artifacts in Proper CMake Targets ---

# Create a STATIC library target from the .o file.
# The dependency is handled by making this target depend on the file output by the custom command.
add_library(${MODEL_BASE_NAME}_static_lib STATIC
  "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
)
target_include_directories(${MODEL_BASE_NAME}_static_lib INTERFACE
  "${CMAKE_CURRENT_BINARY_DIR}"
)
# The dependency is on the custom target that creates the file.
add_dependencies(${MODEL_BASE_NAME}_static_lib compile_inference_model)

# Embed the VMFB bytecode into a C source file.
iree_c_embed_data(
  NAME
    ${MODEL_BASE_NAME}_c
  IDENTIFIER
    ${MODEL_BASE_NAME}
  SRCS
    # This creates an implicit dependency on the command that OUTPUTS this file.
    "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  C_FILE_OUTPUT
    "${MODEL_BASE_NAME}_c.c"
  H_FILE_OUTPUT
    "${MODEL_BASE_NAME}_c.h"
  FLATTEN
  PUBLIC
)
# REMOVED the incorrect add_dependencies command. The SRCS line above handles it.


# --- 4. Build the Final Executable ---
iree_cc_binary(
  NAME
    main # This will create the executable
  SRCS
    "main.cpp"
  DEPS
    # The :: before the target name is a best practice for clarity
    ::${MODEL_BASE_NAME}_c
    ${MODEL_BASE_NAME}_static_lib

    # Core IREE libraries
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::loaders::static_library_loader
    iree::hal::drivers::local_sync
    iree::vm::bytecode::module
    iree::runtime
  INCLUDES
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)