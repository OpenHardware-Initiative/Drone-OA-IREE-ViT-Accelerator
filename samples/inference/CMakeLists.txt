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
  "--iree-llvmcpu-target-cpu-features=+neon"
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
    ${IREE_HOST_BIN_DIR}/iree-compile
    ${_COMPILE_ARGS}
    "${MODEL_FILE_PATH}"
    -o "${VMFB_FILE}"
  DEPENDS "${MODEL_FILE_PATH}"
  COMMENT "Compiling ${MODEL_FILE_PATH} for the aarch64 target"
  COMMAND_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)

add_custom_target(compile_inference_model ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
)

message(STATUS "Inference sample configured. VMFB will be generated at: ${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}")

# --- 3. Wrap the Generated Artifacts in Proper CMake Targets ---
add_library(${MODEL_BASE_NAME}_static_lib STATIC
  "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
)
target_include_directories(${MODEL_BASE_NAME}_static_lib INTERFACE
  "${CMAKE_CURRENT_BINARY_DIR}"
)
add_dependencies(${MODEL_BASE_NAME}_static_lib compile_inference_model)

set_target_properties(${MODEL_BASE_NAME}_static_lib PROPERTIES
  LINKER_LANGUAGE C
)

iree_c_embed_data(
  NAME
    ${MODEL_BASE_NAME}_c
  IDENTIFIER
    ${MODEL_BASE_NAME}
  SRCS
    "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  C_FILE_OUTPUT
    "${MODEL_BASE_NAME}_c.c"
  H_FILE_OUTPUT
    "${MODEL_BASE_NAME}_c.h"
  FLATTEN
  PUBLIC
)

# --- 4. Define Third-Party Dependencies (stb_image) ---
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE
  "${IREE_ITA_SOURCE_DIR}/samples/inference/include"
)
target_compile_definitions(stb_image INTERFACE
  STB_IMAGE_IMPLEMENTATION
)


################################################################################
# --- 5. Build the Final Executable ---
################################################################################

set(_NAME "ITAViTLSTM_test_data")
add_executable(${_NAME} "")

target_sources(${_NAME} PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_compile_features(${_NAME} PRIVATE cxx_std_17)

# The final, correct way to link the libraries.
target_link_libraries(${_NAME} PRIVATE
  # Our generated model library and embedded bytecode data
  ${MODEL_BASE_NAME}_static_lib
  ${MODEL_BASE_NAME}_c

  # Third party libraries
  stb_image
  stdc++fs

  # FINAL FIX: Link the high-level IREE runtime target.
  # This automatically includes all drivers and backends enabled
  # in the main CMake configuration.
  iree_runtime_runtime
)

add_dependencies(${_NAME} compile_inference_model)