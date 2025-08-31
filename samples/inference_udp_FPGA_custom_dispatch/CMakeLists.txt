cmake_minimum_required(VERSION 3.16)
project(InferenceUDPDISPATCH CXX C)

# --- 1. Include Plugin Subdirectory ---
add_subdirectory(plugin)

# --- 2. Configure Model Compilation ---
if(NOT DEFINED IREE_HOST_BIN_DIR)
  message(FATAL_ERROR "IREE_HOST_BIN_DIR is not set. Please re-run cmake with: -DIREE_HOST_BIN_DIR=/path/to/iree/install/bin")
endif()
find_program(IREE_COMPILE_EXECUTABLE iree-compile HINTS "${IREE_HOST_BIN_DIR}" REQUIRED)
message(STATUS "Using iree-compile: ${IREE_COMPILE_EXECUTABLE}")

# --- Define the compilation flags ---
set(IREE_COMPILE_FLAGS
  --iree-hal-target-backends=llvm-cpu
  --iree-llvmcpu-target-triple=aarch64-linux-gnu
  --iree-llvmcpu-target-cpu=cortex-a53
  --iree-opt-aggressively-propagate-transposes=true
  --iree-global-opt-propagate-transposes=true
  --iree-hal-executable-debug-level=3
  --iree-opt-data-tiling=false
  --aarch64-use-aa
  --iree-dispatch-creation-enable-aggressive-fusion=false
  --iree-vm-target-truncate-unsupported-floats
  --iree-scheduling-dump-statistics-format=verbose
  --iree-scheduling-dump-statistics-file=compilation_info.txt
  --iree-llvmcpu-enable-ukernels=all
  --iree-llvmcpu-target-cpu-features=+neon,+fp-armv8,+crypto
  --aarch64-neon-syntax=generic
  --iree-input-demote-f32-to-f16
  --dump-compilation-phases-to=${CMAKE_CURRENT_SOURCE_DIR}/compilation_phases
)

set(SOURCE_MODEL_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/../../output/ITAViTLSTM.mlir")

# --- 3. Custom Command to Compile the CUSTOM DISPATCH Model ---
set(COMPILED_MODEL_CUSTOM_FILENAME "lstmnetvit_custom_aarch64.vmfb")
set(COMPILED_MODEL_CUSTOM "${CMAKE_CURRENT_BINARY_DIR}/${COMPILED_MODEL_CUSTOM_FILENAME}")
set(CUSTOM_SPEC_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/plugin/ITA_spec.mlir")

add_custom_command(
  OUTPUT ${COMPILED_MODEL_CUSTOM}
  COMMAND ${IREE_COMPILE_EXECUTABLE}
          ${IREE_COMPILE_FLAGS}
          --iree-hal-executable-object-search-path=${CMAKE_CURRENT_BINARY_DIR}/plugin/
          --iree-preprocessing-transform-spec-filename=${CUSTOM_SPEC_MLIR}
          ${SOURCE_MODEL_MLIR}
          -o ${COMPILED_MODEL_CUSTOM}
  DEPENDS ${IREE_COMPILE_EXECUTABLE} ${CUSTOM_OP_OBJECT_FILE} ${SOURCE_MODEL_MLIR} ${CUSTOM_SPEC_MLIR}
  VERBATIM
  COMMENT "Compiling CUSTOM DISPATCH model for aarch64"
)
add_custom_target(compile_custom_model DEPENDS ${COMPILED_MODEL_CUSTOM})


# --- 4. Build the Final Executable ---
set(_NAME "inference_udp_fpga_dispatch")
add_executable(${_NAME} "")
target_sources(${_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_compile_features(${_NAME} PRIVATE cxx_std_17)

# --- 5. Link Libraries ---
target_link_libraries(${_NAME} PRIVATE stdc++fs iree_runtime_runtime)

# --- 6. Configure Executable for Custom Dispatch ---
# Hardcode the model filename and dependency, since it's the only option.
message(STATUS "Configuring build for CUSTOM DISPATCH model only.")
target_compile_definitions(${_NAME} PRIVATE
  MODEL_VMFB_FILENAME="${COMPILED_MODEL_CUSTOM_FILENAME}"
)
add_dependencies(${_NAME} compile_custom_model)


# --- 7. Define Installation Path ---
set(ACCELERATOR_PROJECT_ROOT "$ENV{WORKSPACE_DIR}"
  CACHE PATH "Root of the Drone-OA-IREE-ViT-Accelerator project")
if(NOT ACCELERATOR_PROJECT_ROOT)
  set(ACCELERATOR_PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../..")
endif()
message(STATUS "Accelerator Project Root: ${ACCELERATOR_PROJECT_ROOT}")
set(INSTALL_DIR "${ACCELERATOR_PROJECT_ROOT}/third_party/kria_inference/samples")
message(STATUS "Installation directory set to: ${INSTALL_DIR}")

# --- 8. Add Install Rules ---
# Install the executable
install(
  TARGETS ${_NAME}
  RUNTIME DESTINATION "${INSTALL_DIR}"
  COMPONENT applications
)
# Install the custom model alongside the executable
install(
  FILES ${COMPILED_MODEL_CUSTOM}
  DESTINATION ${INSTALL_DIR}
  RENAME ${COMPILED_MODEL_CUSTOM_FILENAME}
  COMPONENT models
)