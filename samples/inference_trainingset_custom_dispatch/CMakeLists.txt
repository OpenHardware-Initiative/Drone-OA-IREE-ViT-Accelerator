cmake_minimum_required(VERSION 3.16)
project(ITAViTLSTM_Inference CXX C)

# --- 1. Include Subdirectories for Plugins and Utilities ---
# This adds our custom C ops and the MLIR transform spec.
# It is expected to define the CUSTOM_OP_OBJECT_FILE variable.
add_subdirectory(plugin)

# This adds our shared, header-only utility libraries (like stb_image).
# It is expected to define the 'project_utils' INTERFACE target.
# The path is relative; adjust if your 'utils' folder is elsewhere.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../utils utils)

# --- 2. Define and Automate Model Compilation ---
if(NOT DEFINED IREE_HOST_BIN_DIR)
  message(FATAL_ERROR "IREE_HOST_BIN_DIR is not set. Please re-run cmake with: -DIREE_HOST_BIN_DIR/path/to/iree/install/bin")
endif()

# Find the iree-compile executable using the provided path as a hint.
find_program(IREE_COMPILE_EXECUTABLE iree-compile 
  HINTS "${IREE_HOST_BIN_DIR}" 
  REQUIRED
)
message(STATUS "Using iree-compile: ${IREE_COMPILE_EXECUTABLE}")

# Define paths for our source model and the final compiled artifact.
set(SOURCE_MODEL_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/../../output/ITAViTLSTM.mlir")
set(COMPILED_MODEL_VMFB "${CMAKE_CURRENT_BINARY_DIR}/model.vmfb")
set(CUSTOM_SPEC_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/plugin/dummy_spec.mlir")
set(DUMP_COMPILATION_PHASES "${CMAKE_CURRENT_SOURCE_DIR}/compilation_phases")

# This command runs iree-compile to generate the .vmfb file.
# CMake will automatically run this command if the output file is missing
# or if any of the dependencies have changed.
add_custom_command(
  OUTPUT ${COMPILED_MODEL_VMFB}
  COMMAND ${IREE_COMPILE_EXECUTABLE}
    --iree-hal-target-backends=llvm-cpu
    --iree-llvmcpu-target-triple=x86_64-pc-linux-gnu
    --iree-llvmcpu-target-cpu=host
    --iree-global-opt-propagate-transposes=true
    --iree-opt-aggressively-propagate-transposes=true
    --iree-dispatch-creation-enable-aggressive-fusion=false 
    --iree-vm-target-truncate-unsupported-floats 
    --iree-llvmcpu-enable-ukernels=all
    --iree-input-demote-f32-to-f16 
    #debug help
    --dump-compilation-phases-to=${DUMP_COMPILATION_PHASES}
    --iree-hal-executable-object-search-path=${CMAKE_CURRENT_BINARY_DIR}/plugin/
    --iree-preprocessing-transform-spec-filename=${CUSTOM_SPEC_MLIR}
    ${SOURCE_MODEL_MLIR}
    -o ${COMPILED_MODEL_VMFB}
  DEPENDS
    ${IREE_COMPILE_EXECUTABLE}
    ${CUSTOM_OP_OBJECT_FILE}    # Dependency on the .o file from the 'plugin' subdirectory
    ${SOURCE_MODEL_MLIR}
    ${CUSTOM_SPEC_MLIR}
  VERBATIM
  COMMENT "Compiling model with custom ops to ${COMPILED_MODEL_VMFB}"
)

# This target provides a clean way to trigger the custom command above.
add_custom_target(compile_model DEPENDS ${COMPILED_MODEL_VMFB})


# --- 3. Build the Final C++ Executable ---
set(_NAME "inference_trainingset_custom_dispatch")
add_executable(${_NAME} "")

target_sources(${_NAME} PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

# Pass the path of the compiled model to the C++ code as a macro.
# This makes the executable self-contained.
target_compile_definitions(${_NAME} PRIVATE
  MODEL_VMFB_PATH="${COMPILED_MODEL_VMFB}"
)

target_compile_features(${_NAME} PRIVATE cxx_std_17)


# --- 4. Link Libraries and Set Final Dependencies ---
target_link_libraries(${_NAME} PRIVATE
  # Third party libraries via our modular utils target
  project_utils
  stdc++fs

  # The high-level IREE runtime library
  iree_runtime_runtime
)

# This crucial line ensures that CMake will always run the 'compile_model'
# target before it attempts to build the main executable.
add_dependencies(${_NAME} compile_model)