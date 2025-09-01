cmake_minimum_required(VERSION 3.16)
project(ITAViTLSTM_Inference CXX C)

# --- 1. Include Subdirectories for Plugins and Utilities ---
# This adds our custom C ops and the MLIR transform spec.
# It is expected to define the CUSTOM_OP_OBJECT_FILE variable.
add_subdirectory(plugin)


# --- 2. Define and Automate Model Compilation ---
if(NOT DEFINED IREE_HOST_BIN_DIR)
  message(FATAL_ERROR "IREE_HOST_BIN_DIR is not set. Please re-run cmake with: -DIREE_HOST_BIN_DIR=/path/to/iree/install/bin")
endif()

# Find the iree-compile executable using the provided path as a hint.
find_program(IREE_COMPILE_EXECUTABLE iree-compile 
  HINTS "${IREE_HOST_BIN_DIR}" 
  REQUIRED
)
message(STATUS "Using iree-compile: ${IREE_COMPILE_EXECUTABLE}")

# --- Define paths for model, output, and all debug artifacts ---
set(SOURCE_MODEL_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/../../output/ITAViTLSTM.mlir")
set(COMPILED_MODEL_VMFB "${CMAKE_CURRENT_BINARY_DIR}/model.vmfb")
set(CUSTOM_SPEC_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/plugin/dummy_spec.mlir")

# Create a dedicated directory for all debug outputs
set(DEBUG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/debug_output")
set(DUMP_COMPILATION_PHASES "${DEBUG_OUTPUT_DIR}/compilation_phases")
set(HAL_EXECUTABLE_DUMPS "${DEBUG_OUTPUT_DIR}/hal_executables")
set(DISPATCH_GRAPH_DOT "${DEBUG_OUTPUT_DIR}/dispatch_graph.dot")
set(VM_SOURCE_LISTING_MLIR "${DEBUG_OUTPUT_DIR}/annotated_vm_module.mlir")

# Ensure the directories for debug dumps exist before compilation
file(MAKE_DIRECTORY ${DUMP_COMPILATION_PHASES})
file(MAKE_DIRECTORY ${HAL_EXECUTABLE_DUMPS})


# This command runs iree-compile to generate the .vmfb file.
add_custom_command(
  OUTPUT ${COMPILED_MODEL_VMFB}
  COMMAND ${IREE_COMPILE_EXECUTABLE}
    # --- Standard compilation flags from your original file ---
    --iree-hal-target-backends=llvm-cpu
    --iree-llvmcpu-target-triple=x86_64-pc-linux-gnu
    --iree-llvmcpu-target-cpu=host
    --iree-global-opt-propagate-transposes=true
    --iree-opt-aggressively-propagate-transposes=true
    --iree-dispatch-creation-enable-aggressive-fusion=false 
    --iree-vm-target-truncate-unsupported-floats 
    --iree-llvmcpu-enable-ukernels=all
    #--iree-input-demote-f32-to-f16 
    --iree-hal-executable-object-search-path=${CMAKE_CURRENT_BINARY_DIR}/plugin/
    --iree-preprocessing-transform-spec-filename=${CUSTOM_SPEC_MLIR}

    # =================================================================
    # --- Comprehensive Debugging and Verbosity Flags ---
    # =================================================================
    
    # [1] Enable verification at each step of the pipeline. Aborts on failure.
    --verify
    
    # [2] Print MLIR IR verbosely before and after every pass.
    # WARNING: This will produce a very large amount of console output.
    #--mlir-print-ir-before-all
    #--mlir-print-ir-after-all
    #--mlir-print-ir-after-change
    
    # [3] On any diagnostic error, print the failing operation and a stack trace.
    --mlir-print-op-on-diagnostic
    --mlir-print-stacktrace-on-diagnostic

    # [4] Display statistics and timing information for each compiler pass.
    --mlir-pass-statistics
    --mlir-timing
    
    # [5] Dump MLIR for each major compilation phase to a directory.
    --dump-compilation-phases-to=${DUMP_COMPILATION_PHASES}
    
    # [6] Dump all HAL executable artifacts (sources, intermediates, binaries).
    #--iree-hal-dump-executable-files-to=${HAL_EXECUTABLE_DUMPS}
    
    # [7] Generate a .dot graph file of the dispatch flow.
    #--iree-flow-dump-dispatch-graph-output-file=${DISPATCH_GRAPH_DOT}
    
    # [8] Set the highest debug level for executable translation.
    --iree-hal-executable-debug-level=3
    
    # [9] For the LLVMCPU backend, generate and embed DWARF/PDB symbols.
    --iree-llvmcpu-debug-symbols=true
    
    # [10] Keep intermediate linker artifacts (like .o files) for inspection.
    --iree-llvmcpu-keep-linker-artifacts=true
    
    # [11] Make the output .vmfb file a polyglot zip for easier inspection.
    --iree-vm-emit-polyglot-zip=true
    
    # [12] Dump an annotated version of the final VM MLIR module.
    --iree-vm-bytecode-source-listing=${VM_SOURCE_LISTING_MLIR}

    # --- Input/Output Files ---
    ${SOURCE_MODEL_MLIR}
    -o ${COMPILED_MODEL_VMFB}
  DEPENDS
    ${IREE_COMPILE_EXECUTABLE}
    ${CUSTOM_OP_OBJECT_FILE}    # Dependency on the .o file from the 'plugin' subdirectory
    ${SOURCE_MODEL_MLIR}
    ${CUSTOM_SPEC_MLIR}
  VERBATIM
  COMMENT "Compiling model with MAX DEBUG FLAGS to ${COMPILED_MODEL_VMFB}"
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