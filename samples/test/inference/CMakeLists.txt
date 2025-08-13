# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(fpga_iree_project CXX C) # Changed project name for clarity

set(CMAKE_CXX_STANDARD 17)

# Find IREE package. This works seamlessly inside your Docker container.
find_package(IREE REQUIRED CONFIG)

# --- Stage 1: Compile the MLIR model (This section is shared) ---
# (This section is identical to the previous CMakeLists.txt)
set(IREE_COMPILE_ARGS
  "--iree-input-type=stablehlo"
  "--iree-hal-target-backends=llvm-cpu"
  "--iree-llvmcpu-target-triple=aarch64-linux-gnu"
  "--iree-llvmcpu-target-cpu=cortex-a53"
  "--iree-llvmcpu-target-cpu-features=+neon,+crypto,+fp-armv8"
  "--iree-llvmcpu-link-static"
  "--iree-vm-target-index-bits=64"
)
set(MODEL_BASE_NAME "ITAViTLSTM")
set(STATIC_LIB_OBJECT_FILE "${MODEL_BASE_NAME}_static_lib.o")
set(STATIC_LIB_HEADER_FILE "${MODEL_BASE_NAME}_static_lib.h")
set(VMFB_FILE "${MODEL_BASE_NAME}.vmfb")
set(MODEL_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/output/ITAViTLSTM.mlir")
list(APPEND IREE_COMPILE_ARGS "--iree-llvmcpu-static-library-output-path=${STATIC_LIB_OBJECT_FILE}")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_HEADER_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  COMMAND ${IREE_HOST_BIN_DIR}/iree-compile ${IREE_COMPILE_ARGS} "${MODEL_FILE_PATH}" -o "${VMFB_FILE}"
  DEPENDS ${MODEL_FILE_PATH}
  COMMENT "Compiling ${MODEL_FILE_PATH} to a static library and VMFB"
  COMMAND_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)
add_custom_target(compile_model ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}")

# --- Stage 2: Embed the .vmfb file (This section is shared) ---
iree_c_embed_data(
  NAME ${MODEL_BASE_NAME}_c
  IDENTIFIER ${MODEL_BASE_NAME}
  SRCS "${CMAKE_CURRENT_BINARY_DIR}/${VMFB_FILE}"
  C_FILE_OUTPUT "${MODEL_BASE_NAME}_c.c"
  H_FILE_OUTPUT "${MODEL_BASE_NAME}_c.h"
  FLATTEN
  PUBLIC
)
add_dependencies(iree-c-${MODEL_BASE_NAME}_c compile_model)

# --- Stage 3: Build the executables ---

# Target 1: The UDP network server
add_executable(fpga_server
  src/server.cpp
)

# Target 2: The self-contained inference test
add_executable(test_inference
  src/test_inference.cpp
)

# Link both targets to the same common dependencies
foreach(TARGET IN ITEMS fpga_server test_inference)
  target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}" # For generated headers
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  # Both targets need the compiled model object file
  target_sources(${TARGET} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB_OBJECT_FILE}"
  )
  target_link_libraries(${TARGET} PRIVATE
    iree::runtime
    iree::hal
    iree::hal-local-sync
    iree::hal::loaders::static_library_loader
    iree-c-${MODEL_BASE_NAME}_c # Link the embedded bytecode
  )
endforeach()