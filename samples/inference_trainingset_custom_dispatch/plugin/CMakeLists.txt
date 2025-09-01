# plugin/CMakeLists.txt
# Build configuration for custom operations

set(_ARCH "x86_64")
set(_LLVM_ARCH "embedded-elf-x86_64")

# Find clang compiler
if(DEFINED IREE_CLANG_BINARY AND EXISTS "${IREE_CLANG_BINARY}")
  set(CLANG_EXECUTABLE ${IREE_CLANG_BINARY})
else()
  find_program(CLANG_EXECUTABLE clang REQUIRED)
endif()
message(STATUS "Using clang for custom ops: ${CLANG_EXECUTABLE}")

# Build the object file with the correct name
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/dummy_dispatch_${_ARCH}.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/dummy_dispatch.c
    ${CLANG_EXECUTABLE}
  COMMAND ${CLANG_EXECUTABLE}
    # --- Target and environment ---
    -target ${_LLVM_ARCH}-none-elf
    -std=c17
    -ffreestanding
    -fvisibility=hidden
    -fno-plt
    -fno-rtti
    -fno-exceptions
    -fPIC  # Position independent code required for shared objects
    # --- Optimization and code gen ---
    -O2
    -g
    -fdata-sections
    -ffunction-sections
    # --- Architecture specific ---
    -march=znver2  # Match the CPU in the MLIR spec
    -mtune=znver2  # Optimize for Zen 2 architecture
    # --- Input / Output ---
    -c ${CMAKE_CURRENT_SOURCE_DIR}/dummy_dispatch.c
    -o ${CMAKE_CURRENT_BINARY_DIR}/dummy_dispatch_${_ARCH}.o
  VERBATIM
  COMMENT "Building custom ITA ops object for ${_ARCH}"
)

# Create target for the object file
add_custom_target(dummy_dispatch_object_${_ARCH} 
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dummy_dispatch_${_ARCH}.o
)

# Export the object file path for the parent CMakeLists.txt
set(CUSTOM_OP_OBJECT_FILE 
  ${CMAKE_CURRENT_BINARY_DIR}/dummy_dispatch_${_ARCH}.o 
  PARENT_SCOPE
)

# Convenience target
add_custom_target(custom_ops_objects DEPENDS dummy_dispatch_object_${_ARCH})