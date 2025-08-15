# custom_ops/CMakeLists.txt

# This sample only builds for x86-64. You can add more architectures
# by creating new commands and updating the MLIR spec.
set(_ARCH "x86_64")
set(_LLVM_ARCH "x86_64") # LLVM target triple component

# Find a working clang compiler. IREE's build system provides IREE_CLANG_BINARY,
# but we can try to find it on the system as well.
if(DEFINED IREE_CLANG_BINARY AND EXISTS "${IREE_CLANG_BINARY}")
  set(CLANG_EXECUTABLE ${IREE_CLANG_BINARY})
else()
  find_program(CLANG_EXECUTABLE clang REQUIRED)
endif()
message(STATUS "Using clang for custom ops: ${CLANG_EXECUTABLE}")

# This custom command compiles our C code into a freestanding ELF object file.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/custom_unary_ops_${_ARCH}.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_unary_ops.c
    ${CLANG_EXECUTABLE}
  COMMAND ${CLANG_EXECUTABLE}
    # --- Target and environment ---
    -target ${_LLVM_ARCH}-none-elf # Crucial: specifies a bare-metal target
    -std=c17
    -ffreestanding # Don't link against libc
    -fvisibility=hidden # Hide symbols by default
    -fno-plt
    -fno-rtti
    -fno-exceptions
    # --- Optimization and code gen ---
    -O2
    -g # Add debug symbols
    -fdata-sections
    -ffunction-sections
    # --- Input / Output ---
    -c ${CMAKE_CURRENT_SOURCE_DIR}/custom_unary_ops.c
    -o ${CMAKE_CURRENT_BINARY_DIR}/custom_unary_ops_${_ARCH}.o
  VERBATIM
  COMMENT "Building custom unary op object for ${_ARCH}"
)

# Create a target that represents the build step for our object file.
add_custom_target(custom_unary_ops_object_${_ARCH} DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/custom_unary_ops_${_ARCH}.o
)

# This is a convenience target. Building it will ensure all custom objects are created.
add_custom_target(custom_ops_objects DEPENDS
    custom_unary_ops_object_${_ARCH}
)