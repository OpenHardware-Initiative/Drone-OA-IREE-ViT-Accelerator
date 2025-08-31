# plugin/CMakeLists.txt
# MODIFIED: Corrected clang flags for aarch64

# --- 1. Target Architecture Configuration ---
set(_ARCH "aarch64")
set(_LLVM_ARCH "aarch64")

# --- 2. Compiler Discovery ---
if(DEFINED IREE_CLANG_BINARY AND EXISTS "${IREE_CLANG_BINARY}")
  set(CLANG_EXECUTABLE ${IREE_CLANG_BINARY})
else()
  find_program(CLANG_EXECUTABLE clang REQUIRED)
endif()
message(STATUS "Using clang for custom ops: ${CLANG_EXECUTABLE}")

# --- 3. Custom Command to Build the Object File ---
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/ITA_dispatch_${_ARCH}.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/ITA_dispatch.c
    ${CLANG_EXECUTABLE}
  COMMAND ${CLANG_EXECUTABLE}
    # --- Target and environment ---
    -target aarch64-linux-gnu
    -std=c17
    -ffreestanding
    -fvisibility=hidden
    -fno-plt
    -fno-rtti
    -fno-exceptions
    -fPIC

    # --- Optimization and code generation ---
    -O3
    -g0 # Remove debug symbols
    -fdata-sections
    -ffunction-sections

    # --- Architecture specific flags for Cortex-A53 ---
    -mcpu=cortex-a53
    # MODIFIED: Removed the invalid '--features' flag and used the correct '-march' flag
    # to enable optional extensions. NEON is already implied by cortex-a53.
    -march=armv8-a+crypto

    # --- Input / Output ---
    -c ${CMAKE_CURRENT_SOURCE_DIR}/ITA_dispatch.c
    -o ${CMAKE_CURRENT_BINARY_DIR}/ITA_dispatch_${_ARCH}.o
  VERBATIM
  COMMENT "Building custom ITA ops object for ${_ARCH}"
)

# --- 4. CMake Target Definition ---
add_custom_target(ITA_dispatch_object_${_ARCH}
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ITA_dispatch_${_ARCH}.o
)

# --- 5. Export Variable to Parent Scope ---
set(CUSTOM_OP_OBJECT_FILE
  ${CMAKE_CURRENT_BINARY_DIR}/ITA_dispatch_${_ARCH}.o
  PARENT_SCOPE
)

# --- 6. Convenience Target ---
add_custom_target(custom_ops_objects_ITA DEPENDS ITA_dispatch_object_${_ARCH})