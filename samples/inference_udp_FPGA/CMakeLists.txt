# --- 1. Build the Final Executable ---
set(_NAME "inference_udp_FPGA")
add_executable(${_NAME} "")

target_sources(${_NAME} PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_compile_features(${_NAME} PRIVATE cxx_std_17)

# --- 2. Link Libraries ---
# This is the crucial step to fix the "stb_image.h not found" error.
# It links your application to the 'project_utils' library, which tells
# the compiler to look inside the 'utils/include' directory.
target_link_libraries(${_NAME} PRIVATE
  # Third party libraries
  stdc++fs

  # The high-level IREE runtime library
  iree_runtime_runtime
)

set(ACCELERATOR_PROJECT_ROOT "$ENV{WORKSPACE_DIR}"
  CACHE PATH "Root of the Drone-OA-IREE-ViT-Accelerator project")

# If it's still not defined, calculate it relative to the IREE project.
if(NOT ACCELERATOR_PROJECT_ROOT)
  set(ACCELERATOR_PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../..")
endif()

message(STATUS "Accelerator Project Root: ${ACCELERATOR_PROJECT_ROOT}")


# --- 3. Add Install Rule ---
# This rule tells CMake where to copy the final compiled executable
# when you run the "install" step.
install(
  TARGETS ${_NAME}
  RUNTIME DESTINATION "${ACCELERATOR_PROJECT_ROOT}/third_party/kria_inference/samples"
  COMPONENT applications
)