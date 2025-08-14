# --- 1. Build the Final Executable ---

set(_NAME "ITAViTLSTM_test_data")
add_executable(${_NAME} "")

target_sources(${_NAME} PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_compile_features(${_NAME} PRIVATE cxx_std_17)

# --- 2. Define Third-Party Dependencies (stb_image) ---
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE
  "${IREE_ITA_SOURCE_DIR}/samples/inference/include"
)
target_compile_definitions(stb_image INTERFACE STB_IMAGE_IMPLEMENTATION)

# --- 3. Link Libraries ---
target_link_libraries(${_NAME} PRIVATE
  # Third party libraries
  stb_image
  stdc++fs

  # The high-level IREE runtime library that pulls in all configured dependencies
  iree_runtime_runtime
  #iree_hal_driver_local_sync
)