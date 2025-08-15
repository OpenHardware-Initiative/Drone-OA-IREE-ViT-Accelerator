# samples/utils/CMakeLists.txt

# Change the library type from INTERFACE to STATIC.
# This creates a real library file (.a) containing compiled code.
add_library(project_utils STATIC
  "libs.cpp"
)

# Change target_include_directories from INTERFACE to PUBLIC.
# PUBLIC applies the include path to the library itself (so libs.cpp can find stb_image.h)
# AND to any targets that link against it (like your main executable).
target_include_directories(project_utils PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)