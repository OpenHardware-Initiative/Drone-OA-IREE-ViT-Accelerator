list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(IREE_ITA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/plugins)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/samples/ samples)