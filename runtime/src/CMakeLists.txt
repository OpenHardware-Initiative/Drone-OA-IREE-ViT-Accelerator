set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/nonloal-runtime")

set(IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")

iree_register_external_hal_driver(
  NAME
    nonlocal-sync
  SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/src/iree/hal/drivers/nonlocal_sync"
  DRIVER_TARGET
    drivers::nonlocal_sync::registration
  REGISTER_FN
    iree_hal_nonlocal_sync_driver_module_register
)

iree_register_external_hal_driver(
  NAME
    nonlocal-task
  SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/src/iree/hal/drivers/nonlocal_task"
  DRIVER_TARGET
    drivers::nonlocal_task::registration
  REGISTER_FN
    iree_hal_nonlocal_task_driver_module_register
)