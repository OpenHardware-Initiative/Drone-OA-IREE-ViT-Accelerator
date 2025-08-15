#include <stdint.h>
#include <stddef.h>
#include <stdio.h> // For printf debugging

/**
 * @brief Custom kernel for ITAFF operation (formerly abs).
 */
void ITAFF_workgroup(
    // Input tensor (%in)
    const uint16_t* restrict binding0, const uint16_t* restrict binding0_aligned,
    size_t binding0_offset, size_t binding0_dims_0, size_t binding0_dims_1, size_t binding0_dims_2,
    size_t binding0_strides_0, size_t binding0_strides_1, size_t binding0_strides_2,
    // Output tensor (%out)
    uint16_t* restrict binding1, uint16_t* restrict binding1_aligned,
    size_t binding1_offset, size_t binding1_dims_0, size_t binding1_dims_1, size_t binding1_dims_2,
    size_t binding1_strides_0, size_t binding1_strides_1, size_t binding1_strides_2,
    // Dimensions passed as arguments
    size_t dim0, size_t dim1, size_t dim2, size_t tid) {

  // Print statement with the new name
  printf("--- Custom Dispatch: ITAFF_workgroup called ---\n");

  size_t total_elements = dim0 * dim1 * dim2;
  size_t workgroup_size = 64; // A common workgroup size
  size_t end = tid + workgroup_size;
  if (end > total_elements) {
    end = total_elements;
  }

  for (size_t i = tid; i < end; ++i) {
    // f16 abs is clearing the sign bit (MSB)
    binding1[i] = binding0[i] & 0x7FFF;
  }
}

/**
 * @brief Custom kernel for ITASelfAttention operation (formerly neg).
 */
void ITASelfAttention_workgroup(
    // Input tensor (%in)
    const uint16_t* restrict binding0, const uint16_t* restrict binding0_aligned,
    size_t binding0_offset, size_t binding0_dims_0, size_t binding0_dims_1, size_t binding0_dims_2,
    size_t binding0_strides_0, size_t binding0_strides_1, size_t binding0_strides_2,
    // Output tensor (%out)
    uint16_t* restrict binding1, uint16_t* restrict binding1_aligned,
    size_t binding1_offset, size_t binding1_dims_0, size_t binding1_dims_1, size_t binding1_dims_2,
    size_t binding1_strides_0, size_t binding1_strides_1, size_t binding1_strides_2,
    // Dimensions passed as arguments
    size_t dim0, size_t dim1, size_t dim2, size_t tid) {

  // Print statement with the new name
  printf("--- Custom Dispatch: ITASelfAttention_workgroup called ---\n");

  size_t total_elements = dim0 * dim1 * dim2;
  size_t workgroup_size = 64;
  size_t end = tid + workgroup_size;
  if (end > total_elements) {
    end = total_elements;
  }

  for (size_t i = tid; i < end; ++i) {
    // Only flip the sign bit if the value is not zero.
    if (binding0[i] != 0x0000 && binding0[i] != 0x8000) {
        binding1[i] = binding0[i] ^ 0x8000;
    } else {
        binding1[i] = binding0[i];
    }
  }
}