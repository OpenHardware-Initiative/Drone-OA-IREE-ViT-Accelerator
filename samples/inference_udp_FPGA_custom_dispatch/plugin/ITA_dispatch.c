// dummy_dispatch.c
// Fixed version accounting for how IREE actually passes static memrefs

#include <stddef.h>
#include <stdint.h>

typedef uint16_t float16_t;

// CRITICAL: For static-sized memrefs like memref<1x128x128xf16>,
// IREE might pass them differently than dynamic memrefs.
// 
// Based on the crash pattern, it seems IREE is passing:
// - A struct or packed representation
// - Or the parameters are in a different order
//
// Let's try the simplest possible signature first:
void ITASelfAttention_workgroup(
    const float16_t* input,
    float16_t* output)
{
    const size_t total_elements = 1 * 128 * 128;
    
    // Simple copy for testing
    for (size_t i = 0; i < total_elements; ++i) {
        output[i] = input[i];
    }
}

// Alternative: If IREE still expects the 5-parameter expansion
// but the actual data comes through differently
void ITASelfAttention_workgroup_expanded(
    const float16_t* binding0,
    const float16_t* binding0_aligned,
    size_t binding0_offset,
    size_t binding0_size,
    size_t binding0_stride,
    float16_t* binding1,
    float16_t* binding1_aligned,
    size_t binding1_offset,
    size_t binding1_size,
    size_t binding1_stride)
{
    // Use aligned pointers if base pointers are null
    const float16_t* input = binding0 ? binding0 : binding0_aligned;
    float16_t* output = binding1 ? binding1 : binding1_aligned;
    
    if (!input || !output) {
        // Something is very wrong - just return
        return;
    }
    
    const size_t total_elements = 1 * 128 * 128;
    
    for (size_t i = 0; i < total_elements; ++i) {
        output[i] = input[i];
    }
}