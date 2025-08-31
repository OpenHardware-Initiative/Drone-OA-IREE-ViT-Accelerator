// dummy_dispatch.c
// Custom ITA operations - bare metal, no standard library

#include <stddef.h>
#include <stdint.h>

// Define half-precision float type
typedef uint16_t float16_t;

// Helper function to convert float16 to float32
static inline float half_to_float(float16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        return 0.0f;
    }
    
    uint32_t f32_bits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    return *(float*)&f32_bits;
}

// Helper function to convert float32 to float16  
static inline float16_t float_to_half(float f) {
    uint32_t f32_bits = *(uint32_t*)&f;
    uint32_t sign = (f32_bits >> 31) & 0x1;
    uint32_t exponent = (f32_bits >> 23) & 0xFF;
    uint32_t mantissa = f32_bits & 0x7FFFFF;
    
    int32_t adj_exp = (int32_t)exponent - 127 + 15;
    
    if (adj_exp <= 0) {
        return (float16_t)(sign << 15);
    } else if (adj_exp >= 31) {
        return (float16_t)((sign << 15) | (0x1F << 10));
    }
    
    return (float16_t)((sign << 15) | (adj_exp << 10) | (mantissa >> 13));
}

// Simple inline absolute value for float
static inline float simple_fabsf(float x) {
    return x < 0.0f ? -x : x;
}

// ITAFF_workgroup: Computes absolute value for entire tensor
// Following IREE's memref expansion pattern
void ITAFF_workgroup(
    // Input binding (binding 0) - memref expansion
    const float16_t* restrict binding0, 
    const float16_t* restrict binding0_aligned,
    size_t binding0_offset, 
    size_t binding0_size, 
    size_t binding0_stride,
    // Output binding (binding 1) - memref expansion  
    float16_t* restrict binding1, 
    float16_t* restrict binding1_aligned,
    size_t binding1_offset, 
    size_t binding1_size, 
    size_t binding1_stride,
    // Dimensions
    size_t d0, size_t d1, size_t d2) {
    
    size_t total_elements = d0 * d1 * d2;
    
    // Process all elements in single workgroup
    for (size_t i = 0; i < total_elements; i++) {
        float val = half_to_float(binding0[i]);
        float abs_val = simple_fabsf(val);
        binding1[i] = float_to_half(abs_val);
    }
}

// ITASelfAttention_workgroup: Computes negation for entire tensor
// Following IREE's memref expansion pattern
void ITASelfAttention_workgroup(
    // Input binding (binding 0) - memref expansion
    const float16_t* restrict binding0, 
    const float16_t* restrict binding0_aligned,
    size_t binding0_offset, 
    size_t binding0_size, 
    size_t binding0_stride,
    // Output binding (binding 1) - memref expansion
    float16_t* restrict binding1, 
    float16_t* restrict binding1_aligned,
    size_t binding1_offset, 
    size_t binding1_size, 
    size_t binding1_stride,
    // Dimensions
    size_t d0, size_t d1, size_t d2) {
    
    size_t total_elements = d0 * d1 * d2;
    
    // Process all elements in single workgroup
    for (size_t i = 0; i < total_elements; i++) {
        float val = half_to_float(binding0[i]);
        float neg_val = -val;
        binding1[i] = float_to_half(neg_val);
    }
}