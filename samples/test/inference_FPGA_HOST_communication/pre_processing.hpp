#pragma once

#include <vector>
#include <cstdint>
#include "fpga_link.hpp" // For HostToFpgaPacket

/**
 * @brief Converts the uint8 image from the network packet to a float32 vector.
 * The model expects float values (likely normalized between 0.0 and 1.0).
 * @param packet_in The network packet containing the uint8_t image.
 * @return A vector of floats representing the processed image.
 */
std::vector<float> pre_process_image(const HostToFpgaPacket& packet_in) {
    const size_t num_pixels = 60 * 90;
    std::vector<float> float_image(num_pixels);

    for (size_t i = 0; i < num_pixels; ++i) {
        // Normalize the uint8 pixel value (0-255) to a float (0.0-1.0).
        // This mirrors the common np.astype(float) / 255.0 pattern.
        float_image[i] = static_cast<float>(packet_in.image[i]) / 255.0f;
    }

    return float_image;
}