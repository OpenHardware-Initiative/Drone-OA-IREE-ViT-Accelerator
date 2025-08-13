// fpga_link.hpp
#pragma once

#include <cstdint>
#include <vector>
#include <cstring>      // For memcpy
#include <arpa/inet.h>  // For ntohl, htonl (standard on Linux/POSIX)

// Network Configuration
const int PORT = 10001;

// Define the structure of the packet from Host -> FPGA.
// Matches Python format: '>5400sff16s' (5400 + 4 + 4 + 16 = 5424 bytes)
// Using #pragma pack to ensure no compiler padding interferes with the layout.
#pragma pack(push, 1)
struct HostToFpgaPacket {
    uint8_t image[5400];
    float   desired_velocity;
    float   position_x;
    float   quaternion[4];
};

// Define the structure of the packet from FPGA -> Host.
// Matches Python format: '>12s' (3 * 4 = 12 bytes)
struct FpgaToHostPacket {
    float velocity_command[3];
};
#pragma pack(pop)


// --- Endianness Conversion for Floats ---
// C/C++ standard library is missing a direct equivalent for floats,
// so we use a safe, portable method.

// network-to-host float
inline float ntohf(uint32_t net) {
    net = ntohl(net);
    float host;
    std::memcpy(&host, &net, sizeof(float));
    return host;
}

// host-to-network float
inline uint32_t htonf(float host) {
    uint32_t net;
    std::memcpy(&net, &host, sizeof(float));
    return htonl(net);
}


// --- Serialization / Deserialization ---

/**
 * @brief Unpacks a raw network buffer into our C++ struct, handling byte order.
 * @param buffer The raw byte buffer received from the network.
 * @param packet The C++ struct to populate.
 * @return True on success, false if the buffer is the wrong size.
 */
bool unpack_frame(const std::vector<char>& buffer, HostToFpgaPacket& packet) {
    if (buffer.size() != sizeof(HostToFpgaPacket)) {
        return false;
    }

    const char* ptr = buffer.data();

    // Image data (uint8_t doesn't need endian conversion)
    std::memcpy(packet.image, ptr, 5400);
    ptr += 5400;

    // desired_velocity
    packet.desired_velocity = ntohf(*reinterpret_cast<const uint32_t*>(ptr));
    ptr += 4;

    // position_x
    packet.position_x = ntohf(*reinterpret_cast<const uint32_t*>(ptr));
    ptr += 4;

    // quaternion
    for (int i = 0; i < 4; ++i) {
        packet.quaternion[i] = ntohf(*reinterpret_cast<const uint32_t*>(ptr));
        ptr += 4;
    }

    return true;
}

/**
 * @brief Packs our C++ reply struct into a raw network buffer for sending.
 * @param packet The C++ struct to serialize.
 * @return A std::vector<char> ready to be sent over the network.
 */
std::vector<char> pack_reply(const FpgaToHostPacket& packet) {
    std::vector<char> buffer(sizeof(FpgaToHostPacket));
    char* ptr = buffer.data();

    // velocity_command
    for (int i = 0; i < 3; ++i) {
        *reinterpret_cast<uint32_t*>(ptr) = htonf(packet.velocity_command[i]);
        ptr += 4;
    }
    return buffer;
}