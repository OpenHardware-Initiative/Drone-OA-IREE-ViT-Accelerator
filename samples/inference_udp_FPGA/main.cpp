#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <arpa/inet.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <array>
#include <cmath>

// --- IREE C API Headers ---
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/base/allocator.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"

// --- Image Loading Library ---
#include "include/stb_image.h"

// Network Configuration
const int PORT = 10001;
const char *FPGA_IP = "10.42.0.14";

// Packet sizes
const size_t IMAGE_SIZE = 5400; // 60x90 uint8
const size_t VELOCITY_SIZE = 4; // float32
const size_t POSX_SIZE = 4;     // float32
const size_t QUAT_SIZE = 16;    // 4x float32
const size_t width = 60;
const size_t height = 90;

const size_t VEL_CMD_SIZE = 12; // 3x float32

const size_t SEND_PACKET_SIZE =
    IMAGE_SIZE + VELOCITY_SIZE + POSX_SIZE + QUAT_SIZE; // 5424 bytes
const size_t REPLY_PACKET_SIZE = VEL_CMD_SIZE;

typedef uint16_t half_float_t;

// --- Data Structures ---
struct TelemetryData {
    float desired_velocity;
    float quaternion[4];
};

struct ReceivedPacket {
  std::vector<uint8_t> image;
  float desired_velocity;
  float position_x;
  std::vector<float> quaternion;
};

// This union is used to convert float to bytes.
union floatToBytes {
  float floatValue;
  unsigned char bytes[4];
};
// --- Forward Declarations & Helpers ---

// FINAL FIX #1: Corrected function declaration with "_module".
extern "C" iree_status_t iree_hal_local_sync_driver_module_register(
    iree_hal_driver_registry_t* registry);

extern "C" iree_status_t iree_allocator_libc_ctl(
    void* self, iree_allocator_command_t command,
    const void* params, void** inout_ptr);



iree_status_t create_tensor_view(iree_hal_device_t* device, const void* data, const iree_hal_dim_t* shape, iree_host_size_t shape_rank, iree_hal_element_type_t element_type, iree_hal_buffer_view_t** out_buffer_view);
std::vector<float> print_output_tensor(iree_hal_buffer_view_t* view);
bool load_telemetry_for_image(const std::filesystem::path& csv_path, const std::string& image_timestamp_str, TelemetryData& out_telemetry);
ReceivedPacket unpack_frame(const char *packet);
std::array<float, 3>calculate_final_velocity(float* raw_output,
                         float desired_vel, float pos_x);
std::vector<unsigned char> pack_reply(float* velocity_cmd);
iree_status_t convert_f16_view_to_f32_view(iree_hal_device_t* device, iree_hal_buffer_view_t* f16_view, iree_hal_buffer_view_t** out_f32_view);

// --- Main Application ---
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " /path/to/model.vmfb /path/to/root_data_folder" << std::endl;
        return 1;
    }
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }

    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY; // 0.0.0.0
    server_address.sin_port = htons(PORT);

    if (bind(sock, (struct sockaddr *)&server_address, sizeof(server_address)) <
        0) {
        std::cerr << "Error binding socket" << std::endl;
        close(sock);
        return -1;
    }

    std::cout << "UDP server listening on port " << PORT << std::endl;

    std::filesystem::path vmfb_path(argv[1]);
    std::filesystem::path root_data_dir(argv[2]);

    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_t* instance = NULL;
    iree_allocator_t host_allocator = { .self = NULL, .ctl = iree_allocator_libc_ctl };
    IREE_CHECK_OK(iree_runtime_instance_create(&instance_options, host_allocator, &instance));
    
    // FINAL FIX #2: Corrected function call with "_module".
    IREE_CHECK_OK(iree_hal_local_sync_driver_module_register(
        iree_runtime_instance_driver_registry(instance)));

    iree_hal_device_t* device = NULL;
    IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
        instance, iree_make_cstring_view("local-sync"), &device));

    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    iree_runtime_session_t* session = NULL;
    IREE_CHECK_OK(iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session));

    std::cout << "Loading model: " << vmfb_path << std::endl;
    IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(session, vmfb_path.c_str()));

    iree_hal_buffer_view_t* hidden_state_h = NULL;
    iree_hal_buffer_view_t* hidden_state_c = NULL;
    std::vector<char> zero_buffer(3 * 1 * 128 * sizeof(float), 0);
    const iree_hal_dim_t hidden_shape[] = {3, 1, 128};
    IREE_CHECK_OK(create_tensor_view(device, zero_buffer.data(), hidden_shape, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &hidden_state_h));
    IREE_CHECK_OK(create_tensor_view(device, zero_buffer.data(), hidden_shape, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &hidden_state_c));

    while (true) {
        // Buffer for incoming packet
        char packet[8192];
        struct sockaddr_in client_address;
        socklen_t client_address_len = sizeof(client_address);
        
        // Receive packet from client
        ssize_t bytes_received =
            recvfrom(sock, packet, sizeof(packet), 0,
                    (struct sockaddr *)&client_address, &client_address_len);

        if (bytes_received < 0) {
            std::cerr << "Error receiving packet" << std::endl;
            continue;
        }

        ReceivedPacket received_data = unpack_frame(packet);

        std::vector<float> image_f32(width * height);
        for (int i = 0; i < width * height; ++i) image_f32[i] = static_cast<float>(received_data.image.data()[i]) / 255.0f;

        iree_runtime_call_t call;
        const char* func_name = "module.main_graph";
        IREE_CHECK_OK(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(func_name), &call));

        const iree_hal_dim_t img_shape[] = {1, 1, 60, 90};
        iree_hal_buffer_view_t* img_view = NULL;
        IREE_CHECK_OK(create_tensor_view(device, image_f32.data(), img_shape, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &img_view));
        const iree_hal_dim_t vel_shape[] = {1, 1};
        iree_hal_buffer_view_t* vel_view = NULL;
        IREE_CHECK_OK(create_tensor_view(device, &received_data.desired_velocity, vel_shape, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &vel_view));
        const iree_hal_dim_t quat_shape[] = {1, 4};
        iree_hal_buffer_view_t* quat_view = NULL;
        IREE_CHECK_OK(create_tensor_view(device, received_data.quaternion.data(), quat_shape, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &quat_view));

        IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, img_view));
        IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, vel_view));
        IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, quat_view));
        IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, hidden_state_h));
        IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, hidden_state_c));

        IREE_CHECK_OK(iree_runtime_call_invoke(&call, 0));
        iree_hal_buffer_view_t* raw_output_view = NULL;
        iree_hal_buffer_view_t* new_hidden_state_h = NULL;
        iree_hal_buffer_view_t* new_hidden_state_c = NULL;
        IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &raw_output_view));
        IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &new_hidden_state_h));
        IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &new_hidden_state_c));

        std::vector<float> raw_output_data = print_output_tensor(raw_output_view);

        auto final_velocity = calculate_final_velocity(raw_output_data.data(), received_data.desired_velocity, received_data.position_x);
        
        auto packed_velocity = pack_reply(final_velocity.data());

        // Send reply back to client
        ssize_t bytes_sent =
            sendto(sock, packed_velocity.data(), packed_velocity.size(), 0,
                (struct sockaddr *)&client_address, client_address_len);

        if (bytes_sent < 0) {
        std::cerr << "Error sending reply" << std::endl;
        } else {
        std::cout << "Sent " << bytes_sent << " bytes reply to client"
                    << std::endl;
        }
        
        iree_hal_buffer_view_release(hidden_state_h);
        iree_hal_buffer_view_release(hidden_state_c);
        hidden_state_h = new_hidden_state_h;
        hidden_state_c = new_hidden_state_c;

        iree_hal_buffer_view_release(img_view);
        iree_hal_buffer_view_release(vel_view);
        iree_hal_buffer_view_release(quat_view);
        iree_hal_buffer_view_release(raw_output_view);
        iree_runtime_call_deinitialize(&call);
    }
        
    iree_hal_buffer_view_release(hidden_state_h);
    iree_hal_buffer_view_release(hidden_state_c);

    iree_runtime_session_release(session);
    iree_hal_device_release(device);
    iree_runtime_instance_release(instance);
    return 0;
}


// --- Implementations of Helper Functions ---

static iree_host_size_t get_element_byte_size(iree_hal_element_type_t element_type) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: return 4;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16: return 2;
    case IREE_HAL_ELEMENT_TYPE_INT_32: return 4;
    case IREE_HAL_ELEMENT_TYPE_INT_16: return 2;
    case IREE_HAL_ELEMENT_TYPE_INT_8: return 1;
    default: return 0;
  }
}

iree_status_t create_tensor_view(iree_hal_device_t* device, const void* data, const iree_hal_dim_t* shape, iree_host_size_t shape_rank, iree_hal_element_type_t element_type, iree_hal_buffer_view_t** out_buffer_view) {
    iree_hal_buffer_params_t buffer_params = {0};
    buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    
    iree_host_size_t byte_length = get_element_byte_size(element_type);
    for(int i = 0; i < shape_rank; ++i) {
        byte_length *= shape[i];
    }

    return iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), shape_rank, shape, element_type, 
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params, 
        iree_make_const_byte_span(data, byte_length), out_buffer_view);
}

inline void swap_endian_4(unsigned char *data) {
  std::swap(data[0], data[3]);
  std::swap(data[1], data[2]);
}

float half_to_float32(half_float_t h) {
    union { float f; uint32_t u; } p;
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    if (exponent == 0) {
        if (mantissa == 0) { p.u = sign << 31; return p.f; }
        else {
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            exponent++; mantissa &= ~0x400;
        }
    } else if (exponent == 31) {
        p.u = (sign << 31) | 0x7f800000 | (mantissa << 13); return p.f;
    }
    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;
    p.u = (sign << 31) | (exponent << 23) | mantissa;
    return p.f;
}


std::vector<float> print_output_tensor(iree_hal_buffer_view_t* view) {
    if (!view) { std::cout << "  <null>" << std::endl; return {}; }
    iree_hal_buffer_mapping_t mapped_memory;
    IREE_CHECK_OK(iree_hal_buffer_map_range(iree_hal_buffer_view_buffer(view), IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &mapped_memory));
    iree_hal_element_type_t element_type = iree_hal_buffer_view_element_type(view);
    iree_host_size_t element_count = iree_hal_buffer_view_element_count(view);

    std::vector<float> model_output;

    if (element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
        const half_float_t* data_ptr = (const half_float_t*)mapped_memory.contents.data;
        for (iree_host_size_t i = 0; i < element_count; ++i) { model_output.push_back(half_to_float32(data_ptr[i])); }
    } else if (element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
        const float* data_ptr = (const float*)mapped_memory.contents.data;
        for (iree_host_size_t i = 0; i < element_count; ++i) { model_output.push_back(data_ptr[i]); }
    }
    
    std::cout << "  Model Output:    [";
    for(size_t i = 0; i < model_output.size(); ++i) { std::cout << model_output[i] << (i < model_output.size() - 1 ? ", " : ""); }
    std::cout << "]" << std::endl;

    iree_hal_buffer_unmap_range(&mapped_memory);
    return model_output;
}

ReceivedPacket unpack_frame(const char *packet) {
  ReceivedPacket received_packet;
  size_t offset = 0;

  // Unpack image (5400 bytes)
  received_packet.image.resize(IMAGE_SIZE);
  std::memcpy(received_packet.image.data(), packet + offset, IMAGE_SIZE);
  offset += IMAGE_SIZE;

  // Unpack desired velocity (4 bytes, big-endian)
  unsigned char velocity_bytes[VELOCITY_SIZE];
  std::memcpy(velocity_bytes, packet + offset, VELOCITY_SIZE);
  swap_endian_4(velocity_bytes);
  received_packet.desired_velocity = *reinterpret_cast<float *>(velocity_bytes);
  offset += VELOCITY_SIZE;

  // Unpack position x (4 bytes, big-endian)
  unsigned char pos_x_bytes[POSX_SIZE];
  std::memcpy(pos_x_bytes, packet + offset, POSX_SIZE);
  swap_endian_4(pos_x_bytes);
  received_packet.position_x = *reinterpret_cast<float *>(pos_x_bytes);
  offset += POSX_SIZE;

  // Unpack quaternion (16 bytes, big-endian)
  received_packet.quaternion.resize(4);
  for (int i = 0; i < 4; i++) {
    unsigned char quat_bytes[4];
    std::memcpy(quat_bytes, packet + offset, sizeof(quat_bytes));
    swap_endian_4(quat_bytes);
    received_packet.quaternion[i] = *reinterpret_cast<float *>(quat_bytes);
    offset += sizeof(QUAT_SIZE / 4);
  }

  return received_packet;
}

inline void htonf_noswap(float value, unsigned char *buffer) {
  floatToBytes ftb;
  ftb.floatValue = value;
  std::memcpy(buffer, ftb.bytes, 4);
}

std::vector<unsigned char> pack_reply(float* velocity_cmd) {
  std::vector<unsigned char> reply(VEL_CMD_SIZE); // 12 bytes

  for (int i = 0; i < 3; i++) {
    htonf_noswap(velocity_cmd[i], reply.data() + i * 4);
  }

  return reply;
}

float frobenius_norm(const std::array<float, 3> &vec) {
  return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

template <typename T> T clip(T value, T lower_bound, T upper_bound) {
  return std::min(std::max(value, lower_bound), upper_bound);
}


std::array<float, 3>
calculate_final_velocity(float* raw_output,
                         float desired_vel, float pos_x) {
  // 1. Create a copy of the input vector to avoid modifying the original
  std::array<float, 3> final_velocity;
  for (size_t i = 0; i < final_velocity.size(); ++i) {
      final_velocity[i] = raw_output[i];
  }

  // 2. Clip the first component and normalize the vector
  final_velocity[0] = clip(final_velocity[0], -1.0f, 1.0f);

  float norm = frobenius_norm(final_velocity);
  if (norm > 0.0f) {
    final_velocity[0] /= norm;
    final_velocity[1] /= norm;
    final_velocity[2] /= norm;
  }

  // 3. Scale the normalized vector by the desired velocity
  final_velocity[0] *= desired_vel;
  final_velocity[1] *= desired_vel;
  final_velocity[2] *= desired_vel;

  // 4. Apply a hardcoded threshold to the x-velocity component
  // We use constants for clarity instead of magic numbers
  constexpr float MIN_X_VELOCITY_CMD = 1.0f;
  constexpr float CONTROL_THRESHOLD = 2.0f;

  if (pos_x < CONTROL_THRESHOLD) {
    final_velocity[0] =
        std::max(MIN_X_VELOCITY_CMD, (pos_x / CONTROL_THRESHOLD) * desired_vel);
  }

  // 5. Return the result by value
  return final_velocity;
}

iree_status_t convert_f16_view_to_f32_view(iree_hal_device_t* device, iree_hal_buffer_view_t* f16_view, iree_hal_buffer_view_t** out_f32_view) {
    iree_hal_buffer_mapping_t mapped_f16;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(f16_view), IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &mapped_f16));
    iree_host_size_t element_count = iree_hal_buffer_view_element_count(f16_view);
    std::vector<float> f32_data(element_count);
    const half_float_t* f16_ptr = (const half_float_t*)mapped_f16.contents.data;
    for (iree_host_size_t i = 0; i < element_count; ++i) { f32_data[i] = half_to_float32(f16_ptr[i]); }
    iree_hal_buffer_unmap_range(&mapped_f16);
    return create_tensor_view(
        device, f32_data.data(), iree_hal_buffer_view_shape_dims(f16_view),
        iree_hal_buffer_view_shape_rank(f16_view), IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        out_f32_view);
}