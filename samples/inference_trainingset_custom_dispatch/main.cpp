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

#include <cstdint>
#include <cstring>

// --- IREE C API Headers ---
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/base/allocator.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"

// --- Image Loading Library ---
#include "stb_image.h"
#include "stb_image_resize2.h"



// --- Data Structures ---
struct TelemetryData {
    float desired_velocity;
    float quaternion[4];
    float ground_truth_velocity[3]; // x, y, z
};

typedef uint16_t half_float_t;

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

// --- Forward Declarations & Helpers ---
extern "C" iree_status_t iree_hal_local_sync_driver_module_register(iree_hal_driver_registry_t* registry);
extern "C" iree_status_t iree_allocator_libc_ctl(void* self, iree_allocator_command_t command, const void* params, void** inout_ptr);
iree_status_t create_tensor_view(iree_hal_device_t* device, const void* data, const iree_hal_dim_t* shape, iree_host_size_t shape_rank, iree_hal_element_type_t element_type, iree_hal_buffer_view_t** out_buffer_view);
void print_output_tensor(iree_hal_buffer_view_t* view, const TelemetryData& telemetry);
bool load_telemetry_for_image(const std::filesystem::path& csv_path, const std::string& image_timestamp_str, TelemetryData& out_telemetry);
iree_status_t convert_f16_view_to_f32_view(iree_hal_device_t* device, iree_hal_buffer_view_t* f16_view, iree_hal_buffer_view_t** out_f32_view);

// --- Main Application ---
int main(int argc, char** argv) {
    // The model path is now handled by the build system.
    // We only need the data folder path from the command line.
    if (argc != 2) { 
        std::cerr << "Usage: " << argv[0] << " /path/to/root_data_folder" << std::endl; 
        return 1; 
    }
    std::filesystem::path vmfb_path(MODEL_VMFB_PATH); // Use the macro here
    std::filesystem::path root_data_dir(argv[1]);     // Get data path from args
    
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);
    iree_runtime_instance_t* instance = NULL;
    iree_allocator_t host_allocator = { .self = NULL, .ctl = iree_allocator_libc_ctl };
    IREE_CHECK_OK(iree_runtime_instance_create(&instance_options, host_allocator, &instance));
    iree_hal_device_t* device = NULL;
    IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(instance, iree_make_cstring_view("local-sync"), &device));
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    iree_runtime_session_t* session = NULL;
    IREE_CHECK_OK(iree_runtime_session_create_with_device(instance, &session_options, device, iree_runtime_instance_host_allocator(instance), &session));
    std::cout << "Loading model: " << vmfb_path << std::endl;
    IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(session, vmfb_path.c_str()));
    std::vector<std::filesystem::path> trajectory_paths;
    for (const auto& entry : std::filesystem::directory_iterator(root_data_dir)) { if (entry.is_directory()) { trajectory_paths.push_back(entry.path()); } }
    std::sort(trajectory_paths.begin(), trajectory_paths.end());
    std::cout << "Found " << trajectory_paths.size() << " trajectories to process." << std::endl;

    for (const auto& traj_path : trajectory_paths) {
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Processing Trajectory: " << traj_path.filename() << std::endl;
        std::cout << "==================================================" << std::endl;

        const iree_hal_dim_t hidden_shape[] = {3, 1, 128};
        std::vector<char> zero_buffer(3 * 1 * 128 * sizeof(float), 0);
        iree_hal_buffer_view_t* hidden_state_h = NULL;
        iree_hal_buffer_view_t* hidden_state_c = NULL;
        IREE_CHECK_OK(create_tensor_view(device, zero_buffer.data(), hidden_shape, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &hidden_state_h));
        IREE_CHECK_OK(create_tensor_view(device, zero_buffer.data(), hidden_shape, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &hidden_state_c));
        
        std::vector<std::filesystem::path> image_paths;
        std::filesystem::path csv_path = traj_path / "data.csv";
        if (!std::filesystem::exists(csv_path)) { std::cerr << "Warning: data.csv not found in " << traj_path << ". Skipping trajectory." << std::endl; continue; }
        for (const auto& file_entry : std::filesystem::directory_iterator(traj_path)) { if (file_entry.path().extension() == ".png") { image_paths.push_back(file_entry.path()); } }
        std::sort(image_paths.begin(), image_paths.end());

        for (const auto& image_path : image_paths) {
            std::cout << "\n--- Processing Image: " << image_path.filename() << " ---" << std::endl;

            int width, height, channels;
            unsigned char *image_data_uint8 = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
            if (!image_data_uint8) { std::cerr << "Warning: Failed to load image " << image_path << ", skipping." << std::endl; continue; }
            const int target_width = 90;
            const int target_height = 60;
            std::vector<float> image_f32(target_width * target_height);
            if (width != target_width || height != target_height) {
                std::vector<unsigned char> resized_data(target_width * target_height);
                stbir_resize_uint8_linear(image_data_uint8, width, height, 0, resized_data.data(), target_width, target_height, 0, STBIR_1CHANNEL);
                for (int i = 0; i < target_width * target_height; ++i) { image_f32[i] = static_cast<float>(resized_data[i]) / 255.0f; }
            } else {
                for (int i = 0; i < target_width * target_height; ++i) { image_f32[i] = static_cast<float>(image_data_uint8[i]) / 255.0f; }
            }
            stbi_image_free(image_data_uint8);

            TelemetryData telemetry;
            std::string image_timestamp = image_path.stem().string();
            if (!load_telemetry_for_image(csv_path, image_timestamp, telemetry)) {
                std::cerr << "Warning: Could not find telemetry for timestamp " << image_timestamp << ", using default." << std::endl;
                telemetry.desired_velocity = 0.0f;
                telemetry.quaternion[0] = 1.0f; telemetry.quaternion[1] = 0.0f;
                telemetry.quaternion[2] = 0.0f; telemetry.quaternion[3] = 0.0f;
                telemetry.ground_truth_velocity[0] = 0.0f; telemetry.ground_truth_velocity[1] = 0.0f; telemetry.ground_truth_velocity[2] = 0.0f;
            }

            iree_runtime_call_t call;
            const char* func_name = "module.main_graph";
            IREE_CHECK_OK(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(func_name), &call));

            float scaled_velocity = telemetry.desired_velocity / 10.0f;

            const iree_hal_dim_t img_shape[] = {1, 1, 60, 90};
            iree_hal_buffer_view_t* img_view = NULL;
            IREE_CHECK_OK(create_tensor_view(device, image_f32.data(), img_shape, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &img_view));
            
            const iree_hal_dim_t vel_shape[] = {1, 1};
            iree_hal_buffer_view_t* vel_view = NULL;
            IREE_CHECK_OK(create_tensor_view(device, &scaled_velocity, vel_shape, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &vel_view));
            
            const iree_hal_dim_t quat_shape[] = {1, 4};
            iree_hal_buffer_view_t* quat_view = NULL;
            IREE_CHECK_OK(create_tensor_view(device, telemetry.quaternion, quat_shape, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &quat_view));

            IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, img_view));
            IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, vel_view));
            IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, quat_view));
            IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, hidden_state_h));
            IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, hidden_state_c));

            IREE_CHECK_OK(iree_runtime_call_invoke(&call, 0));
            iree_hal_buffer_view_t* raw_output_view = NULL;
            iree_hal_buffer_view_t* new_hidden_state_h_f16 = NULL;
            iree_hal_buffer_view_t* new_hidden_state_c_f16 = NULL;
            IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &raw_output_view));
            IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &new_hidden_state_h_f16));
            IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &new_hidden_state_c_f16));
            
            print_output_tensor(raw_output_view, telemetry);

            iree_hal_buffer_view_release(hidden_state_h);
            iree_hal_buffer_view_release(hidden_state_c);

            IREE_CHECK_OK(convert_f16_view_to_f32_view(device, new_hidden_state_h_f16, &hidden_state_h));
            IREE_CHECK_OK(convert_f16_view_to_f32_view(device, new_hidden_state_c_f16, &hidden_state_c));

            iree_hal_buffer_view_release(new_hidden_state_h_f16);
            iree_hal_buffer_view_release(new_hidden_state_c_f16);

            iree_hal_buffer_view_release(img_view);
            iree_hal_buffer_view_release(vel_view);
            iree_hal_buffer_view_release(quat_view);
            iree_hal_buffer_view_release(raw_output_view);
            iree_runtime_call_deinitialize(&call);
        }
        
        iree_hal_buffer_view_release(hidden_state_h);
        iree_hal_buffer_view_release(hidden_state_c);
    }

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

bool load_telemetry_for_image(const std::filesystem::path& csv_path, const std::string& image_timestamp_str, TelemetryData& out_telemetry) {
    std::ifstream file(csv_path);
    if (!file.is_open()) return false;
    const double epsilon = 0.001; 
    double image_timestamp = std::stod(image_timestamp_str);
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) { row.push_back(cell); }
        if (row.size() > 12) { // Ensure there are enough columns for velocity
            try {
                double csv_timestamp = std::stod(row[1]);
                if (std::abs(csv_timestamp - image_timestamp) < epsilon) {
                    out_telemetry.desired_velocity = std::stof(row[2]);
                    // Reverted: The CSV shows quat_1 is w, so we read in w,x,y,z order.
                    out_telemetry.quaternion[0] = std::stof(row[3]); // w
                    out_telemetry.quaternion[1] = std::stof(row[4]); // x
                    out_telemetry.quaternion[2] = std::stof(row[5]); // y
                    out_telemetry.quaternion[3] = std::stof(row[6]); // z
                    
                    // NEW: Load ground truth velocity
                    out_telemetry.ground_truth_velocity[0] = std::stof(row[10]); // vel_x
                    out_telemetry.ground_truth_velocity[1] = std::stof(row[11]); // vel_y
                    out_telemetry.ground_truth_velocity[2] = std::stof(row[12]); // vel_z
                    return true;
                }
            } catch (const std::invalid_argument&) {}
        }
    }
    return false;
}

iree_status_t create_tensor_view(iree_hal_device_t* device, const void* data, const iree_hal_dim_t* shape, iree_host_size_t shape_rank, iree_hal_element_type_t element_type, iree_hal_buffer_view_t** out_buffer_view) {
    iree_hal_buffer_params_t buffer_params = {0};
    buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    iree_host_size_t byte_length = get_element_byte_size(element_type);
    for(int i = 0; i < shape_rank; ++i) { byte_length *= shape[i]; }
    return iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), shape_rank, shape, element_type, 
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params, 
        iree_make_const_byte_span(data, byte_length), out_buffer_view);
}

void print_output_tensor(iree_hal_buffer_view_t* view, const TelemetryData& telemetry) {
    if (!view) { std::cout << "  <null>" << std::endl; return; }
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
    
    std::cout << "  Ground Truth Vel: [" << telemetry.ground_truth_velocity[0] << ", " << telemetry.ground_truth_velocity[1] << ", " << telemetry.ground_truth_velocity[2] << "]" << std::endl;

    if (model_output.size() == 3) {
        float dx = model_output[0] - telemetry.ground_truth_velocity[0];
        float dy = model_output[1] - telemetry.ground_truth_velocity[1];
        float dz = model_output[2] - telemetry.ground_truth_velocity[2];
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        std::cout << "  Error Distance:  " << distance << std::endl;
    }

    iree_hal_buffer_unmap_range(&mapped_memory);
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