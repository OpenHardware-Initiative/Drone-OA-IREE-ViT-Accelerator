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

// --- IREE C API Headers ---
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/base/allocator.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"

// --- Image Loading Library ---
#include "include/stb_image.h"

// --- Data Structures ---
struct TelemetryData {
    float desired_velocity;
    float quaternion[4];
};

// --- Forward Declarations & Helpers ---

// FINAL FIX #1: Corrected function declaration with "_module".
extern "C" iree_status_t iree_hal_local_sync_driver_module_register(
    iree_hal_driver_registry_t* registry);

extern "C" iree_status_t iree_allocator_libc_ctl(
    void* self, iree_allocator_command_t command,
    const void* params, void** inout_ptr);

iree_status_t create_tensor_view(iree_hal_device_t* device, const void* data, const iree_hal_dim_t* shape, iree_host_size_t shape_rank, iree_hal_element_type_t element_type, iree_hal_buffer_view_t** out_buffer_view);
void print_output_tensor(iree_hal_buffer_view_t* view);
bool load_telemetry_for_image(const std::filesystem::path& csv_path, const std::string& image_timestamp_str, TelemetryData& out_telemetry);

// --- Main Application ---
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " /path/to/model.vmfb /path/to/root_data_folder" << std::endl;
        return 1;
    }
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

    std::vector<std::filesystem::path> trajectory_paths;
    for (const auto& entry : std::filesystem::directory_iterator(root_data_dir)) {
        if (entry.is_directory()) {
            trajectory_paths.push_back(entry.path());
        }
    }
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
        if (!std::filesystem::exists(csv_path)) {
            std::cerr << "Warning: data.csv not found in " << traj_path << ". Skipping trajectory." << std::endl;
            continue;
        }
        for (const auto& file_entry : std::filesystem::directory_iterator(traj_path)) {
            if (file_entry.path().extension() == ".png") {
                image_paths.push_back(file_entry.path());
            }
        }
        std::sort(image_paths.begin(), image_paths.end());

        for (const auto& image_path : image_paths) {
            int width, height, channels;
            unsigned char *image_data_uint8 = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
            if (!image_data_uint8) {
                std::cerr << "Warning: Failed to load image " << image_path << ", skipping." << std::endl;
                continue;
            }
            std::vector<float> image_f32(width * height);
            for (int i = 0; i < width * height; ++i) image_f32[i] = static_cast<float>(image_data_uint8[i]) / 255.0f;
            stbi_image_free(image_data_uint8);

            TelemetryData telemetry;
            std::string image_timestamp = image_path.stem().string();
            if (!load_telemetry_for_image(csv_path, image_timestamp, telemetry)) {
                std::cerr << "Warning: Could not find telemetry for timestamp " << image_timestamp << ", skipping image." << std::endl;
                continue;
            }
            std::cout << "\n--- Processing Image: " << image_path.filename() << " ---" << std::endl;

            iree_runtime_call_t call;
            const char* func_name = "module.main_graph";
            IREE_CHECK_OK(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(func_name), &call));

            const iree_hal_dim_t img_shape[] = {1, 1, 60, 90};
            iree_hal_buffer_view_t* img_view = NULL;
            IREE_CHECK_OK(create_tensor_view(device, image_f32.data(), img_shape, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &img_view));
            const iree_hal_dim_t vel_shape[] = {1, 1};
            iree_hal_buffer_view_t* vel_view = NULL;
            IREE_CHECK_OK(create_tensor_view(device, &telemetry.desired_velocity, vel_shape, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &vel_view));
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
            iree_hal_buffer_view_t* new_hidden_state_h = NULL;
            iree_hal_buffer_view_t* new_hidden_state_c = NULL;
            IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &raw_output_view));
            IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &new_hidden_state_h));
            IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &new_hidden_state_c));
            
            print_output_tensor(raw_output_view);

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
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        if (row.size() > 6) {
            try {
                double csv_timestamp = std::stod(row[1]);
                if (std::abs(csv_timestamp - image_timestamp) < epsilon) {
                    out_telemetry.desired_velocity = std::stof(row[2]);
                    out_telemetry.quaternion[0] = std::stof(row[3]);
                    out_telemetry.quaternion[1] = std::stof(row[4]);
                    out_telemetry.quaternion[2] = std::stof(row[5]);
                    out_telemetry.quaternion[3] = std::stof(row[6]);
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
    for(int i = 0; i < shape_rank; ++i) {
        byte_length *= shape[i];
    }

    return iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), shape_rank, shape, element_type, 
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params, 
        iree_make_const_byte_span(data, byte_length), out_buffer_view);
}

void print_output_tensor(iree_hal_buffer_view_t* view) {
    if (!view) {
        std::cout << "  <null>" << std::endl;
        return;
    }
    iree_hal_buffer_mapping_t mapped_memory;
    IREE_CHECK_OK(iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(view), 
        IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, 0, 
        iree_hal_buffer_view_byte_length(view), 
        &mapped_memory));

    const float* data_ptr = reinterpret_cast<const float*>(mapped_memory.contents.data);
    iree_host_size_t element_count = iree_hal_buffer_view_element_count(view);
    std::cout << "  Output Tensor (Shape: ";
    for (int i = 0; i < iree_hal_buffer_view_shape_rank(view); ++i) {
        std::cout << iree_hal_buffer_view_shape_dim(view, i) << (i < iree_hal_buffer_view_shape_rank(view) - 1 ? "x" : "");
    }
    std::cout << ", Type: f32):" << std::endl;
    std::cout << "  [";
    for (iree_host_size_t i = 0; i < element_count; ++i) {
        std::cout << data_ptr[i] << (i < element_count - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    iree_hal_buffer_unmap_range(&mapped_memory);
}