// dummy_inference.hpp
#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "fpga_link.hpp" // We need the packet definition

// Define a struct to hold the results of our dummy inference step
struct DummyInferenceResult {
    std::vector<float> raw_output;
    std::vector<float> new_hidden_state;
};

// Replaces `initialize_model` from Python.
// Creates a fake "hidden state" of the correct size.
std::vector<float> initialize_dummy_state() {
    // Matches torch.zeros(3, 128)
    std::cout << "Dummy hidden state initialized (3x128)." << std::endl;
    return std::vector<float>(3 * 128, 0.0f);
}

// Replaces `run_inference_step` from Python.
// This is the core dummy function.
DummyInferenceResult run_dummy_inference_step(
    const std::vector<float>& old_hidden_state)
{
    // This function simulates the I/O of your model.
    // It takes the old state and produces a new state and a raw output.
    
    // 1. Create a fake raw output (3 floats for velocity command)
    std::vector<float> dummy_raw_output = {0.1f, 0.2f, -0.1f}; // Just some non-zero values

    // 2. Create a fake new hidden state. We can just copy the old one
    // or modify it slightly to prove the state is being passed through.
    std::vector<float> new_hidden_state = old_hidden_state;
    if (!new_hidden_state.empty()) {
        new_hidden_state[0] += 0.01f; // "Update" the state
    }

    return {dummy_raw_output, new_hidden_state};
}

// C++ version of `calculate_final_velocity` from your Python script.
// This is the actual post-processing logic.
void calculate_final_velocity(
    const std::vector<float>& raw_output,
    float desired_vel,
    float pos_x,
    FpgaToHostPacket& reply) // Output is written directly to the reply packet
{
    float vel_cmd[3];
    vel_cmd[0] = std::max(-1.0f, std::min(1.0f, raw_output[0])); // np.clip
    vel_cmd[1] = raw_output[1];
    vel_cmd[2] = raw_output[2];

    float norm = std::sqrt(vel_cmd[0]*vel_cmd[0] + vel_cmd[1]*vel_cmd[1] + vel_cmd[2]*vel_cmd[2]);
    if (norm > 0) {
        vel_cmd[0] /= norm;
        vel_cmd[1] /= norm;
        vel_cmd[2] /= norm;
    }

    reply.velocity_command[0] = vel_cmd[0] * desired_vel;
    reply.velocity_command[1] = vel_cmd[1] * desired_vel;
    reply.velocity_command[2] = vel_cmd[2] * desired_vel;

    const float min_xvel_cmd = 1.0f;
    const float hardcoded_ctl_threshold = 2.0f;
    if (pos_x < hardcoded_ctl_threshold) {
        reply.velocity_command[0] = std::max(min_xvel_cmd, (pos_x / hardcoded_ctl_threshold) * desired_vel);
    }
}