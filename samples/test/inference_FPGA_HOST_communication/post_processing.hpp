#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "fpga_link.hpp" // For FpgaToHostPacket

// This is a direct C++ port of your Python `calculate_final_velocity` logic.
void calculate_final_velocity(
    const float* raw_output,
    float desired_vel,
    float pos_x,
    FpgaToHostPacket& reply)
{
    float vel_cmd[3];
    vel_cmd[0] = std::max(-1.0f, std::min(1.0f, raw_output[0]));
    vel_cmd[1] = raw_output[1];
    vel_cmd[2] = raw_output[2];

    float norm = std::sqrt(vel_cmd[0]*vel_cmd[0] + vel_cmd[1]*vel_cmd[1] + vel_cmd[2]*vel_cmd[2]);
    if (norm > 1e-6) { // Avoid division by zero
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