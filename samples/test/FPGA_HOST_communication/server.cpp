// server.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h> // For close()

#include "fpga_link.hpp"
#include "dummy_inference.hpp"

int main() {
    std::cout << "--- C++ Dummy Inference Server ---" << std::endl;

    // --- 1. Initialize the "Model" (dummy state) ---
    std::vector<float> hidden_state = initialize_dummy_state();

    // --- 2. Setup Network Server ---
    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        std::cerr << "Error: Could not create socket." << std::endl;
        return 1;
    }

    struct sockaddr_in server_addr, client_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY; // Listen on all available network interfaces
    server_addr.sin_port = htons(PORT);

    if (bind(sock_fd, (const struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error: Could not bind to port " << PORT << std::endl;
        close(sock_fd);
        return 1;
    }

    std::cout << "Network server is running. Listening for packets on port " << PORT << "..." << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "  Inference Time  |      Velocity Command (X, Y, Z)" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    // --- 3. Main Network and Inference Loop ---
    while (true) {
        try {
            std::vector<char> recv_buffer(65535); // Max UDP packet size
            socklen_t client_len = sizeof(client_addr);

            ssize_t n = recvfrom(sock_fd, recv_buffer.data(), recv_buffer.size(), 0,
                                 (struct sockaddr*)&client_addr, &client_len);

            if (n <= 0) continue;
            recv_buffer.resize(n);

            HostToFpgaPacket packet_in;
            if (!unpack_frame(recv_buffer, packet_in)) {
                std::cerr << "Warning: Received malformed packet of size " << n << std::endl;
                continue;
            }

            auto inference_start_time = std::chrono::high_resolution_clock::now();

            // This replaces the call to the real model
            DummyInferenceResult result = run_dummy_inference_step(hidden_state);
            hidden_state = result.new_hidden_state; // CRITICAL: Update the state

            auto inference_end_time = std::chrono::high_resolution_clock::now();
            
            FpgaToHostPacket packet_out;
            calculate_final_velocity(result.raw_output, packet_in.desired_velocity, packet_in.position_x, packet_out);

            std::vector<char> send_buffer = pack_reply(packet_out);
            sendto(sock_fd, send_buffer.data(), send_buffer.size(), 0,
                   (const struct sockaddr*)&client_addr, client_len);

            double inference_duration_ms = std::chrono::duration<double, std::milli>(inference_end_time - inference_start_time).count();
            float vx = packet_out.velocity_command[0];
            float vy = packet_out.velocity_command[1];
            float vz = packet_out.velocity_command[2];
            printf("   %7.2f ms   |   Vx: %6.2f, Vy: %6.2f, Vz: %6.2f\n", inference_duration_ms, vx, vy, vz);

        } catch (const std::exception& e) {
            std::cerr << "An error occurred in the server loop: " << e.what() << std::endl;
            // Re-initialize state on error, just like the Python server
            hidden_state = initialize_dummy_state();
            std::cerr << "Internal hidden state has been reset due to an error." << std::endl;
        }
    }

    close(sock_fd);
    return 0;
}