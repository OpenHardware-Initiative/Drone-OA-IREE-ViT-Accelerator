# Framework for `Drone` `O`bstacle `A`voidance using `IREE` for a `Vi`sual `T`ransformer Hardware `Accelerator` platform

## Link to our video submission for the AMD Open Hardware Competition 2025

[![Embedded YouTube Video](https://img.youtube.com/vi/RXjw670piBA/0.jpg)](https://www.youtube.com/watch?v=RXjw670piBA)

## Project report

We recommend you to have a look at our written report to know about the structure of the project.

### ➡️ Report PDF: 
> [docs/Project Report_ AI Compilers Meet FPGAs_ A HW_SW Co-Design Approach for Vision Transformers.pdf](https://github.com/OpenHardware-Initiative/Drone-OA-IREE-ViT-Accelerator/blob/main/docs/Project%20Report_%20AI%20Compilers%20Meet%20FPGAs_%20A%20HW_SW%20Co-Design%20Approach%20for%20Vision%20Transformers.pdf)

## [OpenHardware Initiative](https://open-hardware-initiative.com/)

If you are interested in collaborating or joining our student initiative at tum please reach out to [team@open-hardware-initiative.com](team@open-hardware-initiative.com) or visit our website [open-hardware-initiative.com](https://open-hardware-initiative.com/)

# Drone-ViT-HW-Accelerator

This repository contains the complete toolchain, hardware designs, and software for accelerating a quantized Vision Transformer (ViT) on a Xilinx Kria KR260 SOM. The project integrates a custom hardware accelerator (ITA) with the IREE compiler framework to provide an end-to-end workflow from model training to hardware-accelerated inference.

## ✨ Key Features

- *End-to-End Workflow:* Covers the entire lifecycle: model training, Quantization-Aware Training (QAT), model compilation, hardware synthesis, and deployment.

- *Custom Hardware Acceleration:* Integrates the Integer Transformer Accelerator (ITA) for efficient 8-bit integer MHA operations.

- *Advanced Compilation:* Utilizes the IREE (Intermediate Representation Execution Environment) compiler to target the embedded ARM CPU and bridge to the custom hardware.

- *Cross-Compilation Toolchain:* Provides a Docker-based environment for compiling the model and runtime on an x86_64 host for the aarch64 Kria target.

## 📂 Repository Structure

```
Drone-ViT-HW-Accelerator/
├── docs/                      # All HOW-TO guides and documentation.
├── docker/                    # Dockerfile and resources for the cross-compilation environment.
├── models/                    # ML model definitions tailored for the ITA accelerator.
├── output/                    # Default location for compiled models (.mlir, .vmfb) and validation data.
├── training/                  # Scripts and configuration for model training and QAT.
├── tests/                     # Scripts for model validation and ONNX export.
├── third_party/               # Git submodules for external dependencies (IREE, ITA, etc.).
│   ├── ITA_FPGA/              # Integer Transformer Accelerator RTL and simulation environment.
│   ├── iree/                  # IREE compiler and runtime framework.
│   └── vitfly_FPGA/           # Base model architecture and dataset source.
└── kria-inference/            # Files to be deployed on the Kria board for inference.
```


## 🚀 Getting Started

To get started with this project, it is highly recommended to follow the comprehensive workflow guide, which details every step from environment setup to final deployment.

### ➡️ Full Project Guide: 
> [docs/HOW-TO-run-the-full-project-workflow.md](https://github.com/OpenHardware-Initiative/Drone-OA-IREE-ViT-Accelerator/blob/main/docs/HOW-TO-run-the-full-project-workflow.md)

#### Prerequisites:

Ensure you have the following software installed on your host machine:

- Git 
- Git LFS
- Conda / Miniconda
- Docker

## 📚 Documentation and HOW-TO Guides

All documentation is located in the [`/docs`](https://github.com/OpenHardware-Initiative/Drone-OA-IREE-ViT-Accelerator/tree/main/docs) directory. These guides provide detailed instructions for specific parts of the workflow.

### Core Workflow

- `Full Project Workflow:` The main step-by-step guide to replicate the project.

- `Cross-Compile for Kria:` Details on using the Docker environment to build IREE and the application for the Kria target.

### Hardware (FPGA)

- `Setup and Simulate ITA with Vivado:` How to set up the Vivado project for the ITA hardware accelerator.

- `Setup AXI DMA on Kria KR260:` Guide for creating the Vivado hardware design and kernel module for DMA.

### Kria Board Setup

- `Initial Kria KR260 Setup:` First steps for connecting to and configuring the Kria board.

- `Test Host-FPGA Communication:` A simple guide for testing the UDP link between the host PC and the FPGA.

### Development & Debugging

- `Compile ONNX MLIR Model:` Commands and flags for compiling the MLIR model with IREE.

- `Debug with GDB:` A small guide on how to debug C++ applications using GDB.

- `Avoid Git Problems on Kria:` Using deploy keys for easier Git operations on the board.

- `Import Python Packages from a Folder:` How to handle local Python imports within the project.

### Legacy/Alternative Hardware

- `Connect to ZedBoard:` How to establish a serial connection to a ZedBoard.

- `Build PyTorch for PYNQ:` Cross-compiling PyTorch for a PYNQ-Z2 board.
