# Drone-ViT-HW-Accelerator

This repository provides the toolchain, scripts, and documentation for compiling and accelerating a Vision Transformer (ViT) model for deployment on a Xilinx Kria SOM. It utilizes the IREE (Intermediate Representation Execution Environment) compiler framework to target the embedded ARM architecture from an x86_64 host machine.

The primary workflow involves two main stages:
1.  **Host Build**: Building the IREE compiler and tools on your local x86_64 machine.
2.  **Cross-Compilation**: Using the host tools within a Docker container to compile the model for the aarch64 Kria target.

---

## 📋 Prerequisites

Before you begin, ensure you have the following software installed on your host machine:

* [Git](https://git-scm.com/downloads)
* [Git LFS](https://git-lfs.github.com/)
* [Conda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* [Docker](https://docs.docker.com/get-docker/)

---

## ⚙️ Initial Project Setup

These steps only need to be performed once to set up the project environment.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd Drone-ViT-HW-Accelerator
    ```

2.  **Pull LFS Files and Submodules**
    ```bash
    git lfs pull
    git submodule update --init --recursive
    ```

3.  **Create and Activate Conda Environment**
    This environment is primarily for model preparation, quantization, and analysis on the host machine.
    ```bash
    conda env create -f IREE_environment.yml
    conda activate IREE
    ```
    *To update an existing environment, run `conda env update -f environment.yml --prune`.*

---

## 🎯 Primary Workflow: Cross-Compilation for Kria SOM

This workflow details the end-to-end process from building on the host to cross-compiling for the target.

### Step 1: Set Environment Variables

From the root of the repository, export the following variables. These paths are essential for both the host build and the cross-compilation steps.

```bash
# Set the root directory for the entire project
export WORKSPACE_DIR=${PWD}

# Directories for the x86_64 host build
export BUILD_HOST_DIR=${WORKSPACE_DIR}/build-host
export INSTALL_HOST_DIR=${BUILD_HOST_DIR}/install

# Directories for the aarch64 Kria cross-compilation build
export BUILD_KRIA_DIR=${WORKSPACE_DIR}/build-kria
export INSTALL_KRIA_DIR=${BUILD_KRIA_DIR}/install

# Verify paths
echo "Host Install Dir: ${INSTALL_HOST_DIR}"
echo "Kria Build Dir: ${BUILD_KRIA_DIR}"
```

### Step 2: Build Host Tools (IREE Compiler)

This command compiles the IREE compiler on your x86_64 host. These tools are **required** before proceeding to cross-compilation, as the Docker environment will use them. Note that Python bindings are disabled (`-DIREE_BUILD_PYTHON_BINDINGS=OFF`) as they are not needed for this stage.

```bash
cmake \
    -G Ninja \
    -B "${BUILD_HOST_DIR}" \
    -S third_party/iree \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_HOST_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF

# Build and install the host tools
cmake --build "${BUILD_HOST_DIR}" --target install
```

### Step 3: Set Up and Enter the Cross-Compilation Docker Environment

Now that the host tools are built, you can prepare the Docker environment for cross-compilation.

1.  **Navigate to the Docker Directory**
    ```bash
    cd docker
    ```

2.  **Download the Kria System Root (Sysroot)**
    The build process requires a sysroot containing the libraries and headers for the target Kria operating system.
    ```bash
    # Ensure you are in the `docker/` directory
    wget -c -O kria-sysroot.tar.xz "[https://people.canonical.com/~platform/images/xilinx/kria24-ubuntu-22.04/iot-limerick-kria-classic-server-2204-classic-22.04-kd05-20240223-170-sysroot.tar.xz](https://people.canonical.com/~platform/images/xilinx/kria24-ubuntu-22.04/iot-limerick-kria-classic-server-2204-classic-22.04-kd05-20240223-170-sysroot.tar.xz)"
    ```
    > For more information on official AMD/Xilinx LTS Ubuntu images and sysroots, visit the [Ubuntu downloads page for AMD](https://ubuntu.com/download/amd).

3.  **Build the Docker Image**
    This command builds the Docker image using the `Dockerfile` in this directory.
    ```bash
    docker build -t kria-cross-compiler .
    ```

4.  **Run the Docker Container**
    To start an interactive session, navigate back to the **root directory of the project** and run:
    ```bash
    # Make sure you are at the project's root level
    cd ..
    docker run -it --rm -v "$(pwd):/workspace" kria-cross-compiler
    ```
    * **What this does:** This command runs the container in interactive mode (`-it`), automatically removes it on exit (`--rm`), and critically, mounts your project directory on the host to `/workspace` inside the container (`-v "$(pwd):/workspace"`). This allows you to edit files on your host and have the changes immediately reflected inside the container for compilation.

### Step 4: Cross-Compile the Application (Inside Docker)

Once inside the container, you can proceed with building the target application. The environment variables from Step 1 are automatically set for you.

1.  **Configure the Kria Build**
    This `cmake` command configures the build to use the Kria toolchain and the host tools you built previously (`-DIREE_HOST_BIN_DIR=$INSTALL_HOST_DIR/bin`).

    ```bash
    cmake \
        -G Ninja \
        -B $BUILD_KRIA_DIR \
        -S third_party/iree \
        -DCMAKE_TOOLCHAIN_FILE=/opt/kria.toolchain.cmake \
        -DIREE_HOST_BIN_DIR=$INSTALL_HOST_DIR/bin \
        -DIREE_CMAKE_PLUGIN_PATHS=$PWD \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_KRIA_DIR \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_BUILD_COMPILER=OFF \
        -DIREE_BUILD_SAMPLES=ON \
        -DIREE_BUILD_TESTS=OFF \
        -DIREE_BUILD_PYTHON_BINDINGS=OFF \
        -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
        -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
        -DIREE_HAL_DRIVER_DEFAULTS=OFF \
        -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
    ```

2.  **Build the Target Application**
    Compile the specific application you need. For example:
    ```bash
    cmake --build ${BUILD_KRIA_DIR} --target ITAViTLSTM_test_data
    ```
    *Tip: To see a list of all available targets, run `cmake --build ${BUILD_KRIA_DIR} --target help`.*

### Step 5: Test with QEMU (Inside Docker)

You can test the cross-compiled binary within the Docker container using QEMU, which emulates the aarch64 architecture.

```bash
qemu-aarch64-static -L /opt/sysroot/sysroots/aarch64-xilinix-linux \
    ${BUILD_KRIA_DIR}/runtime/plugins/ita-samples/inference/ITAViTLSTM_test_data \
    /workspace/output/ITAViTLSTM_f16.vmfb \
    /workspace/training/small_data
```

---

## 💻 Local Development and Debugging (x86_64 Host)

This workflow is for developing, profiling, and debugging on your local machine *without* cross-compiling. It builds IREE with Python bindings enabled, allowing you to interact with the framework through scripts and notebooks.

### 1. Build IREE with Python Bindings for Local Use

This configuration is different from the host tools build. It enables Python bindings (`-DIREE_BUILD_PYTHON_BINDINGS=ON`) for local use.

```bash
# It is recommended to use a different build directory for local development
mkdir build-local
cd build-local

cmake \
    -G Ninja \
    -S ../third_party/iree \
    -DIREE_CMAKE_PLUGIN_PATHS=$PWD/.. \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON

# Build the targets
cmake --build .
```

**Optional: Enable CUDA Backend**
If you have a compatible NVIDIA GPU, replace the last 5 flags above with:
```bash
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_TARGET_BACKEND_CUDA=ON \
    -DIREE_HAL_DRIVER_CUDA=ON
```

### 2. Local Debugging of C++ Applications

To build a C++ application with full debug symbols, configure a separate build with `CMAKE_BUILD_TYPE=Debug`. This uses the pre-built host tools to avoid rebuilding the entire compiler.

```bash
# Configure debug build
cmake \
    -G Ninja \
    -B build-debug \
    -S third_party/iree \
    -DIREE_HOST_BIN_DIR=$INSTALL_HOST_DIR/bin \
    -DIREE_CMAKE_PLUGIN_PATHS=$PWD \
    -DCMAKE_BUILD_TYPE=Debug \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_SAMPLES=ON \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON

# Build the specific debug target
cmake --build build-debug --target inference_trainingset_custom_dispatch

# Execute the application
./build-debug/runtime/plugins/ita-samples/inference_trainingset_custom_dispatch/inference_trainingset_custom_dispatch training/data
```

---

## 📦 Model Management

### Profiling

Initial analysis suggests that standard PyTorch and ONNX profilers may not provide sufficient layer-specific detail. The recommended approach is to use the IREE profiling tools for more informative, hardware-aware performance metrics.

### Quantization and ONNX Runtime Versioning

There is a known dependency conflict:
* **Model Quantization**: Requires `onnxruntime==1.16.0`.
* **Model Profiling**: Benefits from the latest `onnxruntime` version (e.g., `1.19.x`).

**Recommendation**: Use separate, dedicated scripts or notebooks for these tasks. Before running a quantization script, execute `pip install onnxruntime==1.16.0`. Before profiling, run `pip install --upgrade onnxruntime`. A more robust long-term solution is to maintain two separate Conda environments.

### Visualization

You can inspect the model architecture using two methods:
1.  **SVG**: Use the **SVG Preview** extension in VS Code to view `.svg` files generated by the model conversion process (see the `/media` directory).
2.  **ONNX**: Use the **ONNX Viewer** extension in VS Code to view the `.onnx` model files directly (see the `/models` directory).

---

## 🧐 Troubleshooting

* **Build Dependencies**: If you encounter issues related to missing dependencies during the IREE build, consult the official [IREE prerequisites guide](https://iree.dev/building-from-source/getting-started/#prerequisites).
* **Conda Environment Issues**: If the `environment.yml` file causes conflicts, consider creating a clean environment (`conda create -n iree-dev python=3.10`) and manually installing the build requirements: `pip install -r third_party/iree/runtime/bindings/python/iree/runtime/build_requirements.txt`.
* **Build Types**: The default build type is `RelWithDebInfo` (Release with Debug Information). For production or performance testing, use `-DCMAKE_BUILD_TYPE=Release`. For full debugging capabilities with LLDB/GDB, use `-DCMAKE_BUILD_TYPE=Debug`.
* **Finding Targets**: To see a list of all available applications to build, run `cmake --build <your-build-dir> --target help`.

---

## 🗒️ Developer TODOs

### Documentation Improvements
* [ ] Automate environment variable setup via a shell script (`source env_setup.sh`).

### Code & Model Improvements
* [ ] Refactor Python scripts to use relative paths starting from a defined `PROJECT_ROOT`.
* [ ] Investigate and remove unnecessary dimensions in the model's inference graph to optimize performance.
