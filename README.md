# Drone-ViT-HW-Accelerator

## Setup

1. Ensure Conda is installed.
2. Clone Repo
3. Setup Conda Environment
    ```bash
    conda env create -f environment.yml
    conda activate mobileSAM
    ```
4. Update submodules
    ```bash
    git submodule update --init --recursive
    ```
5. Build IREE (for CPU)
    ```bash
    mkdir build
    cmake \
        -G Ninja \
        -B build/ \
        -S third_party/iree \
        -DCMAKE_INSTALL_PREFIX=./build/ \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_BUILD_PYTHON_BINDINGS=ON \
        -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
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
    cmake --build build/
    ```

### Optional:

- Update an exisitng environment
    ```bash
    conda env update -f environment.yml --prune
    ```
- Enable CUDA if GPUs available: (you will have to remove the last 5 compiler flags and replace them with the ones bellow)
    ```bash
    -DIREE_TARGET_BACKEND_CUDA=ON \
    -DIREE_HAL_DRIVER_CUDA=ON
    ```

### Debug:
- If you encounter problems regarding dependencies have a look at the prerequisites for building IREE [HERE](https://iree.dev/building-from-source/getting-started/#prerequisites).
- We are building using the `RelWithDebInfo` build type. If you want to have full LLVM debug functionality use the `Debug` build type. **(Dont forget to build on release mode for submission)**
- If you are having problems with the current conda env, consider installing the build requirements provided by IREE on a new venv.
    ```bash
    conda create -n iree python=3.10
    conda activate iree

    # Upgrade PIP before installing other requirements
    pip install --upgrade pip

    # Install IREE build requirements
    pip install -r third_party/iree/runtime/bindings/python/iree/runtime/build_requirements.txt

## How to profile the model?

So far ONNX and Pytorch havent been that useful, this could be due to the specificity on the naming of each layer. I will try with the IREE profiler and see if it is more informative.

### Known problems:
- For quantizing the model we need onnxruntime 1.16.0, but for doing profiling we need the latest version which is 1.19.2. So far I have only manage to go around it by executing profiling and quantize cells separately. LESS THAN IDEAL. I will do separate scripts for this.
    ```bash
    pip install onnxruntime==1.16.0
    ```
    vs.
    ```bash
    pip install --upgrade onnxruntime
    ```