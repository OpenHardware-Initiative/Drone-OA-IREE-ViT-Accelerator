# Drone-ViT-HW-Accelerator

## Setup

1. Ensure Conda is installed.
2. Clone Repo
3. Call:
   ```bash
   git pull
   git lfs pull
    ```
4. Setup Conda Environment
    ```bash
    conda env create -f environment.yml
    conda activate mobileSAM
    ```
5. Update submodules
    ```bash
    git submodule update --init --recursive
    ```
6. Build IREE (for CPU)
    ```bash
    mkdir build
    cmake \
        -G Ninja \
        -B $HOST_BUILD_DIR \
        -S third_party/iree \
        -DIREE_CMAKE_PLUGIN_PATHS=$PWD \
        -DCMAKE_INSTALL_PREFIX=$HOST_INSTALL_DIR \
        -DCMAKE_INSTALL_PREFIX=./build-host/install \
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

The noteebok `onnx_model_example.ipynb` is supposed to be replace the original version found in `third_party/MobileSAM/notebooks/`.

### Known problems:
- For quantizing the model we need onnxruntime 1.16.0, but for doing profiling we need the latest version which is 1.19.2. So far I have only manage to go around it by executing profiling and quantize cells separately. LESS THAN IDEAL. I will do separate scripts for this.
    ```bash
    pip install onnxruntime==1.16.0
    ```
    vs.
    ```bash
    pip install --upgrade onnxruntime
    ```

## Visualizing the model

You can either download a SVG visualization extension for VSCode called `Svg Preview` (look inside `/media`) or the NN visualization extension from ONNX called `ONNX Viewer` (look inside `/models`).


# TODO for documentation:

[ ] Include global variables for things like conda env or other similar things so that we can just call those in the documentation
[ ] Remove all absolute paths (in python scripts we can always build starting from `$PROJECT_ROOT`)

# TODO for improvements:

[ ] Remove unnecessary dimension in inference.


# Build host:

1. Set macros (from repo level):
```bash
export BUILD_HOST_DIR=build-host
export INSTALL_HOST_DIR=build-host/install
```

```bash
cmake \
        -G Ninja \
        -B $HOST_BUILD_DIR \
        -S third_party/iree \
        -DCMAKE_INSTALL_PREFIX=$HOST_INSTALL_DIR \
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
```

```bash
cmake --build "${HOST_BUILD_DIR}" --target install
```


For the kria docker commands:

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



UPDATE:

cmake     -G Ninja     -B $BUILD_KRIA_DIR     -S third_party/iree     -DCMAKE_TOOLCHAIN_FILE=/opt/kria.toolchain.cmake     -DIREE_HOST_BIN_DIR=$INSTALL_HOST_DIR/bin     -DIREE_CMAKE_PLUGIN_PATHS=$PWD     -DCMAKE_INSTALL_PREFIX=$INSTALL_KRIA_DIR     -DCMAKE_BUILD_TYPE=RelWithDebInfo     -DIREE_BUILD_COMPILER=OFF     -DIREE_BUILD_SAMPLES=ON     -DIREE_BUILD_TESTS=OFF     -DIREE_BUILD_PYTHON_BINDINGS=OFF     -DIREE_TARGET_BACKEND_DEFAULTS=OFF     -DIREE_TARGET_BACKEND_LLVM_CPU=ON     -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON

cmake --build ${BUILD_KRIA_DIR} --target ITAViTLSTM_test_data

qemu-aarch64-static -L /opt/sysroot/sysroots/aarch64-xilinix-linux \
    /workspace/build-kria/runtime/plugins/ita-samples/inference/ITAViTLSTM_test_data \
    /workspace/output/ITAViTLSTM_f16.vmfb \
    /workspace/training/small_data