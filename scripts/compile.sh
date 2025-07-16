#!/bin/zsh

# python attention.py 

# iree-import-onnx \
#   attention.onnx \
#   --opset-version 17 \
#   -o attention.mlir

# iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-linux-gnu \
#     --iree-llvmcpu-target-cpu=cortex-a53 --iree-flow-dump-dispatch-graph --dump-compilation-phases-to=./irs/ \
#     model.mlir -o model.vmfb --iree-opt-aggressively-propagate-transposes=true \
#     --iree-global-opt-propagate-transposes=true --iree-hal-executable-debug-level=3 --iree-opt-data-tiling=false \
#     --aarch64-use-aa --iree-dispatch-creation-enable-aggressive-fusion=true --iree-vm-target-truncate-unsupported-floats \
#     --iree-scheduling-dump-statistics-format=verbose --iree-scheduling-dump-statistics-file=compilation_info.txt \
#     --iree-llvmcpu-enable-ukernels=all  --iree-llvmcpu-target-cpu-features=+neon,+fp-armv8,+crypto \
#     --aarch64-neon-syntax=generic --iree-input-demote-f32-to-f16

clang -target aarch64  -std=c17  -ffreestanding \
    -fvisibility=hidden \
    -fno-plt \
    -fno-rtti \
    -fno-exceptions \
    -fdata-sections \
    -ffunction-sections \
    -funique-section-names \
    -c functions.c 

iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-apple-darwin \
    --iree-hal-executable-object-search-path=./ \
    --iree-preprocessing-transform-spec-filename=./example_transform_spec.mlir \
    --iree-flow-dump-dispatch-graph --dump-compilation-phases-to=./irs/ onnx_model.mlir -o ./onnx_model.vmfb 

iree-run-module \
    --device=local-sync \
    --function=main_graph \
    --input=1xf32=7.0 \
    --input=1xf32=4.5 \
    --module=./onnx_model.vmfb

# dot -Tpng dispatch.dot -o dispatch.png