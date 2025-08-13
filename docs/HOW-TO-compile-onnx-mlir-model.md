# How do we compile our model for the arm target?

## Our ONNX model with dummy layers and NO QAT:

In this case we use this command:

```bash
iree-compile output/ita_model_for_hardware.mlir -o output/ITAViTLSTM_f16.vmfb \
--iree-hal-target-backends=llvm-cpu \
--iree-llvmcpu-target-triple=aarch64-linux-gnu \
--iree-llvmcpu-target-cpu=cortex-a53  \
--iree-opt-aggressively-propagate-transposes=true \
--iree-global-opt-propagate-transposes=true \
--iree-hal-executable-debug-level=3 \
--iree-opt-data-tiling=true \
--aarch64-use-aa \
--iree-dispatch-creation-enable-aggressive-fusion=false \
--iree-vm-target-truncate-unsupported-floats \
--iree-scheduling-dump-statistics-format=verbose \
--iree-scheduling-dump-statistics-file=output/compilation_info.txt \
--iree-llvmcpu-enable-ukernels=all  \
--iree-llvmcpu-target-cpu-features=+neon,+fp-armv8,+crypto \
--aarch64-neon-syntax=generic \
--iree-input-demote-f32-to-f16 \
--dump-compilation-phases-to=compilation_phases 
```

Interesting about this is that it wouldnt work with `--iree-dispatch-creation-enable-aggressive-fusion=true`.

The error would be:
```bash
error: One or more operations with large vector sizes (8192 bytes)
```