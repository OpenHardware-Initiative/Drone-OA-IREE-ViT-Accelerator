#!/usr/bin/env bash
set -euo pipefail
#set -x                        # ← trace every command
#shopt -s nullglob             # ← so *.onnx expands to zero files if none match

# adjust these if you move the script
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models/dummy"
MLIR_DIR="$MODEL_DIR/mlir"
VMBF_DIR="$MODEL_DIR/vmbf"

mkdir -p "$MLIR_DIR" "$VMBF_DIR"

echo "➡️  Starting ONNX→MLIR import..."
echo "  • Importing models from $MODEL_DIR"

# explicit check for matches
onnx_files=( "$MODEL_DIR"/*.onnx )
if (( ${#onnx_files[@]} == 0 )); then
  echo "⚠️  No .onnx files found in $MODEL_DIR"
  exit 1
fi

# DEBUG: inspect the array contents
echo "DEBUG: onnx_files[${#onnx_files[@]}]:"
for idx in "${!onnx_files[@]}"; do
  printf "  [%2d] -> %s\n" "$idx" "${onnx_files[$idx]}"
done
echo

total_onnx=0; succ_import=0; fail_import=0
for onnx in "${onnx_files[@]}"; do
  # DEBUG: we've entered the loop
  echo ">>> iteration $((total_onnx+1)): processing '$onnx'"
  #((total_onnx++))

  # the rest of your logic
  name="$(basename "$onnx" .onnx)"
  echo "  • [$name] importing…"
  out_mlir="$MLIR_DIR/$name.mlir"

  echo "  • [$name] importing…"
  echo "    ▶️  iree-import-onnx \"$onnx\" --opset-version 17 -o \"$out_mlir\""
  if
    iree-import-onnx "$onnx" --opset-version 17 -o "$out_mlir" \
        2> >(tee "$MODEL_DIR/${name}_import_error.txt" >&2); then
    echo "    ✅ OK"
    #((succ_import++))
  else
    echo "    ❌ FAIL or TIMEOUT (see ${name}_import_error.txt)"
    #((fail_import++))
  fi
done

echo
echo "  → Imported $succ_import / $total_onnx models, $fail_import failures."

echo
echo "➡️  Starting MLIR→VMFB compile..."
total_mlir=0; succ_comp=0; fail_comp=0
for mlir in "$MLIR_DIR"/*.mlir; do
  #((total_mlir++))
  name="$(basename "$mlir" .mlir)"
  out_vmfb="$VMBF_DIR/${name}_cpu.vmfb"
  echo -n "  • [$name] compiling… "
  if iree-compile \
       "$mlir" \
       --iree-hal-target-device=local \
       --iree-hal-local-target-device-backends=llvm-cpu \
       --iree-llvmcpu-target-cpu=host \
       -o "$out_vmfb"; then
    echo "OK"
   # ((succ_comp++))
  else
    echo "FAIL"
    #((fail_comp++))
    # rerun to capture stderr
    iree-compile \
       "$mlir" \
       --iree-hal-target-device=local \
       --iree-hal-local-target-device-backends=llvm-cpu \
       --iree-llvmcpu-target-cpu=host \
       -o /dev/null \
       2> "$MODEL_DIR/${name}_compile_error.txt" || true
  fi
done
echo "  → Compiled $succ_comp / $total_mlir MLIR files, $fail_comp failures."

# final exit code non-zero if any stage had failures
if (( fail_import + fail_comp > 0 )); then
  echo
  echo "⚠️  Completed with failures. See *_import_error.txt or *_compile_error.txt in $MODEL_DIR"
  exit 1
else
  echo
  echo "✅  All models processed successfully."
  exit 0
fi