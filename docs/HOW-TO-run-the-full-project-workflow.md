# HOW TO run the full project workflow

This guide provides a comprehensive, step-by-step walkthrough of the entire project, from setting up the environment and training the model to synthesizing the hardware and deploying the accelerated application on the Kria SOM.

## PART 1: Model Training and Compilation (Host Machine)

### 1. Setup Conda Environment

First, set up the Conda environment which contains all the Python dependencies for model training, quantization, and analysis.

```bash
# From the project root
conda env create -f IREE_environment.yml
conda activate IREE
```

### 2. Initialize Submodules

Ensure all `third_party` repositories, including `vitfly_FPGA`, `ITA_FPGA`, `IREE` and `kria_inference` are cloned and updated.

```bash
git submodule update --init --recursive
```

### 3. Prepare the Dataset

The model is trained on the dataset from the original ViT-Fly project. Download and prepare it according to the instructions found here: `third_party/vitfly_FPGA/README.md`

### 4. Train the Baseline Model

Train the initial floating-point model. The training configuration is specified in the `.txt` file.

```bash
python3 -m third_party.vitfly_FPGA.training.train --config training/config/train.txt
```

You can find examples of our model architecture, designed to match the ITA accelerator, under the `/models` directory.

### 5. Run Quantization-Aware Training (QAT)

This is a crucial step, as the ITA hardware accelerator requires 8-bit quantized weights and inputs. QAT is necessary to maintain model accuracy with true 8-bit representations. Our custom QAT script handles the unique data type requirements of the ITA's custom softmax function (signed 8-bit in, unsigned 8-bit out).

Run the QAT script using its configuration file:
```bash
python3 -m training/qa_train.py --config training/config/qat.txt
```
### 6. Validate Model and Export Weights

After QAT, run the validation script. This script compares the PyTorch model's output with a simulation of the ITA's integer-based calculations to ensure correctness. It also exports the quantized weights, biases, and requantization parameters.

```bash
python3 -m tests.export_and_validation_W_B --checkpoint path/to/your/QAT/model.pth --image /path/to/test/image.jpg
```

This will generate several files, including `mem.txt`, which will be used to load the model's parameters into the hardware accelerator.

### 7. Export Model to ONNX for IREE

Next, convert the PyTorch model to an ONNX graph. This script replaces our custom PyTorch modules (like the special Softmax) with standard ONNX operations that IREE can understand.

```bash
python3 -m tests.export_onnx_for_FPGA
```

### 8. Convert ONNX to ONNX-MLIR

Finally, import the ONNX model into IREE's MLIR format. This is the last step before IREE's own compilation.

```bash
iree-import-onnx path/to/model.onnx --opset-version 17 -o path/to/output.mlir
```

## PART 2: IREE Compilation and Hardware Deployment

### 9. Build and Cross-Compile with IREE

Now, we switch to the IREE workflow to compile the MLIR model into a deployable binary for the Kria. This process involves building host tools and then using a Docker container for cross-compilation.

➡️ Follow the detailed guide here: 
> HOW-TO-cross-compile-ViT-model-for-Kria.md

After completing that guide, you will have a cross-compiled binary (e.g., `ITAViTLSTM_test_data`) ready for the Kria.

### 10. Prepare Files for Deployment

Move the generated binary and any other necessary files (like the compiled `.vmfb` model) into the `kria-inference` directory. This directory acts as a staging area for all files that need to be on the FPGA.

### 11. Synthesize the Hardware (Vivado)

Switching to the hardware side, generate the bitstream for the FPGA.

1. Navigate to third_party/ITA_FPGA. A Tcl script is provided to set up the entire Vivado project automatically.

2. Before generating the bitstream, ensure the project is set as Vitis-extensible and that you generate a binary (.bin) file along with the bitstream.

3. Run synthesis, implementation, and generate the bitstream.

➡️ For more details, see: 
> HOW-TO-setup-and-simulate-ITA-with-Vivado.md

### 12. Generate Hardware Platform Artifacts

After synthesis, export the hardware platform to generate an `.xsa` file. From this, you will create the Device Tree Overlay (`.dtbo`) file required by Linux.

➡️ For instructions on creating the `.xsa` and `.dtbo`, see: 
> HOW-TO-setup-AXI-DMA-on-Kria-KR260.md

Place the final `.bin`, `.dtbo`, and `shell.json` files into the `kria-inference `repository.

## PART 3: Final Deployment and Execution (Kria Board)

### 13. Prepare the Kria Board

1. Boot your Kria board with Ubuntu
2. Clone the `kria-inference` repo onto the board

➡️ For board setup, see: 
> HOW-TO-setup-Kria-KR260.md

### 14. Load the FPGA Firmware (Flash the PL)

Use the files you generated to load the hardware design onto the Programmable Logic (PL) side of the Kria.

➡️ See the "Deploy and Load the Hardware" section in: 
> HOW-TO-setup-AXI-DMA-on-Kria-KR260.md

### 15. Load the DMA Kernel Module

Load the custom Linux kernel module that allows the ARM processor to communicate with the hardware accelerator via AXI DMA.

### 16. Run the Final Application

1. Connect the Kria board to your host machine via an Ethernet cable.
2. On the host machine, run the simulation script from the vitfly repository to begin sending data.
```bash
# On Host PC
bash /path/to/vitfly/launch_evaluation_FPGA.bash 1 vision
```
3. Immediately after, execute the cross-compiled IREE binary on the FPGA. This will open the UDP communication channel and begin processing the data sent from the host, using custom dispatch calls to interact with the kernel module and accelerate computations on the PL.
```bash 
# On Kria Board
./path/to/binary
```

**Congratulations! You have completed the full workflow.**