# How to setup and run the project on Vivado

## Step 1) Clone the required repository from ITA-FPGA

From the ITA-FPGA repository root, go to the third_party folder and clone the needed repository:

cd /path/to/ITA-FPGA
cd third_party
git clone <REPO_URL>

## Step 2) Generate data files

Run testGenerator.py with your desired configuration for the dataset.

## Example for generating test vectors:
python testGenerator.py -H 1 -S 64 -E 128 -P 192 -F 256 --no-bias --activation=identity

## Step 3) Non-default configs: update Vivado macros

If you’re using non-default configurations, edit setup_vivado.tcl so the defines/macros (H, S, E, P, F, activation, bias) match your chosen parameters.

## Step 4) Refresh file dependencies
make bender

## Step 5) Create the Vivado project

Run the setup script (from the repo root or adjust the path accordingly):

vivado -mode batch -source setup_vivado.tcl

This will create the full Vivado project for you.

## Step 6) Simulate

In Vivado, select the testbench you want to simulate and make sure the data file paths point to the vectors generated in step 2.

## Step 7) Synthesis (Out-of-Context)

If you want to run synthesis, don’t forget to set out-of-context mode:

synth_design -top <TOP_MODULE> -part <FPGA_PART> -mode out_of_context
