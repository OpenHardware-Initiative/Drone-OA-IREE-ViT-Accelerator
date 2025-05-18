#!/bin/bash

# Set up the catkin workspace
echo "Initializing catkin workspace"
cd ~
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color

# Clone the ViTFly repository
echo "Cloning ViTFly repository"
cd ~/catkin_ws/src
git clone https://github.com/anish-bhattacharya/ViT-for-quadrotor-obstacle-avoidance.git vitfly
cd vitfly

# Install the Conda environment
echo "Setting up Python environment"
conda env create -f ./flightmare/environment.yml python=3.8
source ~/.bashrc
conda activate flightmare

# Install additional Python dependencies
echo "Installing additional Python dependencies"
pip install \
    empy \
    catkin-tools \
    rospkg \
    pyyaml \
    uniplot \
    torchvision

# Unzip models and environments
echo "Extracting environment and model files"
tar -xvf /root/data/environments.tar -C flightmare/flightpy/configs/vision
tar -xvf /root/data/flightrender.tar -C flightmare/flightrender
tar -xvf /root/data/pretrained_models.tar -C models

# Set up Flightmare
echo "Setting up Flightmare"
bash setup_ros.bash
cd ../..

# Build the catkin workspace
echo "Building catkin workspace"
catkin build
source devel/setup.bash

# Return to the ViTFly directory
cd src/vitfly

echo "Setup complete!"