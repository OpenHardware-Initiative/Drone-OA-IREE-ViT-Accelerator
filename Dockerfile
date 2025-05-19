FROM ros:noetic

# Install basic dependencies and Python 3.8
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    python3-catkin-tools \
    python3-empy \
    ros-noetic-catkin \
    ros-noetic-roscpp \
    ros-noetic-genmsg \
    ros-noetic-genpy \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-geometry-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-std-srvs \
    ros-noetic-visualization-msgs \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-tf \
    ros-noetic-rviz \
    ros-noetic-rqt-gui \
    ros-noetic-rqt-gui-py \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-msgs \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-rospy \
    ros-noetic-roslaunch \
    ros-noetic-common-msgs \
    ros-noetic-rosbash \
    ros-noetic-rosboost-cfg \
    libyaml-cpp-dev \
    libeigen3-dev \
    libopencv-dev \
    libzmq3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libglvnd-dev \
    libxrender1 \
    libxext6 \
    libsm6 \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip and setuptools
RUN python3 -m pip install --upgrade "pip<24.1" "setuptools<60.0" "importlib-metadata==4.8.3"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3.8 -m pip install -r /tmp/requirements.txt

# Copy the setup script
COPY ./set_up.sh /root/set_up.sh
COPY ./catkin_ws/src/data/ /root/data/
RUN chmod +x /root/set_up.sh
RUN /root/set_up.sh

WORKDIR /root

# Keep the container running
CMD ["bash"]