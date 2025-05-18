FROM ros:noetic

# Install dependencies for Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    python3-catkin-tools \
    python3-empy \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-numpy \
    python3-yaml \
    python3-empy \
    python3-rospkg \
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
    libsm6

# Environment variables for Conda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    mkdir -p $CONDA_DIR && \
    bash /tmp/miniconda.sh -b -u -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda init bash && \
    conda clean -afy

# Copy the setup script
COPY ./set_up.sh /root/set_up.sh

COPY ./catkin_ws/src/data/ /root/data/

RUN chmod +x /root/set_up.sh

WORKDIR /root


# Keep the container running
CMD ["bash"]