#
# Dockerfile for a Comprehensive Kria Cross-Compilation Environment
#
# This multi-stage Dockerfile sets up a complete build environment on an
# x86_64 host for cross-compiling applications for the aarch64 Kria SOM.
# It uses a dedicated CMake toolchain file and a structured sysroot path.
#

# --- Stage 1: Base Setup and Core System Tools ---
    FROM ubuntu:22.04 AS base
    ENV DEBIAN_FRONTEND=noninteractive
    RUN  apt-get update && apt-get install -y --no-install-recommends \
          # Core build and system tools
          build-essential \
          git \
          ccache \
          wget \
          gnupg \
          ninja-build \
          graphviz \
          software-properties-common \
          pkg-config \
          # Libcurl development files for CMake build
          libcurl4-openssl-dev \
          # Python build dependencies
          zlib1g-dev \
          libncurses5-dev \
          libgdbm-dev \
          libnss3-dev \
          libssl-dev \
          libreadline-dev \
          libffi-dev \
          libsqlite3-dev \
          libbz2-dev \
          # Fortran and Linear Algebra (for NumPy, SciPy, PyTorch)
          gfortran \
          libopenblas-dev \
          liblapack-dev \
          # Image & Video I/O (for OpenCV and TorchVision)
          libjpeg-dev \
          libpng-dev \
          libtiff-dev \
          libavcodec-dev \
          libavformat-dev \
          libswscale-dev \
          libv4l-dev \
          # The cross-compiler toolchain for aarch64
          crossbuild-essential-arm64 && \
        rm -rf /var/lib/apt/lists/*
    
    # --- Stage 2: Install Modern Compilers (GCC 11 & Clang 18) ---
    FROM base AS compilers
    RUN apt-get update && \
        # Install GCC-11 from the standard repo
        apt-get install -y gcc-11 g++-11 && \
        # Set up the official LLVM APT repository to get Clang 18
        wget https://apt.llvm.org/llvm.sh && \
        chmod +x llvm.sh && \
        ./llvm.sh 18 all && \
        rm llvm.sh && \
        # Set GCC 11 as the default gcc/g++
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11 && \
        # Set Clang 18 as the default clang/clang++
        update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 180 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-18 && \
        rm -rf /var/lib/apt/lists/*


    
    # --- Stage 3: Build and Install CMake from Source ---
    FROM compilers AS cmake_build
    RUN cd /usr/src && \
        wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz && \
        tar -xzvf cmake-3.29.3.tar.gz && \
        cd cmake-3.29.3/ && \
        ./bootstrap --system-curl && make -j$(nproc) && make install && hash -r && \
        cd / && rm -rf /usr/src/cmake-3.29.3.tar.gz /usr/src/cmake-3.29.3

    # --- Stage 4: Build and Install FlatBuffers (for flatc compiler) ---
    # IREE requires the flatc compiler to process .fbs files.
    FROM cmake_build AS flatbuffers_build
    RUN cd /usr/src && \
        git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers.git && \
        cd flatbuffers && \
        cmake -B build -G "Ninja" . && \
        cmake --build build --target install && \
        cd / && rm -rf /usr/src/flatbuffers
    
    # --- Stage 5: Install Python 3.10 and Dependencies ---
    FROM flatbuffers_build AS python_build
    RUN add-apt-repository ppa:deadsnakes/ppa -y && \
        apt-get update && \
        apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
        python3 -m pip install --no-cache-dir --upgrade pip 'setuptools>=62.4.0' \
        'numpy>=2.0.0b1' 'requests>=2.28.0' 'wheel>=0.36.2' 'sympy==1.12.1' && \
        rm -rf /var/lib/apt/lists/*
    
    # --- Stage 6: Final Environment Setup for Cross-Compilation ---
    FROM python_build AS final_image
    
    # Copy the CMake toolchain file into the image
    COPY kria.toolchain.cmake /opt/kria.toolchain.cmake
    
    # Add the Kria sysroot to the image in a structured path
    COPY kria-sysroot.tar.xz /tmp/
    RUN mkdir -p /opt/sysroot && \
        tar -xvf /tmp/kria-sysroot.tar.xz -C /opt/sysroot && \
        rm /tmp/kria-sysroot.tar.xz
    
    # Set up environment variables
    ENV PATH="/usr/local/bin:${PATH}"
    ENV WORKSPACE_DIR=/workspace
    ENV BUILD_HOST_DIR=${WORKSPACE_DIR}/build-host
    ENV INSTALL_HOST_DIR=${BUILD_HOST_DIR}/install
    ENV BUILD_KRIA_DIR=${WORKSPACE_DIR}/build-kria
    ENV INSTALL_KRIA_DIR=${BUILD_KRIA_DIR}/install
    ENV CMAKE_TOOLCHAIN_FILE=/opt/kria.toolchain.cmake
    RUN mkdir -p ${BUILD_HOST_DIR} ${INSTALL_HOST_DIR} ${BUILD_KRIA_DIR} ${INSTALL_KRIA_DIR}
    
    # --- Stage 7: Entrypoint and Welcome Message ---
    RUN echo '#!/bin/bash' > /usr/local/bin/entrypoint.sh && \
        echo 'echo "================================================================="' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "  Welcome to the Custom Kria Cross-Compilation Environment"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "================================================================="' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "GCC:       $(gcc --version | head -n 1)"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "Clang:     $(clang --version | head -n 1)"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "CMake:     $(cmake --version | head -n 1)"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "FlatC:     $(flatc --version 2>&1)"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "Python:    $(python3 --version)"' >> /usr/local/bin/entrypoint.sh && \
        echo "echo 'Toolchain: ${CMAKE_TOOLCHAIN_FILE}'" >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "Sysroot:   /opt/sysroot"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "================================================================="' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "Available Build Environment Variables:"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "- BUILD_HOST_DIR:   ${BUILD_HOST_DIR}"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "- INSTALL_HOST_DIR: ${INSTALL_HOST_DIR}"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "- BUILD_KRIA_DIR:   ${BUILD_KRIA_DIR}"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "- INSTALL_KRIA_DIR: ${INSTALL_KRIA_DIR}"' >> /usr/local/bin/entrypoint.sh && \
        echo 'echo "================================================================="' >> /usr/local/bin/entrypoint.sh && \
        echo 'exec "$@"' >> /usr/local/bin/entrypoint.sh && \
        chmod +x /usr/local/bin/entrypoint.sh
    
    WORKDIR /workspace
    ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
    CMD ["bash"]
    