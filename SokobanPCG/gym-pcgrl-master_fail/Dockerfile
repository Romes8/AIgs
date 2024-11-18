# Use the specified TensorFlow base image with GPU support
FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Remove any existing CUDA repositories to prevent GPG errors
RUN rm -rf /etc/apt/sources.list.d/cuda*.list || true

# Update package lists and install required packages in a single RUN statement
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-opengl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/src/app

# Copy setup files
COPY setup.py README.md ./

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Install a specific version of gym compatible with stable_baselines
RUN pip3 install gym==0.21.0

# Install stable_baselines
RUN pip3 install stable_baselines==2.10.2

# Install the package in editable mode
RUN pip3 install -e .

# Copy the rest of the application code
COPY . ./

# Set the default command
CMD ["python3", "train.py"]
