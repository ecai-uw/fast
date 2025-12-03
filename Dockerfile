# Use the official Ubuntu 22.04 image with CUDA 12.0.1 runtime.
FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies.
RUN apt-get update && \
    apt-get install -y curl git build-essential libssl-dev zlib1g-dev libbz2-dev cmake \
    libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv.
ENV CODING_ROOT="/opt"

WORKDIR $CODING_ROOT
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="$CODING_ROOT/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install Python 3.10 using pyenv.
RUN pyenv install 3.10
RUN pyenv global 3.10

# Install specific version of pip and setuptools.
RUN pip install --upgrade pip==25.2 setuptools==80.9.0

# Install mujoco system dependencies.
RUN apt-get update && apt-get install -y \
    patchelf \
    libgl1-mesa-dev \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libosmesa6-dev \
    libglu1-mesa-dev \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxmu6 \
    libxrandr2 \
    libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# Install mujoco as specificed by DPPO.
RUN mkdir -p /root/.mujoco && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xvf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco && \
    rm mujoco210-linux-x86_64.tar.gz

# Set MUJOCO env vars
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# Install some additional Python dependencies. TODO: eventually, put this in a pyproject.toml or something.
RUN pip install moviepy==1.0.3 imageio
RUN pip install PyOpenGL-accelerate

# Make the working directory the home directory.
RUN mkdir $CODING_ROOT/code

# Only copy in the source code that is necessary for the dependencies to install.
WORKDIR $CODING_ROOT/code/fast
COPY ./cfg $CODING_ROOT/code/fast/cfg
COPY ./dppo $CODING_ROOT/code/fast/dppo
COPY ./stable-baselines3 $CODING_ROOT/code/fast/stable-baselines3
# TODO: eventually abstract these scripts into a project hierarchy.
COPY ./env_utils.py $CODING_ROOT/code/fast/env_utils.py
COPY ./train_fast.py $CODING_ROOT/code/fast/train_fast.py
COPY ./utils.py $CODING_ROOT/code/fast/utils.py

# Install DPPO.
RUN cd dppo && \
    pip install -e . && \
    pip install -e .[robomimic] && \
    pip install -e .[gym]

# Install Stable Baselines3
RUN cd stable-baselines3 && \
    pip install -e .[extra]

# Make a logs directory.
RUN mkdir $CODING_ROOT/logs

# Small hack to avoid run-time mujoco re-compilation.
RUN python -c "import mujoco_py"

# Avoid permission issues with git in mounted volumes.
RUN git config --global --add safe.directory /opt/code/fast