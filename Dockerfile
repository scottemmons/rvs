# Based on HumanCompatibleAI/eirli Dockerfile, but with the following changes:
# - Update to cuda 11.0
# - Remove requirements.txt and MineRL installations
# - Remove X server display configuration
# - Include both mujoco210 and mujoco-2.1.1
# The Conda bits are based on https://hub.docker.com/r/continuumio/miniconda3/dockerfile
FROM ubuntu:20.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libegl1-mesa \
    xvfb \
    rsync \
    gcc \
    g++ \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
  && chmod +x /usr/local/bin/patchelf

# Install mujoco210
RUN mkdir -p /root/.mujoco \
  && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
  && tar -zxvf mujoco.tar.gz --no-same-owner --directory /root/.mujoco \
  && rm mujoco.tar.gz \
  && wget https://roboti.us/file/mjkey.txt -O /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
# Install mujoco-2.1.1
RUN mkdir -p /root/.mujoco \
  && wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz -O mujoco.tar.gz \
  && tar -zxvf mujoco.tar.gz --no-same-owner --directory /root/.mujoco \
  && rm mujoco.tar.gz \
  && wget https://roboti.us/file/mjkey.txt -O /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco-2.1.1/bin:${LD_LIBRARY_PATH}

# tini is a simple init which is used by the official Conda Dockerfile (among
# other things). It can do stuff like reap zombie processes & forward signals
# (e.g. from "docker stop") to subprocesses. This may be useful if the code
# breaks in such a way that it creates lots of zombies or cannot easily be
# killed (e.g. maybe a Python extension segfaults and doesn't wait on its
# children, which keep running). That said, Sam hasn't yet run into a
# situation where it was necessary with the il-representations code base, at
# least as of October 2020.
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Install Conda and make it the default Python
ENV PATH /opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/conda.sh || true \
  && bash /root/conda.sh -b -p /opt/conda || true \
  && rm /root/conda.sh
RUN conda update -n base -c defaults conda \
  && conda install -c anaconda python=3.7 \
  && pip install --upgrade pip \
  && conda clean -ay

# Install CUDA for PyTorch
RUN conda install -c anaconda cudatoolkit \
  && conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch

# Always run under tini (see explanation above)
ENTRYPOINT [ "/usr/bin/tini", "--" ]
