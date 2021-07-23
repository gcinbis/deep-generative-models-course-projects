FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV PROJECT=testcahit
ENV PYTORCH_VERSION=1.8.0
ENV TORCHVISION_VERSION=0.9.0
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2
ENV NCCL_VERSION=2.5.6-1+cuda10.2
# apt-cache policy libnccl2
ENV DEBIAN_FRONTEND=noninteractive

ARG python=3.7
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update

RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg


# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install PyTorch
RUN pip install torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} && ldconfig

RUN pip install \
    cython \
    numpy \
    pillow \
    pillow-simd \
    opencv-python \
    opencv-contrib-python \
    opencv-python-headless \
    numba
RUN pip install \
    tqdm \
    jupyter \
    matplotlib \
    'tensorflow==1.15' \
    imageio \
    pytorch_fid \
    pandas

# create project workspace dir
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/${PROJECT}
WORKDIR /workspace/${PROJECT}

# Copy project source last (to avoid cache busting)
# WORKDIR /workspace/${PROJECT}
# COPY . /workspace/${PROJECT}
ENV PYTHONPATH="workspace/${PROJECT}:$PYTHONPATH"
