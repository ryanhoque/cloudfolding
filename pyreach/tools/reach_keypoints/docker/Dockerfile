FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1-mesa-glx \
      libhdf5-dev \
      libglib2.0-0 \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim wget \
    build-essential cmake \
    libopenblas-dev

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

RUN pip install --upgrade pip && \
    pip install \
        torch==1.1.0 \
        opencv_python==3.4.2.17 \
        numpy \
        torchvision==0.2.0 \
        Cython \
        matplotlib \
        imgaug

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip install imgaug

WORKDIR /host
CMD ["/bin/bash"]
