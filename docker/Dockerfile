ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"
ARG DEBIAN_FRONTEND=noninteractive

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt install -y git vim

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*

### Installing DualCoOp
RUN git clone https://github.com/sunxm2357/DualCoOp.git /DualCoOp
WORKDIR /DualCoOp
RUN pip install torchvision torchaudio pycocotools randaugment

### Installing Dassl
# Clone this repo
RUN git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
WORKDIR /DualCoOp/Dassl.pytorch/

# Create a conda environment
#conda create -y -n dassl python=3.8

# Install Dassl dependencies
RUN pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
RUN python setup.py develop
