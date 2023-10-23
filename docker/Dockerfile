ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"
ARG DEBIAN_FRONTEND=noninteractive

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt install -y git nano
#RUN apt-get update && apt install -y git nano conda?

### Installing DualCoOp
RUN git clone https://github.com/CarmelaCalabrese/DualCoOp.git /DualCoOp
WORKDIR /DualCoOp
#RUN conda env create -f environment.yml
RUN pip install torchvision torchaudio pycocotools randaugment

### Installing Dassl
# Clone this repo
RUN git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
WORKDIR /DualCoOp/Dassl.pytorch/

# Create a conda environment
RUN conda create -y -n dassl python=3.8

# Activate the environment
RUN conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install Dassl dependencies
RUN pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
RUN python setup.py develop
WORKDIR /DualCoOp