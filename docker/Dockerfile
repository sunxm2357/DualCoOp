ARG DEBIAN_FRONTEND=noninteractive
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt install -y git nano vim wget

### Installing DualCoOp
RUN git clone https://github.com/CarmelaCalabrese/DualCoOp.git /DualCoOp
WORKDIR /DualCoOp

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

#RUN conda env create -f environment.yml
RUN conda create -n dualcoop python=3.9
SHELL ["conda","run","-n","dualcoop","/bin/bash","-c"]

#Install pytorch for cuda 11.7
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
RUN pip install pycocotools randaugment

### Installing Dassl
# Clone this repo
RUN git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
WORKDIR /DualCoOp/Dassl.pytorch/

# Install Dassl dependencies
RUN pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
RUN python setup.py develop

WORKDIR /DualCoOp
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dualcoop", "/bin/bash", "-c"]
