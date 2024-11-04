FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3.10 \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# Install any python packages you need
RUN pip list
RUN pip install numpy==2.0.0

# Upgrade pip

RUN python3 -m pip install --upgrade pip
RUN pip install --upgrade pip

RUN pip install fastai
RUN pip install accelerate

COPY train.py train.py

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Set the working directory
WORKDIR /app

# Set the entrypoint
#ENTRYPOINT [ "python3" ]
#CMD [ "python3", "test.py" ]
# docker run -d --name fastai -v /media/FastDataMama/zigaB/DentAge/:/data/ image:tag
# sudo docker run --name pytorch-container --gpus all -it --rm -v $(pwd):/app pytorch-gpu
# sudo docker run --name pytorch-container --gpus 0 -it --rm -v $(pwd):/app pytorch-gpu python3 kakec.py
