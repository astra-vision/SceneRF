# Dockerfile
#
# Created on Tue Nov 30 2023 by Florian Pfleiderer
#
# Copyright (c) 2023 TU Wien

# nvidia cuda base image
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# Set the working directory in the container
WORKDIR /scene-rf

# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install git and wget
RUN apt-get update && \
    apt-get install -y git wget cmake gcc g++ libgl1-mesa-glx libglib2.0-0 unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Reset noninteractive mode
ENV DEBIAN_FRONTEND=dialog

# Install miniconda.
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
# Make non-activate conda commands available.
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda activate command available from /bin/bash --login shells.
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# Make conda activate command available from /bin/bash --interative shells.
RUN conda init bash

# start shell in login mode
SHELL ["/bin/bash", "--login", "-c"]

# run updates
RUN conda update -n base -c defaults conda

# Download pretrained model
RUN wget https://www.rocq.inria.fr/rits_files/computer-vision/scenerf/scenerf_bundlefusion.ckpt

COPY requirements.txt setup.py ./

# Create a Conda environment
RUN conda create -y -n scenerf python=3.7

RUN conda install -y -n scenerf -c pytorch pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2
RUN conda install -y -n scenerf -c bioconda tbb=2020.2
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN conda activate scenerf && pip install -r requirements.txt
RUN conda activate scenerf && pip install torchmetrics==0.6.0

#install scenerf
RUN conda activate scenerf && pip install -e .

# start container in cyws3d env
RUN touch ~/.bashrc && echo "conda activate scenerf" >> ~/.bashrc

COPY scenerf/ scenerf/
COPY teaser/ teaser/

# Set the default command to run when the container starts
CMD ["bash"]