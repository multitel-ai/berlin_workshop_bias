FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update -y && apt install -y sudo
RUN addgroup --gid 1000 user
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user
RUN usermod -aG sudo user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN pip install jupyter
RUN pip install captum
RUN pip install pipreqs
USER user
WORKDIR /App 
