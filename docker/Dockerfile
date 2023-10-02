FROM nvcr.io/nvidia/pytorch:22.01-py3
# Install mish-cuda required for YOLOv4

RUN git clone https://github.com/JunnYu/mish-cuda /mish-cuda
WORKDIR /mish-cuda
RUN python setup.py build install
WORKDIR /workspace
