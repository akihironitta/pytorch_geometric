ARG PYTHON_VERSION=3.10

FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}

RUN sudo apt update && sudo apt install -y cmake

RUN pip install -U pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# RUN pip install git+https://github.com/pyg-team/pyg-lib.git

RUN pip install pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
