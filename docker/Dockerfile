FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG USERNAME=user
ENV TORCH_CUDA_ARCH_LIST=8.0;8.6
ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.8.4
ARG TORCH_VERSION=1.12.1+cu116
ARG TORCHVISION_VERSION=0.13.1
ARG MIM_VERSION=0.3.3
ARG MMCV_VERSION=1.7.0
ARG MMDET_VERSION=2.28.0
ARG ONNX_VERSION=1.13.1
ARG ONNXRUNTIME_VERSION=1.15.1
ARG ONNXSIM_VERSION=0.4.33

SHELL ["/bin/bash", "-c"]

# packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo git zip curl nano libgtk2.0-dev \
    libopencv-dev libffi-dev liblapack-dev libsqlite3-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev \
    ssh-client wget vim &&\
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# user setting
RUN echo "root:root" | chpasswd \
    && useradd \
        --create-home \
        --home-dir /home/${USERNAME} \
        --shell /bin/bash \
        --user-group \
        --groups adm,sudo \
        ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && cat /dev/null > /etc/sudoers.d/${USERNAME} \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> \
        /etc/sudoers.d/${USERNAME} \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

# python
ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ARG PYTHON_VERSION
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

# torch
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
RUN pip install torch==${TORCH_VERSION} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==${TORCHVISION_VERSION}

# mmcv
ARG MIM_VERSION
ARG MMCV_VERSION
RUN pip install -U openmim==${MIM_VERSION}
RUN mim install mmcv-full==${MMCV_VERSION}

# mmdet
ARG MMDET_VERSION
WORKDIR /home/${USERNAME}/
RUN pip install Cython
RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection -b v${MMDET_VERSION} --depth 1
WORKDIR /home/${USERNAME}/mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -e .

# requirements
RUN pip install -U pip setuptools
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# packages
RUN pip install pip -U \
    && pip install onnx==${ONNX_VERSION} \
    && pip install onnxruntime==${ONNXRUNTIME_VERSION} \
    && pip install onnxsim==${ONNXSIM_VERSION} \
    && pip install opencv-contrib-python==4.8.0.76

# lane NMS
COPY --chown=${USERNAME} libs/models/layers/nms /tmp/nms
WORKDIR /tmp/nms
RUN python /tmp/nms/setup.py install

# path
ENV PYTHONPATH=$PYTHONPATH:/work
ENV PYTHONPATH=$PYTHONPATH:/home/${USERNAME}/

WORKDIR /work

USER ${USERNAME}
