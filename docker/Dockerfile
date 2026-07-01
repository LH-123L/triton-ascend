ARG CANN_BASE_IMAGE=quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11
ARG PIP_SOURCE="https://mirrors.ustc.edu.cn/pypi/simple"
FROM ${CANN_BASE_IMAGE}
ARG PIP_SOURCE
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list 2>/dev/null || true && \
    echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries && \
    apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    clang-15 \
    clang-format \
    cmake \
    ccache \
    curl \
    git \
    gnupg \
    lld-15 \
    make \
    ninja-build \
    sudo \
    unzip \
    vim \
    wget \
    libzstd-dev \
    zlib1g-dev && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 100 && \
    update-ca-certificates

RUN mkdir -p /root/.config/pip && \
    echo -e "[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple\nextra-index-url = https://download.pytorch.org/whl/cpu" > /root/.config/pip/pip.conf

RUN pip3 install --no-cache-dir \
    ninja \
    cmake \
    wheel \
    pybind11 \
    --index-url ${PIP_SOURCE}

COPY requirements.txt requirements_dev.txt ./

RUN \
    python3 -m pip install --no-cache-dir -r requirements_dev.txt && \
    python3 -m pip install --no-cache-dir -r requirements.txt

RUN \
   mkdir -p /home/triton && \
   cd /home/triton && \
   git clone https://gitcode.com/Ascend/triton-ascend.git && \
   cd triton-ascend && \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install

WORKDIR /home/triton
