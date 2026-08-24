# =====================================================================
# Triton-Ascend 基础镜像：Debian 12 + 全部 apt 依赖 + Python 3.11.15
#
# 构建（在 Docker 正常的机器上执行；CodeArts 构建机为 arm64）：
#   docker build --network host --security-opt seccomp=unconfined \
#       --platform linux/arm64 \
#       -f docker/base/debian12/Dockerfile \
#       -t <SWR地址>/triton-ascend-base:debian12-py3.11-arm64 .
#
# 说明：
#   - 合并了 3.2.2 各镜像 Dockerfile 四个阶段（python/cann/llvm/final）的全部 apt 依赖；
#   - Python 3.11.15 为源码编译产物，镜像按架构区分标签（arm64/amd64 不通用）；
#   - 9 个 3.2.2 变体 Dockerfile 均以本镜像为 FROM，不再执行任何 apt 步骤。
# =====================================================================

FROM debian:12

ARG APT_MIRROR=mirrors.ustc.edu.cn

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Python 环境
ENV PATH=/usr/local/python3.11.15/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/python3.11.15/lib:${LD_LIBRARY_PATH}
ENV PIP_DEFAULT_TIMEOUT=100 PIP_RETRIES=5

# 换 apt 源 + 安装全部依赖（统一用 libncurses-dev）
RUN sed -i \
        -e "s|deb\.debian\.org|${APT_MIRROR}|g" \
        -e "s|security\.debian\.org/debian-security|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y \
        apt-transport-https \
        ca-certificates \
        bash \
        curl \
        wget \
        git \
        vim \
        jq \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        clang-15 \
        ccache \
        lld-15 \
        zlib1g \
        zlib1g-dev \
        libssl-dev \
        libncurses-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        libnss3-dev \
        libgdbm-dev \
        liblzma-dev \
        libev-dev \
        libzstd-dev \
        libnuma-dev \
        libblas-dev \
        gfortran \
        patchelf \
        pciutils \
        net-tools \
        openssl \
        unzip \
        sudo \
        procps \
        sysstat \
        systemd \
        iproute2 \
        grep \
        tree \
        rsync \
        tar \
        zip \
        findutils \
        python3-dev \
        openssh-server \
        openssh-client \
        dos2unix \
        libc6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/tmp/* /tmp/* \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++

# 编译安装 Python 3.11.15（华为云源码，--retry 防断流）
RUN curl -fsSL --retry 5 --retry-all-errors --retry-delay 5 \
        https://repo.huaweicloud.com/python/3.11.15/Python-3.11.15.tgz \
        -o /tmp/Python-3.11.15.tgz \
    && tar -xf /tmp/Python-3.11.15.tgz -C /tmp \
    && cd /tmp/Python-3.11.15 \
    && mkdir -p /usr/local/python3.11.15/lib \
    && ./configure \
        --enable-shared \
        LDFLAGS="-Wl,-rpath,/usr/local/python3.11.15/lib" \
        --prefix=/usr/local/python3.11.15 \
    && make -j"$(nproc)" \
    && make altinstall \
    && ln -sf /usr/local/python3.11.15/bin/python3.11 /usr/local/python3.11.15/bin/python3 \
    && ln -sf /usr/local/python3.11.15/bin/pip3.11 /usr/local/python3.11.15/bin/pip3 \
    && ln -sf /usr/local/python3.11.15/bin/python3 /usr/local/python3.11.15/bin/python \
    && ln -sf /usr/local/python3.11.15/bin/pip3 /usr/local/python3.11.15/bin/pip \
    && rm -rf /tmp/*

# pip 默认走华为云镜像
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple\nextra-index-url = https://download.pytorch.org/whl/cpu\n' > /root/.config/pip/pip.conf
