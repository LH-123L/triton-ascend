# =====================================================================
# Triton-Ascend 基础镜像：openEuler 24.03 + 全部 yum 依赖 + Python 3.11.15
#
# 构建（在 Docker 正常的机器上执行；CodeArts 构建机为 arm64）：
#   docker build --network host --security-opt seccomp=unconfined \
#       --platform linux/arm64 \
#       -f docker/base/openeuler24.03/Dockerfile \
#       -t <SWR地址>/triton-ascend-base:openeuler24.03-py3.11-arm64 .
#
# 说明：
#   - 合并了 3.2.2 各镜像 Dockerfile 四个阶段（python/cann/llvm/final）的全部 yum 依赖；
#   - Python 3.11.15 为源码编译产物，镜像按架构区分标签（arm64/amd64 不通用）；
#   - 9 个 3.2.2 变体 Dockerfile 均以本镜像为 FROM，不再执行任何 yum 步骤。
# =====================================================================

FROM openeuler/openeuler:24.03

ARG APT_MIRROR=mirrors.ustc.edu.cn

SHELL ["/bin/bash", "-c"]

# Python 环境
ENV PATH=/usr/local/python3.11.15/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/python3.11.15/lib:${LD_LIBRARY_PATH}
ENV PIP_DEFAULT_TIMEOUT=100 PIP_RETRIES=5

# 换 yum 源 + 安装全部依赖
RUN sed -i \
        -e "s|https\?://repo\.openeuler\.org/|https://${APT_MIRROR}/openeuler/|g" \
        -e "/^metalink=/s/^/#/" \
        /etc/yum.repos.d/openEuler.repo \
    && yum update -y \
    && yum install -y \
        ca-certificates \
        bash \
        glibc \
        gcc \
        gcc-c++ \
        g++ \
        make \
        cmake \
        curl \
        wget \
        git \
        vim \
        jq \
        zlib-devel \
        bzip2-devel \
        openssl-devel \
        ncurses-devel \
        sqlite-devel \
        readline-devel \
        tk-devel \
        gdbm-devel \
        libpcap-devel \
        xz-devel \
        libev-devel \
        expat-devel \
        libffi-devel \
        systemtap-sdt-devel \
        unzip \
        pciutils \
        net-tools \
        lapack-devel \
        gcc-gfortran \
        util-linux \
        findutils \
        libzstd \
        libzstd-devel \
        clang \
        ccache \
        lld \
        numactl-devel \
        sudo \
        procps-ng \
        sysstat \
        systemd \
        iproute \
        openssl \
        grep \
        tree \
        rsync \
        tar \
        zip \
        python3-devel \
        dos2unix \
    && yum clean all \
    && rm -rf /var/cache/yum /tmp/*

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
