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

FROM hub.oepkgs.net/openeuler/openeuler:24.03

ARG APT_MIRROR=repo.huaweicloud.com

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

# ==================== LLVM/MLIR 预编译 ====================
# LLVM 编译产物与 910b/950/a3 芯片无关，只与 OS/架构相关，直接编进基础镜像，
# CI 不再编译 LLVM。源码由宿主机预下载后放到仓库根目录（llvm-project.tar.gz），
# 与旧版 Dockerfile 的 COPY 方式保持一致。
ARG PIP_SOURCE="https://mirrors.ustc.edu.cn/pypi/simple"
# CI 无法稳定访问 github.com，LLVM 源码改为宿主机预下载后 COPY 进构建上下文
# 地址可在构建参数里覆盖：--build-arg LLVM_SRC_URL=<其他代理或 Gitee 镜像的 tarball 地址>
ARG LLVM_SRC_URL=https://ghproxy.com/https://github.com/llvm/llvm-project/archive/b5cc222d7429fe6f18c787f633d5262fac2e676f.tar.gz

RUN pip3 install --no-cache-dir ninja wheel pybind11 --index-url ${PIP_SOURCE}

# 直接 COPY 宿主机预下载的固定 commit 源码包，避免 git clone 依赖 github.com 连通性
# -DLLVM_CCACHE_BUILD=ON：启用已安装的 ccache，缓存写入镜像层 /root/.ccache
# 若面向发布镜像，可把 LLVM_ENABLE_ASSERTIONS 改为 OFF 以提升运行性能
COPY llvm-project.tar.gz /tmp/llvm-project.tar.gz

RUN tar -xzf /tmp/llvm-project.tar.gz -C /tmp \
    && mv /tmp/llvm-project-* /tmp/llvm-project \
    && mkdir -p /tmp/llvm-project/build \
    && cd /tmp/llvm-project/build \
    && export LLVM_INSTALL_PREFIX=/usr/local/llvm-install \
    && cmake ../llvm \
        -G Ninja \
        -DCMAKE_C_COMPILER=/usr/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
        -DCMAKE_LINKER=/usr/bin/lld \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_CCACHE_BUILD=ON \
        -DLLVM_CCACHE_DIR=/root/.ccache \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_LLD=ON \
        -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX} \
    && ninja install \
    && cd / \
    && rm -rf /tmp/llvm-project /tmp/llvm-project.tar.gz
