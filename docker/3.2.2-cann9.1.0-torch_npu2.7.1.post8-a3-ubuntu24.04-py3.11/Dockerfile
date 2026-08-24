ARG PIP_SOURCE="https://mirrors.ustc.edu.cn/pypi/simple"

# ==================== Stage 1: 编译安装 Python 3.11.15 ====================
FROM ubuntu:24.04 AS python-installer

ARG APT_MIRROR=mirrors.ustc.edu.cn

# Python Environment variables
ENV PATH=/usr/local/python3.11.15/bin:${PATH}
# pip 网络参数（被下游 cann/llvm 阶段继承）
ENV PIP_DEFAULT_TIMEOUT=100 PIP_RETRIES=5

# 换 apt 源 + 安装编译依赖（DEBIAN_FRONTEND 防止 tzdata 等交互弹窗）
RUN sed -i \
        -e "s|archive\.ubuntu\.com|${APT_MIRROR}|g" \
        -e "s|security\.ubuntu\.com|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        bash \
        curl \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        libnss3-dev \
        libgdbm-dev \
        liblzma-dev \
        libev-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/tmp/* /tmp/*

# 下载 Python 源码：--retry 防断流；SHA256 校验示例见本 RUN 之后的注释
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
# 可选：从 python.org 获取官方 SHA256 后，在 tar 之前插入这一行做完整性校验：
#   && echo "<官方SHA256>  /tmp/Python-3.11.15.tgz" | sha256sum -c - \
# 可选：追求运行时性能可加 --enable-optimizations（代价是编译时间显著变长）

# ==================== Stage 2: 安装 CANN（含 torch_npu / triton-ascend） ====================
FROM python-installer AS cann-installer

ARG PIP_SOURCE
# CodeArts 传统构建器不会自动提供 TARGETPLATFORM（BuildKit 才会），这里给默认值
ARG TARGETPLATFORM=linux/amd64
ARG APT_MIRROR=mirrors.ustc.edu.cn
ARG CANN_VERSION=9.1.0

# libblas3 由 libblas-dev 依赖自动带入，不再单独安装
RUN sed -i \
        -e "s|archive\.ubuntu\.com|${APT_MIRROR}|g" \
        -e "s|security\.ubuntu\.com|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        git \
        wget \
        gcc \
        g++ \
        make \
        cmake \
        zlib1g \
        openssl \
        unzip \
        pciutils \
        net-tools \
        libblas-dev \
        gfortran \
        patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip：传统构建器无法使用 --mount，此处用 --no-cache-dir 防镜像膨胀
# 如需 pip 缓存加速，请使用 Dockerfile.buildkit（需 CI 开启 BuildKit/buildx）
RUN pip install --no-cache-dir --upgrade pip --index-url ${PIP_SOURCE}

# 按架构生成下载地址并下载 CANN 安装包（--tries/--timeout 防大文件断流）
RUN ARCH=$(uname -m) \
    && test -n "${ARCH}" \
        || { echo "Unsupported ARCH: ${ARCH}"; exit 1; } \
    && echo "Building for ARCH=${ARCH}" \
    && \
    CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run && \
    CANN_OPS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-A3-ops_${CANN_VERSION}_linux-${ARCH}.run && \
    wget --quiet --tries=5 --timeout=60 --waitretry=10 \
        --header="Referer: https://www.hiascend.com/" \
        ${CANN_TOOLKIT_URL} -O ~/Ascend-cann-toolkit.run && \
    wget --quiet --tries=5 --timeout=60 --waitretry=10 \
        --header="Referer: https://www.hiascend.com/" \
        ${CANN_OPS_URL} -O ~/Ascend-cann-ops.run
# 可选：从华为官方渠道获取 SHA256 后，用 sha256sum -c 校验两个 .run 文件

# 安装 CANN Toolkit 与 Ops 包（合并成一层，装完即删安装包）
RUN chmod +x ~/Ascend-cann-toolkit.run \
    && ~/Ascend-cann-toolkit.run --quiet --install --install-for-all \
    && rm -f ~/Ascend-cann-toolkit.run \
    && chmod +x ~/Ascend-cann-ops.run \
    && ~/Ascend-cann-ops.run --quiet --install --install-for-all \
    && rm -f ~/Ascend-cann-ops.run

# Note: 安装 torch_npu 与 triton-ascend
# 注意：这里没有新的 FROM，属于 cann-installer 阶段，最终镜像才能通过 COPY 带走
RUN pip3 install --no-cache-dir \
        pyyaml \
        setuptools \
        pybind11 \
        ninja \
        wheel \
        tabulate \
        --index-url ${PIP_SOURCE}
# cmake 已在上面 apt 安装，不再重复 pip 安装

RUN case "$(uname -m)" in \
        "x86_64")  \
            echo "install torch-npu for the x86 architecture"; \
            pip3 install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu;; \
        "aarch64") \
            echo "install torch-npu for the arm architecture"; \
            pip3 install torch==2.7.1 --index-url ${PIP_SOURCE};; \
        *) echo "Unsupported ARCH: $(uname -m)" && exit 1 ;; \
    esac && \
    pip3 install torch-npu==2.7.1.post8 --index-url ${PIP_SOURCE} && \
    pip3 install triton-ascend==3.2.2 \
        --index-url ${PIP_SOURCE} \
        --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

# ==================== Stage 3: 编译 LLVM / MLIR ====================
FROM python-installer AS llvm-installer

ARG PIP_SOURCE
ARG APT_MIRROR=mirrors.ustc.edu.cn
# CI 无法稳定访问 github.com，LLVM 源码改为走代理下载固定 commit 的 tar.gz
# 代理失效时可在构建参数里覆盖：--build-arg LLVM_SRC_URL=<其他代理或 Gitee 镜像的 tarball 地址>
ARG LLVM_SRC_URL=https://ghproxy.com/https://github.com/llvm/llvm-project/archive/b5cc222d7429fe6f18c787f633d5262fac2e676f.tar.gz

RUN sed -i \
        -e "s|archive\.ubuntu\.com|${APT_MIRROR}|g" \
        -e "s|security\.ubuntu\.com|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        gcc \
        g++ \
        make \
        cmake \
        wget \
        zlib1g-dev \
        libzstd-dev \
        clang-15 \
        ccache \
        lld-15 \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/tmp/* /tmp/* \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++

RUN pip3 install --no-cache-dir \
        ninja \
        wheel \
        pybind11 \
        --index-url ${PIP_SOURCE}
# cmake 已 apt 安装，不再重复 pip 安装

# 直接下载固定 commit 的源码包，避免 git clone 依赖 github.com 连通性
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
        -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
        -DCMAKE_LINKER=/usr/bin/lld-15 \
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

# ==================== Stage 4: 最终镜像 ====================
FROM ubuntu:24.04 AS official-ubuntu

ARG PIP_SOURCE
ARG TARGETPLATFORM=linux/amd64
ARG APT_MIRROR=mirrors.ustc.edu.cn

# Python Environment variables
ENV PATH=/usr/local/python3.11.15/bin:${PATH}

# Toolkit Environment variables（来自 /usr/local/Ascend/ascend-toolkit/set_env.sh，已合并为少量 ENV）
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
ENV ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest

ENV LD_LIBRARY_PATH=\
${ASCEND_TOOLKIT_HOME}/devlib:\
${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64:\
${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64/plugin:\
${ASCEND_TOOLKIT_LATEST_HOME}/lib64:\
${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/opskernel:\
${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/nnengine:\
${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:\
${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:\
${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:\
${ASCEND_TOOLKIT_HOME}/lib64:\
${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:\
${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:\
${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/python3.11.15/lib:${LD_LIBRARY_PATH}

ENV PYTHONPATH=\
${ASCEND_TOOLKIT_LATEST_HOME}/python/site-packages:\
${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe:\
${ASCEND_TOOLKIT_HOME}/python/site-packages:\
${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}

ENV PATH=\
/usr/local/python3.11.15/bin:\
${ASCEND_TOOLKIT_LATEST_HOME}/bin:\
${ASCEND_TOOLKIT_LATEST_HOME}/compiler/ccec_compiler/bin:\
${ASCEND_TOOLKIT_LATEST_HOME}/tools/ccec_compiler/bin:\
${ASCEND_TOOLKIT_HOME}/tools/show_kernel_debug_data:\
${ASCEND_TOOLKIT_HOME}/tools/msobjdump:\
${ASCEND_TOOLKIT_HOME}/bin:\
${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:\
${ASCEND_TOOLKIT_HOME}/tools/profiler/bin:\
${ASCEND_TOOLKIT_HOME}/tools/ascend_system_advisor/asys:${PATH}

ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
ENV CMAKE_PREFIX_PATH=${TOOLCHAIN_HOME}/tools/tikicpulib/lib/cmake:${ASCEND_TOOLKIT_HOME}/lib64/cmake

SHELL [ "/bin/bash", "-c" ]

# 若这是对外分发的运行时镜像，可考虑 debian:12-slim 并裁剪 vim/systemd/openssh 等开发工具
RUN sed -i \
        -e "s|archive\.ubuntu\.com|${APT_MIRROR}|g" \
        -e "s|security\.ubuntu\.com|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        bash \
        libc6 \
        libsqlite3-dev \
        gcc \
        g++ \
        make \
        cmake \
        git \
        vim \
        wget \
        jq \
        curl \
        build-essential \
        libnuma-dev \
        sudo \
        procps \
        sysstat \
        systemd \
        net-tools \
        iproute2 \
        openssl \
        grep \
        tree \
        rsync \
        tar \
        zip \
        unzip \
        findutils \
        python3-dev \
        zlib1g-dev \
        libzstd-dev \
        clang-15 \
        ccache \
        lld-15 \
        openssh-server \
        openssh-client \
        libssl-dev \
        dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/tmp/* /tmp/* \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++

COPY --from=cann-installer /usr/local/python3.11.15 /usr/local/python3.11.15
COPY --from=cann-installer /usr/local/Ascend /usr/local/Ascend
COPY --from=cann-installer /etc/Ascend /etc/Ascend
COPY --from=llvm-installer /usr/local/llvm-install /usr/local/llvm-install
# 注意：docker/copyright.sh 必须在构建上下文里存在，否则本步骤报错
COPY --chown=root:root docker/copyright.sh /etc/profile.d/copyright.sh

# 用 /etc/profile.d 脚本替代往 ~/.bashrc 追加，所有用户登录自动生效
RUN printf 'source /usr/local/Ascend/ascend-toolkit/set_env.sh\nexport LLVM_INSTALL_PREFIX=/usr/local/llvm-install\nalias ll="ls -l"\nbash /etc/profile.d/copyright.sh\n' \
        > /etc/profile.d/ascend-env.sh \
    && chmod 644 /etc/profile.d/ascend-env.sh \
    && chmod 555 /etc/profile.d/copyright.sh

ENTRYPOINT ["/bin/bash", "-c", "\
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    bash /etc/profile.d/copyright.sh && \
    exec \"$@\"", "--"]
