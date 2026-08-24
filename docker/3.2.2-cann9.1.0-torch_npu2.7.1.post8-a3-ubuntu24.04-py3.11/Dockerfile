ARG PIP_SOURCE="https://mirrors.ustc.edu.cn/pypi/simple"

# ==================== 预构建基础镜像 ====================
# 系统依赖 + Python 3.11.15 + LLVM 已预构建为基础镜像（见 docker/base/README.zh.md），
# 本 Dockerfile 不再执行 apt/yum 联网步骤，也不编译 LLVM。
# 基础镜像为多架构 manifest（标签不带架构后缀），Docker 按宿主机架构自动拉取。
ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/opentile/triton-ascend-base:ubuntu24.04-py3.11

# ==================== Stage 1: Python 环境（来自预构建基础镜像） ====================
FROM ${BASE_IMAGE} AS python-installer

ARG PIP_SOURCE

# Python Environment variables
ENV PATH=/usr/local/python3.11.15/bin:${PATH}
# pip 网络参数（被下游 cann/llvm 阶段继承）
ENV PIP_DEFAULT_TIMEOUT=100 PIP_RETRIES=5

# ==================== Stage 2: 安装 CANN（含 torch_npu / triton-ascend） ====================
FROM python-installer AS cann-installer

ARG PIP_SOURCE
# CodeArts 传统构建器不会自动提供 TARGETPLATFORM，改用 uname -m 按实际运行架构探测
ARG TARGETPLATFORM=linux/amd64
ARG CANN_VERSION=9.1.0

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
# cmake 已在基础镜像中安装，不再重复 pip 安装

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

# ==================== Stage 3: 最终镜像 ====================
FROM ${BASE_IMAGE} AS official-ubuntu

ARG PIP_SOURCE
ARG TARGETPLATFORM=linux/amd64

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

COPY --from=cann-installer /usr/local/python3.11.15 /usr/local/python3.11.15
COPY --from=cann-installer /usr/local/Ascend /usr/local/Ascend
COPY --from=cann-installer /etc/Ascend /etc/Ascend
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
