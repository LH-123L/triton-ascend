<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
  </picture>
</p>

<h3 align="center">
Triton-Ascend 
</h3>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

<p align="center">
| <a href="https://triton-ascend.readthedocs.io/zh-cn/latest/"><b>官方文档</b></a> | <a href="https://www.hiascend.com/developer/operator?tag=triton"><b>算子开发用户旅程</b></a> | <a href="https://etherpad-ascend.meeting.osinfra.cn/p/sig-AscendNPU-IR"><b>社区例会</b></a> | <a href="https://www.hiascend.com/"><b>关于昇腾</b></a> |
</p>

---

## 🔥 最新消息

- [2026.04.30] Triton-Ascend 3.2.1 正式版本上线
- [2026.01.20] Triton-Ascend 3.2.0 正式版本上线

<div style="margin-left:1em">
<details>
<summary>更多最新消息</summary>

- [2025.11.14] Triton-Ascend 3.2.0rc4预发布版本上线：<br>- [扩展 tt.fp_to_fp 接口，新增对 FP8的类型转换支持](https://gitcode.com/Ascend/triton-ascend/pull/891) <br>- [新增 scatter_ub_to_out 接口，支持从UB到GM的高效数据分散操作](https://gitcode.com/Ascend/triton-ascend/pull/864)
- [2025.09.30] 完善Scan/Sort类Triton Python API，支持非连续访存，完成vLLM、sglang开源仓中重点Triton算子适配
- [2025.09.19] 支持Triton-Ascend [nightly包](https://test.pypi.org/project/triton-ascend/#history)提取
- [2025.08.15] 完善Atomic类Triton Python API支持，完成Flaggems开源仓重点Triton算子适配，提供Matmul等简单算子高性能实现参考用例
- [2025.06.30] 支持85% Triton Python API，支持连续访存，覆盖基本使用场景需求
- [2025.05.20] Triton-Ascend开源，Gitcode代码仓Alive！

</details>
</div>

---

## 📌 简介

Triton-Ascend是面向昇腾平台构建的Triton编译框架，旨在让Triton代码能够在昇腾硬件上高效运行。
Triton-Ascend通过Triton编译栈对昇腾NPU进行适配与专项优化，大幅提升功能与性能泛化能力。
Triton-Ascend框架打通Triton与昇腾硬件壁垒，降低开发门槛，同时补齐昇腾软件栈能力、丰富算子与应用生态。

## 📖 快速安装

### 环境准备

#### 硬件要求

支持的操作系统: linux(arch64/x86_64)

支持的 Ascend 产品: Atlas A2/A3 系列

最小硬件配置: 单卡 32GB 显存（推荐）

#### 软件依赖

确定 Python、CANN 和 torch_npu 软件版本并安装，软件包安装和源码编译安装均需要先完成这一步。

- Python 版本选择：py3.9-py3.11 均可。

- CANN 版本选择：可以访问昇腾社区官网，根据其提供的<a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">社区软件安装指引</a>完成 CANN 的安装与配置。建议下载安装 8.5.0 版本。

- torch_npu 版本选择：当前配套的 torch_npu 版本为 2.7.1.post4。
  
### 具体实施（以whl包安装为例）

```bash
# 以安装 triton-ascend 3.2.1 为例
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

注意：triton-ascend 3.2.1 及以上，Triton-Ascend 通过将 Triton 声明为安装依赖来缓解安装覆盖问题。安装 Triton-Ascend 时会先安装社区 Triton，再由 Triton-Ascend 覆盖同名目录，从而避免后续安装其他依赖 Triton 的软件包时再次安装 Triton 而覆盖 Triton-Ascend。

## 基于pip安装

### 最新稳定版本

- 您可以通过pip安装Triton-Ascend的最新稳定版本。

```shell
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

- 注意：triton-ascend 3.2.0 及以下 Triton-Ascend 和 Triton 不能同时存在。需要先卸载社区 Triton，再安装 Triton-Ascend。<br>
triton-ascend 3.2.1 及以上，Triton-Ascend 通过将 Triton 声明为安装依赖来缓解安装覆盖问题。
安装 Triton-Ascend 时会先安装社区 Triton，再由 Triton-Ascend 覆盖同名目录，从而避免后续安装其他依赖 Triton 的软件包时再次安装 Triton 而覆盖 Triton-Ascend。
x86 与 arm 使用不同版本的社区 Triton 安装包的原因是社区从 3.5 版本开始才提供 arm 版本安装包：x86 依赖 triton==3.2.0，arm 依赖 triton==3.5.0。

```shell
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

### 历史稳定版本

```shell
pip install triton-ascend==3.2.0
```

## 基于源码安装

### 快速安装

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
git checkout main

# 执行安装命令
pip install -e .
```

## 基于LLVM构建

Triton 使用 LLVM 22 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。LLVM的构建支持两种构建方式，**以下两种方式二选一即可**，无需重复执行。
<div style="margin-left:1em">
<details>
<summary>更多基于LLVM构建</summary>

### 代码准备: `git checkout` 检出指定版本的LLVM

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout fad3272286528b8a491085183434c5ad4b59ab92
   wget https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch
   git apply fad3272.patch
   ```

### clang构建安装LLVM

- 步骤1：使用clang安装LLVM，环境上请安装clang、lld，并指定版本（推荐版本clang>=15，lld>=15），
  如未安装，请按下面指令安装clang、lld、ccache：

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

- 步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- 步骤3：执行以下命令进行构建和安装LLVM：

  ```bash
  cd {PATH_TO}/llvm_project # 路径为用户拉取LLVM代码的路径,需根据实际调整
  mkdir build
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
    -DCMAKE_LINKER=/usr/bin/lld-15 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
  ninja install
  ```

- 步骤4：需要拷贝FILECHECK到目标安装路径：

   ```bash
   cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
   ```

### 克隆 Triton-Ascend

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
```

### 构建 Triton-Ascend

- 步骤1：请确认已设置 [基于LLVM构建] 章节中，LLVM安装的目标路径 ${LLVM_INSTALL_PREFIX}
- 步骤2：请确认已安装clang>=15，lld>=15，ccache

   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

注1：推荐GCC版本见前段章节“系统推荐”，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”，说明链接器无法找到 stdc++fs 库。
该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中相关代码片段的注释打开：

triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

  取消注释后重新构建项目即可解决该问题。
<a id="docker-build"></a>

## 基于Dockerfile安装

- 我们提供了Dockerfile帮助您安装Docker环境镜像。构建过程使用`quay.io/ascend/cann`预构建镜像作为基础镜像，跳过CANN安装步骤，显著加快构建速度。

- 您需要通过`--build-arg`指定`CANN_BASE_IMAGE`参数来选择适合您机器的CANN基础镜像。可用的CANN基础镜像标签可在[quay.io/ascend/cann](https://quay.io/repository/ascend/cann?tab=tags)查看。

- 您可以通过 npu-smi 命令查看系统上的NPU型号。

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

- 根据该镜像启动容器，可以参考下面的命令：

```bash
docker run -u 0 -dit --shm-size=512g --name=triton-ascend_container --net=host --privileged \
--security-opt seccomp=unconfined \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /home:/home \
-v /etc/ascend_install.info:/etc/ascend_install.info \
triton-ascend-image:latest \
/bin/bash

# 进入容器
docker exec -u root -it triton-ascend_container /bin/bash
```

</details>
</div>

## 🏘️ 社区活动信息

1. [会议日历](https://meeting.osinfra.cn/ascend)
2. [会议纪要看板](https://etherpad-ascend.meeting.osinfra.cn/p/sig-AscendNPU-IR)

## 🤝 社区与贡献

欢迎参与Triton-Ascend的开发及代码贡献，详情请参阅[贡献指南](./CONTRIBUTING_zh.md)。

- 请通过[Issue](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何Bug。

- 请通过[用户论坛](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support)来交流使用问题和寻求帮助。
