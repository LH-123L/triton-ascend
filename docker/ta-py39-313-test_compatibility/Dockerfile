# Stage 1: Install Python
FROM ubuntu:22.04 AS python-installer
ARG PIP_SOURCE="https://mirrors.ustc.edu.cn/pypi/simple"
RUN apt-get update \
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
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/*


# Install Python 3.9.15
RUN curl -fsSL https://repo.huaweicloud.com/python/3.9.15/Python-3.9.15.tgz -o /tmp/Python-3.9.15.tgz && \
    tar -xf /tmp/Python-3.9.15.tgz -C /tmp && \
    cd /tmp/Python-3.9.15 && \
    mkdir -p /usr/local/python3.9.15/lib && \
    ./configure --enable-shared --enable-shared LDFLAGS="-Wl,-rpath /usr/local/python3.9.15/lib" --prefix=/usr/local/python3.9.15 && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/python3.9.15/bin/python3.9 /usr/local/python3.9.15/bin/python3 && \
    ln -sf /usr/local/python3.9.15/bin/pip3.9 /usr/local/python3.9.15/bin/pip3 && \
    ln -sf /usr/local/python3.9.15/bin/python3 /usr/local/python3.9.15/bin/python && \
    ln -sf /usr/local/python3.9.15/bin/pip3 /usr/local/python3.9.15/bin/pip && \
    rm -rf /tmp/* 

# Install Python 3.10.15
RUN curl -fsSL https://repo.huaweicloud.com/python/3.10.15/Python-3.10.15.tgz -o /tmp/Python-3.10.15.tgz && \
    tar -xf /tmp/Python-3.10.15.tgz -C /tmp && \
    cd /tmp/Python-3.10.15 && \
    mkdir -p /usr/local/python3.10.15/lib && \
    ./configure --enable-shared --enable-shared LDFLAGS="-Wl,-rpath /usr/local/python3.10.15/lib" --prefix=/usr/local/python3.10.15 && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/python3.10.15/bin/python3.10 /usr/local/python3.10.15/bin/python3 && \
    ln -sf /usr/local/python3.10.15/bin/pip3.10 /usr/local/python3.10.15/bin/pip3 && \
    ln -sf /usr/local/python3.10.15/bin/python3 /usr/local/python3.10.15/bin/python && \
    ln -sf /usr/local/python3.10.15/bin/pip3 /usr/local/python3.10.15/bin/pip && \
    rm -rf /tmp/* 

# Install Python 3.11.15
RUN curl -fsSL https://repo.huaweicloud.com/python/3.11.15/Python-3.11.15.tgz -o /tmp/Python-3.11.15.tgz && \
    tar -xf /tmp/Python-3.11.15.tgz -C /tmp && \
    cd /tmp/Python-3.11.15 && \
    mkdir -p /usr/local/python3.11.15/lib && \
    ./configure --enable-shared --enable-shared LDFLAGS="-Wl,-rpath /usr/local/python3.11.15/lib" --prefix=/usr/local/python3.11.15 && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/python3.11.15/bin/python3.11 /usr/local/python3.11.15/bin/python3 && \
    ln -sf /usr/local/python3.11.15/bin/pip3.11 /usr/local/python3.11.15/bin/pip3 && \
    ln -sf /usr/local/python3.11.15/bin/python3 /usr/local/python3.11.15/bin/python && \
    ln -sf /usr/local/python3.11.15/bin/pip3 /usr/local/python3.11.15/bin/pip && \
    rm -rf /tmp/* 

# Install Python 3.12.13
RUN curl -fsSL https://repo.huaweicloud.com/python/3.12.13/Python-3.12.13.tgz -o /tmp/Python-3.12.13.tgz && \
    tar -xf /tmp/Python-3.12.13.tgz -C /tmp && \
    cd /tmp/Python-3.12.13 && \
    mkdir -p /usr/local/python3.12.13/lib && \
    ./configure --enable-shared --enable-shared LDFLAGS="-Wl,-rpath /usr/local/python3.12.13/lib" --prefix=/usr/local/python3.12.13 && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/python3.12.13/bin/python3.12 /usr/local/python3.12.13/bin/python3 && \
    ln -sf /usr/local/python3.12.13/bin/pip3.12 /usr/local/python3.12.13/bin/pip3 && \
    ln -sf /usr/local/python3.12.13/bin/python3 /usr/local/python3.12.13/bin/python && \
    ln -sf /usr/local/python3.12.13/bin/pip3 /usr/local/python3.12.13/bin/pip && \
    rm -rf /tmp/*

# Install Python 3.13.13
RUN curl -fsSL https://repo.huaweicloud.com/python/3.13.13/Python-3.13.13.tgz -o /tmp/Python-3.13.13.tgz && \
    tar -xf /tmp/Python-3.13.13.tgz -C /tmp && \
    cd /tmp/Python-3.13.13 && \
    mkdir -p /usr/local/python3.13.13/lib && \
    ./configure --enable-shared --enable-shared LDFLAGS="-Wl,-rpath /usr/local/python3.13.13/lib" --prefix=/usr/local/python3.13.13 && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/python3.13.13/bin/python3.13 /usr/local/python3.13.13/bin/python3 && \
    ln -sf /usr/local/python3.13.13/bin/pip3.13 /usr/local/python3.13.13/bin/pip3 && \
    ln -sf /usr/local/python3.13.13/bin/python3 /usr/local/python3.13.13/bin/python && \
    ln -sf /usr/local/python3.13.13/bin/pip3 /usr/local/python3.13.13/bin/pip && \
    rm -rf /tmp/*


RUN \
    pythons = "python3.9 python3.10 python3.11 python3.12 python3.13";  \
    for python in $pythons;do \
        ${python} -m pip3 install --no-cache-dir \
        pyyaml \
        setuptools \
        pybind11 \
        ninja \
        cmake \
        wheel \
        tabulate \
        auditwheel \
        einops \
        expecttest \
        lit \
        ninja \
        patchelf \
        pybind11 \
        pytest \
        pytest-xdist \
        twine \
        wheel \
        pandas \
        setuptools \
        nanobind>=2.4 \
        --index-url ${PIP_SOURCE}; \
    done

# Stage 2: Make image
FROM ubuntu:22.04 AS official-ubuntu

COPY --from=python-installer /usr/local/python3.9.15 /usr/local/python3.9.15
COPY --from=python-installer /usr/local/python3.10.15 /usr/local/python3.10.15
COPY --from=python-installer /usr/local/python3.11.15 /usr/local/python3.11.15
COPY --from=python-installer /usr/local/python3.12.13 /usr/local/python3.12.13
COPY --from=python-installer /usr/local/python3.13.13 /usr/local/python3.13.13

RUN apt-get update \
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
        clang-15  \
        ccache \
        lld-15 \
        openssh-server \
        openssh-client \
        libssl-dev \
        dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/* \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++


