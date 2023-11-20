FROM dolfinx/dolfinx:v0.6.0

LABEL org.opencontainers.image.source https://github.com/equinor/warmth
ARG DEBIAN_FRONTEND=noninteractive

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        ca-certificates \
        git \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev

RUN useradd  -rm -d /home/vscode -s /bin/bash -g root -G sudo -u 1000 vscode -p ""
USER 1000
WORKDIR /home/vscode
ENV HOME=/home/vscode
ENV PATH="/root/.local/bin:$HOME/.local/bin::$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - 