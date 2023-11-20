FROM dolfinx/dolfinx:v0.6.0
# FROM mcr.microsoft.com/devcontainers/base:jammy 
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

# # Python and poetry installation
# # USER $USER
# # ARG HOME="/home/$USER"

# # Don't change. Dolfinx in the base image is only compiled with 3.10
# #ARG PYTHON_VERSION=3.10.6

# # Use poetry pyenv
# #ENV PYENV_ROOT="${HOME}/.pyenv"
# #ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:/root/.local/bin:$PATH"
RUN useradd  -rm -d /home/vscode -s /bin/bash -g root -G sudo -u 1000 vscode -p ""
USER vscode
WORKDIR /home/vscode
ENV HOME=/home/vscode
ENV PATH="/root/.local/bin:$HOME/.local/bin::$PATH"
RUN echo "done 0" \
#     # && curl https://pyenv.run | bash \
#     # && echo "done 1" \
#     # && pyenv install ${PYTHON_VERSION} \
#     # && echo "done 2" \
#     # && pyenv global ${PYTHON_VERSION} \
#     # && echo "done 3" \
     && curl -sSL https://install.python-poetry.org | python3 - 
