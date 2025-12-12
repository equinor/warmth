FROM dolfinx/dolfinx:v0.10.0

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y curl

ENV PATH="/root/.local/bin:$HOME/.local/bin::$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - 
