FROM dolfinx/dolfinx:v0.10.0

LABEL org.opencontainers.image.source https://github.com/equinor/warmth
WORKDIR /home/warmth

ENTRYPOINT pip install . && pip install pytest==7.4.2 pytest-cov==4.1.0 pytest-mpi==0.6 && mpirun -n 2 python3 -m pytest --with-mpi tests
