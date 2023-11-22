FROM dolfinx/dolfinx:v0.6.0

LABEL org.opencontainers.image.source https://github.com/equinor/warmth
WORKDIR /home/warmth

ENTRYPOINT pip install . pytest==7.4.2 pytest-cov==4.1.0 && pytest --cov-report=term-missing --cov=warmth tests | tee pytest-coverage.txt 
