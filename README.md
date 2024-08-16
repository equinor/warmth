# Warmth
## Forward modeling of thermal evolution through geological time

![Build Status](https://github.com/equinor/warmth/actions/workflows/python-test.yml/badge.svg?branch=main)
![Build Status](https://github.com/equinor/warmth/actions/workflows/docs.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/equinor/warmth/graph/badge.svg?token=A9LWISA7OI)](https://codecov.io/gh/equinor/warmth)
![Python](https://img.shields.io/pypi/pyversions/warmth)
[![PyPI version](https://badge.fury.io/py/warmth.svg)](https://badge.fury.io/py/wamrth)
![License](https://img.shields.io/github/license/equinor/warmth)

[Documentation](https://equinor.github.io/warmth/)

warmth is a python package used for modeling thermal evolution based on McKenzie's type basin extension. It can be use for:

- Finding beta factor
- Calculating subsidence and thermal history
- Basement heat flow through time

## Features
- Multi-1D simulation
- Full 3D simulation with dolfinx
- Build model from either: 
    - Python objects
    - [XTGeo](https://github.com/equinor/xtgeo/) supported surface formats
- Multi-rift phase support
- Ensemble models with ERT https://github.com/equinor/ert

## Installation

Until it is available on pypi, you will need to clone the repo

```
git clone git@github.com:equinor/warmth.git
pip install .
```
For a specific release
```
git clone git@github.com:equinor/warmth.git --branch <VERSION>
pip install .
```

For full 3D simulation, dolfinx is required.

See https://docs.fenicsproject.org/dolfinx/main/python/installation.html for installation instructions.

## Tests
Requires dolfinx v0.6.0

```mpirun -n 2 python3 -m pytest --with-mpi --cov-report=term-missing --cov-report=xml --cov=warmth tests```
