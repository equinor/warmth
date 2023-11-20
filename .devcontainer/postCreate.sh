poetry config virtualenvs.create false
poetry env use system
poetry install --with dev --no-interaction
#poetry run pip install petsc4py==3.20.0 mpi4py==3.1.5 https://github.com/FEniCS/ufl/archive/refs/tags/2023.2.0.zip https://github.com/FEniCS/ffcx/archive/v0.7.0.zip https://github.com/FEniCS/basix/archive/v0.7.0.zip