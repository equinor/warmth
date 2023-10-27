# Background theory


3D simulations in warmth are under active development. Currently, yhey are based on a uniform reactangular grid of 1D nodes (defined in a NodeGrid data structure). The sediment inputs, horizons are present day, are provided as .gri files, which are read into a SedimentStack class. 

### Pre-requisite: grid of 1D node simulation
The complete 1D warmth simulation is run for some of the 1D nodes. For other 1D nodes the subsidence, beta factor, and crustal thickness are interpolated. The 1D simulations can be run using the script [parallel-1Dsed.py](warmth3D/parallel-1Dsed.py).  Results for each node are pickled to a separate file (this is to be improved!).

### 3D heat equation simulation using dolfinx
The 3D simulation performs a series of heat equation solves, regularly updating the mesh positions from the 1D nodes. The equations are solved using the PETSc solver from the dolfinx package (part of the FeNiCs project). The compute mesh is built by defining hexahedra for every rectangle of 1D nodes and for every layer (i.e. each sediment, the crust, the lithosphere, and the aesthenosphere), which are then subdivided into tetrahedra. 

The dolfinx model building and solving is managed by the class [UniformNodeGridFixedSizeMeshModel](warmth3D/fixed_mesh_model.py).  The use of this class is demonstrated in [warmth3D_mapA_example.py](tests/warmth3D_mapA_example.py). Note that the NodeGrid class definition in this script should match the definition used in [parallel-1Dsed.py](warmth3D/parallel-1Dsed.py) to compute the 1D solutions. This script writes the results (mesh positions and function values) at every 1M years in xdmf format for visualization in ParaView. 

### RESQML output
The test script [warmth3D_mapA_example.py](tests/warmth3D_mapA_example.py) further demonstrates writing the unstructured grid (with properties) in RESQML format, as a pair of .epc and .h5 files.  The RESQML I/O functions are in a separate file, [resqpy_helpers.py](warmth3D/resqpy_helpers.py), and require a modified version of the resqpy library.  To visualise RESQML data in ParaView, a 3rd-party plug-in can be installed, see [fespp](https://github.com/F2I-Consulting/fespp). 

### 3D dependencies
The dolfinx package is Linux-only(?) and has to be compiled from source or installed using apt-get.  The resqpy dependency can be installed with pip, but, for now, some writing of properties on unstructured grids requires a change in resqpy that is not yet merged.  The other dependencies xtgeo and meshio can be installed using pip (requirements file is to be added).

