import pytest
import warmth
from pathlib import Path
from warmth.mesh_model import run_3d
import os
import numpy as np
from mpi4py import MPI
import pickle
import time

@pytest.mark.mpi
def test_3d_compare():
    comm = MPI.COMM_WORLD
    inc = 5000
    model_pickled = f"model-out-inc_{inc}.p"
    if comm.rank == 0 and not os.path.isfile(model_pickled):
        global runtime_1D_sim
        st = time.time()
        maps_dir = Path("./docs/notebooks/data/")
        model = warmth.Model()
        model.parameters.output_path = Path('./simout_3d')

        inputs = model.builder.input_horizons_template
    
        #Add surface grids to the table. You can use other method as well
        inputs.loc[0]=[0,"0.gri",None,"Onlap"]
        inputs.loc[1]=[66,"66.gri",None,"Onlap"]
        inputs.loc[2]=[100,"100.gri",None,"Onlap"]
        inputs.loc[3]=[163,"163.gri",None,"Erosive"]
        inputs.loc[6]=[182,"182.gri",None,"Erosive"]
        model.builder.input_horizons=inputs

        model.builder.define_geometry(maps_dir/"0.gri",xinc=inc,yinc=inc,fformat="irap_binary")

        model.builder.extract_nodes(4,maps_dir)

        from warmth.data import haq87
        model.builder.set_eustatic_sea_level(haq87)

        for i in model.builder.iter_node():
            i.rift=[[182,175]]
            #
            # set 1D node parameters to be most similar to those in the (later) 3D simulation, for better comparison
            #
            i.bflux = False
            i.adiab = 0.3e-3


        model.simulator.simulate_every = 2

        #
        # set 1D simulation parameters to be most similar to those in the (later) 3D simulation, for better comparison
        #
        model.parameters.HPdcr = 1e32   # "infinite" decay of crustal HP
        model.parameters.bflux = False 
        model.parameters.tetha = 0
        model.parameters.alphav = 0
        print(f"Using MPI? {comm.size>1}")
        model.simulator.run(save=True, purge=True, parallel=True, use_mpi=(comm.size>1))

        runtime_1D_sim = time.time() - st
        print("Total time 1D simulations:", runtime_1D_sim)

        pickle.dump( model, open( model_pickled, "wb" ) )
        try:
            os.mkdir('mesh')
        except FileExistsError:
            pass
        try:
            os.mkdir('temp')
        except FileExistsError:
            pass

    comm.Barrier()
    model = pickle.load( open( model_pickled, "rb" ) )
    comm.Barrier()
    mm2 = run_3d(model.builder,model.parameters,start_time=model.parameters.time_start,end_time=0, pad_num_nodes=2,writeout=False, base_flux=None)
    
    comm.Barrier()
    if comm.rank == 0:            
        nnx = (model.builder.grid.num_nodes_x+2*mm2.padX) 
        nny = (model.builder.grid.num_nodes_y+2*mm2.padX) 
        hx = nnx // 2
        hy = nny // 2

        nn0 = model.builder.nodes[hy-mm2.padX][hx-mm2.padX]
        
        node_result_path = str(nn0.node_path).replace(".pickle", "_results")
        assert os.path.exists(node_result_path), f"ERROR: Node result file {node_result_path} is missing."

        nn = pickle.load(open(node_result_path,"rb"))
        dd = nn._depth_out[:,0]

        mm2_pos, mm2_temp = mm2.get_node_pos_and_temp(-1)

        node_ind = hy*nnx + hx
        v_per_n = int(mm2_pos.shape[0]/(mm2.num_nodes))

        temp_1d = np.nan_to_num(nn.temperature_out[:,0], nan=5.0)
        
        # temp_3d_ind = np.array([ np.where([mm2.mesh_reindex==i])[1][0] for i in range(node_ind*v_per_n, (node_ind+1)*v_per_n) ] )
        temp_3d_ind = np.array([ i for i in range(node_ind*v_per_n, (node_ind+1)*v_per_n) ] )

        # dd_mesh = mm2.mesh.geometry.x[temp_3d_ind,2] - mm2.sed_diff_z[range(node_ind*v_per_n, (node_ind+1)*v_per_n)]
        dd_mesh = mm2_pos[temp_3d_ind,2] - mm2.sed_diff_z[range(node_ind*v_per_n, (node_ind+1)*v_per_n)]
        # temp_3d_mesh = mm2.u_n.x.array[temp_3d_ind]
        temp_3d_mesh = mm2_temp[temp_3d_ind]

        temp_1d_at_mesh_pos = np.interp(dd_mesh, dd, temp_1d)
        dd_subset = np.where(dd_mesh<5000)

        max_abs_error = np.amax(np.abs(temp_1d_at_mesh_pos-temp_3d_mesh))
        max_abs_error_shallow = np.amax(np.abs(temp_1d_at_mesh_pos[dd_subset]-temp_3d_mesh[dd_subset]))    

        print("Max error overall:", max_abs_error)
        print("Max error <5000m:", max_abs_error_shallow)

    # assert (max_abs_error<25.0), "Temperature difference between 3D and 1D simulations is >25"
    # assert (max_abs_error_shallow<5.0),  "Temperature difference between 3D and 1D simulations is >5 in the sediments."

