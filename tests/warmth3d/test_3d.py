import warmth
from pathlib import Path
from warmth.mesh_model import run_3d
import os
import numpy as np
from mpi4py import MPI
import pickle

def test_3d_compare():
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        maps_dir = Path("./docs/notebooks/data/")
        model = warmth.Model()

        inputs = model.builder.input_horizons_template
    
        #Add surface grids to the table. You can use other method as well
        inputs.loc[0]=[0,"0.gri",None,"Onlap"]
        inputs.loc[1]=[66,"66.gri",None,"Onlap"]
        inputs.loc[2]=[100,"100.gri",None,"Onlap"]
        inputs.loc[3]=[163,"163.gri",None,"Erosive"]
        inputs.loc[6]=[182,"182.gri",None,"Erosive"]
        model.builder.input_horizons=inputs


        inc = 1000
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


        model.simulator.simulate_every = 1

        #
        # set 1D simulation parameters to be most similar to those in the (later) 3D simulation, for better comparison
        #
        model.parameters.HPdcr = 1e32   # "infinite" decay of crustal HP
        model.parameters.bflux = False 
        model.parameters.tetha = 0
        model.parameters.alphav = 0

        model.simulator.run(save=True,purge=True, parallel=False)

        import pickle
        pickle.dump( model, open( "model-out.p", "wb" ) )
        model = pickle.load( open( "model-out.p", "rb" ) )

        try:
            os.mkdir('mesh')
        except FileExistsError:
            pass
        try:
            os.mkdir('temp')
        except FileExistsError:
            pass

    comm.Barrier()
    import pickle
    model = pickle.load( open( "model-out.p", "rb" ) )
    comm.Barrier()
    # if comm.rank == 0:            
    # mm2 = 
    mm2, posarr, Tarr = run_3d(model.builder,model.parameters,start_time=model.parameters.time_start,end_time=0, pad_num_nodes=2,writeout=False, base_flux=None)

    
    nnx = (model.builder.grid.num_nodes_x+2*mm2.padX) 
    nny = (model.builder.grid.num_nodes_y+2*mm2.padX) 
    # hx = model.builder.grid.num_nodes_x // 2
    # hy = model.builder.grid.num_nodes_y // 2
    hx = nnx // 2
    hy = nny // 2

    nn = model.builder.nodes[hy-mm2.padX][hx-mm2.padX]
    dd = nn._depth_out[:,0]

    node_ind = hy*nnx + hx
    # v_per_n = int(mm2.mesh_vertices.shape[0]/(model.builder.grid.num_nodes_y*model.builder.grid.num_nodes_x))
    v_per_n = int(mm2.mesh_vertices.shape[0]/(mm2.num_nodes))

    temp_1d = np.nan_to_num(nn.temperature_out[:,0], nan=5.0)
    temp_3d_ind = np.array([ np.where([mm2.mesh_reindex==i])[1][0] for i in range(node_ind*v_per_n, (node_ind+1)*v_per_n) ] )
    dd_mesh = mm2.mesh.geometry.x[temp_3d_ind,2] - mm2.sed_diff_z[range(node_ind*v_per_n, (node_ind+1)*v_per_n)]
    temp_3d_mesh = mm2.u_n.x.array[temp_3d_ind]

    temp_1d_at_mesh_pos = np.interp(dd_mesh, dd, temp_1d)
    dd_subset = np.where(dd_mesh<5000)

    max_abs_error = np.amax(np.abs(temp_1d_at_mesh_pos-temp_3d_mesh))
    max_abs_error_shallow = np.amax(np.abs(temp_1d_at_mesh_pos[dd_subset]-temp_3d_mesh[dd_subset]))    

    print("Max error overall:", max_abs_error)
    print("Max error <5000m:", max_abs_error_shallow)

    # assert (max_abs_error<25.0), "Temperature difference between 3D and 1D simulations is >25"
    # assert (max_abs_error_shallow<5.0),  "Temperature difference between 3D and 1D simulations is >5 in the sediments."




if __name__ == "__main__":
    test_3d_compare()