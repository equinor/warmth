import warmth
from pathlib import Path

maps_dir = Path("../../docs/notebooks/data/")
model = warmth.Model()

inputs = model.builder.input_horizons_template

#Add surface grids to the table. You can use other method as well
inputs.loc[0]=[0,"0.gri",None,"Onlap"]
inputs.loc[1]=[66,"66.gri",None,"Onlap"]
inputs.loc[2]=[100,"100.gri",None,"Onlap"]
inputs.loc[3]=[163,"163.gri",None,"Erosive"]
inputs.loc[4]=[168,"168.gri",None,"Erosive"]
inputs.loc[5]=[170,"170.gri",None,"Onlap"]
inputs.loc[6]=[182,"182.gri",None,"Erosive"]
model.builder.input_horizons=inputs


inc = 2000
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
    # i.crustRHP = 0.0
    # i.sediments['rhp']=[0,0,0,0,0,0]

model.simulator.simulate_every = 1

#
# set 1D simulation parameters to be most similar to those in the (later) 3D simulation, for better comparison
#
model.parameters.HPdcr = 1e32   # "infinite" decay of crustal HP
model.parameters.bflux = False 
model.parameters.tetha = 0
model.parameters.alphav = 0

model.simulator.run(save=True,purge=True)



# interpolate and extrapolate the missing nodes
# find nearby nodes from the array indexer_full_sim, which is sorted by x-index
import itertools
from subsheat3D.fixed_mesh_model import interpolateNode
for ni in range(len(model.builder.nodes)):
    for nj in range(len(model.builder.nodes[ni])):
        if model.builder.nodes[ni][nj] is False:
            closest_x_up = []
            for j in range(ni,len(model.builder.nodes[nj])):
                matching_x = [ i[0] for i in model.builder.indexer_full_sim if i[0]==j ]
                closest_x_up = closest_x_up + list(set(matching_x))
                if len(matching_x)>0:
                    break
            closest_x_down = []
            for j in range(ni-1,-1,-1):
                matching_x = [ i[0] for i in model.builder.indexer_full_sim if i[0]==j ]
                closest_x_down = closest_x_down + list(set(matching_x))
                if len(matching_x)>0:
                    break
            closest_y_up = []
            for j in range(nj,len(model.builder.nodes[ni])):
                matching_y = [ i[1] for i in model.builder.indexer_full_sim if (i[1]==j and ((i[0] in closest_x_up) or i[0] in closest_x_down)) ]
                closest_y_up = closest_y_up + list(set(matching_y))
                if len(matching_y)>0:
                    break
            closest_y_down = []
            for j in range(nj-1,-1,-1):
                matching_y = [ i[1] for i in model.builder.indexer_full_sim if (i[1]==j and (i[0] in closest_x_up or i[0] in closest_x_down) ) ]
                closest_y_down = closest_y_down + list(set(matching_y))
                if len(matching_y)>0:
                    break

            interpolationNodes = [  model.builder.nodes[i[0]][i[1]] for i in itertools.product(closest_x_up+closest_x_down, closest_y_up+closest_y_down)  ]
            interpolationNodes = [nn for nn in interpolationNodes if nn is not False]
            node = interpolateNode(interpolationNodes)
            node.X, node.Y = model.builder.grid.location_grid[ni,nj,:]
            model.builder.nodes[ni][nj] = node
        else:
            node = interpolateNode([model.builder.nodes[ni][nj]])  # "interpolate" the node from itself to make sure the same member variables exist at the end
            model.builder.nodes[ni][nj] = node

from warmth.mesh_model import run
import os
try:
    os.mkdir('mesh')
except FileExistsError:
    pass
try:
    os.mkdir('temp')
except FileExistsError:
    pass
mm2 = run(model,start_time=model.parameters.time_start,end_time=0)

import numpy as np
hx = model.builder.grid.num_nodes_x // 2
hy = model.builder.grid.num_nodes_y // 2
# hx = model.builder.grid.num_nodes_x - 1 - pad
# hy = model.builder.grid.num_nodes_y - 1 - pad

nn = model.builder.nodes[hy][hx]
dd = nn._depth_out[:,0]

node_ind = hy*model.builder.grid.num_nodes_x + hx
v_per_n = int(mm2.mesh_vertices.shape[0]/(model.builder.grid.num_nodes_y*model.builder.grid.num_nodes_x))

temp_1d = np.nan_to_num(nn.temperature_out[:,0], nan=5.0)
temp_3d_ind = np.array([ np.where([mm2.mesh_reindex==i])[1][0] for i in range(node_ind*v_per_n, (node_ind+1)*v_per_n) ] )
dd_mesh = mm2.mesh.geometry.x[temp_3d_ind,2]
temp_3d_mesh = mm2.u_n.x.array[temp_3d_ind]

temp_1d_at_mesh_pos = np.interp(dd_mesh, dd, temp_1d)
dd_subset = np.where(dd_mesh<5000)

max_abs_error = np.amax(np.abs(temp_1d_at_mesh_pos-temp_3d_mesh))
max_abs_error_shallow = np.amax(np.abs(temp_1d_at_mesh_pos[dd_subset]-temp_3d_mesh[dd_subset]))
print(f'Max. absolute error in temperature at 3D mesh vertex positions: {max_abs_error}')
print(f'Max. absolute error at depths < 5000m: {max_abs_error_shallow}')

assert (max_abs_error<25.0), "Temperature difference between 3D and 1D simulations is >25"
assert (max_abs_error_shallow<5.0),  "Temperature difference between 3D and 1D simulations is >5 in the sediments."




