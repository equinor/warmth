
from pathlib import Path

import warmth

def test_model_from_grid():  
    maps_dir = Path("./docs/notebooks/data/")

    model = warmth.Model()

    inputs = model.builder.input_horizons_template

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


    # simulate every 10 nodes
    for index in model.builder.grid.indexing_arr:
        if (index[0] % 10 > 0):
            pass
        else:
            if isinstance(model.builder.nodes[index[0]][index[1]],bool) is False:
                model.builder.nodes[index[0]][index[1]] = False
        if (index[1] % 10 > 0):
            pass
        else:
            if isinstance(model.builder.nodes[index[0]][index[1]],bool) is False:
                model.builder.nodes[index[0]][index[1]] = False


    model.parameters.output_path = Path('./simout_grids')
    model.simulator.run(save=False,purge=True,parallel=False,use_mpi=False)

    for i in model.builder.iter_node():
        if i is not False:
            print(i.result.heatflow(0))




