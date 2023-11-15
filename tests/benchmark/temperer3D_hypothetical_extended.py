import numpy as np
import pickle
import itertools
from warmth3d.fixed_mesh_model import UniformNodeGridFixedSizeMeshModel
from warmth3d.Helpers import NodeGrid
from warmth3d.resqpy_helpers import read_mesh_resqml

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(msg=""):
    import time
    if 'startTime_for_tictoc' in globals():
        delta = time.time() - startTime_for_tictoc
        print (msg+": Elapsed time is " + str(delta) + " seconds.")
        return delta
    else:
        print ("Toc: start time not set")

def nodeFromDataArray(nodeGrid, node_data_array, nodeid_x, nodeid_y):
    origin_x = nodeGrid.origin_x
    origin_y = nodeGrid.origin_y
    stepx = nodeGrid.step_x
    stepy = nodeGrid.step_y
    class GenericObject(object):
        pass
    node = GenericObject
    node.X, node.Y = node_data_array[0:2]
    node.subsidence = node_data_array[3]
    node.crust_ls = node_data_array[4]
    node.lith_ls = node_data_array[5]
    node.sed = node_data_array[2]
    node.X, node.Y = (origin_x + stepx*nodeid_x, origin_y + stepy*nodeid_y)
    node.sed_thickness_ls =  node.sed[-1,1,:] - node.sed[0,0,:]
    node.sediments = node_data_array[6]
    if len(node_data_array)>7:
        node.beta = node_data_array[7]
    if len(node_data_array)>9:
        node.depth_out = node_data_array[8].copy()
        node.temperature_out = node_data_array[9].copy()
    if len(node_data_array)>10:
        node.parameters = node_data_array[10]
    return node

def interpolateNode(interpolationNodes, interpolationWeights):
    assert len(interpolationNodes)>0
    assert len(interpolationNodes)==len(interpolationWeights)
    wsum = np.sum(np.array(interpolationWeights))
    iWeightNorm = [ w/wsum for w in interpolationWeights]
    class GenericObject(object):
        pass
    node = GenericObject
    node.X = np.sum( np.array( [node.X * w for node,w in zip(interpolationNodes,iWeightNorm)] ) ) 
    node.Y = np.sum( np.array( [node.Y * w for node,w in zip(interpolationNodes,iWeightNorm)] ) ) 
    node.subsidence = np.sum( np.array( [node.subsidence * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.crust_ls = np.sum( np.array( [node.crust_ls * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.lith_ls = np.sum( np.array( [node.lith_ls * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.beta = np.sum( np.array( [node.beta * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    # node.kAsth = np.sum( np.array( [node.kAsth * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    # node.kLith = np.sum( np.array( [node.kLith * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.sediments = interpolationNodes[0].sediments.copy()
    return node

def loadOrInterpolateNode(nodeGrid, nodeDirectoryPrefix, nodeid_x, nodeid_y):
    # full_simulation = (nodeid_x % 10 == 0) and (nodeid_y % 10 == 0)
    full_simulation = False
    if not full_simulation:
        #
        # TODO: replace the hard-coded min/max node ranges
        nodeid_x_expanded = max(min(nodeid_x,96),2) 
        nodeid_y_expanded = max(min(nodeid_y,96),2)

        sedFileName = nodeDirectoryPrefix+"node-sed-"+str(nodeid_x_expanded)+"-"+str(nodeid_y_expanded)+".pkl"
        try:
            sed_data_array = pickle.load( open(sedFileName,"rb") )
            # X, Y = sed_data_array[0], sed_data_array[1]
            sed = sed_data_array[2]
        except FileNotFoundError as e:
            print("1D sediment solution not found: ", sedFileName)
            breakpoint()
            return
        # node = interpolateNode(interpolationNodes, interpolationWeight)
        class GenericObject(object):
            pass        
        node = GenericObject
        node.subsidence = np.zeros(sed.shape[2])
        node.crust_ls = np.zeros(sed.shape[2])
        node.lith_ls = np.zeros(sed.shape[2])
        node.beta = 1

        props_shale = { 
            'k_cond': 1.64,
            'rhp': 2.034,
            'phi': 0.7,
            'decay': 0.83,
            'solidus': 2700,
        }
        props_sand = { 
            'k_cond': 3.95,
            'rhp': 0.703,
            'phi': 0.41,
            'decay': 0.31,
            'solidus': 2720,
        }
        props_salt = { 
            'k_cond': 6.5,
            'rhp': 0.012,
            'phi': 0.01,
            'decay': 0.0001,
            'solidus': 2200,
        }

        num_sed = len(node.sediments['k_cond'])
        props = [props_sand, props_sand]
        # props = [props_sand, props_sand, props_salt]

        for key in ['k_cond', 'rhp', 'phi', 'decay', 'solidus']:
            node.sediments[key] = [props[sed][key] for sed in range(num_sed) ]

        node.sed = sed
        node.sed_thickness_ls =  node.sed[-1,1,:] - node.sed[0,0,:]
        origin_x = nodeGrid.origin_x
        origin_y = nodeGrid.origin_y
        stepx = nodeGrid.step_x
        stepy = nodeGrid.step_y
        node.X = origin_x + stepx*nodeid_x
        node.Y = origin_y + stepy*nodeid_y

        expandEdgeElements = 1200
        if (expandEdgeElements>0):
            if (nodeid_x== nodeGrid.start_index_x):
                node.X = node.X - 6*expandEdgeElements
            if (nodeid_x== nodeGrid.start_index_x+1):
                node.X = node.X - 3*expandEdgeElements
            if (nodeid_x== nodeGrid.start_index_x+2):
                node.X = node.X - expandEdgeElements
            if (nodeid_x== nodeGrid.start_index_x+nodeGrid.num_nodes_x-1):
                node.X = node.X + 6*expandEdgeElements
            if (nodeid_x== nodeGrid.start_index_x+nodeGrid.num_nodes_x-2):
                node.X = node.X + 3*expandEdgeElements
            if (nodeid_x== nodeGrid.start_index_x+nodeGrid.num_nodes_x-3):
                node.X = node.X + expandEdgeElements
            if (nodeid_y== nodeGrid.start_index_y):
                node.Y = node.Y - 6*expandEdgeElements
            if (nodeid_y== nodeGrid.start_index_y+1):
                node.Y = node.Y - 3*expandEdgeElements
            if (nodeid_y== nodeGrid.start_index_y+2):
                node.Y = node.Y - expandEdgeElements
            if (nodeid_y== nodeGrid.start_index_y+nodeGrid.num_nodes_y-1):
                node.Y = node.Y + 6*expandEdgeElements
            if (nodeid_y== nodeGrid.start_index_y+nodeGrid.num_nodes_y-2):
                node.Y = node.Y + 3*expandEdgeElements
            if (nodeid_y== nodeGrid.start_index_y+nodeGrid.num_nodes_y-3):
                node.Y = node.Y + expandEdgeElements

        return node
    else:
        nodeFileName = nodeDirectoryPrefix+"node-slim-"+str(nodeid_x)+"-"+str(nodeid_y)+".pkl"
        try:
            node_data_array = pickle.load( open(nodeFileName,"rb") )
        except FileNotFoundError as e:
            print("1D Node solution not found: ", nodeFileName)
            breakpoint()
            return
        node = nodeFromDataArray(nodeGrid, node_data_array, nodeid_x, nodeid_y)
        return node

def run( nodeGrid, run_simulation=True, start_time=182, end_time=0, out_dir = "out-hypothetical/"):
    nodes = []
    modelNamePrefix = ng.modelNamePrefix

    cols = ng.num_nodes_x
    rows = ng.num_nodes_y
    ind_step = ng.step_index
    start_index_x = ng.start_index_x
    start_index_y = ng.start_index_y    

    print("PING A")
    tic()

    nodeFileName = nodeGrid.nodeDirectoryPrefix+"nodeX.pkl"
    # node_data_array = pickle.load( open(nodeFileName,"rb") )
    # nodeX = nodeFromDataArray(nodeGrid, node_data_array, nodeid_x, nodeid_y)
    nodeX = pickle.load( open(nodeFileName,"rb") )

    for j in range(rows):
        for i in range(cols):
            nodeid_x = start_index_x + i * ind_step
            nodeid_y = start_index_y + j * ind_step
 
            node = loadOrInterpolateNode(nodeGrid , nodeGrid.nodeDirectoryPrefix, nodeid_x, nodeid_y)
            node.sediments = nodeX.sediments.copy()
            nodes.append(node)

    print("PING B")
    toc("load nodes")

    #
    # shift sediments to lower base
    #
    for ti in range(nodes[0].sed_thickness_ls.shape[0]):
        maxes = [ np.amax(n.sed[:,:,ti]) for n in nodes ]
        globalmax = np.amax(np.array(maxes))
        for i,n in enumerate(nodes):
            diff = (globalmax - np.amax(n.sed[:,:,ti]))
            n.sed[:,:,ti] = n.sed[:,:,ti] + diff
            n.subsidence[ti] = diff


    print("=====",nodes[0].subsidence)
    characteristic_length = ind_step * ng.step_x

    nums = 4
    dt = 314712e8 / nums

    mms2 = []
    mms_tti = []

    tti = 0

    writeout = True

    if not run_simulation:
        return
    time_solve = 0.0    
    
    for tti in range(start_time, end_time-1,-1): 
        rebuild_mesh = (tti==start_time)
        
        build_tti = 0     # use the full stack at every time step
        # build_tti = tti   # do sedimentation: update stack positions at time steps

        if rebuild_mesh:
            print("Rebuild/reload mesh at tti=", tti)          
            mm2 = UniformNodeGridFixedSizeMeshModel(nodeGrid, nodes, modelName=modelNamePrefix+str(tti), sedimentsOnly=True)
            mm2.buildMesh(build_tti)
        else:
            print("Re-generating mesh vertices at tti=", tti)
            mm2.updateMesh(build_tti)

        mm2.baseFluxMagnitude = 0.04 if (tti>50) else 0.04
        print("===",tti,"  ",mm2.baseFluxMagnitude,"=========== ")
        if ( len(mms2) == 0):
            tic()
            mm2.setupSolverAndSolve(no_steps=nums, time_step = dt, skip_setup=False)   
            time_solve = time_solve + toc(msg="setup solver and solve")
        else:    
            tic()
            # mm2.setupSolverAndSolve( initial_state_model=mms2[-1], no_steps=nums, time_step=dt, skip_setup=(not rebuild_mesh))
            mm2.setupSolverAndSolve( no_steps=nums, time_step=dt, skip_setup=(not rebuild_mesh))
            time_solve = time_solve + toc(msg="setup solver and solve")
        # subvolumes.append(mm2.evaluateVolumes())
        if (writeout):
            tic()
            mm2.writeLayerIDFunction(out_dir+"LayerID-"+str(tti)+".xdmf", tti=tti)
            mm2.writeTemperatureFunction(out_dir+"Temperature-"+str(tti)+".xdmf", tti=tti)
            toc(msg="write function")
            if (tti==0):
                fn = out_dir+"mesh-pos-"+str(tti)+".npy"
                np.save(fn, mm2.mesh.geometry.x)
                fn = out_dir+"T-value-"+str(tti)+".npy"
                np.save(fn, mm2.uh.x.array[:])            
                fn = out_dir+"cell-pos-"+str(tti)+".npy"
                np.save(fn, mm2.getCellMidpoints())
                fn = out_dir+"k-value-"+str(tti)+".npy"
                np.save(fn, mm2.thermalCond.x.array[:])
                fn = out_dir+"phi-value-"+str(tti)+".npy"
                np.save(fn, mm2.mean_porosity.x.array[:])

        mms2.append(mm2)
        mms_tti.append(tti)
    print("total time solve: " , time_solve)
    EPCfilename = mm2.write_hexa_mesh_resqml("temp/")
    print("RESQML model written to: " , EPCfilename)
    # read_mesh_resqml(EPCfilename, meshTitle="hexamesh")  # test reading of the .epc file


#
# NOTE: to compute the required 1D node solutions, you must first run  warmth3d/parallel-1Dsed.py using the same NodeGrid parameters as below!
#

# ng = NodeGrid(0, 0, 97, 97, 2, 2, 1, 100, 100)
ng = NodeGrid(0, 0, 97+27+25, 97+27+25, -25, -25, 1, 100, 100)   # pad/extend the system by 25 nodes in each direction

ng.modelNamePrefix = "hypo-all-nodes-"
ng.nodeDirectoryPrefix = "nodes-hypothetical/"

import os
for output_dir in ["out-hypothetical/", "mesh", "temp"]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

run( ng, run_simulation=True, start_time=2, end_time=0, out_dir = "out-hypothetical/")

