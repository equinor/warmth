import numpy as np
import pickle
import itertools
from warmth3d.fixed_mesh_model import UniformNodeGridFixedSizeMeshModel
from warmth3d.Helpers import NodeGrid
from warmth3d.resqpy_helpers import read_mesh_resqml

#
# a map for how many meters were eroded between 100-120 My
# a map for how many meters were eroded between 120-140 My  => constant 50m
#

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
    node.kAsth = np.sum( np.array( [node.kAsth * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.kLith = np.sum( np.array( [node.kLith * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.sediments = interpolationNodes[0].sediments.copy()
    return node

def loadOrInterpolateNode(nodeGrid, nodeDirectoryPrefix, nodeid_x, nodeid_y):
    if (nodeid_x % 10 > 0) or (nodeid_y % 10 > 0):
        nidx0 = nodeid_x - (nodeid_x % 10)
        nidy0 = nodeid_y - (nodeid_y % 10)
        interpolationNodes = []
        interpolationWeight = []
        sedFileName = nodeDirectoryPrefix+"node-sed-"+str(nodeid_x)+"-"+str(nodeid_y)+".pkl"
        try:
            sed_data_array = pickle.load( open(sedFileName,"rb") )
            X, Y = sed_data_array[0], sed_data_array[1]
            sed = sed_data_array[2]
        except FileNotFoundError as e:
            print("1D sediment solution not found: ", sedFileName)
            breakpoint()
            return
        ddx = [0,10] if (nodeid_x % 10 > 0) else [0]
        ddy = [0,10] if (nodeid_y % 10 > 0) else [0]
        for dx,dy in list(itertools.product(ddx, ddy)):
            nxpos, nypos = nidx0+dx, nidy0+dy
            nodeFileName = nodeDirectoryPrefix+"node-slim-"+str(nxpos)+"-"+str(nypos)+".pkl"
            try:
                node_data_array = pickle.load( open(nodeFileName,"rb") )
                interpolationNodes.append(nodeFromDataArray(nodeGrid,node_data_array,nxpos,nypos))
                dist = np.sqrt((nodeid_x-nxpos)**2 + (nodeid_y-nypos)**2)
                interpolationWeight.append(1/dist)
            except FileNotFoundError as e:
                print("1D Node solution not found: ", nodeFileName)
                breakpoint()
                return
        node = interpolateNode(interpolationNodes, interpolationWeight)
        node.sed = sed
        node.sed_thickness_ls =  node.sed[-1,1,:] - node.sed[0,0,:]
        origin_x = nodeGrid.origin_x
        origin_y = nodeGrid.origin_y
        stepx = nodeGrid.step_x
        stepy = nodeGrid.step_y
        node.X, node.Y = (origin_x + stepx*nodeid_x, origin_y + stepy*nodeid_y)
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

def run( nodeGrid, run_simulation=True, start_time=182, end_time=0, out_dir = "out-mapA/"):
    nodes = []
    modelNamePrefix = ng.modelNamePrefix

    cols = ng.num_nodes_x
    rows = ng.num_nodes_y
    ind_step = ng.step_index
    start_index_x = ng.start_index_x
    start_index_y = ng.start_index_y    

    convexHullEdges = []

    for j in range(rows):
        for i in range(cols):
            nodeid_x = start_index_x + i * ind_step
            nodeid_y = start_index_y + j * ind_step
 
            node = loadOrInterpolateNode(nodeGrid , nodeGrid.nodeDirectoryPrefix, nodeid_x, nodeid_y)
            nodes.append(node)

    characteristic_length = ind_step * ng.step_x

    nums = 4
    dt = 314712e8 / nums

    mms2 = []
    mms_tti = []

    tti = 0
    subvolumes = []
   
    writeout = True

    if not run_simulation:
        return
    time_solve = 0.0    
    
    for tti in range(start_time, end_time-1,-1): 
        rebuild_mesh = (tti==start_time)
        if rebuild_mesh:
            print("Rebuild/reload mesh at tti=", tti)          
            mm2 = UniformNodeGridFixedSizeMeshModel(nodeGrid, nodes, modelName=modelNamePrefix+str(tti))
            mm2.buildMesh(tti)
        else:
            print("Re-generating mesh vertices at tti=", tti)
            mm2.updateMesh(tti)

        print("===",tti,"=========== ")
        if ( len(mms2) == 0):
            tic()
            mm2.setupSolverAndSolve(no_steps=40, time_step = 314712e8 * 2e2, skip_setup=False)   
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
            # mm2.writeOutputFunctions(out_dir+"test4-"+str(tti)+".xdmf", tti=tti)
            toc(msg="write function")
        mms2.append(mm2)
        mms_tti.append(tti)
    print("total time solve: " , time_solve)
    EPCfilename = mm2.write_hexa_mesh_resqml("temp/")
    print("RESQML model written to: " , EPCfilename)
    read_mesh_resqml(EPCfilename)  # test reading of the .epc file



#
# NOTE: to compute the required 1D node solutions, you must first run  warmth3d/parallel-1Dsed.py using the same NodeGrid parameters as below!
#
ng = NodeGrid(150, 0, 485, 548, 500, 1000, 5, 1000, 1000)
ng = NodeGrid(0, 0, 11, 11, 500, 1000, 5, 100, 100)
# ng = NodeGrid(25400, 41600, 31, 31, 100, 100, 2, 100, 100)

ng.modelNamePrefix = "mapB-121-nodes-"
ng.nodeDirectoryPrefix = "nodes-mapB/"

import os
for output_dir in ["out-mapB/", "mesh", "temp"]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# import cProfile
# cProfile.run('run( ng, run_simulation=True, start_time=182, end_time=0, out_dir = "out-mapA/")')

run( ng, run_simulation=True, start_time=170, end_time=0, out_dir = "out-mapB/")


