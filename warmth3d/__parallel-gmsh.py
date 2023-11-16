import numpy as np
import pickle
import pandas as pd
# import subsheat
# from subsheat.data import haq87
from warmth3d.SedimentStack import SedimentStack
from warmth3d.deforming_model import MultiNodeModel

import os

from threading import Thread
from multiprocessing import Pool
from dataclasses import dataclass
import itertools

from warmth3d.Helpers import NodeGrid

# logger = get_logger(__name__)


class GMSHworker(object):
    def __init__(self, nodeGrid, tti):
        self.tti = tti
        self.modelNamePrefix = nodeGrid.modelNamePrefix
        self.nodeDirectoryPrefix = nodeGrid.nodeDirectoryPrefix
        self.nodeGrid = nodeGrid

    def run(self):
        nodes = []
        triangles = []

        rows = self.nodeGrid.num_nodes_x
        cols = self.nodeGrid.num_nodes_y
        origin_x = self.nodeGrid.origin_x
        origin_y = self.nodeGrid.origin_y
        stepx = self.nodeGrid.step_x
        stepy = self.nodeGrid.step_y
        ind_step = self.nodeGrid.step_index
        start_index_x = self.nodeGrid.start_index_x
        start_index_y = self.nodeGrid.start_index_y    

        convexHullEdges = []

        for j in range(cols):
            for i in range(rows):
                nodeid_x = start_index_x + i * ind_step
                nodeid_y = start_index_y + j * ind_step
                # nodeFileName = self.nodeDirectoryPrefix+"node-"+str(nodeid_x)+"-"+str(nodeid_y)+".pkl"
                nodeFileName = self.nodeDirectoryPrefix+"node-slim-"+str(nodeid_x)+"-"+str(nodeid_y)+".pkl"
                # dataArray = [node.X,node.Y,node.sed.copy(), node.subsidence.copy(), node.crust_ls.copy(), node.lith_ls.copy()]
                # pickle.dump(dataArray, open(self.sedStack.nodeDirectoryPrefix+"node-slim-"+str(nodei)+"-"+str(nodej)+".pkl","wb") )                    
                try:
                    # node = pickle.load( open(nodeFileName,"rb") )
                    node_data_array = pickle.load( open(nodeFileName,"rb") )
                except FileNotFoundError as e:
                    print("1D Node solution not found: ", nodeFileName)
                    breakpoint()
                    return
                class GenericObject(object):
                    pass
                node = GenericObject
                node.X = node_data_array[0]
                node.Y = node_data_array[1]
                node.subsidence = node_data_array[3]
                node.crust_ls = node_data_array[4]
                node.lith_ls = node_data_array[5]
                node.sed = node_data_array[2]
                node.X, node.Y = (origin_x + stepx*nodeid_x, origin_y + stepy*nodeid_y)
                node.sed_thickness_ls =  node.sed[-1,1,:] - node.sed[0,0,:]
                # 
                # temporary; need to save sediments in node_slim
                hack_sediments = pd.DataFrame(columns=['solidus', 'k_cond'])
                hack_sediments['solidus'] = [2720.0, 2708.0, 2618.0, 2720.0, 2708.0, 2618.0, 2720.0, 2708.0, 2618.0, 2720.0, 2708.0, 2618.0 ]
                hack_sediments['k_cond'] = [ 1.500, 1.538462, 1.904762, 1.500, 1.538462, 1.904762, 1.500, 1.538462, 1.904762, 1.500, 1.538462, 1.904762 ]
                node.sediments = hack_sediments

                nodes.append(node)

        for j in range(cols-1):
            for i in range(rows-1):
                lin_ind = i + j*rows
                triangles.append( [lin_ind+0, lin_ind+1, lin_ind+rows] )
                triangles.append( [lin_ind+1, lin_ind+rows+1, lin_ind+rows] )

        for j in range(cols-1):
            lin_ind_0 = 0 + j*rows
            lin_ind_1 = 0 + (j+1)*rows
            convexHullEdges.append( [lin_ind_0, lin_ind_1] )
            lin_ind_0 = (rows-1) + j*rows
            lin_ind_1 = (rows-1) + (j+1)*rows
            convexHullEdges.append( [lin_ind_0, lin_ind_1] )

        for i in range(rows-1):
            lin_ind_0 = i + 0*rows
            lin_ind_1 = i+1 + (0)*rows
            convexHullEdges.append( [lin_ind_0, lin_ind_1] )
            lin_ind_0 = i + (cols-1)*rows
            lin_ind_1 = i+1 + (cols-1)*rows
            convexHullEdges.append( [lin_ind_0, lin_ind_1] )

        characteristic_length = ind_step * stepx
        time_mesh = 0.0
        writeout = True
        tti = self.tti
        pfc = 10
        mm2 = MultiNodeModel(nodes, triangles, convexHullEdges, modelName=self.modelNamePrefix+str(tti))
        mm2.buildVerticesAndFaces(time_index=tti)
        print("=== Meshing ",tti,"=========== " )
        try: 
            mm2.buildMesh(writeToFile=True, loadFromFileIfAvailable=False, meshSizeTop = characteristic_length*pfc)
        except Exception as inst:
            print("exception, ", inst)
            pfc = pfc*0.8
            print("retry pfc, ", pfc)
            mm2.buildMesh(writeToFile=True, loadFromFileIfAvailable=False, meshSizeTop = characteristic_length*pfc)
            print("post exception, ")
        # print("total time meshing time ", tti, time_mesh)

def runWorker(args):
    print("runWorker", args[1])
    w = GMSHworker(args[0], args[1]) # for tti in range(time_start, time_end-1, -1)
    w.run()
    return args[1]

if __name__ == '__main__': 

    class GMSHProcessSpawner(object):
        def __init__(self, nodeGrid, times):
            self.nodeGrid = nodeGrid
            self.times = times
        def run_pool(self):
            with Pool() as pool:
                # call the same function with different data in parallel
                for result in pool.map(runWorker, zip(list(itertools.repeat(self.nodeGrid, len(self.times))), list(self.times ))):
                    # report the value to show progress
                    print("Process finished: ", result)            

    ng = NodeGrid(25400, 41600, 31, 31, 100, 100, 2, 100, 100)
    # ng = NodeGrid(25400, 41600, 7, 9, 106, 122, 2, 100, 100)
    ng.modelNamePrefix = "new_11_31_2_"
    ng.nodeDirectoryPrefix = "nodes-mapA/"
    # print(ng)  

    # times = range(self.time_start, self.time_end-1, -1)
    # remeshTTI = [182,181,169,167,162,99,65]    
    remeshTTI = [182,181,169,167]    
    ws = GMSHProcessSpawner(ng, remeshTTI)
    ws.run_pool()





