import numpy as np
import pickle
from multiprocessing import Pool
import itertools

import warmth
from warmth.data import haq87
from warmth3d.SedimentStack import SedimentStack

from warmth3d.Helpers import NodeGrid, getNodeParameters

# import redis 

# logger = get_logger(__name__)

class NodeWorker(object):
    def __init__(self, sedimentStack, node_ij):
        self.node_ij = node_ij
        self.sedStack = sedimentStack

    def run(self):
        nodei, nodej = self.node_ij

        full_beta_solve = ( (nodei % 10 == 0) and (nodej % 10 == 0))  # only a fraction of the nodes is simulated, the rest is interpolated

        node = self.sedStack.create1DNodeAtIJ(nodei, nodej)
        invalid = np.any(np.isnan(np.array(node.sediments_inputs['top'].to_list())))
        if (invalid):
            return False
        model = warmth.Model()
        baseage = int(node.sediments["baseage"].iloc[-1])
        model.parameters.time_start = baseage
        model.parameters.time_end = 0
        model.parameters.starting_beta = 1.15
        model.builder.all_nodes = [node]
        model.builder.set_eustatic_sea_level(haq87)
        for i in model.builder.all_nodes:
            i.rift = np.array([[baseage, int(node.sediments["baseage"].iloc[-2])]])
        model.parameters.experimental = True
        if (full_beta_solve):
            model.simulator.run()
        else:
            #
            # compute only the sedimentation for this node, the crustal thickness and subsidence have to be interpolated 
            #   when the 3D simulation is set up! 
            #
            fw = warmth.forward_modelling.Forward_model(model.parameters, node)
            fw._sedimentation()
            #
            # pad sed array with bottom (zero-sized?) sediments
            while (node.sed.shape[0] < len(node.sediments_inputs)-1):
                mm = [ np.amax(node.sed[:,:,i]) for i in range(node.sed.shape[2]) ]                                
                # node.sed = np.concatenate([node.sed, np.zeros((1,node.sed.shape[1],node.sed.shape[2]))],axis=0)
                node.sed = np.concatenate([node.sed, np.tile(mm, (1,2,1))],axis=0)
            node.sed[np.where( np.abs(node.sed)<1e-6 )] = 0.0
            self.dataArray = [fw.current_node.X, fw.current_node.Y, node.sed.copy(), node.sediments.copy(), getNodeParameters(node) ]
            pickle.dump(self.dataArray, open(self.sedStack.nodeDirectoryPrefix+"node-sed-"+str(nodei)+"-"+str(nodej)+".pkl","wb") )
            return True

        node = model.builder.all_nodes[0]
        node.sed[np.where( np.abs(node.sed)<1e-6 )] = 0.0
        print(node.crust_ls)
        print(node.crust_ls.shape)
        #
        # pad sed array with bottom (zero-sized?) sediments
        while (node.sed.shape[0] < len(node.sediments_inputs)-1):
            mm = [ np.amax(node.sed[:,:,i]) for i in range(node.sed.shape[2]) ]                                
            # node.sed = np.concatenate([node.sed, np.zeros((1,node.sed.shape[1],node.sed.shape[2]))],axis=0)
            node.sed = np.concatenate([node.sed, np.tile(mm, (1,2,1))],axis=0)

        dataArray = [node.X, node.Y, node.sed.copy(), node.subsidence.copy(), node.crust_ls.copy(), node.lith_ls.copy(), node.sediments.copy(), node.beta, node.depth_out.copy(), node.temperature_out.copy(), getNodeParameters(node)]
        pickle.dump(dataArray, open(self.sedStack.nodeDirectoryPrefix+"node-slim-"+str(nodei)+"-"+str(nodej)+".pkl","wb") )
        print("Starting at resulting beta at node i,j:", nodei, nodej, model.parameters.starting_beta, node.beta)
        return True

def runWorker(args):
    print("runWorker", args[1])
    w = NodeWorker(args[0], args[1])    # for tti in range(time_start, time_end-1, -1)
    res = w.run()
    print("runworker ", args[1], res)
    return args[1]

if __name__ == '__main__': 
    class NodeProcessSpawner(object):
        def __init__(self, sedimentStack, node_pos_i, node_pos_j):
            self.sedimentStack = sedimentStack
            self.node_pos_i = node_pos_i
            self.node_pos_j = node_pos_j
        def run_pool(self):
            with Pool() as pool:
                # call the same function with different data in parallel
                node_i_j = list(itertools.product(node_pos_i, node_pos_j))          
                print("node pairs", node_i_j)      
                for result in pool.map(runWorker, zip(list(itertools.repeat(self.sedimentStack, len(node_i_j))), node_i_j )):
                    # report the value to show progress
                    print("Process finished: ", result)            

    gridFiles = []
    gridFiles.append(("docs/notebooks/data/0.gri",0))
    gridFiles.append(("docs/notebooks/data/66.gri",66))
    gridFiles.append(("docs/notebooks/data/100.gri",100))
    gridFiles.append(("docs/notebooks/data/163.gri",163))
    gridFiles.append(("docs/notebooks/data/168.gri",168))
    gridFiles.append(("docs/notebooks/data/170.gri",170))
    gridFiles.append(("docs/notebooks/data/182.gri",182))
    sedstack = SedimentStack()
    sedstack.loadFromGridFiles(gridFiles)
    sedstack.nodeDirectoryPrefix = "nodes-mapA/"

    # node = sedstack.create1DNodeAtIJ(100,100)
    import os
    if not os.path.exists(sedstack.nodeDirectoryPrefix):
        os.makedirs(sedstack.nodeDirectoryPrefix)

    #
    # Run all combinations of the following i,j indices for the nodes in the grid stack
    #  only a small subset of nodes is fully computed (when i and j are multiples of ten)
    #  the rest only has their sediment stacks computed, and the crust movement and subsidence is interpolated!
    # 
    # ng = NodeGrid(25400, 41600, 76, 76, 50, 50, 2, 100, 100)
    ng = NodeGrid(25400, 41600, 10, 10, 100, 100, 5, 100, 100)
    # ng = NodeGrid(25400, 41600, 1, 1, 100, 100, 2, 100, 100)
    # node_pos_i = list(range(100,160+1,2))
    # node_pos_j = list(range(100,160+1,2))
    node_pos_i = list(range(ng.start_index_x, ng.start_index_x + ng.num_nodes_x *ng.step_index, ng.step_index))
    node_pos_j = list(range(ng.start_index_y, ng.start_index_y + ng.num_nodes_y *ng.step_index, ng.step_index))

    print(sedstack.nodeDirectoryPrefix, node_pos_i, node_pos_j)

    ws = NodeProcessSpawner(sedstack, node_pos_i, node_pos_j)
    ws.run_pool()


