import numpy as np
import pickle
from multiprocessing import Pool
import itertools

import warmth
from warmth.data import haq87
from warmth3d.SedimentStack import SedimentStack

from warmth3d.Helpers import NodeGrid

# logger = get_logger(__name__)

class NodeWorker(object):
    def __init__(self, sedimentStack, node_ij, full_beta_solve=False, save_full_node=""):
        self.node_ij = node_ij
        self.sedStack = sedimentStack
        self.full_beta_solve = full_beta_solve
        self.save_full_node = save_full_node

    def run(self):
        nodei, nodej = self.node_ij

        # full_beta_solve = ( (nodei % 10 == 0) and (nodej % 10 == 0))  # only a fraction of the nodes is simulated, the rest is interpolated
        full_beta_solve = self.full_beta_solve

        node = self.sedStack.create1DNodeAtIJ(nodei, nodej)
        invalid = np.any(np.isnan(np.array(node.sediments_inputs['top'].to_list())))
        if (invalid):
            print ("invalid", invalid, node.sediments_inputs['top'].to_list())
            return False
        model = warmth.Model()
        baseage = int(node.sediments["baseage"].iloc[-1])
        model.parameters.time_start = baseage
        model.parameters.time_end = 0
        model.parameters.starting_beta = 1.05
        model.builder.all_nodes = [node]
        model.builder.set_eustatic_sea_level(haq87)
        for i in model.builder.all_nodes:
            i.rift = np.array([[baseage, int(node.sediments["baseage"].iloc[-2])]])
        model.parameters.experimental = True
        if (full_beta_solve):
            print("NODE: ", node)
            model.simulator.run_all()
        else:
            #
            # compute only the sedimentation for this node, 
            #   if crustal thickness and subsidence are required in the 3D simulation, they will have to be interpolated 
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
            self.dataArray = [fw.current_node.X, fw.current_node.Y, node.sed.copy(), node.sediments.copy() ]
            pickle.dump(self.dataArray, open(self.sedStack.nodeDirectoryPrefix+"node-sed-"+str(nodei)+"-"+str(nodej)+".pkl","wb") )
            return True

        if (len(self.save_full_node)>0):
            pickle.dump(node, open(self.save_full_node, "wb"))
        node = model.builder.all_nodes[0]
        node.sed[np.where( np.abs(node.sed)<1e-6 )] = 0.0
        print(node.crust_ls)
        print(node.crust_ls.shape)
        #
        # pad sed array with bottom (zero-sized?) sediments
        while (node.sed.shape[0] < len(node.sediments_inputs)-1):
            mm = [ np.amax(node.sed[:,:,i]) for i in range(node.sed.shape[2]) ]                                
            node.sed = np.concatenate([node.sed, np.tile(mm, (1,2,1))],axis=0)

        dataArray = [node.X, node.Y, node.sed.copy(), node.subsidence.copy(), node.crust_ls.copy(), node.lith_ls.copy(), node.sediments.copy(), node.beta]
        pickle.dump(dataArray, open(self.sedStack.nodeDirectoryPrefix+"node-slim-"+str(nodei)+"-"+str(nodej)+".pkl","wb") )
        return True

def runWorker(args):
    print("runWorker", args[1])
    w = NodeWorker(args[0], args[1], args[2], args[3])    # for tti in range(time_start, time_end-1, -1)
    res = w.run()
    print("runworker ", args[1], res)
    return args[1]

if __name__ == '__main__': 
    class NodeProcessSpawner(object):
        def __init__(self, sedimentStack, node_pos_i, node_pos_j, full_beta_solve=False, save_full_node=""):
            self.sedimentStack = sedimentStack
            self.node_pos_i = node_pos_i
            self.node_pos_j = node_pos_j
            self.full_beta_solve = full_beta_solve
            self.save_full_node = save_full_node
        def run_pool(self):
            with Pool() as pool:
                # call the same function with different data in parallel
                node_i_j = list(itertools.product(node_pos_i, node_pos_j))          
                print("node pairs", node_i_j)     
                for result in pool.map(runWorker, zip( list(itertools.repeat(self.sedimentStack, len(node_i_j))), node_i_j, list(itertools.repeat(self.full_beta_solve , len(node_i_j))), list(itertools.repeat(self.save_full_node, len(node_i_j)))) ):
                    # report the value to show progress
                    print("Process finished: ", result)            

    gridFiles = []
    gridFiles.append(("hypothetical/0.gri",0))
    gridFiles.append(("hypothetical/30.gri",30))
    # gridFiles.append(("hypothetical/50.gri",50))
    gridFiles.append(("hypothetical/60.gri",60))

    sedstack = SedimentStack()
    sedstack.loadFromGridFiles(gridFiles)
    sedstack.nodeDirectoryPrefix = "nodes-hypothetical/"

    import os
    if not os.path.exists(sedstack.nodeDirectoryPrefix):
        os.makedirs(sedstack.nodeDirectoryPrefix)

    #
    # simulate one node fully
    #
    ng = NodeGrid(0, 0, 1, 1, 50, 50, 1, 100, 100)
    node_pos_i = [50]
    node_pos_j = [50]
    ws = NodeProcessSpawner(sedstack, node_pos_i, node_pos_j, full_beta_solve=True, save_full_node="nodes-hypothetical/nodeX.pkl")
    ws.run_pool()

    #
    # Run all combinations of the following i,j indices for the nodes in the grid stack
    #  only has the sediment stacks are compute
    # 
    ng = NodeGrid(0, 0, 97, 97, 2, 2, 1, 100, 100)
    node_pos_i = list(range(ng.start_index_x, ng.start_index_x + ng.num_nodes_x *ng.step_index, ng.step_index))
    node_pos_j = list(range(ng.start_index_y, ng.start_index_y + ng.num_nodes_y *ng.step_index, ng.step_index))
    print(sedstack.nodeDirectoryPrefix, node_pos_i, node_pos_j)
    ws = NodeProcessSpawner(sedstack, node_pos_i, node_pos_j, full_beta_solve=False, save_full_node="")
    ws.run_pool()



