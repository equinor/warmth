from dataclasses import dataclass
from typing import List
import itertools
import numpy as np
from .build import Builder, single_node

@dataclass
class NodeGrid:
    origin_x: float
    origin_y: float
    num_nodes_x: int
    num_nodes_y: int
    start_index_x: int
    start_index_y: int
    step_index: int   # including every N:th node (in x and y)
    step_x: float    # node separation in x
    step_y: float    # node separation in y
    modelNamePrefix: str = "new_test_X_"
    nodeDirectoryPrefix: str = "nodes-mapA/"


@dataclass
class NodeParameters1D:
    shf: float = 30e-3
    hc: float = 30e3
    hw: float = 3.6e3
    hLith: float = 130e3
    kLith: float = 3.109
    kCrust: float = 2.5
    kAsth: float = 100
    rhp: float = 2
    crustliquid: float = 2500.0
    crustsolid: float = 2800.0
    lithliquid: float = 2700.0
    lithsolid: float = 3300.0
    asthliquid: float = 2700.0
    asthsolid: float = 3200.0
    T0: float = 5
    Tm: float = 1330.0
    qbase: float = 30e-3

def getNodeParameters(node):
    #
    # TODO: better implementation
    #
    xx = NodeParameters1D()
    xx.shf = node.shf        
    xx.hc = node.hc        
    xx.hw = node.hw        
    xx.hLith = node.hLith        
    xx.kLith = getattr(node, 'kLith', 3.108)  
    xx.kCrust = node.kCrust        
    xx.kAsth = getattr(node, 'kAsth', 100)
    xx.rhp = node.rhp        
    xx.crustliquid = node.crustliquid        
    xx.crustsolid = node.crustsolid        
    xx.lithliquid = node.lithliquid        
    xx.lithsolid = node.lithsolid        
    xx.asthliquid = node.asthliquid        
    xx.asthsolid = node.asthsolid        
    xx.T0 = node.T0        
    xx.Tm = node.Tm        
    xx.qbase = node.qbase        
    return xx


def top_crust(nn, tti):
    if (tti > nn.subsidence.shape[0]-1):    
        return 0.0
    return nn.subsidence[tti] + nn.sed_thickness_ls[tti]
def top_sed(nn:single_node, tti):
    if (tti > nn.subsidence.shape[0]-1):    
        return 0.0
    return nn.subsidence[tti]
def thick_crust(nn, tti):
    if (tti > nn.crust_ls.shape[0]-1):    
        return 0.0
    return nn.crust_ls[tti]
def thick_lith(nn, tti):
    if (tti > nn.lith_ls.shape[0]-1):    
        return 0.0
    return nn.lith_ls[tti]
def top_lith(nn, tti):
    return top_crust(nn,tti) + thick_crust(nn,tti)
def top_asth(nn, tti):
    # return 130e3
    # return 130e3+nn.subsidence[tti]+nn.sed_thickness_ls[tti]
    return thick_crust(nn,tti)+thick_lith(nn,tti)+nn.subsidence[tti]+nn.sed_thickness_ls[tti]
    # return thick_crust(nn,tti)+thick_lith(nn,nn.lith_ls.shape[0]-1)+nn.subsidence[tti]+nn.sed_thickness_ls[tti]
    # return thick_lith(nn,tti) + top_lith(nn,tti)
def top_sed_id(nn, sed_id, tti):
    if (tti > nn.sed.shape[2]-1):    
        return 0.0
    if (sed_id==100):
        sed_id = 0
    return nn.sed[sed_id,0,tti]
def bottom_sed_id(nn, sed_id, tti):
    if (tti > nn.sed.shape[2]-1):    
        return 0.0
    if (sed_id==100):
        sed_id = 0
    return nn.sed[sed_id,1,tti]
def thick_sed(nn, sed_id, tti):
    return bottom_sed_id(nn,sed_id,tti) - top_sed_id(nn,sed_id,tti)

def volumeOfTet(points):
    """ Computes the volume of a tetrahedron, given as four 3D-points
    """ 
    import numpy as np
    ad = points[0]-points[3]
    bd = points[1]-points[3]
    cd = points[2]-points[3]
    bdcd = np.cross(bd,cd)
    return np.linalg.norm(np.dot(ad,bdcd))/6


def volumeOfHex(points):
    """ Computes the volume of a hexahedron, given as eight 3D-points
    """ 
    import numpy as np
    tetsplit1 = [ [1,2,4,8], [1,2,5,8], [4,8,2,3], [2,3,7,8], [2,5,6,8], [2,6,7,8] ]
    vol = 0.0
    for f in tetsplit1:
        tet = points[[p-1 for p in f],:]
        vol = vol + volumeOfTet(tet)
    return vol

def interpolateNode(interpolationNodes: List[single_node], interpolationWeights=None) -> single_node:
    assert len(interpolationNodes)>0
    if interpolationWeights is None:
        interpolationWeights = np.ones([len(interpolationNodes),1])
    assert len(interpolationNodes)==len(interpolationWeights)
    wsum = np.sum(np.array(interpolationWeights))
    iWeightNorm = [ w/wsum for w in interpolationWeights]

    node = single_node()
    node.__dict__.update(interpolationNodes[0].__dict__)
    node.X = np.sum( np.array( [node.X * w for node,w in zip(interpolationNodes,iWeightNorm)] ) ) 
    node.Y = np.sum( np.array( [node.Y * w for node,w in zip(interpolationNodes,iWeightNorm)] ) )

    # self.top_crust_arr = [ self._depth_out[ np.where(self._idsed[:,age] == -1)[0][0], age] for age in range(self.max_time)]
    # #print ("PING B")
    # self.top_lith_arr = [ self._depth_out[ np.where(self._idsed[:,age] == -2)[0][0], age] for age in range(self.max_time)]
    # #print ("PING C")
    # self.top_aest_arr = [ self._depth_out[ np.where(self._idsed[:,age] == -3)[0][0], age] for age in range(self.max_time)]
    # #print ("PING D")

    # self.top_lithosphere(age)-self.top_crust(age)

    if node.subsidence is None:
        node.subsidence = np.sum( np.array( [ node.seabed_arr[:] * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.crust_ls is None:
        node.crust_ls = np.sum( np.array( [ (node.top_lith_arr[:]-node.top_crust_arr[:]) * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.lith_ls is None:
        node.crust_ls = np.sum( np.array( [ (node.top_aest_arr[:]-node.top_lithosphere[:]) * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 

    if node.beta is None:
        node.beta = np.sum( np.array( [node.beta * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.kAsth is None:
        node.kAsth = np.sum( np.array( [node.kAsth * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.kLith is None:
        node.kLith = np.sum( np.array( [node.kLith * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    # if node._depth_out is None:
    #     node._depth_out = np.sum([node.result._depth_out*w for n,w in zip(interpolationNodes[0:1], [1] )], axis=0)
    # if node.temperature_out is None:
    #     node.temperature_out = np.sum([n.result.temperature_out*w for n,w in zip(interpolationNodes[0:1], [1] )], axis=0)

    if node.sed is None:
        node.sed = np.sum([n.sed*w for n,w in zip(interpolationNodes,iWeightNorm)], axis=0)
    if node.sed_thickness_ls is None:
        node.sed_thickness_ls =  node.sed[-1,1,:] - node.sed[0,0,:]    
    return node


def interpolate_all_nodes(builder:Builder)->Builder:
    for ni in range(len(builder.nodes)):
        for nj in range(len(builder.nodes[ni])):
            if (builder.nodes[ni][nj] is False) or (not builder.nodes[ni][nj]._full_simulation):
                closest_x_up = []
                for j in range(ni,len(builder.nodes[nj])):
                    matching_x = [ i[0] for i in builder.indexer_full_sim if i[0]==j ]
                    closest_x_up = closest_x_up + list(set(matching_x))
                    if len(matching_x)>0:
                        break
                closest_x_down = []
                for j in range(ni-1,-1,-1):
                    matching_x = [ i[0] for i in builder.indexer_full_sim if i[0]==j ]
                    closest_x_down = closest_x_down + list(set(matching_x))
                    if len(matching_x)>0:
                        break
                closest_y_up = []
                for j in range(nj,len(builder.nodes[ni])):
                    matching_y = [ i[1] for i in builder.indexer_full_sim if (i[1]==j and ((i[0] in closest_x_up) or i[0] in closest_x_down)) ]
                    closest_y_up = closest_y_up + list(set(matching_y))
                    if len(matching_y)>0:
                        break
                closest_y_down = []
                for j in range(nj-1,-1,-1):
                    matching_y = [ i[1] for i in builder.indexer_full_sim if (i[1]==j and (i[0] in closest_x_up or i[0] in closest_x_down) ) ]
                    closest_y_down = closest_y_down + list(set(matching_y))
                    if len(matching_y)>0:
                        break

                interpolationNodes = [  builder.nodes[i[0]][i[1]] for i in itertools.product(closest_x_up+closest_x_down, closest_y_up+closest_y_down)  ]
                interpolationNodes = [nn for nn in interpolationNodes if nn is not False]
                interpolationNodes = [nn for nn in interpolationNodes if nn._full_simulation]
                node = interpolateNode(interpolationNodes)
                node.X, node.Y = builder.grid.location_grid[ni,nj,:]
                builder.nodes[ni][nj] = node
            else:
                node = interpolateNode([builder.nodes[ni][nj]])  # "interpolate" the node from itself to make sure the same member variables exist at the end
                builder.nodes[ni][nj] = node
    return builder