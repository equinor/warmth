from dataclasses import dataclass

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
    crustRHP: float = 2
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
def top_sed(nn, tti):
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
    # return 130e3+nn.subsidence[tti]+nn.sed_thickness_ls[tti]
    return thick_crust(nn,tti)+thick_lith(nn,tti)+nn.subsidence[tti]+nn.sed_thickness_ls[tti]
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
