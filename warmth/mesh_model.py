from typing import Tuple
from pathlib import Path
import numpy as np
from mpi4py import MPI
import meshio
import dolfinx  
from os import path
from petsc4py import PETSc
import ufl
import sys
import time
from scipy.interpolate import LinearNDInterpolator
from progress.bar import Bar
from warmth.build import single_node, Builder
from .parameters import Parameters
from warmth.logging import logger
from warmth.postprocessing import VR
from .mesh_utils import  top_crust,top_sed,thick_crust,  top_lith, top_asth, top_sed_id, bottom_sed_id,interpolateNode, interpolate_all_nodes
from .resqpy_helpers import write_tetra_grid_with_properties, write_hexa_grid_with_timeseries, write_hexa_grid_with_properties,read_mesh_resqml_hexa
def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(msg=""):
    if 'startTime_for_tictoc' in globals():
        delta = time.time() - startTime_for_tictoc
        print (msg+": Elapsed time is " + str(delta) + " seconds.")
        return delta
    else:
        print ("Toc: start time not set")
class UniformNodeGridFixedSizeMeshModel:
    """Manages a 3D heat equation computation using dolfinx
       Input is a uniform x-, y-grid of Nodes solved by 1D-SubsHeat 
       The mesh vertex and cell order stays the same across the simulation
       Zero-sized cells will be increased to be of a small minimum size

       The constructor takes a NodeGrid class and the list of 1D nodes
    """    
    point_domain_edge_map = {}
    point_top_vertex_map = {}
    point_bottom_vertex_map = {}
    def __init__(self, builder:Builder,parameters:Parameters, sedimentsOnly = False, padding_num_nodes=0):
        self._builder = builder
        self._parameters=parameters
        self.node1D = [n for n in self._builder.iter_node()]
        self.num_nodes = len(self.node1D)
        self.mesh = None

        self.modelName = self._parameters.name
        self.Temp0 = 5
        self.TempBase = 1369
        self.verbose = True
        self.minimumCellThick = 0.05
        self.averageLABdepth = 260000
        
        self.runSedimentsOnly = sedimentsOnly
        self.posarr = []
        self.Tarr = []
        self.time_indices = []

        # 2 6 6 6
        self.numElemPerSediment = 2
        self.numElemInCrust = 0 if self.runSedimentsOnly else 4    # split crust hexahedron into pieces
        self.numElemInLith = 0 if self.runSedimentsOnly else 4  # split lith hexahedron into pieces
        self.numElemInAsth = 0 if self.runSedimentsOnly else 4  # split asth hexahedron into pieces

        self.num_nodes_x = self._builder.grid.num_nodes_x
        self.num_nodes_y = self._builder.grid.num_nodes_y

        nodes_padded = []
        self.padX = padding_num_nodes
        for j in range(-self.padX,self.num_nodes_y+self.padX):
            for i in range(-self.padX,self.num_nodes_x+self.padX):
                in_padding = False
                in_padding = (i<0) or (j<0) or (i>=self.num_nodes_x) or (j>=self.num_nodes_y)
                if (in_padding):
                    source_x = max(0, min(i, self.num_nodes_x-1))
                    source_y = max(0, min(j, self.num_nodes_y-1))
                    new_node = interpolateNode( [self._builder.nodes[source_y][source_x] ] )
                    new_node.X = self._builder.grid.origin_x + i*self._builder.grid.step_x
                    new_node.Y = self._builder.grid.origin_y + j*self._builder.grid.step_y
                    nodes_padded.append(new_node)
                else:
                    nodes_padded.append(self._builder.nodes[j][i])

        self.node1D = nodes_padded
        self.num_nodes = len(self.node1D)
        self.num_nodes_x = self._builder.grid.num_nodes_x + 2*self.padX
        self.num_nodes_y = self._builder.grid.num_nodes_y + 2*self.padX

        # self.convexHullEdges = []
        # for i in range(self.num_nodes_x-1):
        #     edge = [i, i+1]
        #     self.convexHullEdges.append(edge)
        #     edge = [i+(self.num_nodes_y-1*self.num_nodes_x), i+1+(self.num_nodes_y-1*self.num_nodes_x)]
        #     self.convexHullEdges.append(edge)
        # for i in range(self.num_nodes_y-1):
        #     edge = [i*self.num_nodes_x, (i+1)*self.num_nodes_x]
        #     self.convexHullEdges.append(edge)
        #     edge = [i*self.num_nodes_x + (self.num_nodes_x-1), (i+1)*self.num_nodes_x+ (self.num_nodes_x-1)]
        #     self.convexHullEdges.append(edge)

        self.useBaseFlux = False
        self.baseFluxMagnitude = 0.06

        self.mesh0_geometry_x = None
        self.CGorder = 1

        self.layer_id_per_vertex = None
        self.thermalCond = None
        self.mean_porosity = None
        self.c_rho = None
        self.numberOfSediments = self._builder.input_horizons.shape[0]-1 #skip basement
        self.numberOfSedimentCells = self.numberOfSediments * self.numElemPerSediment

        self.interpolators = {}
   
    def write_tetra_mesh_resqml( self, out_path):
        """Prepares arrays and calls the RESQML output helper function:  the lith and aesth are removed, and the remaining
           vertices and cells are renumbered;  the sediment properties are prepared for output.

           out_path: string: path to write the resqml model to (.epc and .h5 files)

           returns the filename (of the .epc file) that was written 
        """            
        def boundary(x):
            return np.full(x.shape[1], True)
        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)
        p0 = self.mesh.geometry.x[tet,:]
        tet_to_keep = []
        p_to_keep = set()
        lid_to_keep = []
        cell_id_to_keep = []
        for i,t in enumerate(tet):
            ps = p0[i]
            minY = np.amin( np.array( [p[1] for p in ps] ) )
            midpoint = np.sum(ps,axis=0)*0.25
            lid0 = self.findLayerID(self.tti, midpoint)
            # 
            # discard aesth and lith (layer IDs -2, -3)
            #
            if (lid0>=-1) and (lid0<100):
                tet_to_keep.append(t)
                lid_to_keep.append(lid0)
                cell_id_to_keep.append(self.node_index[i])
                if abs(self.node_index[i].Y-minY)>1:
                    logger.warning( f"unusual Y coordinate:, {minY}, {self.node1D[self.node_index[i]].Y}, {i}, {self.node_index[i]}, {self.node1D[self.node_index[i]]}")
                for ti in t:
                    p_to_keep.add(ti)
        poro0_per_cell = np.array( [ self.getSedimentPropForLayerID('phi', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ] )
        decay_per_cell = np.array( [ self.getSedimentPropForLayerID('decay', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        density_per_cell = np.array( [ self.getSedimentPropForLayerID('solidus', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        cond_per_cell = np.array( [ self.getSedimentPropForLayerID('k_cond', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        rhp_per_cell = np.array( [ self.getSedimentPropForLayerID('rhp', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])

        lid_per_cell = np.array(lid_to_keep)

        points_cached = []
        point_original_to_cached = np.ones(self.mesh.geometry.x.shape[0], dtype = np.int32)  * (-1)
        for i in range(self.mesh.geometry.x.shape[0]):
            if (i in p_to_keep):
                point_original_to_cached[i] = len(points_cached)
                points_cached.append(self.mesh.geometry.x[i,:])
        tet_renumbered = [ [point_original_to_cached[i] for i in tet] for tet in tet_to_keep ]
        T_per_vertex = [ self.uh.x.array[i] for i in range(self.mesh.geometry.x.shape[0]) if i in p_to_keep  ]
        age_per_vertex = [ self.mesh_vertices_age[i] for i in range(self.mesh.geometry.x.shape[0]) if i in p_to_keep  ]
        
        filename = path.join(out_path, self.modelName+'_'+str(self.tti)+'.epc')
        write_tetra_grid_with_properties(filename, np.array(points_cached), tet_renumbered, "tetramesh",
            np.array(T_per_vertex), np.array(age_per_vertex), poro0_per_cell, decay_per_cell, density_per_cell,
            cond_per_cell, rhp_per_cell, lid_per_cell)
        return filename

    # def send_mpi_messages(self):
    #     comm = MPI.COMM_WORLD
    #     comm.send(mm2.mesh.topology.index_map(0).local_to_global(list(range(mm2.mesh.geometry.x.shape[0]))) , dest=0, tag=((comm.rank-1)*10)+21)
    #     comm.send(mm2.mesh_reindex, dest=0, tag=((comm.rank-1)*10)+23)
    #     comm.send(mm2.mesh_vertices_age, dest=0, tag=((comm.rank-1)*10)+25)
    #     comm.send(self.posarr, dest=0, tag=( (comm.rank-1)*10)+20)
    #     comm.send(self.Tarr, dest=0, tag=( (comm.rank-1)*10)+24)

    def receive_mpi_messages(self):
        comm = MPI.COMM_WORLD
        import time
        st = time.time()

        self.sub_posarr_s = [self.posarr]
        self.sub_Tarr_s = [self.Tarr]

        self.index_map_s = [self.mesh.topology.index_map(0).local_to_global(list(range(self.mesh.geometry.x.shape[0])))]
        self.mesh_reindex_s = [self.mesh_reindex]
        self.mesh_vertices_age_s = [self.mesh_vertices_age]

        for i in range(1,comm.size):
            self.index_map_s.append(comm.recv(source=i, tag=((i-1)*10)+21))
            self.mesh_reindex_s.append(comm.recv(source=i, tag=((i-1)*10)+23))
            self.mesh_vertices_age_s.append(comm.recv(source=i, tag=((i-1)*10)+25))
            self.sub_posarr_s.append(comm.recv(source=i, tag=((i-1)*10)+20))
            self.sub_Tarr_s.append(comm.recv(source=i, tag=((i-1)*10)+24))

        nv = np.amax(np.array( [np.amax(index_map) for index_map in self.index_map_s ] )) + 1   # no. vertices/nodes
        mri = np.arange( nv, dtype=np.int32)

        self.x_original_order = [ (np.ones( [nv,3], dtype= np.float32) * -1) for _ in range(len(self.posarr)) ]
        self.T_per_vertex = [ (np.ones( nv, dtype= np.float32) * -1) for _ in range(len(self.Tarr)) ]
        self.age_per_vertex = np.ones( nv, dtype= np.int32) * -1

        for k in range(len(self.mesh_reindex_s)):
            for ind,val in enumerate(self.mesh_reindex_s[k]):
                for i in range(len(self.posarr)):
                    self.x_original_order[i][val,:] = self.sub_posarr_s[k][i][ind,:] 
                    self.T_per_vertex[i][val] = self.sub_Tarr_s[k][i][ind]
                self.age_per_vertex[val] = self.mesh_vertices_age_s[k][ind]

        delta = time.time() - st
        logger.debug( f"receive_mpi_messages delta: {delta}")

    def get_node_pos_and_temp(self, tti=-1):
        #
        # This function can only be called after all MPI compute processes have finished and receive_mpi_messages has been called.
        #
        start_time = self.time_indices[0]
        end_time = self.time_indices[-1]
        ind = start_time - ( tti if (tti>=0) else end_time)
        if (ind >= len(self.x_original_order)):
            return None,None
        if (ind < 0):
            return None,None
        return self.x_original_order[ind], self.T_per_vertex[ind]

    def write_hexa_mesh_resqml( self, out_path, tti):
        """Prepares arrays and calls the RESQML output helper function for hexa meshes:  the lith and aesth are removed, and the remaining
           vertices and cells are renumbered;  the sediment properties are prepared for output.

           out_path: string: path to write the resqml model to (.epc and .h5 files)

           returns the filename (of the .epc file) that was written 
        """            
        comm = MPI.COMM_WORLD
        nv = np.amax(np.array([np.amax(index_map) for index_map in self.index_map_s])) +1  # no. vertices/nodes
        hexaHedra, hex_data_layerID, hex_data_nodeID = self.buildHexahedra(keep_padding=False)

        hexa_to_keep = []
        p_to_keep = set()
        lid_to_keep = []
        cond_per_cell = []
        cell_id_to_keep = []
        x_original_order, T_per_vertex = self.get_node_pos_and_temp(tti)
        for i,h in enumerate(hexaHedra):
            lid0 = hex_data_layerID[i]
            # 
            # discard aesth and lith (layer IDs -2, -3)
            #
            if (lid0>=-1) and (lid0<100):
                hexa_to_keep.append(h)
                lid_to_keep.append(lid0)
                cell_id_to_keep.append(hex_data_nodeID[i])
                minY = np.amin(np.array ( [x_original_order[hi,1] for hi in h] ))
                if abs( self.node1D[hex_data_nodeID[i]].Y - minY)>1:
                    logger.warning( f"weird Y:  {minY}, {self.node1D[hex_data_nodeID[i]].Y}, {abs( self.node1D[hex_data_nodeID[i]].Y - minY)}, {i}, {hex_data_nodeID[i]}" )
                    breakpoint()
                for hi in h:
                    p_to_keep.add(hi)
                    # k_cond_mean.append(self.thermalCond.x.array[hi])   # the actual, Sekiguchi-derived conductivitues
                # cond_per_cell.append( np.mean(np.array(k_cond_mean)))

        poro0_per_cell = np.array( [ self.getSedimentPropForLayerID('phi', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ] )
        decay_per_cell = np.array( [ self.getSedimentPropForLayerID('decay', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        density_per_cell = np.array( [ self.getSedimentPropForLayerID('solidus', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        cond_per_cell = np.array( [ self.getSedimentPropForLayerID('k_cond', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        rhp_per_cell = np.array( [ self.getSedimentPropForLayerID('rhp', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        lid_per_cell = np.array(lid_to_keep)

        points_cached = []
        point_original_to_cached = np.ones(nv, dtype = np.int32)  * (-1)
        for i in range(nv):
            if (i in p_to_keep):
                point_original_to_cached[i] = len(points_cached)
                points_cached.append(x_original_order[i,:])
        hexa_renumbered = [ [point_original_to_cached[i] for i in hexa] for hexa in hexa_to_keep ]
        
        # for i,h in enumerate(hexa_renumbered):
        #     minY = np.amin(np.array ( [np.array(points_cached)[hi][1] for hi in h] ))
        #     poro0 = poro0_per_cell[i]
        #     lid0  = lid_to_keep[i]

        T_per_vertex_keep = [ T_per_vertex[i] for i in range(nv) if i in p_to_keep ]
        age_per_vertex_keep = [ self.age_per_vertex[i] for i in range(nv) if i in p_to_keep ]

        filename_hex = path.join(out_path, self.modelName+'_hexa_'+str(tti)+'.epc')
        points_cached=np.array(points_cached)
        write_hexa_grid_with_properties(filename_hex, points_cached, hexa_renumbered, "hexamesh",
            np.array(T_per_vertex_keep), np.array(age_per_vertex_keep), poro0_per_cell, decay_per_cell, density_per_cell,
            cond_per_cell, rhp_per_cell, lid_per_cell)
        return filename_hex


    def write_hexa_mesh_timeseries( self, out_path):
        """Prepares arrays and calls the RESQML output helper function for hexa meshes:  the lith and aesth are removed, and the remaining
           vertices and cells are renumbered;  the sediment properties are prepared for output.

           out_path: string: path to write the resqml model to (.epc and .h5 files)

           returns the filename (of the .epc file) that was written 
        """            
        hexaHedra, hex_data_layerID, hex_data_nodeID = self.buildHexahedra(keep_padding=False)
        hexa_to_keep = []
        p_to_keep = set()
        lid_to_keep = []
        cell_id_to_keep = []
        for i,h in enumerate(hexaHedra):
            lid0 = hex_data_layerID[i]
            # 
            # discard aesth and lith (layer IDs -2, -3)
            #
            if (lid0>=-1) and (lid0<100):
                hexa_to_keep.append(h)
                lid_to_keep.append(lid0)
                cell_id_to_keep.append(hex_data_nodeID[i])
                for hi in h:
                    p_to_keep.add(hi)


        poro0_per_cell = np.array( [ self.getSedimentPropForLayerID('phi', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ] )
        decay_per_cell = np.array( [ self.getSedimentPropForLayerID('decay', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        density_per_cell = np.array( [ self.getSedimentPropForLayerID('solidus', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        cond_per_cell = np.array( [ self.getSedimentPropForLayerID('k_cond', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        rhp_per_cell = np.array( [ self.getSedimentPropForLayerID('rhp', lid,cid) for lid,cid in zip(lid_to_keep,cell_id_to_keep) ])
        lid_per_cell = np.array(lid_to_keep)
        
        x_original_order, T_per_vertex = self.get_node_pos_and_temp(self.time_indices[0]) # oldest first
        n_vertices = x_original_order.shape[0]
        age_per_vertex_keep = np.array([ self.age_per_vertex[i] for i in range(n_vertices) if i in p_to_keep ])
        Temp_per_vertex_series = np.empty([len(self.time_indices), len(p_to_keep)])
        points_cached_series = np.empty([len(self.time_indices), len(p_to_keep),3])
        Ro_per_vertex_series = np.empty([len(self.time_indices), len(p_to_keep)])

        for idx, tti in enumerate(self.time_indices): # oldest first
            if idx > 0:
                x_original_order, T_per_vertex = self.get_node_pos_and_temp(tti)
            T_per_vertex_filt = [ T_per_vertex[i] for i in range(n_vertices) if i in p_to_keep  ]
            Temp_per_vertex_series[idx,:] = T_per_vertex_filt
            point_original_to_cached = np.full(n_vertices,-1,dtype = np.int32)
            count = 0
            for i in range(n_vertices):
                if (i in p_to_keep):
                    points_cached_series[idx,count,:]=x_original_order[i,:]
                    point_original_to_cached[i]= count
                    count += 1
        s = time.time()
        logger.debug("Calculating vitrinite reflectance EasyRo%DL")
        for i in range(Temp_per_vertex_series.shape[1]):
            ts = Temp_per_vertex_series[:,i]
            ro = VR.easyRoDL(ts)
            Ro_per_vertex_series[:,i] = ro.flatten()
        logger.debug(f"VR calculation {time.time()-s}s")
        hexa_renumbered = [ [point_original_to_cached[i] for i in hexa] for hexa in hexa_to_keep ]

        

        filename_hex = path.join(out_path, self.modelName+'_hexa_ts_'+str(self.tti)+'.epc')
        write_hexa_grid_with_timeseries(filename_hex, points_cached_series, hexa_renumbered, "hexamesh",
            Temp_per_vertex_series, Ro_per_vertex_series, age_per_vertex_keep, poro0_per_cell, decay_per_cell, density_per_cell,
            cond_per_cell, rhp_per_cell, lid_per_cell )
        return filename_hex

    def heatflow_at_crust_sed_boundary(self):
        hf_array = np.zeros([self.num_nodes_x-2*self.padX, self.num_nodes_y-2*self.padX])
        for hy in range(self.padX, self.num_nodes_y-self.padX):
            for hx in range(self.padX, self.num_nodes_x-self.padX):
                v_per_n = int(self.mesh_vertices.shape[0]/(self.num_nodes_y*self.num_nodes_x))
                ind_base_of_sed = v_per_n - self.numElemInAsth - self.numElemInLith - self.numElemInCrust -1
                # first_ind_in_crust = v_per_n - self.numElemInAsth - self.numElemInLith - self.numElemInCrust

                node_ind = hy*self.num_nodes_x + hx
                nn = self.node1D[node_ind]
                temp_3d_ind = np.array([ np.where([self.mesh_reindex==i])[1][0] for i in range(node_ind*v_per_n+ind_base_of_sed, node_ind*v_per_n+ind_base_of_sed+2 ) ] )
                dd = self.mesh.geometry.x[temp_3d_ind,2]
                tt = self.u_n.x.array[temp_3d_ind]
                hf = nn.kCrust*(tt[1]-tt[0])/(dd[1]-dd[0])

                hx_unpad = hx - self.padX            
                hy_unpad = hy - self.padX            
                hf_array[hx_unpad, hy_unpad] = hf
        return hf_array


    def getSubsidenceAtMultiplePos(self, pos_x, pos_y):
        """Returns subsidence values at given list of x,y positions.
            TODO: re-design
        """            
        subs1 = []
        for px,py in zip(pos_x,pos_y):
            fkey = self.floatKey2D([px+2e-2, py+2e-2])
            dz = UniformNodeGridFixedSizeMeshModel.point_top_vertex_map.get(fkey, 1e10)
            subs1.append(dz)
        return np.array(subs1)

    def getBaseAtMultiplePos(self, pos_x, pos_y):
        """Returns lowest mesh z values at given list of x,y positions.
            TODO: re-design
        """            
        subs1 = []
        for px,py in zip(pos_x,pos_y):
            fkey = self.floatKey2D([px+2e-2, py+2e-2])
            dz = UniformNodeGridFixedSizeMeshModel.point_bottom_vertex_map.get(fkey, 1e10)
            subs1.append(dz)
        return np.array(subs1)

    def getTopOfLithAtNode(self, tti, node:single_node):
        """Returns crust-lith boundary depth at the given time at the given node
        """           
        z0 = top_lith( node, tti ) if not self.runSedimentsOnly else 0
        return z0

    def getTopOfAsthAtNode(self, tti, node:single_node):
        """Returns crust-lith boundary depth at the given time at the given node
        """           
        z0 = top_asth( node, tti ) if not self.runSedimentsOnly else 0
        return z0

    #
    def getSedimentPropForLayerID(self, property, layer_id:int, node_index:int) ->float:
        """
        """           
        assert property in ['k_cond', 'rhp', 'phi', 'decay', 'solidus', 'liquidus'], "Unknown property " + property
        if (layer_id>=0) and (layer_id<self.numberOfSediments):
            node = self.node1D[node_index]
            prop = node.sediments[property][layer_id]
            return prop
        if (layer_id<=-1) and (layer_id>=-3):
            lid = -layer_id -1
            node = self.node1D[node_index]
            if (property=='k_cond'):
                return [node.kCrust, node.kLith, node.kAsth][lid]
            if (property=='rhp'):
                return [node.crustRHP,0.0,0.0][lid] 
            if (property=='phi'):
                return [0.0,0.0,0.0][lid]   # porosity for crust, lith, aest
            if (property=='decay'):
                return [0.5,0.5,0.5][lid]   # porosity decay for crust, lith, aest ?
            if (property=='solidus'):
                return [node.crustsolid,node.lithsolid,node.asthsolid][lid]   # solid density for crust, lith, aest
            if (property=='liquidus'):
                return [node.crustliquid,node.lithliquid,node.asthliquid][lid]   # liquid density for crust, lith, aest
        return np.nan

    def porosity0ForLayerID(self, layer_id:int, node_index:int)->Tuple[float, float]:
        """Porosity (at surface) conductivity value for the given layer index
        """           
        if (layer_id==-1):
            return 0.0,1e8 # porosity (at surface) of crust
        if (layer_id==-2):
            return 0.0,1e8 # porosity (at surface) of lith
        if (layer_id==-3):
            return 0.0,1e8  # porosity (at surface) of aesth
        if (layer_id>=0) and (layer_id<self.numberOfSediments):
            # assert (node_index < len(self.node1D)) and node_index > 0
            node = self.node1D[node_index]
            phi = node.sediments.phi[layer_id]
            decay = node.sediments.decay[layer_id]
            return phi, decay
        return 0.0, 0.0

    def cRhoForLayerID(self, ss:int, node_index:int)->float:      
        node = self.node1D[node_index]
        if (ss==-1):
            return self._parameters.cp*node.crustsolid
        if (ss==-2):
            return self._parameters.cp*node.lithsolid
        if (ss==-3):
            return self._parameters.cp*node.asthsolid
        if (ss>=0) and (ss<self.numberOfSediments):
            rho = node.sediments.solidus[ss]
            return self._parameters.cp*rho
        return self._parameters.cp*node.crustsolid

    def kForLayerID(self, ss:int, node_index:int)->float:
        """Thermal conductivity for a layer ID index
        """
        if (node_index > len(self.node1D)-1):
            raise Exception(f"Node index {node_index} larger then node length {len(self.node1D)}")
        node = self.node1D[node_index]
        if (ss==-1):
            return node.kCrust
        elif  (ss==-2):
            return node.kLith
        elif (ss==-3):
            return node.kAsth
        elif (ss>=0) and (ss<self.numberOfSediments):
            # kc = self.globalSediments.k_cond[ss]
            # node_index = cell_id // (self.numberOfSediments+6)  # +6 because crust, lith, aest are each cut into two
            kc = node.sediments.k_cond[ss]
            return kc


    def rhpForLayerID(self, ss:int, node_index:int)->float:
        """Radiogenic heat production for a layer ID index
        """
        if (ss==-1):
            node = self.node1D[node_index]
            kc = node.crustRHP * node._upperCrust_ratio
            return kc
        elif (ss>=0) and (ss<self.numberOfSediments):
            node = self.node1D[node_index]
            kc = node.sediments.rhp[ss]
            return kc
        else:
            return 0


    def buildVertices(self, time_index=0, optimized=False):
        """Determine vertex positions, node-by-node.
           For every node, the same number of vertices is added (one per sediment, one per crust, lith, asth, and one at the bottom)
           Degenerate vertices (e.g. at nodes where sediment is yet to be deposited or has been eroded) are avoided by a small shift, kept in self.sed_diff_z
        """           
        tti = time_index
        self.tti = time_index
        import time

        compare = False
        if (optimized) and hasattr(self, 'mesh_vertices'):            
            compare = True
            xxT = self.top_sed_at_nodes[:,tti]
            bc = self.base_crust_at_nodes[:,tti]
            bl = self.base_lith_at_nodes[:,tti]
            ba = bl.copy()+130000
            aa = np.array([self.top_sed_at_nodes[:,tti]])

            for ss in range(self.numberOfSediments):
                for j in range(self.numElemPerSediment):
                    base_of_prev_sediments = self.bottom_sed_id_at_nodes[ss-1][:,tti] if (ss>0) else (self.bottom_sed_id_at_nodes[ss][:,tti]*0)
                    base_of_current_sediments = self.bottom_sed_id_at_nodes[ss][:,tti]
                    base_of_current_sediments = base_of_prev_sediments + (base_of_current_sediments-base_of_prev_sediments)* (j+1) / self.numElemPerSediment
                    if self.runSedimentsOnly:
                        zpos = base_of_current_sediments
                        aa = np.concatenate([aa,np.array([zpos])])
                    else:
                        zpos = xxT + base_of_current_sediments
                        aa = np.concatenate([aa,np.array([zpos])])

            base_of_last_sediments = (xxT+self.bottom_sed_id_at_nodes[-1][:,tti]) if (len(self.bottom_sed_id_at_nodes)>0) else xxT

            for i in range(1,self.numElemInCrust+1):
                zp = base_of_last_sediments+ (bc-base_of_last_sediments)*(i/self.numElemInCrust) 
                aa = np.concatenate([aa,np.array([zp])])

            for i in range(1,self.numElemInLith+1):
                zp = bc+ (bl-bc)*(i/self.numElemInLith) 
                aa = np.concatenate([aa,np.array([zp])])
            for i in range(1,self.numElemInAsth+1):
                zp = bl+ (ba-bl)*(i/self.numElemInLith) 
                aa = np.concatenate([aa,np.array([zp])])

            new_z_pos_0 = np.transpose(aa).flatten()
            mm0 = self.mesh_vertices.copy()
            mm0[:,2] = new_z_pos_0
            self.mesh_vertices_0 = mm0
        else:
            self.mesh_vertices_0 = []
            self.sed_diff_z = []
            self.mesh_vertices_age_unsorted = []

            for ind,node in enumerate(self.node1D):
                top_of_sediments = top_sed(node, tti)
                self.mesh_vertices_0.append( [ node.X, node.Y, top_of_sediments - 0.0*(self.numberOfSedimentCells+1) ] )
                self.sed_diff_z.append(-self.minimumCellThick*(self.numberOfSedimentCells+1))
                self.mesh_vertices_age_unsorted.append(node.sediments.topage[0])  # append top age of top sediment
                if (ind==0):
                    st = time.time()
                for ss in range(self.numberOfSediments):
                    for j in range(self.numElemPerSediment):
                        base_of_prev_sediments = bottom_sed_id(node, ss-1, tti) if (ss>0) else top_of_sediments
                        base_of_current_sediments = bottom_sed_id(node, ss, tti)
                        base_of_current_sediments = base_of_prev_sediments + (base_of_current_sediments-base_of_prev_sediments)* (j+1) / self.numElemPerSediment
                        if self.runSedimentsOnly:
                            zpos = base_of_current_sediments
                        else:
                            zpos = top_of_sediments + base_of_current_sediments
                        vert = np.array([ node.X, node.Y, zpos ])
                        self.mesh_vertices_0.append( vert )
                        self.sed_diff_z.append(-self.minimumCellThick*(self.numberOfSedimentCells - (ss*self.numElemPerSediment+j) ))
                        age_of_previous = node.sediments.baseage[ss-1] if (ss>0) else 0.0
                        self.mesh_vertices_age_unsorted.append( age_of_previous + ((j+1) / self.numElemPerSediment) * (node.sediments.baseage[ss]-age_of_previous) )  # append interpolatedbase age of current sediment
                if (ind==0):
                    delta = time.time() - st
                    logger.debug(f"delta 2: {delta}")
                    st = time.time()
                if not self.runSedimentsOnly:
                    base_of_last_sediments = bottom_sed_id(node, self.numberOfSediments-1, tti) if (self.numberOfSediments>0) else top_of_sediments
                    base_crust = node.subsidence[tti] + node.sed_thickness_ls[tti] + node.crust_ls[tti]

                    for i in range(1,self.numElemInCrust+1):
                        self.mesh_vertices_0.append( [ node.X, node.Y, base_of_last_sediments+ (base_crust-base_of_last_sediments)*(i/self.numElemInCrust) ] )
                        self.sed_diff_z.append(0.0)
                        self.mesh_vertices_age_unsorted.append(1000)

                    base_lith = node.crust_ls[tti]+node.lith_ls[tti]+node.subsidence[tti]+node.sed_thickness_ls[tti]
                    for i in range(1,self.numElemInLith+1):
                        self.mesh_vertices_0.append( [ node.X, node.Y, base_crust+ (base_lith-base_crust)*(i/self.numElemInLith) ] )
                        self.sed_diff_z.append(0.0)
                        self.mesh_vertices_age_unsorted.append(1000)

                    base_aest = base_lith+130000
                    for i in range(1,self.numElemInAsth+1):
                        self.mesh_vertices_0.append( [ node.X, node.Y, base_lith+(base_aest-base_lith)*(i/self.numElemInAsth) ] )
                        self.sed_diff_z.append(0.0)
                        self.mesh_vertices_age_unsorted.append(1000)
                if (ind==0):
                    delta = time.time() - st
                    logger.debug(f"delta 3: {delta}")

            assert len(self.mesh_vertices_0) % self.num_nodes ==0
            self.mesh_vertices_0 = np.array(self.mesh_vertices_0)
            self.sed_diff_z = np.array(self.sed_diff_z)
            self.mesh_vertices = self.mesh_vertices_0.copy()
        self.mesh_vertices[:,2] = self.mesh_vertices_0[:,2] + self.sed_diff_z

    def updateVertices(self):
        """Update the mesh vertex positions using the values in self.mesh_vertices, and using the known dolfinx-induded reindexing
        """        
        self.mesh.geometry.x[:] = self.mesh_vertices[self.mesh_reindex].copy()
        self.mesh_vertices_age = np.array(self.mesh_vertices_age_unsorted)[self.mesh_reindex].copy()
        self.mesh0_geometry_x = self.mesh.geometry.x.copy()      
        self.mesh_vertices[:,2] = self.mesh_vertices_0[:,2] - self.sed_diff_z
        
        if hasattr(self, 'top_sed_at_nodes'):         
            self.updateTopVertexMap()
        if self.runSedimentsOnly or (self.useBaseFlux): 
            self.updateBottomVertexMap()
        self.mesh_vertices[:,2] = self.mesh_vertices_0[:,2] + self.sed_diff_z

    def buildMesh(self,tti:int):
        """Construct a new mesh at the given time index tti, and determine the vertex re-indexing induced by dolfinx
        """        
        self.tti = tti
        self.buildVertices(time_index=tti)
        logger.info("Built vertices")
        self.constructMesh()
        logger.info("Built mesh")
        self.updateMesh(tti)
        logger.info("Updated vertices")
     

    def updateMesh(self,tti:int, optimized=False):
        """Construct the mesh positions at the given time index tti, and update the existing mesh with the new values
        """   
        import time
        assert self.mesh is not None
        self.tti = tti        
        self.buildVertices(time_index=tti, optimized=optimized)
        self.updateVertices()        
        self.posarr.append(self.mesh.geometry.x.copy())
        self.time_indices.append(self.tti)

    def buildHexahedra(self, keep_padding=True):
        xpnum = self.num_nodes_x
        ypnum = self.num_nodes_y
        # xpnum = self.num_nodes_x - 2* self.padX
        # ypnum = self.num_nodes_y - 2* self.padX

        nodeQuads = []
        for j in range(ypnum-1):
            for i in range(xpnum-1):
                is_not_padded = (j>=self.padX) and (j<ypnum-self.padX) and (i>=self.padX) and (i<xpnum-self.padX)
                if (keep_padding) or (is_not_padded):
                    i0 = j * (xpnum)+i
                    q = [ i0, i0+1, i0 + xpnum+1, i0 + xpnum ]
                    nodeQuads.append(q)

        v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        assert len(self.mesh_vertices) % self.num_nodes ==0

        hexaHedra = []
        hex_data_layerID = []
        hex_data_nodeID = []
        for q in nodeQuads:
            for s in range(v_per_n-1):
                h = []
                #
                for i in range(4):
                    i0 = q[i]*v_per_n + s+1
                    h.append(i0)
                for i in range(4):
                    h.append(q[i]*v_per_n + s)
                hexaHedra.append(h)
                lid = s // self.numElemPerSediment
                if (s >= self.numberOfSedimentCells):
                    ss = s - self.numberOfSedimentCells
                    if (ss>=0) and (ss<self.numElemInCrust):
                        lid = -1                        
                    if (ss>=self.numElemInCrust) and (ss < self.numElemInCrust+self.numElemInLith):
                        lid = -2                        
                    if (ss>=self.numElemInCrust+self.numElemInLith) and (ss<self.numElemInCrust+self.numElemInLith+self.numElemInAsth):
                        lid = -3                        
                hex_data_layerID.append(lid)
                hex_data_nodeID.append(q[0])
        return hexaHedra, hex_data_layerID, hex_data_nodeID

    def constructMesh(self):
        """Generates a pseudo-structured tetrahedral mesh based on the vertex positions in self.mesh_vertices.
           Vertices are grouped by node, and nodes are arranged in a uniform grid.
           One hexahedron is constructed per layer per four corner nodes, and each hexahedron is split into six tetrahedra.
           Since dolfinx does not allow zero-sized cells, the mesh vertices must have been separated slightly at degenrate positions.

           The meshio library is used to write the mesh to file, from which dolfinx reads it.
        """   
        self.thermalCond = None
        v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        hexaHedra, hex_data_layerID, hex_data_nodeID = self.buildHexahedra()

        comm = MPI.COMM_WORLD


        # https://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
        tetsplit1 = [ [1,2,4,8], [1,2,5,8], [4,8,2,3], [2,3,7,8], [2,5,6,8], [2,6,7,8] ]
        tetsplit0 = [ [ p-1 for p in v ] for v in tetsplit1 ]
        lid_per_node = [100]
        for i in range(self.numberOfSedimentCells):
            lid_per_node.append(i)
        if not self.runSedimentsOnly: 
            for i in range(1,self.numElemInCrust+1):
                lid_per_node.append(-1)
            for i in range(1,self.numElemInLith+1):
                lid_per_node.append(-2)
            for i in range(1,self.numElemInAsth+1):
                lid_per_node.append(-3)
        assert len(lid_per_node) == v_per_n

        cells = []
        cell_data_layerID = []
        node_index = []
        c_count = 0
        for h,lid,nid in zip(hexaHedra, hex_data_layerID, hex_data_nodeID):
            for t in tetsplit0:
                candidate_tet_ind = [ h[k] for k in t ]
                cells.append(candidate_tet_ind)
                cell_data_layerID.append(lid)
                node_index.append(nid)
            c_count = c_count + 1

        points = self.mesh_vertices.copy()

        mesh = meshio.Mesh(
            points,
            [ ("tetra", cells ) ],
            # Only one cell-spefific data array can be recovered by dolfinx (using read_meshtags), so we can write only one!
            cell_data={"layer": [ (np.array(cell_data_layerID, dtype=np.float64)+3)*1e7 + np.array(node_index, dtype=np.float64) ] },
        )
        
        def mpi_print(s):
            logger.info(f"Rank {comm.rank}: {s}")

        fn = self.modelName+"_mesh.xdmf"
        if comm.rank==0:
            mesh.write( fn )
            logger.info(f"saved mesh to {fn}")             
        comm.Barrier()
        enc = dolfinx.io.XDMFFile.Encoding.HDF5
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fn, "r", encoding=enc) as file:   # MPI.COMM_SELF
            self.mesh = file.read_mesh(name="Grid", ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)
            aa = file.read_meshtags(self.mesh, name="Grid")
            self.cell_data_layerID = np.floor(aa.values.copy()*1e-7)-3
            self.node_index = np.mod(aa.values.copy(),1e7).astype(np.int32)
        self.mesh.topology.create_connectivity(3,0)  # create_connectivity_all()
        mpi_print(f"Number of local cells: {self.mesh.topology.index_map(3).size_local}")
        mpi_print(f"Number of global cells: {self.mesh.topology.index_map(3).size_global}")
        mpi_print(f"Number of local vertices: {self.mesh.topology.index_map(0).size_local}")
        mpi_print(f"Ghost cells (global numbering): {self.mesh.topology.index_map(3).ghosts}")
        mpi_print(f"Ghost nodes (global numbering): {self.mesh.topology.index_map(0).ghosts}")

        mpi_print(f"Dir local cells: {type(self.mesh.topology.index_map(3))} {dir(self.mesh.topology.index_map(3))}")
        mpi_print(f"Dir local verts: {dir(self.mesh.topology.index_map(0))}")  # local_to_global ! ?
        mpi_print(f"Type local verts: {type(self.mesh.topology.index_map(0))}")

        # store original vertex order 
        self.mesh_reindex = np.array(self.mesh.geometry.input_global_indices).astype(np.int32)
        self.mesh0_geometry_x = self.mesh.geometry.x.copy()


    def TemperatureGradient(self, x):
        self.averageLABdepth = np.mean(np.array([ top_asth(n, self.tti) for n in self.node1D]))

        nz = (x[2,:] - self.Zmin) / (self.averageLABdepth - self.Zmin)
        nz[nz>1.0] = 1.0
        res = nz * (self.TempBase-self.Temp0) + self.Temp0
        for i in range(x.shape[1]):
            p = x[:,i]
            fkey = self.floatKey2D(p+[2e-2, 2e-2,0.0])
            dz = UniformNodeGridFixedSizeMeshModel.point_top_vertex_map.get(fkey, 1e10)
            Zmin0 = dz if (dz<1e9) else self.Zmin 
            nz0 = (p[2] - Zmin0) / (self.averageLABdepth - Zmin0)
            nz0 = min(nz0, 1.0)
            res[i] = nz0 * (self.TempBase-self.Temp0) + self.Temp0
            if (p[2] > self.averageLABdepth):
                res[i] =  1330 + (p[2]-self.averageLABdepth)*0.0003  # 1369
        return res


    def floatKey2D(self, xp):
        """Returns a tuple of integers from the input xy-point.  This is useful as a dict key for fast lookups. 
        """   
        return( int(xp[0]*10), int(xp[1]*10) )


    def sedimentsConductivitySekiguchi(self):
        """Scale surface conductivity of sediments to effective conductivity of sediments at depth. Scaler of 0.6 based on Allen & Allen p345. porosity dependent conductivity
        Args:
            mean_porosity (npt.NDArray[np.float64]): Mean porosity of sediments at depth
            conductivity (npt.NDArray[np.float64]): Conductivity of sediments at 20C
        Returns:
            npt.NDArray[np.float64]: Effective conductivity of sediments
        """
        import time
        def boundary(x):
            return np.full(x.shape[1], True)
        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)
        self.layer_id_per_vertex = [ [] for _ in range(self.mesh.geometry.x.shape[0]) ]
        ## 

        top_km = (np.min(self.mesh.geometry.x[tet,2], 1)-self.subsidence_at_nodes[:,self.tti]) * 1e-3
        bottom_km = (np.max(self.mesh.geometry.x[tet,2], 1)-self.subsidence_at_nodes[:,self.tti]) * 1e-3
        poro0 = self.porosity0.x.array[:]
        decay = self.porosityDecay.x.array[:]
        ii = np.where(top_km<0.0)

        diff_z = np.amin(top_km)
        top_km[top_km<0.0] = top_km[top_km<0.0]+0.0002-diff_z
        bottom_km[bottom_km<0.0] = bottom_km[bottom_km<0.0]+0.0002-diff_z
        f1 = np.divide( self.porosity0.x.array[:], np.multiply( self.porosityDecay.x.array[:], bottom_km-top_km ) )
        f2 = np.exp(-1 * np.multiply(decay, top_km)) - np.exp(-1 * np.multiply(decay, bottom_km))
        mean_porosity = np.multiply(f1,f2)

        # temperature_C is the mean temperature in the cells
        temperature_C = np.mean(self.uh.x.array[tet[:,:]],1)
        temperature_K_inv = np.reciprocal(273.15 + temperature_C)
        conductivity_effective = 1.84 + 358 * np.multiply( ( (1.0227*self.thermalCond0)-1.882) , ((temperature_K_inv)-0.00068) )
        conductivity_effective = conductivity_effective * (1.0-mean_porosity) # * np.sqrt(1-mean_porosity)
        self.thermalCond.x.array[self.layerIDsFcn.x.array[:]>=0] = conductivity_effective[self.layerIDsFcn.x.array[:]>=0]
        self.mean_porosity.x.array[self.layerIDsFcn.x.array[:]>=0] = mean_porosity[self.layerIDsFcn.x.array[:]>=0]

        newrho = self._parameters.cp * np.multiply( \
            ((self.c_rho0.x.array[:]/self._parameters.cp)), (1-mean_porosity)) + mean_porosity*self._parameters.rhowater
        self.c_rho.x.array[self.layerIDsFcn.x.array[:]>=0] = newrho[self.layerIDsFcn.x.array[:]>=0] 

    def getCellMidpoints(self):  
        def boundary(x):
            return np.full(x.shape[1], True)
        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)
        self.layer_id_per_vertex = [ [] for _ in range(self.mesh.geometry.x.shape[0]) ]
        midp = []
        for i,t in enumerate(tet):
            lidval = int(self.layerIDsFcn.x.array[i])
            if (lidval<0):
                # only relevant for sediment
                continue
            zpos = np.mean(self.mesh.geometry.x[t,:], axis=0)
            midp.append(zpos)
        return np.array(midp)


    def buildKappaAndLayerIDs(self):
        """Returns two dolfinx functions, constant-per-cell, on the current mesh:
            one contains thermal conductivity (kappa) values, one contains layer IDs
        """   

        self.mesh_vertex_layerIDs = np.full_like(self.mesh.geometry.x[:,2], 100, dtype=np.int32 )
        # piecewise constant Kappa in the tetrahedra
        Q = dolfinx.fem.FunctionSpace(self.mesh, ("DG", 0))  # discontinuous Galerkin, degree zero
        thermalCond = dolfinx.fem.Function(Q)
        c_rho = dolfinx.fem.Function(Q)
        self.c_rho0 = dolfinx.fem.Function(Q)
        lid = dolfinx.fem.Function(Q)
        rhp = dolfinx.fem.Function(Q)
        self.porosity0 = dolfinx.fem.Function(Q)
        self.mean_porosity = dolfinx.fem.Function(Q)
        self.porosityDecay = dolfinx.fem.Function(Q)
        self.porosityAtDepth = dolfinx.fem.Function(Q)
        self.rhp0 = dolfinx.fem.Function(Q)

        #
        # subdomains:
        # https://jorgensd.github.io/dolfinx-tutorial/chapter3/subdomains.html
        #
        def boundary(x):
            return np.full(x.shape[1], True)

        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)

        p0 = self.mesh.geometry.x[tet,:]
        midp = np.sum(p0,1)*0.25   # midpoints of tetrahedra

        ls = self.cell_data_layerID.copy()
        lid.x.array[:] = np.array(ls, dtype=PETSc.ScalarType).flatten()

        ks = [ self.kForLayerID(lid,self.node_index[i]) for i,lid in enumerate(ls)]
        thermalCond.x.array[:] = np.array(ks, dtype=PETSc.ScalarType).flatten()
        self.thermalCond0 = np.array(ks, dtype=PETSc.ScalarType).flatten()

        rhps = [ self.rhpForLayerID(lid,self.node_index[i]) for i,lid in enumerate(ls)]
        rhp.x.array[:] = np.array(rhps, dtype=PETSc.ScalarType).flatten()
        self.rhp0.x.array[:] = np.array(rhps, dtype=PETSc.ScalarType).flatten()

        crs = [ self.cRhoForLayerID(lid,self.node_index[i]) for i,lid in enumerate(ls)]
        c_rho.x.array[:] = np.array(crs, dtype=PETSc.ScalarType).flatten()
        self.c_rho0.x.array[:] = np.array(crs, dtype=PETSc.ScalarType).flatten()

        poro = [ self.porosity0ForLayerID(lid, self.node_index[i])[0] for i,lid in enumerate(ls)]
        self.porosity0.x.array[:] = np.array(poro, dtype=PETSc.ScalarType).flatten()
        self.porosityAtDepth.x.array[:] = np.array(poro, dtype=PETSc.ScalarType).flatten()

        decay = [ self.porosity0ForLayerID(lid, self.node_index[i])[1] for i,lid in enumerate(ls)]
        self.porosityDecay.x.array[:] = np.array(decay, dtype=PETSc.ScalarType).flatten()

        self.layer_id_per_vertex = [ [] for _ in range(self.mesh.geometry.x.shape[0]) ]
        for i,t in enumerate(tet):
            lidval = int(lid.x.array[i])
            for ti in t:
                self.layer_id_per_vertex[ti].append(lidval)
        for i,t in enumerate(tet):
            lidval = int(lid.x.array[i])
            midp_z = midp[i][2]
            for j,ti in enumerate(t):
                vertex_on_top_of_tet = (self.mesh.geometry.x[ti,2] < midp_z)
                if (lidval>=0):
                    next_lidval = lidval+1 
                    while (next_lidval not in self.layer_id_per_vertex[ti]) and (next_lidval<self.numberOfSediments):
                        next_lidval = next_lidval + 1
                    if (next_lidval>self.numberOfSediments-1):
                        next_lidval = -1
                    if vertex_on_top_of_tet:
                        self.mesh_vertex_layerIDs[ti] = lidval
                    elif self.mesh_vertex_layerIDs[ti]==100:
                        self.mesh_vertex_layerIDs[ti] = next_lidval
                else:
                    if ((lidval) > self.mesh_vertex_layerIDs[ti]) or (self.mesh_vertex_layerIDs[ti]>=100):
                        self.mesh_vertex_layerIDs[ti] = lidval
        self.updateTopVertexMap()
        if self.runSedimentsOnly:
            self.updateBottomVertexMap()
        return thermalCond, c_rho, lid, rhp

    def updateTopVertexMap(self):
        """ Updates the point_top_vertex_map, used for fast lookup of subsidence values.
        """ 
        UniformNodeGridFixedSizeMeshModel.point_top_vertex_map = {}
        v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        for i,n in enumerate(self.node1D):
            aa = self.top_sed_at_nodes[i, self.tti] + (-self.minimumCellThick*(self.numberOfSedimentCells+1))
            fkey = self.floatKey2D( np.array([n.X,n.Y,0.0]) + [2e-2, 2e-2,0.0])
            UniformNodeGridFixedSizeMeshModel.point_top_vertex_map[fkey] = aa
            fkey = self.floatKey2D( np.array([n.X,n.Y,0.0]) - [2e-2, 2e-2,0.0])
            UniformNodeGridFixedSizeMeshModel.point_top_vertex_map[fkey] = aa
            fkey = self.floatKey2D( np.array([n.X,n.Y,0.0]) + [0e-2, 0e-2,0.0])
            UniformNodeGridFixedSizeMeshModel.point_top_vertex_map[fkey] = aa

    def updateBottomVertexMap(self):
        """ Updates the point_bottom_vertex_map, used for fast lookup of subsidence values.
            (to be re-designed?)
        """ 
        UniformNodeGridFixedSizeMeshModel.point_bottom_vertex_map = {}
        #v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        for i in range(self.mesh.geometry.x.shape[0]):
            p = self.mesh.geometry.x[i,:]
            fkey = self.floatKey2D(p+[2e-2, 2e-2,0.0])
            dz = UniformNodeGridFixedSizeMeshModel.point_bottom_vertex_map.get(fkey, -1e10)
            if p[2]>dz:
                UniformNodeGridFixedSizeMeshModel.point_bottom_vertex_map[fkey] = p[2]

    def updateDirichletBaseTemperature(self):
        assert False, "to be re-implemented"


    def updateDBC(self):
        self.averageLABdepth = np.mean(np.array([ top_asth(n, self.tti) for n in self.node1D]))
        logger.debug(f"self.averageLABdepth: {self.averageLABdepth}")
        ii = np.where(self.mesh.geometry.x[:,2]>250000)
        self.bc.value.x.array[ii] = 1330+(self.mesh.geometry.x[ii,2]-self.averageLABdepth)*0.0003


    def buildDirichletBC(self):
        """ Generate a dolfinx Dirichlet Boundary condition that applies at the top and bottom vertices.
            The values at the edges are those in function self.TemperatureStep
        """ 
        import time
        comm = MPI.COMM_WORLD          

        st = time.time()
        self.averageLABdepth = np.mean(np.array([ top_asth(n, self.tti) for n in self.node1D]))
        self.Zmax = self.averageLABdepth + 130000
        logger.debug(f"buildDirichletBC delta 1: {time.time()-st}")

        def boundary_D_top_bottom(x):
            subs0 = self.getSubsidenceAtMultiplePos(x[0,:], x[1,:])
            xx = np.logical_or( np.abs(x[2]-subs0) < 0.9*self.minimumCellThick, np.abs(x[2]-self.Zmax)<10000 )
            return xx
        def boundary_D_bottom(x):
            subs0 = self.getSubsidenceAtMultiplePos(x[0,:], x[1,:])
            xx = np.logical_or( np.abs(x[2]-self.Zmax)<10000, np.abs(x[2]-self.Zmax)<10000 )
            return xx
        def boundary_D_top(x):
            subs0 = self.getSubsidenceAtMultiplePos(x[0,:], x[1,:])
            xx = np.logical_or( np.abs(x[2]-subs0) < 0.9*self.minimumCellThick, np.isclose(x[2], 1e6*self.Zmax) )            
            return xx
            
        if (self.useBaseFlux):
            st = time.time()
            dofs_D = dolfinx.fem.locate_dofs_geometrical(self.V, boundary_D_top)
            logger.debug(f"buildDirichletBC delta 2: {time.time()-st}")
        else:
            st = time.time()
            dofs_D = dolfinx.fem.locate_dofs_geometrical(self.V, boundary_D_top_bottom)
            # dofs_D2 = dolfinx.fem.locate_dofs_geometrical(self.V, boundary_D_bottom)
            # dofs_D3 = dolfinx.fem.locate_dofs_geometrical(self.V, boundary_D_top)
            logger.debug(f"buildDirichletBC delta 3: {time.time()-st}")
        u_bc = dolfinx.fem.Function(self.V)
        st = time.time()
        u_bc.interpolate(self.TemperatureGradient)
        logger.debug(f"buildDirichletBC delta 4: {time.time()-st}")
        bc = dolfinx.fem.dirichletbc(u_bc, dofs_D)
        return bc


    def resetMesh(self):
        self.mesh.geometry.x[:,2] = self.mesh0_geometry_x.copy()[:,2]

    def writeLayerIDFunction(self, outfilename, tti=0):
        """ Writes the mesh and the layer ID function (constant value per cell) to the given output file in XDMF format
        """         
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.layerIDsFcn, tti)

    def writePoroFunction(self, outfilename, tti=0):
        """ Writes the mesh and poro0 function (constant value per cell) to the given output file in XDMF format
        """         
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.porosity0, tti)

    def writeTemperatureFunction(self, outfilename, tti=0):
        """ Writes the mesh and the current temperature solution to the given output file in XDMF format
        """         
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.u_n, tti)

    def writeOutputFunctions(self, outfilename, tti=0):
        """ Writes the mesh, layer IDs, and current temperature solution to the given output file in XDMF format
            #
            # TODO: this does not work
            #
        """         
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.layerIDsFcn, tti)
        xdmf.write_function(self.u_n, tti)

    def setupSolver(self, initial_state_model = None):
        comm = MPI.COMM_WORLD          
        def mpi_print(s):
            print(f"Rank {comm.rank}: {s}")

        self.resetMesh()
        self.Zmin = np.min(self.mesh_vertices, axis=0)[2]
        self.Zmax = np.max(self.mesh_vertices, axis=0)[2]
        # mpi_print(f"Number of local cells: {self.mesh.topology.index_map(3).size_local}")
        # mpi_print(f"Number of global cells: {self.mesh.topology.index_map(3).size_global}")
        # mpi_print(f"Number of local vertices: {self.mesh.topology.index_map(0).size_local}")

        #
        # define function space
        self.FE = ufl.FiniteElement("CG", self.mesh.ufl_cell(), self.CGorder)
        self.V = dolfinx.fem.FunctionSpace(self.mesh, self.FE)

        # Define solution variable uh
        self.uh = dolfinx.fem.Function(self.V)
        self.uh.name = "uh"

        # u_n: solution at previous time step
        self.u_n = dolfinx.fem.Function(self.V)
        self.u_n.name = "u_n"

        import time

        st = time.time()
        nn = self.node1D[self.node_index[0]]
        
        self.subsidence_at_nodes = np.zeros([ len(self.cell_data_layerID), nn.subsidence.shape[0] ])
        
        for i in range(len(self.node1D)):
            nn = self.node1D[i]
            iix = np.where(self.node_index==i)
            self.subsidence_at_nodes[iix,:] = nn.subsidence
        logger.debug(f"setup delay 1.3: {time.time()-st}")

        st = time.time()
        top_of_sediments = self.subsidence_at_nodes
        self.bottom_sed_id_at_nodes = [] 
        self.top_sed_at_nodes = np.zeros([self.num_nodes, nn.subsidence.shape[0] ])
        self.base_crust_at_nodes = np.zeros([self.num_nodes, nn.subsidence.shape[0] ])
        self.base_lith_at_nodes = np.zeros([self.num_nodes, nn.subsidence.shape[0] ])
        for k in range(0,self.numberOfSediments):
            bottom_sed_id_at_nodes = np.zeros([self.num_nodes, nn.subsidence.shape[0] ])
            for i in range(len(self.node1D)):
                nn = self.node1D[i]
                bottom_sed_id_at_nodes[i, :] = nn.sed[k,1,:]
            self.bottom_sed_id_at_nodes.append(bottom_sed_id_at_nodes)
        for i in range(len(self.node1D)):
            nn = self.node1D[i]
            self.top_sed_at_nodes[i,:] = nn.subsidence
            self.base_crust_at_nodes[i,:] = nn.subsidence + nn.sed_thickness_ls + nn.crust_ls
            self.base_lith_at_nodes[i,:]  = self.base_crust_at_nodes[i,:] + nn.lith_ls
        logger.debug(f"setup delay 1.4: {time.time()-st}")

        st = time.time()
        self.thermalCond, self.c_rho, self.layerIDsFcn, self.rhpFcn = self.buildKappaAndLayerIDs()
        assert not np.any(np.isnan(self.thermalCond.x.array))
        logger.debug(f"setup delay 1: {time.time()-st}")

        st = time.time()
        # initialise both with initial condition: either a step function, or the solution from another Model instance
        if (initial_state_model is None):
            self.u_n.interpolate(self.TemperatureGradient)
        else:
            self.u_n.x.array[:] = initial_state_model.uh.x.array[:].copy()
        self.uh.x.array[:] = self.u_n.x.array[:].copy()
        logger.debug(f"setup delay 1.5: {time.time()-st}")

    def setupSolverAndSolve(self, n_steps:int=100, time_step:int=-1, skip_setup:bool = False, initial_state_model = None, update_bc=False):
        """ Sets up the function spaces, output functions, input function (kappa values), boundary conditions, initial conditions.
            Sets up the heat equation in dolfinx, and solves the system in time for the given number of steps.
            
            Use skip_setup = True to continue a computation (e.g. after deforming the mesh), instead of starting one from scratch 
        """   
        comm = MPI.COMM_WORLD          
        def mpi_print(s):
            logger.debug(f"Rank {comm.rank}: {s}")
    
        if (not skip_setup):
            self.setupSolver(initial_state_model)

        if (not skip_setup) or update_bc:
            self.bc = self.buildDirichletBC()

        import time
        st = time.time()
        self.sedimentsConductivitySekiguchi()
        logger.debug(f"solve delay 2: {time.time()-st}")

        t=0
        dt = time_step if (time_step>0) else  3600*24*365 * 5000000
        num_steps = n_steps

        #
        # Time-dependent heat problem:
        #   time-discretized variational form with backwards Euler,
        #   see: https://fenicsproject.org/pub/tutorial/html/._ftut1006.html
        #
        #  solver setup, see:
        #  https://jorgensd.github.io/dolfinx-tutorial/chapter2/diffusion_code.html
        #

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = self.c_rho*u*v*ufl.dx + dt*ufl.dot(self.thermalCond*ufl.grad(u), ufl.grad(v)) * ufl.dx
        f = self.rhpFcn 
        #logger.info(f"mean RHP {np.mean(self.rhpFcn.x.array[:])}")

        if ( self.useBaseFlux ):
            baseFlux = self.baseFluxMagnitude
            # define Neumann condition: constant flux at base
            # expression g defines values of Neumann BC (heat flux at base)
            domain_c = dolfinx.fem.Function(self.V)
            domain_c.x.array[ : ] = 0.0
            if (self.CGorder>1):
                #
                # NOTE: CGorder>1 is under development, not functional
                #
                def marker(x):
                    return x[2,:]>3990
                facets = dolfinx.mesh.locate_entities_boundary(self.mesh, dim=(self.mesh.topology.dim - 2),
                                        marker=marker )
                dofs = dolfinx.fem.locate_dofs_topological(V=self.V, entity_dim=1, entities=facets)
                if (len(facets)>0):
                    #print( np.amax(facets))
                    pass
                if (len(dofs)>0):
                    #print( np.amax(dofs))
                    pass
                domain_c.x.array[ dofs ] = 1
            else:
                basepos = self.getBaseAtMultiplePos(self.mesh.geometry.x[:,0], self.mesh.geometry.x[:,1])
                domain_c.x.array[  self.mesh.geometry.x[:,2] > basepos*0.99 ] = 1
                xmin, xmax = np.amin(self.mesh.geometry.x[:,0]), np.amax(self.mesh.geometry.x[:,0])
                ymin, ymax = np.amin(self.mesh.geometry.x[:,1]), np.amax(self.mesh.geometry.x[:,1])
                #
                # remove corners from base heat flow domain
                # domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] < xmin+1, self.mesh.geometry.x[:,1] < ymin+1) ] = 0
                # domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] < xmin+1, self.mesh.geometry.x[:,1] > ymax-1) ] = 0
                # domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] > xmax-1, self.mesh.geometry.x[:,1] < ymin+1) ] = 0
                # domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] > xmax-1, self.mesh.geometry.x[:,1] > ymax-1) ] = 0

            domain_zero = dolfinx.fem.Function(self.V)
            toppos = self.getSubsidenceAtMultiplePos(self.mesh.geometry.x[:,0], self.mesh.geometry.x[:,1])
            domain_zero.x.array[  self.mesh.geometry.x[:,2] < toppos+0.01 ] = 1
            #logger.debug(f"Neumann conditions: , {self.tti}, {np.count_nonzero(domain_c.x.array)}, {np.count_nonzero(domain_zero.x.array)}")

            g = (-1.0*baseFlux) * ufl.conditional( domain_c > 0, 1.0, 0.0 )
            L = (self.c_rho*self.u_n + dt*f)*v*ufl.dx - dt * g * v * ufl.ds    # last term reflects Neumann BC 
        else:
            L = (self.c_rho*self.u_n + dt*f)*v*ufl.dx   # no Neumann BC 

        bilinear_form = dolfinx.fem.form(a)
        linear_form = dolfinx.fem.form(L)

        A = dolfinx.fem.petsc.assemble_matrix(bilinear_form, bcs=[self.bc])
        A.assemble()
        b = dolfinx.fem.petsc.create_vector(linear_form)

        comm = MPI.COMM_WORLD
        solver = PETSc.KSP().create(self.mesh.comm)

        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        import time
        for i in range(num_steps):
            t += dt
            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            dolfinx.fem.petsc.assemble_vector(b, linear_form)

            # TODO: update Dirichlet BC at every time step:
            #       the temperature at the base of Asth is set such that it reaches Tm at the current depth of the LAB (using the slope adiab=0.0003)
            # bc = self.buildDirichletBC()

            # Apply Dirichlet boundary condition to the vector
            dolfinx.fem.petsc.apply_lifting(b, [bilinear_form], [[self.bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            dolfinx.fem.petsc.set_bc(b, [self.bc])

            # Solve linear problem
            solver.solve(b, self.uh.vector)
            self.uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            # diffnorm = np.sum(np.abs(self.u_n.x.array - self.uh.x.array)) / self.u_n.x.array.shape[0]
            self.u_n.x.array[:] = self.uh.x.array
            # comm.Barrier()
        self.Tarr.append(self.uh.x.array[:].copy())
        # print("latest Tarr", self.Tarr[-1], np.mean(self.Tarr[-1]))




    #
    # =====================================
    #     Helper functions, not used by the main workflow
    #
    # =====================================
    #

    def safeInterpolation(self, interp, pos_x, pos_y, epsilon=1e-2):
        #
        # NDLinearInterpolator cannot extrapolate beyond the data points;
        #   use an epsilon to avoid NaN in sitations where the query point is marginally outside
        #
        res = interp([pos_x, pos_y])[0]
        if (np.isnan(res)):
            manyres = np.array( [ interp([pos_x-epsilon, pos_y-epsilon])[0], \
                interp([pos_x+epsilon, pos_y-epsilon])[0],\
                interp([pos_x-epsilon, pos_y+epsilon])[0],\
                interp([pos_x+epsilon, pos_y+epsilon])[0]])
            res = np.nanmean(manyres)
        if (np.isnan(res)):
            logger.warning(f'NaN encounered in safeInterpolation pos_x {pos_x}:  pos_y: {pos_y};  {interp([pos_x, pos_y])} {interp([0,0])[0]} ')
        # assert not np.isnan(res), "interpolation is nan in safeInterpolation"
        return res

    def getThickOfCrustAtPos(self, tti, pos_x, pos_y):
        interp = self.getInterpolator(tti, "thick_crust")
        thick_crust_1 = self.safeInterpolation(interp, pos_x, pos_y)        
        assert not np.isnan(thick_crust_1), "interpolation is nan in thick crust!"
        return thick_crust_1

    def getTopOfCrustAtPos(self, tti, pos_x, pos_y):
        interp = self.getInterpolator(tti, "top_crust")
        top_crust_1 = self.safeInterpolation(interp, pos_x, pos_y)        
        assert not np.isnan(top_crust_1), "interpolation is nan in top crust!"
        return top_crust_1

    def getTopOfLithAtPos(self, tti, pos_x, pos_y):
        interp = self.getInterpolator(tti, "topoflith")
        top_lith_1 = self.safeInterpolation(interp, pos_x, pos_y)        
        assert not np.isnan(top_lith_1), "interpolation is nan in top lith!"
        return top_lith_1

    def getTopOfAsthAtPos(self, tti, pos_x, pos_y):
        interp = self.getInterpolator(tti, "topofasth")
        top_asth_1 = self.safeInterpolation(interp, pos_x, pos_y)        
        assert not np.isnan(top_asth_1), "interpolation is nan in top asth!"
        return top_asth_1

    def getSubsidenceAtPos(self, tti, pos_x, pos_y):
        interp = self.getInterpolator(tti, "subsidence")
        subs1 = self.safeInterpolation(interp, pos_x, pos_y)        
        assert not np.isnan(subs1), "interpolation is nan in subsidence!"
        return subs1

    def getSedPosAtPos(self, tti, pos_x, pos_y, sediment_id, use_top_instead_of_bottom=False):
        interp = self.getInterpolator(tti, "sedimentpos", sed_id=sediment_id, use_top_instead_of_bottom=use_top_instead_of_bottom)
        z_c_1 = self.safeInterpolation(interp, pos_x, pos_y)        
        return z_c_1

    def getPosAtNode(self, tti, node_index, sediment_id, use_top_instead_of_bottom=False):
        z_c = top_sed(self.node1D[node_index],tti)
        if (use_top_instead_of_bottom):
            z_c = z_c + top_sed_id(self.node1D[node_index], sediment_id, tti)
        else:
            z_c = z_c + bottom_sed_id(self.node1D[node_index], sediment_id, tti)
        return z_c

    #
    def findLayerID(self, tti, point):
        """Helper function to determine the layer ID for the given point.  Not used by the main simulation workflow
        """
        px, py, pz = point[0],point[1],point[2]
        subs = self.getSubsidenceAtPos(tti, px, py)
        if (pz<subs-0.1):
            return 100
        top_crust = self.getTopOfCrustAtPos(tti, px, py)
        top_lith  = self.getTopOfLithAtPos(tti, px, py)
        top_asth  = self.getTopOfAsthAtPos(tti, px, py)
        if (pz > top_crust) and (pz<=top_lith):
            return -1
        if (pz > top_lith) and (pz<=top_asth):
            return -2
        if (pz > top_asth):
            return -3
        for ss in range(self.numberOfSediments):
            if (ss==0):
                top_sed = self.getSedPosAtPos(tti, px, py, 0, use_top_instead_of_bottom=True)
            else:
                top_sed = self.getSedPosAtPos(tti, px, py, ss-1)
            top_next_sed = self.getSedPosAtPos(tti, px, py, ss)
            if ( ss == self.numberOfSediments-1):
                top_next_sed = top_next_sed + 0.1
            if (pz >= top_sed) and (pz < top_next_sed):
                return ss
        return 100


    def interpolatorKey(self, tti, dataname, sed_id = -1, use_top_instead_of_bottom=False):
        key = str(tti)+"_"+dataname
        if (sed_id>=0):
            key=key+"SED"+str(sed_id)
        if (use_top_instead_of_bottom):
            key=key+"TOP"
        return key

    def getInterpolator(self, tti, dataname, sed_id = -1, use_top_instead_of_bottom=False):
        key = self.interpolatorKey(tti, dataname, sed_id=sed_id, use_top_instead_of_bottom=use_top_instead_of_bottom)
        if (key in self.interpolators):
            return self.interpolators[key]
        
        xpos = [ node.X for node in self.node1D ]
        ypos = [ node.Y for node in self.node1D ]

        val = None
        if (dataname=="thick_crust"):
            val = [ thick_crust(node, tti) for node in self.node1D ]
        if (dataname=="top_crust"):
            val = [ top_crust(node, tti) for node in self.node1D ]
        if (dataname=="subsidence"):
            val = [ top_sed(node, tti) for node in self.node1D ]
        if (dataname=="sedimentpos"):
            val = [ self.getPosAtNode(tti, i, sed_id, use_top_instead_of_bottom=use_top_instead_of_bottom) for i in range(len(self.node1D)) ]
        if (dataname=="topoflith"):
            val = [ self.getTopOfLithAtNode(tti, i) for i in range(len(self.node1D)) ]
        if (dataname=="topofasth"):
            val = [ self.getTopOfAsthAtNode(tti, i) for i in range(len(self.node1D)) ]
        assert val is not None, "unknown interpolator datanme " + dataname

        interp = LinearNDInterpolator(list(zip(xpos, ypos)), val)
        self.interpolators[key] = interp
        return interp


    def evaluateVolumes(self):
        def boundary(x):
            return np.full(x.shape[1], True)

        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)

        p0 = self.mesh.geometry.x[tet,:]
        totalvol = 0
        num_sed = self.numberOfSediments
        subvols = [0.0 for _ in range(num_sed+1)]
        for i,t in enumerate(tet):
            ps = p0[i]
            ps = p0[i]
            vol = self.volumeOfTet(ps)
            lid = self.cell_data_layerID[i]
            # lid = self.findLayerID(self.tti, midpoint)
            commonset = self.layer_id_per_vertex[t[0]].intersection(self.layer_id_per_vertex[t[1]]).intersection(self.layer_id_per_vertex[t[2]]).intersection(self.layer_id_per_vertex[t[3]])
            lid = int(list(commonset)[0])

            vol = self.volumeOfTet(ps)
            totalvol = totalvol + vol
            if (lid==-1):
                subvols[num_sed] = subvols[num_sed] + vol
            if (lid>=0) and (lid<num_sed):
                subvols[lid] = subvols[lid] + vol
        return subvols


    def pointIsInTriangle2D(self, pt, triangle):
        def sign (p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(pt, triangle[0,:], triangle[1,:])
        d2 = sign(pt, triangle[1,:], triangle[2,:])
        d3 = sign(pt, triangle[2,:], triangle[0,:])

        # allow for a small tolerance here since vertices are often at cell edges
        has_neg = (d1 < -0.001) or (d2 < -0.001) or (d3 < -0.001)   
        has_pos = (d1 > 0.001) or (d2 > 0.001) or (d3 > 0.001)
        is_in_triangle = not (has_neg and has_pos)
        return is_in_triangle

    # 
    def interpolateResult(self, x):
        """interpolates the result at given positions x;
           depends on dolfinx vertex-cell association functions which sometimes fail for no obvious reason..
        """

        #
        # the interpolation code is prone to problems, especially when vertices are ouside the mesh
        # 
        tol = 1.0    # Avoid hitting the outside of the domain
        tol_z = 1.0  # Avoid hitting the outside of the domain
        plot_points = []
        meshZmax = np.amax(self.mesh.geometry.x[:,2])
        
        midpoint = np.mean(self.mesh_vertices,axis=0)
        mzm = []

        transpose = x.shape[0]==3 and x.shape[1]!=3
        if transpose:
            for xp in x.T:
                fkey = self.floatKey2D([xp[0]+2e-2, xp[1]+2e-2])
                meshZmin = UniformNodeGridFixedSizeMeshModel.point_top_vertex_map.get(fkey, 1e10)
                mzm.append(meshZmin)
            meshZminV = np.array(mzm)
            meshZminV2 = np.max([ x.T[:,2], meshZminV], axis=0)
        else:
            for xp in x:
                fkey = self.floatKey2D([xp[0]+2e-2, xp[1]+2e-2])
                meshZmin = UniformNodeGridFixedSizeMeshModel.point_top_vertex_map.get(fkey, 1e10)
                # meshZmin = self.getSubsidenceNew(self.tti, xp[0], xp[1])
                mzm.append(meshZmin)
            meshZminV = np.array(mzm)
            meshZminV2 = np.max([ x[:,2], meshZminV], axis=0)
        
        meshZminV3 = np.min([ meshZminV2, np.ones(meshZminV.shape) * meshZmax], axis=0)
        meshZminV4 = meshZminV3.copy()
        meshZminV4[meshZminV3<midpoint[2]] = meshZminV3[meshZminV3<midpoint[2]] + tol_z
        meshZminV4[meshZminV3>midpoint[2]] = meshZminV3[meshZminV3>midpoint[2]] - tol_z
        meshZminV4[meshZminV3>200000] = meshZminV3[meshZminV3>200000] - 100.0
        pl_po = x.T.copy() if transpose else x.copy()
        pl_po[:,2] = meshZminV4

        plot_points = []
        for i in range(pl_po.shape[0]):
            pt = pl_po[i,:]
            fkey = self.floatKey2D(pt)
            on_edge = UniformNodeGridFixedSizeMeshModel.point_domain_edge_map.get(fkey, True)
            dx, dy = 0.0, 0.0
            if on_edge:
                if pt[0]<midpoint[0]:
                    dx = tol
                if (pt[0]>midpoint[0]):
                    dx = -tol
                if pt[1]<midpoint[1]:
                    dy = tol
                if pt[1]>midpoint[1]:
                    dy = -tol
            plot_points.append( [ pt[0]+dx, pt[1]+dy, pt[2]] )
        plot_points = np.array(plot_points)

        bb_tree = dolfinx.geometry.BoundingBoxTree(self.mesh, self.mesh.topology.dim)
        
        points_cells = []
        points_on_proc = []
        
        # Find cells whose bounding-box collide with the the points
        cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, plot_points)
        
        
        res = []
        for i, point in enumerate(plot_points):
            # if len(colliding_cells.links(i))>0:
            #     points_on_proc.append(point)
            #     points_cells.append(colliding_cells.links(i)[0])
            if len(cell_candidates.links(i))>0:
                #points_on_proc.append(point)
                #points_cells.append(cell_candidates.links(i)[0])
                for bb in cell_candidates.links(i):
                    val = self.uh.eval(point, [bb])
                    if (not np.isnan(val)):
                        break
                res.append( val )
            else:
                logger.debug(f"need to extrapolate cell for point {i}, {point}")
                if (point[2]>200000):
                    try:
                        points_cells.append(cell_candidates.links(i)[0])
                        points_on_proc.append(point)
                    except IndexError:
                        logger.warning(f"IndexError, {point}, {cell_candidates.links(i)}" )
                        breakpoint()
                        raise
                else:
                    logger.debug(f"PING V, {point}")
                    if len(cell_candidates.links(i))==0:
                        logger.debug(f"PING V V, {point}")
                        def boundary(x):
                            return np.full(x.shape[1], True)
                        #entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
                        breakpoint()
                    points_on_proc.append(point)
                    points_cells.append(cell_candidates.links(i)[0])
        res = np.array(res)
        aa = np.any(np.isnan(res))
        bb = np.any(np.isnan(self.uh.x.array))
        if aa or bb:
            logger.debug(f"aa {aa},  bb {bb}")
            breakpoint()

        if transpose:
            assert res.flatten().shape[0] == x.shape[1]
        else:
            assert res.flatten().shape[0] == x.shape[0]
        return res.flatten()


def global_except_hook(exctype, value, traceback):
    """https://github.com/chainer/chainermn/issues/236
    """
    try:
        sys.stderr.write("\n*****************************************************\n")
        sys.stderr.write("Uncaught exception was detected on rank {}. \n".format(
            MPI.COMM_WORLD.Get_rank()))
        from traceback import print_exception
        print_exception(exctype, value, traceback)
        sys.stderr.write("*****************************************************\n\n\n")
        sys.stderr.write("\n")
        sys.stderr.write("Calling MPI_Abort() to shut down MPI processes...\n")
        sys.stderr.flush()
    finally:
        try:       
            MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            sys.stderr.write("*****************************************************\n")
            sys.stderr.write("Sorry, we failed to stop MPI, this process will hang.\n")
            sys.stderr.write("*****************************************************\n")
            sys.stderr.flush()
            raise e

sys.excepthook = global_except_hook

def run_3d( builder:Builder, parameters:Parameters,  start_time=182, end_time=0, pad_num_nodes=0,
            out_dir = "out-mapA/",sedimentsOnly=False, writeout=True, base_flux=None):
    logger.setLevel(10)  # numeric level equals DEBUG
    comm = MPI.COMM_WORLD
    builder=interpolate_all_nodes(builder)
    nums = 4
    dt = parameters.myr2s / nums # time step is 1/4 of 1Ma
    mms2 = []
    mms_tti = []
    tti = 0
    # base_flux = 0.0033
    writeout_final = True
    time_solve = 0.0
    with Bar('Processing...',check_tty=False, max=(start_time-end_time)) as bar:
        for tti in range(start_time, end_time-1,-1): #start from oldest
            rebuild_mesh = (tti==start_time)
            if rebuild_mesh:
                logger.info(f"Rebuild/reload mesh at {tti}")          
                mm2 = UniformNodeGridFixedSizeMeshModel(builder, parameters,sedimentsOnly, padding_num_nodes=pad_num_nodes)
                mm2.buildMesh(tti)
                if (base_flux is not None):
                    mm2.baseFluxMagnitude = base_flux
            else:
                logger.info(f"Re-generating mesh vertices at {tti}")
                tic()
                mm2.updateMesh(tti, optimized=True)
                toc(msg="update mesh")
            mm2.useBaseFlux = (base_flux is not None)
            mm2.baseFluxMagnitude = base_flux

            if ( len(mms2) == 0):
                tic()
                mm2.useBaseFlux = False
                mm2.setupSolverAndSolve(n_steps=40, time_step = 314712e8 * 2e2, skip_setup=False)   
                time_solve = time_solve + toc(msg="setup solver and solve")
            else:    
                mm2.useBaseFlux = (base_flux is not None)
                tic()
                mm2.setupSolverAndSolve( n_steps=nums, time_step=dt, skip_setup=(not rebuild_mesh), update_bc = mm2.useBaseFlux and (len(mms2) == 1))
                time_solve = time_solve + toc(msg="setup solver and solve")
            if (writeout):
                tic()
                mm2.writeLayerIDFunction(out_dir+"LayerID-"+str(tti)+".xdmf", tti=tti)
                mm2.writeTemperatureFunction(out_dir+"Temperature-"+str(tti)+".xdmf", tti=tti)
                # mm2.writeOutputFunctions(out_dir+"test4-"+str(tti)+".xdmf", tti=tti)
                toc(msg="write function")
            
            mms2.append(mm2)
            mms_tti.append(tti)
            bar.next()
    logger.info(f"total time solve 3D: {time_solve}")
    if (writeout_final):
        comm.Barrier()
        if comm.rank>=1:
            comm.send(mm2.mesh.topology.index_map(0).local_to_global(list(range(mm2.mesh.geometry.x.shape[0]))) , dest=0, tag=((comm.rank-1)*10)+21)
            comm.send(mm2.mesh_reindex, dest=0, tag=((comm.rank-1)*10)+23)
            comm.send(mm2.mesh_vertices_age, dest=0, tag=((comm.rank-1)*10)+25)
            comm.send(mm2.posarr, dest=0, tag=((comm.rank-1)*10)+20)
            comm.send(mm2.Tarr, dest=0, tag=((comm.rank-1)*10)+24)
        if comm.rank==0:
            mm2.receive_mpi_messages()
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            # EPCfilename = mm2.write_hexa_mesh_resqml("temp/", end_time)
            # logger.info(f"RESQML model written to: {EPCfilename}")
            EPCfilename_ts = mm2.write_hexa_mesh_timeseries(out_dir)
            logger.info(f"RESQML partial model with timeseries written to: {EPCfilename_ts}")
            read_mesh_resqml_hexa(EPCfilename_ts)  # test reading of the .epc file
    return mm2
