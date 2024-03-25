import numpy as np
from mpi4py import MPI
from scipy.interpolate import LinearNDInterpolator
from typing import Iterator, List, Literal

from warmth.build import single_node
from warmth.logging import logger
from warmth3d.Helpers import  NodeParameters1D, top_crust,top_sed,thick_crust,  top_lith, top_asth, top_sed_id, bottom_sed_id
from warmth3d.resqpy_helpers import write_tetra_grid_with_properties, write_hexa_grid_with_properties

#
# RHP: in sediments: constant, part of the sediment lithology parametrization (e.g. 2.301755e-09)
# RHP: none in lith and aesth
# RHP: in crust:  total HP in crust estimated at the start of the rift from the difference in surface and base heat flows (shf and qbase);
#      this total HP at rift start will decrease with the crust shortening..  The total HP is not evenly distributed in the crust, but
#      follows an exp-decay
#      NOTE: the default values of surface and base heat flow imply zero RHP in the crust
#


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

    times = range(node.result._depth.shape[1])
    if node.subsidence is None:
        node.subsidence = np.sum( np.array( [ [node.result.seabed(t) for t in times] * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.crust_ls is None:
        node.crust_ls = np.sum( np.array( [ [node.result.crust_thickness(t) for t in times] * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.lith_ls is None:
        node.lith_ls = np.sum( np.array( [ [node.result.lithosphere_thickness(t) for t in times] * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 

    if node.beta is None:
        node.beta = np.sum( np.array( [node.beta * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.kAsth is None:
        node.kAsth = np.sum( np.array( [node.kAsth * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node.kLith is None:
        node.kLith = np.sum( np.array( [node.kLith * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    if node._depth_out is None:
        node._depth_out = np.sum([node.result._depth_out*w for n,w in zip(interpolationNodes[0:1], [1] )], axis=0)
    if node.temperature_out is None:
        node.temperature_out = np.sum([n.result.temperature_out*w for n,w in zip(interpolationNodes[0:1], [1] )], axis=0)

    if node.sed is None:
        node.sed = np.sum([n.sed*w for n,w in zip(interpolationNodes,iWeightNorm)], axis=0)
    if node.sed_thickness_ls is None:
        node.sed_thickness_ls =  node.sed[-1,1,:] - node.sed[0,0,:]    
    return node


#
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
    def __init__(self, nodeGrid, nodes, modelName="test", sedimentsOnly = False):
        self.node1D = nodes
        self.num_nodes = len(self.node1D)
        self.mesh = None

        self.modelName = modelName
        self.Temp0 = 5
        self.TempBase = 1369
        self.verbose = True
        self.minimumCellThick = 0.05
        
        self.runSedimentsOnly = sedimentsOnly

        self.numElemInCrust = 0 if self.runSedimentsOnly else 4    # split crust hexahedron into pieces
        self.numElemInLith = 0 if self.runSedimentsOnly else 2  # split lith hexahedron into pieces
        self.numElemInAsth = 0 if self.runSedimentsOnly else 2  # split asth hexahedron into pieces


        self.num_nodes_x = nodeGrid.num_nodes_x
        self.num_nodes_y = nodeGrid.num_nodes_y
        self.convexHullEdges = []
        for i in range(self.num_nodes_x-1):
            edge = [i, i+1]
            self.convexHullEdges.append(edge)
            edge = [i+(self.num_nodes_y-1*self.num_nodes_x), i+1+(self.num_nodes_y-1*self.num_nodes_x)]
            self.convexHullEdges.append(edge)
        for i in range(self.num_nodes_y-1):
            edge = [i*self.num_nodes_x, (i+1)*self.num_nodes_x]
            self.convexHullEdges.append(edge)
            edge = [i*self.num_nodes_x + (self.num_nodes_x-1), (i+1)*self.num_nodes_x+ (self.num_nodes_x-1)]
            self.convexHullEdges.append(edge)

        self.useBaseFlux = False
        self.baseFluxMagnitude = 0.06

        self.mesh0_geometry_x = None
        self.CGorder = 1

        self.layer_id_per_vertex = None
        self.thermalCond = None
        self.mean_porosity = None
        self.c_rho = None
        self.numberOfSediments = self.node1D[0].sed.shape[0]
        # self.globalSediments = self.node1D[0].sediments.copy()

        self.parameters1D = self.node1D[0].parameters if hasattr(self.node1D[0], 'parameters') else NodeParameters1D()
        self.interpolators = {}
        print("Using 1D Node parameters", self.parameters1D)
   
    def write_tetra_mesh_resqml( self, out_path):
        """Prepares arrays and calls the RESQML output helper function:  the lith and aesth are removed, and the remaining
           vertices and cells are renumbered;  the sediment properties are prepared for output.

           out_path: string: path to write the resqml model to (.epc and .h5 files)

           returns the filename (of the .epc file) that was written 
        """            
        import dolfinx
        def boundary(x):
            return np.full(x.shape[1], True)
        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)

        # self.mesh.geometry.x[:] = self.mesh_vertices[self.mesh_reindex].copy()

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
                    print("unusual Y coordinate:", minY, self.node1D[self.node_index[i]].Y, i, self.node_index[i], self.node1D[self.node_index[i]])
                for ti in t:
                    p_to_keep.add(ti)

        # poro0_per_cell = np.array( [ self.getSedimentPropForLayerID('phi', lid) for lid in lid_to_keep ] )
        # decay_per_cell = np.array( [ self.getSedimentPropForLayerID('decay', lid) for lid in lid_to_keep ])
        # density_per_cell = np.array( [ self.getSedimentPropForLayerID('solidus', lid) for lid in lid_to_keep ])
        # cond_per_cell = np.array( [ self.getSedimentPropForLayerID('k_cond', lid) for lid in lid_to_keep ])
        # rhp_per_cell = np.array( [ self.getSedimentPropForLayerID('rhp', lid) for lid in lid_to_keep ])
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
        
        from os import path
        filename = path.join(out_path, self.modelName+'_'+str(self.tti)+'.epc')
        write_tetra_grid_with_properties(filename, np.array(points_cached), tet_renumbered, "tetramesh",
            np.array(T_per_vertex), np.array(age_per_vertex), poro0_per_cell, decay_per_cell, density_per_cell,
            cond_per_cell, rhp_per_cell, lid_per_cell)
        return filename


    def write_hexa_mesh_resqml( self, out_path):
        """Prepares arrays and calls the RESQML output helper function for hexa meshes:  the lith and aesth are removed, and the remaining
           vertices and cells are renumbered;  the sediment properties are prepared for output.

           out_path: string: path to write the resqml model to (.epc and .h5 files)

           returns the filename (of the .epc file) that was written 
        """            
        x_original_order = self.mesh.geometry.x[:].copy()
        reverse_reindex_order = np.arange( self.mesh_vertices.shape[0] )
        for ind,val in enumerate(self.mesh_reindex):
            x_original_order[val,:] = self.mesh.geometry.x[ind,:]
            reverse_reindex_order[val] = ind
        hexaHedra, hex_data_layerID, hex_data_nodeID = self.buildHexahedra()

        hexa_to_keep = []
        p_to_keep = set()
        lid_to_keep = []
        cond_per_cell = []
        cell_id_to_keep = []
        for i,h in enumerate(hexaHedra):
            lid0 = hex_data_layerID[i]
            # 
            # discard aesth and lith (layer IDs -2, -3)
            #
            if (lid0>=-1) and (lid0<100):
                hexa_to_keep.append(h)
                lid_to_keep.append(lid0)
                # cell_id_to_keep.append(self.node_index[i])
                cell_id_to_keep.append(hex_data_nodeID[i])
                minY = np.amin(np.array ( [x_original_order[hi,1] for hi in h] ))
                if abs( self.node1D[hex_data_nodeID[i]].Y - minY)>1:
                    print("weird Y:", minY, self.node1D[hex_data_nodeID[i]].Y, abs( self.node1D[hex_data_nodeID[i]].Y - minY), i, hex_data_nodeID[i])
                    breakpoint()
                # if (minY>40000):
                #     pp = self.getSedimentPropForLayerID('phi', lid0, hex_data_nodeID[i])
                #     if (pp<0.7) and lid0>=0:
                #         print("weird phi: ", pp, minY, self.node1D[hex_data_nodeID[i]].Y, abs( self.node1D[hex_data_nodeID[i]].Y - minY), i, hex_data_nodeID[i])
                #         breakpoint()
                k_cond_mean = []
                for hi in h:
                    p_to_keep.add(hi)
                    k_cond_mean.append(self.thermalCond.x.array[hi])
                cond_per_cell.append( np.mean(np.array(k_cond_mean)))

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
                points_cached.append(x_original_order[i,:])
        hexa_renumbered = [ [point_original_to_cached[i] for i in hexa] for hexa in hexa_to_keep ]
        
        for i,h in enumerate(hexa_renumbered):
            minY = np.amin(np.array ( [np.array(points_cached)[hi,1] for hi in h] ))
            poro0 = poro0_per_cell[i]
            lid0  = lid_to_keep[i]
            # if (minY>40000) and poro0 < 0.8 and lid0>=0:
            #     print("problem A", minY, poro0, i, h)
            #     breakpoint()
            # if (minY<40000) and poro0 > 0.8:
            #     print("problem B", minY, poro0, i, h)
            #     breakpoint()


        T_per_vertex = [ self.uh.x.array[reverse_reindex_order[i]] for i in range(self.mesh.geometry.x.shape[0]) if i in p_to_keep  ]
        age_per_vertex = [ self.mesh_vertices_age[reverse_reindex_order[i]] for i in range(self.mesh.geometry.x.shape[0]) if i in p_to_keep  ]

        from os import path
        filename_hex = path.join(out_path, self.modelName+'_hexa_'+str(self.tti)+'.epc')
        write_hexa_grid_with_properties(filename_hex, np.array(points_cached), hexa_renumbered, "hexamesh",
            np.array(T_per_vertex), np.array(age_per_vertex), poro0_per_cell, decay_per_cell, density_per_cell,
            cond_per_cell, rhp_per_cell, lid_per_cell)
        return filename_hex

    def getNodeX(self, node_index):
        return self.node1D[node_index].X

    def getNodeY(self, node_index):
        return self.node1D[node_index].Y    

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

    def getTopOfLithAtNode(self, tti, node_index):
        """Returns crust-lith boundary depth at the given time at the given node
        """           
        z0 = top_lith( self.node1D[node_index], tti ) if not self.runSedimentsOnly else 0
        return z0

    def getTopOfAsthAtNode(self, tti, node_index):
        """Returns crust-lith boundary depth at the given time at the given node
        """           
        z0 = top_asth( self.node1D[node_index], tti ) if not self.runSedimentsOnly else 0
        return z0

    #
    def getSedimentPropForLayerID(self, property, layer_id, node_index):
        """
        """           
        assert property in ['k_cond', 'rhp', 'phi', 'decay', 'solidus', 'liquidus'], "Unknown property " + property
        if (layer_id>=0) and (layer_id<self.numberOfSediments):
            # node_index = ind2 // (self.numberOfSediments+6)  # +6 because crust, lith, aest are each cut into two
            node = self.node1D[node_index]
            phi = node.sediments[property][layer_id]
            # if (property=='phi') and phi>0.7 and node.Y<40000:
            #     print("phi", property, phi, node_index, node)
            #     breakpoint()
            # if (property=='phi') and phi<0.7 and node.Y>40000:
            #     print("phi", property, phi, node_index, node)
            #     breakpoint()
            # phi = self.globalSediments[property][layer_id]
            # assert abs(phi-phi0)<1e-6
            return phi
        if (layer_id<=-1) and (layer_id>=-3):
            lid = -layer_id -1
            if (property=='k_cond'):
                return [0.1,0.1,0.1][lid]
            if (property=='rhp'):
                return [0.0,0.0,0.0][lid]  # RHP in crust is not yet supported!
            if (property=='phi'):
                return [0.1,0.0,0.0][lid]   # porosity for crust, lith, aest
            if (property=='decay'):
                return [0.5,0.5,0.5][lid]   # porosity decay for crust, lith, aest
            if (property=='solidus'):
                return [0.5,0.5,0.5][lid]   # solid density decay for crust, lith, aest
            if (property=='liquidus'):
                return [0.5,0.5,0.5][lid]   # solid density decay for crust, lith, aest
        return np.nan

    def porosity0ForLayerID(self, layer_id, node_index):
        """Porosity (at surface) conductivity value for the given layer index
        """           
        if (layer_id==-1):
            return 0.0,0.0 # porosity (at surface) of crust
        if (layer_id==-2):
            return 0.0,0.0 # porosity (at surface) of lith
        if (layer_id==-3):
            return 0.0,0.0  # porosity (at surface) of aesth
        if (layer_id>=0) and (layer_id<self.numberOfSediments):
            # assert (node_index < len(self.node1D)) and node_index > 0
            node = self.node1D[node_index]
            phi = node.sediments.phi[layer_id]
            decay = node.sediments.decay[layer_id]
            return phi, decay
        return 0.0, 0.0

    def cRhoForLayerID(self, ss, node_index):
        #
        # prefactor 1000 is the heat capacity.. assumed constant
        #
        if (ss==-1):
            return 1000*self.parameters1D.crustsolid
        if (ss==-2):
            return 1000*self.parameters1D.lithsolid
        if (ss==-3):
            return 1000*self.parameters1D.crustsolid
        if (ss>=0) and (ss<self.numberOfSediments):
            node = self.node1D[node_index]
            rho = node.sediments.solidus[ss]
            return 1000*rho
        return 1000*self.parameters1D.crustsolid

    def kForLayerID(self, ss, node_index):
        """Thermal conductivity for a layer ID index
        """
        # ind0 = cell_id
        # ind1 = self.cell_index[cell_id]
        # ind2 = np.where(self.cell_index==cell_id)[0][0]
        # if self.cell_data_layerID[ind2] != ss:
        #     print("ind", cell_id, ind0, ind1, ind2)
        #     print(len(self.cell_index), np.amin(self.cell_index), np.amax(self.cell_index))
        #     print(len(self.cell_data_layerID), np.amin(self.cell_data_layerID), np.amax(self.cell_data_layerID))
        #     print("kForLayerID", ss, ind0, ind1, ind2, self.cell_data_layerID[self.cell_index[cell_id]] )
        # assert self.cell_data_layerID[ind2] == ss
        if (ss==-1):
            return self.parameters1D.kCrust
        if (ss==-2):
            return self.parameters1D.kLith
        if (ss==-3):
            return self.parameters1D.kAsth
        if (ss>=0) and (ss<self.numberOfSediments):
            # kc = self.globalSediments.k_cond[ss]
            # node_index = cell_id // (self.numberOfSediments+6)  # +6 because crust, lith, aest are each cut into two
            if (node_index > len(self.node1D)-1):
                print("cell ID", node_index, len(self.node1D))
                breakpoint()
            node = self.node1D[node_index]
            kc = node.sediments.k_cond[ss]
            return kc
        return self.parameters1D.kCrust

    def rhpForLayerID(self, ss, node_index):
        """Radiogenic heat production for a layer ID index
        """
        if (ss==-1):
            return 0
        if (ss==-2):
            return 0
        if (ss==-3):
            return 0
        if (ss>=0) and (ss<self.numberOfSediments):
            node = self.node1D[node_index]
            kc = node.sediments.rhp[ss]
            return kc
        return 0


    def buildVertices(self, time_index=0, useFakeEncodedZ=False):
        """Determine vertex positions, node-by-node.
           For every node, the same number of vertices is added (one per sediment, one per crust, lith, asth, and one at the bottom)
           Degenerate vertices (e.g. at nodes where sediment is yet to be deposited or has been eroded) are avoided by a small shift, kept in self.sed_diff_z
           
           When the option useFakeEncodedZ is set, the z-values are repurposed to encode the index of the vertex in the original, deterministic indexing.
           This is necessary because dolfinx will re-index the vertices upon mesh generation. 
        """           
        tti = time_index
        self.tti = time_index

        self.mesh_vertices_0 = []
        self.sed_diff_z = []
        self.mesh_vertices_age_unsorted = []
        
        mean_top_of_lith = np.mean( np.array( [ self.getTopOfLithAtNode(tti, i) for i in range(len(self.node1D)) ] ) )
        mean_top_of_asth = np.mean( np.array( [ self.getTopOfAsthAtNode(tti, i) for i in range(len(self.node1D)) ] ) )
        logger.info(f'Time {tti}: mean top of lith: {mean_top_of_lith:.1f}; of aesth: {mean_top_of_asth:.1f} ')

        for k in range(self.num_nodes):
            node = self.node1D[k]
            top_of_sediments = top_sed(self.node1D[k], tti)
            self.mesh_vertices_0.append( [ self.getNodeX(k), self.getNodeY(k), top_of_sediments - 0.0*(self.numberOfSediments+1) ] )
            self.sed_diff_z.append(-self.minimumCellThick*(self.numberOfSediments+1))
            
            # self.mesh_vertices_age_unsorted.append(self.globalSediments.topage[0])  # append top age of top sediment
            self.mesh_vertices_age_unsorted.append(node.sediments.topage[0])  # append top age of top sediment
            for ss in range(self.numberOfSediments):
                base_of_current_sediments = bottom_sed_id(node, ss, tti)
                if self.runSedimentsOnly:
                    zpos = base_of_current_sediments
                else:
                    zpos = top_of_sediments + base_of_current_sediments
                vert = np.array([ self.getNodeX(k),self.getNodeY(k), zpos ])
                self.mesh_vertices_0.append( vert )
                self.sed_diff_z.append(-self.minimumCellThick*(self.numberOfSediments-ss))
                # self.mesh_vertices_age_unsorted.append(self.globalSediments.baseage[ss])  # append base age of current sediment
                self.mesh_vertices_age_unsorted.append(node.sediments.baseage[ss])  # append base age of current sediment

            if not self.runSedimentsOnly:
                base_of_last_sediments = bottom_sed_id(node, self.numberOfSediments-1, tti) if (self.numberOfSediments>0) else top_of_sediments
                base_crust = mean_top_of_lith
                for i in range(1,self.numElemInCrust+1):
                    self.mesh_vertices_0.append( [ self.getNodeX(k), self.getNodeY(k), base_of_last_sediments+ (base_crust-base_of_last_sediments)*(i/self.numElemInCrust) ] )
                    self.sed_diff_z.append(0.0)
                    self.mesh_vertices_age_unsorted.append(1000)

                base_lith = mean_top_of_asth
                for i in range(1,self.numElemInLith+1):
                    self.mesh_vertices_0.append( [ self.getNodeX(k), self.getNodeY(k), base_crust+ (base_lith-base_crust)*(i/self.numElemInLith) ] )
                    self.sed_diff_z.append(0.0)
                    self.mesh_vertices_age_unsorted.append(1000)

                base_aest = 260000
                for i in range(1,self.numElemInAsth+1):
                    self.mesh_vertices_0.append( [ self.getNodeX(k), self.getNodeY(k), base_lith+(base_aest-base_lith)*(i/self.numElemInAsth) ] )
                    self.sed_diff_z.append(0.0)
                    self.mesh_vertices_age_unsorted.append(1000)
            
            # self.mesh_vertices_age_unsorted.extend([1000,1000,1000])  # append age of base of crust, list, asth

        assert len(self.mesh_vertices_0) % self.num_nodes ==0
        self.mesh_vertices_0 = np.array(self.mesh_vertices_0)
        self.sed_diff_z = np.array(self.sed_diff_z)
        self.mesh_vertices = self.mesh_vertices_0.copy()
        self.mesh_vertices[:,2] = self.mesh_vertices_0[:,2] + self.sed_diff_z
        if (useFakeEncodedZ):
            self.mesh_vertices[:,2] = np.ceil(self.mesh_vertices[:,2])*1000 + np.array(list(range(self.mesh_vertices.shape[0])))*0.001
            # zz = self.mesh_vertices[:,2].copy()
            # zz2=np.mod(zz,1000)
            # # mesh_reindex = (1e-4+zz2*10).astype(np.int32)

    def updateVertices(self):
        """Update the mesh vertex positions using the values in self.mesh_vertices, and using the known dolfinx-induded reindexing
        """        
        self.mesh.geometry.x[:] = self.mesh_vertices[self.mesh_reindex].copy()
        self.mesh_vertices_age = np.array(self.mesh_vertices_age_unsorted)[self.mesh_reindex].copy()
        self.mesh0_geometry_x = self.mesh.geometry.x.copy()      
        self.mesh_vertices[:,2] = self.mesh_vertices_0[:,2] - self.sed_diff_z
        self.updateTopVertexMap()
        if self.runSedimentsOnly: 
            self.updateBottomVertexMap()
        self.mesh_vertices[:,2] = self.mesh_vertices_0[:,2] + self.sed_diff_z

    def buildMesh(self,tti):
        """Construct a new mesh at the given time index tti, and determine the vertex re-indexing induced by dolfinx
        """        
        self.tti = tti
        self.buildVertices(time_index=tti, useFakeEncodedZ=True)
        self.constructMesh()
        self.buildVertices(time_index=tti, useFakeEncodedZ=False)
        self.updateVertices()        

    def updateMesh(self,tti):
        """Construct the mesh positions at the given time index tti, and update the existing mesh with the new values
        """   
        assert self.mesh is not None
        self.tti = tti
        self.buildVertices(time_index=tti, useFakeEncodedZ=False)
        self.updateVertices()        

    def buildHexahedra(self):
        xpnum = self.num_nodes_x
        ypnum = self.num_nodes_y

        nodeQuads = []
        for j in range(ypnum-1):
            for i in range(xpnum-1):
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
                lid = s
                if (s >= self.numberOfSediments):
                    ss = s - self.numberOfSediments
                    # lid = -((s+1)-self.numberOfSediments)
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
        import meshio
        
        self.thermalCond = None

        xpnum = self.num_nodes_x
        ypnum = self.num_nodes_y

        # nodeQuads = []
        # for j in range(ypnum-1):
        #     for i in range(xpnum-1):
        #         i0 = j * (xpnum)+i
        #         q = [ i0, i0+1, i0 + xpnum+1, i0 + xpnum ]
        #         nodeQuads.append(q)

        # v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        # assert len(self.mesh_vertices) % self.num_nodes ==0

        # hexaHedra = []
        # hex_data_layerID = []
        # for q in nodeQuads:
        #     for s in range(v_per_n-1):
        #         h = []
        #         #
        #         # TODO: cut crust, lith, aesth into at least two hexahedra each
        #         # 
        #         for i in range(4):
        #             i0 = q[i]*v_per_n + s+1
        #             h.append(i0)
        #         for i in range(4):
        #             h.append(q[i]*v_per_n + s)
        #         hexaHedra.append(h)
        #         lid = s
        #         if (s >= self.numberOfSediments):
        #             ss = s - self.numberOfSediments
        #             # lid = -((s+1)-self.numberOfSediments)
        #             if (ss>=0) and (ss<self.numElemInCrust):
        #                 lid = -1                        
        #             if (ss>=self.numElemInCrust) and (ss < self.numElemInCrust+self.numElemInLith):
        #                 lid = -2                        
        #             if (ss>=self.numElemInCrust+self.numElemInLith) and (ss<self.numElemInCrust+self.numElemInLith+self.numElemInAsth):
        #                 lid = -3                        
        #         hex_data_layerID.append(lid)

        v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        hexaHedra, hex_data_layerID, hex_data_nodeID = self.buildHexahedra()


        # https://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
        tetsplit1 = [ [1,2,4,8], [1,2,5,8], [4,8,2,3], [2,3,7,8], [2,5,6,8], [2,6,7,8] ]
        tetsplit0 = [ [ p-1 for p in v ] for v in tetsplit1 ]

        lid_per_node = [100]
        for i in range(self.numberOfSediments):
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
                # candidate_tet_pos = np.array(self.mesh_vertices[candidate_tet_ind,:] )
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
        
        # mesh.write( "mesh/"+self.modelName+"_mesh.vtk")
        fn = "mesh/"+self.modelName+"_mesh.xdmf"
        mesh.write( fn )            
        import dolfinx   
        enc = dolfinx.io.XDMFFile.Encoding.HDF5
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, fn, "r", encoding=enc) as file:
            self.mesh = file.read_mesh(name="Grid" )
            aa=file.read_meshtags(self.mesh, name="Grid")
            self.cell_data_layerID = np.floor(aa.values.copy()*1e-7)-3
            self.node_index = np.mod(aa.values.copy(),1e7).astype(np.int32)
        # print("min max", np.amin(self.node_index), np.amax(self.node_index), np.amin(cell_index), np.amax(cell_index))
        # print( len(self.cell_data_layerID), np.amin(self.cell_data_layerID), np.amax(self.cell_data_layerID) )
        # print(len(hexaHedra), c_count)
        # breakpoint()

        #
        # obtain original vertex order as encoded in z-pos digits
        zz  = self.mesh.geometry.x[:,2].copy()
        zz2 = np.mod(zz,1000)
        self.mesh_reindex = (1e-4+zz2*1000).astype(np.int32)
        self.mesh0_geometry_x = self.mesh.geometry.x.copy()

        # for i,c in enumerate(self.node_index):
        #     h1,lid1 = hexaHedra[c], hex_data_layerID[c]
        #     lid2 = self.node_index[i]
        #     if (lid2 != self.node_index[i])
        #     assert lid2==h1[0]


    def normalizedZ(self, z):
        # return (z-4000) / (4000-0 )
        Zmin, Zmax = np.amin(z), np.amax(z)
        return (z-Zmin) / (Zmax-Zmin)

    def TemperatureGradient(self, x):
        self.averageLABdepth = np.mean(np.array([ top_asth(n, self.tti) for n in self.node1D]))
        Zmin, Zmax = np.amin(x[2,:]), np.amax(x[2,:])
        nz = (x[2,:] - Zmin) / (self.averageLABdepth - Zmin)
        nz[nz>1.0] = 1.0
        res = nz * (self.TempBase-self.Temp0) + self.Temp0
        for i in range(x.shape[1]):
            p = x[:,i]
            fkey = self.floatKey2D(p+[2e-2, 2e-2,0.0])
            dz = UniformNodeGridFixedSizeMeshModel.point_top_vertex_map.get(fkey, 1e10)
            Zmin0 = dz if (dz<1e9) else np.amin(x[2,:])
            nz0 = (p[2] - Zmin0) / (self.averageLABdepth - Zmin0)
            nz0 = min(nz0, 1.0)
            res[i] = nz0 * (self.TempBase-self.Temp0) + self.Temp0
            if (p[2]>250000):
                res[i] = 1369
            # Zmax = np.amax(x[2,:])
        # res[x[2,:]<self.Zmin] = self.Temp0 + (( x[2,:][x[2,:]<self.Zmin] - self.Zmin)/1000)*12
        return res

    def getLABTemperature(self):
        return self.parameters1D.Tm

    def TemperatureStep(self, x):
        self.averageLABdepth = np.mean(np.array([ top_asth(n, self.tti) for n in self.node1D]))
        # self.TempBase = self.getLABTemperature() + (260000-self.averageLABdepth) * 0.3e-3
        self.TempBase = 1369
        #
        # self.current_node.kAsth = self.current_node.qbase/self._parameters.adiab
        # self.adiab: float = 0.3e-3
        # qbase: 30e-3
        # 
        # out = np.floor(1.99*self.normalizedZ(x[2])) * (self.TempBase-self.Temp0) + self.Temp0
        return np.floor(1.99*self.normalizedZ(x[2])) * (self.TempBase-self.Temp0) + self.Temp0

    def TemperatureFromNode(self, x):
        dd = self.node1D[0].depth_out[:,self.tti]
        tt = self.node1D[0].temperature_out[:,self.tti]
        zz = x[2]
        val = np.interp(zz, dd, tt)
        # print("node1D 0 data", self.node1D[0].depth_out.shape, self.node1D[0].temperature_out.shape, x, val)
        return val

    def floatKey2D(self, xp):
        """Returns a tuple of integers from the input xy-point.  This is useful as a dict key for fast lookups. 
        """   
        return( int(xp[0]*10), int(xp[1]*10) )

    def updateSedimentsConductivity(self):
        import dolfinx     
        # mean_porosity_arr, sed_idx_arr = self._sediments_mean_porosity(
        #     xsed,  idsed)
        # # density_sed = self._sediment_density(
        # #     mean_porosity_arr, self.current_node.sediments["solidus"].values[sed_idx_arr])
        # conductivity_sed = self._sediment_conductivity(
        #     mean_porosity_arr, self.current_node.sediments["k_cond"].values[sed_idx_arr])
        def boundary(x):
            return np.full(x.shape[1], True)

        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)

        # p0 = self.mesh.geometry.x[tet,:]
        # midp = np.sum(p0,1)*0.25   # midpoints of tetrahedra
        # poro = [ self.porosity0ForLayerID(lid)[0] for lid in ls]
        # self.porosity0.x.array[:] = np.array(poro, dtype=PETSc.ScalarType).flatten()
        # self.porosityAtDepth.x.array[:] = np.array(poro, dtype=PETSc.ScalarType).flatten()

        self.layer_id_per_vertex = [ [] for _ in range(self.mesh.geometry.x.shape[0]) ]
        for i,t in enumerate(tet):
            lidval = int(self.layerIDsFcn.x.array[i])
            if (lidval<0):
                # only relevant for sediment
                continue
            zpos = [ self.mesh.geometry.x[ti,2] for ti in t]
            top_km = np.amin(zpos) / 1e3
            bottom_km = np.amax(zpos) / 1e3
            poro0 = self.porosity0.x.array[i]
            decay = self.porosityDecay.x.array[i]

            # f1 = sed_phi0[sed_id_filtered] / (sed_decay[sed_id_filtered] * thickness_km_arr)
            # f2 = np.exp(-1 * sed_decay[sed_id_filtered] * top_km_arr) - np.exp(-1 * sed_decay[sed_id_filtered] * base_km_arr)
            f1 = poro0 / (decay * (bottom_km-top_km))
            f2 = np.exp(-1 * decay * top_km) - np.exp(-1 * decay * bottom_km)
            mean_porosity = f1*f2
            self.porosityAtDepth.x.array[i] = mean_porosity
            conductivity_effective = self.kForLayerID(lidval, self.node_index[i]) ** (1 - mean_porosity) * 0.6 ** mean_porosity
            self.thermalCond.x.array[i] = conductivity_effective

    def sedimentsConductivitySekiguchi(self): #mean_porosity, conductivity, temperature_C):
        """Scale surface conductivity of sediments to effective conductivity of sediments at depth. Scaler of 0.6 based on Allen & Allen p345. porosity dependent conductivity
        Args:
            mean_porosity (npt.NDArray[np.float64]): Mean porosity of sediments at depth
            conductivity (npt.NDArray[np.float64]): Conductivity of sediments at 20C
        Returns:
            npt.NDArray[np.float64]: Effective conductivity of sediments
        """

        import dolfinx     
        def boundary(x):
            return np.full(x.shape[1], True)

        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)
        self.layer_id_per_vertex = [ [] for _ in range(self.mesh.geometry.x.shape[0]) ]
        for i,t in enumerate(tet):
            lidval = int(self.layerIDsFcn.x.array[i])
            if (lidval<0):
                # only relevant for sediment
                continue
            zpos = [ self.mesh.geometry.x[ti,2] for ti in t]
            top_km = np.amin(zpos) / 1e3
            bottom_km = np.amax(zpos) / 1e3
            poro0 = self.porosity0.x.array[i]
            decay = self.porosityDecay.x.array[i]
            f1 = poro0 / (decay * (bottom_km-top_km))
            f2 = np.exp(-1 * decay * top_km) - np.exp(-1 * decay * bottom_km)
            mean_porosity = f1*f2
            self.porosityAtDepth.x.array[i] = mean_porosity
            
            # cond_local = self.kForLayerID(lidval, i)
            cond_local = self.kForLayerID(lidval, self.node_index[i])

            # conductivity_effective = self.kForLayerID(lidval) ** (1 - mean_porosity) * 0.6 ** mean_porosity

            temperature_C = np.mean(np.array([ self.uh.x.array[ti] for ti in t]))
            temperature_K = 273.15 + temperature_C
            conductivity_effective = 1.84 + 358 * ( (1.0227*cond_local)-1.882) * ((1/temperature_K)-0.00068)
            conductivity_effective = conductivity_effective * (1.0-mean_porosity) # * np.sqrt(1-mean_porosity)
            # if (i % 200 == 0):
            #     print("Sekiguchi", i, self.tti,  conductivity_effective, conductivity_effective / (1.0-mean_porosity), cond_local, self.kForLayerID(lidval), temperature_C, mean_porosity, poro0, top_km, bottom_km)
            #     print("Sekiguchi", i, top_km, bottom_km, f1, f2)

            self.thermalCond.x.array[i] = conductivity_effective
            self.mean_porosity.x.array[i] = mean_porosity
            self.c_rho.x.array[i] = 1000*((self.c_rho.x.array[i]/1000) * (1-mean_porosity) + mean_porosity*1000)

        self.rhpFcn.x.array[:] = np.multiply( self.rhp0.x.array[:], (1.0-self.mean_porosity.x.array[:]) )

    def getCellMidpoints(self):
        import dolfinx     
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
        import dolfinx     
        from petsc4py import PETSc           

        self.mesh_vertex_layerIDs = np.full_like(self.mesh.geometry.x[:,2], 100, dtype=np.int32 )

        #
        # piecewise constant Kappa in the tetrahedra
        #
        Q = dolfinx.fem.FunctionSpace(self.mesh, ("DG", 0))  # discontinuous Galerkin, degree zero
        thermalCond = dolfinx.fem.Function(Q)
        c_rho = dolfinx.fem.Function(Q)
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

        rhps = [ self.rhpForLayerID(lid,self.node_index[i]) for i,lid in enumerate(ls)]
        rhp.x.array[:] = np.array(rhps, dtype=PETSc.ScalarType).flatten()
        self.rhp0.x.array[:] = np.array(rhps, dtype=PETSc.ScalarType).flatten()

        crs = [ self.cRhoForLayerID(lid,self.node_index[i]) for i,lid in enumerate(ls)]
        c_rho.x.array[:] = np.array(crs, dtype=PETSc.ScalarType).flatten()

        # self.node_index[i]
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
            (to be re-designed?)
        """ 
        UniformNodeGridFixedSizeMeshModel.point_top_vertex_map = {}
        v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        if not self.runSedimentsOnly:
            indices = np.where( np.mod(self.mesh_reindex,v_per_n)==0)[0]
        else:
            indices = range(self.mesh.geometry.x.shape[0])
        for i in indices:
            p = self.mesh.geometry.x[i,:]
            fkey = self.floatKey2D(p+[2e-2, 2e-2,0.0])
            dz = UniformNodeGridFixedSizeMeshModel.point_top_vertex_map.get(fkey, 1e10)
            if p[2]<dz:
                UniformNodeGridFixedSizeMeshModel.point_top_vertex_map[fkey] = p[2]

    def updateBottomVertexMap(self):
        """ Updates the point_bottom_vertex_map, used for fast lookup of subsidence values.
            (to be re-designed?)
        """ 
        UniformNodeGridFixedSizeMeshModel.point_bottom_vertex_map = {}
        v_per_n = int(len(self.mesh_vertices) / self.num_nodes)
        indices = np.where( np.mod(self.mesh_reindex,v_per_n)==0)
        # for i in indices[0]:
        for i in range(self.mesh.geometry.x.shape[0]):
            p = self.mesh.geometry.x[i,:]
            fkey = self.floatKey2D(p+[2e-2, 2e-2,0.0])
            dz = UniformNodeGridFixedSizeMeshModel.point_bottom_vertex_map.get(fkey, -1e10)
            if p[2]>dz:
                UniformNodeGridFixedSizeMeshModel.point_bottom_vertex_map[fkey] = p[2]

    def updateDirichletBaseTemperature(self):
        assert False, "to be re-implemented"
        # iuv = len(self.u_values)                 
        # i0 = np.argmin( np.abs( self.u_values[iuv-1]-1330) )
        # i1 = i0+1
        # if (i0>len(self.u_values[iuv-1])):
        #     return
        # if (self.u_values[iuv-1][i0]>1330):
        #     i0=i0-1
        #     i1 = i0+1
        # d0 = self.plot_points[2, i0]
        # d1 = self.plot_points[2, i1]
        # w = (1330-self.u_values[iuv-1].flatten()[i0]) / (self.u_values[iuv-1].flatten()[i1] - self.u_values[iuv-1].flatten()[i0])
        # w = max(min(w,1),0)
        # d_LAB = d0 + w * (d1-d0)
        
        # d_LAB = top_asth(self.node1D[0], self.tti)
        # TempBase = 1330 + (260000-d_LAB) * 0.0003
        # self.TempBase = TempBase
        # logger.info(f'update Dirichlet Base Temp:  LAB {d_LAB}m  baseTemp {TempBase}C')

    def buildDirichletBC(self):
        """ Generate a dolfinx Dirichlet Boundary condition that applies at the top and bottom vertices.
            The values at the edges are those in function self.TemperatureStep
        """ 
        import dolfinx        
        #
        # Dirichlet BC at top and bottom
        #
        self.Zmax = np.amax(self.mesh.geometry.x[:,2])
        Zmin = np.amin(self.mesh.geometry.x[:,2])
        self.averageLABdepth = np.mean(np.array([ top_sed(n, self.tti) for n in self.node1D]))
        def boundary_D_top_bottom(x):
            subs0 = self.getSubsidenceAtMultiplePos(x[0,:], x[1,:])
            xx = np.logical_or( np.abs(x[2]-subs0)<5, np.isclose(x[2], self.Zmax) )
            return xx
        def boundary_D_top(x):
            subs0 = self.getSubsidenceAtMultiplePos(x[0,:], x[1,:])
            # print("subs0", self.tti, subs0.shape, subs0[np.abs(subs0)>1].shape, subs0[np.abs(subs0)>1] )
            #print("subs0", self.tti, subs0.shape, subs0[np.abs(subs0)>1].shape, subs0[np.abs(subs0)>1] )
            xx = np.logical_or( np.abs(x[2]-subs0)<5, np.isclose(x[2], 1e6*self.Zmax) )            
            return xx
            
        if (self.useBaseFlux):
            dofs_D = dolfinx.fem.locate_dofs_geometrical(self.V, boundary_D_top)
            # print("dofs_D", self.tti, dofs_D.shape)
        else:
            dofs_D = dolfinx.fem.locate_dofs_geometrical(self.V, boundary_D_top_bottom)
        u_bc = dolfinx.fem.Function(self.V)
        u_bc.interpolate(self.TemperatureGradient)
        print("buildDirichletBC", np.amin(u_bc.x.array), np.amax(u_bc.x.array) )
        # u_bc.interpolate(self.TemperatureStep)
        bc = dolfinx.fem.dirichletbc(u_bc, dofs_D)
        return bc


    def resetMesh(self):
        self.mesh.geometry.x[:,2] = self.mesh0_geometry_x.copy()[:,2]

    def writeLayerIDFunction(self, outfilename, tti=0):
        """ Writes the mesh and the layer ID function (constant value per cell) to the given output file in XDMF format
        """         
        import dolfinx
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.layerIDsFcn, tti)

    def writePoroFunction(self, outfilename, tti=0):
        """ Writes the mesh and poro0 function (constant value per cell) to the given output file in XDMF format
        """         
        import dolfinx
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.porosity0, tti)
        # xdmf.write_function(self.thermalCond, tti)

    def writeTemperatureFunction(self, outfilename, tti=0):
        """ Writes the mesh and the current temperature solution to the given output file in XDMF format
        """         
        import dolfinx
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.u_n, tti)

    def writeOutputFunctions(self, outfilename, tti=0):
        """ Writes the mesh, layer IDs, and current temperature solution to the given output file in XDMF format
            #
            # TODO: this does not work
            #
        """         
        import dolfinx
        xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outfilename, "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(self.layerIDsFcn, tti)
        xdmf.write_function(self.u_n, tti)

    def setupSolverAndSolve(self, time_step=-1, no_steps=100, skip_setup = False, initial_state_model = None):
        """ Sets up the function spaces, output functions, input function (kappa values), boundary conditions, initial conditions.
            Sets up the heat equation in dolfinx, and solves the system in time for the given number of steps.
            
            Use skip_setup = True to continue a computation (e.g. after deforming the mesh), instead of starting one from scratch 
        """     
        import dolfinx
        import ufl
        from petsc4py import PETSc

        if (not skip_setup):
            self.resetMesh()
            self.Zmin = np.min(self.mesh_vertices, axis=0)[2]
            self.Zmax = np.max(self.mesh_vertices, axis=0)[2]
        
        # Time-dependent heat problem:
        #   time-discretized variational form with backwards Euler,
        #   see: https://fenicsproject.org/pub/tutorial/html/._ftut1006.html
    
        if (not skip_setup):
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

            # initialise both with initial condition: either a step function, or the solution from another Model instance
            if (initial_state_model is None):
                # self.u_n.interpolate(self.TemperatureStep)
                self.u_n.interpolate(self.TemperatureGradient)
                # self.u_n.interpolate(self.TemperatureFromNode)
            else:
                self.u_n.x.array[:] = initial_state_model.uh.x.array[:].copy()
            self.uh.x.array[:] = self.u_n.x.array[:].copy()


        self.thermalCond, self.c_rho, self.layerIDsFcn, self.rhpFcn = self.buildKappaAndLayerIDs()
        assert not np.any(np.isnan(self.thermalCond.x.array))
        
        # self.updateSedimentsConductivity()
        self.sedimentsConductivitySekiguchi()

        self.bc = self.buildDirichletBC()

        t=0
        dt = time_step if (time_step>0) else  3600*24*365 * 5000000
        num_steps = no_steps

        #
        #  solver setup, see:
        #  https://jorgensd.github.io/dolfinx-tutorial/chapter2/diffusion_code.html
        #

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = self.c_rho*u*v*ufl.dx + dt*ufl.dot(self.thermalCond*ufl.grad(u), ufl.grad(v)) * ufl.dx

        # f = dolfinx.fem.Constant(self.mesh, PETSc.ScalarType(source))  # source term 
        f = self.rhpFcn  # source term, units of W/m^3
        print("mean RHP", np.mean(self.rhpFcn.x.array[:]))

        if ( self.useBaseFlux ):
            # baseFlux = 0.03 if (self.tti>50) else 0.03 
            baseFlux = self.baseFluxMagnitude
            # define Neumann condition: constant flux at base
            # expression g defines values of Neumann BC (heat flux at base)
            x = ufl.SpatialCoordinate(self.mesh)

            domain_c = dolfinx.fem.Function(self.V)

            if (self.CGorder>1):
                def marker(x):
                    print(x.shape, x)
                    return x[2,:]>3990
                facets = dolfinx.mesh.locate_entities_boundary(self.mesh, dim=(self.mesh.topology.dim - 2),
                                        marker=marker )
                print(type(facets), facets.shape)
                dofs = dolfinx.fem.locate_dofs_topological(V=self.V, entity_dim=1, entities=facets)
                print( type(dofs), len(dofs))
                print(facets.shape, dofs.shape)
                if (len(facets)>0):
                    print( np.amax(facets))
                if (len(dofs)>0):
                    print( np.amax(dofs))
                print(type(domain_c.x.array), len(domain_c.x.array))
                domain_c.x.array[ dofs ] = 1
            else:
                basepos = self.getBaseAtMultiplePos(self.mesh.geometry.x[:,0], self.mesh.geometry.x[:,1])
                domain_c.x.array[  self.mesh.geometry.x[:,2] > basepos*0.99 ] = 1
                xmin, xmax = np.amin(self.mesh.geometry.x[:,0]), np.amax(self.mesh.geometry.x[:,0])
                ymin, ymax = np.amin(self.mesh.geometry.x[:,1]), np.amax(self.mesh.geometry.x[:,1])
                #
                # remove corners from base heat flow domain
                domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] < xmin+1, self.mesh.geometry.x[:,1] < ymin+1) ] = 0
                domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] < xmin+1, self.mesh.geometry.x[:,1] > ymax-1) ] = 0
                domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] > xmax-1, self.mesh.geometry.x[:,1] < ymin+1) ] = 0
                domain_c.x.array[  np.logical_and( self.mesh.geometry.x[:,0] > xmax-1, self.mesh.geometry.x[:,1] > ymax-1) ] = 0

            domain_zero = dolfinx.fem.Function(self.V)
            toppos = self.getSubsidenceAtMultiplePos(self.mesh.geometry.x[:,0], self.mesh.geometry.x[:,1])
            domain_zero.x.array[  self.mesh.geometry.x[:,2] < toppos+0.01 ] = 1
            print("Neumann conditions: ", self.tti, np.count_nonzero(domain_c.x.array), np.count_nonzero(domain_zero.x.array))

            g = (-1.0*baseFlux) * ufl.conditional( domain_c > 0, 1.0, 0.0 )

            # print("building L", self.c_rho.x.array[:].shape, self.u_n.x.array[:].shape, type(self.c_rho*self.u_n))
            # # print("building L", (self.c_rho*self.u_n).shape )
            # print("building L", self.rhpFcn.x.array.shape )
            L = (self.c_rho*self.u_n + dt*f)*v*ufl.dx - dt * g * v * ufl.ds    # last term reflects Neumann BC 
        else:
            L = (self.c_rho*self.u_n + dt*f)*v*ufl.dx   # no Neumann BC 

        bilinear_form = dolfinx.fem.form(a)
        linear_form = dolfinx.fem.form(L)

        A = dolfinx.fem.petsc.assemble_matrix(bilinear_form, bcs=[self.bc])
        A.assemble()
        b = dolfinx.fem.petsc.create_vector(linear_form)

        from petsc4py import PETSc
        solver = PETSc.KSP().create(self.mesh.comm)

        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        
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
            # print(pos_x, pos_y, interp([pos_x, pos_y]), interp([0,0])[0] )
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
        
        xpos = [ self.getNodeX(i) for i in range(len(self.node1D)) ]
        ypos = [ self.getNodeY(i) for i in range(len(self.node1D)) ]

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
        import dolfinx
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
        import dolfinx
        import numpy as np

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
        
        # # Choose one of the cells that contains the point
        # try:
        #     colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, plot_points)
        # except RuntimeError:
        #     print("colliding cells failed once: ", meshZmin, meshZmax, plot_points)
        #     def boundary(x):
        #         return np.full(x.shape[1], True)
        #     entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
        #     tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)            
        #     breakpoint()
        #     ## self.mesh.geometry.x[np.all([self.mesh.geometry.x[:,0]==37900, self.mesh.geometry.x[:,1]==54100], axis=0), :]
        #     # fkey = self.floatKey2D([37900,54100])
        #     # on_edge = UniformNodeGridFixedSizeMeshModel.point_domain_edge_map.get(fkey, True)            
        #     for k in range(plot_points.shape[0]):
        #         try:
        #             colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, [plot_points[k,:]])
        #         except RuntimeError as e:
        #             print("colliding cells failed at k: ", e, k, plot_points[k-2:k])
        #             breakpoint()
        #     breakpoint()
        #     raise
        
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
                print("need to extrapolate cell for point", i, point)
                if (point[2]>200000):
                    try:
                        points_cells.append(cell_candidates.links(i)[0])
                        points_on_proc.append(point)
                    except IndexError:
                        print("IndexError", point, cell_candidates.links(i) )
                        breakpoint()
                        raise
                else:
                    print("PING V", point)
                    if len(cell_candidates.links(i))==0:
                        print("PING V V", point)
                        def boundary(x):
                            return np.full(x.shape[1], True)
                        entities = dolfinx.mesh.locate_entities(self.mesh, 3, boundary )
                        tet = dolfinx.cpp.mesh.entities_to_geometry(self.mesh, 3, entities, False)
                        breakpoint()
                    points_on_proc.append(point)
                    points_cells.append(cell_candidates.links(i)[0])
        #points_on_proc = np.array(points_on_proc, dtype=np.float64)
        #res = self.uh.eval(points_on_proc, points_cells)
        res = np.array(res)
        aa = np.any(np.isnan(res))
        bb = np.any(np.isnan(self.uh.x.array))
        if aa or bb:
            print(aa,bb)
            breakpoint()

        if transpose:
            assert res.flatten().shape[0] == x.shape[1]
        else:
            assert res.flatten().shape[0] == x.shape[0]
        return res.flatten()

    def nodeIsOnDomainEdge(self, node0):
        return any([ e[0]==node0 or e[1]==node0 for e in self.convexHullEdges])

    def pointIsOnDomainEdge(self, pt, node0, node1, weight):
        if (abs(weight)<0.01):
            return self.nodeIsOnDomainEdge(node0)
        if (abs(weight-1.0)<0.01):
            return self.nodeIsOnDomainEdge(node1)
        b0 = [node0, node1] in self.convexHullEdges
        b1 = [node1, node0] in self.convexHullEdges
        if b0 or b1:
            return True
        return False
