# https://resqpy.readthedocs.io/en/latest/tutorial/high_level_objects.html#reading-and-writing-objects

import numpy as np

import resqpy.property as rqp
import resqpy.crs as rqc
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.unstructured as rug
import resqpy.time_series as rts
from warmth.logging import logger
#
# 
#

def read_mesh_resqml(epcfilename, meshTitle = 'tetramesh'):
    """Example code how to read the .epc file written by the write_tetra_grid_with_properties function.
       Extracts arrays of node positions and of tetrahedra indices.
       Extracts arrays of properties (per-cell and per-node)
    """
    model = rq.Model(epcfilename)
    assert model is not None

    #
    # read mesh:  vertex positions and cell/tetrahedra definitions
    #
    tetra_uuid = model.uuid(obj_type = 'UnstructuredGridRepresentation', title = meshTitle)
    assert tetra_uuid is not None
    tetra = rug.TetraGrid(model, uuid = tetra_uuid)
    assert tetra is not None
    print(f"Mesh {tetra.title}: {tetra.node_count} nodes, {tetra.cell_count} cells, {tetra.cell_shape} cell shape")
    assert tetra.cell_shape == 'tetrahedral'
    
    cells = np.array( [ tetra.distinct_node_indices_for_cell(i) for i in range(tetra.cell_count) ]  ) # cell indices are read using this function(?)
    
    tetra.check_tetra()

    #
    # read properties
    #

    temp_uuid = model.uuid(title = 'Temperature')
    assert temp_uuid is not None
    temp_prop = rqp.Property(model, uuid = temp_uuid)
    assert temp_prop.uom() == 'degC'
    assert temp_prop.indexable_element() == 'nodes'   # properties are defined either on nodes or on cells

    layerID_uuid = model.uuid(title = 'LayerID')
    assert layerID_uuid is not None
    layerID_prop = rqp.Property(model, uuid = layerID_uuid)
    # assert layerID_prop.uom() == 'Euc'
    assert layerID_prop.is_continuous() == False
    assert layerID_prop.indexable_element() == 'cells'
    # print( layerID_prop.array_ref().shape, layerID_prop.array_ref()[0:10] )  # .array_ref() exposes the values as numpy array
 
    titles=['Temperature', 'Age', 'LayerID', 'Porosity_initial', 'Porosity_decay', 'Density_solid', 'thermal_conductivity', 'Radiogenic_heat_production']
    for title in titles:
        prop_uuid = model.uuid(title = title)
        prop = rqp.Property(model, uuid = prop_uuid)
        print(f"Property {title}: defined on {prop.indexable_element()}, unit {prop.uom()}, first values: {prop.array_ref()[0:10]}")
    
def write_tetra_grid_with_properties(filename, nodes, cells, modelTitle = "tetramesh",
    Temp_per_vertex=None, age_per_vertex=None, poro0_per_cell=None, decay_per_cell=None, density_per_cell=None,
    cond_per_cell=None, rhp_per_cell=None, lid_per_cell=None ):
    """Writes the given tetrahedral mesh, defined by arrays of nodes and cell indices, into a RESQML .epc file
       Given SubsHeat properties are optionally written.
       NOTE: writing properties that are defines per-node (have 'nodes' as indexable element) requires a patched version of resqpy!
    """
    node_count = len(nodes)
    faces_per_cell = []
    nodes_per_face = []
    faces_dict = {}
    faces_repeat = np.zeros(node_count*100, dtype = bool)
    cell_face_is_right_handed = np.zeros(len(cells)*4, dtype = bool)
    for it,tet in enumerate(cells):
        midp = ( nodes[tet[0],:] + nodes[tet[1],:] + nodes[tet[2],:] + nodes[tet[3],:] ) * 0.25
        for ir,tri in enumerate([[0,1,2],[0,1,3],[1,2,3],[2,0,3]]):
            face0 = [tet[x] for x in tri ]
            assert -1 not in face0

            #
            # the point order in the tetrahedra may not be consistent
            #   best to test every face individually
            #
            e0 = nodes[face0[1],:] - nodes[face0[0],:]
            e1 = nodes[face0[2],:] - nodes[face0[0],:]
            normal = np.cross(e0,e1)
            midp_face = ( nodes[face0[0],:] + nodes[face0[1],:] + nodes[face0[2],:]) / 3.0
            sign = np.dot( midp-midp_face, normal )
            face_handedness = (sign>0)

            fkey0 = ( x for x in sorted(face0) )
            #
            # keep track of which faces are encountered once vs. more than once
            # faces that are encountered the second time will need to use the reverse handedness
            #
            face_is_repeated = False
            if (fkey0 not in faces_dict):
                faces_dict[fkey0] = len(nodes_per_face)
                nodes_per_face.extend(face0)
                cell_face_is_right_handed[it*4+ir] = face_handedness
            else:
                face_is_repeated = True
                cell_face_is_right_handed[it*4+ir] = not face_handedness
            fidx0 = faces_dict.get(fkey0)            
            faces_per_cell.append(fidx0/3)
            faces_repeat[int(fidx0/3)] = face_is_repeated
    
    set_cell_count = int(len(faces_per_cell)/4)
    face_count = int(len(nodes_per_face)/3)

    # cell_face_is_right_handed = np.zeros(face_count, dtype = bool)
    # cell_face_is_right_handed[faces_repeat[0:face_count]] = True

    model = rq.new_model(filename)
    crs = rqc.Crs(model)
    crs.create_xml()

    # create an empty TetraGrid
    tetra = rug.TetraGrid(model, title = modelTitle)
    assert tetra.cell_shape == 'tetrahedral'

    # hand craft all attribute data
    tetra.crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')
    assert tetra.crs_uuid is not None
    assert bu.matching_uuids(tetra.crs_uuid, crs.uuid)
    tetra.set_cell_count(set_cell_count)
    # faces
    tetra.face_count = face_count
    tetra.faces_per_cell_cl = np.arange(4, 4 * set_cell_count + 1, 4, dtype = int)
    tetra.faces_per_cell = np.array(faces_per_cell)

    # nodes
    tetra.node_count = node_count
    tetra.nodes_per_face_cl = np.arange(3, 3 * face_count + 1, 3, dtype = int)
    tetra.nodes_per_face = np.array(nodes_per_face)

    # face handedness
    tetra.cell_face_is_right_handed = cell_face_is_right_handed  # False for all faces for external cells (1 to 4)

    # points
    tetra.points_cached = nodes

    # basic validity check
    tetra.check_tetra()

    # write arrays, create xml and store model
    tetra.write_hdf5()
    tetra.create_xml()
    
    # https://github.com/bp/resqpy/blob/master/tests/unit_tests/property/test_property.py
    #
    # 'Temperature'  'Age'    'LayerID' 'Porosity_initial'  'Porosity_decay' 'Density_solid'  'thermal_conductivity''Radiogenic_heat_production'

    if Temp_per_vertex is not None:
        _ = rqp.Property.from_array(model,
                                    Temp_per_vertex,
                                    source_info = 'SubsHeat',
                                    keyword = 'Temperature',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'thermodynamic temperature',
                                    indexable_element = 'nodes',
                                    uom = 'degC')

    if age_per_vertex is not None:
        _ = rqp.Property.from_array(model,
                                    age_per_vertex,
                                    source_info = 'SubsHeat',
                                    keyword = 'Age',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'geological age',
                                    indexable_element = 'nodes',
                                    uom = 'Ma')

    if lid_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    lid_per_cell.astype(np.int32),
                                    source_info = 'SubsHeat',
                                    keyword = 'LayerID',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'layer ID',
                                    indexable_element = 'cells',
                                    uom = 'Euc',
                                    discrete=True)
         
    if poro0_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    poro0_per_cell,
                                    source_info = 'SubsHeat',
                                    keyword = 'Porosity_initial',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'porosity',
                                    indexable_element = 'cells',
                                    uom = 'm3/m3')
    if decay_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    decay_per_cell,
                                    source_info = 'SubsHeat',
                                    keyword = 'Porosity_decay',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'porosity decay',
                                    indexable_element = 'cells',
                                    uom = 'Euc')
    if density_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    density_per_cell,
                                    source_info = 'SubsHeat',
                                    keyword = 'Density_solid',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'density',
                                    indexable_element = 'cells',
                                    uom = 'kg/m3')
    if cond_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    cond_per_cell,
                                    source_info = 'SubsHeat',
                                    keyword = 'thermal_conductivity',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'thermal conductivity',
                                    indexable_element = 'cells',
                                    uom = 'W/(m.deltaK)')
    if rhp_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    rhp_per_cell,
                                    source_info = 'SubsHeat',
                                    keyword = 'Radiogenic_heat_production',
                                    support_uuid = tetra.uuid,
                                    property_kind = 'heat',
                                    indexable_element = 'cells',
                                    uom = 'W/m3')

    model.store_epc()
    # read_mesh_resqml(filename)


def read_mesh_resqml_hexa(epcfilename, meshTitle = 'hexamesh'):
    """Example code how to read the .epc file written by the write_hexa_grid_with_properties function.
       Extracts arrays of node positions and of hexahedra indices.
       Extracts arrays of properties (per-cell and per-node)
    """
    model = rq.Model(epcfilename)
    assert model is not None

    #
    # read mesh:  vertex positions and cell definitions
    #
    hexa_uuid = model.uuid(obj_type = 'UnstructuredGridRepresentation', title = meshTitle)
    assert hexa_uuid is not None
    hexa = rug.HexaGrid(model, uuid = hexa_uuid)
    assert hexa is not None
    print(hexa.title, hexa.node_count, hexa.cell_count, hexa.cell_shape)
    assert hexa.cell_shape == 'hexahedral'
    
    print( hexa.points_ref().shape )   # numpy array of vertex positions
    cells = np.array( [ hexa.distinct_node_indices_for_cell(i) for i in range(hexa.cell_count) ]  ) # cell indices are read using this function(?)
    print( cells.shape )   # numpy array of vertex positions
    
    hexa.check_hexahedral()

    #
    # read properties
    #

    temp_uuid = model.uuids(title = 'Temperature')
    temp_uuid = temp_uuid[0]
    assert temp_uuid is not None
    temp_prop = rqp.Property(model, uuid = temp_uuid)
    assert temp_prop.uom() == 'degC'
    assert temp_prop.indexable_element() == 'nodes'   # properties are defined either on nodes or on cells
    print( temp_prop.array_ref().shape, temp_prop.array_ref()[0:10] )  # .array_ref() exposes the values as numpy array

    layerID_uuid = model.uuid(title = 'LayerID')
    assert layerID_uuid is not None
    layerID_prop = rqp.Property(model, uuid = layerID_uuid)
    # assert layerID_prop.uom() == 'Euc'
    assert layerID_prop.is_continuous() == False
    assert layerID_prop.indexable_element() == 'cells'
    print( layerID_prop.array_ref().shape, layerID_prop.array_ref()[0:10] )  # .array_ref() exposes the values as numpy array
 
    titles=[ 'Age', 'LayerID', 'Porosity_initial', 'Porosity_decay', 'Density_solid', 'thermal_conductivity', 'Radiogenic_heat_production']
    titles_uuid = [model.uuid(title = title) for title in titles]
    titles_uuid.append(temp_uuid)
    for prop_uuid in titles_uuid:
        prop = rqp.Property(model, uuid = prop_uuid)
        print(prop.title, prop.indexable_element(), prop.uom(), prop.array_ref()[0:10] )
    

def write_hexa_grid_with_properties(filename, nodes, cells, modelTitle = "hexamesh",
    Temp_per_vertex=None, age_per_vertex=None, poro0_per_cell=None, decay_per_cell=None, density_per_cell=None,
    cond_per_cell=None, rhp_per_cell=None, lid_per_cell=None ):
    """Writes the given hexahedral mesh, defined by arrays of nodes and cell indices, into a RESQML .epc file
       Given SubsHeat properties are optionally written.
 
       cells is an array of 8-arrays in which the nodes are ordered:     
               7------6
              /      /|
             /      / |
            4------5  |
            |         |
            |  3------2
            | /      /
            |/      /
            0------1

       NOTE: writing properties that are defines per-node (have 'nodes' as indexable element) requires a patched version of resqpy!
    """
    node_count = len(nodes)
    faces_per_cell = []
    nodes_per_face = []
    faces_dict = {}
    faces_repeat = np.zeros(node_count*100, dtype = bool)

    cell_face_is_right_handed = np.zeros( len(cells)*6, dtype = bool)
    for ih,hexa in enumerate(cells):
        faces= [[0,3,2,1], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [4,5,6,7]]
        for iq,quad in enumerate(faces):
            face0 = [hexa[x] for x in quad ]
            assert -1 not in face0
            fkey0 = ( x for x in sorted(face0) )
            #
            # keep track of which faces are encountered once vs. more than once
            # faces that are encountered the second time will need to use the reverse handedness
            #
            face_is_repeated = False
            if (fkey0 not in faces_dict):
                faces_dict[fkey0] = len(nodes_per_face)
                nodes_per_face.extend(face0)
                cell_face_is_right_handed[(ih*6 + iq)] = False
            else:
                face_is_repeated = True
                cell_face_is_right_handed[(ih*6 + iq)] = True
            fidx0 = faces_dict.get(fkey0)            
            faces_per_cell.append(fidx0/4)
            faces_repeat[int(fidx0/4)] = face_is_repeated
    
    set_cell_count = int(len(faces_per_cell)/6)
    face_count = int(len(nodes_per_face)/4)


    model = rq.new_model(filename)
    crs = rqc.Crs(model)
    crs.create_xml()

    # gts = rts.GeologicTimeSeries(model, title="warmth simulation")
    # gts.create_time_index(182)
    # gts.create_xml()

    # create an empty HexaGrid
    hexa = rug.HexaGrid(model, title = modelTitle)
    assert hexa.cell_shape == 'hexahedral'

    # hand craft all attribute data
    hexa.crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')
    assert hexa.crs_uuid is not None
    assert bu.matching_uuids(hexa.crs_uuid, crs.uuid)
    hexa.set_cell_count(set_cell_count)
    # faces
    hexa.face_count = face_count
    hexa.faces_per_cell_cl = np.arange(6, 6 * set_cell_count + 1, 6, dtype = int)
    hexa.faces_per_cell = np.array(faces_per_cell)

    # nodes
    hexa.node_count = node_count
    hexa.nodes_per_face_cl = np.arange(4, 4 * face_count + 1, 4, dtype = int)
    hexa.nodes_per_face = np.array(nodes_per_face)

    # face handedness
    hexa.cell_face_is_right_handed = cell_face_is_right_handed  # False for all faces for external cells

    # points
    hexa.points_cached = nodes

    # basic validity check
    hexa.check_hexahedral()

    # write arrays, create xml and store model
    hexa.write_hdf5()
    hexa.create_xml()

    if Temp_per_vertex is not None:
        _ = rqp.Property.from_array(model,
                                    Temp_per_vertex.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Temperature',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'thermodynamic temperature',
                                    indexable_element = 'nodes',
                                    uom = 'degC')

    if age_per_vertex is not None:
        _ = rqp.Property.from_array(model,
                                    age_per_vertex.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Age',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'geological age',
                                    indexable_element = 'nodes',
                                    uom = 'y')

    if lid_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    lid_per_cell.astype(np.int32),
                                    source_info = 'SubsHeat',
                                    keyword = 'LayerID',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'layer ID',
                                    indexable_element = 'cells',
                                    uom = 'Euc',
                                    discrete=True)
         
    if poro0_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    poro0_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Porosity_initial',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'porosity',
                                    indexable_element = 'cells',
                                    uom = 'm3/m3')
    if decay_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    decay_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Porosity_decay',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'porosity decay',
                                    indexable_element = 'cells',
                                    uom = 'Euc')
    if density_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    density_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Density_solid',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'density',
                                    indexable_element = 'cells',
                                    uom = 'kg/m3')
    if cond_per_cell is not None:

        _ = rqp.Property.from_array(model,
                                    cond_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'thermal_conductivity',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'thermal conductivity',
                                    indexable_element = 'cells',
                                    uom = 'W/(m.deltaK)')
    if rhp_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    rhp_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Radiogenic_heat_production',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'heat',
                                    indexable_element = 'cells',
                                    uom = 'W/m3')

    model.store_epc()


def write_hexa_grid_with_timeseries(filename, nodes_series, cells, modelTitle = "hexamesh",
    Temp_per_vertex_series=None, Ro_per_vertex_series= None, 
    age_per_vertex=None, poro0_per_cell=None, decay_per_cell=None, density_per_cell=None,
    cond_per_cell=None, rhp_per_cell=None, lid_per_cell=None ):
    """Writes the given hexahedral mesh, defined by arrays of nodes and cell indices, into a RESQML .epc file
       Given SubsHeat properties are optionally written.
 
       cells is an array of 8-arrays in which the nodes are ordered:     
               7------6
              /      /|
             /      / |
            4------5  |
            |         |
            |  3------2
            | /      /
            |/      /
            0------1

       NOTE: writing properties that are defines per-node (have 'nodes' as indexable element) requires a patched version of resqpy!
    """
    logger.info("Creating RESQML model")
    nodes_time_0 = nodes_series[-1,:,:] # present-day at last index
    node_count = nodes_time_0.shape[0]
    faces_per_cell = []
    nodes_per_face = []
    faces_dict = {}
    faces_repeat = np.zeros(node_count*100, dtype = bool)

    cell_face_is_right_handed = np.zeros( len(cells)*6, dtype = bool)
    for ih,hexa in enumerate(cells):
        faces= [[0,3,2,1], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [4,5,6,7]]
        for iq,quad in enumerate(faces):
            face0 = [hexa[x] for x in quad ]
            assert -1 not in face0
            fkey0 = ( x for x in sorted(face0) )
            #
            # keep track of which faces are encountered once vs. more than once
            # faces that are encountered the second time will need to use the reverse handedness
            #
            face_is_repeated = False
            if (fkey0 not in faces_dict):
                faces_dict[fkey0] = len(nodes_per_face)
                nodes_per_face.extend(face0)
                cell_face_is_right_handed[(ih*6 + iq)] = False
            else:
                face_is_repeated = True
                cell_face_is_right_handed[(ih*6 + iq)] = True
            fidx0 = faces_dict.get(fkey0)            
            faces_per_cell.append(fidx0/4)
            faces_repeat[int(fidx0/4)] = face_is_repeated
    
    set_cell_count = int(len(faces_per_cell)/6)
    face_count = int(len(nodes_per_face)/4)


    model = rq.new_model(filename)
    crs = rqc.Crs(model)
    crs.create_xml()
    # present-day is set as 1 due resqpy
    million_years_offset = 0 
    times_in_years = [ int(max((t+million_years_offset)*1e6, million_years_offset)) for t in list(range(nodes_series.shape[0]-1,-1,-1))]

    gts = rts.GeologicTimeSeries.from_year_list(model, times_in_years, title="warmth simulation")
    gts.create_xml()
    rts.timeframe_for_time_series_uuid(model, gts.uuid)

    # create an empty HexaGrid
    hexa = rug.HexaGrid(model, title = modelTitle)
    assert hexa.cell_shape == 'hexahedral'

    # hand craft all attribute data
    hexa.crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')
    assert hexa.crs_uuid is not None
    assert bu.matching_uuids(hexa.crs_uuid, crs.uuid)
    hexa.set_cell_count(set_cell_count)
    # faces
    hexa.face_count = face_count
    hexa.faces_per_cell_cl = np.arange(6, 6 * set_cell_count + 1, 6, dtype = int)
    hexa.faces_per_cell = np.array(faces_per_cell)

    # nodes
    hexa.node_count = node_count
    hexa.nodes_per_face_cl = np.arange(4, 4 * face_count + 1, 4, dtype = int)
    hexa.nodes_per_face = np.array(nodes_per_face)

    # face handedness
    hexa.cell_face_is_right_handed = cell_face_is_right_handed  # False for all faces for external cells

    # points
    hexa.points_cached = nodes_time_0

    # basic validity check
    hexa.check_hexahedral()

    hexa.create_xml()
    hexa.write_hdf5()

    if hexa.property_collection is None:
        hexa.property_collection = rqp.PropertyCollection(support = hexa)
    pc = hexa.property_collection

    # nodes0 = nodes.copy()
    for time_index in range(nodes_series.shape[0]):  #oldest first

        nodes2 = nodes_series[time_index,:,:].astype(np.float32)

        pc.add_cached_array_to_imported_list(nodes2,
                                                'dynamic nodes',
                                                "points",
                                                uom = 'm',
                                                property_kind = 'length',
                                                realization = 0,
                                                time_index = time_index,
                                                indexable_element = 'nodes',
                                                points = True)
        # active_array = np.ones([2160], dtype = bool)
        tt = Temp_per_vertex_series[time_index].astype(np.float32)
        pc.add_cached_array_to_imported_list(tt,
                                                'Temperature',
                                                "Temperature",
                                                uom = 'degC',
                                                property_kind = 'thermodynamic temperature',
                                                realization = 0,
                                                time_index = time_index,
                                                indexable_element = 'nodes')
                                                # points = True)
        if (Ro_per_vertex_series is not None):
            ro = Ro_per_vertex_series[time_index,:].astype(np.float32)
            pc.add_cached_array_to_imported_list(ro,
                                                    'Vitrinite reflectance',
                                                    "%Ro",
                                                    uom = 'percent',
                                                    property_kind = 'dimensionless',
                                                    realization = 0,
                                                    time_index = time_index,
                                                    indexable_element = 'nodes')
                                                    # points = True)


    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = gts.uuid)


    if age_per_vertex is not None:
        _ = rqp.Property.from_array(model,
                                    age_per_vertex.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Age',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'geological age',
                                    indexable_element = 'nodes',
                                    uom = 'y')

    if lid_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    lid_per_cell.astype(np.int32),
                                    source_info = 'SubsHeat',
                                    keyword = 'LayerID',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'layer ID',
                                    indexable_element = 'cells',
                                    uom = 'Euc',
                                    discrete=True)
         
    if poro0_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    poro0_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Porosity_initial',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'porosity',
                                    indexable_element = 'cells',
                                    uom = 'm3/m3')
    if decay_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    decay_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Porosity_decay',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'porosity decay',
                                    indexable_element = 'cells',
                                    uom = 'Euc')
    if density_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    density_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Density_solid',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'density',
                                    indexable_element = 'cells',
                                    uom = 'kg/m3')
    if cond_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    cond_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'thermal_conductivity',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'thermal conductivity',
                                    indexable_element = 'cells',
                                    uom = 'W/(m.deltaK)')
    if rhp_per_cell is not None:
        _ = rqp.Property.from_array(model,
                                    rhp_per_cell.astype(np.float32),
                                    source_info = 'SubsHeat',
                                    keyword = 'Radiogenic_heat_production',
                                    support_uuid = hexa.uuid,
                                    property_kind = 'heat',
                                    indexable_element = 'cells',
                                    uom = 'W/m3')
    model.store_epc()

