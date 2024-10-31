import pandas as pd
import warmth
from warmth.data import haq87
import numpy as np
import pickle
from warmth.forward_modelling import Forward_model

def sediments(template:pd.DataFrame)->pd.DataFrame:
    h1 = [152.0,0.0,1.500000,2.301755e-09,0.620000,0.500,2720.0,2448.0]
    h2=[810.0,20.0,1.538462,2.079433e-09,0.599730,0.490,2708.0,2437.2]
    h3=[1608.0,66,1.500000,2.301755e-09,0.2,0.500,2720.0,2448.0]
    h4=[1973.0,100,1.500000,2.301755e-09,0.620000,0.500,2720.0,2448.0]
    h5=[2262.0,145,1.500000,2.301755e-09,0.620000,0.500,2720.0,2448.0]
    h6=[2362.0,152,1.904762,4.719506e-10,0.447705,0.415,2618.0,2356.2]
    h7=[2427.0,160,1.500000,2.301755e-09,0.620000,0.500,2720.0,2448.0]
    horizons=[h1,h2,h3,h4,h5,h6,h7]
    for i in horizons:
        new =pd.DataFrame.from_dict({'top': [i[0]], 'topage': [int(i[1])], 'k_cond': [i[2]], 'rhp':[i[3]], 'phi':[i[4]], 'decay':[i[5]], 'solidus':[i[6]],'liquidus':[i[7]]})
        template = pd.concat([template, new], ignore_index=True)
    return template

def load_reference_data_single_rift():
    with open("./tests/data/integration_single_temperature.npy","rb") as f:
        t = np.load(f)
    with open("./tests/data/integration_single_depth.npy","rb") as f:
        d = np.load(f)
    with open("./tests/data/integration_single_sedID.npy","rb") as f:
        s = np.load(f)
    return t,d,s
def load_reference_data_multi_rift():
    with open("./tests/data/integration_multi_temperature.npy","rb") as f:
        t = np.load(f)
    with open("./tests/data/integration_multi_depth.npy","rb") as f:
        d = np.load(f)
    with open("./tests/data/integration_multi_sedID.npy","rb") as f:
        s = np.load(f)
    return t,d,s
def test_integration_single_rift():
    model = warmth.Model()
    sed = sediments(model.builder.single_node_sediments_inputs_template)
    node = warmth.single_node()
    node.sediments_inputs = sed
    node.qbase = 30e-3
    node.crustRHP = (60e-3-node.qbase)/node.hc/0.5
    node.rift = np.array([[160,145]])
    model.parameters.time_start = 160
    model.parameters.time_end = 0
    model.builder.nodes = [[node]]
    model.builder.set_eustatic_sea_level(haq87)
    model.parameters.experimental = True
    fw = Forward_model(model.parameters, node)
    fw.simulate_single_node()
    node = fw.current_node
    ref_t, ref_d, red_s = load_reference_data_single_rift()



def test_integration_multi_rift():
    model = warmth.Model()
    sed = sediments(model.builder.single_node_sediments_inputs_template)
    node = warmth.single_node()
    node.sediments_inputs = sed
    node.qbase = 30e-3
    node.crustRHP = (60e-3-node.qbase)/node.hc/0.5
    node.rift = np.array([[160,145],[100,80]])
    node.paleoWD=np.array([200])
    model.parameters.time_start = 160
    model.parameters.time_end = 0
    model.builder.nodes = [[node]]
    model.builder.set_eustatic_sea_level(haq87)
    model.parameters.experimental = True
    fw = Forward_model(model.parameters, node)
    fw.simulate_single_node()
    node = fw.current_node
    ref_t, ref_d, red_s = load_reference_data_multi_rift()

