from warmth.build import single_node, Builder,Results,Grid
from warmth.parameters import Parameters
import numpy as np
import pandas as pd
tidy_sediments = single_node._tidy_sediments

def test_tidy_sediments_onlap():
    d = {"top":[100,200,300,400,500],"topage":[0,10,20,30,40],"k_cond":np.full(5,2),"rhp":np.full(5,1e-7),
    "phi":np.full(5,0.55),"decay":np.full(5,0.49),"solidus":np.full(5,2700),"liquidus":np.full(5,2400),
    "strat":["Onlap","Onlap","Onlap","Onlap","Erosion"],"horizonIndex":np.arange(5)}
    test_df = pd.DataFrame.from_dict(d)
    test_df.at[2, 'top'] =2600
    df = tidy_sediments(test_df)
    assert np.allclose(df["top"].values, np.array([100,200,400,400]))
    assert np.allclose(df["base"].values, np.array([200,400,400,500]))

def test_tidy_sediments_erosion():
    d = {"top":[100,200,300,400,500],"topage":[0,10,20,30,40],"k_cond":np.full(5,2),"rhp":np.full(5,1e-7),
    "phi":np.full(5,0.55),"decay":np.full(5,0.49),"solidus":np.full(5,2700),"liquidus":np.full(5,2400),
    "strat":["Onlap","Onlap","Erosion","Onlap","Erosion"],"horizonIndex":np.arange(5)}
    test_df = pd.DataFrame.from_dict(d)
    test_df.at[2, 'top'] =2600
    df = tidy_sediments(test_df)
    assert np.allclose(df["top"].values, np.array([100,200,500,500]))
    assert np.allclose(df["base"].values, np.array([200,500,500,500]))
    return
def test_check_nan_in_sediments():
    b = Builder(Parameters())
    d = {"top":[100,200,300,400,500],"topage":[0,10,20,30,40],"k_cond":np.full(5,2),"rhp":np.full(5,1e-7),
    "phi":np.full(5,0.55),"decay":np.full(5,0.49),"solidus":np.full(5,2700),"liquidus":np.full(5,2400),
    "strat":["Onlap","Onlap","Onlap","Onlap","Erosion"],"horizonIndex":np.arange(5)}
    test_df = pd.DataFrame.from_dict(d)
    r = b._check_nan_sed(test_df)
    assert r == True

def test_consecutive_nan_in_sediments():
    b = Builder(Parameters())
    d = {"top":[100,np.nan,np.nan,np.nan,500],"topage":[0,10,20,30,40],"k_cond":np.full(5,2),"rhp":np.full(5,1e-7),
        "phi":np.full(5,0.55),"decay":np.full(5,0.49),"solidus":np.full(5,2700),"liquidus":np.full(5,2400),
        "strat":["Onlap","Onlap","Onlap","Onlap","Erosion"],"horizonIndex":np.arange(5)}
    test_df = pd.DataFrame.from_dict(d)
    r = b._check_nan_sed(test_df)
    assert r == False
def test_max_nan_in_sediments():
    b = Builder(Parameters())
    d = {"top":[100,np.nan,np.nan,np.nan,np.nan,500],"topage":[0,10,20,30,35,40],"k_cond":np.full(6,2),"rhp":np.full(6,1e-7),
        "phi":np.full(6,0.55),"decay":np.full(6,0.49),"solidus":np.full(6,2700),"liquidus":np.full(6,2400),
        "strat":["Onlap","Onlap","Onlap","Onlap","Onlap","Erosion"],"horizonIndex":np.arange(6)}
    test_df = pd.DataFrame.from_dict(d)
    r = b._check_nan_sed(test_df)
    assert r == False
def test_nan_in_seabed_basement():
    b = Builder(Parameters())
    d = {"top":[np.nan,200,300,400,500],"topage":[0,10,20,30,40],"k_cond":np.full(5,2),"rhp":np.full(5,1e-7),
    "phi":np.full(5,0.55),"decay":np.full(5,0.49),"solidus":np.full(5,2700),"liquidus":np.full(5,2400),
    "strat":["Onlap","Onlap","Onlap","Onlap","Erosion"],"horizonIndex":np.arange(5)}
    test_df = pd.DataFrame.from_dict(d)
    r = b._check_nan_sed(test_df)
    assert r == False
    d = {"top":[100,200,300,400,np.nan],"topage":[0,10,20,30,40],"k_cond":np.full(5,2),"rhp":np.full(5,1e-7),
    "phi":np.full(5,0.55),"decay":np.full(5,0.49),"solidus":np.full(5,2700),"liquidus":np.full(5,2400),
    "strat":["Onlap","Onlap","Onlap","Onlap","Erosion"],"horizonIndex":np.arange(5)}
    test_df = pd.DataFrame.from_dict(d)
    r = b._check_nan_sed(test_df)
    assert r == False


def test_results():
    sed_data = {"top":[0,100,200],"topage":[0,10,20],"k_cond":np.full(3,1),"rhp":np.full(3,1e-7),
        "phi":np.full(3,0.55),"decay":np.full(3,0.49),"solidus":np.full(3,2700),"liquidus":np.full(3,2400),
        "strat":["Onlap","Onlap","Erosion"],"horizonIndex":np.arange(3)}
    sed = pd.DataFrame.from_dict(sed_data)
    d = np.array([[0,100,200,300,400,500,600],[np.nan,0,100,200,400,500,600]])
    d=d.T
    t = np.array([[5,20,30,40,50,60,70],[np.nan,5,20,30,50,60,70]])
    t=t.T
    idsed = np.array([[0,1,-1,-2,-3,-3],[np.nan,1,-1,-2,-3,-3]])
    idsed =idsed.T
    k_crust = 2
    k_lith = 3
    k_asth=5
    r = Results(d,t,idsed,sed,k_crust,k_lith,k_asth)
    assert np.allclose(r.ages,np.arange(2))
    np.testing.assert_almost_equal(r.top_crust(1),100,decimal=2)
    np.testing.assert_almost_equal(r.crust_thickness(1),100,decimal=2)
    np.testing.assert_almost_equal(r.top_lithosphere(1),200,decimal=2)
    np.testing.assert_almost_equal(r.lithosphere_thickness(1),200,decimal=2)
    np.testing.assert_almost_equal(r.top_asthenosphere(1),400,decimal=2)
    np.testing.assert_array_almost_equal(r.depth(1),d[:,1])
    np.testing.assert_array_almost_equal(r.sediment_porosity(1)["values"],np.array([0, 0.53674242, 0, 0,0,  0]),decimal=1,verbose=True)
    np.testing.assert_array_almost_equal(r._reference_conductivity(1),np.array([np.nan,1,2,3,5,5]),decimal=1,verbose=True)
    np.testing.assert_array_almost_equal(r.effective_conductivity(1)["values"],np.array([np.nan,0.45, 1.99, 2.90, 4.57,4.47]),decimal=1,verbose=True)
    np.testing.assert_array_almost_equal(r.heatflow(1)["values"],np.array([np.nan,0.0675, 0.1996, 0.2907, 0.4578,0.4474]),decimal=1,verbose=True)
    np.testing.assert_almost_equal(r.basement_heatflow(1),0.1996,decimal=1)
    assert r.heatflow(1)["layerId"].size == r.sediment_ids(1).size
    assert r.heatflow(1)["depth"].size == r.sediment_ids(1).size
    assert r.heatflow(1)["values"].size == r.sediment_ids(1).size
    assert r.temperature(1)["layerId"].size == r.sediment_ids(1).size
    assert r.temperature(1)["depth"].size == r.depth(1).size
    assert r.temperature(1)["values"].size == r.depth(1).size
    assert np.allclose(np.unique(r.heatflow(1,1)["layerId"]) , np.array([1]))
    assert r.heatflow(1,1)["depth"].size == 1
    assert r.heatflow(1,1)["layerId"].size == 1
    assert r.heatflow(1,1)["values"].size == 1
    assert np.allclose(np.unique(r.temperature(1,1)["layerId"]) , np.array([1]))
    assert r.temperature(1,1)["depth"].size == 2
    assert r.temperature(1,1)["layerId"].size == 1
    assert r.temperature(1,1)["values"].size == 2
    assert r.seabed(0) == 0
    assert r.seabed(1) == 0

def test_grid():
    g = Grid(10.2, 100.3, 5, 6, 105.5, 102.5)
    xtgeo_check = np.array([[10.2, 100.3,  0.], [115.7, 100.3,   0.], [221.2, 100.3,  0.], [326.7, 100.3,  0.],
                            [432.2, 100.3, 0.],
                            [10.2, 202.8, 0.],
                            [115.7, 202.8, 0.],
                            [221.2, 202.8, 0.],
                            [326.7, 202.8, 0.],
                            [432.2, 202.8, 0.],
                            [10.2, 305.3, 0.],
                            [115.7, 305.3, 0.],
                            [221.2, 305.3, 0.],
                            [326.7, 305.3, 0.],
                            [432.2, 305.3, 0.],
                            [10.2, 407.8, 0.],
                            [115.7, 407.8, 0.],
                            [221.2, 407.8, 0.],
                            [326.7, 407.8, 0.],
                            [432.2, 407.8, 0.],
                            [10.2, 510.3, 0.],
                            [115.7, 510.3, 0.],
                            [221.2, 510.3, 0.],
                            [326.7, 510.3, 0.],
                            [432.2, 510.3, 0.],
                            [10.2, 612.8,  0.],
                            [115.7, 612.8,  0.],
                            [221.2, 612.8, 0.],
                            [326.7, 612.8,  0.],
                            [432.2, 612.8,  0.]])

    assert np.allclose(g._location_xtgeo_z, xtgeo_check) == True
    grid = g.make_grid_arr()
    assert len(grid) == 6
    for row in grid:
        assert len(row) == 5
    indexer_check = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1],
                              [2, 2],
                              [2, 3],
                              [2, 4],
                              [3, 0],
                              [3, 1],
                              [3, 2],
                              [3, 3],
                              [3, 4],
                              [4, 0],
                              [4, 1],
                              [4, 2],
                              [4, 3],
                              [4, 4],
                              [5, 0],
                              [5, 1],
                              [5, 2],
                              [5, 3],
                              [5, 4]])
    assert np.allclose(g.indexing_arr,indexer_check) == True
    assert g.indexing_arr.dtype == 'int64'
