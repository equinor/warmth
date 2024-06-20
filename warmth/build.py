from pathlib import Path
import pickle
from typing import Iterator, List, Literal
import xtgeo
import numpy as np

import concurrent.futures
# import geopandas as gpd
import pandas as pd
#from scipy.spatial import ConvexHull
# from shapely.geometry import Point, Polygon
import math
import copy
from dataclasses import dataclass
from warmth.utils import compressed_pickle_open, compressed_pickle_save
from .logging import logger
from .parameters import Parameters
from .postprocessing import Results


class single_node:
    """Properties of 1D location for forward model

    Attributes
    ----------
    hc : float
        Initial crustal thickness (m)
    hLith : float
        Initial depth of base lithosphere (Thermal Lithosphere-Asthenosphere boundary) (m)
    kCrust : float
        Reference conductivity of crust at 20C (W/K.m^2)
    kLith : float
        Reference conductivity of lithospheric mantle at 20C (W/K.m^2)
    kAsth : float
        Reference conductivity of asthenosphere at 20C (W/K.m^2)
    rhp : float
        Radiogenic heat production of the crust (W/m^3)
    crustsolid : float
        Density of the crust
    lithsolid : float
        Density of the lithospheric mantle
    asthsolid : float
        Density of th asthenosphere
    T0 : float
        Seabed temperature (C)
    Tm : float
        Temperature at the Lithosphere-Asthenosphere boundary (LAB) (m)
    qbase : float
        Heat flow at the base of the crust (Moho) (W/m^2)
    sediments_inputs : pd.DataFrame
        Present-day sediments. See Builder.single_node_sediments_inputs_template
    X : float
        X location of the node
    Y : float
        Y location of the node
    paleoWD : np.ndarray[np.float64]
        Paleo-water depth for multi-rift
    rift : List[List[int]]
        Rifting episodes
    water_depth_difference : float
        Difference between forward model and observed present-day water depth
    sediment_fill_margin : int
        Maximum difference between modelled and observed present-day water depth when a fit is considered achieved
    total_beta_tested : int
        Total number of beta factors tested in forward model
    error : str | None
        Error from forward model
    simulated_at : str | None
        Timestamp when forward model is finished
    
    """
    def __init__(self):
        self.hc: float = 30e3
        self.hLith: float = 130e3
        self.kLith: float = 2
        self.kCrust: float = 2.5
        self.kAsth:float = 100
        self.crustRHP: float = 2e-6  #microW
        self._upperCrust_ratio =0.5
        self.crustliquid: float = 2500.0
        self.crustsolid: float = 2800.0
        self.lithliquid: float = 2700.0
        self.lithsolid: float = 3300.0
        self.asthliquid: float = 2700.0
        self.asthsolid: float = 3200.0
        self.T0: float = 5
        self.Tm: float = 1330.0
        self.qbase: float = 30e-3
        self.bflux: bool = True
        self.sediments_inputs : pd.DataFrame | None= None
        self.X:float = 0.0
        self.Y:float = 0.0
        self.indexer = [0, 0]
        self.paleoWD = np.empty(0, dtype=float)
        self.hc_calibration: str = ""
        self.rift = [[]]
        self.water_depth_difference: float = 0
        self.sediment_fill_margin: int = 100
        self.total_beta_tested: int = 0
        self._sediments = None
        self._full_simulation: bool = True
        self.error: str | None = None
        self.simulated_at: float | None = None
        self._depth_out:np.ndarray[np.float64]|None=None
        self.temperature_out:np.ndarray[np.float64]|None=None
        self._idsed:np.ndarray[np.int32]|None=None
        self._ht:float = self.hLith+self.hc+150e3
        self._crust_ls:np.ndarray[np.float64]|None=None
        self._lith_ls:np.ndarray[np.float64]|None=None
        self._subsidence:np.ndarray[np.float64]|None=None
        self.seabed_arr:np.ndarray[np.float64]|None=None
        self.top_crust_arr:np.ndarray[np.float64]|None=None
        self.top_lith_arr:np.ndarray[np.float64]|None=None
        self.top_aest_arr:np.ndarray[np.float64]|None=None

    @property
    def shf(self)->float:
        return ((self.crustRHP*self._upperCrust_ratio)*self.hc) + self.qbase

    @property
    def result(self)-> Results|None:
        """Results of 1D simulation

        Returns
        -------
        Results|None
            None if not simulated
        """
        return Results(self._depth_out,self.temperature_out,self._idsed,self.sediments,self.kCrust,self.kLith,self.kAsth)

    def clear_unused_data(self):
        """Removes most arrays of detailed input and output that are not needed by warmth3D, in order to save memory. 
        """
        # self.max_time = self._depth_out.shape[1]
        self._depth_out = None
        self.temperature_out =None
        self._idsed = None
        self.coord_initial = None
        self._crust_ls = None
        self._lith_ls = None
        self._subsidence =None

    def compute_derived_arrays(self):
        """Computes depths of seabed, top crust, top lithosphere and top aestenosphere, and stores them with the node.
           This allows the depth and temperature arrays to be discarded to save memory.
        """
        self.top_crust_arr = [ self._depth_out[ np.where(self._idsed[:,age] == -1)[0][0], age] for age in range(self.max_time)]
        self.top_lith_arr = [ self._depth_out[ np.where(self._idsed[:,age] == -2)[0][0], age] for age in range(self.max_time)]
        self.top_aest_arr = [ self._depth_out[ np.where(self._idsed[:,age] == -3)[0][0], age] for age in range(self.max_time)]
        self.seabed_arr = np.array( [ self._depth_out[np.where(~np.isnan(self.temperature_out[:,age]))[0][0],age] for age in range(self.max_time)])


    @property
    def crust_ls(self)->np.ndarray[np.float64]:
        if isinstance(self.result,Results):
            all_age = np.arange(len(self.top_lith_arr),dtype=np.int32)
            val = np.zeros(all_age.size)
            for age in all_age:
                val[age] = self.top_lith_arr[age] - self.top_crust_arr[age]
    
            return val
        else:
            return self._crust_ls
    @property
    def lith_ls(self)->np.ndarray[np.float64]:
        if isinstance(self.result,Results):
            all_age = np.arange(len(self.top_lith_arr),dtype=np.int32)
            val = np.zeros(all_age.size)
            for age in all_age:
                val[age] = self.top_aest_arr[age] - self.top_lith_arr[age]
            return val
        else:
            return self._lith_ls          
    @property
    def subsidence(self)->np.ndarray[np.float64]:
        if isinstance(self.result,Results):
            all_age = np.arange(len(self.top_lith_arr),dtype=np.int32)
            val = np.zeros(all_age.size)
            for age in all_age:
                val[age] = self.seabed_arr[age]
                # val[age] = self.result.seabed(age)
            return val
        else:
            return self._subsidence 
    @property
    def sed_thickness_ls(self)->float:
        if isinstance(self.result,Results):
            all_age = np.arange(len(self.top_lith_arr),dtype=np.int32)
            val = np.zeros(all_age.size)
            for age in all_age:
                # seabed = self.result.seabed(age)
                # top_crust = self.result.top_crust(age)
                seabed = self.seabed_arr[age]
                top_crust = self.top_crust_arr[age]
                val[age] = top_crust - seabed
            return val
        else:
            return self.sed[-1,1,:] - self.sed[0,0,:]
        
    
    @property
    def _name(self) -> str:
        return str(self.X).replace(".", "_")+"__"+str(self.Y).replace(".", "_")

    @property
    def fitting(self) -> bool:
        """Whether a beta factor is found

        Returns
        -------
        bool
            True if the modelled water depth difference is smaller than the acceptable difference
        """
        fitting = False
        if self.water_depth_difference*-1 <= self.sediment_fill_margin:
            fitting = True
        return fitting

    # @property
    # def ht(self) -> int:
    #     return self.hLith+self.hc+150e3
    @staticmethod
    def _tidy_sediments(df:pd.DataFrame)->pd.DataFrame:
        check_ascending = df.apply(lambda x: x.is_monotonic_increasing)
        if check_ascending["top"] == False and check_ascending["topage"] == False:
            raise ValueError(
                "topage and top have to be in ascending order")
        # TODO trucation
        #df.drop_duplicates(subset=["top"], keep="last", inplace=True)

        base = df["top"].values[1:]
        top = df["top"].values[:-1]

        basement = np.where(top>base[-1])[0]
        top[basement] = base[-1]

        thickness = base - top
        #check for crossing
        idx = np.where(thickness <0)[0]
        while True:
            for i in reversed(idx):
                if df.iloc[i]["strat"] == 'Onlap':
                    top[i] = df.iloc[i+1]["top"]
                else: # erod everything below
                    idx_top = np.where(top<top[i])[0]
                    idx_top = idx_top[idx_top>i]
                    top[idx_top]=top[i]
            base = np.append(top[1:],base[-1])
            thickness = base - top
            idx = np.where(thickness <0)[0]
            if idx.size==0:
                break

        baseage=df["topage"].values[1:]
        df = df[:-1]
        WD = top[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            # Return 0 if there is crossing/overlapping horizon. i.e. top > base
            PhiMean = (
                df["phi"]
                / df["decay"]
                / (thickness / 1e3)
                * (
                    np.exp((-1 * df["decay"] * ((top - WD) / 1e3)))
                    - np.exp((-1 * df["decay"] *
                                ((base - WD) / 1e3)))
                )
            )
            PhiMean[PhiMean == np.inf] = 0.0
            PhiMean = np.nan_to_num(PhiMean)
            PhiMean[PhiMean < 0] = 0
            grain_thickness = (thickness / 1e3) * (1 - PhiMean)
            grain_thickness[grain_thickness == np.inf] = 0.0
            grain_thickness = np.nan_to_num(grain_thickness)

        df_out = df.assign(top=top,base=base,baseage = baseage,thickness=thickness,grain_thickness=grain_thickness,phi_mean=PhiMean)
        return df_out

    @property
    def sediments(self) -> pd.DataFrame:
        """Cleaned-up sediments for the 1D location

        Returns
        -------
        pd.DataFrame
            Sediment input
        """
        if self._sediments is None:
            #self._tidy_sediments()
            self._sediments = self._tidy_sediments(self.sediments_inputs)
        return self._sediments

    def _dump(self, filepath: Path):
        compressed_pickle_save(self, filepath)
        return


def load_node(filepath: Path) -> single_node:
    logger.info(f"Loading node from {filepath}")
    data = compressed_pickle_open(filepath)
    return data

@dataclass
class _sediment_layer_:
    """Properties of a single sedimentary layer. Only used during model building. Sediment data are stored in class single_node after model building
    """
    X: float = 0
    Y: float = 0
    top: float = 0
    topage: int = 0
    thermoconductivity: float = 2
    rhp: float = 0.1e-6
    phi: float = 0.55
    decay: float = 0.49
    solidus: float = 2700
    liquidus: float = 2400
    strat: Literal["Onlap"] |Literal["Erosive"]="Erosive"
    horizon_index: int|None = None


class Grid:
    """Defines geometry of a 3D model
    """
    def __init__(self, origin_x: float, origin_y: float, num_nodes_x: int, num_nodes_y: int, step_x: float, step_y: float):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.num_nodes_x = num_nodes_x
        self.num_nodes_y = num_nodes_y
        self.step_x = step_x   # node separation in x
        self.step_y = step_y  # node separation in y
        self._location_grid = None
        self.__location_xtgeo = None
        self.__location_xtgeo_z = None
        self._indexing_arr = None
    @property
    def xmax(self)->float:
        return self.origin_x+ (self.num_nodes_x*self.step_x)    
    @property
    def ymax(self)->float:
        return self.origin_y+ (self.num_nodes_y*self.step_y)
    @property
    def location_grid(self)->np.ndarray:
        """Locations of all 1D nodes

        Returns
        -------
        np.ndarray
            A 2D array of locations of all nodes
        """
        if isinstance(self._location_grid, type(None)):
            x = np.arange(self.origin_x, self.origin_x+(self.num_nodes_x *
                          self.step_x), self.step_x, dtype=np.float64)
            y = np.arange(self.origin_y, self.origin_y+(self.num_nodes_y *
                          self.step_y), self.step_y, dtype=np.float64)
            X, Y = np.meshgrid(x, y)
            self._location_grid = np.dstack([X, Y])
        return self._location_grid

    @property
    def _location_xtgeo(self)->np.ndarray:
        """X, Y location to extract using xtgeo

        Returns
        -------
        np.ndarray
            2D X, Y location
        """
        if isinstance(self.__location_xtgeo, type(None)):
            loc_grid = self.location_grid
            self.__location_xtgeo = loc_grid.reshape(
                (loc_grid.shape[0]*loc_grid.shape[1], loc_grid.shape[2]))
        return self.__location_xtgeo

    @property
    def _location_xtgeo_z(self)->np.ndarray:
        """X, Y, Z location to extract using xtgeo

        Returns
        -------
        np.ndarray
            2D X, Y, Z location
        """
        if isinstance(self.__location_xtgeo_z, type(None)):
            arr = self._location_xtgeo
            self.__location_xtgeo_z = np.hstack(
                (arr, np.full((arr.shape[0], 1), 0)))
        return self.__location_xtgeo_z

    def make_grid_arr(self)->List[List]:
        """list of list defining model geometry

        Returns
        -------
        List[List]
            Template geometry to store 1D node object
        """
        return [[False for _ in range(self.num_nodes_x)] for _ in range(self.num_nodes_y)]

    @property
    def indexing_arr(self) -> np.ndarray:
        """Array of indices of all 1D node object

        Returns
        -------
        np.ndarray
            Arry of indices
        """
        if isinstance(self._indexing_arr, type(None)):
            loc_grid = self.location_grid
            ind = np.indices((loc_grid.shape[0], loc_grid.shape[1]))
            self._indexing_arr = np.dstack(
                (ind[0, :].ravel(), ind[1, :].ravel()))[0]
        return self._indexing_arr

    def dump(self, filepath: Path):
        """Save the object

        Parameters
        ----------
        filepath : Path
            File path to save
        """
        self._location_grid = None
        self.__location_xtgeo = None
        self.__location_xtgeo_z = None
        self._indexing_arr = None
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

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
    node.subsidence = np.sum( np.array( [ [node.result.seabed(t) for t in times] * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.crust_ls = np.sum( np.array( [ [node.result.crust_thickness(t) for t in times] * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.lith_ls = np.sum( np.array( [ [node.result.lithosphere_thickness(t) for t in times] * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 

    node.beta = np.sum( np.array( [node.beta * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.kAsth = np.sum( np.array( [node.kAsth * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.kLith = np.sum( np.array( [node.kLith * w for node,w in zip(interpolationNodes,iWeightNorm)] ) , axis = 0) 
    node.depth_out = np.sum([node.result._depth*w for n,w in zip(interpolationNodes[0:1], [1] )], axis=0)
    node.temperature_out = np.sum([n.result._temperature*w for n,w in zip(interpolationNodes[0:1], [1] )], axis=0)

    node.sed = np.sum([n.sed*w for n,w in zip(interpolationNodes,iWeightNorm)], axis=0)
    node.sed_thickness_ls =  node.sed[-1,1,:] - node.sed[0,0,:]    
    return node


class Builder:
    def __init__(self, parameters: Parameters):
        """Utilities to build a model

        Parameters
        ----------
        parameters : Parameters
            Model parameters
        """
        self.parameters = parameters
        self._xmin = 0
        self._xmax = 0
        self._ymin = 0
        self._ymax = 0
        self.boundary = None
        self.grid: Grid | None = None
        self.nodes: list[single_node] = []

    @property
    def single_node_sediments_inputs_template(self):
        """Template for creating sediment for single node

        Returns
        -------
        pd.Dataframe
            Single node sediment template
        """
        return pd.DataFrame(
            columns=[
                "top",
                "topage",
                "k_cond",
                "rhp",
                "phi",
                "decay",
                "solidus",
                "liquidus",
            ]
        )

    @property
    def input_horizons_template(self)->pd.DataFrame:
        """Template dataframe for model input using maps

        Returns
        -------
        pd.DataFrame
            Emtpy dataframe for appending input data
        """
        return pd.DataFrame({'Age': pd.Series(dtype='int'),
                             'File_name': pd.Series(dtype='str'),
                             'Facies_maps': pd.Series(dtype='str'),
                             'Stratigraphy': pd.Series(dtype='str')})


    def _extract_single_horizon(self,
                               path:Path, input_data_row:pd.Series,  row_index: int, formatfile:str="irap_binary", facies_dict:dict|None=None
                               ) -> List[List]:
        """Extract data from one map

        Parameters
        ----------
        path : Path
            Directory of input maps
        input_data_row : pd.Series
            One row of input data maps
        row_index : int
            Index of input row. Use to identify seabed
        formatfile : str, optional
            Map format supported by xtgeo, by default "irap_binary"
        facies_dict : dict | None, optional
            Lithology value mapping with facies map, by default None

        Returns
        -------
        List[List]
            list of list containing sediment objects

        Raises
        ------
        Exception
            Absence of facies_dict when facies map is specified in input
        """
        name = input_data_row["File_name"]
        fullpath = path / name
        topage = input_data_row["Age"]
        facies_map_flag = False
        top = xtgeo.surface_from_file(fullpath, fformat=formatfile)
        #boundary_polygon = self.model_bound(top)
        sed = self.grid.make_grid_arr()
        location = self.grid._location_xtgeo_z
        loc_depth_val = top.get_fence(location)
        loc_depth_val = loc_depth_val.filled(np.nan)
        if (
            isinstance(input_data_row["Facies_maps"], str)
            and (input_data_row["Facies_maps"]) != "faci_m//-1.pmd"
        ):  # skip basement facies map
            facies_map_flag = True
            facies_path = path / input_data_row["Facies_maps"]
            facies_map = xtgeo.surface_from_file(
                facies_path, fformat=formatfile)
            loc_facies_code = copy.deepcopy(loc_depth_val)
            loc_facies_code = facies_map.get_fence(loc_facies_code)

            if isinstance(facies_dict, type(None)):
                raise Exception("No facies dictionary supplied")
            else:
                pass
        else:
            pass
        indexing_arr = self.grid.indexing_arr
        for ind, i in enumerate(loc_depth_val):
            node_index = indexing_arr[ind]
            if (np.isnan(i[2])):
                if row_index == 1:
                    self.nodes[node_index[0]][node_index[1]] = False
            else:
                if row_index == 1:
                    self.nodes[node_index[0]][node_index[1]] = True
                # loop all locations for 1 horizon
                y = [_sediment_layer_()]
                topdepth = round(i[2], 0)
                y[0].X = i[0]
                y[0].Y = i[1]
                y[0].top = topdepth
                y[0].topage = topage
                y[0].strat = input_data_row["Stratigraphy"]
                y[0].horizon_index = row_index
                # Double check if facies dict contains map values
                if facies_map_flag == True:
                    facies_map_val = loc_facies_code[ind, -1]
                    if np.ma.is_masked(facies_map_val) == False:
                        facies_map_val = int(facies_map_val)
                        facies_map_val = str(facies_map_val)
                        if facies_map_val in facies_dict:
                            facies_val = facies_dict[facies_map_val]
                            y[0].thermoconductivity = float(
                                facies_val["Thermal Conduct. at 20°C"]
                            )
                            y[0].solidus = float(facies_val["Density"])
                            y[0].liquidus = float(facies_val["Density"]) * 0.9
                            # Rybach 1986
                            rhp = (0.00001 * float(facies_val["Density"])) * (
                                (9.52 * float(facies_val["Uranium"]))
                                + (2.56 * float(facies_val["Thorium"]))
                                + (3.48 * float(facies_val["Potassium"]))
                            )
                            rhp = rhp * 1e-6  # microwatt to watt
                            y[0].rhp = rhp
                            y[0].phi = float(facies_val["Initial Porosity"])
                            if facies_val["Compaction Model Key"] == "5":
                                y[0].decay = float(
                                    facies_val["Athy's Factor k (depth)"])
                            else:
                                logger.warning(
                                    f"Input facies properties not using Athy's Factor k for facies ID: {facies_map_val}. Using default {y[0].decay} for compaction"
                                )
                        else:
                            logger.warning(
                                f"No facies map value for X:{y[0].X},Y:{y[0].Y}"
                            )
                sed[node_index[0]][node_index[1]] = y[0]

        return sed

    # Main function to extract sediments

    def extract_nodes(
        self, thread:int, path:Path, formatfile:str="irap_binary", facies_dict:dict|None=None,
    ):
        """Extract model nodes from input data

        Parameters
        ----------
        thread : int
            Number of concurrent process
        path : Path
            Path to map directory
        formatfile : str, optional
            Map format supported by xtgeo, by default "irap_binary"
        facies_dict : dict | None, optional
            Lithology value mapping with facies map, by default None

        Raises
        ------
        ValueError
            Invalid input table. Check self.input_horizons
        ValueError
            self.input_horizons not sorted with ascending age
        """
        self.input_horizons.reset_index(drop=True,inplace=True)
        self.input_horizons = self.input_horizons

        if (
            isinstance(self.input_horizons, pd.DataFrame)
            and len(self.input_horizons.columns) == 4
            and list(self.input_horizons.columns) == ["Age", "File_name", "Facies_maps","Stratigraphy"]
        ):
            self.input_horizons = self.input_horizons.astype(
                dtype={"Age": "int64", "File_name": "object",
                       "Facies_maps": "object","Stratigraphy":"object"}
            )
        else:
            raise ValueError(
                "Invalid input table. Check warmth.input_data_template")
        # Check if age is sorted
        chk = self.input_horizons.apply(lambda x: x.is_monotonic_increasing)
        chk = chk["Age"]
        if chk == True:
            pass
        else:
            raise ValueError(
                "input_data table must be sorted according to Age")
        self.parameters.time_start = int(self.input_horizons.iloc[-1]['Age'])
        sediments_all = []
        poolx = concurrent.futures.ThreadPoolExecutor(max_workers=thread)
        with poolx as executor:
            futures = [
                executor.submit(
                    self._extract_single_horizon,
                    path,
                    row,
                    index,
                    formatfile=formatfile,
                    facies_dict=facies_dict,
                )
                for index, row in self.input_horizons.iterrows()
            ]
            logger.info('Extracting %s sedimentary packages with %s horizons', len(
                futures), len(futures) + 1)
            logger.info('Threads:%s', len(poolx._threads))

            # When each job finishes
            for future in concurrent.futures.as_completed(futures):
                sed = future.result()  # This will also raise any exceptions
                sediments_all.append(sed)

        self._create_nodes(sediments_all)

        return

    def _create_nodes(self, all_sediments_grid: List[List[List]]):
        """Create 1D node from extracted sediment objects

        Parameters
        ----------
        all_sediments_grid : List[List[List]]
            Extracted sediment objects
        """
        indexer = self.grid.indexing_arr


        for index in indexer:

            if self.nodes[index[0]][index[1]] != False:
                node_sed: list[_sediment_layer_] = []
                for sed_grid in all_sediments_grid:
                    node_sed.append(sed_grid[index[0]][index[1]])
                if all(node_sed) is False:
                    logger.warning(f"dropping node {index}. One of the layer has no depth value")
                else:
                    top = np.empty(0)
                    topage = np.empty(0)
                    k_cond = np.empty(0)
                    rhp = np.empty(0)
                    phi = np.empty(0)
                    decay = np.empty(0)
                    solidus = np.empty(0)
                    liquidus = np.empty(0)
                    strat = np.empty(0,dtype=str)
                    inputRef = np.empty(0,dtype=int)
                    for hor in node_sed:
                        top = np.append(top, float(hor.top))
                        topage = np.append(topage, int(hor.topage))
                        k_cond = np.append(k_cond, float(hor.thermoconductivity))
                        rhp = np.append(rhp, float(hor.rhp))
                        phi = np.append(phi, float(hor.phi))
                        decay = np.append(decay, float(hor.decay))
                        solidus = np.append(solidus, float(hor.solidus))
                        liquidus = np.append(liquidus, float(hor.liquidus))
                        strat = np.append(strat,hor.strat)
                        inputRef= np.append(inputRef,hor.horizon_index)

        
                    df = pd.DataFrame({'top': top, 'topage': topage, 'k_cond': k_cond,
                                                'rhp': rhp, 'phi': phi, 'decay': decay, 'solidus': solidus, 'liquidus': liquidus,'strat':strat,'horizonIndex':inputRef})
                    df = df.sort_values(by=["topage"],ignore_index=True)

                    # print(df)
                    #df.reset_index(drop=True,inplace=True)
                    # df.at[2, 'top'] = np.nan            
                    # df.at[3, 'top'] = np.nan
                    checker = self._check_nan_sed(df)

                    if checker is False:
                        self.nodes[index[0]][index[1]] = False
                    else:
                        df = self._fix_nan_sed(df)
                        n = single_node()
                        n.X=node_sed[0].X
                        n.Y=node_sed[0].Y
                        n.sediments_inputs=df
                        n.indexer = index
                        self.nodes[index[0]][index[1]] = n
            else:
                pass
        return
    
    def _check_nan_sed(self,df:pd.DataFrame)-> bool:
        """Validate node sediment.
        Top and base must not be NaN
        Max 3 NaN allowed in sedimentary column
        Max 2 consecutive NaN allow


        Parameters
        ----------
        df : pd.DataFrame
            node.sediment object

        Returns
        -------
        bool
            True if passed validation
        """
        if np.isnan(df.iloc[-1]["top"]):
            return False
        if np.isnan(df.iloc[0]["top"]):
            return False
        max_nan_allowed = 3
        if df['top'].isna().sum() > max_nan_allowed:
            return False
        max_consecutive_nan = 2
        consecutive_nan = df['top'].isnull().astype(int).groupby((df['top'].notnull() != df['top'].shift().notnull()).cumsum()).sum()
        if consecutive_nan.max()> max_consecutive_nan:
            return False
        return True
    
    def _fix_nan_sed(self, df:pd.DataFrame)->pd.DataFrame:
        """Cleanup cross-cutting sedimentary column

        Parameters
        ----------
        df : pd.DataFrame
            node.sediment object

        Returns
        -------
        pd.DataFrame
            Cleaned node.sediment object
        """
        idx_nan = df['top'].index[df['top'].apply(np.isnan)]
        for i in idx_nan:
            above_idx=i-1
            above = df["top"][above_idx]
            below_idx = i+1
            below = df["top"][below_idx]
            while np.isnan(below):
                below_idx+=1
                below= df["top"][below_idx]
            top_strat = df["strat"][above_idx]
            if top_strat == "Erosive":
                new_top = above
            else:
                new_top = below
            df.at[i,"top"] = new_top
        return df


    def define_geometry(self, path: Path, xinc: float = None, yinc: float = None, fformat="irap_binary"):
        """Define geometry of a 3D model by using a map

        Parameters
        ----------
        path : Path
            Path to the map used in defining model geometry
        xinc : float, optional
            Overwrite node distance in x direction from input map, by default None
        yinc : float, optional
            Overwrite node distance in y direction from input map, by default None
        fformat : str, optional
            Map format supported by xtgeo, by default "irap_binary"
        """
        hor = xtgeo.surface_from_file(path, values=False, fformat=fformat)
        hor.unrotate()
        hor.autocrop()
        if hor.yflip != 1:
            raise Exception("Flipped surface not supported")
        if isinstance(xinc, type(None)):
            xinc = hor.xinc
        if isinstance(yinc, type(None)):
            yinc = hor.yinc
        xmax = hor.xori+(hor.ncol*hor.xinc)
        ymax = hor.yori+(hor.nrow*hor.yinc)
        new_ncol = math.floor((xmax-hor.xori)/xinc)
        new_nrow = math.floor((ymax-hor.yori)/yinc)
        self.grid = Grid(hor.xori, hor.yori, new_ncol, new_nrow, xinc, yinc)
        self.nodes = self.grid.make_grid_arr()
        return

    @property
    def locations(self)->np.ndarray:
        """Locations of all 1D nodes

        Returns
        -------
        np.ndarray
            A 2D array of locations of all nodes
        """
        return self.grid.location_grid

    def iter_node(self)->Iterator[single_node]:
        """Iterate all 1D nodes

        Yields
        ------
        Iterator[single_node]
            1D node
        """
        for row in self.nodes:
            for col in row:
                if isinstance(col,bool)==False:
                    yield col
    @property
    def indexer_full_sim(self)->list:
        return [i.indexer for i in self.iter_node() if i._full_simulation is True]
    @property
    def n_valid_node(self)->int:
        return len([i for i in self.iter_node()])
    def set_eustatic_sea_level(self, sealevel:dict|None=None):
        """Set eustatic sea level correction for subsidence modelling

        Parameters
        ----------
        sealevel : dict | None, optional
            Eustatic sea level data, by default None
        """
        if isinstance(sealevel, dict):
            time = np.arange(self.parameters.time_end,
                             self.parameters.time_start + 1, self.parameters.time_step_Ma * -1)
            sealevel = np.interp(time, list(
                sealevel.keys()), list(sealevel.values()))
            self.parameters.eustatic_sea_level = {
                "age": time, "sea_level_changes": sealevel}
        elif isinstance(sealevel, type(None)):
            self.parameters.eustatic_sea_level = {
                "age": np.arange(
                    self.parameters.time_end, self.parameters.time_start +
                    1, self.parameters.time_step_Ma * -1
                ),
                "sea_level_changes": np.full(
                    np.arange(
                        self.parameters.time_end, self.parameters.time_start +
                        1, self.parameters.time_step_Ma * -1
                    ).size,
                    0.0,
                ),
            }
        else:
            logger.warning("Invalid sealevel data. Expect dict")
        return
