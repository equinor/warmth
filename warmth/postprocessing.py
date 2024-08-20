import time
from typing import Tuple, TypedDict
from scipy import interpolate
import numpy as np
import pandas as pd
from .logging import logger


class Results:
    """Simulation results
    """
    def __init__(self, depth:np.ndarray, temperature:np.ndarray,
                 sediments_ids:np.ndarray,sediment_input:pd.DataFrame,k_crust:float,k_lith:float,k_asth:float):
        self._depth=depth
        self._temperature=temperature
        self._sediments_ids=sediments_ids
        self._sediment_input=sediment_input
        self._k_crust=k_crust
        self._k_lith=k_lith
        self._k_asth=k_asth

    class resultValues(TypedDict):
        depth: np.ndarray[np.float64]
        layerId: np.ndarray[np.int32]
        values:np.ndarray[np.float64]

    def top_crust(self,age:int)->float:
        """Depth of crust

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        float
            Depth of crust from sea level (m)
        """
        depth_idx= np.where(self.sediment_ids(age) == -1)[0][0]
        return self._depth[depth_idx,age]

    def top_lithosphere(self,age:int)->float:
        """Depth of lithospheric mantle

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        float
            Depth of lithospheric mantle / Moho from sea level (m)
        """    
        depth_idx= np.where(self.sediment_ids(age) == -2)[0][0]
        return self._depth[depth_idx,age]

    def top_asthenosphere(self,age:int)->float:
        """Depth of Asthenosphere

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        float
            Depth of Asthenosphere from sea level (m)
        """         
        depth_idx= np.where(self.sediment_ids(age) == -3)[0][0]
        return self._depth[depth_idx,age]

    def crust_thickness(self,age:int)->float:
        """Thickness of crust

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        float
            Thickness of crust (m)
        """
        return self.top_lithosphere(age)-self.top_crust(age)
    
    def lithosphere_thickness(self,age:int)->float:
        """Thickness of lithospheric mantle

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        float
            Thickness of lithospheric mantle (m)
        """
        return self.top_asthenosphere(age)-self.top_lithosphere(age)
    
    def depth(self,age:int)->np.ndarray[np.float64]:
        """Depth reference for results

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        np.ndarray
            Top and base of all cells
        """
        return self._depth[:,age]

    def temperature(self,age:int,sediment_id:int|None=None)->resultValues:
        """Temperature at top and base of cells

        Parameters
        ----------
        age : int
            Geological age
        sediment_id : int | None, optional
            Optional filter using id of layer by default None

        Returns
        -------
        np.ndarray
            Temperature at top and base of cells
        """
        v = self._temperature[:,age]
        sed_id = self.sediment_ids(age)
        d = self.depth(age)
        if isinstance(sediment_id,int):
            top_idx,base_idx=self._filter_sed_id_index(sediment_id,sed_id)
            d = d[top_idx:base_idx+1]
            sed_id=sed_id[top_idx:base_idx]
            v=v[top_idx:base_idx+1]
        return {"depth":d,"layerId":sed_id,"values":v}

    def sediment_ids(self,age:int)->np.ndarray[np.int32]:
        """Layer ids at the centre of cells

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        np.ndarray
            Layer ids at the center of cells
        """
        return self._sediments_ids[:,age]

    def sediment_porosity(self,age:int,sediment_id:int|None=None)->resultValues:
        """Porosity at the centre of cells

        Parameters
        ----------
        age : int
            Geological age
        sediment_id : int | None, optional
            Optional filter using id of layer by default None

        Returns
        -------
        dict
            Porosity at centre of cells
        """
        sed_id = self.sediment_ids(age)
        initial_poro = np.full(sed_id.size,fill_value=0,dtype=float)
        initial_decay = np.full(sed_id.size,fill_value=0,dtype=float)
        for idx, row in self._sediment_input.iterrows():
            sed_idx = np.argwhere(sed_id == idx).flatten()
            if sed_idx.size >0:
                initial_poro[sed_idx] = row["phi"]
                initial_decay[sed_idx] = row["decay"]
        d = self.depth(age)
        x1=d[1:]/1e3
        x2 = d[:-1]/1e3
        diff = x2 - x1
        exp = -1*initial_decay
        phi1 = np.exp(exp*x1)*np.expm1(exp*diff)/diff
        v=-1*initial_poro/initial_decay*phi1
        v[np.isnan(v)] = 0
        d = (d[1:]+d[:-1])/2
        if isinstance(sediment_id,int):
            top_idx,base_idx=self._filter_sed_id_index(sediment_id,sed_id)
            d = d[top_idx:base_idx]
            sed_id=sed_id[top_idx:base_idx]
            v=v[top_idx:base_idx]
        return {"depth":d,"layerId":sed_id,"values":v}

    def sediment_density(self, age:int,sediment_id:int|None=None)->resultValues:
        from .forward_modelling import Forward_model
        sed_poro = self.sediment_porosity(age, sediment_id)
        start_idx = np.argwhere(sed_poro["layerId"]==0)[0][0]
        end_idx = np.argwhere(sed_poro["layerId"]==-1)[0][0]
        bulk_density = []
        for layer_id in sed_poro["layerId"][start_idx:end_idx]:
            bulk_density.append(self._sediment_input.iloc[layer_id]["solidus"])
        sed_poro["values"][start_idx:end_idx]=Forward_model._sediment_density(sed_poro["values"][start_idx:end_idx],bulk_density,1000)
        return sed_poro


    def _reference_conductivity(self,age:int)->np.ndarray:
        """Conductivity of layers at 20C reference temperature

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        np.ndarray
            Conductivity of layers at 20C reference temperature (W/K.m^2)
        """
        sed_id = self.sediment_ids(age)
        cond = np.full(sed_id.size,fill_value=np.nan,dtype=float)
        cond[sed_id == -1 ] = self._k_crust
        cond[sed_id == -2 ] = self._k_lith
        cond[sed_id == -3 ] = self._k_asth
        for idx, row in self._sediment_input.iterrows():
            cond[sed_id == idx ] = row["k_cond"]
        return cond

    def effective_conductivity(self,age:int,sediment_id:int|None=None)->resultValues:
        """Effective conductivity at the centre of cells

        Parameters
        ----------
        age : int
            Geological age
        sediment_id : int | None, optional
            Optional filter using id of layer by default None

        Returns
        -------
        resultValues
            Effective conductivity at centre of cells (W/K.m^2)
        """
        from .forward_modelling import Forward_model
        v = Forward_model._sediment_conductivity_sekiguchi(self.sediment_porosity(age)["values"],self._reference_conductivity(age),self.temperature(age)["values"])
        d = self.depth(age)
        d = (d[1:]+d[:-1])/2
        sed_id = self.sediment_ids(age)
        if isinstance(sediment_id,int):
            top_idx,base_idx=self._filter_sed_id_index(sediment_id,sed_id)
            d = d[top_idx:base_idx]
            sed_id=sed_id[top_idx:base_idx]
            v=v[top_idx:base_idx]
        return {"depth":d,"layerId":sed_id,"values":v,"reference":self._reference_conductivity(age)}

    def heatflow(self,age:int,sediment_id:int|None=None)->resultValues:
        """Heat flow at the centre of cells

        Parameters
        ----------
        age : int
            Geological age
        sediment_id : int | None, optional
            Optional filter using id of layer by default None

        Returns
        -------
        dict
            Heat flow at centre of cells
        """
        t = self.temperature(age)["values"]
        d = self.depth(age)
        sed_id = self.sediment_ids(age)
        eff_con = self.effective_conductivity(age)
        combined_con = eff_con["reference"].copy()
        combined_con[ sed_id>=0 ] = eff_con["values"][sed_id>=0]
        v = combined_con*(t[1:]-t[:-1])/(d[1:]-d[:-1])
        d = (d[1:]+d[:-1])/2
        if isinstance(sediment_id,int):
            top_idx,base_idx=self._filter_sed_id_index(sediment_id,sed_id)
            d = d[top_idx:base_idx]
            sed_id=sed_id[top_idx:base_idx]
            v=v[top_idx:base_idx]
        return {"depth":d,"layerId":sed_id,"values":v}
    
    def basement_heatflow(self,age:int)-> float:
        """Heat flow from the crust to the base of sediments

        Parameters
        ----------
        age : int
            Geological age

        Returns
        -------
        float
            Basement heat flow (W/m3)
        """
        hf=self.heatflow(age)
        top_crust_idx = np.argwhere(hf["layerId"] == -1)[0][0]
        res = hf["values"][top_crust_idx]
        if top_crust_idx>0:
            above = hf["values"][top_crust_idx-1]
            if np.isnan(above) is False: 
                res = (res+above)/2
        return res
    def seabed(self,age:int)->np.ndarray[np.float64]:
        idx = np.where(~np.isnan(self._temperature[:,age]))[0][0]
        return self._depth[idx,age]
    
    def _filter_sed_id_index(self,sed_id:int,sed_id_arr:np.ndarray)->Tuple[int,int]:
        """Filter results by layer id

        Parameters
        ----------
        sed_id : int
            layer id
        sed_id_arr : np.ndarray
            Array of all layer id

        Returns
        -------
        Tuple[int,int]
            Indices for top and base of array

        Raises
        ------
        Exception
            Layer id not existing at the time step
        """
        if sed_id in sed_id_arr:
            top_sediment_index= np.argwhere(sed_id_arr==sed_id)[0][0]
            base_sediment_index = np.argwhere(sed_id_arr==sed_id)[-1][0]+1
            return top_sediment_index,base_sediment_index
        else:
            raise Exception(f"Invalid sediment id {sed_id}. Valid ids: {np.unique(sed_id_arr[~np.isnan(sed_id_arr)])}")
    
    def _mid_pt_temperature(self,arr:np.ndarray[np.float64])->np.ndarray[np.float64]:
        return (arr[1:]+arr[:-1])/2
    
    
    def temperature_history(self,sed_id:int)->np.ndarray[np.float64]:
        max_age_for_sed= np.max(np.argwhere(self._sediments_ids==sed_id)[:,1])
        all_valid_sed_age=np.arange(max_age_for_sed+1)
        total_cells = self.sediment_ids(0)[self.sediment_ids(0)==sed_id].size
        res = np.full((total_cells,max_age_for_sed+1),np.nan)
        cell_idx = total_cells-1
        starting_idx=-1
        while cell_idx >=0:
            for age in reversed(all_valid_sed_age): 
                t_old = self._mid_pt_temperature(self.temperature(age,sed_id)["values"])
                if t_old.size >=starting_idx*-1:
                    res[cell_idx,age]=t_old[starting_idx]
                else:
                    pass
            starting_idx = starting_idx-1
            cell_idx = cell_idx-1
        return res
    def vitrinite_reflectance_history(self,sed_id,Rotype="Easy%RoDL"):
        temp_h = self.temperature_history(sed_id)
        for i in range(temp_h.shape[0]):
            cell = temp_h[i]
            if Rotype=="Easy%RoDL":
                vr = VR.easyRoDL(cell[~np.isnan(cell)])
            else:
                raise Exception (f"{Rotype} not implemented")
            temp_h[i,:vr.size]=vr.flatten()
        return temp_h
    def vitrinite_reflectance(self,sed_id:int|None=None,Rotype="Easy%RoDL")->resultValues: 
        sed_ids_all = self.sediment_ids(0)
        d = self.depth(0)
        d = (d[1:]+d[:-1])/2
        v = self.vitrinite_reflectance_history(0,Rotype)[:,0]
        if isinstance(sed_id,type(None)):
            only_sed= np.unique(sed_ids_all)
            only_sed = only_sed[only_sed>-1]
            v = np.empty(0)
            for i in only_sed:
                v_new = self.vitrinite_reflectance_history(int(i),Rotype)[:,0]
                v = np.append(v,v_new)
            top_sed_idx,_=self._filter_sed_id_index(0,sed_ids_all)
            top_crust_idx,_=self._filter_sed_id_index(-1,sed_ids_all)
            d=d[top_sed_idx:top_crust_idx]
            sed_ids_all=sed_ids_all[top_sed_idx:top_crust_idx]
        else:
            v = self.vitrinite_reflectance_history(sed_id,Rotype)[:,0]
            top_idx,base_idx=self._filter_sed_id_index(sed_id,sed_ids_all)
            d=d[top_idx:base_idx]
            sed_ids_all=sed_ids_all[top_idx:base_idx]
        return {"depth":d,"layerId":sed_ids_all,"values":v}
    
class VR:
    def __init__(self) -> None:
        pass
    @staticmethod
    def cum_reacted(A, E, weights, time, temp_k):
        c = 0.0019858775  # 8.3144621/4184 (  Convert from kilocalorie to Joule by multiplication of 4184)
        # Dimensions
        nt = temp_k.size
        nw = weights.size

        # Heating rate between different time-steps, degC/s
        heat_rate = np.zeros([nt, 1])
        for i in range(1, time.shape[0]):
            heat_rate[i] = (
                (temp_k[i] - temp_k[i - 1]) / (time[i] - time[i - 1]) / 31600000000000
            )

        # I: nt x nw, Equation 10 in Sweeney and Burnham, 1990
        I = np.zeros([nt, nw])
        for i in range(0, time.size):
            for j in range(0, weights.size):
                E_RT_temp = E[j] / (c * temp_k[i])
                I[i, j] = (
                    A
                    * temp_k[i]
                    * np.exp(-E_RT_temp)
                    * (
                        1.0
                        - (E_RT_temp**2 + 2.334733 * E_RT_temp + 0.250621)
                        / (E_RT_temp**2 + 3.330657 * E_RT_temp + 1.681534)
                    )
                )

        # Cumulative Reacted
        CR_easy = np.zeros([nt, 1])
        DI = np.zeros([nt, nw])

        # When computing the deltas, drop index 0, start at index 1
        with np.errstate(divide='ignore',invalid='ignore'):
            for i in range(1, time.size):
                for j in range(0, weights.size):
                    DI[i, j] = DI[i - 1, j] + (I[i, j] - I[i - 1, j]) / heat_rate[i]
                    CR_easy[i] += weights[j] * (1 - np.exp(-DI[i, j]))

        return CR_easy
    
    @staticmethod
    def easyRoDL(temperature:np.ndarray[np.float64])->np.ndarray[np.float64]:
        """Easy%RoDL from temperature history

        Parameters
        ----------
        temperature : np.ndarray[np.float64]
            Temperature history in ascending time per Ma (C)

        Returns
        -------
        Vitrinite reflectance np.ndarray[np.float64]
            Vitrinite reflectance based on Easy%RoDL
        """
        temp_k = temperature+273
        temp_k = np.flip(temp_k)
        time=np.arange(np.count_nonzero(~np.isnan(temp_k)))
        A_V = 2e15
        E_easy_V = np.array(
            [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76]
        )
        weights_easy_V = np.array(
            [
                0.03,
                0.04,
                0.045,
                0.045,
                0.045,
                0.04,
                0.045,
                0.05,
                0.055,
                0.06,
                0.07,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.03,
                0.025,
            ]
        )

        Cr_easy_V = VR.cum_reacted(A_V, E_easy_V, weights_easy_V, time, temp_k)
        RoV = 0.223 * np.exp(3.7 * Cr_easy_V)
        return np.flip(RoV)
    
class Results_interpolator:
    def __init__(self, builder) -> None:
        self._builder = builder
        self._values = ["kAsth","crustRHP","qbase","T0"]
        self._values_arr = ["subsidence","crust_ls","lith_ls"]
        self._n_age=None
        self._n_valid_node= None
        self._x = None
        self._y=None
        pass
    
    def iter_full_sim_nodes(self):
        for node in self._builder.iter_node():
            if node._full_simulation:
                yield node

    
    def _get_x_y(self)->None:
        # x = np.zeros(self.n_valid_node)
        # y = np.zeros(self.n_valid_node)
        x = []
        y = []
        for count, node in enumerate(self.iter_full_sim_nodes()):
            # x[count]=node.X
            # y[count]= node.Y
            x.append(node.X)
            y.append(node.Y)
            if count == 0:
                self._n_age = node.crust_ls.size
        self._n_valid_node = len(x)
        self._x = np.array(x)
        self._y= np.array(y)
        return
    @property
    def n_valid_node(self)->np.ndarray[np.float64]:
        if isinstance(self._n_valid_node,type(None)):
            self._get_x_y()
        return self._n_valid_node
    @property
    def x(self)->np.ndarray[np.float64]:
        if isinstance(self._x,type(None)):
            self._get_x_y()
        return self._x
    @property
    def y(self)->np.ndarray[np.float64]:
        if isinstance(self._y,type(None)):
            self._get_x_y()
        return self._y
    @property
    def n_age(self)->int:
        if isinstance(self._n_age,type(None)):
            self._get_x_y()
        return self._n_age
    
    def interpolator(self,val):
        grid = self._builder.grid
        grid_x, grid_y = np.mgrid[
            grid.origin_x: grid.xmax: grid.step_x,
            grid.origin_y: grid.ymax: grid.step_y,
        ]
        rbfi = interpolate.Rbf(self.x, self.y, val)
        di = rbfi(grid_x, grid_y)
        return di
    
    def interp_value(self):
        for prop in self._values:
            logger.warning(f"Interpolating {prop}")
            val = np.zeros(self.n_valid_node)
            for count, node in enumerate(self.iter_full_sim_nodes()):
                val[count] = getattr(node,prop)

            interped = self.interpolator(val)
            for n in self._builder.iter_node():
                if n._full_simulation is False:
                    idx = n.indexer
                    val =interped[idx[1],idx[0]]
                    setattr(n,prop,val)
        return
    
    def interp_arr(self):
        for prop in self._values_arr:
            logger.warning(f"Interpolating {prop}")
            #extract all data from all full simulated nodes
            val = np.zeros((self.n_valid_node,self.n_age))
            for count, node in enumerate(self.iter_full_sim_nodes()):
                val[count,:] = getattr(node,prop)
            #Handle not simulated nodes
            prop ="_"+prop
            for age in range(self.n_age):
                # filter to age
                interp_all_this_age = self.interpolator(val[:,age])
                #set the nodes
                for node in self._builder.iter_node():
                    if node._full_simulation is False:
                        if isinstance(getattr(node,prop),type(None)):
                            setattr(node,prop,np.zeros(self.n_age))
                        idx = node.indexer
                        interpolated_val =interp_all_this_age[idx[1],idx[0]]
                        arr = getattr(node,prop)
                        arr[age] =interpolated_val
                        setattr(node,prop,arr)
        return

    def run(self):
        self.interp_value()
        self.interp_arr()
        return