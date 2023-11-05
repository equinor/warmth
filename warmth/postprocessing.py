from typing import Tuple, TypedDict
import numpy as np
import pandas as pd


class Results:
    """Simulation results
    """
    def __init__(self,depth:np.ndarray, temperature:np.ndarray,sediments_ids:np.ndarray,sediment_input:pd.DataFrame,k_crust:float,k_lith:float,k_asth:float):
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
        value:np.ndarray[np.float64]

    @property
    def ages(self)->np.ndarray[np.int32]:
        """Array of all simulated ages

        Returns
        -------
        np.ndarray
            Array of ages
        """
        return np.arange(self._depth.shape[1],dtype=np.int32)

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
        return {"depth":d,"layerId":sed_id,"values":v}

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
        v = self.effective_conductivity(age)["values"]*(t[1:]-t[:-1])/(d[1:]-d[:-1])
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
        sed_id = self.sediment_ids(age)
        top_crust_idx= np.argwhere(sed_id==-1)[0][0]
        hf=self.heatflow(age)["values"]
        res = hf[top_crust_idx]
        if top_crust_idx>0:
            above = hf[top_crust_idx-1]
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