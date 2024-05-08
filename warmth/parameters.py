from pathlib import Path
import pandas as pd
import pickle
from warmth.utils import compressed_pickle_open, compressed_pickle_save
from .logging import logger


class Parameters:
    """Parameters of the model
    """
    def __init__(self) -> None:
        self.alphav: float = 3.0e-5
        self.adiab: float = 0.3e-3
        self.cp: float = 1000
        self.g: float = 9.8
        self.tetha: float = 0.01
        self.convergence: float = 1e-4
        self.rhowater: float = 1000.0
        self.rhoAir: float = 1.0
        self.HPdcr: float = 16e3  # Length scale of heat production decay
        self.bflux: bool = False
        self.vertical_resolution_sediments: int = 100
        self.resolution: int = 1000
        self.experimental: bool = True
        self.initial_hc_max = 60000
        self.initial_hc_min = 15000
        self.initial_hLith_max = 120000
        self.initial_hLith_min = 60000
        self.hc_calibration_outer_loop = 3
        self.hc_calibration_inner_loop = 6
        self.hc_calibration_max_nodes = 20
        self.time_start: int = 10
        self.time_end: int = 0
        self.projection = 0  # EPSG code
        self.sediment_fill_margin: float = 100
        self.melt = False
        self.partialmelting_LL = 0.02
        self.partialmelting_UL = 0.2
        self.partialmelting_extrude = True
        self.melt_time: int = -100
        self.time_step_Ma: int = -1
        self.max_beta: float = 15.0
        self.myr2s = 314712e8
        self.maxContLithFlag: bool = True
        self.maxContLith: float = 130000.0
        self.starting_beta: float = 1.1
        self.positive_down = True
        self.name:str = "model"
        self.output_path:Path=Path('./simout')
        pass

    @property
    def alphav(self) -> float:
        return self._alphav

    @alphav.setter
    def alphav(self, val):
        if isinstance(val, (float, int)):
            self._alphav = val
        else:
            logger.warning("Float")

    @property
    def adiab(self) -> float:
        return self._adiab

    @adiab.setter
    def adiab(self, val):
        if isinstance(val, (float, int)):
            self._adiab = val
        else:
            logger.warning("Float")

    @property
    def cp(self) -> float:
        return self._cp

    @cp.setter
    def cp(self, val):
        if isinstance(val, int):
            self._cp = val
        else:
            logger.warning("Int")

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, val):
        if isinstance(val, (float, int)):
            self._g = val
        else:
            logger.warning("Float")

    @property
    def tetha(self):
        return self._tetha

    @tetha.setter
    def tetha(self, val):
        if isinstance(val, (float, int)):
            self._tetha = val
        else:
            logger.warning("Float")

    @property
    def conv(self):
        return self._conv

    @conv.setter
    def conv(self, val):
        if isinstance(val, (float, int)):
            self._conv = val
        else:
            logger.warning("Float")

    @property
    def rhowater(self):
        return self._rhowater

    @rhowater.setter
    def rhowater(self, val):
        if isinstance(val, (float, int)):
            self._rhowater = val
        else:
            logger.warning("Float")

    @property
    def rhoAir(self):
        return self._rhoAir

    @rhoAir.setter
    def rhoAir(self, val):
        if isinstance(val, (float, int)):
            self._rhoAir = val
        else:
            logger.warning("Float")

    @property
    def HPdcr(self):
        return self._HPdcr

    @HPdcr.setter
    def HPdcr(self, val):
        if isinstance(val, (float, int)):
            self._HPdcr = val
        else:
            logger.warning("Float")

    @property
    def bflux(self):
        return self._bflux

    @bflux.setter
    def bflux(self, val):
        if isinstance(val, bool):
            self._bflux = val
        else:
            logger.warning("Accept boolean")
        return

    @property
    def vertical_resolution_sediments(self):
        return self._sedres

    @vertical_resolution_sediments.setter
    def vertical_resolution_sediments(self, val):
        if isinstance(val, int):
            self._sedres = val
        else:
            logger.warning("Int")

    @property
    def resolution(self):
        return self._res

    @resolution.setter
    def resolution(self, val):
        if isinstance(val, int):
            self._res = val
        else:
            logger.warning("Int")

    @property
    def experimental(self):
        return self._experimental

    @experimental.setter
    def experimental(self, val):
        if isinstance(val, bool):
            self._experimental = val
        else:
            logger.warning("Accept boolean")
        return

    @property
    def initial_hc_max(self):
        return self._initial_hc_max

    @initial_hc_max.setter
    def initial_hc_max(self, val):
        if isinstance(val, int):
            if hasattr(self, 'initial_hc_min'):
                if val < self.initial_hc_min:
                    logger.warning("must be larger than initial_hc_min")
                else:
                    self._initial_hc_max = val
            else:
                self._initial_hc_max = val
        else:
            logger.warning("Accept int and must be larger than initial_hc_max")

    @property
    def initial_hc_min(self):
        return self._initial_hc_min

    @initial_hc_min.setter
    def initial_hc_min(self, val):
        if isinstance(val, int):
            if hasattr(self, 'initial_hc_max'):
                if val > self.initial_hc_max:
                    logger.warning("must be smaller than initial_hc_max")
                else:
                    self._initial_hc_min = val
            else:
                self._initial_hc_min = val
        else:
            logger.warning(
                "Accept int and must be smaller than initial_hc_min")

    @property
    def initial_hLith_max(self):
        """ Maximum allowed lithosphere thickness for crustal thickness calibration"""
        return self._initial_hLith_max

    @initial_hLith_max.setter
    def initial_hLith_max(self, val):
        if isinstance(val, int):
            if hasattr(self, 'initial_hLith_min'):
                if val < self.initial_hc_min:
                    logger.warning("must be larger than initial_hc_min")
                else:
                    self._initial_hLith_max = val
            else:
                self._initial_hLith_max = val
        else:
            logger.warning(
                "Accept int and must be larger than initial_hLith_min")

    @property
    def initial_hLith_min(self):
        """ Minimum allowed lithosphere thickness for crustal thickness calibration"""
        return self._initial_hLith_min

    @initial_hLith_min.setter
    def initial_hLith_min(self, val):
        if isinstance(val, int):
            if hasattr(self, 'initial_hLith_max'):
                if val > self.initial_hLith_max:
                    logger.warning("must be smaller than initial_hc_max")
                else:
                    self._initial_hLith_min = val
            else:
                self._initial_hLith_min = val
        else:
            logger.warning(
                "Accept int and must be smaller than initial_hLith_max")

    @property
    def hc_calibration_outer_loop(self):
        return self._hc_calibration_outer_loop

    @hc_calibration_outer_loop.setter
    def hc_calibration_outer_loop(self, val):
        if isinstance(val, int):
            self._hc_calibration_outer_loop = val
        else:
            logger.warning("Int")

    @property
    def hc_calibration_inner_loop(self):
        return self._hc_calibration_inner_loop

    @hc_calibration_inner_loop.setter
    def hc_calibration_inner_loop(self, val):
        if isinstance(val, int):
            self._hc_calibration_inner_loop = val
        else:
            logger.warning("Int")

    @property
    def hc_calibration_max_nodes(self):
        return self._hc_calibration_max_nodes

    @hc_calibration_max_nodes.setter
    def hc_calibration_max_nodes(self, val):
        if isinstance(val, int):
            self._hc_calibration_max_nodes = val
        else:
            logger.warning("Int")

    @property
    def time_start(self):
        """Start age of the model in Ma

        :return: Start age of model
        :rtype: int
        """
        return self._time_start

    @time_start.setter
    def time_start(self, val):
        if isinstance(val, int):
            if hasattr(self, 'time_end'):
                if val < self.time_end:
                    logger.warning(
                        f"must be larger/older than time_end {self.time_end}")
                else:
                    self._time_start = val
            else:
                self._time_start = val
        else:
            logger.warning("Accept int")

    @property
    def time_end(self):
        """End age of the model in Ma

        :return: End age of model
        :rtype: int
        """
        return self._time_end

    @time_end.setter
    def time_end(self, val):
        if isinstance(val, int):
            if hasattr(self, 'time_start'):
                if val > self.time_start:
                    logger.warning(
                        f"must be smaller/younger than time_start {self.time_start}")
                else:
                    self._time_end = val
            else:
                self._time_end = val
        else:
            logger.warning("Accept int")

    @property
    def positive_down(self):
        """Depth values are positive downwards

        :return: True if positve downwards
        :rtype: bool
        """
        return self._positive_down

    @positive_down.setter
    def positive_down(self, val):
        if isinstance(val, bool):
            self._positive_down = val
        else:
            logger.warning("Accept boolean")
        return

    @property
    def time_step_Ma(self):
        """Time step to solve

        :return: Time step
        :rtype: -1
        """
        return self._time_step_Ma

    @time_step_Ma.setter
    def time_step_Ma(self, val):
        if val == -1:
            self._time_step_Ma = val
        else:
            logger.warning("Accept -1 only")
        return

    def dump(self,filepath:Path):
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    


