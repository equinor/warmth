from typing import Tuple
import numpy as np
import math
from .logging import logger
from .build import single_node
from .parameters import Parameters
from scipy import interpolate
from scipy.linalg import solve_banded 


class Forward_model:
    """1D simulation
    """
    def __init__(self, parameters: Parameters, current_node: single_node) -> None:
        self._parameters = parameters
        self.current_node = current_node
        pass

    def simulate_single_node(self):
        """Start simulating self.current_node
        """
        if isinstance(self.current_node.rift, list):
            self.current_node.rift = np.stack(self.current_node.rift)
        if isinstance(self.current_node.paleoWD, list):
            self.current_node.paleoWD = np.array(self.current_node.paleoWD)
        self.current_node.maximum_burial_depth = np.zeros_like(self.current_node.paleoWD)
        if self.current_node.rift.size < 2:
            self.current_node.error = 'No rift event'
        else:
            self._setup_initial_conditions()
            # Solve continental points
            self.simulate_continental()
            self.current_node.Tinit = None
        return

    def _setup_initial_conditions(self):
        """Setup initial model condition. Make sure model start is at thermal equilibrium
        """
        self.current_node.sediment_fill_margin = self._parameters.sediment_fill_margin
        # Create mesh in crust and lithosphere
        self._generate_lithosphere_cells()
        # Calculated sedimentation rate and sediment thickness through time
        self._sedimentation()
        # Initial radiogenic heat production without sediments
        self.current_node.initial_crustal_HP = self._heat_production(
            self.current_node.coord_initial,
            0,
            self.current_node.hc,
            self.current_node.shf-self.current_node.qbase,
        )
        # Update k lithospheric mantle to keep LAB at 1300C
        self._update_initial_kLith()
        self._initial_temperature()
        self._initial_height_of_sealevel()
        return

    def sediment_density(self, mean_porosity: np.ndarray[np.float64], density: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Effective density of sediment cells taking into account of water density in pores

        Parameters
        ----------
        mean_porosity : np.ndarray[np.float64]
            Porosity of the sediments
        density : np.ndarray[np.float64]
            Density of sediment

        Returns
        -------
        mean_sediments_density : np.ndarray[np.float64]
            Effective density of sediment cells
        """
        return self._sediment_density(mean_porosity, density,self._parameters.rhowater)
    
    @staticmethod
    def _sediment_density(mean_porosity: np.ndarray[np.float64], density: np.ndarray[np.float64],rhowater) -> np.ndarray[np.float64]:
        """Effective density of sediment cells taking into account of water density in pores

        Parameters
        ----------
        mean_porosity : np.ndarray[np.float64]
            Porosity of the sediments
        density : np.ndarray[np.float64]
            Density of sediment

        Returns
        -------
        mean_sediments_density : np.ndarray[np.float64]
            Effective density of sediment cells
        """
        mean_sediments_density = (
            mean_porosity * rhowater +
            (1 - mean_porosity) * density
        )
        return mean_sediments_density

    @staticmethod
    def _sediment_conductivity_sekiguchi(mean_porosity: np.ndarray[np.float64], conductivity: np.ndarray[np.float64],temperature_C:np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Effective conductivity of sediments based on Sekiguchi 1984

        Parameters
        ----------
        mean_porosity : np.ndarray[np.float64]
            Porosity of sediments
        conductivity : np.ndarray[np.float64]
            Reference 20C conductivity of sediments
        temperature_C : np.ndarray[np.float64]
            Temperature of sediments

        Returns
        -------
        effective_conductivity : np.ndarray[np.float64]
            Effective conductivity of sediments
        """
        mid_pt_temperautureC = (temperature_C[1:] + temperature_C[:-1]) / 2
        temperature_K=273.15+mid_pt_temperautureC
        conductivity = 1.84+358*((1.0227*conductivity)-1.882)*((1/temperature_K)-0.00068)
        effective_conductivity = conductivity*(1-mean_porosity)
        return effective_conductivity

    def _check_beta(self, wd_diff: float, beta_current: float, beta_all: np.ndarray[np.float64], Wd_diff_all: np.ndarray[np.float64]) -> tuple[bool, np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Check if current beta factor matches the observed subsidence

        Parameters
        ----------
        wd_diff : float
            Current difference between observed and modelled water depth. Postive values indicate modelled seabed is deeper than observation. Beta too high
        beta_current : float
            Current beta factor
        beta_all : np.ndarray[np.float64]
            1D Array of all beta factor tested
        Wd_diff_all : np.ndarray[np.float64]
            1D Array of all water depth differences

        Returns
        -------
        beta_found : bool
            True if beta is found
        beta_all : np.ndarray[np.float64]
            1D Array of all beta factor tested
        Wd_diff_all : np.ndarray[np.float64]
            1D Array of all water depth differences
        """
        beta_found = False
        beta_all = np.append(beta_all, beta_current)
        if Wd_diff_all.size > 0:
            if wd_diff >= Wd_diff_all[0]:
                wd_diff +=1
        Wd_diff_all = np.append(Wd_diff_all, wd_diff)
        if wd_diff > 0:
            beta_found = True
        return beta_found, beta_all, Wd_diff_all

    def _check_convergence(self, T_last_step: np.ndarray[np.float64], T_this_step: np.ndarray[np.float64]) -> bool:
        """Check convergence by L2 Norm. Zeinkiewicz et al. Finite Element Method For Fluid Dynamics, 6th Ed. P. 114, Eq. 4.9

        Parameters
        ----------
        T_last_step : np.ndarray[np.float64]
            Temperature from previous discret time step
        T_this_step : np.ndarray[np.float64]
            Temperature from current discret time step

        Returns
        -------
        converged : bool
            True if convergence is lower than self._parameters.convergence
        """
        converged = False
        T_diff = T_this_step - T_last_step
        T_diff = T_diff.astype(np.complex128)
        T_diff = np.sqrt(T_diff)
        convergence = np.sum((T_diff * (T_this_step - T_last_step))) / \
            np.sum(T_this_step)
        if convergence.real < self._parameters.convergence:
            converged = True
        return converged

    @staticmethod
    def _build_crust_lithosphere_properties(coord: np.ndarray[np.float64], base_crust_depth: float, base_lith_depth: float, crust_properties: float, lith_properties: float, asth_properties: float) -> np.ndarray[np.float64]:
        """Build cells properties in crust-lithopsheric mantle-asthenosphere

        Parameters
        ----------
        coord : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere
        hc : float
            Depth to base of crust
        hLith : float
            Depth to base of lithospheric mantle
        crust_properties : float
            Crustal properties
        lith_properties : float
            Lithospheric mantle properties
        asth_properties : float
            Asthenosphere properties

        Returns
        -------
        properties : np.ndarray[np.float64]
            Array of properties. Length is coord.size-1 as properties is at the centre of cells.
        """
        hc_idx = np.argwhere(coord <= base_crust_depth)[-1][0]
        hlith_idx = np.argwhere(coord <= base_lith_depth)[-1][0]
        properties = np.zeros(coord.size-1)
        properties[:hc_idx] = crust_properties
        properties[hc_idx:hlith_idx] = lith_properties
        properties[hlith_idx:] = asth_properties
        return properties

    def _effective_density(self, density: np.ndarray[np.float64], T_arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Effective density due to thermal expansion

        Parameters
        ----------
        density : np.ndarray[np.float64]dt
            Density
        T_arr : np.ndarray[np.float64]
            Temperature

        Returns
        -------
        effective_density : np.ndarray[np.float64]
            Effective density
        """
        T_avg = 0.5*(T_arr[:-1]+T_arr[1:])
        effective_density = density * \
            (1-self._parameters.alphav*(T_avg-self.current_node.T0))
        return effective_density

    def _update_initial_kLith(self) -> None:
        """Update thermal conductivity of lithosphere to keep base of lithosphere at model starts at self.current_node.Tm (Thermal base of lithosphere)
        """
        q_observ = np.full(self.current_node.ncrust, self.current_node.qbase)
        dx_arr = self.current_node.coord_initial[1:] - \
            self.current_node.coord_initial[:-1]
        HP_elem = self.current_node.initial_crustal_HP*dx_arr
        for i in range((self.current_node.ncrust - 2), -1, -1):  # skip first and reverse
            q_observ[i] = q_observ[i + 1] + HP_elem[i]
        T_observ = np.full(self.current_node.ncrust, self.current_node.T0)
        # Vectorize
        k_dx = dx_arr**2/2/self.current_node.kCrust
        k_thermal_elem_arr = dx_arr/self.current_node.kCrust
        for i in range(1, self.current_node.ncrust):  # skip first
            T_observ[i] = (
                T_observ[i - 1] + (q_observ[i - 1] * k_thermal_elem_arr[i]) -
                (self.current_node.initial_crustal_HP[i - 1] * k_dx[i])
            )
        self.current_node.kLith = self.current_node.qbase / (self.current_node.Tm - (
            T_observ[self.current_node.ncrust - 1])) * (self.current_node.hLith - self.current_node.coord_initial[self.current_node.ncrust - 1])
        return

    def _generate_lithosphere_cells(self) -> None:
        """Generate vertical cells in the crust, lithospheric mantle, asthenosphere based on model.parameters.resolution
        """
        coord = np.arange(0, self.current_node._ht,
                          self._parameters.resolution, dtype=np.float64)
        coord = np.append(coord, self.current_node._ht)
        if not (self.current_node.hc in coord):
            idx = np.argwhere(coord < self.current_node.hc)[-1][0]
            coord_temp = np.append(coord[: idx + 1], self.current_node.hc)
            coord = np.append(coord_temp, coord[idx + 1:])
        else:
            pass
        if not (self.current_node.hLith in coord):
            idx = np.argwhere(coord < self.current_node.hLith)[-1][0]
            coord_temp = np.append(coord[: idx + 1], self.current_node.hLith)
            coord = np.append(coord_temp, coord[idx + 1:])
        else:
            pass
        self.current_node.ncrust = np.argwhere(
            coord == self.current_node.hc)[0][0]+1
        self.current_node.coord_initial = coord
        return

    @staticmethod
    def _remesh_crust_lith_asth(coord: np.ndarray[np.float64], key_depths_arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Add key depths (new hc, new hLith) to the vertical 1D cells of crust, lithospheric mantl and asthenosphere

        Parameters
        ----------
        coord : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere
        list_key_depths : np.ndarray[np.float64]
            Key depths to be added to the mesh

        Returns
        -------
        coord_new : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere with key depths
        """
        coord_new = np.copy(coord)
        margin = 10
        for key_depth in key_depths_arr:
            if (key_depth in coord_new):
                pass
            else:
                upper_idx = np.argwhere(coord_new < key_depth)
                if upper_idx.size == 0:
                    raise Exception
                else:
                    upper_idx = upper_idx[-1][0]

                lower_idx = np.argwhere(coord_new > key_depth)
                if lower_idx.size == 0:
                    # raise Exception
                    lower_idx = len(coord_new)-1
                else:
                    lower_idx = lower_idx[0][0]

                upper_diff = key_depth - coord_new[upper_idx]
                lower_diff = coord_new[lower_idx] - key_depth
                differences = np.array([upper_diff, lower_diff])
                if np.any(differences < margin):
                    smallest_idx = np.argmin(differences)
                    differences_idx = np.array([upper_idx, lower_idx])
                    coord_new[differences_idx[smallest_idx]] = key_depth
                else:
                    coord_first_slice = np.append(
                        coord_new[:upper_idx+1], key_depth)
                    coord_new = np.append(
                        coord_first_slice, coord_new[lower_idx:])
        if coord_new[0] != 0:
            coord_new = np.append(0, coord_new)
        return coord_new

    def _advection(self, coord_rift: np.ndarray[np.float64], T_previous: np.ndarray[np.float64], coord_previous: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Calculate new temperature due to advection of asthenosphere

        Parameters
        ----------
        coord_rift : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere after thinning of the crust
        T_previous : np.ndarray[np.float64]
            Temperature before thinning of the crust
        coord_previous : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere before thinning of the crust

        Returns
        -------
        np.ndarray[np.float64]
            Temperature in crust, lithospheric mantle and asthenosphere after accounting for advection
        """
        # todo ulf: this pchip is slow, how about linear interpolation?
        # interpolated_temperature = interpolate.pchip_interpolate(coord_previous, T_previous, coord_rift)
        interpolated_temperature = T_previous
        n = coord_previous.size-1
        # if asthenosphere goes up
        # todo: ulf: replace with argmax
        coord_went_up = np.argwhere(coord_rift > coord_previous[n])
        if coord_went_up.size > 0:
            coord_went_up = coord_went_up[0][0]
            interpolated_temperature[coord_went_up:-1] = T_previous[n] + self._parameters.adiab * \
                (coord_rift[coord_went_up:-1] - coord_previous[n])

        # if surface below 0
        # todo ulf: optimize
        coords_went_above_zero = np.argwhere(coord_rift < coord_previous[0])
        if coords_went_above_zero.size > 0:
            coords_went_above_zero = coords_went_above_zero[-1][0]+1
            interpolated_temperature[:coords_went_above_zero] = 0
        return interpolated_temperature

    def _initial_temperature(self,
                             ) -> None:
        """Calculate steady state temperature at the start of the model
        """
        n_coord = self.current_node.coord_initial.size
        nelem = n_coord-1
        L = np.zeros((n_coord, n_coord))
        R = np.zeros((n_coord, 1))
        self.current_node.kAsth = self.current_node.qbase/self._parameters.adiab
        k_thermal_arr = self._build_crust_lithosphere_properties(
            self.current_node.coord_initial, self.current_node.hc, self.current_node.hLith, self.current_node.kCrust, self.current_node.kLith, self.current_node.kAsth)
        dx_arr = self.current_node.coord_initial[1:] - \
            self.current_node.coord_initial[:-1]
        heat_production_elem = dx_arr*self.current_node.initial_crustal_HP
        # Shape
        N = 0.5 * np.ones((2, 1))
        dNdN = np.array([[1, -1], [-1, 1]])
        for i in range(nelem):
            conductivity = dNdN/dx_arr[i]*k_thermal_arr[i]
            left = conductivity  # +advection_val
            # Internal heating
            Right = N*heat_production_elem[i]
            L[i: i + 2, i: i + 2] = L[i: i + 2, i: i + 2] + left
            R[i: i + 2] = R[i: i + 2] + Right
        R[0] = self.current_node.T0
        L[0, :] = 0
        L[0, 0] = 1
        if self.current_node.bflux is True:
            L[-1, -2] = -1
            L[-1, -1] = 1
            R[-1] = (
                self.current_node.qbase * dx_arr[-1] / k_thermal_arr[-1]
                + self.current_node.initial_crustal_HP[nelem - 1] *
                dx_arr[-1] * dx_arr[-1] / 2 / k_thermal_arr[-1]
            )
        else:
            R[-1] = self.current_node.Tm + self.current_node.adiab * \
                (self.current_node.coord_initial[n_coord -
                 1] - self.current_node.hLith)
            L[-1, :] = 0
            L[-1, -1] = 1
        T = np.linalg.solve(L, R)
        self.current_node.Tinit = T.flatten()
        return

    def _initial_height_of_sealevel(self) -> None:
        """Elevation of the asthenosphere in the abscence of sediments and lithosphere
        """
        dx_arr = self.current_node.coord_initial[1:] - \
            self.current_node.coord_initial[:-1]
        density = self._build_crust_lithosphere_properties(
            self.current_node.coord_initial, self.current_node.hc, self.current_node.hLith, self.current_node.crustsolid, self.current_node.lithsolid, self.current_node.asthsolid)
        density_effective = self._effective_density(
            density, self.current_node.Tinit)
        Wref = np.sum(density_effective*dx_arr)
        self.current_node.H0 = Wref / (self._parameters.rhowater - self.current_node.asthsolid) - self.current_node.asthsolid * \
            self.current_node._ht / \
            (self._parameters.rhowater - self.current_node.asthsolid)
        return 

    def _heat_production(self, coord: np.ndarray[np.float64], top_crust: float, base_crust: float, HP_total: float) -> np.ndarray[np.float64]:
        """Distribute radiogenic heat production at each crust cells based on the total heat production

        Parameters
        ----------
        coord : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere
        top_crust : float
            Top depth of the crust (m)
        base_crust : float
            Base depth of the crust (m)
        HP_total : float
            Total radiogenic heat production in the crust

        Returns
        -------
        heat_product : np.ndarray[np.float64]
            Radiogenic heat production per cell of crust, lithospheric mantle and asthenosphere
        """
        heat_product = np.zeros((coord.size - 1))
        top_idx = np.argwhere(coord <= top_crust)[-1][0]
        base_idx = np.argwhere(coord <= base_crust)[-1][0]
        coord_filtered = coord[top_idx:base_idx+1]
        coord_mid_point = 0.5*(coord_filtered[1:]+coord_filtered[:-1])
        coord_elem_thickness = coord_filtered[1:]-coord_filtered[:-1]
        HP_total = 0 if HP_total < 0 else HP_total
        HP_elem = np.exp(-1*coord_mid_point/self._parameters.HPdcr)
        HP_sum = np.sum(HP_elem*coord_elem_thickness)
        heat_product_crust = (HP_total / HP_sum)*HP_elem
        heat_product[:base_idx] = heat_product_crust
        return heat_product

    def _update_lithosphere_depth(self, T_arr: np.ndarray[np.float64], coord: np.ndarray[np.float64]) -> float:
        """Depth of the base of lithosphere (Thermal LAB) based on new temperature profile

        Parameters
        ----------
        T_arr : np.ndarray[np.float64]
            Temperature profile
        coord : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere

        Returns
        -------
        depth_LAB : float
            Depth to the base of thermal lithosphere
        """
        T_LAB = (1 - self._parameters.tetha) * self.current_node.Tm
        if (T_LAB in T_arr):
            idx = np.argwhere(T_arr == T_LAB)[0][0]
            depth_LAB = coord[idx]
        else:  # no T=Tm, then calculates new LAB between nodes
            y = np.argwhere(T_arr <= T_LAB)
            if y.size == 0:
                y = -1
            else:
                y = y[-1][0]
            z = np.argwhere(T_arr >= T_LAB)
            if z.size == 0:
                z = 0
            else:
                z = z[0][0]
            T1 = T_arr[y]
            T2 = T_arr[z]
            X1 = coord[y]
            X2 = coord[z]
            diff = (X2 - X1) / (T2 - T1)
            depth_LAB = diff * T_LAB - diff * T1 + X1
        return depth_LAB

    def _subsidence(self,
                    density_effective_crust_lith: np.ndarray[np.float64],
                    coord: np.ndarray[np.float64],
                    xsed: np.ndarray[np.float64],
                    density_effective_sed: np.ndarray[np.float64],
                    time_Ma: float,
                    ) -> float:
        """Seabed depth with Airy isostasy and Eustatic Sea level correction

        Parameters
        ----------
        density_effective_crust_lith : np.ndarray[np.float64]
            Effective density of crust, lithospheric mantle and asthenosphere
        coord : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere
        xsed : np.ndarray[np.float64]
            Top and base of sediments
        density_effective_sed : np.ndarray[np.float64]
            Effective density of sediments
        time_Ma : float
            current time step

        Returns
        -------
        seabed
            Depth of seabed at the given time step (m)
        """
        dx_arr = coord[1:]-coord[:-1]
        W1 = np.sum(dx_arr*density_effective_crust_lith)
        W0 = self.current_node.asthsolid * (coord[-1] - coord[0])
        if xsed.size > 1:
            dx_arr = xsed[1:]-xsed[:-1]
            W1 = W1 + np.sum(dx_arr*density_effective_sed)
        W0 += self.current_node.asthsolid * (xsed[-1] - xsed[0])
        # Dynamic topography to be implemented
        # W1 = W1+dynamicTopo.drho(ipoint)*dynamicTopo.h(itime+1)
        a = W0 + (self._parameters.rhowater -
                  self.current_node.asthsolid) * self.current_node.H0
        b = (self._parameters.rhowater -
             self.current_node.asthsolid) * self.current_node.H0
        if W1 == a:  # sealevel and no sed
            seabed = 0
        elif W1 > a:  # sealevel and subsidence
            seabed = ((-1 * W1) + W0 + b) / \
                (self._parameters.rhowater - self.current_node.asthsolid)
        else:  # sealevel and uplift
            seabed = (-1 * (W1 - W0 - b)) / \
                (self._parameters.rhoAir - self.current_node.asthsolid)
        # Eustatic sealevel
        sealevel_ind = np.argwhere(
            self._parameters.eustatic_sea_level["age"] == time_Ma)[0][0]
        eustatic_sealevel = self._parameters.eustatic_sea_level[
            "sea_level_changes"][sealevel_ind]
        seabed = seabed - eustatic_sealevel * self.current_node.asthsolid / \
            (self.current_node.asthsolid - self._parameters.rhowater)
        return seabed

    def _coord_rift_scaler(self, rift_end_time: int, rift_start_time: int, total_beta: float, coord_start_this_rift: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Coord scaler for each time step during rifting

        Parameters
        ----------
        rift_end_time : int
            End of rift in Ma
        rift_start_time : int
            Start of rift in Ma. Must be smaller value than rift_end_time
        total_beta : float
            Beta factor
        coord_start_this_rift : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere at the start of the rift event

        Returns
        -------
        coord_rift_scaler : np.ndarray[np.float64]
            Scaler for top and base of cells to represent thinning of the crust
        """
        rift_duration_s = (rift_end_time - rift_start_time) * \
            self._parameters.myr2s
        time_step_s = self._parameters.time_step_Ma * self._parameters.myr2s
        thinning_factor = (1 / total_beta - 1)
        coord_rift_scaler = (
            thinning_factor
            / rift_duration_s
            * coord_start_this_rift
        ) * time_step_s
        return coord_rift_scaler

    def _distribute_beta_factor(self, coord_scaler: np.ndarray[np.float64], T_new: np.ndarray[np.float64],  coord_before_this_time_step: np.ndarray[np.float64]) -> Tuple[float, np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Evenly distribute stretching through the whole rifting episode.

        Parameters
        ----------
        coord_scaler : np.ndarray[np.float64]
            Scaling factor for a thinning crust
        T_new : np.ndarray[np.float64]
            Temperature profile before rifting
        coord_before_this_time_step : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere before rifting

        Returns
        -------
        hcUpdated : float
            Thickness of the crust after rifting (m)
        T_new : np.ndarray[np.float64]
            Temperature profile after thinning of the crust
        coord_after_rift : np.ndarray[np.float64]
            Top and base of cells of crust, lithospheric mantle and asthenosphere after rifting
        """

        coord_after_rift = coord_before_this_time_step + coord_scaler

        # advection
        T_new = self._advection(coord_after_rift, T_new,
                                coord_before_this_time_step)

        # add moho (base crust)
        hcUpdated:float = coord_after_rift[self.current_node.ncrust - 1]
        return hcUpdated, T_new, coord_after_rift

    def simulate_continental(
        self
    ):
        """Run forward model for single or multi rift
        """
        coord_init = self.current_node.coord_initial
        seabed = self.current_node.sediments.iloc[0]["top"]
        time_start = self._parameters.time_start
        time_end = self._parameters.time_end

        density_arr = self._build_crust_lithosphere_properties(
            coord_init, self.current_node.hc, self.current_node.hLith, self.current_node.crustsolid, self.current_node.lithsolid, self.current_node.asthsolid)
        density_effective_initial_arr = self._effective_density(
            density_arr, self.current_node.Tinit)

        Wdepth_start = self._subsidence(
            density_effective_initial_arr,
            coord_init,
            np.zeros(1),
            np.zeros(1),
            time_start,
        )  # initial WDepth
        Tsed = np.zeros(1)
        xsed = np.zeros(1)
        idsed = np.empty(0, np.int32)
        HPsed = np.empty(0)
        nrift = self.current_node.rift.shape[0]
        self.current_node.beta = np.empty(0)
        total_HP_model_start = self.current_node.shf-self.current_node.qbase
        if nrift > 1:  # multi-rift. More than 1 nan to fill
            T_last = self.current_node.Tinit
            coord_last = coord_init
            hLith_last = self.current_node.hLith
            total_crustal_HP_rift_start = total_HP_model_start
            for irift in range(nrift):  # start from the oldest rift and loop all rift
                if irift + 1 < nrift:  # if not the last rift
                    # PWD before the next rift
                    WdepthEnd = self.current_node.paleoWD[irift]
                    start_time = self.current_node.rift[irift, 0]
                    # run until next rift
                    end_time = self.current_node.rift[irift + 1, 0]
                else:  # for the last rift event
                    WdepthEnd = seabed  # present-day WD
                    start_time = self.current_node.rift[irift, 0]
                if irift == 0:  # first rift
                    start_time = time_start
                if irift + 1 == nrift:  # last rift
                    end_time = time_end
                beta, total_crustal_HP_rift_start, T_last, coord_last, xsed, Tsed, HPsed, idsed, hLith_last,  temperature, depth_out,layer_ids_one_rift ,num_elements= self.simulate_one_rift_event(
                    start_time,
                    end_time,
                    self.current_node.rift[irift, 0],
                    self.current_node.rift[irift, 1],
                    WdepthEnd,
                    coord_last,
                    T_last,
                    xsed,
                    Tsed,
                    HPsed,
                    idsed,
                    hLith_last,
                    total_crustal_HP_rift_start
                )
                # oldest last
                if irift == 0:
                    self.current_node._depth_out = np.zeros((num_elements+1, time_start+1))
                    self.current_node.temperature_out = np.zeros((num_elements+1, time_start+1))
                    self.current_node.temperature_out.fill(np.nan)
                    self.current_node._idsed = np.zeros((num_elements, time_start+1),dtype=np.int32)
                    self.current_node._idsed.fill(-9999)
                self.current_node.beta = np.append(self.current_node.beta, beta)
                if temperature.shape[0]!=self.current_node.temperature_out.shape[0]:
                    self.current_node.temperature_out,temperature=self._equalise_array_shape(self.current_node.temperature_out,temperature,np.nan)
                self.current_node.temperature_out[:, end_time:start_time] = temperature[:,
                                                                      end_time:start_time]
                if depth_out.shape[0]!=self.current_node._depth_out.shape[0]:
                    self.current_node._depth_out,depth_out=self._equalise_array_shape(self.current_node._depth_out,depth_out,np.nan)
                self.current_node._depth_out[:, end_time:start_time] = depth_out[:,
                                                                  end_time:start_time]
                if layer_ids_one_rift.shape[0]!=self.current_node._idsed.shape[0]:
                    self.current_node._idsed,layer_ids_one_rift=self._equalise_array_shape(self.current_node._idsed,layer_ids_one_rift,-9999)
                self.current_node._idsed[:, end_time:start_time] = layer_ids_one_rift[:,
                                                              end_time:start_time]
                num_elements = self.current_node._depth_out.shape[0]
        else:  # one rift
            if nrift == 1:  # make sure beta is not placeholder
                beta, _a_, _hp_, _b_, _c_, Tsed, _e_, idsed, _g_,  self.current_node.temperature_out, self.current_node._depth_out,self.current_node._idsed,num_elements = self.simulate_one_rift_event(
                    self._parameters.time_start,
                    self._parameters.time_end,
                    self.current_node.rift[0, 0],
                    self.current_node.rift[0, 1],
                    seabed,
                    coord_init,
                    self.current_node.Tinit,
                    xsed,
                    Tsed,
                    HPsed,
                    idsed,
                    self.current_node.hLith,
                    total_HP_model_start
                )
                self.current_node.beta = np.append(self.current_node.beta, beta)
        # append initial temperature to output data
        initial_depth = np.linspace(0.0, self.current_node._ht, self.current_node._depth_out.shape[0])
        idx_initial_seabed = np.abs(initial_depth - Wdepth_start).argmin()
        initial_depth[idx_initial_seabed] = Wdepth_start
        self.current_node.temperature_out[idx_initial_seabed:, -1] = np.interp(
            initial_depth[idx_initial_seabed:], self.current_node.coord_initial+Wdepth_start, self.current_node.Tinit)
        base_crust = Wdepth_start+coord_init[(self.current_node.ncrust - 1)]
        idx_base_crust = np.abs(initial_depth - base_crust).argmin()

        base_lith = idx_initial_seabed+self.current_node.hLith
        idx_base_lith = np.abs(initial_depth - base_lith).argmin()
        self.current_node._idsed[idx_initial_seabed:idx_base_crust, -
                 1] = -1
        self.current_node._idsed[idx_base_crust:idx_base_lith, -1] = -2
        self.current_node._idsed[idx_base_lith:, -1] = -3
        self.current_node._depth_out[:, -1] = initial_depth
        return

    @staticmethod
    def _phi1(x1: np.ndarray[np.float64], x2: np.ndarray[np.float64], exponent: np.ndarray[np.float64])->np.ndarray[np.float64]:
        """calculated porosity exponent in a way that is accurate at x2 == x1

        Parameters
        ----------
        x1 : np.ndarray[np.float64]
            base of sediment (km)
        x2 : np.ndarray[np.float64]
            top of sediment (km)
        exponent : np.ndarray[np.float64]
            Exponential decay of porosity

        Returns
        -------
        exp : np.ndarray[np.float64]
            Exponent of porosity
        """
        d = x2 - x1
        d[d==0] = 1e-16
        return np.exp(exponent*x1)*np.expm1(exponent*d)/d
        

    def _sediments_mean_porosity(self, sed_coord: np.ndarray[np.float64],  sed_id: np.ndarray[np.float64]) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Effective porosity of sediments at depth

        Parameters
        ----------
        sed_coord : np.ndarray[np.float64]
            Depth of sediment column
        sed_id : np.ndarray[np.float64]
            Ids of sediment properties

        Returns
        -------
        mean_porosity : np.ndarray[np.float64]
            Effective porosity of sediment
        sed_ids : np.ndarray[np.float64]
            Ids of sediment
        """
        sed_phi0 = self.current_node.sediments["phi"].values
        sed_decay = self.current_node.sediments["decay"].values
        top_km_arr = sed_coord[:-1]/1e3
        base_km_arr = sed_coord[1:]/1e3
        n_sed_elem = top_km_arr.size
        mean_porosity = np.zeros(n_sed_elem)
        sed_id = sed_id[np.arange(n_sed_elem)]
        mean_porosity = -sed_phi0[sed_id] / sed_decay[sed_id] * self._phi1(base_km_arr, top_km_arr, -sed_decay[sed_id])
        return mean_porosity, sed_id


    def _compact_many_layers(self, seddep:np.ndarray[np.float64], phi0:np.ndarray[np.float64], decay:np.ndarray[np.float64], niter=12, maximum_burial_depth=None):
        """Run the compation iteration for all layers at once. This converges
        almost as fast as solving each layer thickness top-down, but runs much faster
        because of vector operations. Since we vectorize we have to run the
        same number of iterations for all layers.
        Note: Layer 0 is the top layer here! Maximum burial depth is the depth of the bottom of the layers.
        """ 
        layer_thickness = seddep
        for i in range(niter):
            # np.cumsum(layer_thickness) is the depth of the base of each layer.
            layer_base_depths = np.cumsum(layer_thickness)
            if maximum_burial_depth is not None:
                # Here we use that if one layer is above its maximum burial depth
                # then all lower layers are as well.
                layer_base_depths = np.maximum(layer_base_depths, maximum_burial_depth)
            x = np.maximum(1e-14, decay*layer_thickness)
            phiavg = phi0*np.exp(-decay*layer_base_depths)*np.expm1(x)/x
            layer_thickness = seddep/(1-phiavg)
        return layer_thickness
    
    def _sedimentation(self, reference_implementation=False):
        """Calculated sediment top and base and sedimentation rate through time
        """
        sed_pack = self.current_node.sediments
        ntime = self._parameters.time_start - self._parameters.time_end + 1
        itime = np.arange(self._parameters.time_start)
        # itime = itime[::-1]  # oldest time first # TODO: Verify that we can iterate forwards in time (because we want to track maximum burial in this way)
        if sed_pack.baseage.max() > self._parameters.time_start:
            # logger.warning(f"Some sediments are older than start of the model. Ignoring sediments older than {time_start}")
            sed_pack = sed_pack[sed_pack.baseage <=
                                self._parameters.time_start]
        sed_lay = sed_pack.shape[0]
        sed = np.zeros((sed_lay, 2, ntime))
        # 2D array col = sed_pack, row = age/time
        sedrate = np.zeros((ntime, sed_lay))
        if not reference_implementation: # Vectorized compaction including erosion.
            # Dataframe access by name in the inner loop is very slow, we extract
            # the relevant data to numpy arrays to avoid this.
            baseage = sed_pack['baseage'][:sed_lay].to_numpy()
            topage = sed_pack['topage'][:sed_lay].to_numpy()
            grain_thickness = sed_pack['grain_thickness'][:sed_lay].to_numpy()
            have_erosion = np.any(grain_thickness<0)
            phi = sed_pack['phi'][:sed_lay].to_numpy()
            decay = sed_pack['decay'][:sed_lay].to_numpy()
            # Vectorized inner loop (over layers)
            for i in itime:
                seddep = np.zeros(sed_lay)
                # Assume that layer are stored in increasing age order
                # We have a number of already deposited layers and
                # maybe one layer which is in the middle of deposition.
                # Already deposited means i < topage
                # Started means i > baseage

                deposited_start = np.argmax(i < topage)
                if topage[deposited_start] >= i:
                    seddep[deposited_start:] = grain_thickness[deposited_start:]
                else:
                    deposited_start = 10000000
                # Active layer
                active_index = np.argmax(i < baseage)
                if baseage[active_index] > i and active_index < deposited_start:
                    duration = np.abs(baseage[active_index] - topage[active_index])
                    if duration > 0:
                        sedrate[i,active_index] = grain_thickness[active_index]/duration   
                    seddep[active_index] = sedrate[i,active_index]*(baseage[active_index] - i)
                # Erode (if negative values in seddep)
                if have_erosion:
                    for j in range(active_index,0,-1):
                        if seddep[j] < 0:
                            seddep[j-1] += seddep[j]
                            seddep[j] = 0
                    if seddep[0] < 0:
                        seddep[0] = 0
                    if i == itime[0]:
                        maximum_burial_depth = np.zeros_like(seddep)
                    else:
                        maximum_burial_depth = np.maximum(maximum_burial_depth, sed[active_index:, 1, i-1])
                else:
                    maximum_burial_depth = None

                # Compact
                if baseage[active_index] > i: # We have at least one layer
                    layer_thickness = self._compact_many_layers(seddep[active_index:],
                                                                phi[active_index:],
                                                                decay[active_index:],
                                                                maximum_burial_depth=maximum_burial_depth[active_index:] if maximum_burial_depth is not None else None)
                    z = np.cumsum(layer_thickness)
                    sed[active_index+1:, 0, i] = z[:-1] # top depth
                    sed[active_index:, 1, i] = z # bottom depth
        # for rows in cont_pts: #row is 1d locations
        else: # if reference_implementation: Note: does not support erosion!
            for i in itime:  # time is reversed
                for x in range(sed_lay):  # skip basement
                    sed_layer_properties = sed_pack.iloc[x]
                    # Move seddep calculation to outside the compaction loop
                    if i < sed_layer_properties.baseage and i >= sed_layer_properties.topage: # depositing, slice sedimentary package
                        deposition_duration = abs(
                            sed_layer_properties.baseage - sed_layer_properties.topage
                        )
                        if deposition_duration == 0:
                            sedrate[i, x] = 0
                            seddep = 0
                        else:
                            sedrate[i, x] = sed_layer_properties.grain_thickness / \
                                deposition_duration
                            seddep = sedrate[i, x] * \
                                (sed_layer_properties.baseage - i)
                    elif i < sed_layer_properties.topage:  # already deposited
                        seddep = sed_layer_properties.grain_thickness
                    else:  # future sediments
                        seddep = 0
                    if seddep > 0:  # set sed
                        y1 = sed[x, 0, i]  # top
                        check = True
                        y2 = seddep
                        while check == True:
                            if sed_layer_properties.phi > 0:
                                average_porosity = (
                                    sed_layer_properties.phi
                                    / sed_layer_properties.decay
                                    / (y2 - y1)
                                    * (
                                        math.exp(
                                            (-1 * sed_layer_properties.decay * y1))
                                        - math.exp((-1 * sed_layer_properties.decay * y2))
                                    )
                                )
                            else:
                                average_porosity = sed_layer_properties.phi
                            y2_new = y1 + seddep / (1 - average_porosity)
                            if abs((y2 - y2_new)) <= 1e-6:
                                check = False
                                sed[x, 1, i] = y2
                                if x + 1 < sed_lay:
                                    sed[x + 1, 0, i] = y2
                            y2 = y2_new
                    else:
                        sed[x, 1, i] = sed[x, 0, i]  # base = top
                        if x + 1 < sed_lay:
                            sed[x + 1, 0, i] = sed[x, 0, i]
        self.current_node.sed = sed * 1000
        self.current_node.sedrate = sedrate * 1000
        return
    
    @staticmethod
    def compaction(top_m: float, base_m: float, phi0: float, phi_decay: float, base_maximum_burial_depth_m: float = 0) -> float:
        """Compact sediment at depth

        Parameters
        ----------
        top_m : float
            Top of sediment (m)
        base_m : float
            Base of sediment (m)
        phi0 : float
            Porosity at surface (fraction)
        phi_decay : float
            Exponential decay of porosity with depth (fraction)
        base_maximum_burial_depth_m : float, optional
            Maximum burial depth of sediment (To make sure compaction is irreversable), by default 0

        Returns
        -------
        base_result : float
            Depth of sediment base (m)
        """
        if base_m == 0:
            return top_m
        top_km = top_m / 1000
        base_km = base_m / 1000
        base_new = max(base_km, base_maximum_burial_depth_m/1000)
        while True:
            if phi0 > 0:
                phiav = (
                    phi0
                    / phi_decay
                    / (base_new - top_km)
                    * (math.exp((-1 * phi_decay * top_km)) - math.exp((-1 * phi_decay * base_new)))
                )
            else:
                phiav = phi0
            y2_new = top_km + base_km / (1 - phiav)  # add porosity
            y2_new = max(y2_new, base_maximum_burial_depth_m/1000)
            if (
                abs(base_new - y2_new) <= 1e-6
            ):  # loop until there is little compaction at deeper sediments
                base_result = base_new * 1000
                break
            base_new = y2_new
        return base_result

    @staticmethod
    def decompaction(top_m: float, base_m: float, phi0: float, phi_decay: float) -> float:
        """Thickness of sediment without pore space. For calculating deposition rate

        Parameters
        ----------
        top_m : float
            Top of sediment (m)
        base_m : float
            Base of sediment (m)
        phi0 : float
            Porosity at surface (fraction)
        phi_decay : float
            Exponential decay of porosity with depth (fraction)

        Returns
        -------
        grain_thickness: float
            Thickness of sediment without pore space (m)
        """
        if top_m == base_m:
            return 0
        top_km = top_m / 1000
        base_km = base_m / 1000
        phi_avg = (
            phi0
            / phi_decay
            / (base_km - top_km)
            * (math.exp(-phi_decay * top_km) - math.exp(-phi_decay * base_km))
        )
        grain_thickness = (base_km - top_km) * (1 - phi_avg)
        return grain_thickness * 1000 

    def add_sediments(
        self,
        sedrate:np.ndarray[np.float64],
        sed:np.ndarray[np.float64],
        xsed_old:np.ndarray[np.float64],
        Tsed_old:np.ndarray[np.float64],
        HPsed_old:np.ndarray[np.float64],
        idsed_old:np.ndarray[np.int32],

    ) ->Tuple[bool,np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.int32]]:
        """Take care of sedimentation at this time step

        Parameters
        ----------
        sedrate : np.ndarray[np.float64]
            Sedimentation rate at current time steps (m/Ma)
        sed : np.ndarray[np.float64]
            Top and base of all sediment unit at current time step referenced to seabed at 0 m (m)
        xsed_old : np.ndarray[np.float64]
            Top and base of sedimentary column at current time step referenced to seabed at 0 m (m)
        Tsed_old : np.ndarray[np.float64]
            Temperature of sediments at xsed (C)
        HPsed_old : np.ndarray[np.float64]
            Radiogenic heat production between xsed (W/m3)
        idsed_old : np.ndarray[np.int32]
            Sediment ids between xsed

        Returns
        -------
        sedflag : bool
            True if sedimentation exists at current time step
        xsed : np.ndarray[np.float64]
            Top and base of sedimentary column at current time step referenced to seabed at 0 m (m)
        Tsed : np.ndarray[np.float64]
            Temperature of sediments at xsed (C)
        HPsed : np.ndarray[np.float64]
            Radiogenic heat production between xsed (W/m3)
        idsed : np.ndarray[np.int32]
            Sediment ids between xsed
        """

        sedflag = True
        if np.sum(sedrate) > 0:  # new sediment exist
            xsed, Tsed, HPsed, idsed = self._combine_new_old_sediments(
                sedrate,
                sed,
                xsed_old,
                Tsed_old,
                HPsed_old,
                idsed_old,

            )
        else:
            if xsed_old[-1] == 0:  # no sed for all time
                sedflag = False
                xsed = np.zeros(1)
                HPsed = np.zeros(1)
                idsed = np.empty(0, dtype=np.int32)
                Tsed = np.zeros(1)
            else:  # no sed only this time. passthrough
                xsed = xsed_old
                Tsed = Tsed_old
                HPsed = HPsed_old
                idsed = idsed_old
        return sedflag, xsed, Tsed, HPsed, idsed

    def _combine_new_old_sediments(
        self,
        sedrate:np.ndarray[np.float64],
        sed:np.ndarray[np.float64],
        xsed_old:np.ndarray[np.float64],
        Tsed_old:np.ndarray[np.float64],
        HPsed_old:np.ndarray[np.float64],
        idsed_old:np.ndarray[np.int32],
    ) ->Tuple[np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.int32]]:
        """Add new sediment for current time step and combine with previous sediments

        Parameters
        ----------
        sedrate : np.ndarray[np.float64]
            Sedimentation rate at current time steps (m/Ma)
        sed : np.ndarray[np.float64]
            Top and base of all sediment unit at current time step referenced to seabed at 0 m (m)
        xsed_old : np.ndarray[np.float64]
            Top and base of sedimentary column at current time step referenced to seabed at 0 m (m)
        Tsed_old : np.ndarray[np.float64]
            Temperature of sediments at xsed (C)
        HPsed_old : np.ndarray[np.float64]
            Radiogenic heat production between xsed (W/m3)
        idsed_old : np.ndarray[np.int32]
            Sediment ids between xsed

        Returns
        -------
        xsed : np.ndarray[np.float64]
            Top and base of sedimentary column at current time step referenced to seabed at 0 m (m)
        Tsed : np.ndarray[np.float64]
            Temperature of sediments at xsed (C)
        HPsed : np.ndarray[np.float64]
            Radiogenic heat production between xsed (W/m3)
        idsed : np.ndarray[np.int32]
            Sediment ids between xsed
        """
        # Get new sediments at this time step
        xsed_new, idsed_new = self._get_new_sediments(
            sedrate, self.current_node.sediments["phi"].values, self.current_node.sediments["decay"].values)
        HPsed_new = self.current_node.sediments["rhp"].values[idsed_new]

        # decompact and recompat old_sed for new depth
        if xsed_old[-1] > 0:  # old sediments exist
            new_sed_base = xsed_new[-1]
            xsed_old_recompacted = self._recompact_old_sediments(
                new_sed_base, xsed_old,  idsed_old, self.current_node.sediments[
                    "phi"].values, self.current_node.sediments["decay"].values
            )
            xsed = np.append(xsed_new, xsed_old_recompacted)
            idsed = np.append(idsed_new, idsed_old)
            HPsed = np.append(HPsed_new, HPsed_old)
        else:
            xsed = xsed_new
            HPsed = HPsed_new
            idsed = idsed_new
        Tsed = np.zeros(idsed_new.size)
        Tsed = np.append(Tsed, Tsed_old)

        # refine sediments mesh
        if xsed.size > 3:
            xsed_remeshed, idsed, HPsed = self._remesh_sediments(
                xsed, idsed, HPsed, self.current_node.sediments["rhp"].values, sed)
            Tsed = np.interp(xsed_remeshed, xsed, Tsed)
            xsed = xsed_remeshed

        return xsed, Tsed, HPsed, idsed
    
    def _recompact_old_sediments(
        self, xsed_new_base: np.ndarray[np.float64], xsed_old: np.ndarray[np.float64],  idsed_old: np.ndarray[np.int32], sed_phi0: np.ndarray[np.float64], sed_decay: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Calculate depth of previous sedimentary column under the weight of new sediments

        Parameters
        ----------
        xsed_new_base : np.ndarray[np.float64]
            Top and base of new sedimentary column (m)
        xsed_old : np.ndarray[np.float64]
            Top and base of previous sedimentary column (m)
        idsed_old : np.ndarray[np.int32]
            Sediment ids between previous xsed
        sed_phi0 : np.ndarray[np.float64]
            Surface porosity of sediments (fraction)
        sed_decay : np.ndarray[np.float64]
            Exponential decay of sediment porosity with depth (fraction)

        Returns
        -------
        np.ndarray[np.float64]
            _description_
        """

        xsed_old_recompacted = np.zeros(xsed_old.size - 1)
        previous_base = xsed_new_base
        for i in range(xsed_old.size - 1):
            sed_idx = idsed_old[i]
            phi0 = sed_phi0[sed_idx]
            maximum_burial = 0
            shift = np.maximum(0,maximum_burial-xsed_old[i+1])
            grain_thickness_m = self.decompaction( # Fix for maximum burial
                xsed_old[i]+shift, xsed_old[i + 1]+shift, phi0, sed_decay[sed_idx])
            xsed_old_recompacted[i] = self.compaction(
                previous_base, grain_thickness_m, phi0, sed_decay[sed_idx], base_maximum_burial_depth_m=maximum_burial)
            previous_base = xsed_old_recompacted[i]
        return xsed_old_recompacted

    def _get_new_sediments(self, sedrate: np.ndarray[np.float64], sed_phi0: np.ndarray[np.float64], sed_decay: np.ndarray[np.float64]) -> tuple[np.ndarray[np.float64], np.ndarray[np.int32]]:
        """Generate new sediments that was deposited at this time step

        Parameters
        ----------
        sedrate : np.ndarray[np.float64]
            Sedimentation rate at current time steps (m/Ma)
        sed_phi0 : np.ndarray[np.float64]
            Surface porosity of sediments (fraction)
        sed_decay : np.ndarray[np.float64]
            Exponential decay of sediment porosity with depth (fraction)

        Returns
        -------
        xsed_new : np.ndarray[np.float64]
            Top and base of new sedimentary column (m)
        idsed_new : np.ndarray[np.float64]
            Sediment ids of new sedimentary column
        """
        xsed_new = np.zeros(1)  # keep track of sediment depth
        idsed_new = np.empty(0, dtype=np.int32)
        for i in range(sedrate.size):
            if (
                sedrate[i] > 0
            ):  # sedrate filtered to time in input. sedrate is now 1d array
                new_sediment_depth = self.compaction(
                    xsed_new[-1], sedrate[i], sed_phi0[i], sed_decay[i])
                xsed_new = np.append(xsed_new, new_sediment_depth)
                idsed_new = np.append(idsed_new, i)
        return xsed_new, idsed_new

    #TODO: ulf speedup
    def _remesh_sediments(self, xsed: np.ndarray[np.float64], idsed: np.ndarray[np.int32], HPsed: np.ndarray[np.float64], sed_rhp: np.ndarray[np.float64], sed: np.ndarray[np.float64]) -> tuple[np.ndarray[np.float64], np.ndarray[np.int32], np.ndarray[np.float64]]:
        """Remesh sedimentary column to maintain resolution

        Parameters
        ----------
        xsed : np.ndarray[np.float64]
            Top and base of sedimentary column at current time step referenced to seabed at 0 m (m)
        idsed : np.ndarray[np.int32]
            Sediment ids between xsed
        HPsed : np.ndarray[np.float64]
            Radiogenic heat production between xsed (W/m3)
        sed_rhp : np.ndarray[np.float64]
            Radiogenic heat production for all sediments (W/m3)
        sed : np.ndarray[np.float64]
            Top and base of all sediment unit at current time step referenced to seabed at 0 m (m)

        Returns
        -------
        xsed_remeshed : np.ndarray[np.float64]
            Top and base of sedimentary column at current time step referenced to seabed at 0 m (m)
        idsed_remeshed : np.ndarray[np.int32]
            Sediment ids between xsed
        HPsed_remeshed : np.ndarray[np.float64]
            Radiogenic heat production between xsed (W/m3)
        """
        xsed_remeshed = xsed[:2]
        idsed_remeshed = np.array([idsed[0]], np.int32)
        HPsed_remeshed = np.array([HPsed[0]])
        for i in range(sed.shape[0]):
            if abs((xsed_remeshed[-1] - sed[i, 1])) > 1 and sed[i, 1] > 0:
                n_new_nodes = math.floor(
                    (sed[i, 1] - xsed_remeshed[-1]) / self._parameters.vertical_resolution_sediments)
                x_interp = np.linspace(
                    xsed_remeshed[-1], sed[i, 1], n_new_nodes + 2)
                xsed_remeshed = np.append(xsed_remeshed, x_interp[1:]) # todo: ulf: append is slow, build list and concatenate at the end
                id_temp = i * np.ones(n_new_nodes + 1, np.int32)
                idsed_remeshed = np.append(idsed_remeshed, id_temp)
                HPsed_remeshed = np.append(
                    HPsed_remeshed, (sed_rhp[i] * np.ones(n_new_nodes + 1)))
        return xsed_remeshed, idsed_remeshed, HPsed_remeshed

    def simulate_one_rift_event(
        self,
        time_start:int,
        time_end:int,
        rift_start_time:int,
        rift_end_time:int,
        observed_seabed:float,
        coord_start_this_rift:np.ndarray[np.float64],
        T_init:np.ndarray[np.float64],
        xsed_first:np.ndarray[np.float64],
        Tsed_first:np.ndarray[np.float64],
        HPsed_first:np.ndarray[np.float64],
        idsed_first:np.ndarray[np.int32],
        hLith:float,
        total_crustal_HP_time_start:float
    ):
        all_tested_beta = np.empty(0)
        all_water_depth_difference = np.empty(0)
        beta = self._parameters.starting_beta
        save_results = beta_found = False
        n_depth_out = 2000 # Not used. Will be overridden in last time step. Keep here for linter
        # Start searching beta factor
        while True:
            if save_results == True:
                # Setup result holder
                num = (math.floor(self.current_node._ht /
                        self._parameters.vertical_resolution_sediments)) + 1
                num=n_depth_out
                depth_out_all = np.zeros((num, time_start+1))
                temperature_out = np.zeros((num, time_start+1))
                temperature_out.fill(np.nan)
                idsed_out = np.zeros((num-1, time_start+1),dtype=np.int32)
                idsed_out.fill(-9999)
            # initial condition for new beta trial
            xsed = xsed_first
            Tsed = Tsed_first
            HPsed = HPsed_first
            idsed = idsed_first
            T_new = T_init
            coord_before_this_time_step = coord_current = coord_start_this_rift
            hcUpdated = hc_start_this_rift = coord_current[(
                self.current_node.ncrust - 1)]
            total_crustal_HP_current = total_crustal_HP_time_start
            coord_rift_scaler = self._coord_rift_scaler(
                rift_end_time, rift_start_time, beta, coord_start_this_rift)

            # Start from oldest time
            for i in range(time_start - 1, time_end - 1, self._parameters.time_step_Ma):

                # vel = np.zeros(coord_first.size)
                if i < rift_start_time and i >= rift_end_time:
                    #dtrift_distributed = rift_end_time - rift_start_time
                    hcUpdated, T_new, coord_current = self._distribute_beta_factor(
                        coord_rift_scaler, T_new, coord_before_this_time_step)
                    coord_before_this_time_step = coord_current.copy()
                    total_crustal_HP_current = total_crustal_HP_time_start / (
                        hc_start_this_rift / hcUpdated
                    )
                # new LAB
                lithUpdated = self._update_lithosphere_depth(
                    T_new, coord_start_this_rift)
     
                # remesh to add new hc and hlith

                coord_current = self._remesh_crust_lith_asth(
                    coord_start_this_rift, np.array([hcUpdated, lithUpdated]))

                # heat production crust
                HP_new = self._heat_production(
                    coord_current, 0.0, hcUpdated, total_crustal_HP_current)

                # Interpolate temperature to current coord
                T_new = np.interp(coord_current, coord_start_this_rift, T_new)

                # TODO: underplate, asthenospheric anamaly, melt for all modes
                # if (self.current_node.X==12150) and (self.current_node.Y==12000):
                #     print("Lith", i, hLith, lithUpdated)

                # Take care of sedimentation
                sedflag, xsed, Tsed, HPsed, idsed = self.add_sediments(
                    self.current_node.sedrate[i, :],
                    self.current_node.sed[:, :, i],
                    xsed,
                    Tsed,
                    HPsed,
                    idsed,
                )
                (
                    T_newtemp,
                    densityeff_crust_lith,
                    Tsed,
                    densityeff_sed

                    # Solve the heat equation
                ) = self.calculate_new_temperature(
                    sedflag,
                    coord_current,
                    T_new,
                    HP_new,
                    xsed,
                    hcUpdated,
                    # hLith,
                    lithUpdated,
                    Tsed,
                    HPsed,
                    idsed,
                )

                # Interpolate temperature back to original coord
                T_new = np.interp(coord_start_this_rift,
                                  coord_current, T_newtemp)

                lithUpdated = self._update_lithosphere_depth(
                    T_new, coord_start_this_rift)

                # Sanity check for maximum lithosphere depth
                # if self._parameters.maxContLithFlag:
                #     if hLith > self._parameters.maxContLith:
                #         hLith = self._parameters.maxContLith
                #     if lithUpdated > self._parameters.maxContLith:
                #         lithUpdated = self._parameters.maxContLith

                # New water depth after running this time step
                densityeff_crust_lith = np.interp(
                    ((coord_start_this_rift[:-1] +
                     coord_start_this_rift[1:]) / 2),
                    ((coord_current[:-1] + coord_current[1:]) / 2),
                    densityeff_crust_lith,
                )
                modelled_seabed = self._subsidence(
                    densityeff_crust_lith,
                    coord_start_this_rift,
                    xsed,
                    densityeff_sed,
                    i,
                )
                if i == time_end:
                    n_depth_out = xsed.size+coord_current.size+4

                # Save result to holder
                if save_results:
                    if sedflag == True:
                        Tout = np.append(Tsed, T_newtemp[1:])
                        coord_T = np.append(xsed, coord_current[1:] + xsed[-1])
                    else:  # no sed
                        Tout = T_newtemp
                        coord_T = coord_current
    
                    seabed = modelled_seabed
                    coord_seabed = coord_T+seabed
                    depth_out = np.linspace(0.0, self.current_node._ht, num)
                    depth_out[1:coord_seabed.size+1]=coord_seabed
                    depth_out[coord_seabed.size+1:]= np.linspace(coord_seabed[-1]+100,self.current_node._ht,depth_out[coord_seabed.size+1:].size)
                    idx_seabed = np.abs(depth_out - seabed).argmin()        
                    if xsed.size > 1:
                        base_crust = hcUpdated+xsed[-1]+seabed
                        base_lith = lithUpdated+xsed[-1]+seabed
                        idx_lith = np.abs(depth_out - base_lith).argmin()
                        idx_base_crust = np.abs(depth_out - base_crust).argmin()
                        xsed_with_seabed = xsed+seabed
                        base_sed = xsed_with_seabed[-1]
                        idx_base_sed = np.abs(depth_out - base_sed).argmin()
                        depth_out_mid_point = (depth_out[1:] + depth_out[:-1]) / 2

                        idsed_out[idx_seabed:idx_base_sed, i] = np.interp(
                            depth_out_mid_point[idx_seabed:idx_base_sed], xsed_with_seabed[:-1], idsed)
                        idsed_out[idx_base_sed:idx_base_crust,
                                 i] = -1
                    else:
                        base_crust = hcUpdated+seabed
                        base_lith = lithUpdated+seabed
                        idx_lith = np.abs(depth_out - base_lith).argmin()
                        idx_base_crust = np.abs(depth_out - base_crust).argmin()
                        idsed_out[idx_seabed:idx_base_crust,
                                 i] = -1
                    depth_out[idx_lith] = base_lith
                    idsed_out[idx_base_crust:idx_lith, i] = -2
                    idsed_out[idx_lith:, i] = -3
                    depth_out_all[:, i] = depth_out
                    temperature_out[idx_seabed:, i] = np.interp(
                        depth_out[idx_seabed:], coord_seabed, Tout)
                    temperature_out[idx_lith:i] = (
                        1 - self._parameters.tetha) * self.current_node.Tm

            if save_results and beta_found:  # Results saved. Exit loop
                self.current_node.water_depth_difference = modelled_seabed-observed_seabed
                self.current_node.total_beta_tested = all_tested_beta.size+1
                break
            # Check condition
            beta_found, all_tested_beta, all_water_depth_difference = self._check_beta(modelled_seabed-observed_seabed,
                                                                                       beta, all_tested_beta, all_water_depth_difference)
            if beta_found:  # Start saving results
                beta = self._approximate_true_beta(
                    all_water_depth_difference, all_tested_beta)
                save_results = True
            else:  # Not fitting. Try with new beta
                beta = self._get_new_beta(
                    all_tested_beta, all_water_depth_difference, beta)
                if beta >= self._parameters.max_beta:
                    beta = self._parameters.max_beta
                    beta_found = save_results = True
        
        return beta, total_crustal_HP_current, T_new, coord_before_this_time_step, xsed, Tsed, HPsed, idsed, lithUpdated, temperature_out, depth_out_all,idsed_out,n_depth_out



    def _approximate_true_beta(self, Wd_diff_all: np.ndarray[np.float64], beta_all: np.ndarray[np.float64]) -> float:
        """Interpolate the best beta factor that fit the observed subsidence

        Parameters
        ----------
        Wd_diff_all : np.ndarray[np.float64]
            (modelled seabed depth - observed seabed depth) for all tested beta factor
        beta_all : np.ndarray[np.float64]
            All beta factors that have been tested

        Returns
        -------
        beta : float
            Interpolated beta factor result
        """
        if Wd_diff_all.size == 1:
            beta = 1
        else:
            beta = interpolate.pchip_interpolate(
                Wd_diff_all, beta_all, 0)
            beta = float(beta)
        return round(beta, 2)

    def _get_new_beta(self, beta_list: np.ndarray[np.float64], wd_diff_list: np.ndarray[np.float64], old_beta: float) -> float:
        """Find a new beta factor for new trial

        Parameters
        ----------
        beta_list : np.ndarray[np.float64]
            All beta factors that have been tested
        wd_diff_list : np.ndarray[np.float64]
            (modelled seabed depth - observed seabed depth) for all tested beta factor
        old_beta : float
            Tested beta factor from this run

        Returns
        -------
        beta : float
            New beta factor for next trial
        """
        if self._parameters.experimental:
            if beta_list.size >= 2:
                beta = interpolate.pchip_interpolate(
                    wd_diff_list, beta_list, 100)
                beta = float(beta)
                if beta <= beta_list[-1]:  # Sometimes pchip will return negative number
                    beta = beta_list[-1] * 1.2
            else:
                beta = old_beta*1.2

            # multiplier = self._get_beta_multiplier(wd_diff_list[-1])

            # if beta*multiplier < self._parameters.max_beta:
            #     beta = beta * multiplier
        else:
            beta = old_beta+0.2
        return beta

    @staticmethod
    def _get_beta_multiplier(wd_difference: float) -> float:
        """Interpolation tends to underestimate beta. A multiplier is used to avoid unnecessary trial
            Linear scaler based on testing

        Parameters
        ----------
        wd_difference : float
            (modelled seabed depth - observed seabed depth)

        Returns
        -------
        multiplier : float
            Multiplier for beta factor
        """
        multiplier = wd_difference*(-5e-4)+0.5
        if multiplier <= 1:
            multiplier = 1.1
        return multiplier
    
    @staticmethod
    def _equalise_array_shape(output_store_arr:np.ndarray,new_data_arr:np.ndarray,pad_val:int|float)->np.ndarray:
        """Pad arrays for mult rift simulation

        Parameters
        ----------
        output_store_arr : np.ndarray
            Output array from previous rift event
        new_data_arr : np.ndarray
            Output array from current rift event
        pad_val : int | float
            Padding value to add at the end of the array

        Returns
        -------
        output_store_arr : np.ndarray
            Output array from previous rift event
        new_data_arr : np.ndarray
            Output array from current rift event
        """
        if (new_data_arr.shape[0]>output_store_arr.shape[0]):
            missing_length=new_data_arr.shape[0]-output_store_arr.shape[0]
            output_store_arr = np.pad(output_store_arr, ((0,missing_length),(0,0)), 'constant', constant_values=pad_val)
        else:
            missing_length=output_store_arr.shape[0]-new_data_arr.shape[0]
            new_data_arr = np.pad(new_data_arr, ((0,missing_length),(0,0)), 'constant', constant_values=pad_val)
        return output_store_arr,new_data_arr
    
    def calculate_new_temperature(self,
                                  sedflag: bool,
                                  coord_crust_lith: np.ndarray[np.float64],
                                  t_old: np.ndarray[np.float64],
                                  HP: np.ndarray[np.float64],
                                  xsed: np.ndarray[np.float64],
                                  hc: float,
                                  hLith: float,
                                  Tsed: np.ndarray[np.float64],
                                  HPsed: np.ndarray[np.float64],
                                  idsed: np.ndarray[np.int32],
                                  ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Calculate new temperature profile in new time step

        Args:
            sedflag (bool): If sediments exists
            coord (np.ndarray[np.float64]): 1D vertical mesh of crust, lithosphere and asthenosphere
            t_old (np.ndarray[np.float64]): Temperature profile of crust, lithosphere and asthenosphere at previous time step
            HP (np.ndarray[np.float64]): RHP of crust, lithosphere and asthenosphere at this new time step
            xsed (np.ndarray[np.float64]): 1D vertical mesh of sedimentary column
            hc (float): Depth of base crust
            hLith (float): Depth of base lithosphere
            Tsed (np.ndarray[np.float64]): Temperature profile of sediments at previous time step
            HPsed (np.ndarray[np.float64]): RHP of sediments at this new time step
            idsed (np.ndarray[np.int32]): ID of sediments

        Returns:
            tuple[np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64],np.ndarray[np.float64]]: _description_
        """
        if sedflag == True:
            coord_all = np.append(xsed[:-1], (coord_crust_lith + xsed[-1]))
            T_all = np.append(Tsed[:-1], t_old)
            HP_all = np.append(HPsed, HP)
        else:
            coord_all = coord_crust_lith
            T_all = t_old
            HP_all = HP
        mean_porosity_arr, sed_idx_arr = self._sediments_mean_porosity(
            xsed,  idsed)
        # if (self.current_node.X==12150) and (self.current_node.Y==12000):
        #     print("Lith", i, hLith, lithUpdated)
        
        density_sed = self.sediment_density(
            mean_porosity_arr, self.current_node.sediments["solidus"].values[sed_idx_arr])
        conductivity_sed = self._sediment_conductivity_sekiguchi(
            mean_porosity_arr, self.current_node.sediments["k_cond"].values[sed_idx_arr],Tsed)


        conductivity_crust_lith = self._build_crust_lithosphere_properties(
            coord_crust_lith, hc, hLith, self.current_node.kCrust, self.current_node.kLith, self.current_node.kAsth)
        conductivity = np.append(conductivity_sed, conductivity_crust_lith)
        dx_arr = coord_all[1:]-coord_all[:-1]
        conductivity_diag = conductivity/dx_arr
        COND_diag = np.convolve(conductivity_diag, [1, 1])
        COND_subdiag = -conductivity_diag
        # packed format suitable for np.linalg.solve_banded()
        COND_packed = np.zeros((3,len(COND_diag)))
        COND_packed[0,1:] = COND_subdiag
        COND_packed[1,:] = COND_diag
        COND_packed[2,:-1] = COND_subdiag
        # todo: avoid forming this matrix
        # COND = np.diag(COND_diag)+np.diag(COND_subdiag,1)+np.diag(COND_subdiag,-1)
        SOUR = np.convolve(HP_all*dx_arr, [0.5, 0.5])
        density_crust_lith = self._build_crust_lithosphere_properties(
            coord_crust_lith, hc, hLith, self.current_node.crustsolid, self.current_node.lithsolid, self.current_node.asthsolid)
        density_all = np.append(density_sed, density_crust_lith)

        T, densityeff = self.implicit_euler_solve(
            T_all,
            coord_all,
            density_all,
            COND_packed,
            SOUR,
            self.current_node.kAsth,
            HP_all[-1],
            # self.current_node.Tinit[-1],
            1330+(coord_all[-1]-hLith)*self._parameters.adiab
        )

        # split sed from lithos
        if sedflag == True:
            sednode = xsed.size
            Tsed = T[:sednode]
            densityeffsed = densityeff[: sednode - 1]
            T = T[sednode - 1:]
            coord_all = coord_all[sednode - 1:] - xsed[-1]
            densityeff = densityeff[sednode - 1:]
        else:
            Tsed = np.zeros(1)
            densityeffsed = np.empty(0)
        T = np.interp(coord_crust_lith, coord_all, T)
        return T, densityeff, Tsed, densityeffsed

    def implicit_euler_solve(
        self,
        T_start: np.ndarray[np.float64],
        coord_all: np.ndarray[np.float64],
        density_all: np.ndarray[np.float64],
        conductivity_packed: np.ndarray[np.float64],
        source: np.ndarray[np.float64], # SOUR
        k_last_node: float,
        HP_last_node: float,
        T_base: float,
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Solve heat equation using backward Euler scheme

        Args:
            T_start (np.ndarray[np.float64]): Temperature profile of starting condition
            coord_all (np.ndarray[np.float64]): 1D vertical mesh of whole model
            density_all (np.ndarray[np.float64]): Density of whole model
            conductivity (np.ndarray[np.float64]): Conductivity of whole model
            source (np.ndarray[np.float64]): Heat source term
            k_last_node (float): Conductivity at the base of model
            HP_last_node (float): RHP at the base of model

        Returns:
            tuple[np.ndarray[np.float64],np.ndarray[np.float64]]: _description_
        """
        # setup first discretize scheme. Run all time in one step.
        discret_steps = 1
        T_last_step = T_start
        total_time_in_s_to_simulate = (
            abs(self._parameters.time_step_Ma)) * self._parameters.myr2s
        time_in_s_per_discret_step = total_time_in_s_to_simulate
        dx_arr = coord_all[1:]-coord_all[:-1]
        while True:
            T_old = T_start
            for step in range(discret_steps):
                (
                    density_effective,
                    time_derivative_packed
                ) = self._assemble_time_derivative(
                    dx_arr,
                    T_old,
                    density_all,
                )
                # implicit assembly
                Lpacked = time_derivative_packed / time_in_s_per_discret_step + conductivity_packed
                # compute R from the packed time derivative matrix
                Rpacked = (time_derivative_packed[1,:] / time_in_s_per_discret_step) * T_old + source
                Rpacked[:-1] += (time_derivative_packed[0,1:] / time_in_s_per_discret_step) * T_old[1:]
                Rpacked[1:] += (time_derivative_packed[2,:-1] / time_in_s_per_discret_step) * T_old[:-1]
                # fixed surface temp BC
                Rpacked[0] = self.current_node.T0
                Lpacked[0,1] = 0
                Lpacked[1,0] = 1
                # basal BC
                if self._parameters.bflux:
                    Rpacked[-1] = self.current_node.qbase * dx_arr[-1] / k_last_node + \
                        HP_last_node * dx_arr[-1] * \
                        dx_arr[-1] / 2 / k_last_node
                    Lpacked[2,-2] = -1
                    Lpacked[1,-1] = 1
                else:
                    Rpacked[-1] = T_base
                    Lpacked[2,-2] = 0
                    Lpacked[1,-1] = 1
                T_this_step = solve_banded((1,1),Lpacked,Rpacked)
                T_old = T_this_step.copy()
            if self._check_convergence(T_last_step, T_this_step):
                break
            else:
                # new time discretize scheme
                discret_steps = 2 * discret_steps
                time_in_s_per_discret_step = total_time_in_s_to_simulate / discret_steps
                T_last_step = T_this_step.copy()
        return T_this_step, density_effective

    def _assemble_time_derivative(
        self,
        dx_arr: np.ndarray[np.float64],
        T_initial: np.ndarray[np.float64],
        density: np.ndarray[np.float64],
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Assemble effective density and time derivative for backward Euler scheme

        Args:
            dx_arr (np.ndarray[np.float64]): Thickness of each mesh elements
            T_initial (np.ndarray[np.float64]): Temperature at the start of model
            density (np.ndarray[np.float64]): Surface density of all elements

        Returns:
            tuple[np.ndarray[np.float64],np.ndarray[np.float64]]: _description_
        """
        density_effective_arr = self._effective_density(density, T_initial)
        density_effective_sum_arr = density_effective_arr * \
            dx_arr*self._parameters.cp
        time_derivative_diag = np.convolve(density_effective_sum_arr,[1/3,1/3])
        time_derivative_subdiag = density_effective_sum_arr/6
        td_packed = np.zeros((3,len(time_derivative_diag)))
        td_packed[0,1:] = time_derivative_subdiag
        td_packed[1,:] = time_derivative_diag
        td_packed[2,:-1] = time_derivative_subdiag
        return density_effective_arr, td_packed
