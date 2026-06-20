from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

PLANCK_CONSTANT_eVS = 4.135667696e-15
SPEED_OF_LIGHT_mmPs = 2.99792458e11
HC_eVnm = 1239.841984
HC_eVA = 12398.41984

SI_LATTICE_CONSTANT_A = 5.431020511

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

SI_K_EDGE_eV = 1839.0
SI_L1_EDGE_eV = 149.7
SI_L2_EDGE_eV = 99.8
SI_L3_EDGE_eV = 99.2

R_ELECTRON_CM = 2.8179403227e-5


def d_spacing_si111() -> float:
    return SI_LATTICE_CONSTANT_A / math.sqrt(3.0)


def energy_to_wavelength_A(energy_eV: float) -> float:
    return HC_eVA / energy_eV


def wavelength_to_energy_eV(wavelength_A: float) -> float:
    return HC_eVA / wavelength_A


def si_absorption_coefficient(energy_eV: float) -> float:
    energy_keV = energy_eV / 1000.0
    if energy_eV < SI_L3_EDGE_eV:
        mu = 1.0e4 * energy_keV ** (-2.5)
    elif energy_eV < SI_L2_EDGE_eV:
        mu = 500.0 * energy_keV ** (-2.8)
    elif energy_eV < SI_L1_EDGE_eV:
        mu = 400.0 * energy_keV ** (-2.8)
    elif energy_eV < SI_K_EDGE_eV:
        mu = 100.0 * energy_keV ** (-2.7)
    else:
        mu = 25.0 * energy_keV ** (-2.7)

    if energy_eV > SI_K_EDGE_eV:
        delta_E = energy_eV - SI_K_EDGE_eV
        if delta_E < 50.0:
            fine_structure = 1.0 + 0.3 * math.sin(delta_E * 0.5) * math.exp(-delta_E / 20.0)
            mu *= fine_structure

    return mu


@dataclass
class BraggSolution:
    bragg_angle_rad: float
    bragg_angle_deg: float
    d_spacing_A: float
    wavelength_A: float
    energy_eV: float
    miller_h: int
    miller_k: int
    miller_l: int
    order: int


class BraggSolver:
    def __init__(self, h: int = 1, k: int = 1, l: int = 1,
                 lattice_const_A: float = 5.431020511):
        self._h = h
        self._k = k
        self._l = l
        self._lattice = lattice_const_A
        self._d_spacing = self._compute_d_spacing()

    def _compute_d_spacing(self) -> float:
        sum_sq = self._h ** 2 + self._k ** 2 + self._l ** 2
        if sum_sq == 0:
            raise ValueError("Miller indices cannot all be zero")
        return self._lattice / math.sqrt(sum_sq)

    @property
    def d_spacing(self) -> float:
        return self._d_spacing

    def bragg_angle(self, energy_eV: float, order: int = 1) -> float:
        lam = energy_to_wavelength_A(energy_eV)
        sin_theta = (order * lam) / (2.0 * self._d_spacing)
        if abs(sin_theta) > 1.0:
            raise ValueError(
                f"Energy {energy_eV} eV below accessible range for "
                f"Si({self._h}{self._k}{self._l}) at order {order}"
            )
        return math.asin(sin_theta)

    def solve(self, energy_eV: float, order: int = 1) -> BraggSolution:
        lam = energy_to_wavelength_A(energy_eV)
        sin_theta = (order * lam) / (2.0 * self._d_spacing)
        if abs(sin_theta) > 1.0:
            raise ValueError(
                f"Energy {energy_eV} eV below accessible range for "
                f"Si({self._h}{self._k}{self._l}) at order {order}"
            )
        theta = math.asin(sin_theta)
        return BraggSolution(
            bragg_angle_rad=theta,
            bragg_angle_deg=theta * RAD_TO_DEG,
            d_spacing_A=self._d_spacing,
            wavelength_A=lam,
            energy_eV=energy_eV,
            miller_h=self._h,
            miller_k=self._k,
            miller_l=self._l,
            order=order,
        )

    def solve_energy_scan(self, energy_start_eV: float, energy_end_eV: float,
                          num_points: int, order: int = 1) -> List[BraggSolution]:
        results = []
        step = (energy_end_eV - energy_start_eV) / (num_points - 1)
        for i in range(num_points):
            e = energy_start_eV + i * step
            try:
                results.append(self.solve(e, order))
            except ValueError:
                break
        return results

    def _structure_factor_si(self) -> float:
        f_si = 14.0
        multiplier = 4.0
        all_odd = (self._h % 2 != 0) and (self._k % 2 != 0) and (self._l % 2 != 0)
        all_even = (self._h % 2 == 0) and (self._k % 2 == 0) and (self._l % 2 == 0)
        sum_even = (self._h + self._k + self._l) % 2 == 0

        if all_odd:
            return multiplier * f_si * math.sqrt(2.0)
        elif all_even and sum_even:
            return multiplier * f_si * 2.0
        else:
            return 0.0

    def darwin_width_rad(self, energy_eV: float) -> float:
        theta = self.bragg_angle(energy_eV)
        F_h = self._structure_factor_si()
        lam_A = energy_to_wavelength_A(energy_eV)
        V_cell_A3 = self._lattice ** 3
        r_e_A = 2.8179403227e-5
        dw = (2.0 * r_e_A * lam_A ** 2 * F_h) / \
             (math.pi * V_cell_A3 * math.sin(2.0 * theta))
        return dw

    def reflectivity(self, energy_eV: float, angle_offset_rad: float) -> float:
        dw = self.darwin_width_rad(energy_eV)
        delta = angle_offset_rad / (dw / 2.0)
        if abs(delta) <= 1.0:
            return 1.0
        x = abs(delta) - math.sqrt(delta * delta - 1.0)
        return x * x


@dataclass
class CrystalPosition:
    pitch_angle_deg: float = 0.0
    pitch_angle_rad: float = 0.0
    yaw_angle_deg: float = 0.0
    yaw_angle_rad: float = 0.0
    translation_x_mm: float = 0.0
    translation_y_mm: float = 0.0
    translation_z_mm: float = 0.0


@dataclass
class DCMConfiguration:
    crystal1: CrystalPosition = field(default_factory=CrystalPosition)
    crystal2: CrystalPosition = field(default_factory=CrystalPosition)
    offset_mm: float = 15.0
    bragg_angle_deg: float = 0.0
    energy_eV: float = 0.0
    beam_exit_height_mm: float = 0.0


class CrystalGeometry:
    def __init__(self, offset_mm: float = 15.0,
                 crystal_thickness_mm: float = 1.0,
                 crystal_width_mm: float = 50.0,
                 crystal_height_mm: float = 50.0):
        self._offset = offset_mm
        self._thickness = crystal_thickness_mm
        self._width = crystal_width_mm
        self._height = crystal_height_mm

    @property
    def fixed_offset(self) -> float:
        return self._offset

    @property
    def crystal_thickness(self) -> float:
        return self._thickness

    def compute_dcm(self, energy_eV: float, solver: BraggSolver) -> DCMConfiguration:
        sol = solver.solve(energy_eV)
        theta = sol.bragg_angle_rad
        sin_theta = math.sin(theta)
        half_offset = self._offset / 2.0

        c1 = CrystalPosition(
            pitch_angle_deg=sol.bragg_angle_deg,
            pitch_angle_rad=theta,
        )
        c2 = CrystalPosition(
            pitch_angle_deg=sol.bragg_angle_deg,
            pitch_angle_rad=theta,
            translation_x_mm=half_offset / sin_theta,
            translation_y_mm=self._offset,
        )

        return DCMConfiguration(
            crystal1=c1,
            crystal2=c2,
            offset_mm=self._offset,
            bragg_angle_deg=sol.bragg_angle_deg,
            energy_eV=energy_eV,
            beam_exit_height_mm=self._offset,
        )

    def compute_dcm_scan(self, energy_start_eV: float, energy_end_eV: float,
                         num_points: int, solver: BraggSolver) -> List[DCMConfiguration]:
        configs = []
        step = (energy_end_eV - energy_start_eV) / (num_points - 1)
        for i in range(num_points):
            e = energy_start_eV + i * step
            try:
                configs.append(self.compute_dcm(e, solver))
            except ValueError:
                break
        return configs

    def compute_beam_path_length(self, energy_eV: float, solver: BraggSolver) -> float:
        sol = solver.solve(energy_eV)
        theta = sol.bragg_angle_rad
        path_c1 = self._thickness / math.sin(theta)
        path_c2 = self._thickness / math.sin(theta)
        free = self._offset / math.cos(theta)
        return path_c1 + path_c2 + free

    def compute_crystal_penetration(self, energy_eV: float, solver: BraggSolver) -> float:
        sol = solver.solve(energy_eV)
        theta = sol.bragg_angle_rad
        return self._thickness / math.sin(theta)


@dataclass
class PhotonTransportResult:
    transmitted_weight: float = 0.0
    absorbed_weight: float = 0.0
    total_path_length_mm: float = 0.0
    crystal1_absorption: float = 0.0
    crystal2_absorption: float = 0.0
    crystal1_path_mm: float = 0.0
    crystal2_path_mm: float = 0.0
    reflectivity_crystal1: float = 0.0
    reflectivity_crystal2: float = 0.0
    transmission_fraction: float = 0.0


@dataclass
class SpectrumPoint:
    energy_eV: float = 0.0
    transmitted_intensity: float = 0.0
    absorption_fraction: float = 0.0
    path_length_mm: float = 0.0
    crystal1_pitch_deg: float = 0.0
    crystal2_pitch_deg: float = 0.0
    crystal2_x_mm: float = 0.0
    crystal2_y_mm: float = 0.0


class PhotonTransport:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def _crystal_absorption(self, energy_eV: float, path_length_mm: float) -> float:
        mu = si_absorption_coefficient(energy_eV)
        path_cm = path_length_mm * 0.1
        return 1.0 - math.exp(-mu * path_cm)

    def _sample_angular_deviation(self, darwin_width_rad: float) -> float:
        u = self._rng.uniform(0.0, 1.0)
        return darwin_width_rad * (2.0 * u - 1.0)

    def _reflect_from_crystal(self, deviation: float, darwin_width_rad: float) -> bool:
        normalized = deviation / (darwin_width_rad / 2.0)
        if abs(normalized) <= 1.0:
            return True
        x = abs(normalized) - math.sqrt(normalized * normalized - 1.0)
        reflectivity = x * x
        return self._rng.uniform() < reflectivity

    def simulate_single_energy(self, energy_eV: float, bragg_angle_rad: float,
                               offset_mm: float, crystal_thickness_mm: float,
                               darwin_width_rad: float,
                               num_photons: int) -> PhotonTransportResult:
        sin_theta = math.sin(bragg_angle_rad)
        cos_theta = math.cos(bragg_angle_rad)
        path_in_crystal = crystal_thickness_mm / sin_theta

        reflected1 = 0
        reflected2 = 0
        transmitted_weight = 0.0
        absorbed_weight = 0.0
        c1_abs = 0.0
        c2_abs = 0.0
        total_path = 0.0
        c1_path = 0.0
        c2_path = 0.0

        for _ in range(num_photons):
            weight = 1.0
            photon_path = 0.0

            dev1 = self._sample_angular_deviation(darwin_width_rad)
            if not self._reflect_from_crystal(dev1, darwin_width_rad):
                absorbed_weight += weight
                continue
            reflected1 += 1

            abs_frac1 = self._crystal_absorption(energy_eV, path_in_crystal)
            weight *= (1.0 - abs_frac1)
            c1_abs += abs_frac1
            photon_path += path_in_crystal

            dev2 = self._sample_angular_deviation(darwin_width_rad)
            if not self._reflect_from_crystal(dev2, darwin_width_rad):
                absorbed_weight += weight
                continue
            reflected2 += 1

            abs_frac2 = self._crystal_absorption(energy_eV, path_in_crystal)
            weight *= (1.0 - abs_frac2)
            c2_abs += abs_frac2
            photon_path += path_in_crystal

            free_path = offset_mm / cos_theta
            photon_path += free_path

            transmitted_weight += weight
            total_path += photon_path
            c1_path += path_in_crystal
            c2_path += path_in_crystal

        result = PhotonTransportResult()
        if num_photons > 0:
            result.reflectivity_crystal1 = reflected1 / num_photons
            result.reflectivity_crystal2 = reflected2 / num_photons if reflected1 > 0 else 0.0
            result.transmission_fraction = reflected2 / num_photons
            result.absorbed_weight = absorbed_weight / num_photons
            result.transmitted_weight = transmitted_weight / num_photons
            if reflected2 > 0:
                result.total_path_length_mm = total_path / reflected2
                result.crystal1_path_mm = c1_path / reflected2
                result.crystal2_path_mm = c2_path / reflected2
            result.crystal1_absorption = c1_abs / num_photons
            result.crystal2_absorption = c2_abs / num_photons

        return result

    def simulate_single_energy_vectorized(self, energy_eV: float, bragg_angle_rad: float,
                                          offset_mm: float, crystal_thickness_mm: float,
                                          darwin_width_rad: float,
                                          num_photons: int) -> PhotonTransportResult:
        sin_theta = math.sin(bragg_angle_rad)
        cos_theta = math.cos(bragg_angle_rad)
        path_in_crystal = crystal_thickness_mm / sin_theta
        free_path = offset_mm / cos_theta

        mu = si_absorption_coefficient(energy_eV)
        path_cm = path_in_crystal * 0.1
        abs_fraction = 1.0 - math.exp(-mu * path_cm)

        dev1 = self._rng.uniform(-darwin_width_rad, darwin_width_rad, num_photons)
        dev2 = self._rng.uniform(-darwin_width_rad, darwin_width_rad, num_photons)

        normalized1 = dev1 / (darwin_width_rad / 2.0)
        normalized2 = dev2 / (darwin_width_rad / 2.0)

        reflect1 = np.abs(normalized1) <= 1.0
        ref1_outside = ~reflect1
        if np.any(ref1_outside):
            x1 = np.abs(normalized1[ref1_outside]) - np.sqrt(normalized1[ref1_outside] ** 2 - 1.0)
            r1 = x1 ** 2
            u1 = self._rng.uniform(size=r1.shape)
            reflect1[ref1_outside] = u1 < r1

        reflect2 = np.abs(normalized2) <= 1.0
        ref2_outside = ~reflect2
        if np.any(ref2_outside):
            x2 = np.abs(normalized2[ref2_outside]) - np.sqrt(normalized2[ref2_outside] ** 2 - 1.0)
            r2 = x2 ** 2
            u2 = self._rng.uniform(size=r2.shape)
            reflect2[ref2_outside] = u2 < r2

        both_reflect = reflect1 & reflect2
        n_ref1 = int(np.sum(reflect1))
        n_both = int(np.sum(both_reflect))

        weight_after_c1 = (1.0 - abs_fraction) * reflect1.astype(float)
        weight_after_c2 = weight_after_c1 * (1.0 - abs_fraction) * reflect2.astype(float)

        transmitted_weight = float(np.sum(weight_after_c2)) / num_photons
        absorbed_weight = 1.0 - float(np.sum(weight_after_c1 * (1.0 - abs_fraction) * reflect2.astype(float))) / num_photons

        result = PhotonTransportResult()
        result.reflectivity_crystal1 = n_ref1 / num_photons
        result.reflectivity_crystal2 = n_both / n_ref1 if n_ref1 > 0 else 0.0
        result.transmission_fraction = n_both / num_photons
        result.transmitted_weight = transmitted_weight
        result.absorbed_weight = absorbed_weight
        result.total_path_length_mm = 2.0 * path_in_crystal + free_path
        result.crystal1_path_mm = path_in_crystal
        result.crystal2_path_mm = path_in_crystal
        result.crystal1_absorption = abs_fraction * n_ref1 / num_photons
        result.crystal2_absorption = abs_fraction * n_both / num_photons

        return result

    def simulate_energy_scan(self, energy_start_eV: float, energy_end_eV: float,
                             num_energy_points: int,
                             bragg_angles_rad: List[float],
                             offset_mm: float, crystal_thickness_mm: float,
                             darwin_widths_rad: List[float],
                             num_photons_per_point: int,
                             vectorized: bool = True) -> List[SpectrumPoint]:
        spectrum = []
        step = (energy_end_eV - energy_start_eV) / (num_energy_points - 1)

        for i in range(num_energy_points):
            e = energy_start_eV + i * step
            if i >= len(bragg_angles_rad) or i >= len(darwin_widths_rad):
                break

            sim_fn = self.simulate_single_energy_vectorized if vectorized else self.simulate_single_energy
            res = sim_fn(e, bragg_angles_rad[i], offset_mm, crystal_thickness_mm,
                         darwin_widths_rad[i], num_photons_per_point)

            sin_theta = math.sin(bragg_angles_rad[i])
            pt = SpectrumPoint(
                energy_eV=e,
                transmitted_intensity=res.transmitted_weight,
                absorption_fraction=res.absorbed_weight,
                path_length_mm=res.total_path_length_mm,
                crystal1_pitch_deg=bragg_angles_rad[i] * RAD_TO_DEG,
                crystal2_pitch_deg=bragg_angles_rad[i] * RAD_TO_DEG,
                crystal2_x_mm=(offset_mm / 2.0) / sin_theta,
                crystal2_y_mm=offset_mm,
            )
            spectrum.append(pt)

        return spectrum


class HDF5Writer:
    @staticmethod
    def write_spectrum(filename: str, spectrum: List[SpectrumPoint],
                       bragg_solutions: List[BraggSolution],
                       dcm_configs: List[DCMConfiguration]):
        import h5py
        with h5py.File(filename, 'w') as f:
            n = len(spectrum)

            energy = np.array([p.energy_eV for p in spectrum])
            transmitted = np.array([p.transmitted_intensity for p in spectrum])
            absorption = np.array([p.absorption_fraction for p in spectrum])
            path_length = np.array([p.path_length_mm for p in spectrum])
            c1_pitch = np.array([p.crystal1_pitch_deg for p in spectrum])
            c2_pitch = np.array([p.crystal2_pitch_deg for p in spectrum])
            c2_x = np.array([p.crystal2_x_mm for p in spectrum])
            c2_y = np.array([p.crystal2_y_mm for p in spectrum])

            spec_grp = f.create_group('spectrum')
            spec_grp.create_dataset('energy_eV', data=energy)
            spec_grp.create_dataset('transmitted_intensity', data=transmitted)
            spec_grp.create_dataset('absorption_fraction', data=absorption)
            spec_grp.create_dataset('total_path_length_mm', data=path_length)

            crystal_grp = f.create_group('crystal_positions')
            crystal_grp.create_dataset('crystal1_pitch_deg', data=c1_pitch)
            crystal_grp.create_dataset('crystal2_pitch_deg', data=c2_pitch)
            crystal_grp.create_dataset('crystal2_x_mm', data=c2_x)
            crystal_grp.create_dataset('crystal2_y_mm', data=c2_y)

            bragg_grp = f.create_group('bragg')
            if bragg_solutions:
                bragg_angle = np.array([s.bragg_angle_deg for s in bragg_solutions])
                wavelength = np.array([s.wavelength_A for s in bragg_solutions])
                d_sp = np.array([s.d_spacing_A for s in bragg_solutions])
                bragg_grp.create_dataset('bragg_angle_deg', data=bragg_angle)
                bragg_grp.create_dataset('wavelength_A', data=wavelength)
                bragg_grp.create_dataset('d_spacing_A', data=d_sp)

    @staticmethod
    def write_simulation_report(filename: str,
                                energy_start_eV: float, energy_end_eV: float,
                                num_energy_points: int, num_photons: int,
                                offset_mm: float,
                                miller_h: int, miller_k: int, miller_l: int,
                                spectrum: List[SpectrumPoint],
                                bragg_solutions: List[BraggSolution],
                                dcm_configs: List[DCMConfiguration]):
        import h5py
        with h5py.File(filename, 'w') as f:
            params = f.create_group('simulation_parameters')
            params.attrs['energy_start_eV'] = energy_start_eV
            params.attrs['energy_end_eV'] = energy_end_eV
            params.attrs['num_energy_points'] = num_energy_points
            params.attrs['num_photons_per_point'] = num_photons
            params.attrs['offset_mm'] = offset_mm
            params.attrs['miller_h'] = miller_h
            params.attrs['miller_k'] = miller_k
            params.attrs['miller_l'] = miller_l
            params.attrs['si_lattice_constant_A'] = SI_LATTICE_CONSTANT_A

            n = len(spectrum)
            energy = np.array([p.energy_eV for p in spectrum])
            transmitted = np.array([p.transmitted_intensity for p in spectrum])
            absorption = np.array([p.absorption_fraction for p in spectrum])
            path_length = np.array([p.path_length_mm for p in spectrum])
            c1_pitch = np.array([p.crystal1_pitch_deg for p in spectrum])
            c2_pitch = np.array([p.crystal2_pitch_deg for p in spectrum])
            c2_x = np.array([p.crystal2_x_mm for p in spectrum])
            c2_y = np.array([p.crystal2_y_mm for p in spectrum])

            spec_grp = f.create_group('spectrum')
            spec_grp.create_dataset('energy_eV', data=energy)
            spec_grp.create_dataset('transmitted_intensity', data=transmitted)
            spec_grp.create_dataset('absorption_fraction', data=absorption)
            spec_grp.create_dataset('total_path_length_mm', data=path_length)

            crystal_grp = f.create_group('crystal_positions')
            crystal_grp.create_dataset('crystal1_pitch_deg', data=c1_pitch)
            crystal_grp.create_dataset('crystal2_pitch_deg', data=c2_pitch)
            crystal_grp.create_dataset('crystal2_x_mm', data=c2_x)
            crystal_grp.create_dataset('crystal2_y_mm', data=c2_y)

            bragg_grp = f.create_group('bragg')
            if bragg_solutions:
                bragg_angle = np.array([s.bragg_angle_deg for s in bragg_solutions])
                wavelength = np.array([s.wavelength_A for s in bragg_solutions])
                d_sp = np.array([s.d_spacing_A for s in bragg_solutions])
                bragg_grp.create_dataset('bragg_angle_deg', data=bragg_angle)
                bragg_grp.create_dataset('wavelength_A', data=wavelength)
                bragg_grp.create_dataset('d_spacing_A', data=d_sp)

            if dcm_configs:
                dcm_grp = f.create_group('dcm_configurations')
                dcm_bragg = np.array([c.bragg_angle_deg for c in dcm_configs])
                dcm_exit = np.array([c.beam_exit_height_mm for c in dcm_configs])
                dcm_c1_x = np.array([c.crystal1.translation_x_mm for c in dcm_configs])
                dcm_c1_y = np.array([c.crystal1.translation_y_mm for c in dcm_configs])
                dcm_c2_x = np.array([c.crystal2.translation_x_mm for c in dcm_configs])
                dcm_c2_y = np.array([c.crystal2.translation_y_mm for c in dcm_configs])
                dcm_grp.create_dataset('bragg_angle_deg', data=dcm_bragg)
                dcm_grp.create_dataset('beam_exit_height_mm', data=dcm_exit)
                dcm_grp.create_dataset('crystal1_x_mm', data=dcm_c1_x)
                dcm_grp.create_dataset('crystal1_y_mm', data=dcm_c1_y)
                dcm_grp.create_dataset('crystal2_x_mm', data=dcm_c2_x)
                dcm_grp.create_dataset('crystal2_y_mm', data=dcm_c2_y)


@dataclass
class SimulationConfig:
    energy_start_eV: float = 5000.0
    energy_end_eV: float = 25000.0
    num_energy_points: int = 100
    num_photons_per_point: int = 10000
    offset_mm: float = 15.0
    crystal_thickness_mm: float = 1.0
    miller_h: int = 1
    miller_k: int = 1
    miller_l: int = 1
    lattice_constant_A: float = 5.431020511


@dataclass
class SimulationResult:
    spectrum: List[SpectrumPoint] = field(default_factory=list)
    bragg_solutions: List[BraggSolution] = field(default_factory=list)
    dcm_configs: List[DCMConfiguration] = field(default_factory=list)


class MPIScheduler:
    def __init__(self):
        from mpi4py import MPI
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    def run_simulation(self, config: SimulationConfig) -> SimulationResult:
        solver = BraggSolver(config.miller_h, config.miller_k, config.miller_l,
                             config.lattice_constant_A)

        bragg_solutions = solver.solve_energy_scan(
            config.energy_start_eV, config.energy_end_eV, config.num_energy_points
        )

        geometry = CrystalGeometry(config.offset_mm, config.crystal_thickness_mm)
        dcm_configs = geometry.compute_dcm_scan(
            config.energy_start_eV, config.energy_end_eV,
            config.num_energy_points, solver
        )

        bragg_angles = [s.bragg_angle_rad for s in bragg_solutions]
        darwin_widths = [solver.darwin_width_rad(s.energy_eV) for s in bragg_solutions]

        total_points = len(bragg_solutions)
        points_per_rank = total_points // self._size
        remainder = total_points % self._size
        local_start = self._rank * points_per_rank + min(self._rank, remainder)
        local_count = points_per_rank + (1 if self._rank < remainder else 0)

        transport = PhotonTransport(seed=42 + self._rank)

        step = (config.energy_end_eV - config.energy_start_eV) / (config.num_energy_points - 1)
        local_spectrum = []
        for i in range(local_start, local_start + local_count):
            if i >= total_points:
                break
            e = config.energy_start_eV + i * step
            res = transport.simulate_single_energy_vectorized(
                e, bragg_angles[i], config.offset_mm,
                config.crystal_thickness_mm, darwin_widths[i],
                config.num_photons_per_point
            )
            sin_theta = math.sin(bragg_angles[i])
            pt = SpectrumPoint(
                energy_eV=e,
                transmitted_intensity=res.transmitted_weight,
                absorption_fraction=res.absorbed_weight,
                path_length_mm=res.total_path_length_mm,
                crystal1_pitch_deg=bragg_angles[i] * RAD_TO_DEG,
                crystal2_pitch_deg=bragg_angles[i] * RAD_TO_DEG,
                crystal2_x_mm=(config.offset_mm / 2.0) / sin_theta,
                crystal2_y_mm=config.offset_mm,
            )
            local_spectrum.append(pt)

        gathered = self._comm.gather(local_spectrum, root=0)

        if self._rank == 0:
            global_spectrum = []
            for rank_spectrum in gathered:
                global_spectrum.extend(rank_spectrum)
            global_spectrum.sort(key=lambda p: p.energy_eV)
        else:
            global_spectrum = []

        global_spectrum = self._comm.bcast(global_spectrum, root=0)

        result = SimulationResult(
            spectrum=global_spectrum,
            bragg_solutions=bragg_solutions,
            dcm_configs=dcm_configs,
        )
        return result


@dataclass
class BeamlineConfig:
    energy_start_eV: float = 5000.0
    energy_end_eV: float = 25000.0
    num_energy_points: int = 100
    num_photons_per_point: int = 10000
    offset_mm: float = 15.0
    crystal_thickness_mm: float = 1.0
    miller_h: int = 1
    miller_k: int = 1
    miller_l: int = 1
    lattice_constant_A: float = 5.431020511


class BeamlineSimulator:
    def __init__(self, config: Optional[BeamlineConfig] = None):
        self._config = config or BeamlineConfig()
        self._solver = BraggSolver(
            self._config.miller_h,
            self._config.miller_k,
            self._config.miller_l,
            self._config.lattice_constant_A,
        )
        self._geometry = CrystalGeometry(
            self._config.offset_mm,
            self._config.crystal_thickness_mm,
        )
        self._result: Optional[SimulationResult] = None

    @property
    def config(self) -> BeamlineConfig:
        return self._config

    @property
    def solver(self) -> BraggSolver:
        return self._solver

    @property
    def geometry(self) -> CrystalGeometry:
        return self._geometry

    def solve_bragg(self, energy_eV: float) -> BraggSolution:
        return self._solver.solve(energy_eV)

    def solve_bragg_scan(self, energy_start_eV: float, energy_end_eV: float,
                         num_points: int) -> List[BraggSolution]:
        return self._solver.solve_energy_scan(energy_start_eV, energy_end_eV, num_points)

    def compute_dcm(self, energy_eV: float) -> DCMConfiguration:
        return self._geometry.compute_dcm(energy_eV, self._solver)

    def compute_dcm_scan(self, energy_start_eV: float, energy_end_eV: float,
                         num_points: int) -> List[DCMConfiguration]:
        return self._geometry.compute_dcm_scan(energy_start_eV, energy_end_eV,
                                               num_points, self._solver)

    def run_simulation(self, use_mpi: bool = False, vectorized: bool = True) -> Dict[str, np.ndarray]:
        if use_mpi:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                if comm.Get_size() < 2:
                    import warnings
                    warnings.warn(
                        "MPI requested but only 1 rank available. "
                        "Run with: mpiexec -n <N> python your_script.py"
                    )
                    use_mpi = False
            except ImportError:
                import warnings
                warnings.warn("mpi4py not available, falling back to single-process mode")
                use_mpi = False

        if use_mpi:
            cfg = SimulationConfig(
                energy_start_eV=self._config.energy_start_eV,
                energy_end_eV=self._config.energy_end_eV,
                num_energy_points=self._config.num_energy_points,
                num_photons_per_point=self._config.num_photons_per_point,
                offset_mm=self._config.offset_mm,
                crystal_thickness_mm=self._config.crystal_thickness_mm,
                miller_h=self._config.miller_h,
                miller_k=self._config.miller_k,
                miller_l=self._config.miller_l,
                lattice_constant_A=self._config.lattice_constant_A,
            )
            scheduler = MPIScheduler()
            self._result = scheduler.run_simulation(cfg)
        else:
            bragg_solutions = self._solver.solve_energy_scan(
                self._config.energy_start_eV, self._config.energy_end_eV,
                self._config.num_energy_points
            )
            dcm_configs = self._geometry.compute_dcm_scan(
                self._config.energy_start_eV, self._config.energy_end_eV,
                self._config.num_energy_points, self._solver
            )
            bragg_angles = [s.bragg_angle_rad for s in bragg_solutions]
            darwin_widths = [self._solver.darwin_width_rad(s.energy_eV) for s in bragg_solutions]

            transport = PhotonTransport(seed=42)
            spectrum = transport.simulate_energy_scan(
                self._config.energy_start_eV, self._config.energy_end_eV,
                self._config.num_energy_points, bragg_angles,
                self._config.offset_mm, self._config.crystal_thickness_mm,
                darwin_widths, self._config.num_photons_per_point,
                vectorized=vectorized,
            )

            self._result = SimulationResult(
                spectrum=spectrum,
                bragg_solutions=bragg_solutions,
                dcm_configs=dcm_configs,
            )

        return self._result_to_numpy(self._result)

    def save_hdf5(self, filename: str):
        if self._result is None:
            raise RuntimeError("No simulation results. Call run_simulation() first.")
        HDF5Writer.write_simulation_report(
            filename,
            self._config.energy_start_eV,
            self._config.energy_end_eV,
            self._config.num_energy_points,
            self._config.num_photons_per_point,
            self._config.offset_mm,
            self._config.miller_h,
            self._config.miller_k,
            self._config.miller_l,
            self._result.spectrum,
            self._result.bragg_solutions,
            self._result.dcm_configs,
        )

    @staticmethod
    def _result_to_numpy(result: SimulationResult) -> Dict[str, np.ndarray]:
        data = {}

        if result.spectrum:
            sp = result.spectrum
            data['energy_eV'] = np.array([p.energy_eV for p in sp])
            data['transmitted_intensity'] = np.array([p.transmitted_intensity for p in sp])
            data['absorption_fraction'] = np.array([p.absorption_fraction for p in sp])
            data['path_length_mm'] = np.array([p.path_length_mm for p in sp])
            data['crystal1_pitch_deg'] = np.array([p.crystal1_pitch_deg for p in sp])
            data['crystal2_pitch_deg'] = np.array([p.crystal2_pitch_deg for p in sp])
            data['crystal2_x_mm'] = np.array([p.crystal2_x_mm for p in sp])
            data['crystal2_y_mm'] = np.array([p.crystal2_y_mm for p in sp])

        if result.bragg_solutions:
            bs = result.bragg_solutions
            data['bragg_angle_deg'] = np.array([s.bragg_angle_deg for s in bs])
            data['wavelength_A'] = np.array([s.wavelength_A for s in bs])
            data['d_spacing_A'] = np.array([s.d_spacing_A for s in bs])

        if result.dcm_configs:
            dc = result.dcm_configs
            data['dcm_bragg_angle_deg'] = np.array([c.bragg_angle_deg for c in dc])
            data['dcm_beam_exit_height_mm'] = np.array([c.beam_exit_height_mm for c in dc])
            data['crystal1_x_mm'] = np.array([c.crystal1.translation_x_mm for c in dc])
            data['crystal1_y_mm'] = np.array([c.crystal1.translation_y_mm for c in dc])
            data['crystal2_x_mm'] = np.array([c.crystal2.translation_x_mm for c in dc])
            data['crystal2_y_mm'] = np.array([c.crystal2.translation_y_mm for c in dc])

        return data
