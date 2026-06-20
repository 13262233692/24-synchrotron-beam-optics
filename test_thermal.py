import os
import sys
import math
os.environ['PYTHONPATH'] = r'd:\SOLO-0619-3\24-synchrotron-beam-optics\python'
sys.path.insert(0, r'd:\SOLO-0619-3\24-synchrotron-beam-optics\python')

import numpy as np
from synchrotron.core import (
    ThermalFDMSolver, ThermalConfig,
    BeamlineSimulator, BeamlineConfig, ThermalConfig,
    HDF5Writer
)

print("=== Test 1: Thermal FDM Solver standalone ===")
tcfg = ThermalConfig(
    crystal_width_mm=50.0,
    crystal_height_mm=50.0,
    crystal_thickness_mm=1.0,
    grid_nx=64,
    grid_ny=64,
    beam_sigma_mm=1.0,
    beam_power_w=100.0,
    heat_sink_temp_k=298.0,
    ambient_temp_k=298.0,
    dt_sec=0.0,
    max_iterations=100000,
    convergence_tolerance_k=1e-2,
)
solver = ThermalFDMSolver(tcfg)
import math
bragg_10kev = 11.4036 * math.pi / 180.0
solver.set_bragg_angle(bragg_10kev)
solver.set_beam_position(0.5, 0.0)
state = solver.solve_steady_state()
print(f"Converged: {state.converged}, Iterations: {state.iterations}")
print(f"Max Temp: {state.max_temp_k:.3f} K, Delta T: {state.delta_t_max_k:.3f} K")
print(f"Max Displacement: {state.deformation.max_displacement_um:.4f} um")
print(f"Max Slope: {state.deformation.max_slope_rad:.6e} rad")
print(f"Mean Slope: {state.deformation.mean_slope_rad:.6e} rad")
print(f"PZT Bias: {state.compensation.pitch_bias_urad:.2f} urad")
print(f"Uncompensated Height Error: {state.compensation.estimated_beam_height_error_um:.4f} um")
print(f"Compensated Height Error: {state.compensation.compensated_height_error_um:.6f} um")
assert state.converged, "Thermal solver should converge"
assert state.max_temp_k > 298.0, "Temperature should rise above ambient"
assert state.deformation.max_displacement_um > 0.0, "Should have thermal displacement"
assert state.compensation.pitch_bias_urad != 0.0, "Should have PZT compensation"
assert state.compensation.compensated_height_error_um < 0.001, "PZT should anchor height to sub-nm"
print("Test 1 PASSED")

print("\n=== Test 2: BeamlineSimulator with thermal chain ===")
bcfg = BeamlineConfig(
    energy_start_eV=5000.0,
    energy_end_eV=25000.0,
    num_energy_points=7,
    num_photons_per_point=100,
    offset_mm=15.0,
    crystal_thickness_mm=1.0,
    miller_h=1, miller_k=1, miller_l=1,
)
bcfg.thermal_config.grid_nx = 32
bcfg.thermal_config.grid_ny = 32
bcfg.thermal_config.dt_sec = 0.0
bcfg.thermal_config.max_iterations = 60000
bcfg.thermal_config.convergence_tolerance_k = 1e-2
bcfg.thermal_config.beam_power_w = 50.0
bcfg.thermal_config.beam_sigma_mm = 1.5
sim = BeamlineSimulator(bcfg)
result = sim.run_simulation(use_mpi=False, vectorized=True)
print(f"Keys in result: {sorted(result.keys())}")
n = len(result['energy_eV'])
print(f"Energy points: {n}")
print(f"Energy range: {result['energy_eV'][0]:.0f} - {result['energy_eV'][-1]:.0f} eV")
print(f"Transmitted range: {result['transmitted_intensity'].min():.4f} - {result['transmitted_intensity'].max():.4f}")
print(f"Thermal max_temp range: {result['thermal_max_temp_k'].min():.2f} - {result['thermal_max_temp_k'].max():.2f} K")
print(f"Thermal disp range: {result['thermal_max_displacement_um'].min():.4f} - {result['thermal_max_displacement_um'].max():.4f} um")
print(f"PZT bias range: {result['pzt_pitch_bias_urad'].min():.2f} - {result['pzt_pitch_bias_urad'].max():.2f} urad")
print(f"Compensated height err max: {result['compensated_height_error_um'].max():.6f} um")
assert 'thermal_max_temp_k' in result
assert 'pzt_pitch_bias_urad' in result
assert 'compensated_height_error_um' in result
assert np.all(result['compensated_height_error_um'] < 0.001), "All points should anchor to sub-nm"
print("Test 2 PASSED")

print("\n=== Test 3: HDF5 output with thermal datasets ===")
outfile = r'd:\SOLO-0619-3\24-synchrotron-beam-optics\test_thermal_output.h5'
sim.save_hdf5(outfile)
import h5py
with h5py.File(outfile, 'r') as f:
    print(f"HDF5 Groups: {sorted(list(f.keys()))}")
    print(f"thermal group datasets: {sorted(list(f['thermal'].keys()))}")
    print(f"pzt_compensation group datasets: {sorted(list(f['pzt_compensation'].keys()))}")
    assert 'thermal' in f
    assert 'pzt_compensation' in f
    assert 'max_temperature_k' in f['thermal']
    assert 'pitch_bias_urad' in f['pzt_compensation']
    temps = f['thermal/max_temperature_k'][:]
    pzt = f['pzt_compensation/pitch_bias_urad'][:]
    comp_err = f['pzt_compensation/compensated_height_error_um'][:]
    print(f"Saved temps: {temps.min():.2f} - {temps.max():.2f} K")
    print(f"Saved PZT bias: {pzt.min():.2f} - {pzt.max():.2f} urad")
    print(f"Saved comp err: {comp_err.max():.8f} um")
os.remove(outfile)
print("Test 3 PASSED")

print("\n=== Test 4: Subnormal number check in thermal ===")
subnormals_found = 0
for v in result['thermal_max_temp_k']:
    exp = math.frexp(v)[1]
    if exp < -1021 and v != 0.0:
        subnormals_found += 1
for v in result['pzt_pitch_bias_rad']:
    exp = math.frexp(v)[1]
    if exp < -1021 and v != 0.0:
        subnormals_found += 1
print(f"Subnormals in thermal output: {subnormals_found}")
assert subnormals_found == 0, "No subnormals should leak out"
print("Test 4 PASSED")

print("\n=== ALL TESTS PASSED ===")
