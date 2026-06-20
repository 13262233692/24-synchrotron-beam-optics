#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bragg_solver.h"
#include "crystal_geometry.h"
#include "photon_transport.h"
#include "hdf5_writer.h"
#include "mpi_scheduler.h"

namespace py = pybind11;

PYBIND11_MODULE(_synchrotron, m) {
    m.doc() = "Synchrotron Radiation Beam Optics Simulation Core";

    py::class_<synchrotron::BraggSolution>(m, "BraggSolution")
        .def_readonly("bragg_angle_rad", &synchrotron::BraggSolution::bragg_angle_rad)
        .def_readonly("bragg_angle_deg", &synchrotron::BraggSolution::bragg_angle_deg)
        .def_readonly("d_spacing_A", &synchrotron::BraggSolution::d_spacing_A)
        .def_readonly("wavelength_A", &synchrotron::BraggSolution::wavelength_A)
        .def_readonly("energy_eV", &synchrotron::BraggSolution::energy_eV)
        .def_readonly("miller_h", &synchrotron::BraggSolution::miller_h)
        .def_readonly("miller_k", &synchrotron::BraggSolution::miller_k)
        .def_readonly("miller_l", &synchrotron::BraggSolution::miller_l)
        .def_readonly("order", &synchrotron::BraggSolution::order)
        .def("__repr__", [](const synchrotron::BraggSolution& s) {
            return "<BraggSolution energy=" + std::to_string(s.energy_eV) +
                   " eV angle=" + std::to_string(s.bragg_angle_deg) + " deg>";
        });

    py::class_<synchrotron::CrystalPosition>(m, "CrystalPosition")
        .def_readonly("pitch_angle_deg", &synchrotron::CrystalPosition::pitch_angle_deg)
        .def_readonly("pitch_angle_rad", &synchrotron::CrystalPosition::pitch_angle_rad)
        .def_readonly("yaw_angle_deg", &synchrotron::CrystalPosition::yaw_angle_deg)
        .def_readonly("yaw_angle_rad", &synchrotron::CrystalPosition::yaw_angle_rad)
        .def_readonly("translation_x_mm", &synchrotron::CrystalPosition::translation_x_mm)
        .def_readonly("translation_y_mm", &synchrotron::CrystalPosition::translation_y_mm)
        .def_readonly("translation_z_mm", &synchrotron::CrystalPosition::translation_z_mm);

    py::class_<synchrotron::DCMConfiguration>(m, "DCMConfiguration")
        .def_readonly("crystal1", &synchrotron::DCMConfiguration::crystal1)
        .def_readonly("crystal2", &synchrotron::DCMConfiguration::crystal2)
        .def_readonly("offset_mm", &synchrotron::DCMConfiguration::offset_mm)
        .def_readonly("bragg_angle_deg", &synchrotron::DCMConfiguration::bragg_angle_deg)
        .def_readonly("energy_eV", &synchrotron::DCMConfiguration::energy_eV)
        .def_readonly("beam_exit_height_mm", &synchrotron::DCMConfiguration::beam_exit_height_mm);

    py::class_<synchrotron::PhotonTransportResult>(m, "PhotonTransportResult")
        .def_readonly("transmitted_weight", &synchrotron::PhotonTransportResult::transmitted_weight)
        .def_readonly("absorbed_weight", &synchrotron::PhotonTransportResult::absorbed_weight)
        .def_readonly("total_path_length_mm", &synchrotron::PhotonTransportResult::total_path_length_mm)
        .def_readonly("crystal1_absorption", &synchrotron::PhotonTransportResult::crystal1_absorption)
        .def_readonly("crystal2_absorption", &synchrotron::PhotonTransportResult::crystal2_absorption)
        .def_readonly("reflectivity_crystal1", &synchrotron::PhotonTransportResult::reflectivity_crystal1)
        .def_readonly("reflectivity_crystal2", &synchrotron::PhotonTransportResult::reflectivity_crystal2)
        .def_readonly("transmission_fraction", &synchrotron::PhotonTransportResult::transmission_fraction);

    py::class_<synchrotron::SpectrumPoint>(m, "SpectrumPoint")
        .def_readonly("energy_eV", &synchrotron::SpectrumPoint::energy_eV)
        .def_readonly("transmitted_intensity", &synchrotron::SpectrumPoint::transmitted_intensity)
        .def_readonly("absorption_fraction", &synchrotron::SpectrumPoint::absorption_fraction)
        .def_readonly("path_length_mm", &synchrotron::SpectrumPoint::path_length_mm)
        .def_readonly("crystal1_pitch_deg", &synchrotron::SpectrumPoint::crystal1_pitch_deg)
        .def_readonly("crystal2_pitch_deg", &synchrotron::SpectrumPoint::crystal2_pitch_deg)
        .def_readonly("crystal2_x_mm", &synchrotron::SpectrumPoint::crystal2_x_mm)
        .def_readonly("crystal2_y_mm", &synchrotron::SpectrumPoint::crystal2_y_mm);

    py::class_<synchrotron::BraggSolver>(m, "BraggSolver")
        .def(py::init<int, int, int, double>(),
             py::arg("h") = 1, py::arg("k") = 1, py::arg("l") = 1,
             py::arg("lattice_const_A") = 5.431020511)
        .def("solve", &synchrotron::BraggSolver::solve,
             py::arg("energy_eV"), py::arg("order") = 1)
        .def("solve_energy_scan", &synchrotron::BraggSolver::solve_energy_scan,
             py::arg("energy_start_eV"), py::arg("energy_end_eV"),
             py::arg("num_points"), py::arg("order") = 1)
        .def("bragg_angle", &synchrotron::BraggSolver::bragg_angle,
             py::arg("energy_eV"), py::arg("order") = 1)
        .def("darwin_width_rad", &synchrotron::BraggSolver::darwin_width_rad,
             py::arg("energy_eV"))
        .def("reflectivity", &synchrotron::BraggSolver::reflectivity,
             py::arg("energy_eV"), py::arg("angle_offset_rad"))
        .def("compute_d_spacing", &synchrotron::BraggSolver::compute_d_spacing);

    py::class_<synchrotron::CrystalGeometry>(m, "CrystalGeometry")
        .def(py::init<double, double, double, double>(),
             py::arg("offset_mm") = 15.0,
             py::arg("crystal_thickness_mm") = 1.0,
             py::arg("crystal_width_mm") = 50.0,
             py::arg("crystal_height_mm") = 50.0)
        .def("compute_dcm", &synchrotron::CrystalGeometry::compute_dcm,
             py::arg("energy_eV"), py::arg("solver"))
        .def("compute_dcm_scan", &synchrotron::CrystalGeometry::compute_dcm_scan,
             py::arg("energy_start_eV"), py::arg("energy_end_eV"),
             py::arg("num_points"), py::arg("solver"))
        .def("compute_beam_path_length", &synchrotron::CrystalGeometry::compute_beam_path_length,
             py::arg("energy_eV"), py::arg("solver"))
        .def("compute_crystal_penetration", &synchrotron::CrystalGeometry::compute_crystal_penetration,
             py::arg("energy_eV"), py::arg("solver"))
        .def("fixed_offset", &synchrotron::CrystalGeometry::fixed_offset)
        .def("crystal_thickness", &synchrotron::CrystalGeometry::crystal_thickness);

    py::class_<synchrotron::PhotonTransport>(m, "PhotonTransport")
        .def(py::init<uint64_t>(), py::arg("seed") = 42)
        .def("simulate_single_energy", &synchrotron::PhotonTransport::simulate_single_energy,
             py::arg("energy_eV"), py::arg("bragg_angle_rad"),
             py::arg("offset_mm"), py::arg("crystal_thickness_mm"),
             py::arg("darwin_width_rad"), py::arg("num_photons"))
        .def("simulate_energy_scan", &synchrotron::PhotonTransport::simulate_energy_scan,
             py::arg("energy_start_eV"), py::arg("energy_end_eV"),
             py::arg("num_energy_points"), py::arg("bragg_angles_rad"),
             py::arg("offset_mm"), py::arg("crystal_thickness_mm"),
             py::arg("darwin_widths_rad"), py::arg("num_photons_per_point"));

    m.def("write_spectrum_hdf5", &synchrotron::HDF5Writer::write_spectrum,
          py::arg("filename"), py::arg("spectrum"),
          py::arg("bragg_solutions"), py::arg("dcm_configs"));

    m.def("write_simulation_report_hdf5", &synchrotron::HDF5Writer::write_simulation_report,
          py::arg("filename"), py::arg("energy_start_eV"), py::arg("energy_end_eV"),
          py::arg("num_energy_points"), py::arg("num_photons"),
          py::arg("offset_mm"), py::arg("miller_h"), py::arg("miller_k"), py::arg("miller_l"),
          py::arg("spectrum"), py::arg("bragg_solutions"), py::arg("dcm_configs"));

    m.def("si_absorption_coefficient", &synchrotron::si_absorption_coefficient,
          py::arg("energy_eV"));

    m.def("d_spacing_si111", &synchrotron::d_spacing_si111);
    m.def("energy_to_wavelength_A", &synchrotron::energy_to_wavelength_A,
          py::arg("energy_eV"));
    m.def("wavelength_to_energy_eV", &synchrotron::wavelength_to_energy_eV,
          py::arg("wavelength_A"));

    py::class_<synchrotron::SimulationConfig>(m, "SimulationConfig")
        .def(py::init<>())
        .def_readwrite("energy_start_eV", &synchrotron::SimulationConfig::energy_start_eV)
        .def_readwrite("energy_end_eV", &synchrotron::SimulationConfig::energy_end_eV)
        .def_readwrite("num_energy_points", &synchrotron::SimulationConfig::num_energy_points)
        .def_readwrite("num_photons_per_point", &synchrotron::SimulationConfig::num_photons_per_point)
        .def_readwrite("offset_mm", &synchrotron::SimulationConfig::offset_mm)
        .def_readwrite("crystal_thickness_mm", &synchrotron::SimulationConfig::crystal_thickness_mm)
        .def_readwrite("miller_h", &synchrotron::SimulationConfig::miller_h)
        .def_readwrite("miller_k", &synchrotron::SimulationConfig::miller_k)
        .def_readwrite("miller_l", &synchrotron::SimulationConfig::miller_l)
        .def_readwrite("lattice_constant_A", &synchrotron::SimulationConfig::lattice_constant_A);

    py::class_<synchrotron::MPIScheduler>(m, "MPIScheduler")
        .def(py::init<>())
        .def("run_simulation", &synchrotron::MPIScheduler::run_simulation,
             py::arg("config"))
        .def("rank", &synchrotron::MPIScheduler::rank)
        .def("size", &synchrotron::MPIScheduler::size);

    py::class_<synchrotron::SimulationResult>(m, "SimulationResult")
        .def_readonly("spectrum", &synchrotron::SimulationResult::spectrum)
        .def_readonly("bragg_solutions", &synchrotron::SimulationResult::bragg_solutions)
        .def_readonly("dcm_configs", &synchrotron::SimulationResult::dcm_configs);
}
