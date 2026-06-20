#include "hdf5_writer.h"
#include <H5Cpp.h>
#include <stdexcept>

namespace synchrotron {

void HDF5Writer::write_spectrum(const std::string& filename,
                                 const std::vector<SpectrumPoint>& spectrum,
                                 const std::vector<BraggSolution>& bragg_solutions,
                                 const std::vector<DCMConfiguration>& dcm_configs) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    hsize_t n = spectrum.size();

    auto write_dataset_1d = [&](const std::string& name, const std::vector<double>& data) {
        H5::DataSpace dataspace(1, &n);
        H5::DataSet dataset = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    };

    std::vector<double> energy(n), transmitted(n), absorption(n), path_length(n);
    std::vector<double> c1_pitch(n), c2_pitch(n), c2_x(n), c2_y(n);
    std::vector<double> bragg_angle(n), wavelength(n), d_spacing(n);

    for (hsize_t i = 0; i < n; ++i) {
        energy[i] = spectrum[i].energy_eV;
        transmitted[i] = spectrum[i].transmitted_intensity;
        absorption[i] = spectrum[i].absorption_fraction;
        path_length[i] = spectrum[i].path_length_mm;
        c1_pitch[i] = spectrum[i].crystal1_pitch_deg;
        c2_pitch[i] = spectrum[i].crystal2_pitch_deg;
        c2_x[i] = spectrum[i].crystal2_x_mm;
        c2_y[i] = spectrum[i].crystal2_y_mm;

        if (i < bragg_solutions.size()) {
            bragg_angle[i] = bragg_solutions[i].bragg_angle_deg;
            wavelength[i] = bragg_solutions[i].wavelength_A;
            d_spacing[i] = bragg_solutions[i].d_spacing_A;
        }
    }

    H5::Group spectrum_grp = file.createGroup("/spectrum");
    file.setName("/", "synchrotron_beam_optics");

    write_dataset_1d("/spectrum/energy_eV", energy);
    write_dataset_1d("/spectrum/transmitted_intensity", transmitted);
    write_dataset_1d("/spectrum/absorption_fraction", absorption);
    write_dataset_1d("/spectrum/total_path_length_mm", path_length);

    H5::Group crystal_grp = file.createGroup("/crystal_positions");
    write_dataset_1d("/crystal_positions/crystal1_pitch_deg", c1_pitch);
    write_dataset_1d("/crystal_positions/crystal2_pitch_deg", c2_pitch);
    write_dataset_1d("/crystal_positions/crystal2_x_mm", c2_x);
    write_dataset_1d("/crystal_positions/crystal2_y_mm", c2_y);

    H5::Group bragg_grp = file.createGroup("/bragg");
    write_dataset_1d("/bragg/bragg_angle_deg", bragg_angle);
    write_dataset_1d("/bragg/wavelength_A", wavelength);
    write_dataset_1d("/bragg/d_spacing_A", d_spacing);

    file.close();
}

void HDF5Writer::write_simulation_report(const std::string& filename,
                                          double energy_start_eV,
                                          double energy_end_eV,
                                          int num_energy_points,
                                          int num_photons,
                                          double offset_mm,
                                          int miller_h, int miller_k, int miller_l,
                                          const std::vector<SpectrumPoint>& spectrum,
                                          const std::vector<BraggSolution>& bragg_solutions,
                                          const std::vector<DCMConfiguration>& dcm_configs) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    H5::Group params_grp = file.createGroup("/simulation_parameters");

    auto write_attr_d = [&](const std::string& grp, const std::string& name, double val) {
        H5::DataSpace scalar(H5S_SCALAR);
        H5::Attribute attr = file.openGroup(grp).createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar);
        attr.write(H5::PredType::NATIVE_DOUBLE, &val);
    };
    auto write_attr_i = [&](const std::string& grp, const std::string& name, int val) {
        H5::DataSpace scalar(H5S_SCALAR);
        H5::Attribute attr = file.openGroup(grp).createAttribute(name, H5::PredType::NATIVE_INT, scalar);
        attr.write(H5::PredType::NATIVE_INT, &val);
    };

    write_attr_d("/simulation_parameters", "energy_start_eV", energy_start_eV);
    write_attr_d("/simulation_parameters", "energy_end_eV", energy_end_eV);
    write_attr_i("/simulation_parameters", "num_energy_points", num_energy_points);
    write_attr_i("/simulation_parameters", "num_photons_per_point", num_photons);
    write_attr_d("/simulation_parameters", "offset_mm", offset_mm);
    write_attr_i("/simulation_parameters", "miller_h", miller_h);
    write_attr_i("/simulation_parameters", "miller_k", miller_k);
    write_attr_i("/simulation_parameters", "miller_l", miller_l);

    file.close();

    H5::H5File file2(filename, H5F_ACC_RDWR);
    hsize_t n = spectrum.size();

    auto write_dataset_1d = [&](const std::string& name, const std::vector<double>& data) {
        H5::DataSpace dataspace(1, &n);
        H5::DataSet dataset = file2.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    };

    std::vector<double> energy(n), transmitted(n), absorption(n), path_length(n);
    std::vector<double> c1_pitch(n), c2_pitch(n), c2_x(n), c2_y(n);
    std::vector<double> bragg_angle(n), wavelength(n), d_spacing(n);

    for (hsize_t i = 0; i < n; ++i) {
        energy[i] = spectrum[i].energy_eV;
        transmitted[i] = spectrum[i].transmitted_intensity;
        absorption[i] = spectrum[i].absorption_fraction;
        path_length[i] = spectrum[i].path_length_mm;
        c1_pitch[i] = spectrum[i].crystal1_pitch_deg;
        c2_pitch[i] = spectrum[i].crystal2_pitch_deg;
        c2_x[i] = spectrum[i].crystal2_x_mm;
        c2_y[i] = spectrum[i].crystal2_y_mm;

        if (i < bragg_solutions.size()) {
            bragg_angle[i] = bragg_solutions[i].bragg_angle_deg;
            wavelength[i] = bragg_solutions[i].wavelength_A;
            d_spacing[i] = bragg_solutions[i].d_spacing_A;
        }
    }

    H5::Group spectrum_grp = file2.createGroup("/spectrum");
    write_dataset_1d("/spectrum/energy_eV", energy);
    write_dataset_1d("/spectrum/transmitted_intensity", transmitted);
    write_dataset_1d("/spectrum/absorption_fraction", absorption);
    write_dataset_1d("/spectrum/total_path_length_mm", path_length);

    H5::Group crystal_grp = file2.createGroup("/crystal_positions");
    write_dataset_1d("/crystal_positions/crystal1_pitch_deg", c1_pitch);
    write_dataset_1d("/crystal_positions/crystal2_pitch_deg", c2_pitch);
    write_dataset_1d("/crystal_positions/crystal2_x_mm", c2_x);
    write_dataset_1d("/crystal_positions/crystal2_y_mm", c2_y);

    H5::Group bragg_grp = file2.createGroup("/bragg");
    write_dataset_1d("/bragg/bragg_angle_deg", bragg_angle);
    write_dataset_1d("/bragg/wavelength_A", wavelength);
    write_dataset_1d("/bragg/d_spacing_A", d_spacing);

    file2.close();
}

}
