#pragma once

#include <vector>
#include <string>
#include "photon_transport.h"
#include "crystal_geometry.h"
#include "bragg_solver.h"

namespace synchrotron {

struct HDF5Dataset {
    std::string name;
    std::vector<double> data;
    std::vector<hsize_t> dims;
};

class HDF5Writer {
public:
    static void write_spectrum(const std::string& filename,
                               const std::vector<SpectrumPoint>& spectrum,
                               const std::vector<BraggSolution>& bragg_solutions,
                               const std::vector<DCMConfiguration>& dcm_configs);

    static void write_simulation_report(const std::string& filename,
                                        double energy_start_eV,
                                        double energy_end_eV,
                                        int num_energy_points,
                                        int num_photons,
                                        double offset_mm,
                                        int miller_h, int miller_k, int miller_l,
                                        const std::vector<SpectrumPoint>& spectrum,
                                        const std::vector<BraggSolution>& bragg_solutions,
                                        const std::vector<DCMConfiguration>& dcm_configs);
};

}
