#pragma once

#include <vector>
#include <string>
#include <cmath>

namespace synchrotron {

struct BraggSolution {
    double bragg_angle_rad;
    double bragg_angle_deg;
    double d_spacing_A;
    double wavelength_A;
    double energy_eV;
    int miller_h;
    int miller_k;
    int miller_l;
    int order;
};

class BraggSolver {
public:
    explicit BraggSolver(int h = 1, int k = 1, int l = 1, double lattice_const_A = 5.431020511);

    BraggSolution solve(double energy_eV, int order = 1) const;

    std::vector<BraggSolution> solve_energy_scan(double energy_start_eV,
                                                  double energy_end_eV,
                                                  int num_points,
                                                  int order = 1) const;

    double compute_d_spacing() const;

    double bragg_angle(double energy_eV, int order = 1) const;

    double darwin_width_rad(double energy_eV) const;

    double reflectivity(double energy_eV, double angle_offset_rad) const;

private:
    int h_, k_, l_;
    double lattice_const_;
    double d_spacing_;

    double compute_structure_factor_si() const;
};

}
