#pragma once

#include "bragg_solver.h"
#include <vector>

namespace synchrotron {

struct CrystalPosition {
    double pitch_angle_deg;
    double pitch_angle_rad;
    double yaw_angle_deg;
    double yaw_angle_rad;
    double translation_x_mm;
    double translation_y_mm;
    double translation_z_mm;
};

struct DCMConfiguration {
    CrystalPosition crystal1;
    CrystalPosition crystal2;
    double offset_mm;
    double bragg_angle_deg;
    double energy_eV;
    double beam_exit_height_mm;
};

class CrystalGeometry {
public:
    CrystalGeometry(double offset_mm = 15.0,
                    double crystal_thickness_mm = 1.0,
                    double crystal_width_mm = 50.0,
                    double crystal_height_mm = 50.0);

    DCMConfiguration compute_dcm(double energy_eV, const BraggSolver& solver) const;

    std::vector<DCMConfiguration> compute_dcm_scan(double energy_start_eV,
                                                    double energy_end_eV,
                                                    int num_points,
                                                    const BraggSolver& solver) const;

    double compute_beam_path_length(double energy_eV, const BraggSolver& solver) const;

    double compute_crystal_penetration(double energy_eV, const BraggSolver& solver) const;

    double fixed_offset() const { return offset_mm_; }
    double crystal_thickness() const { return thickness_mm_; }

private:
    double offset_mm_;
    double thickness_mm_;
    double width_mm_;
    double height_mm_;
};

}
