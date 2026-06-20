#include "crystal_geometry.h"
#include "physics_constants.h"
#include <stdexcept>

namespace synchrotron {

CrystalGeometry::CrystalGeometry(double offset_mm,
                                 double crystal_thickness_mm,
                                 double crystal_width_mm,
                                 double crystal_height_mm)
    : offset_mm_(offset_mm),
      thickness_mm_(crystal_thickness_mm),
      width_mm_(crystal_width_mm),
      height_mm_(crystal_height_mm) {}

DCMConfiguration CrystalGeometry::compute_dcm(double energy_eV,
                                               const BraggSolver& solver) const {
    BraggSolution sol = solver.solve(energy_eV);
    double theta = sol.bragg_angle_rad;

    DCMConfiguration cfg;
    cfg.energy_eV = energy_eV;
    cfg.bragg_angle_deg = sol.bragg_angle_deg;

    cfg.crystal1.pitch_angle_deg = sol.bragg_angle_deg;
    cfg.crystal1.pitch_angle_rad = theta;
    cfg.crystal1.yaw_angle_deg = 0.0;
    cfg.crystal1.yaw_angle_rad = 0.0;
    cfg.crystal1.translation_x_mm = 0.0;
    cfg.crystal1.translation_y_mm = 0.0;
    cfg.crystal1.translation_z_mm = 0.0;

    cfg.crystal2.pitch_angle_deg = sol.bragg_angle_deg;
    cfg.crystal2.pitch_angle_rad = theta;
    cfg.crystal2.yaw_angle_deg = 0.0;
    cfg.crystal2.yaw_angle_rad = 0.0;

    cfg.offset_mm = offset_mm_;
    double cos2theta = std::cos(2.0 * theta);
    double sin_theta = std::sin(theta);
    double half_offset = offset_mm_ / 2.0;

    cfg.crystal2.translation_x_mm = half_offset / sin_theta;
    cfg.crystal2.translation_y_mm = offset_mm_;
    cfg.crystal2.translation_z_mm = 0.0;

    cfg.beam_exit_height_mm = offset_mm_;

    return cfg;
}

std::vector<DCMConfiguration> CrystalGeometry::compute_dcm_scan(
    double energy_start_eV,
    double energy_end_eV,
    int num_points,
    const BraggSolver& solver) const {
    std::vector<DCMConfiguration> configs;
    configs.reserve(num_points);
    double step = (energy_end_eV - energy_start_eV) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        double e = energy_start_eV + i * step;
        try {
            configs.push_back(compute_dcm(e, solver));
        } catch (const std::runtime_error&) {
            break;
        }
    }
    return configs;
}

double CrystalGeometry::compute_beam_path_length(double energy_eV,
                                                  const BraggSolver& solver) const {
    BraggSolution sol = solver.solve(energy_eV);
    double theta = sol.bragg_angle_rad;
    double path_in_crystal1 = thickness_mm_ / std::sin(theta);
    double path_in_crystal2 = thickness_mm_ / std::sin(theta);
    double free_space = offset_mm_ / std::cos(theta);
    return path_in_crystal1 + path_in_crystal2 + free_space;
}

double CrystalGeometry::compute_crystal_penetration(double energy_eV,
                                                     const BraggSolver& solver) const {
    BraggSolution sol = solver.solve(energy_eV);
    double theta = sol.bragg_angle_rad;
    return thickness_mm_ / std::sin(theta);
}

}
