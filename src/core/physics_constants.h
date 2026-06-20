#pragma once

#include <cmath>
#include <cstdint>

namespace synchrotron {

constexpr double PLANCK_CONSTANT_eVs  = 4.135667696e-15;
constexpr double SPEED_OF_LIGHT_mmPs  = 2.99792458e11;
constexpr double HC_eVnm = 1239.841984;
constexpr double HC_eVA  = 12398.41984;

constexpr double SI_LATTICE_CONSTANT_A = 5.431020511;

constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;

inline double d_spacing_si111() {
    return SI_LATTICE_CONSTANT_A / std::sqrt(3.0);
}

inline double d_spacing_si220() {
    return SI_LATTICE_CONSTANT_A / std::sqrt(8.0);
}

inline double d_spacing_si311() {
    return SI_LATTICE_CONSTANT_A / std::sqrt(11.0);
}

inline double energy_to_wavelength_A(double energy_eV) {
    return HC_eVA / energy_eV;
}

inline double wavelength_to_energy_eV(double wavelength_A) {
    return HC_eVA / wavelength_A;
}

constexpr double SI_K_EDGE_eV = 1839.0;
constexpr double SI_L1_EDGE_eV = 149.7;
constexpr double SI_L2_EDGE_eV = 99.8;
constexpr double SI_L3_EDGE_eV = 99.2;

double si_absorption_coefficient(double energy_eV);

}
