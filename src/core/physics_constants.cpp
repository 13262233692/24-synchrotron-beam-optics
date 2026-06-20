#include "physics_constants.h"
#include <cmath>
#include <algorithm>

namespace synchrotron {

double si_absorption_coefficient(double energy_eV) {
    double energy_keV = energy_eV / 1000.0;
    double mu = 0.0;

    if (energy_eV < SI_L3_EDGE_eV) {
        mu = 1.0e4 * std::pow(energy_keV, -2.5);
    } else if (energy_eV < SI_L2_EDGE_eV) {
        mu = 500.0 * std::pow(energy_keV, -2.8);
    } else if (energy_eV < SI_L1_EDGE_eV) {
        mu = 400.0 * std::pow(energy_keV, -2.8);
    } else if (energy_eV < SI_K_EDGE_eV) {
        mu = 100.0 * std::pow(energy_keV, -2.7);
    } else {
        mu = 25.0 * std::pow(energy_keV, -2.7);
    }

    if (energy_eV > SI_K_EDGE_eV) {
        double delta_E = energy_eV - SI_K_EDGE_eV;
        if (delta_E < 50.0) {
            double fine_structure = 1.0 + 0.3 * std::sin(delta_E * 0.5) *
                                   std::exp(-delta_E / 20.0);
            mu *= fine_structure;
        }
    }

    return mu;
}

}
