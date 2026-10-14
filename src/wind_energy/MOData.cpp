#include <limits>
#include "src/wind_energy/MOData.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf {

/*
 * van der Laan, P., Kelly, M. C., & Sørensen, N. N. (2017). A new k-epsilon
 * model consistent with Monin-Obukhov similarity theory. Wind Energy,
 * 20(3), 479–489. https://doi.org/10.1002/we.2017
 *
 * Consistent with Dyer (1974) formulation from page 57, Chapter 2, Modeling
 * the vertical ABL structure in Modelling of Atmospheric Flow Fields,
 * Demetri P Lalas and Corrado F Ratto, January 1996,
 * https://doi.org/10.1142/2975.
 */

// Non-dimensional Potential Temperature Shear, phi_h
amrex::Real MOData::phi_h(amrex::Real z) const
{
    // Neutral
    amrex::Real phi_h = 1.0_rt;

    // Stable
    if((L < L_neutral) && (L > 0.0_rt)){
        phi_h = alpha_h + gamma_h * (z / L);
    }

    // Unstable
    else if((L > -L_neutral) && (L <= 0.0_rt)){
        phi_h = alpha_h*std::pow(1.0_rt - beta_h * (z / L), -0.5_rt);
    }

    return phi_h;
}
amrex::Real MOData::phi_h() const { return phi_h(zref); }

// Velocity Profile Stability Correction Function, psi_m (stateless, takes zeta = z/L)
amrex::Real MOData::calc_psi_m(
    const amrex::Real zeta, const amrex::Real beta_m, const amrex::Real gamma_m)
{
    if (zeta > 0) {
        return -gamma_m * zeta;
    }
    const amrex::Real x = std::sqrt(std::sqrt(1.0_rt - (beta_m * zeta)));
    return (2.0_rt * std::log(0.5_rt * (1.0_rt + x))) +
           std::log(0.5_rt * (1.0_rt + (x * x))) - (2.0_rt * std::atan(x)) +
           utils::half_pi();
}

// Potential Temperature Profile Stability Correction Function, psi_h (stateless, takes zeta = z/L) (for backward compatibility)
amrex::Real MOData::calc_psi_h(
    const amrex::Real zeta, const amrex::Real beta_h, const amrex::Real gamma_h)
{
    if (zeta > 0) {
        return -gamma_h * zeta;
    }
    const amrex::Real x = std::sqrt(1 - (beta_h * zeta));
    return 2.0_rt * std::log(0.5_rt * (1 + x));
}

// Velocity Profile Stability Correction Function, psi_m
amrex::Real MOData::psi_m(amrex::Real z) const
{
    const bool stable = (L < L_neutral) && (L > 0.0_rt);
    const bool unstable = (L > -L_neutral) && (L <= 0.0_rt);
    return (stable || unstable) ? calc_psi_m(z / L, beta_m, gamma_m) : 0.0_rt;
}

// Potential Temperature Profile Stability Correction Function, psi_h (Used in Surface Fluxes Computation)
amrex::Real MOData::psi_h(amrex::Real z) const
{
    amrex::Real psi_h = 0.0_rt;

    // Stable
    if((L < L_neutral) && (L > 0.0_rt)){
        psi_h = 1.0_rt - (phi_h(z) / alpha_h);
    }

    // Unstable
    else if((L > -L_neutral) && (L <= 0.0_rt)){
        psi_h = 2.0_rt * std::log((1.0_rt + std::pow(phi_h(z) / alpha_h, -1.0_rt)) / 2.0_rt);
    }

    return psi_h;
}

// Update Obukhov Length, L
void MOData::update_L()
{
    // Non-neutral
    if (std::abs(surf_temp_flux) > 1.0e-6_rt){
        L = (-std::pow(utau, 3.0_rt) * theta_mean) / (kappa * gravity * surf_temp_flux);
    }
    
    // Neutral
    else{
        L = L_neutral;
    }
}

// Update Friction Velocity, utau
void MOData::update_utau()
{
    utau = (kappa * ref_speed()) / (std::log(zref / z0) - (psi_m(zref) - psi_m(z0)));
    utau = std::max(utau, 1.0e-6_rt);
}

// Update Surface Temperature
void MOData::update_surf_temp()
{
    surf_temp = theta_mean + ((alpha_h * surf_temp_flux) / (kappa * utau)) * (std::log(zref / z0t) - (psi_h(zref) - psi_h(z0t)));
}

// Update Surface Temperature Flux
void MOData::update_surf_temp_flux()
{
    surf_temp_flux = (-(theta_mean - surf_temp) * kappa * utau) / (alpha_h * (std::log(zref / z0t) - (psi_h(zref) - psi_h(z0t))));
}

void MOData::update_fluxes(int max_iters)
{
    amrex::Real utau1 = std::max(1.0e-6_rt, (kappa * ref_speed()) / (std::log(zref / z0))); // Initialize with neutral utau
    amrex::Real utau0 = utau1 + 2.0 * tol;
    utau = utau1;

    // Without a seed, a cold start with Q=0 leaves L neutral and exits after one pass
    if (alg_type == ThetaCalcType::SURFACE_TEMPERATURE) {
        update_surf_temp_flux();
    }

    int iter = 0;
    while ((std::abs(utau - utau0) > tol) && (iter < max_iters)) {
        utau0 = utau1;
        update_L();
        update_utau();
        switch (alg_type) {
            case ThetaCalcType::HEAT_FLUX:
                update_surf_temp();
                break;
            case ThetaCalcType::SURFACE_TEMPERATURE:
                update_surf_temp_flux();
                break;
        }
        utau1 = utau;
        iter++;
    }

    if (iter >= max_iters) {
    amrex::Print()
        << "MOData::update_fluxes: Convergence criteria not met after "
        << max_iters << " iterations"
        << "\nObukhov length = " << L << ", zeta = " << zref / L
        << "\npsi_m = " << psi_m(zref) << ", psi_h = " << psi_h(zref)
        << "\nutau = " << utau <<", Tsurf = " << surf_temp
        << " q = " << surf_temp_flux << std::endl;
    }
}



} // namespace kynema_sgf
