#ifndef MODATA_H
#define MODATA_H

#include <cstdint>
#include "src/utilities/trig_ops.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf {

/** Monin-Obukhov surface layer profile
 *  \ingroup we_abl
 *
 * van der Laan, P., Kelly, M. C., & Sørensen, N. N. (2017). A new k-epsilon
 * model consistent with Monin-Obukhov similarity theory. Wind Energy,
 * 20(3), 479–489. https://doi.org/10.1002/we.2017
 *
 * Consistent with Dyer (1974) formulation from page 57, Chapter 2, Modeling
 * the vertical ABL structure in Modelling of Atmospheric Flow Fields,
 * Demetri P Lalas and Corrado F Ratto, January 1996,
 * https://doi.org/10.1142/2975.
 */
struct MOData
{
    enum class ThetaCalcType : std::uint8_t {
        HEAT_FLUX = 0,      ///< Heat-flux specified
        SURFACE_TEMPERATURE ///< Surface temperature specified
    };

    amrex::Real zref{0.0_rt};           ///< Reference height (m)
    amrex::Real z0{0.1_rt};             ///< Aerodynamic roughness height (m)
    amrex::Real z0t{z0};                ///< Thermal roughness height (m)
    amrex::Real utau;                   ///< Friction velocity (m/s)
    amrex::Real kappa{0.41_rt};         ///< von Karman constant
    amrex::Real gravity{9.81_rt};       ///< Acceleration due to gravity (m/s^2)
    amrex::Real L{1.0e16_rt};           ///< Non-dimensional Obukhov length
    amrex::Real L_neutral{1.0e6_rt};    ///< Neutral Obukhov length threshold

    amrex::RealArray vel_mean; ///< Mean velocity (at zref)
    amrex::Real vmag_mean;     ///< Mean wind speed (at zref)
    amrex::Real Su_mean;       ///< Mean wind speed times x-velocity (at zref)
    amrex::Real Sv_mean;       ///< Mean wind speed times y-velocity (at zref)
    amrex::Real vmean_mag{0.0_rt};  ///< Magnitude of mean velocity (at zref)
    amrex::Real theta_mean;    ///< Mean potential temperature
    amrex::Real Stheta_mean;    ///< Mean wind speed times potential temperature (at zref)

    amrex::Real surf_temp_flux{0.0_rt}; ///< Heat flux
    amrex::Real surf_temp;              ///< Instantaneous surface temperature

    amrex::Real alpha_m{1.0_rt};
    amrex::Real alpha_h{0.95_rt};
    amrex::Real gamma_m{5.0_rt};
    amrex::Real gamma_h{5.0_rt};
    amrex::Real beta_m{19.0_rt};
    amrex::Real beta_h{11.6_rt};
    amrex::Real tol{1.0e-12_rt};    ///< Tolerance for iterative calculations

    ThetaCalcType alg_type{ThetaCalcType::HEAT_FLUX};

    bool use_mean_vel_mag{false}; ///< utau from |<u>| (true) or <|u|> (false)

    [[nodiscard]] amrex::Real ref_speed() const
    {
        return use_mean_vel_mag ? vmean_mag : vmag_mean;
    }

    [[nodiscard]] amrex::Real phi_h(amrex::Real z) const;
    [[nodiscard]] amrex::Real phi_h() const;
    [[nodiscard]] amrex::Real psi_m(amrex::Real z) const;
    [[nodiscard]] amrex::Real psi_h(amrex::Real z) const;

    // Stateless forms for callers that supply their own L, beta and gamma
    [[nodiscard]] static amrex::Real
    calc_psi_m(amrex::Real zeta, amrex::Real beta_m, amrex::Real gamma_m);
    [[nodiscard]] static amrex::Real
    calc_psi_h(amrex::Real zeta, amrex::Real beta_h, amrex::Real gamma_h);

    void update_L();
    void update_utau();
    void update_surf_temp();
    void update_surf_temp_flux();
    void update_fluxes(int max_iters = 25);
};

} // namespace kynema_sgf

#endif /* MODATA_H */
