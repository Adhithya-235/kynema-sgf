#ifndef WALLMODELTERRAINFORCING_H
#define WALLMODELTERRAINFORCING_H

#include "src/equation_systems/icns/MomentumSource.H"
#include "src/core/SimTime.H"
#include "src/CFDSim.H"
#include "src/wind_energy/MOData.H"
#include "src/wind_energy/ShearStress.H"
#include "AMReX_REAL.H"
#include <string>
#include <limits>
#include <type_traits>

using namespace amrex::literals;

namespace kynema_sgf::pde::icns {

/** Drag coefficient selection method for WallModelTerrainForcing */
enum class DragCoefficientMethod {
    Timestep,        // cd_factor/dt
    LinearQuadratic  // DragForcing.cpp implementation  [default]
};

/** Wall shear-stress closure selection for WallModelTerrainForcing */
enum class ShearStressModel {
    Constant,
    Local,
    Schumann,
    Moeng,
    Donelan,
    LocalMoeng
};

/** Local, instantaneous neutral log-law wall stress.
 *
 *  Unlike the ShearStress.H models, ustar is computed per-cell from the local
 *  horizontal velocity, z0, kappa, and height above the wall, rather than from
 *  the ABL-wide MOData::utau.
 */
struct ShearStressLocalMoeng
{
    explicit ShearStressLocalMoeng(amrex::Real kappa, amrex::Real z0)
        : m_kappa(kappa), m_z0(z0)
    {}

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    calc_vel_x(amrex::Real u, amrex::Real wspd, amrex::Real height) const
    {
        return calc_coeff2(height) * wspd * u;
    };

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    calc_vel_y(amrex::Real v, amrex::Real wspd, amrex::Real height) const
    {
        return calc_coeff2(height) * wspd * v;
    };

    amrex::Real m_kappa;
    amrex::Real m_z0;

private:
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
    calc_coeff2(amrex::Real height) const
    {
        const amrex::Real c = m_kappa / std::log(height / m_z0);
        return c * c;
    }
};

/** Wall model based terrain forcing source term 
 * 
 *  Computes target velocity at fluid-adjacent terrain cells using shear stress
 *  computed from the moeng wall model. The terrain surface should then have the 
 *  wall-modelled shear stress.
 * 
 *  Schema: 
 *  - Fluid-adjacent terrain cells: extrapolate velocity using wall-modelled gradient
 *  - Deep terrain cells: force velocity to zero
 *  - Fluid cells: no forcing 
 * 
 *  \ingroup icns_src
*/ 
class WallModelTerrainForcing : public MomentumSource::Register<WallModelTerrainForcing>
{
public:
    static std::string identifier() { return "WallModelTerrainForcing"; }

    explicit WallModelTerrainForcing(const CFDSim& sim);

    ~WallModelTerrainForcing() override;

    void operator()(
        int lev, FieldState fstate, amrex::MultiFab& src_term) const override;    

private:
    template <typename ShearStressType>
    void apply_forcing(
        int lev, FieldState fstate, amrex::MultiFab& src_term,
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
        const ShearStressType& tau) const;

    const SimTime& m_time;
    const CFDSim& m_sim;
    const amrex::AmrCore& m_mesh;
    const Field& m_velocity;
    const MOData* m_mo;

    DragCoefficientMethod m_method{DragCoefficientMethod::LinearQuadratic};
    amrex::Real m_cd_m{10.0_rt};       // LinearQuadratic only
    amrex::Real m_cd_factor{1.0_rt};   // Timestep only
    amrex::Real m_cd_max{1000.0_rt};   // LinearQuadratic only
    bool m_limit_drag{true};           // LinearQuadratic only
    amrex::Real m_feedback_damping{0.02_rt}; // scales (target_old - u) feedback term

    ShearStressModel m_shear_stress_type{ShearStressModel::Moeng};
    amrex::Real m_kappa_local{0.41_rt}; // LocalMoeng only, defaulted from mo.kappa
    amrex::Real m_z0_local{0.1_rt};     // LocalMoeng only, defaulted from mo.z0

    bool m_clamp_xhi_backflow{true};          // clamp negative target_u near an xhi mass_inflow_outflow BC
    amrex::Real m_xhi_backflow_buffer{250.0_rt}; // distance [m] from xhi over which the clamp applies

    Field& m_target_velocity;
    Field& m_surface_stress;
    mutable amrex::Real m_last_advance_time{-std::numeric_limits<amrex::Real>::max()};
};

} // namespace kynema_sgf::pde::icns

#endif /* WALLMODELTERRAINFORCING_H */