#ifndef WALLMODELTERRAINFORCING_H
#define WALLMODELTERRAINFORCING_H

#include "src/equation_systems/icns/MomentumSource.H"
#include "src/core/SimTime.H"
#include "src/CFDSim.H"
#include "src/wind_energy/MOData.H"
#include "src/wind_energy/ShearStress.H"
#include "AMReX_REAL.H"
#include <string>

using namespace amrex::literals;

namespace kynema_sgf::pde::icns {

/** Drag coefficient selection method for WallModelTerrainForcing */
enum class DragCoefficientMethod {
    SpatialTemporal, // min((cd_m/dx)*velmag, cd_factor/dt)  [default]
    Timestep,        // cd_factor/dt
    LinearQuadratic  // DragForcing.cpp implementation
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
    const SimTime& m_time;
    const CFDSim& m_sim;
    const amrex::AmrCore& m_mesh;
    const Field& m_velocity;
    const MOData* m_mo;

    DragCoefficientMethod m_method{DragCoefficientMethod::SpatialTemporal};
    amrex::Real m_cd_m{10.0_rt};       // shared: SpatialTemporal, LinearQuadratic
    amrex::Real m_cd_factor{0.01_rt};   // shared: SpatialTemporal, Timestep
    amrex::Real m_cd_max{1000.0_rt};   // LinearQuadratic only
    bool m_limit_drag{true};           // LinearQuadratic only

    Field& m_target_velocity;
};

} // namespace kynema_sgf::pde::icns

#endif /* WALLMODELTERRAINFORCING_H */