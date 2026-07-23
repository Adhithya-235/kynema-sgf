#ifndef WALLMODELTERRAINFORCING_H
#define WALLMODELTERRAINFORCING_H

#include "src/equation_systems/icns/MomentumSource.H"
#include "src/core/SimTime.H"
#include "src/CFDSim.H"
#include "src/wind_energy/MOData.H"
#include "src/wind_energy/ShearStress.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::icns {

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
    amrex::Real m_cd_m{2.0_rt};
    amrex::Real m_cdm_factor{0.2_rt};
    Field& m_target_velocity;
};

} // namespace kynema_sgf::pde::icns

#endif /* WALLMODELTERRAINFORCING_H */