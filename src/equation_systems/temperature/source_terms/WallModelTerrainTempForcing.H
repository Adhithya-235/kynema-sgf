#ifndef WALLMODELTERRAINTEMPFORCING_H
#define WALLMODELTERRAINTEMPFORCING_H

#include "src/equation_systems/temperature/TemperatureSource.H"
#include "src/CFDSim.H"
#include "src/wind_energy/MOData.H"
#include "src/wind_energy/ShearStress.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::temperature {

/** Wall model based terrain temperature forcing source term 
 * 
 *  Computes target temperature at fluid-adjacent terrain cells using heat flux
 *  computed from the moeng wall model. The terrain surface should then have the 
 *  wall-modelled heat flux.
 * 
 *  Schema: 
 *  - Fluid-adjacent terrain cells: extrapolate temperature using wall-modelled gradient
 *  - Deep terrain cells: force temperature to soil temperature
 *  - Fluid cells: no forcing 
 * 
 *  \ingroup temperature_src
*/ 
class WallModelTerrainTempForcing : public TemperatureSource::Register<WallModelTerrainTempForcing>
{
public:
    static std::string identifier() { return "WallModelTerrainTempForcing"; }

    explicit WallModelTerrainTempForcing(const CFDSim& sim);

    ~WallModelTerrainTempForcing() override;

    void operator()(
        int lev, FieldState fstate, amrex::MultiFab& src_term) const override;    

private:
    const CFDSim& m_sim;
    const amrex::AmrCore& m_mesh;
    const Field& m_velocity;
    const Field& m_temperature;
    const MOData* m_mo;
    amrex::Real m_cd_t{1.0_rt};
    amrex::Real m_cd_max{10.0_rt};
    amrex::Real m_soil_temperature{300.0_rt};
};

} // namespace kynema_sgf::pde::temperature

#endif /* WALLMODELTERRAINTEMPFORCING_H */