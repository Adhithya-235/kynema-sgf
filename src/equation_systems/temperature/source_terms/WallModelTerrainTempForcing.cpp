#include "src/equation_systems/temperature/source_terms/WallModelTerrainTempForcing.H"
#include "src/wind_energy/ABL.H"
#include "src/physics/TerrainDrag.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::temperature {

WallModelTerrainTempForcing::WallModelTerrainTempForcing(const CFDSim& sim)
    : m_sim(sim)
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_temperature(sim.repo().get_field("temperature"))
    , m_mo(nullptr)
{
    amrex::ParmParse pp("WallModelTerrainTempForcing");
    pp.query("cd_t", m_cd_t);
    pp.query("cd_max", m_cd_max);
    pp.query("soil_temperature", m_soil_temperature);
    if (!m_sim.physics_manager().contains("ABL")) {
        amrex::Abort("WallModelTerrainTempForcing: ABL physics not found. "
                     "ABL physics with MOData is required.");
    }
    auto& abl = m_sim.physics_manager().get<kynema_sgf::ABL>();
    m_mo = &abl.abl_wall_function().mo();
    if (!m_sim.repo().int_field_exists("terrain_blank")) {
        amrex::Abort("WallModelTerrainTempForcing: terrain_blank field not found. "
                     "TerrainDrag physics must be enabled.");
    }
    amrex::Print() << "WallModelTerrainTempForcing initialized successfully\n"
                   << "  cd_t = " << m_cd_t << '\n'
                   << "  cd_max = " << m_cd_max << '\n'
                   << "  soil_temperature = " << m_soil_temperature << '\n';
}

WallModelTerrainTempForcing::~WallModelTerrainTempForcing() = default;

void WallModelTerrainTempForcing::operator()(
    const int lev, const FieldState fstate, amrex::MultiFab& src_term) const
{
    auto const& vel_arrs = m_velocity.state(field_impl::dof_state(fstate))(lev).const_arrays();
    auto const& temp_arrs = m_temperature.state(field_impl::dof_state(fstate))(lev).const_arrays();
    auto const& src_arrs = src_term.arrays();
    auto const& blank_arrs = m_sim.repo().get_int_field("terrain_blank")(lev).const_arrays();
    auto const& rho_arrs = m_sim.repo().get_field("density")(lev).const_arrays();
    auto const& diff_arrs = m_sim.repo().get_field("temperature_mueff")(lev).const_arrays();
    auto const& geom = m_mesh.Geom(lev);
    auto const& dx = geom.CellSizeArray();
    const auto tiny = std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real cd_t = m_cd_t / dx[2];
    const amrex::Real cd_max = m_cd_max;
    const amrex::Real T0 = m_soil_temperature;
    const MOData mo = *m_mo;

    amrex::ParallelFor(
        src_term, amrex::IntVect(0), 1,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int){

            if (blank_arrs[nbx](i, j, k, 0) == 0) {
                return;
            }
            amrex::Real target_temp = 0.0_rt;
            amrex::Real velmag = 0.0_rt;

            if (k+1 >= 0 && blank_arrs[nbx](i, j, k+1, 0) == 0) {
                const amrex::Real uold1 = vel_arrs[nbx](i, j, k+1, 0);
                const amrex::Real vold1 = vel_arrs[nbx](i, j, k+1, 1);
                const amrex::Real wold1 = vel_arrs[nbx](i, j, k+1, 2);
                const amrex::Real thetaold1 = temp_arrs[nbx](i, j, k+1, 0);
                const amrex::Real dens1 = rho_arrs[nbx](i, j, k+1);
                const amrex::Real alph1 = diff_arrs[nbx](i, j, k+1);
                const auto tau = ShearStressMoeng(mo);
                const amrex::Real wspd = std::sqrt(uold1*uold1 + vold1*vold1); 
                velmag = std::sqrt(uold1*uold1 + vold1*vold1 + wold1*wold1);
                const amrex::Real dthetadz = tau.calc_theta(wspd, thetaold1) * dens1 / alph1;
                target_temp = thetaold1 - dx[2] * dthetadz;
            } else {
                target_temp = T0;
            }
            
            const amrex::Real CdT = amrex::min<amrex::Real>(cd_t/(velmag + tiny), cd_max / dx[2]);
            const amrex::Real temp_n = temp_arrs[nbx](i, j, k, 0);
            src_arrs[nbx](i, j, k, 0) -= CdT * (temp_n - target_temp);
        }
    );
}
} // namespace kynema_sgf::pde::temperature