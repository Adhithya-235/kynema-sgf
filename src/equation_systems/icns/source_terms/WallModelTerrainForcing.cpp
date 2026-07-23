#include "src/equation_systems/icns/source_terms/WallModelTerrainForcing.H"
#include "AMReX_ParmParse.H"
#include "src/wind_energy/ABL.H"
#include "src/physics/TerrainDrag.H"
#include "src/utilities/constants.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::icns {

WallModelTerrainForcing::WallModelTerrainForcing(const CFDSim& sim)
    : m_time(sim.time())
    , m_sim(sim)
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_target_velocity(sim.repo().declare_field("target_velocity", 3, 1, 1))
    , m_mo(nullptr)
{
    if (!m_sim.physics_manager().contains("ABL")) {
        amrex::Abort("WallModelTerrainForcing: ABL physics not found. "
                     "ABL physics with MOData is required.");
    }
    auto& abl = m_sim.physics_manager().get<kynema_sgf::ABL>();
    m_mo = &abl.abl_wall_function().mo();
    if (!m_sim.repo().int_field_exists("terrain_blank")) {
        amrex::Abort("WallModelTerrainForcing: terrain_blank field not found. "
                     "TerrainDrag physics must be enabled.");
    }
    amrex::ParmParse pp("WallModelTerrainForcing");
    pp.query("cd_m", m_cd_m);
    pp.query("cdm_factor", m_cdm_factor);
    amrex::Print() << "WallModelTerrainForcing initialized successfully\n"
                   << "  cd_m = " << m_cd_m << '\n'
                   << "  cdm_factor = " << m_cdm_factor << '\n';
}

WallModelTerrainForcing::~WallModelTerrainForcing() = default;

void WallModelTerrainForcing::operator()(
    const int lev, const FieldState fstate, amrex::MultiFab& src_term) const
{
    auto const& vel_arrs = m_velocity.state(field_impl::dof_state(fstate))(lev).const_arrays();
    auto const& src_arrs = src_term.arrays();
    auto const& blank_arrs = m_sim.repo().get_int_field("terrain_blank")(lev).const_arrays();
    auto const& rho_arrs = m_sim.repo().get_field("density")(lev).const_arrays();
    auto const& visc_arrs = m_sim.repo().get_field("velocity_mueff")(lev).const_arrays();
    auto const& geom = m_mesh.Geom(lev);
    auto const& dx = geom.CellSizeArray();
    auto const& dt = m_time.delta_t();
    m_target_velocity.setVal(0.0_rt, lev, 0, 3);
    auto target_vel_arrs = m_target_velocity(lev).arrays();
    const MOData mo = *m_mo;

    amrex::ParallelFor(
        src_term, amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n){

            if (blank_arrs[nbx](i, j, k, 0) == 0) {
                return;
            }

            amrex::Real u_target = 0.0_rt;
            amrex::Real v_target = 0.0_rt;
            amrex::Real w_target = 0.0_rt;
            if (k+1 >= 0 && blank_arrs[nbx](i, j, k+1, 0) == 0) {
                const amrex::Real uold1 = vel_arrs[nbx](i, j, k+1, 0);
                const amrex::Real vold1 = vel_arrs[nbx](i, j, k+1, 1);
                const amrex::Real wold1 = vel_arrs[nbx](i, j, k+1, 2);
                const amrex::Real dens1 = rho_arrs[nbx](i, j, k+1);
                const amrex::Real visc1 = 0.5_rt*(visc_arrs[nbx](i, j, k+1) + visc_arrs[nbx](i, j, k));
                const auto tau = ShearStressMoeng(mo);
                const amrex::Real wspd = std::sqrt(uold1*uold1 + vold1*vold1); 
                const amrex::Real dudz = tau.calc_vel_x(uold1, wspd) * dens1 / (2*visc1);
                const amrex::Real dvdz = tau.calc_vel_y(vold1, wspd) * dens1 / (2*visc1);
                u_target = uold1 - dx[2] * dudz;
                v_target = vold1 - dx[2] * dvdz;
                w_target = -wold1;
            } 
            amrex::Real target_vel = 0.0_rt;
            if (n == 0) {
                target_vel = u_target;
            } else if (n == 1) {
                target_vel = v_target;
            } else {
                target_vel = w_target;
            }
            target_vel_arrs[nbx](i, j, k, n) = target_vel;

            const amrex::Real u_rel = vel_arrs[nbx](i, j, k, 0) - u_target;
            const amrex::Real v_rel = vel_arrs[nbx](i, j, k, 1) - v_target;
            const amrex::Real w_rel = vel_arrs[nbx](i, j, k, 2) - w_target;
            const amrex::Real velmag_rel = std::sqrt(u_rel*u_rel + v_rel*v_rel + w_rel*w_rel);
            const amrex::Real spatial_drag_rate = (m_cd_m/dx[2]) * velmag_rel;
            const amrex::Real temporal_safety_rate = m_cdm_factor/dt;
            const amrex::Real CdM_m = amrex::min<amrex::Real>(spatial_drag_rate, temporal_safety_rate);
            const amrex::Real vel_n = vel_arrs[nbx](i, j, k, n);
            src_arrs[nbx](i, j, k, n) -= CdM_m * (vel_n - target_vel);

        }
    );

    m_target_velocity(lev).FillBoundary(m_mesh.Geom(lev).periodicity());
    
}
} // namespace kynema_sgf::pde::icns