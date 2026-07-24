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
    std::string method_str = "spatial_temporal";
    pp.query("drag_coefficient_method", method_str);
    if (method_str == "spatial_temporal") {
        m_method = DragCoefficientMethod::SpatialTemporal;
    } else if (method_str == "timestep") {
        m_method = DragCoefficientMethod::Timestep;
    } else if (method_str == "linear_quadratic") {
        m_method = DragCoefficientMethod::LinearQuadratic;
    } else {
        amrex::Print() << "WARNING: WallModelTerrainForcing: unknown "
                       << "drag_coefficient_method '" << method_str
                       << "', defaulting to 'spatial_temporal'\n";
        m_method = DragCoefficientMethod::SpatialTemporal;
    }
    pp.query("cd_m", m_cd_m);
    pp.query("cd_factor", m_cd_factor);
    pp.query("cd_max", m_cd_max);
    pp.query("limit_drag", m_limit_drag);

    amrex::Print() << "WallModelTerrainForcing initialized successfully\n"
                   << "  Method: " << method_str << '\n';
    if (m_method == DragCoefficientMethod::SpatialTemporal) {
        amrex::Print() << "  cd_m = " << m_cd_m << '\n'
                       << "  cd_factor = " << m_cd_factor << '\n';
    } else if (m_method == DragCoefficientMethod::Timestep) {
        amrex::Print() << "  cd_factor = " << m_cd_factor << '\n';
    } else if (m_method == DragCoefficientMethod::LinearQuadratic) {
        amrex::Print() << "  cd_m = " << m_cd_m << '\n'
                       << "  cd_max = " << m_cd_max << '\n'
                       << "  limit_drag = " << (m_limit_drag ? "yes" : "no") << '\n';
    }
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
    const MOData mo = *m_mo;
    const amrex::Real lq_cd_m = (m_limit_drag && dx[2] < 1.0_rt) ? m_cd_m : m_cd_m / dx[2];
    const amrex::Real lq_cd_max = m_limit_drag ? m_cd_max : kynema_sgf::constants::LARGE_NUM;
    const amrex::Real lq_scale_factor = (m_limit_drag && dx[2] < 1.0_rt) ? 1.0_rt : 1.0_rt / dx[2];
    const DragCoefficientMethod method = m_method;
    const amrex::Real cd_m = m_cd_m;
    const amrex::Real cd_factor = m_cd_factor;
    m_target_velocity.setVal(0.0_rt, lev, 0, 3);
    auto target_vel_arrs = m_target_velocity(lev).arrays();

    amrex::ParallelFor(
        src_term, amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n){

            if (blank_arrs[nbx](i, j, k, 0) == 0) {
                return;
            }

            amrex::Real target_vel = 0.0_rt;
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
                if (n == 0) {
                    target_vel = uold1 - dx[2] * dudz;
                } else if (n == 1) {
                    target_vel = vold1 - dx[2] * dvdz;
                } else {
                    target_vel = -wold1;
                }
            }
            target_vel_arrs[nbx](i, j, k, n) = target_vel;

            const amrex::Real u_k = vel_arrs[nbx](i, j, k, 0);
            const amrex::Real v_k = vel_arrs[nbx](i, j, k, 1);
            const amrex::Real w_k = vel_arrs[nbx](i, j, k, 2);
            const amrex::Real velmag = std::sqrt(u_k*u_k + v_k*v_k + w_k*w_k);

            amrex::Real CdM_m = 0.0_rt;
            switch (method) {
                case DragCoefficientMethod::SpatialTemporal: {
                    const amrex::Real spatial_rate  = (cd_m / dx[2]) * velmag;
                    const amrex::Real temporal_rate = cd_factor / dt;
                    CdM_m = amrex::min<amrex::Real>(spatial_rate, temporal_rate);
                    break;
                }
                case DragCoefficientMethod::Timestep: {
                    CdM_m = cd_factor / dt;
                    break;
                }
                case DragCoefficientMethod::LinearQuadratic: {
                    const amrex::Real CdM = amrex::min<amrex::Real>(
                        lq_cd_m / (velmag + kynema_sgf::constants::EPS),
                        lq_cd_max / lq_scale_factor);
                    CdM_m = CdM * velmag;
                    break;
                }
            }

            const amrex::Real vel_n = vel_arrs[nbx](i, j, k, n);
            src_arrs[nbx](i, j, k, n) -= CdM_m * (vel_n - target_vel);

        }
    );

    m_target_velocity(lev).FillBoundary(m_mesh.Geom(lev).periodicity());

}
} // namespace kynema_sgf::pde::icns