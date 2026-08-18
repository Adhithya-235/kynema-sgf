#include "src/equation_systems/icns/source_terms/WallModelTerrainForcing.H"
#include "AMReX_ParmParse.H"
#include "src/wind_energy/ABL.H"
#include "src/physics/TerrainDrag.H"
#include "src/utilities/constants.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::icns {

// WALLMODELTERRAINFORCING CONSTRUCTOR

WallModelTerrainForcing::WallModelTerrainForcing(const CFDSim& sim)
    : m_time(sim.time())
    , m_sim(sim)
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_target_velocity(sim.repo().declare_field("target_velocity", 3, 1, 2))
    , m_surface_stress(sim.repo().declare_field("surface_stress", 2, 1, 2))
    , m_mo(nullptr)
{
    // CHECK FOR ABL PHYSICS AND MO DATA

    if (!m_sim.physics_manager().contains("ABL")) {
        amrex::Abort("WallModelTerrainForcing: ABL physics not found. "
                     "ABL physics with MOData is required.");
    }
    auto& abl = m_sim.physics_manager().get<kynema_sgf::ABL>();
    m_mo = &abl.abl_wall_function().mo();

    // CHECK FOR TERRAIN BLANKING FIELD 

    if (!m_sim.repo().int_field_exists("terrain_blank")) {
        amrex::Abort("WallModelTerrainForcing: terrain_blank field not found. "
                     "TerrainDrag physics must be enabled.");
    }

    // READ PREFERRED DRAG COEFFICIENT METHOD AND PARAMETERS

    amrex::ParmParse pp("WallModelTerrainForcing");
    std::string method_str = "linear_quadratic";
    pp.query("drag_coefficient_method", method_str);
    if (method_str == "timestep") {
        m_method = DragCoefficientMethod::Timestep;
    } else if (method_str == "linear_quadratic") {
        m_method = DragCoefficientMethod::LinearQuadratic;
    } else {
        amrex::Print() << "WARNING: WallModelTerrainForcing: unknown "
                       << "drag_coefficient_method '" << method_str
                       << "', defaulting to 'linear_quadratic'\n";
        m_method = DragCoefficientMethod::LinearQuadratic;
    }
    pp.query("cd_m", m_cd_m);
    pp.query("cd_factor", m_cd_factor);
    pp.query("cd_max", m_cd_max);
    pp.query("limit_drag", m_limit_drag);
    pp.query("feedback_damping", m_feedback_damping);

    // READ PREFERRED SHEAR STRESS MODEL AND PARAMETERS

    std::string shear_stress_str = "moeng";
    pp.query("wall_shear_stress_type", shear_stress_str);
    shear_stress_str = amrex::toLower(shear_stress_str);
    if (shear_stress_str == "constant") {
        m_shear_stress_type = ShearStressModel::Constant;
    } else if (shear_stress_str == "local") {
        m_shear_stress_type = ShearStressModel::Local;
    } else if (shear_stress_str == "schumann") {
        m_shear_stress_type = ShearStressModel::Schumann;
    } else if (shear_stress_str == "moeng") {
        m_shear_stress_type = ShearStressModel::Moeng;
    } else if (shear_stress_str == "donelan") {
        m_shear_stress_type = ShearStressModel::Donelan;
    } else if (shear_stress_str == "local_moeng") {
        m_shear_stress_type = ShearStressModel::LocalMoeng;
    } else {
        amrex::Print() << "WARNING: WallModelTerrainForcing: unknown "
                       << "wall_shear_stress_type '" << shear_stress_str
                       << "', defaulting to 'moeng'\n";
        shear_stress_str = "moeng";
        m_shear_stress_type = ShearStressModel::Moeng;
    }
    m_kappa_local = m_mo->kappa;
    m_z0_local = m_mo->z0;
    pp.query("kappa", m_kappa_local);
    pp.query("z0", m_z0_local);

    // INITIALIZATION LOG

    amrex::Print() << "WallModelTerrainForcing initialized successfully\n"
                   << "  Method: " << method_str << '\n'
                   << "  wall_shear_stress_type = " << shear_stress_str << '\n';
    if (m_method == DragCoefficientMethod::Timestep) {
        amrex::Print() << "  cd_factor = " << m_cd_factor << '\n';
    } else if (m_method == DragCoefficientMethod::LinearQuadratic) {
        amrex::Print() << "  cd_m = " << m_cd_m << '\n'
                       << "  cd_max = " << m_cd_max << '\n'
                       << "  limit_drag = " << (m_limit_drag ? "yes" : "no") << '\n';
    }
    if (m_shear_stress_type == ShearStressModel::LocalMoeng) {
        amrex::Print() << "  kappa = " << m_kappa_local << '\n'
                       << "  z0 = " << m_z0_local << '\n';
    }
    amrex::Print() << "  feedback_damping = " << m_feedback_damping << '\n';
}

// WALLMODELTERRAINFORCING DESTRUCTOR

WallModelTerrainForcing::~WallModelTerrainForcing() = default;

// WALLMODELTERRAINFORCING OPERATOR

void WallModelTerrainForcing::operator()(
    const int lev, const FieldState fstate, amrex::MultiFab& src_term) const
{   
    // ADVANCE TARGET VELOCITY IF TIME HAS ADVANCED

    if (lev == 0 && m_time.current_time() > m_last_advance_time) {
        m_target_velocity.advance_states();
        m_surface_stress.advance_states();
        m_last_advance_time = m_time.current_time();
    }

    // APPLY WALL MODEL TERRAIN FORCING BASED ON SELECTED SHEAR STRESS MODEL

    const MOData mo = *m_mo;
    const auto& geom = m_mesh.Geom(lev);
    const auto& dx = geom.CellSizeArray();
    switch (m_shear_stress_type) {
    case ShearStressModel::Constant:
        apply_forcing(lev, fstate, src_term, dx, ShearStressConstant(mo));
        break;
    case ShearStressModel::Local:
        apply_forcing(lev, fstate, src_term, dx, ShearStressLocal(mo));
        break;
    case ShearStressModel::Schumann:
        apply_forcing(lev, fstate, src_term, dx, ShearStressSchumann(mo));
        break;
    case ShearStressModel::Moeng:
        apply_forcing(lev, fstate, src_term, dx, ShearStressMoeng(mo));
        break;
    case ShearStressModel::Donelan:
        apply_forcing(lev, fstate, src_term, dx, ShearStressDonelan(mo));
        break;
    case ShearStressModel::LocalMoeng:
        apply_forcing(
            lev, fstate, src_term, dx,
            ShearStressLocalMoeng(m_kappa_local, m_z0_local));
        break;
    }
}

// SOURCE TERM AGGREGATOR

template <typename ShearStressType>
void WallModelTerrainForcing::apply_forcing(
    const int lev, const FieldState fstate, amrex::MultiFab& src_term,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
    const ShearStressType& tau) const
{
    // POPULATE I/O ARRAYS

    auto const& vel_arrs = m_velocity.state(field_impl::dof_state(fstate))(lev).const_arrays();
    auto const& target_vel_old_arrs = m_target_velocity.state(FieldState::Old)(lev).const_arrays();
    auto const& src_arrs = src_term.arrays();
    auto const& blank_arrs = m_sim.repo().get_int_field("terrain_blank")(lev).const_arrays();
    auto const& rho_arrs = m_sim.repo().get_field("density")(lev).const_arrays();
    auto const& visc_arrs = m_sim.repo().get_field("velocity_mueff")(lev).const_arrays();
    auto const& prob_lo = m_mesh.Geom(lev).ProbLoArray();

    // DRAG COEFFICIENT PARAMETERS

    auto const& dtc = m_time.max_cfl()/m_time.conv_cfl_unit(); // Convert CFL to convective time step size
    const amrex::Real lq_cd_m = (m_limit_drag && dx[2] < 1.0_rt) ? m_cd_m : m_cd_m / dx[2];
    const amrex::Real lq_cd_max = m_limit_drag ? m_cd_max : kynema_sgf::constants::LARGE_NUM;
    const amrex::Real lq_scale_factor = (m_limit_drag && dx[2] < 1.0_rt) ? 1.0_rt : 1.0_rt / dx[2];
    const DragCoefficientMethod method = m_method;
    const amrex::Real cd_factor = m_cd_factor;
    const amrex::Real feedback_damping = m_feedback_damping;

    // INITIALIZE TARGET VELOCITY AND SURFACE STRESS ARRAYS

    m_target_velocity(lev).setVal(0.0_rt);
    m_surface_stress(lev).setVal(0.0_rt);
    auto target_vel_arrs = m_target_velocity(lev).arrays();
    auto surface_stress_arrs = m_surface_stress(lev).arrays();

    // PARALLEL FOR LOOP OVER ALL CELLS

    amrex::ParallelFor(
        src_term, amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n){

            // SKIP FLUID CELLS

            if (blank_arrs[nbx](i, j, k, 0) == 0) {
                return;
            }

            // INITIALIZE TARGET VELOCITY (0 FOR DEEP TERRAIN CELLS)

            amrex::Real target_u = 0.0_rt;
            amrex::Real target_v = 0.0_rt;
            amrex::Real target_w = 0.0_rt;

            // GET CURRENT VELOCITY AT TERRAIN CELL

            const amrex::Real u_k = vel_arrs[nbx](i, j, k, 0);
            const amrex::Real v_k = vel_arrs[nbx](i, j, k, 1);
            const amrex::Real w_k = vel_arrs[nbx](i, j, k, 2);

            // IF AT A SURFACE ADJACENT TERRAIN CELL

            if (k+1 >= 0 && blank_arrs[nbx](i, j, k+1, 0) == 0) {

                // GET VELOCITY, DENSITY, INTERFACIAL EFFECTIVE VISCOSITY AT FLUID CELL ABOVE TERRAIN CELL

                const amrex::Real uold1 = vel_arrs[nbx](i, j, k+1, 0);
                const amrex::Real vold1 = vel_arrs[nbx](i, j, k+1, 1);
                const amrex::Real wold1 = vel_arrs[nbx](i, j, k+1, 2);
                const amrex::Real dens1 = rho_arrs[nbx](i, j, k+1);
                const amrex::Real visc1 = 0.5_rt*(visc_arrs[nbx](i, j, k+1) + visc_arrs[nbx](i, j, k));
                const amrex::Real wspd = std::sqrt(uold1*uold1 + vold1*vold1);
                
                // GET OLD TARGET VELOCITY AT TERRAIN CELL

                const amrex::Real u_target_old = target_vel_old_arrs[nbx](i, j, k, 0);
                const amrex::Real v_target_old = target_vel_old_arrs[nbx](i, j, k, 1);
                const amrex::Real w_target_old = target_vel_old_arrs[nbx](i, j, k, 2);
                
                // GET MODELLED VELOCITY GRADIENT AT TERRAIN SURFACE

                amrex::Real tau_x = 0.0_rt;
                amrex::Real tau_y = 0.0_rt;
                if constexpr (std::is_same_v<ShearStressType, ShearStressLocalMoeng>) {
                    const amrex::Real z_kp1 = prob_lo[2] + (k + 1 + 0.5_rt) * dx[2];
                    tau_x = tau.calc_vel_x(uold1, wspd, z_kp1);
                    tau_y = tau.calc_vel_y(vold1, wspd, z_kp1);
                } else {
                    tau_x = tau.calc_vel_x(uold1, wspd);
                    tau_y = tau.calc_vel_y(vold1, wspd);
                }
                const amrex::Real dudz = tau_x * dens1 / (visc1);
                const amrex::Real dvdz = tau_y * dens1 / (visc1);

                // COMPUTE TARGET VELOCITY AT TERRAIN CELL USING WALL-MODELLED GRADIENT AND FEEDBACK DAMPING

                target_u = (uold1 - dx[2] * dudz) + feedback_damping * (u_target_old - u_k);
                target_v = (vold1 - dx[2] * dvdz) + feedback_damping * (v_target_old - v_k);
                target_w = -wold1 + feedback_damping * (w_target_old - w_k);

                // STORE SURFACE STRESS FOR POST-PROCESSING

                surface_stress_arrs[nbx](i, j, k, 0) = tau_x;
                surface_stress_arrs[nbx](i, j, k, 1) = tau_y;
            }

            // ASSIGN TARGET VELOCITY TO TARGET VELOCITY FIELD

            const amrex::Real target_vel = (n == 0) ? target_u : (n == 1) ? target_v : target_w;
            target_vel_arrs[nbx](i, j, k, n) = target_vel;

            // COMPUTE DRAG COEFFICIENT BASED ON SELECTED METHOD

            amrex::Real CdM_m = 0.0_rt;
            switch (method) {
                case DragCoefficientMethod::Timestep: {
                    CdM_m = cd_factor / dtc;
                    break;
                }
                case DragCoefficientMethod::LinearQuadratic: {
                    const amrex::Real du = u_k - target_u;
                    const amrex::Real dv = v_k - target_v;
                    const amrex::Real dw = w_k - target_w;
                    const amrex::Real velmag = std::sqrt(du*du + dv*dv + dw*dw);
                    const amrex::Real CdM = amrex::min<amrex::Real>(lq_cd_m / (velmag + kynema_sgf::constants::EPS), lq_cd_max / lq_scale_factor);
                    CdM_m = CdM * velmag;
                    break;
                }
            }

            // AGGREGATE SOURCE TERM FOR MOMENTUM EQUATION

            const amrex::Real vel_n = vel_arrs[nbx](i, j, k, n);
            src_arrs[nbx](i, j, k, n) -= CdM_m * (vel_n - target_vel);

        }
    );

    // FILL BOUNDARY FOR TARGET VELOCITY AND SURFACE STRESS

    m_target_velocity(lev).FillBoundary(m_mesh.Geom(lev).periodicity());
    m_surface_stress(lev).FillBoundary(m_mesh.Geom(lev).periodicity());

}
} // namespace kynema_sgf::pde::icns