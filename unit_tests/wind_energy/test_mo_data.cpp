#include "gtest/gtest.h"
#include <array>
#include <cmath>
#include <limits>
#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "src/utilities/trig_ops.H"
#include "src/wind_energy/MOData.H"
#include "src/wind_energy/ShearStress.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {

namespace {

constexpr amrex::Real tol =
    std::numeric_limits<amrex::Real>::epsilon() * 1.0e4_rt;

kynema_sgf::MOData make_mo()
{
    kynema_sgf::MOData mo;
    mo.zref = 5.0_rt;
    mo.z0 = 0.1_rt;
    mo.z0t = 0.1_rt;
    mo.vel_mean = {3.0_rt, 4.0_rt, 0.0_rt};
    mo.vmean_mag = 5.0_rt;
    mo.vmag_mean = 5.0_rt;
    mo.Su_mean = 15.0_rt;
    mo.Sv_mean = 20.0_rt;
    mo.theta_mean = 300.0_rt;
    mo.Stheta_mean = 1500.0_rt;
    mo.surf_temp = 300.0_rt;
    mo.surf_temp_flux = 0.0_rt;
    return mo;
}

// Reference implementation of the former phi_m-based psi_m
amrex::Real psi_m_reference(
    const kynema_sgf::MOData& mo, const amrex::Real z, const amrex::Real L)
{
    if (L > 0.0_rt) {
        return 1.0_rt - (1.0_rt + mo.gamma_m * z / L);
    }
    const amrex::Real phi = std::pow(1.0_rt - mo.beta_m * z / L, -0.25_rt);
    return 2.0_rt * std::log((1.0_rt + 1.0_rt / phi) / 2.0_rt) +
           std::log((1.0_rt + 1.0_rt / (phi * phi)) / 2.0_rt) -
           2.0_rt * std::atan(1.0_rt / phi) + kynema_sgf::utils::half_pi();
}

// Returns {calc_vel_x, calc_vel_y, calc_theta}
template <typename Tau>
std::array<amrex::Real, 3> eval_tau(
    const Tau& tau,
    const amrex::Real u,
    const amrex::Real v,
    const amrex::Real wspd,
    const amrex::Real theta)
{
    amrex::Gpu::DeviceVector<amrex::Real> dout(3);
    auto* p = dout.data();
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) {
        p[0] = tau.calc_vel_x(u, wspd);
        p[1] = tau.calc_vel_y(v, wspd);
        p[2] = tau.calc_theta(wspd, theta);
    });
    std::array<amrex::Real, 3> out{};
    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, dout.begin(), dout.end(), out.begin());
    return out;
}

} // namespace

TEST(MODataTest, neutral_utau)
{
    auto mo = make_mo();
    mo.update_fluxes();
    const amrex::Real expected =
        mo.kappa * mo.vmag_mean / std::log(mo.zref / mo.z0);
    EXPECT_NEAR(mo.utau, expected, tol * expected);
    EXPECT_EQ(mo.L, mo.L_neutral);
}

TEST(MODataTest, utau_reference_speed_switch)
{
    auto mo = make_mo();
    mo.vmag_mean = 5.0_rt;
    mo.vmean_mag = 3.0_rt;
    const amrex::Real lg = std::log(mo.zref / mo.z0);

    mo.use_mean_vel_mag = false;
    mo.update_fluxes();
    EXPECT_NEAR(mo.utau, mo.kappa * 5.0_rt / lg, tol);

    mo.use_mean_vel_mag = true;
    mo.update_fluxes();
    EXPECT_NEAR(mo.utau, mo.kappa * 3.0_rt / lg, tol);
}

TEST(MODataTest, psi_m_matches_reference_formula)
{
    kynema_sgf::MOData mo;
    for (const amrex::Real L :
         {-200.0_rt, -20.0_rt, -2.0_rt, 2.0_rt, 20.0_rt, 200.0_rt}) {
        for (const amrex::Real z : {0.1_rt, 1.0_rt, 5.0_rt}) {
            mo.L = L;
            const amrex::Real ref = psi_m_reference(mo, z, L);
            EXPECT_NEAR(mo.psi_m(z), ref, 1.0e-10_rt);
            EXPECT_NEAR(
                kynema_sgf::MOData::calc_psi_m(z / L, mo.beta_m, mo.gamma_m),
                ref, 1.0e-10_rt);
        }
    }
    mo.L = mo.L_neutral * 10.0_rt;
    EXPECT_EQ(mo.psi_m(5.0_rt), 0.0_rt);
}

TEST(MODataTest, stable_unstable_fixed_point)
{
    for (const amrex::Real Q : {-0.05_rt, 0.05_rt}) {
        auto mo = make_mo();
        mo.alg_type = kynema_sgf::MOData::ThetaCalcType::HEAT_FLUX;
        mo.surf_temp_flux = Q;
        mo.update_fluxes();

        const amrex::Real neutral =
            mo.kappa * mo.vmag_mean / std::log(mo.zref / mo.z0);
        if (Q < 0.0_rt) {
            EXPECT_GT(mo.L, 0.0_rt);
            EXPECT_LT(mo.utau, neutral);
        } else {
            EXPECT_LT(mo.L, 0.0_rt);
            EXPECT_GT(mo.utau, neutral);
        }

        const amrex::Real denom =
            std::log(mo.zref / mo.z0) -
            (kynema_sgf::MOData::calc_psi_m(
                 mo.zref / mo.L, mo.beta_m, mo.gamma_m) -
             kynema_sgf::MOData::calc_psi_m(
                 mo.z0 / mo.L, mo.beta_m, mo.gamma_m));
        EXPECT_NEAR(
            mo.utau, mo.kappa * mo.vmag_mean / denom, 1.0e-6_rt * mo.utau);
    }
}

TEST(MODataTest, surface_temp_heat_flux_round_trip)
{
    auto a = make_mo();
    a.alg_type = kynema_sgf::MOData::ThetaCalcType::SURFACE_TEMPERATURE;
    a.surf_temp = 302.0_rt;
    a.update_fluxes();
    EXPECT_GT(a.surf_temp_flux, 0.0_rt);

    auto b = make_mo();
    b.alg_type = kynema_sgf::MOData::ThetaCalcType::HEAT_FLUX;
    b.surf_temp = 302.0_rt;
    b.surf_temp_flux = a.surf_temp_flux;
    b.update_fluxes();
    EXPECT_NEAR(b.surf_temp, 302.0_rt, 1.0e-6_rt * 302.0_rt);
    EXPECT_NEAR(b.utau, a.utau, 1.0e-6_rt * a.utau);
}

TEST(ShearStressNonlinearTest, uniform_wind_gives_utau_squared)
{
    auto mo = make_mo(); // u=(3,4), |u|=5, Su=15, Sv=20
    mo.use_mean_vel_mag = true;
    mo.update_fluxes();
    const kynema_sgf::ShearStressNonlinear tau(mo);
    const auto out = eval_tau(tau, 3.0_rt, 4.0_rt, 5.0_rt, 300.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_NEAR(out[0], 0.6_rt * utau2, tol * utau2);
    EXPECT_NEAR(out[1], 0.8_rt * utau2, tol * utau2);
    EXPECT_NEAR(std::hypot(out[0], out[1]), utau2, tol * utau2);
}

TEST(ShearStressNonlinearTest, mean_stress_equals_utau_squared)
{
    // Two equal-area states: u = 2 and u = -1, so <|u|u> = 1.5
    auto mo = make_mo();
    mo.utau = 0.3_rt;
    mo.vmag_mean = 1.5_rt;
    mo.Su_mean = 1.5_rt;
    mo.Sv_mean = 0.0_rt;
    const kynema_sgf::ShearStressNonlinear tau(mo);
    const auto s1 = eval_tau(tau, 2.0_rt, 0.0_rt, 2.0_rt, 0.0_rt);
    const auto s2 = eval_tau(tau, -1.0_rt, 0.0_rt, 1.0_rt, 0.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_NEAR(0.5_rt * (s1[0] + s2[0]), utau2, tol * utau2);
    EXPECT_NEAR(0.5_rt * (s1[1] + s2[1]), 0.0_rt, tol);
}

TEST(ShearStressNonlinearTest, heat_flux_magnitude_and_sign_vs_moeng)
{
    // Stable state (theta > theta_s) with downward flux Q < 0.
    // Two equal-area states: (theta=10, |u|=2) and (theta=20, |u|=1)
    auto mo = make_mo();
    mo.utau = 0.3_rt;
    mo.vmag_mean = 1.5_rt;
    mo.theta_mean = 15.0_rt;
    mo.Stheta_mean = 20.0_rt; // <|u| theta>
    mo.surf_temp = 5.0_rt;
    mo.surf_temp_flux = -0.02_rt;

    const kynema_sgf::ShearStressNonlinear nl(mo);
    const kynema_sgf::ShearStressMoeng moeng(mo);
    const auto n1 = eval_tau(nl, 0.0_rt, 0.0_rt, 2.0_rt, 10.0_rt);
    const auto n2 = eval_tau(nl, 0.0_rt, 0.0_rt, 1.0_rt, 20.0_rt);
    const auto m1 = eval_tau(moeng, 0.0_rt, 0.0_rt, 2.0_rt, 10.0_rt);
    const auto m2 = eval_tau(moeng, 0.0_rt, 0.0_rt, 1.0_rt, 20.0_rt);
    const amrex::Real nl_mean = 0.5_rt * (n1[2] + n2[2]);
    const amrex::Real moeng_mean = 0.5_rt * (m1[2] + m2[2]);

    EXPECT_NEAR(std::abs(nl_mean), 0.02_rt, tol);
    // Same BC sign convention as Moeng
    EXPECT_GT(nl_mean * moeng_mean, 0.0_rt);
}

} // namespace kynema_sgf_tests
