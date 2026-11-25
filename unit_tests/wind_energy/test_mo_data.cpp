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
    mo.FhFh_mean = 625.0_rt; // |u|^4 for uniform |u| = 5
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
        if constexpr (requires { tau.calc_vel_x(u, v, wspd); }) {
            p[0] = tau.calc_vel_x(u, v, wspd);
            p[1] = tau.calc_vel_y(u, v, wspd);
        } else {
            p[0] = tau.calc_vel_x(u, wspd);
            p[1] = tau.calc_vel_y(v, wspd);
        }
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

TEST(ShearStressNonlinearTest, zero_flux_neutral_is_zero)
{
    // Exactly neutral: <|u| theta> == <|u|> theta_s makes the old
    // normalization 0/0
    auto mo = make_mo();
    mo.surf_temp_flux = 0.0_rt;
    mo.Stheta_mean = mo.vmag_mean * mo.surf_temp;
    const kynema_sgf::ShearStressNonlinear tau(mo);
    EXPECT_FALSE(tau.flux_resolved);
    const auto out = eval_tau(tau, 3.0_rt, 4.0_rt, 5.0_rt, 300.0_rt);
    EXPECT_TRUE(std::isfinite(out[2]));
    EXPECT_EQ(out[2], 0.0_rt);
}

TEST(ShearStressNonlinearTest, unresolved_temperature_difference_is_zero)
{
    // Nonzero flux but speed-weighted dtheta below threshold (roundoff level)
    auto mo = make_mo();
    mo.surf_temp_flux = 0.01_rt;
    mo.Stheta_mean =
        mo.vmag_mean * (mo.surf_temp + 0.1_rt *
                                           kynema_sgf::ShearStressNonlinear::
                                               dtheta_min);
    const kynema_sgf::ShearStressNonlinear tau(mo);
    EXPECT_FALSE(tau.flux_resolved);
    const auto out = eval_tau(tau, 3.0_rt, 4.0_rt, 5.0_rt, 300.0_rt);
    EXPECT_TRUE(std::isfinite(out[2]));
    EXPECT_EQ(out[2], 0.0_rt);
}

TEST(ShearStressWeiBrasseurTest, uniform_wind_gives_utau2_along_M)
{
    // Point equals the plane mean: Fh' = 0, so F = 0 and tau = utau^2 M
    auto mo = make_mo(); // <Fh> = (15, 20), M = (0.6, 0.8)
    mo.use_mean_vel_mag = true;
    mo.update_fluxes();
    const kynema_sgf::ShearStressWeiBrasseur tau(mo, 0.5_rt);
    const auto out = eval_tau(tau, 3.0_rt, 4.0_rt, 5.0_rt, 300.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
    EXPECT_NEAR(out[0], 0.6_rt * utau2, tol * utau2);
    EXPECT_NEAR(out[1], 0.8_rt * utau2, tol * utau2);
}

TEST(ShearStressWeiBrasseurTest, beta_zero_gives_mean_direction)
{
    auto mo = make_mo();
    mo.utau = 0.3_rt;
    const kynema_sgf::ShearStressWeiBrasseur tau(mo, 0.0_rt);
    const amrex::Real u = -2.0_rt;
    const amrex::Real v = 1.0_rt;
    const auto out = eval_tau(tau, u, v, std::hypot(u, v), 300.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_NEAR(out[0], 0.6_rt * utau2, tol * utau2);
    EXPECT_NEAR(out[1], 0.8_rt * utau2, tol * utau2);
}

TEST(ShearStressWeiBrasseurTest, hand_computed_two_component_point)
{
    // Two equal-area states u = 2 and u = -1: <Fh> = (1.5, 0), M = (1, 0),
    // <|Fh|^2> = 8.5, rms(Fh') = sqrt(8.5 - 2.25) = 2.5.
    // Point u = (0, 1): Fh' = (-1.5, 1), F = (-0.6, 0.4)
    auto mo = make_mo();
    mo.utau = 0.3_rt;
    mo.vmag_mean = 1.5_rt;
    mo.Su_mean = 1.5_rt;
    mo.Sv_mean = 0.0_rt;
    mo.FhFh_mean = 8.5_rt;
    const amrex::Real beta = 0.5_rt;
    const kynema_sgf::ShearStressWeiBrasseur tau(mo, beta);
    EXPECT_NEAR(tau.Fp_rms, 2.5_rt, tol);
    const auto out = eval_tau(tau, 0.0_rt, 1.0_rt, 1.0_rt, 0.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_NEAR(out[0], utau2 * (1.0_rt + (beta * -0.6_rt)), tol * utau2);
    EXPECT_NEAR(out[1], utau2 * (beta * 0.4_rt), tol * utau2);
}

TEST(ShearStressWeiBrasseurTest, fluctuation_term_is_beta_Fh_prime_over_rms)
{
    auto mo = make_mo(); // <Fh> = (15, 20), M = (0.6, 0.8)
    mo.utau = 0.3_rt;
    mo.FhFh_mean = 725.0_rt; // rms(Fh') = sqrt(725 - 625) = 10
    const amrex::Real beta = 0.7_rt;
    const kynema_sgf::ShearStressWeiBrasseur tau(mo, beta);
    const amrex::Real utau2 = mo.utau * mo.utau;
    const std::array<std::array<amrex::Real, 2>, 5> pts{
        {{{2.0_rt, 3.0_rt}},
         {{4.5_rt, 3.5_rt}},
         {{-1.0_rt, 2.0_rt}},
         {{3.2_rt, 4.1_rt}},
         {{0.0_rt, -1.0_rt}}}};
    for (const auto& pt : pts) {
        const amrex::Real u = pt[0];
        const amrex::Real v = pt[1];
        const amrex::Real wspd = std::hypot(u, v);
        const auto out = eval_tau(tau, u, v, wspd, 300.0_rt);
        const amrex::Real fx = (wspd * u) - mo.Su_mean;
        const amrex::Real fy = (wspd * v) - mo.Sv_mean;
        EXPECT_NEAR(
            out[0], utau2 * (0.6_rt + (beta * fx / 10.0_rt)), tol * utau2);
        EXPECT_NEAR(
            out[1], utau2 * (0.8_rt + (beta * fy / 10.0_rt)), tol * utau2);
    }
}

TEST(ShearStressWeiBrasseurTest, beta_rms_ratio_matches_nonlinear)
{
    // With beta = rms(Fh') / |<Fh>| the model reduces to ShearStressNonlinear
    auto mo = make_mo(); // |<Fh>| = 25
    mo.utau = 0.3_rt;
    mo.FhFh_mean = 725.0_rt; // rms(Fh') = 10
    const kynema_sgf::ShearStressNonlinear nl(mo);
    const kynema_sgf::ShearStressWeiBrasseur wb(mo, 10.0_rt / 25.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    const std::array<std::array<amrex::Real, 2>, 4> pts{
        {{{2.0_rt, 3.0_rt}},
         {{4.5_rt, 3.5_rt}},
         {{-1.0_rt, 2.0_rt}},
         {{0.0_rt, -1.0_rt}}}};
    for (const auto& pt : pts) {
        const amrex::Real wspd = std::hypot(pt[0], pt[1]);
        const auto n = eval_tau(nl, pt[0], pt[1], wspd, 300.0_rt);
        const auto w = eval_tau(wb, pt[0], pt[1], wspd, 300.0_rt);
        EXPECT_NEAR(w[0], n[0], tol * utau2);
        EXPECT_NEAR(w[1], n[1], tol * utau2);
    }
}

TEST(ShearStressWeiBrasseurTest, symmetric_states_average_to_mean)
{
    // Two equal-area states: u = 2 and u = -1, so <|u|u> = 1.5, M = (1, 0),
    // Fh' = +-2.5 and rms(Fh') = 2.5
    auto mo = make_mo();
    mo.utau = 0.3_rt;
    mo.vmag_mean = 1.5_rt;
    mo.Su_mean = 1.5_rt;
    mo.Sv_mean = 0.0_rt;
    mo.FhFh_mean = 8.5_rt;
    const amrex::Real beta = 0.5_rt;
    const kynema_sgf::ShearStressWeiBrasseur tau(mo, beta);
    const auto s1 = eval_tau(tau, 2.0_rt, 0.0_rt, 2.0_rt, 0.0_rt);
    const auto s2 = eval_tau(tau, -1.0_rt, 0.0_rt, 1.0_rt, 0.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_NEAR(s1[0], utau2 * (1.0_rt + beta), tol * utau2);
    EXPECT_NEAR(s2[0], utau2 * (1.0_rt - beta), tol * utau2);
    EXPECT_NEAR(0.5_rt * (s1[0] + s2[0]), utau2, tol * utau2);
    EXPECT_NEAR(0.5_rt * (s1[1] + s2[1]), 0.0_rt, tol);
}

TEST(ShearStressWeiBrasseurTest, zero_mean_flux_guard)
{
    // <Fh> = 0: M = 0, rms(Fh') = sqrt(<|Fh|^2>) = 25, tau = utau^2 beta F
    auto mo = make_mo();
    mo.utau = 0.3_rt;
    mo.Su_mean = 0.0_rt;
    mo.Sv_mean = 0.0_rt;
    const amrex::Real beta = 0.5_rt;
    const kynema_sgf::ShearStressWeiBrasseur tau(mo, beta);
    EXPECT_EQ(tau.Mx, 0.0_rt);
    EXPECT_EQ(tau.My, 0.0_rt);
    const auto out = eval_tau(tau, 3.0_rt, 4.0_rt, 5.0_rt, 300.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
    EXPECT_NEAR(out[0], utau2 * beta * 0.6_rt, tol * utau2);
    EXPECT_NEAR(out[1], utau2 * beta * 0.8_rt, tol * utau2);
}

TEST(ShearStressWeiBrasseurTest, zero_rms_guard)
{
    // Roundoff can make <|Fh|^2> - |<Fh>|^2 slightly negative: rms is
    // clamped to 0, F = 0 and tau = utau^2 M even for a nonzero local Fh'
    auto mo = make_mo(); // <Fh> = (15, 20), M = (0.6, 0.8)
    mo.utau = 0.3_rt;
    mo.FhFh_mean = 625.0_rt * (1.0_rt - 1.0e-14_rt);
    const kynema_sgf::ShearStressWeiBrasseur tau(mo, 0.5_rt);
    EXPECT_EQ(tau.Fp_rms, 0.0_rt);
    EXPECT_EQ(tau.inv_Fp_rms, 0.0_rt);
    const auto out = eval_tau(tau, -1.0_rt, 2.0_rt, std::sqrt(5.0_rt), 0.0_rt);
    const amrex::Real utau2 = mo.utau * mo.utau;
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
    EXPECT_NEAR(out[0], 0.6_rt * utau2, tol * utau2);
    EXPECT_NEAR(out[1], 0.8_rt * utau2, tol * utau2);
}

TEST(ShearStressWeiBrasseurTest, heat_flux_matches_nonlinear)
{
    auto mo = make_mo();
    mo.utau = 0.3_rt;
    mo.vmag_mean = 1.5_rt;
    mo.theta_mean = 15.0_rt;
    mo.Stheta_mean = 20.0_rt;
    mo.surf_temp = 5.0_rt;
    mo.surf_temp_flux = -0.02_rt;
    const kynema_sgf::ShearStressNonlinear nl(mo);
    const kynema_sgf::ShearStressWeiBrasseur wb(mo, 0.5_rt);
    EXPECT_TRUE(wb.flux_resolved);
    const auto n1 = eval_tau(nl, 0.0_rt, 0.0_rt, 2.0_rt, 10.0_rt);
    const auto w1 = eval_tau(wb, 0.0_rt, 0.0_rt, 2.0_rt, 10.0_rt);
    const auto n2 = eval_tau(nl, 0.0_rt, 0.0_rt, 1.0_rt, 20.0_rt);
    const auto w2 = eval_tau(wb, 0.0_rt, 0.0_rt, 1.0_rt, 20.0_rt);
    EXPECT_EQ(w1[2], n1[2]);
    EXPECT_EQ(w2[2], n2[2]);
}

TEST(ShearStressWeiBrasseurTest, utau_matches_nonlinear)
{
    auto mo = make_mo();
    mo.use_mean_vel_mag = true;
    mo.update_fluxes();
    const kynema_sgf::ShearStressNonlinear nl(mo);
    const kynema_sgf::ShearStressWeiBrasseur wb(mo, 0.5_rt);
    EXPECT_EQ(wb.utau2, nl.utau2);
}

} // namespace kynema_sgf_tests
