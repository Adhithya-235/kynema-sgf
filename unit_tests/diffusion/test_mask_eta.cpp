#include "ks_test_utils/MeshTest.H"
#include "ks_test_utils/pp_utils.H"
#include "src/diffusion/diffusion.H"
#include "AMReX_ParReduce.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {

namespace {

constexpr int ncells = 8;
constexpr amrex::Real eta_val = 2.0_rt;

// Height in cells; h == 0 columns put fluid directly on the bottom wall
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int terrain_h(int i, int j)
{
    const int ii = ((i % ncells) + ncells) % ncells;
    const int jj = ((j % ncells) + ncells) % ncells;
    return (ii + jj) % 3;
}

// Ghost cells below the domain (k < 0) are always solid since h >= 0
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int is_solid(int i, int j, int k)
{
    return static_cast<int>(k < terrain_h(i, j));
}

} // namespace

class MaskEtaTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        pp_utils::default_time_inputs();
        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{ncells, ncells, ncells}};
            pp.add("verbose", 0);
            pp.add("max_level", 0);
            pp.add("max_grid_size", 4);
            pp.addarr("n_cell", ncell);
        }
        {
            amrex::ParmParse pp("geometry");
            amrex::Vector<amrex::Real> problo{{0.0_rt, 0.0_rt, 0.0_rt}};
            amrex::Vector<amrex::Real> probhi{{8.0_rt, 8.0_rt, 8.0_rt}};
            amrex::Vector<int> periodic{{1, 1, 0}};
            pp.addarr("prob_lo", problo);
            pp.addarr("prob_hi", probhi);
            pp.addarr("is_periodic", periodic);
        }
        m_need_params = false;
    }
};

TEST_F(MaskEtaTest, interface_faces_zeroed)
{
    initialize_mesh();
    auto& repo = sim().repo();
    auto& eta = repo.declare_field("eta", 1, 1);
    auto& blank = repo.declare_int_field("terrain_blank", 1, 1);
    const auto& geom = mesh().Geom(0);

    eta(0).setVal(eta_val);

    const auto& blank_arrs = blank(0).arrays();
    amrex::ParallelFor(
        blank(0), blank(0).nGrowVect(),
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            blank_arrs[nbx](i, j, k) = is_solid(i, j, k);
        });
    amrex::Gpu::streamSynchronize();

    auto b = diffusion::average_velocity_eta_to_faces(geom, eta(0));
    diffusion::mask_eta_on_faces(geom, b, blank(0), 0.0_rt);

    const int klo = geom.Domain().smallEnd(2);
    const int khi = geom.Domain().bigEnd(2) + 1;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const auto& b_arrs = b[idim].const_arrays();
        const amrex::IntVect iv = amrex::IntVect::TheDimensionVector(idim);

        auto r = amrex::ParReduce(
            amrex::TypeList<
                amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                amrex::ReduceOpSum>{},
            amrex::TypeList<int, int, int, int>{}, b[idim], amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k)
                -> amrex::GpuTuple<int, int, int, int> {
                const amrex::Real val = b_arrs[nbx](i, j, k);
                const bool straddles =
                    is_solid(i - iv[0], j - iv[1], k - iv[2]) !=
                    is_solid(i, j, k);
                const bool on_wall = (idim == 2) && (k == klo || k == khi);
                const bool expect_zero = straddles && !on_wall;
                return {
                    static_cast<int>(expect_zero),
                    static_cast<int>(expect_zero && val != 0.0_rt),
                    static_cast<int>(!expect_zero && val != eta_val),
                    static_cast<int>(straddles && on_wall)};
            });

        int n_zeroed = amrex::get<0>(r);
        int n_bad_zero = amrex::get<1>(r);
        int n_bad_keep = amrex::get<2>(r);
        int n_wall_straddle = amrex::get<3>(r);
        amrex::ParallelDescriptor::ReduceIntSum(n_zeroed);
        amrex::ParallelDescriptor::ReduceIntSum(n_bad_zero);
        amrex::ParallelDescriptor::ReduceIntSum(n_bad_keep);
        amrex::ParallelDescriptor::ReduceIntSum(n_wall_straddle);

        EXPECT_GT(n_zeroed, 0) << "direction " << idim;
        EXPECT_EQ(n_bad_zero, 0) << "direction " << idim;
        EXPECT_EQ(n_bad_keep, 0) << "direction " << idim;
        if (idim == 2) {
            EXPECT_GT(n_wall_straddle, 0);
        }
    }
}

} // namespace kynema_sgf_tests
