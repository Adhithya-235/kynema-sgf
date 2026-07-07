#ifndef HORIZONTALVELOCITYMAG_H
#define HORIZONTALVELOCITYMAG_H

#include "src/fvm/fvm_utils.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::fvm {

/** HorizontalVelocityMag operator 
 *  \ingroup fvm
 */ 

template <typename FTypeIn, typename FTypeOut>
struct HorizontalVelocityMag
{
    HorizontalVelocityMag(FTypeOut& horvelmagphi, const FTypeIn& phi)
        : m_horvelmagphi(horvelmagphi), m_phi(phi)
    {
        AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM == m_phi.num_comp());
        AMREX_ALWAYS_ASSERT(m_phi.num_grow() > amrex::IntVect(0));
    }

    template <typename Stencil>
    void apply(const int lev) const
    {
        const auto& geom = m_phi.repo().mesh().Geom(lev);
        const auto& dlo = geom.Domain().smallEnd();
        const auto& dhi = geom.Domain().bigEnd();

        auto& out_mf = m_horvelmagphi(lev);
        auto const& in_arrs = m_phi(lev).const_arrays();
        auto const& out_arrs = out_mf.arrays();

        amrex::ParallelFor(
            out_mf, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                if (!Stencil::applies_to(i, j, k, dlo, dhi)) {
                    return;
                }
                auto const& phi = in_arrs[nbx];
                auto const& horvelmagphi = out_arrs[nbx];
                const amrex::Real u = phi(i, j, k, 0);
                const amrex::Real v = phi(i, j, k, 1);
                horvelmagphi(i, j, k) = std::sqrt(utils::powi(u, 2) + utils::powi(v, 2));
            }
        );
    }

    FTypeOut& m_horvelmagphi;
    const FTypeIn& m_phi;
}; 

/** Compute the magnitude of horizontal velocity
 *  \ingroup fvm
 * 
 *  \param horvelmagphi [out] Field where the horizontal velocity magnitude is populated
 *  \param phi [in] The velocity vector field 
 */
template <typename FTypeIn, typename FTypeOut>
void horizontal_velocity_mag(FTypeOut& horvelmagphi, const FTypeIn& phi)
{
    BL_PROFILE("kynema-sgf::fvm::horizontal_velocity_mag");
    HorizontalVelocityMag<FTypeIn, FTypeOut> horvelmag(horvelmagphi, phi);
    impl::apply(horvelmag, phi);
}

/** Compute the magnitude of horizontal velocity return as a ScratchField
 *  \ingroup fvm
 * 
 *  \param phi [in] The velocity vector field
 */
template <typename FType>
std::unique_ptr<ScratchField> horizontal_velocity_mag(const FType& phi)
{
    const std::string gname = phi.name() + "_horizontal_velocity_magnitude";
    auto horvelmagphi = phi.repo().create_scratch_field(gname, 1);
    horizontal_velocity_mag(*horvelmagphi, phi);
    return horvelmagphi;
}

} // namespace kynema_sgf::fvm

#endif /* HORIZONTALVELOCITYMAG_H */