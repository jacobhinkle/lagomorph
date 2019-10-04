#include <torch/extension.h>

#include <vector>

#include "interp.h"

#define CHECK_CPU(x) TORCH_CHECK(! x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

template<typename Real, bool broadcast_I>
void affine_interp_2d(
        at::Tensor out_,
        at::Tensor I_,
        at::Tensor A_,
        at::Tensor T_) {
    auto nn = A_.size(0);
    auto nc = I_.size(1);
    auto nx = I_.size(2);
    auto ny = I_.size(3);
    auto out = out_.data<Real>();
    auto I = I_.data<Real>();
    auto A = A_.data<Real>();
    auto T = T_.data<Real>();
    const int nxy = nx*ny;
    const Real* In = I; // pointer to current vector field v
    Real* outn = out; // pointer to current vector field v
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real hx, hy;
    Real Inx;
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) In = I;
        auto A11 = An[0];
        auto A12 = An[1];
        auto A21 = An[2];
        auto A22 = An[3];
        auto T1 = Tn[0];
        auto T2 = Tn[1];
        for (int c=0; c < nc; ++c) {
            for (int i=0; i < nx; ++i) {
                int ix = i*ny;
                Real fi=static_cast<Real>(i)-ox;
                hx = A11*fi - A12*oy + T1 + ox;
                hy = A21*fi - A22*oy + T2 + oy;
                for (int j=0; j < ny; ++j, ++ix, hx+=A12, hy+=A22) {
                    Inx = biLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                        hx, hy,
                        nx, ny);
                    outn[ix] = Inx;
                }
            }
            outn += nxy;
            In += nxy;
        }
        An += 4;
        Tn += 2;
    }
}

template<typename Real, bool broadcast_I>
void affine_interp_3d(
        at::Tensor out_,
        at::Tensor I_,
        at::Tensor A_,
        at::Tensor T_) {
    auto nn = A_.size(0);
    auto nc = I_.size(1);
    auto nx = I_.size(2);
    auto ny = I_.size(3);
    auto nz = I_.size(4);
    auto out = out_.data<Real>();
    auto I = I_.data<Real>();
    auto A = A_.data<Real>();
    auto T = T_.data<Real>();
    const int nxyz = nx*ny*nz;
    const Real* In = I; // pointer to current vector field v
    Real* outn = out; // pointer to current vector field v
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real oz = .5*static_cast<Real>(nz-1);
    Real hx, hy, hz;
    Real Inx;
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) In = I;
        auto A11 = An[0];
        auto A12 = An[1];
        auto A13 = An[2];
        auto A21 = An[3];
        auto A22 = An[4];
        auto A23 = An[5];
        auto A31 = An[6];
        auto A32 = An[7];
        auto A33 = An[8];
        auto T1 = Tn[0];
        auto T2 = Tn[1];
        auto T3 = Tn[2];
        for (int c=0; c < nc; ++c) {
            for (int i=0; i < nx; ++i) {
                Real fi=static_cast<Real>(i)-ox;
                for (int j=0; j < ny; ++j) {
                    int ix = (i*ny + j)*nz;
                    Real fj=static_cast<Real>(j)-oy;
                    hx = A11*fi + A12*fj - A13*oz + T1 + ox;
                    hy = A21*fi + A22*fj - A23*oz + T2 + oy;
                    hz = A31*fi + A32*fj - A33*oz + T3 + oz;
                    for (int k=0; k < nz; ++k, ++ix, hx+=A13, hy+=A23, hz+=A33) {
                        Inx = triLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                            hx, hy, hz,
                            nx, ny, nz);
                        outn[ix] = Inx;
                    }
                }
            }
            outn += nxyz;
            In += nxyz;
        }
        An += 9;
        Tn += 3;
    }
}

at::Tensor affine_interp_cpu_forward(
    at::Tensor I,
    at::Tensor A,
    at::Tensor T) {
    CHECK_INPUT(I)
    CHECK_INPUT(A)
    CHECK_INPUT(T)
    TORCH_CHECK(A.size(0) == T.size(0), "A and T must have same first dimension");
    auto d = I.dim() - 2;
    TORCH_CHECK(d == 2 || d == 3, "Only two- and three-dimensional affine interpolation is supported");

    const bool broadcast_I = I.size(0) == 1 && A.size(0) > 1;

    at::Tensor Itx;

    if (d == 2) {
        Itx = at::zeros({A.size(0), I.size(1), I.size(2), I.size(3)}, I.type());
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_cpu_forward", ([&] {
            affine_interp_2d<scalar_t, broadcastI>(
                Itx,
                I,
                A,
                T);
            }));
        }));
    } else {
        Itx = at::zeros({A.size(0), I.size(1), I.size(2), I.size(3), I.size(4)}, I.type());
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_cpu_forward", ([&] {
            affine_interp_3d<scalar_t, broadcastI>(
                Itx,
                I,
                A,
                T);
            }));
        }));
    }

    return Itx;
}
