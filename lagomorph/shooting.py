"""
Vector and scalar momentum shooting algorithms
"""
from pycuda import gpuarray
import pycuda.autoinit
import numpy as np
from .arithmetic import multiply_add
from .metric import FluidMetric
from .deform import composeHV
from . import adjrep

def expmap(m0, metric, T=1.0, Nsteps=10, phiinv=None):
    """
    Given an initial momentum (Lie algebra element), compute the exponential
    map.

    What we return is actually only the inverse transformation phi^{-1}
    """
    d = m0.ndim-2

    phiinv = gpuarray.zeros_like(m0)

    dt = T/Nsteps

    # preallocate vector fields
    mv = gpuarray.empty_like(m0)

    for i in range(Nsteps):
        adjrep.Ad_star(phiinv, m0, out=mv)
        metric.sharp(mv, out=mv)
        phiinv = composeHV(phiinv, mv, dt=-dt)

    return phiinv

def jacobi_field_backward(m0, metric, phiinv, lamT, lam=None, mu=None, T=1.0, Nsteps=10):
    """
    Integrate the geodesic regression adjoint equation (Jacobi field). You must
    provide the initial vector momentum for the geodesic, the metric kernel, and
    the initial value of the adjoint variable, at time T (where the data lives).

    This method integrates the exponential map backwards to find phi at each
    time point, and uses the coadjoint mapping to map lamT to each timepoint, so
    two deformations to time t are maintained and updated: one from 0 to t and
    one from T to t.

    The adjoint variable mu is updated by Euler integration of the reduced
    Jacobi equation for actions of Riemannian Lie groups with right-invariant
    metrics (cf. Hinkle 2015, Appendix A):

        L lambda(T) = grad Phi(T).I0 (Phi(T).I0 - I1)
        m(t) = Ad^*_{Phi^{-1}(t)}(m(0))
        v(t) = m(t)^sharp
        lambda(t) = Ad^dagger_{Phi(T to t)}(lambdaT) = K Ad^* Phi_Tt L lambdaT
        d/dt mu(t) = -sym^\dagger_{v(t)} mu(t) - lambda(t)
        d/dt Phi(t) = v \circ Phi(t)
        d/dt Phi^{-1}(t) = Phi^{-1}(t)\circ (x - dt v(t)) -- abuse of notation
    """
    d = m0.ndim-2

    # g maps from 0 to t. h maps from T to t.
    ginv = phiinv.copy()
    hinv = gpuarray.zeros_like(ginv)
    mu = gpuarray.zeros_like(hinv)

    dt = T/Nsteps

    # preallocate vector fields
    mv = gpuarray.empty_like(m0)

    for i in reversed(range(Nsteps)): 
        adjrep.Ad_star(phiinv, m0, out=mv)
        metric.sharp(mv, out=mv)
        lam_flat = adjrep.Ad_star(hinv, lamT)
        lam = metric.sharp(lam_flat)
        dmu = adjrep.sym_dagger(mv, mu, metric)
        dmu += lam
        multiply_add(dmu, dt, out=mu)
        ginv = composeHV(ginv, mv, dt=dt)
        hinv = composeHV(hinv, mv, dt=-dt)

    return lam, mu
