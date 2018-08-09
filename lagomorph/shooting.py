"""
Vector and scalar momentum shooting algorithms
"""
from .metric import FluidMetric
from .deform import identitylikedef, composeHV
from .adjrep import AdjRep

def expmap(m0, metric, T=1.0, Nsteps=10, phi=None):
    """
    Given an initial momentum (Lie algebra element), compute the exponential
    map.
    """
    d = m0.ndim-2

    if phi is None:
        phi = identitylikedef(m0)

    # Helper class for adjoint representation
    adj = AdjRep(dim=d)

    dt = T/Nsteps

    for i in range(Nsteps):
        m = adj.Ad_star(phi, m0)
        #v = metric.sharp(m)
        v = m
        phi = composeHV(phi, v, dt=dt)
        break

    return phi

def jacobi_field_backward(m0, metric, lamT, lam=None, mu=None, phi=None, T=1.0, Nsteps=2):
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

        m(t) = Ad^*_{Phi^{-1}(t)}(m(0))
        v(t) = m(t)^sharp
        lambda(t) = Ad^*_{Phi(T to t)}(lambdaT)
        d/dt mu(t) = -sym^\dagger_{v(t)} mu(t) - lambda(t)
        d/dt Phi(t) = v \circ Phi(t)
        d/dt Phi^{-1}(t) = Phi^{-1}(t)\circ (x - dt v(t)) -- abuse of notation
    """
    lam = lamT
    mu = lam
    print("WARNING: dummy Jacobi field")
    return lam, mu
