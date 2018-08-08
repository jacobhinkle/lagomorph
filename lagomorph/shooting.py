"""
Vector and scalar momentum shooting algorithms
"""
from .kernel import FluidKernel
from .deform import identitylikedef, composeHV
from .adjrep import AdjRep

def expmap(m0, K, T=1.0, Nsteps=100, phi0=None, phi=None):
    """
    Given an initial momentum (Lie algebra element), compute the exponential
    map.
    """
    d = m0.ndim-2

    if phi0 is None:
        phi0 = identitylikedef(m0)

    # Helper classes
    adj = AdjRep(dim=d)

    dt = T/Nsteps
    # initialization
    m = m0.copy()
    if phi is None:
        phi = phi0.copy()

    for i in range(Nsteps):
        m = adj.Ad_star(phi, m0)
        K.inverse(m)
        composeHV(phi, m, dt=dt)

    return phi

