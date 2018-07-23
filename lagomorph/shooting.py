"""
Vector and scalar momentum shooting algorithms
"""
from .kernel import FluidKernel
from .deform import identitylike, composeHV
from .adjrep import AdjRep

def expmap(m0, T=1.0, Nsteps=100, phi0=None, phi=None):
    """
    Given an initial momentum (Lie algebra element), compute the exponential
    map.
    """
    d = m0.ndim-2

    if phi0 is None:
        phi0 = identitylike(m0)

    # Helper classes
    ar = AdjRep(dim=d)
    K = FluidKernel(m0.shape)

    dt = T/Nsteps
    # initialization
    m = m0.copy()
    if phi is None:
        phi = phi0.copy()

    for i in range(Nsteps):
        m = ar.bigcoad(phi, m0)
        K.apply(m)
        composeHV(phi, m, dt=dt)

    return phi
