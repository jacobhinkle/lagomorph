"""
Adjoint representation for Diff(R^d)
"""
import numpy as np
from .diff import jacobian_times_vectorfield, jacobian_times_vectorfield_adjoint
from .deform import interp

def ad(v, w):
    """
    This is ad(v,w), the adjoint action of a velocity v on a
    velocity w.

        ad(v,w) = -[v,w] = Dv w - Dw v
    """
    return jacobian_times_vectorfield(v, w, displacement=False) \
            - jacobian_times_vectorfield(w, v, displacement=False)
def Ad(phi, v):
    """
    This is Ad(phi,v), the big adjoint action of a deformation phi on a
    velocity w.

        Ad(phi,v) = (Dphi \circ phi^{-1}) v\circ phi^{-1}

    This is a tricky computation, is not commonly needed in practice and
    will not be implemented until needed.

    Given phi^{-1}, Ad(phi, v) can be computed by first multiplying v by
    Dphi^{-1} then splatting the components of the resulting vector field.
    """
    #DphiTv = jacobian_times_vectorfield_adjoint(phi, v, displacement=True)
    #return splat(DphiTv, phi)
    raise NotImplementedError
def ad_star(v, m):
    """
    This is ad^*(v,m), the coadjoint action of a velocity v on a
    vector momentum m.

        ad^*(v, m) = (Dv)^T m + Dm v + m div v

    where div denotes the divergence of a vector field.

    Note that this is the numerical adjoint of ad(v,.), which is implemented
    using the common finite difference scheme.
    """
    return jacobian_times_vectorfield(v, m, displacement=False, transpose=True) \
         - jacobian_times_vectorfield_adjoint(m, v)
         
def Ad_star(phiinv, m):
    """
    This is Ad^*(phi,m), the big coadjoint action of a deformation phi on a
    vector momentum m. The formula for this is

        Ad^*(phi,m)(x) = (D phi(x)) m(phi(x))

    where D denotes the Jacobian matrix.
    """
    mphiinv = interp(m, phiinv)
    return jacobian_times_vectorfield(phiinv, mphiinv, displacement=True)
# dagger versions of the above coadjoint operators
# The dagger indicates that instead of a _dual_ action, the _adjoint_ action
# under a metric. These are performed by flatting, applying to dual action,
# then sharping.
def ad_dagger(x, y, metric):
    return metric.sharp(ad_star(x, metric.flat(y)))
def Ad_dagger(phi, y, metric):
    return metric.sharp(Ad_star(phi, metric.flat(y)))
# The sym operator is a negative symmetrized ad_dagger, and is important for
# computing reduced Jacobi fields.
# cf. Bullo 1995 or Hinkle 2015 (PhD thesis)
def sym(x, y, metric):
    return -(ad_dagger(x, y, metric) + ad_dagger(y, x, metric))
def sym_dagger(x, y, metric):
    return ad_dagger(y, x, metric) - ad(x, y)
