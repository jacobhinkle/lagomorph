"""
Adjoint representation for Diff(R^d)
"""
from pycuda import gpuarray
import numpy as np

from .arithmetic import multiply_imvec
from .diff import gradient, jacobian_times_vectorfield, divergence
from .deform import interp_vec

def ad(v, w):
    """
    This is ad(v,w), the adjoint action of a velocity v on a
    velocity w.

        ad(v,w) = -[v,w] = Dv w - Dw v
    """
    return jacobian_times_vectorfield(v,w,displacement=False) \
            - jacobian_times_vectorfield(w,v,displacement=False)
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
    raise NotImplementedError("not implemented yet")
def ad_star(v, m):
    """
    This is ad^*(v,m), the coadjoint action of a velocity v on a
    vector momentum m.

        ad^*(v, m) = (Dv)^T m + Dm v + m div v

    where div denotes the divergence of a vector field
    """
    out = jacobian_times_vectorfield(v, m, displacement=False, transpose=True)
    out += jacobian_times_vectorfield(m, v, displacement=False)
    dv = divergence(v)
    mdv = multiply_imvec(dv, m)
    out += mdv
    return out
def Ad_star(phi, m, displacement=True, out=None):
    """
    This is Ad^*(phi,m), the big coadjoint action of a deformation phi on a
    vector momentum m. The formula for this is

        Ad^*(phi,m)(x) = (D phi(x)) m(phi(x))

    where D denotes the Jacobian matrix.
    """
    # First interpolate m
    mphi = interp_vec(m, phi)
    ret = jacobian_times_vectorfield(phi, mphi, displacement=displacement, out=out)
    return ret
# dagger versions of the above coadjoint operators
# The dagger indicates that instead of a _dual_ action, the _adjoint_ action
# under a metric. These are performed by flatting, applying to dual action,
# then sharping.
def ad_dagger(x, y, metric):
    return metric.sharp(ad_star(x, metric.flat(y)))
def Ad_dagger(phi, y, metric, displacement=True):
    return metric.sharp(Ad_star(phi, metric.flat(y), displacement=displacement))
# The sym operator is a negative symmetrized ad_dagger, and is important for
# computing reduced Jacobi fields.
# cf. Bullo 1995 or Hinkle 2015 (PhD thesis)
def sym(x, y, metric):
    return -(ad_dagger(x, y, metric) + ad_dagger(y, x, metric))
def sym_dagger(x, y, metric):
    return ad_dagger(y, x, metric) - ad(x, y)
