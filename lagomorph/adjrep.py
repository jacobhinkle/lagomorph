"""
Adjoint representation for Diff(R^d)
"""
import numpy as np
from .diff import jacobian_times_vectorfield, jacobian_times_vectorfield_adjoint
from .deform import interp

def ad(v, w):
    r"""
    This is :math:`\ad(v,w)`, the adjoint action of a velocity `v` on a
    velocity `w`.

    .. math:: \ad(v,w) = -[v,w] = Dv w - Dw v
        :label: advw
    """
    return jacobian_times_vectorfield(v, w, displacement=False) \
            - jacobian_times_vectorfield(w, v, displacement=False)

def Ad(phi, v):
    r"""
    This is :math:`\Ad(\varphi,v)`, the big adjoint action of a deformation :math:`\varphi` on a
    velocity `w`.

    .. math:: \Ad(\varphi,v) = (D\varphi \circ \varphi^{-1}) v\circ \varphi^{-1}
        :label: Adphiv

    This is a tricky computation, is not commonly needed in practice and
    will not be implemented until needed.

    Given :math:`\varphi^{-1}`, :math:`\Ad(\varphi, v)` can be computed by first multiplying :math:`v` by
    :math:`D\varphi^{-1}` then splatting the components of the resulting vector field.
    """
    #DphiTv = jacobian_times_vectorfield_adjoint(phi, v, displacement=True)
    #return splat(DphiTv, phi)
    raise NotImplementedError

def ad_star(v, m):
    r"""
    This is :math:`\ad^*(v,m)`, the coadjoint action of a velocity :math:`v` on a
    vector momentum :math:`m`.

    .. math:: \ad^*(v, m) = (Dv)^T m + Dm v + m \div v

    where :math:`\div` denotes the divergence of a vector field.

    Note that this is the numerical adjoint of :math:`ad(v,.)` in :eq:`advw`,
    which is implemented using a central finite difference scheme.
    """
    return jacobian_times_vectorfield(v, m, displacement=False, transpose=True) \
         - jacobian_times_vectorfield_adjoint(m, v)

def Ad_star(phiinv, m):
    r"""
    This is :math:`\Ad^*(\varphi,m)`, the big coadjoint action of a deformation
    :math:`\varphi` on a vector momentum :math:`m`. The formula for this is

    .. math:: \Ad^*(\varphi,m)(x) = (D \varphi(x)) m(\varphi(x))

    where :math:`D` denotes the Jacobian matrix. This is the numerical adjoint
    of :math:`\Ad(\varphi, \dot)` in :eq:`Adphiv`.
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
