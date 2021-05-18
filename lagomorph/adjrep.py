r"""
Adjoint representation for $\Diff(\R^3)$.

For a derivation of these operators, and their significance, please refer to `Jacob Hinkle's Ph.D. thesis <https://core.ac.uk/download/pdf/276266157.pdf>`_, particularly Appendices A and B.

Primal adjoint actions
----------------------

.. autofunction:: ad
.. autofunction:: Ad

Dual actions
------------

.. autofunction:: ad_star
.. autofunction:: Ad_star

"Adjoint" adjoint actions (daggers)
-----------------------------------

.. autofunction:: ad_dagger
.. autofunction:: Ad_dagger

Symmetrized adjoint action
--------------------------

The sym operator and its adjoint arises when deriving reduced Jacobi fields.

.. autofunction:: sym
.. autofunction:: sym_dagger
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
    return jacobian_times_vectorfield(
        v, w, displacement=False
    ) - jacobian_times_vectorfield(w, v, displacement=False)


def Ad(phi, v):
    r"""
    This is $\Ad(\varphi,v)$, the big adjoint action of a deformation $\varphi$ on a
    velocity $w$.

    .. math:: \Ad(\varphi,v) = (D\varphi \circ \varphi^{-1}) v\circ \varphi^{-1}
        :label: Adphiv

    This is a tricky computation, is not commonly needed in practice and
    will not be implemented until needed.

    Given $\varphi^{-1}$, $\Ad(\varphi, v)$ can be computed by first multiplying $v$ by
    $D\varphi^{-1}$ then splatting the components of the resulting vector field.
    """
    # DphiTv = jacobian_times_vectorfield_adjoint(phi, v, displacement=True)
    # return splat(DphiTv, phi)
    raise NotImplementedError


def ad_star(v, m):
    r"""
    This is $\ad^*(v,m)$, the coadjoint action of a velocity $v$ on a
    vector momentum $m$.

    .. math:: \ad^*(v, m) = (Dv)^T m + Dm v + m \div v

    where $\div$ denotes the divergence of a vector field.

    Note that this is the numerical adjoint of $\ad(v, .)$ in :eq:`advw`,
    which is implemented using a central finite difference scheme.
    """
    return jacobian_times_vectorfield(
        v, m, displacement=False, transpose=True
    ) - jacobian_times_vectorfield_adjoint(m, v)


def Ad_star(phiinv, m):
    r"""
    This is $\Ad^*(\varphi, m)$, the big coadjoint action of a deformation
    $\varphi$ on a vector momentum $m$. The formula for this is

    .. math:: \Ad^*(\varphi,m)(x) = (D \varphi(x)) m(\varphi(x))

    where $D$ denotes the Jacobian matrix. This is the numerical adjoint
    of $\Ad(\varphi, \dot)$ in :eq:`Adphiv`.
    """
    mphiinv = interp(m, phiinv)
    return jacobian_times_vectorfield(phiinv, mphiinv, displacement=True)


# dagger versions of the above coadjoint operators
# The dagger indicates that instead of a _dual_ action, the _adjoint_ action
# under a metric.
# then sharping.
def ad_dagger(x, y, metric):
    r"""
    The *adjoint* of the $\ad(v,\dot)$ operator, with respect to a provided
    metric.

    This is performed by flatting, applying the dual action $\ad^*(v,\dot)$, then sharping:

    .. math:: \ad^\dagger(x, y) = \ad^*(x, y^\flat)^\sharp
    """
    return metric.sharp(ad_star(x, metric.flat(y)))


def Ad_dagger(phi, y, metric):
    r"""
    Similar to $\ad^\dagger$, but for $\Ad$.

    .. math:: \Ad^\dagger(x, y) = \Ad^*(x, y^\flat)^\sharp
    """
    return metric.sharp(Ad_star(phi, metric.flat(y)))


def sym(x, y, metric):
    r"""
    The sym operator is a negative symmetrized ad_dagger, and is important for
    computing reduced Jacobi fields.
    cf. `Bullo 1995 <http://www.cds.caltech.edu/~marsden/wiki/uploads/projects/geomech/Bullo1995.pdf>`_ or `Hinkle 2015 (PhD
    thesis) <https://core.ac.uk/download/pdf/276266157.pdf>`_

    .. math:: \sym(x, y) = \ad^\dagger(y, x) - \ad^\dagger(x, y)
        :label: sym
    """
    return -(ad_dagger(x, y, metric) + ad_dagger(y, x, metric))


def sym_dagger(x, y, metric):
    r"""
    We do not implement $\sym^*$, since sym is defined in terms of a metric
    already. This function implements the adjoint to :eq:`sym`:

    .. math:: \sym^\dagger(x, y) = \ad^\dagger(y, x) - \ad(x, y)
    """
    return ad_dagger(y, x, metric) - ad(x, y)
