"""
Adjoint representation for Diff(R^d)
"""
from pycuda import gpuarray

from .diff import grad 
from .deform import interp_def

class AdjRep(object):
    def __init__(self, dim):
        """
        Args:
            dim: integer, either 2 or 3
        """
        self.dim = dim
    def ad(self, v, w):
        """
        This is ad(v,w), the adjoint action of a velocity v on a
        velocity w.
        """
        raise NotImplementedError("not implemented yet")
    def Ad(self, phi, w):
        """
        This is Ad(phi,w), the big adjoint action of a deformation phi on a
        velocity w.
        """
        raise NotImplementedError("not implemented yet")
    def ad_star(self, v, m):
        """
        This is ad^*(v,m), the coadjoint action of a velocity v on a
        vector momentum m.
        """
        raise NotImplementedError("not implemented yet")
    def Ad_star(self, phi, m):
        """
        This is Ad^*(phi,m), the big coadjoint action of a deformation phi on a
        vector momentum m. The formula for this is

            Ad^*(phi,m)(x) = (D phi(x)) m(phi(x))

        where D denotes the Jacobian matrix.
        """
        # First interpolate m
        mphi = interp_def(m, phi)
        if not isinstance(phi, gpuarray.GPUArray):
            phi = gpuarray.to_gpu(phi)
        if not isinstance(m, gpuarray.GPUArray):
            m = gpuarray.to_gpu(m)
        # For each dimension, compute the point-wise dot product between the
        # gradient of that coordinate map of phi and mphi
        out = gpuarray.zeros_like(m)
        for d in range(phi.shape[1]):
            gphi = grad(phi[:,d,...])
            gphi *= mphi
            out[:,d,:] = gphi[:,0,...]
            for dd in range(1, gphi.shape[1]):
                out[:,d,:] += gphi[:,dd,...]
        return out
    # dagger versions
    def ad_dagger(self, x, y, K):
        return K.inverse(self.ad_star(x, K.forward(y)))
    def sym_dagger(self, x, y, K):
        return self.ad_dagger(y, x, K) - self.ad(x, y)
