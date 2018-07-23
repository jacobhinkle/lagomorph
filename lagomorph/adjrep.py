"""
Adjoint representation for Diff(R^d)
"""

class AdjRep(object):
    def __init__(self, dim):
        """
        Args:
            dim: integer, either 2 or 3
        """
        self.dim = dim
    def bigcoad(self, phi, m):
        """
        This is Ad^*, the big coadjoint action of a deformation phi on a
        vector momentum m.
        """
