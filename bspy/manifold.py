import numpy as np
from collections import namedtuple

class Manifold:
    """
    A manifold is an abstract base class for differentiable functions with
    normals and tangent spaces whose range is one dimension higher than their domain.
    """

    minSeparation = 0.01
    """If two points are within 0.01 of each each other, they are coincident."""

    Crossing = namedtuple('Crossing', ('left','right'))
    Coincidence = namedtuple('Coincidence', ('left', 'right', 'alignment', 'transform', 'inverse', 'translation'))
    """Return type for intersect."""

    def __init__(self):
        pass

    def intersect(self, other):
        """
        Intersect two manifolds (self and other).

        Parameters
        ----------
        other : `Manifold`
            The `Manifold` intersecting self.

        Returns
        -------
        intersections : `list` (or `NotImplemented` if other is an unknown type of Manifold)
            A list of intersections between the two manifolds. 
            Each intersection records either a crossing or a coincident region.

            For a crossing, intersection is a `Manifold.Crossing`: (left, right)
            * left : `Manifold` in the manifold's domain where the manifold and the other cross.
            * right : `Manifold` in the other's domain where the manifold and the other cross.
            * Both intersection manifolds have the same domain and range (the crossing between the manifold and the other).

            For a coincident region, intersection is a `Manifold.Coincidence`: (left, right, alignment, transform, inverse, translation)
            * left : `Solid` in the manifold's domain within which the manifold and the other are coincident.
            * right : `Solid` in the other's domain within which the manifold and the other are coincident.
            * alignment : scalar value holding the normal alignment between the manifold and the other (the dot product of their unit normals).
            * transform : `numpy.array` holding the transform matrix from the manifold's domain to the other's domain.
            * inverse : `numpy.array` holding the inverse transform matrix from the other's domain to the boundary's domain.
            * translation : `numpy.array` holding the translation vector from the manifold's domain to the other's domain.
            * Together transform, inverse, and translation form the mapping from the manifold's domain to the other's domain and vice-versa.

        See Also
        --------
        `cached_intersect_manifold` : Intersect two manifolds, caching the result for twins (same intersection but swapping self and other).
        `solid.Solid.slice` : slice the solid by a manifold.

        Notes
        -----
        To invert the mapping to go from the other's domain to the manifold's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        return NotImplemented

class Hyperplane(Manifold):
    """
    A hyperplane is a `Manifold` defined by a unit normal, a point on the hyperplane, and a tangent space orthogonal to the normal.

    Parameters
    ----------
    normal : array-like
        The unit normal.
    
    point : array-like
        A point on the hyperplane.
    
    tangentSpace : array-like
        A array of tangents that are linearly independent and orthogonal to the normal.
    
    Notes
    -----
    The number of coordinates in the normal defines the dimension of the range of the hyperplane. The point must have the same dimension. The tangent space must be shaped: (dimension, dimension-1). 
    Thus the dimension of the domain is one less than that of the range.
    """

    maxAlignment = 0.99 # 1 - 1/10^2
    """If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel."""

    def __init__(self, normal, point, tangentSpace):
        self._normal = np.atleast_1d(np.array(normal))
        self._point = np.atleast_1d(np.array(point))
        self._tangentSpace = np.atleast_1d(np.array(tangentSpace))
        if not np.allclose(self._tangentSpace.T @ self._normal, 0.0): raise ValueError("normal must be orthogonal to tangent space")
