import numpy as np
from collections import namedtuple
from bspy.manifold import Manifold
from bspy.solid import Solid, Boundary

@Manifold.register
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

    maxAlignment = 0.9999 # 1 - 1/10^4
    """ If the absolute value of the dot product of two unit normals is greater than maxAlignment, the manifolds are parallel."""

    def __init__(self, normal, point, tangentSpace):
        self._normal = np.atleast_1d(normal)
        self._point = np.atleast_1d(point)
        self._tangentSpace = np.atleast_1d(tangentSpace)
        if not np.allclose(self._tangentSpace.T @ self._normal, 0.0): raise ValueError("normal must be orthogonal to tangent space")

    def __repr__(self):
        return "Hyperplane({0}, {1}, {2})".format(self._normal, self._point, self._tangentSpace)

    def complete_slice(self, slice, solid):
        """
        Add any missing inherent (implicit) boundaries of this manifold's domain to the given slice of the 
        given solid that are needed to make the slice valid and complete.

        Parameters
        ----------
        slice : `Solid`
            The slice of the given solid formed by the manifold. The slice may be incomplete, missing some of the 
            manifold's inherent domain boundaries. Its dimension must match `self.domain_dimension()`.

        solid : `Solid`
            The solid being sliced by the manifold. Its dimension must match `self.range_dimension()`.

        Parameters
        ----------
        domain : `Solid`
            A domain for this manifold that may be incomplete, missing some of the manifold's inherent domain boundaries. 
            Its dimension must match `self.domain_dimension()`.

        See Also
        --------
        `Solid.slice` : Slice the solid by a manifold.

        Notes
        -----
        Since hyperplanes have no inherent domain boundaries, this operation only tests for 
        point containment for zero-dimension hyperplanes (points).
        """
        assert self.domain_dimension() == slice.dimension
        assert self.range_dimension() == solid.dimension
        if slice.dimension == 0:
            slice.containsInfinity = solid.contains_point(self._point)

    def copy(self):
        """
        Copy the hyperplane.

        Returns
        -------
        hyperplane : `Hyperplane`
        """
        return Hyperplane(self._normal, self._point, self._tangentSpace)

    @staticmethod
    def create_axis_aligned(dimension, axis, offset, flipNormal=False):
        """
        Create an axis-aligned hyperplane.

        Parameters
        ----------
        dimension : `int`
            The dimension of the hyperplane.

        axis : `int`
            The number of the axis (0 for x, 1 for y, ...).
        
        offset : `float`
            The offset from zero along the axis of a point on the hyperplane. 
        
        flipNormal : `bool`, optional
            A Boolean indicating that the normal should point toward in the negative direction along the axis. 
            Default is False, meaning the normal points in the positive direction along the axis.

        Returns
        -------
        hyperplane : `Hyperplane`
            The axis-aligned hyperplane.
        """
        assert dimension > 0
        diagonal = np.identity(dimension)
        sign = -1.0 if flipNormal else 1.0
        normal = sign * diagonal[:,axis]
        point = offset * normal
        if dimension > 1:
            tangentSpace = np.delete(diagonal, axis, axis=1)
        else:
            tangentSpace = np.array([0.0])
        
        return Hyperplane(normal, point, tangentSpace)

    @staticmethod
    def create_hypercube(bounds):
        """
        Create a solid hypercube.

        Parameters
        ----------
        bounds : array-like
            An array with shape (dimension, 2) of lower and upper and lower bounds for the hypercube. 

        Returns
        -------
        hypercube : `Solid`
            The hypercube.
        """
        bounds = np.array(bounds)
        if len(bounds.shape) != 2 or bounds.shape[1] != 2: raise ValueError("bounds must have shape (dimension, 2)")
        dimension = bounds.shape[0]
        solid = Solid(dimension, False)
        for i in range(dimension):
            if dimension > 1:
                domain = Hyperplane.create_hypercube(np.delete(bounds, i, axis=0))
            else:
                domain = Solid(0, True)
            hyperplane = Hyperplane.create_axis_aligned(dimension, i, -bounds[i][0], True)
            solid.add_boundary(Boundary(hyperplane, domain))
            hyperplane = Hyperplane.create_axis_aligned(dimension, i, bounds[i][1], False)
            solid.add_boundary(Boundary(hyperplane, domain))

        return solid

    def domain_dimension(self):
        """
        Return the domain dimension.

        Returns
        -------
        dimension : `int`
        """
        return len(self._normal) - 1

    def evaluate(self, domainPoint):
        """
        Return the value of the manifold (a point on the manifold).

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the point.

        Returns
        -------
        point : `numpy.array`
        """
        return np.dot(self._tangentSpace, np.atleast_1d(domainPoint)) + self._point

    def flip_normal(self):
        """
        Flip the direction of the normal.

        Returns
        -------
        hyperplane : `Hyperplane`
            The hyperplane with flipped normal. The hyperplane retains the same tangent space.

        See Also
        --------
        `Solid.complement` : Return the complement of the solid: whatever was inside is outside and vice-versa.
        """
        return Hyperplane(-self._normal, self._point, self._tangentSpace)

    @staticmethod
    def from_dict(dictionary):
        """
        Create a `Hyperplane` from a data in a `dict`.

        Parameters
        ----------
        dictionary : `dict`
            The `dict` containing `Hyperplane` data.

        Returns
        -------
        hyperplane : `hyperplane`

        See Also
        --------
        `to_dict` : Return a `dict` with `Hyperplane` data.
        """
        return Hyperplane(dictionary["normal"], dictionary["point"], dictionary["tangentSpace"])

    def full_domain(self):
        """
        Return a solid that represents the full domain of the hyperplane.

        Returns
        -------
        domain : `Solid`
            The full (untrimmed) domain of the hyperplane.

        See Also
        --------
        `Boundary` : A portion of the boundary of a solid.
        """
        return Solid(self.domain_dimension(), True)

    def intersect(self, other):
        """
        Intersect two hyperplanes.

        Parameters
        ----------
        other : `Hyperplane`
            The `Hyperplane` intersecting the hyperplane.

        Returns
        -------
        intersections : `list` (or `NotImplemented` if other is not a `Hyperplane`)
            A list of intersections between the two hyperplanes. 
            (Hyperplanes will have at most one intersection, but other types of manifolds can have several.)
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
        `Solid.slice` : slice the solid by a manifold.
        `numpy.linalg.svd` : Compute the singular value decomposition of a matrix array.

        Notes
        -----
        Hyperplanes are parallel when their unit normals are aligned (dot product is nearly 1 or -1). Otherwise, they cross each other.

        To solve the crossing, we find the intersection by solving the underdetermined system of equations formed by assigning points 
        in one hyperplane (`self`) to points in the other (`other`). That is: 
        `self._tangentSpace * selfDomainPoint + self._point = other._tangentSpace * otherDomainPoint + other._point`. This system is `dimension` equations
        with `2*(dimension-1)` unknowns (the two domain points).
        
        There are more unknowns than equations, so it's underdetermined. The number of free variables is `2*(dimension-1) - dimension = dimension-2`.
        To solve the system, we rephrase it as `Ax = b`, where `A = (self._tangentSpace -other._tangentSpace)`, `x = (selfDomainPoint otherDomainPoint)`, 
        and `b = other._point - self._point`. Then we take the singular value decomposition of `A = U * Sigma * VTranspose`, using `numpy.linalg.svd`.
        The particular solution for x is given by `x = V * SigmaInverse * UTranspose * b`,
        where we only consider the first `dimension` number of vectors in `V` (the rest are zeroed out, i.e. the null space of `A`).
        The null space of `A` (the last `dimension-2` vectors in `V`) spans the free variable space, so those vectors form the tangent space of the intersection.
        Remember, we're solving for `x = (selfDomainPoint otherDomainPoint)`. So, the selfDomainPoint is the first `dimension-1` coordinates of `x`,
        and the otherDomainPoint is the last `dimension-1` coordinates of `x`. Likewise for the two tangent spaces.

        For coincident regions, we need the domains, normal alignment, and mapping from the hyperplane's domain to the other's domain. (The mapping is irrelevant and excluded for dimensions less than 2.)
        We can tell if the two hyperplanes are coincident if their normal alignment (dot product of their unit normals) is nearly 1 
        in absolute value (`alignment**2 < Hyperplane.maxAlignment`) and their points are barely separated:
        `-2 * Manifold.minSeparation < dot(hyperplane._normal, hyperplane._point - other._point) < Manifold.minSeparation`. (We give more room 
        to the outside than the inside to avoid compounding issues from minute gaps.)

        Since hyperplanes are flat, the domains of their coincident regions are the entire domain: `Solid(domain dimension, True)`.
        The normal alignment is the dot product of the unit normals. The mapping from the hyperplane's domain to the other's domain is derived
        from setting the hyperplanes to each other: 
        `hyperplane._tangentSpace * selfDomainPoint + hyperplane._point = other._tangentSpace * otherDomainPoint + other._point`. Then solve for
        `otherDomainPoint = inverse(transpose(other._tangentSpace) * other._tangentSpace)) * transpose(other._tangentSpace) * (hyperplane._tangentSpace * selfDomainPoint + hyperplane._point - other._point)`.
        You get the transform is `inverse(transpose(other._tangentSpace) * other._tangentSpace)) * transpose(other._tangentSpace) * hyperplane._tangentSpace`,
        and the translation is `inverse(transpose(other._tangentSpace) * other._tangentSpace)) * transpose(other._tangentSpace) * (hyperplane._point - other._point)`.

        Note that to invert the mapping to go from the other's domain to the hyperplane's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        if not isinstance(other, Hyperplane):
            return NotImplemented
        assert self.range_dimension() == other.range_dimension()

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []
        dimension = self.range_dimension()

        # Check if manifolds intersect (are not parallel)
        alignment = np.dot(self._normal, other._normal)
        if alignment * alignment < Hyperplane.maxAlignment:
            # We're solving the system Ax = b using singular value decomposition, 
            #   where A = (self._tangentSpace -other._tangentSpace), x = (selfDomainPoint otherDomainPoint), and b = other._point - self._point.
            # Construct A.
            A = np.concatenate((self._tangentSpace, -other._tangentSpace),axis=1)
            # Compute the singular value decomposition of A.
            U, sigma, VTranspose = np.linalg.svd(A)
            # Compute the inverse of Sigma and transpose of V.
            SigmaInverse = np.diag(np.reciprocal(sigma))
            V = np.transpose(VTranspose)
            # Compute x = V * SigmaInverse * UTranspose * (other._point - self._point)
            x = V[:, 0:dimension] @ SigmaInverse @ np.transpose(U) @ (other._point - self._point)

            # The self intersection normal is just the dot product of other normal with the self tangent space.
            selfDomainNormal = np.dot(other._normal, self._tangentSpace)
            selfDomainNormal = selfDomainNormal / np.linalg.norm(selfDomainNormal)
            # The other intersection normal is just the dot product of self normal with the other tangent space.
            otherDomainNormal = np.dot(self._normal, other._tangentSpace)
            otherDomainNormal = otherDomainNormal / np.linalg.norm(otherDomainNormal)
            # The self intersection point is the first dimension-1 coordinates of x.
            selfDomainPoint = x[0:dimension-1]
            # The other intersection point is the last dimension-1 coordinates of x.
            otherDomainPoint = x[dimension-1:]
            if dimension > 2:
                # The self intersection tangent space is the first dimension-1 coordinates of the null space (the last dimension-2 vectors in V).
                selfDomainTangentSpace = V[0:dimension-1, dimension:]
                # The other intersection tangent space is the last dimension-1 coordinates of the null space (the last dimension-2 vectors in V).
                otherDomainTangentSpace = V[dimension-1:, dimension:]
            else:
                # There is no null space (dimension-2 <= 0)
                selfDomainTangentSpace = np.array([0.0])
                otherDomainTangentSpace = np.array([0.0])
            intersections.append(Manifold.Crossing(Hyperplane(selfDomainNormal, selfDomainPoint, selfDomainTangentSpace), Hyperplane(otherDomainNormal, otherDomainPoint, otherDomainTangentSpace)))

        # Otherwise, manifolds are parallel. Now, check if they are coincident.
        else:
            insideSeparation = np.dot(self._normal, self._point - other._point)
            # Allow for extra outside separation to avoid issues with minute gaps.
            if -2.0 * Hyperplane.minSeparation < insideSeparation < Hyperplane.minSeparation:
                # These hyperplanes are coincident. Return the domains in which they coincide (entire domain for hyperplanes) and the normal alignment.
                domainCoincidence = Solid(dimension-1, True)
                if dimension > 1:
                    # For non-zero domains, also return the mapping from the self domain to the other domain.
                    tangentSpaceTranspose = np.transpose(other._tangentSpace)
                    map = np.linalg.inv(tangentSpaceTranspose @ other._tangentSpace) @ tangentSpaceTranspose
                    transform =  map @ self._tangentSpace
                    inverseTransform = np.linalg.inv(transform)
                    translation = map @ (self._point - other._point)
                    intersections.append(Manifold.Coincidence(domainCoincidence, domainCoincidence, alignment, transform, inverseTransform, translation))
                else:
                    intersections.append(Manifold.Coincidence(domainCoincidence, domainCoincidence, alignment, None, None, None))

        return intersections

    def normal(self, domainPoint, normalize=True, indices=None):
        """
        Return the normal.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the normal.
        
        normalize : `boolean`, optional
            If True the returned normal will have unit length (the default). Otherwise, the normal's length will
            be the area of the tangent space (for two independent variables, its the length of the cross product of tangent vectors).
        
        indices : `iterable`, optional
            An iterable of normal indices to calculate. For example, `indices=(0, 3)` will return a vector of length 2
            with the first and fourth values of the normal. If `None`, all normal values are returned (the default).

        Returns
        -------
        normal : `numpy.array`
        """
        if normalize:
            normal = self._normal
        else:
            # Compute and cache cofactor normal on demand.
            if not hasattr(self, '_cofactorNormal'):
                dimension = self.range_dimension()
                if dimension > 1:
                    minor = np.zeros((dimension-1, dimension-1))
                    self._cofactorNormal = np.array(self._normal) # We change it, so make a copy.
                    sign = 1.0
                    for i in range(dimension):
                        if i > 0:
                            minor[0:i, :] = self._tangentSpace[0:i, :]
                        if i < dimension - 1:
                            minor[i:, :] = self._tangentSpace[i+1:, :]
                        self._cofactorNormal[i] = sign * np.linalg.det(minor)
                        sign *= -1.0
            
                    # Ensure cofactorNormal points in the same direction as normal.
                    if np.dot(self._cofactorNormal, self._normal) < 0.0:
                        self._cofactorNormal = -self._cofactorNormal
                else:
                    self._cofactorNormal = self._normal
            
            normal = self._cofactorNormal
        
        return normal if indices is None else normal[(indices,)]

    def range_bounds(self):
        """
        Return the range bounds for the hyperplane.

        Returns
        -------
        rangeBounds : `np.array` or `None`
            The range of the hyperplane given as lower and upper bounds on each dependent variable. 
            If the hyperplane has an unbounded range (domain dimension > 0), `None` is returned.
        """
        return None if self.domain_dimension() > 0 else np.array(((self._point[0], self._point[0]),))

    def range_dimension(self):
        """
        Return the range dimension.

        Returns
        -------
        dimension : `int`
        """
        return len(self._normal)

    def tangent_space(self, domainPoint):
        """
        Return the tangent space.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the tangent space.

        Returns
        -------
        tangentSpace : `numpy.array`
        """
        return self._tangentSpace

    def to_dict(self):
        """
        Return a `dict` with `Hyperplane` data.

        Returns
        -------
        dictionary : `dict`

        See Also
        --------
        `from_dict` : Create a `Hyperplane` from a data in a `dict`.
        """
        return {"type" : "Hyperplane", "normal" : self._normal, "point" : self._point, "tangentSpace" : self._tangentSpace}

    def transform(self, matrix, matrixInverseTranspose = None):
        """
        Transform the range of the hyperplane.

        Parameters
        ----------
        matrix : `numpy.array`
            A square matrix transformation.

        matrixInverseTranspose : `numpy.array`, optional
            The inverse transpose of matrix (computed if not provided).

        Returns
        -------
        hyperplane : `Hyperplane`
            The transformed hyperplane.

        See Also
        --------
        `Solid.transform` : Transform the range of the solid.
        """
        if self.range_dimension() > 1:
            if matrixInverseTranspose is None:
                matrixInverseTranspose = np.transpose(np.linalg.inv(matrix))

            normal = matrixInverseTranspose @ self._normal
            normal = normal / np.linalg.norm(normal)
            hyperplane = Hyperplane(normal, matrix @ self._point, matrix @ self._tangentSpace)
        else:
            hyperplane = Hyperplane(self._normal, matrix @ self._point, self._tangentSpace)
    
        return hyperplane
    
    def translate(self, delta):
        """
        translate the range of the hyperplane.

        Parameters
        ----------
        delta : `numpy.array`
            A 1D array translation.

        Returns
        -------
        hyperplane : `Hyperplane`
            The translated hyperplane.

        See Also
        --------
        `Solid.translate` : translate the range of the solid.
        """
        return Hyperplane(self._normal, self._point + delta, self._tangentSpace)

    def trimmed_range_bounds(self, domainBounds):
        """
        Return the trimmed range bounds for the hyperplane.

        Parameters
        ----------
        domainBounds : array-like or `None`
            An array with shape (domain_dimension, 2) of lower and upper and lower bounds on each hyperplane parameter. 
            If domainBounds is `None` then the hyperplane is unbounded.

        Returns
        -------
        trimmedManifold, rangeBounds : `Hyperplane`, `np.array` (or None)
            A manifold trimmed to the given domain bounds, and the range of the trimmed hyperplane given as 
            lower and upper bounds on each dependent variable. If the domain bounds are `None` (meaning unbounded) 
            then rangeBounds is `None`.  

        Notes
        -----
        The returned trimmed manifold is the original hyperplane (no changes).
        """
        if domainBounds is None:
            return self, self.range_bounds()
        else:
            domainBounds = np.atleast_1d(domainBounds)
            rangeBounds = [[value.min(), value.max()] for value in 
                self._tangentSpace @ domainBounds + self._point.reshape(self._point.shape[0], 1)]
            return self, np.array(rangeBounds, self._normal.dtype)