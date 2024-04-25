import numpy as np
from os import path
import json
from bspy.manifold import Manifold
import bspy.spline_block
import bspy._spline_domain
import bspy._spline_evaluation
import bspy._spline_intersection
import bspy._spline_fitting
import bspy._spline_operations

@Manifold.register
class Spline(Manifold):
    """
    A class to model, represent, and process piecewise polynomial tensor product
    functions (splines) as linear combinations of B-splines. 

    Parameters
    ----------
    nInd : `int`
        The number of independent variables of the spline.

    nDep : `int`
        The number of dependent variables of the spline.
    
    order : `tuple`
        A tuple of length nInd where each integer entry represents the
        polynomial order of the spline in that variable.

    nCoef : `tuple`
        A tuple of length nInd where each integer entry represents the
        dimension (i.e. number of B-spline coefficients) of the spline
        in that variable.

    knots : `list`
        A list of the lists of the knots of the spline in each independent variable.

    coefs : array-like
        A list of the B-spline coefficients of the spline.
    
    metadata : `dict`, optional
        A dictionary of ancillary data to store with the spline. Default is {}.
    """
    
    def __init__(self, nInd, nDep, order, nCoef, knots, coefs, metadata = {}):
        if not(nInd >= 0): raise ValueError("nInd < 0")
        self.nInd = int(nInd)
        if not(nDep >= 0): raise ValueError("nDep < 0")
        self.nDep = int(nDep)
        if not(len(order) == self.nInd): raise ValueError("len(order) != nInd")
        self.order = tuple(int(x) for x in order)
        if not(len(nCoef) == self.nInd): raise ValueError("len(nCoef) != nInd")
        self.nCoef = tuple(int(x) for x in nCoef)
        if not(len(knots) == nInd): raise ValueError("len(knots) != nInd")
        for i in range(len(knots)):
            nKnots = self.order[i] + self.nCoef[i]
            if not(len(knots[i]) == nKnots):
                raise ValueError(f"Knots array for variable {i} should have length {nKnots}")
        self.knots = tuple(np.array(kk) for kk in knots)
        for knots, order, nCoef in zip(self.knots, self.order, self.nCoef):
            for i in range(nCoef):
                if not(knots[i] <= knots[i + 1] and knots[i] < knots[i + order]):
                       raise ValueError("Improperly ordered knot sequence")
        totalCoefs = 1
        for nCoef in self.nCoef:
            totalCoefs *= nCoef
        if not(len(coefs) == totalCoefs or len(coefs) == self.nDep):
            raise ValueError(f"Length of coefs should be {totalCoefs} or {self.nDep}")
        self.coefs = np.array(coefs)
        if self.coefs.shape != (self.nDep, *self.nCoef):
            if len(self.coefs) == totalCoefs:
                self.coefs = self.coefs.reshape((*self.nCoef[::-1], self.nDep)).T
            else:
                self.coefs = np.array([c.T for c in self.coefs]).reshape((self.nDep, *self.nCoef))
        self.metadata = dict(metadata)

    def __call__(self, *uvw, **kwargs):
        return self.evaluate(*uvw, **kwargs)

    def __repr__(self):
        return f"Spline({self.nInd}, {self.nDep}, {self.order}, " + \
               f"{self.nCoef}, {self.knots} {self.coefs}, {self.metadata})"

    def __add__(self, other):
        if isinstance(other, Spline):
            return self.add(other, [(ix, ix) for ix in range(min(self.nInd, other.nInd))])
        else:
            return self.translate(other)

    def __radd__(self, other):
        if isinstance(other, Spline):
            return other.add(self, [(ix, ix) for ix in range(min(self.nInd, other.nInd))])
        else:
            return self.translate(other)

    def __matmul__(self, other):
        if isinstance(other, Spline):
            return self.multiply(other, [(ix, ix) for ix in range(min(self.nInd, other.nInd))], 'D')
        else:
            other = np.atleast_1d(other)
            if len(other.shape) > 1:
                return self.transform(other.T)
            else:
                return self.dot(other)

    def __rmatmul__(self, other):
        if isinstance(other, Spline):
            return other.multiply(self, [(ix, ix) for ix in range(min(self.nInd, other.nInd))], 'D')
        else:
            other = np.atleast_1d(other)
            if len(other.shape) > 1:
                return self.transform(other)
            else:
                return self.dot(other)

    def __mul__(self, other):
        if isinstance(other, Spline):
            return self.multiply(other, [(ix, ix) for ix in range(min(self.nInd, other.nInd))], 'S')
        else:
            return self.scale(other)

    def __rmul__(self, other):
        if isinstance(other, Spline):
            return other.multiply(self, [(ix, ix) for ix in range(min(self.nInd, other.nInd))], 'S')
        else:
            return self.scale(other)

    def __sub__(self, other):
        if isinstance(other, Spline):
            return self.subtract(other, [(ix, ix) for ix in range(min(self.nInd, other.nInd))])
        else:
            return self.translate(-np.atleast_1d(other))

    def __rsub__(self, other):
        if isinstance(other, Spline):
            return other.subtract(self, [(ix, ix) for ix in range(min(self.nInd, other.nInd))])
        else:
            spline = self.scale(-1.0)
            return spline.translate(other)
    
    def __truediv__(self, other):
        if not np.isscalar(other):  raise ValueError('Divisor must be a scalar')
        return self * (1.0 / other)
    
    def add(self, other, indMap = None):
        """
        Add two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to add to self. The number of dependent variables must match self.

        indMap : `iterable` or `None`, optional
            An iterable of indices or pairs of indices. Each index refers to an independent variable.
            Within the iterable, a single index, `n`, maps the nth independent variable of self to the same independent variable of other.
            A pair `(n, m)` maps the nth independent variable of self to the mth independent variable of other. 
            For example, if you wanted to compute `self(u, v, w) + other(u, w)`, you'd pass `[0, (2, 1)]` for `indMap`. 
            Unmapped independent variables remain independent (the default).
            The domains of mapped independent variables must match. 
            An independent variable can map to no more than one other independent variable.

        Returns
        -------
        spline : `Spline`
            The result of adding self and other.

        See Also
        --------
        `subtract` : Subtract two splines.
        `multiply` : Multiply two splines.
        `common_basis` : Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Notes
        -----
        Equivalent to spline1 + spline2 with indMap = range(min(spline1.nInd, spline2.nInd)).
        
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
        if indMap is not None:
            indMap = [(mapping, mapping) if np.isscalar(mapping) else mapping for mapping in indMap]
        return bspy._spline_operations.add(self, other, indMap)

    @staticmethod
    def bspline_values(knot, knots, splineOrder, u, derivativeOrder = 0, taylorCoefs = False):
        """
        Compute bspline values or their derivatives for a 1D bspline segment given the rightmost knot of the segment and a parameter value within that segment.

        Parameters
        ----------
        knot : `int`
            The rightmost knot in the bspline segment.  If knot is None, then this value is
            determined by binary search

        knots : array-like
            The array of knots for the bspline.
        
        splineOrder : `int`
            The order of the bspline.

        u : `float`
            The parameter value within the segment at which to evaluate it.
        
        derivativeOrder : `int`, optional
            The order of the derivative. A zero-order derivative (default) just evaluates the bspline normally.
        
        taylorCoefs : `boolean`, optional
            A boolean flag that if true returns the derivatives divided by their degree factorial, that is 
            the taylor coefficients at the given parameter values. Default is false.

        Returns
        -------
        knot : `int`
            The rightmost knot in the bspline segment.  If this is specified on input, then this
            valued is returned.  Otherwise, it is computed and then returned.

        value : `numpy.array`
            The value of the bspline or its derivative at the given parameter.

        See Also
        --------
        `evaluate` : Compute the value of the spline at given parameter values.
        `derivative` : Compute the derivative of the spline at given parameter values.

        Notes
        -----
        This method does not check parameter values. It is used by other evaluation methods. It uses the de Boor recurrence relations for a B-spline.
        """
        return bspy._spline_evaluation.bspline_values(knot, knots, splineOrder, u, derivativeOrder, taylorCoefs)

    @staticmethod
    def circular_arc(radius, angle, tolerance = None):
        """
        Create a 2D circular arc for a given radius and angle accurate to within a given tolerance.

        Parameters
        ----------
        radius : `float`
            The radius of the circular arc.

        angle : `float`
            The angle of the circular arc measured in degrees starting at the x-axis rotating counterclockwise.

        tolerance : `float`, optional
            The maximum allowed error in the circular arc (default is machine epsilon).

        Returns
        -------
        spline : `Spline`
            A spline curve for the circular arc (domain from 0 to 1).

        See Also
        --------
        `section` : Fit a planar section to the list of 4-tuples of data.
        `revolve` : Rotate the spline to create a surface of revolution.
        """
        return bspy._spline_fitting.circular_arc(radius, angle, tolerance)

    def clamp(self, left, right):
        """
        Ensure the leftmost/rightmost knot has a full order multiplicity, clamping the spline's 
        value at the first/last knot to its first/last coefficient.

        Parameters
        ----------
        left : `iterable`
            An iterable of independent variables to clamp on the left side.

        right : `iterable`
            An iterable of independent variables to clamp on the right side.

        Returns
        -------
        spline : `Spline`
            The clamped spline. If the spline was already clamped, the original spline is returned.

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `trim` : Trim the domain of a spline.
        """
        return bspy._spline_domain.clamp(self, left, right)

    @staticmethod
    def common_basis(splines, indMap = None):
        """
        Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Parameters
        ----------
        splines : `iterable`
            The collection of N splines to align.

        indMap : `iterable`
            The collection of independent variables to align. Since each spline can have multiple 
            independent variables, `indMap` is an `iterable` of `iterables` (like a list of lists). 
            Each collection of indices (i0, i1, .. iN) maps the i'th independent variable to each other. 
            The domains of mapped independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            If all the splines are curves (1 independent variable), then `indMap` is ((0, 0, .. 0),).
            If indMap is None (the default), then it is set to [N * [i] for i in range(nInd)]

        Returns
        -------
        splines : `tuple`
            The aligned collection of N splines.

        See Also
        --------
        `elevate_and_insert_knots` : Elevate a spline and insert new knots.

        Notes
        -----
        Uses `elevate_and_insert_knots` to ensure mapped variables share the same order and knots. 
        """
        return bspy._spline_domain.common_basis(splines, indMap)

    def complete_slice(self, slice, solid):
        """
        Add any missing inherent (implicit) boundaries of this spline's domain to the given slice of the 
        given solid that are needed to make the slice valid and complete.

        Parameters
        ----------
        slice : `solid.Solid`
            The slice of the given solid formed by the spline. The slice may be incomplete, missing some of the 
            spline's inherent domain boundaries. Its dimension must match `self.domain_dimension()`.

        solid : `solid.Solid`
            The solid being sliced by the manifold. Its dimension must match `self.range_dimension()`.

        See Also
        --------
        `solid.Solid.slice` : slice the solid by a manifold.
        `domain` : Return the domain of a spline.

        Notes
        -----
        A spline's inherent domain is determined by its knot array for each dimension. This method only works for 
        nInd of 1 or 2.
        """
        if self.domain_dimension() != slice.dimension: raise ValueError("Spline domain dimension must match slice dimension")
        if self.range_dimension() != solid.dimension: raise ValueError("Spline range dimension must match solid dimension")
        if slice.dimension != 1 and slice.dimension != 2: raise ValueError("Only works for nInd = 1 or 2")
        return bspy._spline_intersection.complete_slice(self, slice, solid)

    @staticmethod
    def cone(radius1, radius2, height, tolerance = None):
        """
        Construct a cone of the two given radii and height.

        Parameters
        ----------
        radius1 : scalar
            The desired radius of the cone at the bottom
        
        radius2 : scalar
            The desired radius of the cone at the top
        
        height : scalar
            The desired height of the cone

        tolerance : scalar, optional
            The desired absolute tolerance to which the cylinder should be constructed.  Defaults
            to 1.0e-12 if tolerance == None.

        Returns
        -------
        spline : `Spline`
            A bi-quartic spline approximation to a cone of the specified radii and height,
            accurate to the given tolerance.
        
        Notes
        -----
        The resulting mapping is defined over the unit square.  The first independent variable
        corresponds to the radial position on the cone, while the second independent variable
        is the location along the height.
    
        See Also
        --------
        `circular_arc` : Create a 2D circular arc for a given radius and angle accurate to within
                         a given tolerance.
        `ruled_surface` : Construct a ruled surface between two curves.
        """
        return bspy._spline_fitting.cone(radius1, radius2, height, tolerance)

    def confine(self, range_bounds):
        """
        Confine the range of a curve to the given bounds.

        Parameters
        ----------
        range_bounds : `iterable`
            The collection of `nDep` tuples that specify the lower and upper bounds for the curve.

        Returns
        -------
        spline : `Spline`
            The confined spline. 

        See Also
        --------
        `range_bounds` : Return the range of a spline as lower and upper bounds.
        `contour` : Fit a spline to the contour defined by `F(x) = 0`.

        Notes
        -----
        Only works for curves (`nInd == 1`). Portions of the curve that lie outside the bounds 
        become lines along the boundary.
        """
        return bspy._spline_operations.confine(self, range_bounds)

    @staticmethod
    def contour(F, knownXValues, dF = None, epsilon = None, metadata = {}):
        """
        Fit a spline to the contour defined by `F(x) = 0`, where `F` maps n dimensions to 
        n-1 dimensions. Thus, the solution, `x(t)`, is a contour curve (one degree of freedom) 
        returned in the form of a spline.

        Parameters
        ----------
        F : function or `Spline`
            A function or spline that takes an array-like argument of length `n` and returns an 
            array-like result of length `n - 1`.

        knownXValues : `iterable` of array-like
            An `iterable` of known x values (array-like) that lie on the desired contour. 
            The length of `knownXValues` must be at least 2 (for the two boundary conditions). 
            All x values must be length `n` and be listed in the order they appear on the contour.  
            `F(x)` for all known x values must be a zero vector of length `n-1`.

        dF : `iterable` or `None`, optional
            An `iterable` of the `n` functions representing the `n` first derivatives of `F`. 
            If `dF` is `None` (the default), the first derivatives will be computed for you. 
            If `F` is not a spline, computing the first derivatives involves multiple calls to `F` 
            and can be numerically unstable. 

        epsilon : `float`, optional
            Tolerance for contour precision. Evaluating `F` with contour values will be within epsilon 
            of zero. The default is square root of machine epsilon.
            
        metadata : `dict`, optional
            A dictionary of ancillary data to store with the spline. Default is {}.

        Returns
        -------
        spline : `Spline`
            The spline contour, `x(t)`, with nInd == 1 and nDep == n.

        See Also
        --------
        `least_squares` : Fit a spline to an array of data points using the method of least squares.
        `contours` : Find all the contour curves of a spline.
        `confine` : Confine the range of a curve to the given bounds.

        Notes
        -----
        The returned spline has constant parametric speed (the length of its derivative is constant). 
        If `F` is a `Spline`, then the range of the returned contour is confined to the domain of `F`. 
        Implements the algorithm described in section 7 of Grandine, Thomas A. 
        "Applications of contouring." Siam Review 42, no. 2 (2000): 297-316.
        """
        return bspy._spline_fitting.contour(F, knownXValues, dF, epsilon, metadata)

    def contours(self):
        """
        Find all the contour curves of a spline whose `nInd` is one larger than its `nDep`.

        Returns
        -------
        curves : `iterable`
            A collection of `Spline` curves, `u(t)`, each of whose domain is [0, 1], whose range is
            in the parameter space of the given spline, and which satisfy `self(u(t)) = 0`. 

        See Also
        --------
        `zeros` : Find the roots of a spline (nInd must match nDep).
        `contour` : Fit a spline to the contour defined by `F(x) = 0`.
        `intersect` : Intersect two splines.

        Notes
        -----
        Uses `zeros` to find all intersection points and `contour` to find individual intersection curves. 
        The algorithm used to to find all intersection curves is from Grandine, Thomas A., and Frederick W. Klein IV. 
        "A new approach to the surface intersection problem." Computer Aided Geometric Design 14, no. 2 (1997): 111-134.
        """
        return bspy._spline_intersection.contours(bspy.spline_block.SplineBlock(self))

    def contract(self, uvw):
        """
        Contract a spline by assigning a fixed value to one or more of its independent variables.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable to contract.
            A value of `None` for an independent variable indicates that variable should remain unchanged.

        Returns
        -------
        spline : `Spline`
            The contracted spline.

        See Also
        --------
        `evaluate` : Compute the value of the spline at given parameter values.
        """
        return bspy._spline_operations.contract(self, uvw)

    def convolve(self, other, indMap = None, productType = 'S'):
        """
        Convolve two splines (cross, dot, or scalar product).

        Parameters
        ----------
        other : `Spline`
            The spline to convolve with self.

        indMap : `iterable` or `None`, optional
            An iterable of indices or pairs of indices. Each index refers to an independent variable.
            Within the iterable, a single index, `n`, maps the nth independent variable of self to the same independent variable of other.
            A pair `(n, m)` maps the nth independent variable of self to the mth independent variable of other. 
            For example, if you wanted to convolve `self(u, v, w)` with `other(u, w)`, you'd pass `[0, (2, 1)]` for `indMap`. 
            Unmapped independent variables remain independent (the default).
            An independent variable can map to no more than one other independent variable.

        productType : {'C', 'D', 'S'}, optional
            The type of product to perform on the dependent variables (default is 'S').
                'C' is for a cross product, self x other (nDep must be 2 or 3).
                'D' is for a dot product (nDep must match).
                'S' is for a scalar product (nDep must match or be 1 for one of the splines).
        
        Returns
        -------
        spline : `Spline`
            The result of convolving self with other.

        See Also
        --------
        `multiply` : Multiply two splines (cross, dot, or scalar product).
        `integrate` : Integrate a spline with respect to one of its independent variables, returning the resulting spline.

        Notes
        -----
        Taken in part from Lee, E. T. Y. "Computing a chain of blossoms, with application to products of splines." 
        Computer Aided Geometric Design 11, no. 6 (1994): 597-620.
        """
        if indMap is not None:
            indMap = [(mapping, mapping, True) if np.isscalar(mapping) else (*mapping, True) for mapping in indMap]
        return bspy._spline_operations.multiplyAndConvolve(self, other, indMap, productType)

    def copy(self):
        """
        Create a copy of a spline.
        
        Returns
        -------
        spline : `Spline`
            The spline copy.
        """
        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, self.coefs, self.metadata)

    def cross(self, vector):
        """
        Cross product a spline with `nDep` of 2 or 3 by the given vector.

        Parameters
        ----------
        vector : array-like or `Spline`
            An array of length 2 or 3 or spline with `nDep` of 2 or 3 that specifies the vector.

        Returns
        -------
        spline : `Spline`
            The crossed spline: self x vector.

        See Also
        --------
        `multiply` : Multiply two splines (cross, dot, or scalar product).
        """
        return bspy._spline_operations.cross(self, vector)

    def curvature(self, uvw):
        """
        Compute the curvature of a univariate spline.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).

        Returns
        -------
        value : scalar
            The value of the curvature at the given point on the curve.
        
        Notes
        -----
        Computes the curvature of the graph of the function if nDep == 1.  If nDep == 1 or 2,
        then the curvature is computed as a signed quantity.
        """
        return bspy._spline_evaluation.curvature(self, uvw)

    @staticmethod
    def cylinder(radius, height, tolerance = None):
        """
        Construct a cylinder of the given radius and height.

        Parameters
        ----------
        radius : scalar
            The desired radius of the cylinder
        
        height : scalar
            The desired height of the cylinder

        tolerance : scalar, optional
            The desired absolute tolerance to which the cylinder should be constructed.  Defaults
            to 1.0e-12 if tolerance == None.

        Returns
        -------
        spline : `Spline`
            A bi-quartic spline approximation to a cylinder of the specified radius and height,
            accurate to the given tolerance.
        
        Notes
        -----
        The resulting mapping is defined over the unit square.  The first independent variable
        corresponds to the radial position on the cylinder, while the second independent variable
        is the location along the height.
    
        See Also
        --------
        `circular_arc` : Create a 2D circular arc for a given radius and angle accurate to within
                         a given tolerance.
        `ruled_surface` : Construct a ruled surface between two curves.
        """
        return bspy._spline_fitting.cylinder(radius, height, tolerance)

    def derivative(self, with_respect_to, *uvw, **kwargs):
        """
        Compute the derivative of the spline at given parameter values.

        Parameters
        ----------
        with_respect_to : `iterable`
            An iterable of length `nInd` that specifies the integer order of derivative for each independent variable.
            A zero-order derivative just evaluates the spline normally.
        
        *uvw : `iterable` or iterable of iterables
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values), or 
            an iterable of iterables of length `nInd` that specifies values for each independent variable (numpy ufunc style).

        **kwargs:
            For other keyword-only arguments, see the `ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

        Returns
        -------
        value : `numpy.array`
            The value of the derivative of the spline at the given parameter values (array of size nDep). If multiple points are 
            passed (numpy ufunc style), then multiple arrays are returned, one for each dependent variable.

        See Also
        --------
        `evaluate` : Compute the value of the spline at a given parameter value.
        `differentiate` : Differentiate a spline with respect to one of its independent variables, returning the resulting spline.
        `integral` : Compute the integral of the spline at a given parameter value.
        `integrate` : Integrate a spline with respect to one of its independent variables, returning the resulting spline.

        Notes
        -----
        The derivative method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline.  The non-zero B-splines are
        evaluated, then the dot product of those B-splines with the vector of
        B-spline coefficients is computed.
        """
        if len(uvw) == 0 and self.nInd == 0:
            return np.zeros(self.nDep, self.coefs.dtype)
        elif np.isscalar(uvw[0]):
            return bspy._spline_evaluation.derivative(self, with_respect_to, uvw)
        elif len(uvw) > 1 or len(uvw[0]) > self.nInd:
            def vectorized(*uvwInstance):
                return tuple(bspy._spline_evaluation.derivative(self, with_respect_to, uvwInstance))
            uFunc = np.frompyfunc(vectorized, self.nInd, self.nDep)
            if self.nDep > 1:
                return tuple([a.astype(self.coefs.dtype, copy=False) for a in uFunc(*uvw, **kwargs)])
            else:
                return np.array([x[0] for x in uFunc(*uvw, **kwargs)], self.coefs.dtype)
        else:
            return bspy._spline_evaluation.derivative(self, with_respect_to, *uvw)

    def differentiate(self, with_respect_to = 0):
        """
        Differentiate a spline with respect to one of its independent variables, returning the resulting spline.

        Parameters
        ----------
        with_respect_to : integer, optional
            The number of the independent variable to differentiate. Default is zero.

        Returns
        -------
        spline : `Spline`
            The spline that results from differentiating the original spline with respect to the given independent variable.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at a given parameter value.
        `integral` : Compute the integral of the spline at a given parameter value.
        `integrate` : Integrate a spline with respect to one of its independent variables, returning the resulting spline.
        """
        return bspy._spline_operations.differentiate(self, with_respect_to)

    def domain(self):
        """
        Return the domain of a spline.

        Returns
        -------
        bounds : `numpy.array`
            nInd x 2 array of the lower and upper bounds on each of the independent variables.

        See Also
        --------
        `reparametrize` : Reparametrize a spline to match new domain bounds
        `trim` : Trim the domain of a spline.
        """
        return bspy._spline_evaluation.domain(self)

    def domain_dimension(self):
        """
        Return the domain dimension of a spline (nInd).

        Returns
        -------
        dimension : `int`
            The dimension of the spline's domain (nInd)
        """
        return self.nInd

    def dot(self, vector):
        """
        Dot product a spline by the given vector.

        Parameters
        ----------
        vector : array-like or `Spline`
            An array of length `nDep` or spline with matching `nDep` that specifies the vector.

        Returns
        -------
        spline : `Spline`
            The dotted spline.

        See Also
        --------
        `multiply` : Multiply two splines (cross, dot, or scalar product).

        Notes
        -----
        Equivalent to spline @ vector.
        """
        return bspy._spline_operations.dot(self, vector)

    def elevate(self, m):
        """
        Elevate a spline, increasing its order by `m`.

        Parameters
        ----------
        m : `iterable` of length `nInd`
            An iterable that specifies the non-negative integer amount to increase the order 
            for each independent variable of the spline.

        Returns
        -------
        spline : `Spline`
            A spline with the order of the current spline plus `m`.

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `elevate_and_insert_knots` : Elevate a spline and insert new knots.

        Notes
        -----
        Implements the algorithm from Huang, Qi-Xing, Shi-Min Hu, and Ralph R. Martin. 
        "Fast degree elevation and knot insertion for B-spline curves." Computer Aided Geometric Design 22, no. 2 (2005): 183-197.
        """
        return self.elevate_and_insert_knots(m, self.nInd * [[]])

    def elevate_and_insert_knots(self, m, newKnots):
        """
        Elevate a spline and insert new knots.

        Parameters
        ----------
        m : `iterable` of length `nInd`
            An iterable that specifies the non-negative integer amount to increase the order 
            for each independent variable of the spline.

        newKnots : `iterable` of length `nInd`
            An iterable that specifies the knots to be added to each independent variable's knots. 
            len(newKnots[ind]) == 0 if no knots are to be added for the `ind` independent variable.

        Returns
        -------
        spline : `Spline`
            A spline with the order of the current spline plus `m` that includes the new knots.

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `clamp` : Clamp the left and/or right side of a spline.
        `elevate` : Elevate a spline, increasing its order by `m`.

        Notes
        -----
        Implements the algorithm from Huang, Qi-Xing, Shi-Min Hu, and Ralph R. Martin. 
        "Fast degree elevation and knot insertion for B-spline curves." Computer Aided Geometric Design 22, no. 2 (2005): 183-197.
        """
        return bspy._spline_domain.elevate_and_insert_knots(self, m, newKnots)

    def evaluate(self, *uvw, **kwargs):
        """
        Compute the value of the spline at given parameter values.

        Parameters
        ----------
        *uvw : `iterable` or iterable of iterables
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values), or 
            an iterable of iterables of length `nInd` that specifies values for each independent variable (numpy ufunc style).

        **kwargs:
            For other keyword-only arguments, see the `ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

        Returns
        -------
        value : `numpy.array`
            The value of the spline at the given parameter values (array of size nDep). If multiple points are 
            passed (numpy ufunc style), then multiple arrays are returned, one for each dependent variable.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at given parameter values.

        Notes
        -----
        Equivalent to spline(*uvw, **kwargs).

        The evaluate method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline.  The non-zero B-splines are
        evaluated, then the dot product of those B-splines with the vector of
        B-spline coefficients is computed.
        """
        if len(uvw) == 0 and self.nInd == 0:
            return self.coefs
        elif np.isscalar(uvw[0]):
            return bspy._spline_evaluation.evaluate(self, uvw)
        elif len(uvw) > 1 or len(uvw[0]) > self.nInd:
            def vectorized(*uvwInstance):
                return tuple(bspy._spline_evaluation.evaluate(self, uvwInstance))
            uFunc = np.frompyfunc(vectorized, self.nInd, self.nDep)
            if self.nDep > 1:
                return tuple([a.astype(self.coefs.dtype, copy=False) for a in uFunc(*uvw, **kwargs)])
            else:
                return np.array([x[0] for x in uFunc(*uvw, **kwargs)], self.coefs.dtype)
        else:
            return bspy._spline_evaluation.evaluate(self, *uvw)

    def extrapolate(self, newDomain, continuityOrder):
        """
        Extrapolate a spline out to an extended domain maintaining a given order of continuity.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new lower and upper bounds on each of the independent variables (same form as 
            returned from `domain`). If a bound is None or nan then the original bound (and knots) are left unchanged.

        continuityOrder : `int`
            The order of continuity of the extrapolation (the number of derivatives that match at the endpoints). 
            A continuity order of zero means the extrapolation just matches the spline value at the endpoints. 
            The continuity order is automatically limited to one less than the degree of the spline.

        Returns
        -------
        spline : `Spline`
            Extrapolated spline. If all the knots are unchanged, the original spline is returned.

        See Also
        --------
        `domain` : Return the domain of a spline.
        `trim` : Trim the domain of a spline.
        """
        return bspy._spline_domain.extrapolate(self, newDomain, continuityOrder)

    def flip_normal(self):
        """
        Flip the direction of the normal.

        Returns
        -------
        spline : `Spline`
            The spline with flipped normal. The spline retains the same tangent space.

        See Also
        --------
        `solid.Solid.complement` : Return the complement of the solid: whatever was inside is outside and vice-versa.
        """
        spline = self.copy()
        spline.metadata["flipNormal"] = not self.metadata.get("flipNormal", False)
        return spline

    def fold(self, foldedInd):
        """
        Fold the coefficients of a spline's indicated independent variables into the coefficients of the remaining independent variables, retaining the 
        indicated independent variables' knots and orders in a second spline with no coefficients.

        Parameters
        ----------
        foldedInd : `iterable`
            An iterable that specifies the independent variables whose coefficients should be folded.

        Returns
        -------
        foldedSpline, coefficientlessSpline : `Spline`, `Spline`
            The folded spline and the coefficientless spline that retains the indicated independent variables' knots and orders.

        See Also
        --------
        `unfold` : Unfold the coefficients of an original spline's indicated independent variables back into the spline.

        Notes
        -----
        Given a spline whose coefficients are an nDep x n0 x ... x nk array and a list of (ordered) indices which form a proper subset of {n0, ... , nk}, it should return 2 splines.
        The first one is a spline with k + 1 - length(index subset) independent variables where the knot sets of all the dimensions which have been removed have been removed.  However, 
        all of the coefficient data is still intact, so that the resulting coefficient array has shape (nDep nj0 nj1 ... njj) x nk0 x ... x nkk.  The second spline should be a spline 
        with 0 dependent variables which contains all the knot sequences that were removed from the spline.  The unfold method takes the two splines as input and reverses the process, 
        returning the original spline.

        Here's an example. Suppose spl is a spline with 3 independent variables and 3 dependent variables which has nCoef = [4, 5, 6] and knots = [knots0, knots1, knots2]. 
        Then spl.fold([0, 2]) should return a spline with 1 independent variable and 72 dependent variables.  It should have nCoef = [5], knots = [knots1], and its coefs array should have 
        shape (72, 5).  The other spline should have 0 dependent variables, 2 independent variables, and knots = [knots0, knots2].  How things get ordered in coefs probably doesn't matter 
        so long as unfold unscrambles things in the corresponding way.  The second spline is needed to hold the basis information that was dropped so that it can be undone.
        """
        return bspy._spline_domain.fold(self, foldedInd)
    
    def four_sided_patch(bottom, right, top, left, surfParam = 0.5):
        """
        Construct a surface from its four boundary curves.
    
        Parameters
        ----------
        bottom : `Spline`
            A curve which represents the bottom edge of the resulting surface.
        
        right : `Spline`
            A curve which represents the right edge of the resulting surface.

        top : `Spline`
            A curve which represents the top edge of the resulting surface.

        left : `Spline`
            A curve which represents the left edge of the resulting surface.
    
        surfParam : scalar, optional
            A scalar which selects which member of a one parameter family of interpolating
            surfaces to return.  surfParam = 0.0 will return the Coons patch.  surfParam = 1.0
            will return the surface for which each dependent variable approximately solves the
            Laplace equation for which the four boundary curves are the Dirichlet boundary
            conditions.  Other values of surfParam will return the appropriate affine combination
            of those two surfaces.
    
        Returns
        -------
        surface : `Spline`
            The spline surface requested by the input curves and surfParam.

        See Also
        --------
        `common_basis` : Determine a common B-spline basis for two or more splines.
    
        Notes
        -----
        The variables bottom, right, top, and left are notional in the sense that the curves can
        be given in any order.  This method does the best possible job of matching endpoints, so
        the roles of "right," "top," and "left" may very well get reassigned.  Moreover, the
        parametrization of the curves may also get reversed in order to best match endpoints.
        """
        return bspy._spline_fitting.four_sided_patch(bottom, right, top, left, surfParam)

    @staticmethod
    def from_dict(dictionary):
        """
        Create a `Spline` from a data in a `dict`.

        Parameters
        ----------
        dictionary : `dict`
            The `dict` containing `Spline` data.

        Returns
        -------
        spline : `Spline`

        See Also
        --------
        `to_dict` : Return a `dict` with `Spline` data.
        """
        return Spline(dictionary["nInd"], dictionary["nDep"], dictionary["order"], dictionary["nCoef"],
                [np.array(knots) for knots in dictionary["knots"]], np.array(dictionary["coefs"]), dictionary["metadata"])

    def full_domain(self):
        """
        Return a solid that represents the full domain of the spline.

        Returns
        -------
        domain : `Solid`
            The full (untrimmed) domain of the spline.

        See Also
        --------
        `Boundary` : A portion of the boundary of a solid.
        """
        return bspy._spline_intersection.full_domain(self)
    
    def geodesic(self, uvStart, uvEnd, tolerance = 1.0e-6):
        """
        Determine a geodesic between two points on a surface

        Parameters
        ----------
        uvStart : `array-like`
            The parameter values for the surface at one end of the desired geodesic.
        
        uvEnd : `array-like`
            The parameter values for the surface at the other end of the desired geodesic.
        
        tolerance : scalar
            The maximum error in parameter space to which the geodesic should get computed.
            Defaults to 1.0e-6.

        Returns
        -------
        geodesic : `Spline`
            A spline curve whose range is in the domain of the given surface.  The range of the
            curve is the locus of points whose image under the surface map form the curve of
            minimum distance between the two given points.
        
        See Also
        --------
        `solve_ode` : Solve an ordinary differential equation using spline collocation.

        Notes
        -----
        Solves the second order ODE which are the geodesic equations over the surface.
        """
        return bspy._spline_fitting.geodesic(self, uvStart, uvEnd, tolerance)
    
    def graph(self):
        """
        Generate the spline which is the graph of the given spline.
        
        Returns
        -------
        spline : `Spline`
            A spline with nInd independent variables and nInd + nDep dependent variables, the first nInd of
            which are just the independent variables themselves. For example, given a scalar
            valued function f of two variables u and v, returns the spline of two independent variables whose
            three dependent variables are (u, v, f(u,v)).
        
        See Also
        --------
        `add` : Add two splines together

        Notes
        -----
        Makes use of matrix spline multiply and tensor product addition for splines.
        """
        return bspy._spline_operations.graph(self)
    
    def greville(self, ind = 0):
        """
        Compute and return the Greville abscissae (knot averages) for the given independent
        variable.

        Parameters
        ----------
        ind : integer, optional
            The index of the independent variable (default is 0).
        
        Returns
        -------
        knotAverages: `numpy.ndarray`
            An array with the knot averages for the specified independent variable.

        Notes
        -----
        The Greville abscissae always satisfy the interlacing conditions, so can be used as
        valid collocation points, interpolation points, or quadrature points.
        """
        return bspy._spline_evaluation.greville(self, ind)
        
    def insert_knots(self, newKnots):
        """
        Insert new knots into a spline.

        Parameters
        ----------
        newKnots : `iterable` of length `nInd`
            An iterable that specifies the knots to be added to each independent variable's knots. 
            len(newKnots[ind]) == 0 if no knots are to be added for the `ind` independent variable.

        Returns
        -------
        spline : `Spline`
            A spline with the new knots inserted. If no knots were inserted, the original spline is returned.

        See Also
        --------
        `clamp` : Clamp the left and/or right side of a spline.

        Notes
        -----
        Implements Boehm's standard knot insertion algorithm.
        """
        return bspy._spline_domain.insert_knots(self, newKnots)

    def integral(self, with_respect_to, uvw1, uvw2, returnSpline = False):
        """
        Compute the derivative of the spline at given parameter values.

        Parameters
        ----------
        with_respect_to : `iterable`
            An iterable of length `nInd` that specifies the integer order of integral for each independent variable.
            A zero-order integral just evaluates the spline normally.
        
        uvw1 : `iterable`
            An iterable of length `nInd` that specifies the lower limit of each independent variable (the parameter values).
        
        uvw2 : `iterable`
            An iterable of length `nInd` that specifies the upper limit of each independent variable (the parameter values).

        returnSpline : `boolean`, optional
            A boolean flag that if true returns the integrated spline along with the value of its integral. Default is false.
 
        Returns
        -------
        value : `numpy.ndarray`
            The value of the integral of the spline at the given parameter limits.

        spline : `Spline`
            The integrated spline, which is only returned if `returnSpline` is `True`.

        See Also
        --------
        `integrate` : Integrate a spline with respect to one of its independent variables, returning the resulting spline.
        `evaluate` : Compute the value of the spline at a given parameter value.
        `differentiate` : Differentiate a spline with respect to one of its independent variables, returning the resulting spline.
        `derivative` : Compute the derivative of the spline at a given parameter value.

        Notes
        -----
        The integral method uses the integrate method to integrate the spline `with_respect_to` times for each independent variable.
        Then the method returns that integrated spline's value at `uw2` minus its value at `uw1` (optionally along with the spline).
        The method doesn't calculate the integral directly because the number of operations required is nearly the same as constructing
        the integrated spline.
        """
        return bspy._spline_evaluation.integral(self, with_respect_to, uvw1, uvw2, returnSpline)

    def integrate(self, with_respect_to = 0):
        """
        Integrate a spline with respect to one of its independent variables, returning the resulting spline.

        Parameters
        ----------
        with_respect_to : integer, optional
            The number of the independent variable to integrate. Default is zero.

        Returns
        -------
        spline : `Spline`
            The spline that results from integrating the original spline with respect to the given independent variable.

        See Also
        --------
        `differentiate` : Differentiate a spline with respect to one of its independent variables, returning the resulting spline.
        `derivative` : Compute the derivative of the spline at a given parameter value.
        `integral` : Compute the integral of the spline at a given parameter value.
        """
        return bspy._spline_operations.integrate(self, with_respect_to)

    def intersect(self, other):
        """
        Intersect a spline or hyperplane.

        Parameters
        ----------
        other : `Spline` or `Hyperplane`
            The `Manifold` to intersect with self (must have same range dimension as self).

        Returns
        -------
        intersections : `iterable` (or `NotImplemented` if other is an unknown type of Manifold)
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
        `zeros` : Find the roots of a spline (nInd must match nDep).
        `contours` : Find all the contour curves of a spline.
        `solid.Solid.slice` : slice the solid by a manifold.

        Notes
        -----
        Uses `zeros` to find all intersection points and `contours` to find all the intersection curves.
        """
        if not(self.range_dimension() == other.range_dimension()): raise ValueError("The number of dependent variables for both splines much match.")
        return bspy._spline_intersection.intersect(self, other)

    def jacobian(self, uvw):
        """
        Compute the value of the spline's Jacobian at given parameter values.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).

        Returns
        -------
        value : `numpy.array`
            The value of the spline's Jacobian at the given parameter values. The shape of the return value is (nDep, nInd).

        See Also
        --------
        `evaluate` : Compute the value of the spline at given parameter values.
        `derivative` : Compute the derivative of the spline at given parameter values.

        Notes
        -----
        Calls `derivative` nInd times.
        """
        return bspy._spline_evaluation.jacobian(self, uvw)

    @staticmethod
    def join(splineList):
        """
        Join a list of splines together into a single spline.

        Parameters
        ----------
        splineList : `iterable`
            The list of splines to join together.  All must have the same number of dependent variables.

        Returns
        -------
        joinedSpline : `Spline`
            A single spline whose image is the union of all the images of the input splines.  The resulting spline
            is parametrized over the unit cube.
                
        Notes
        -----
        Currently only works for univariate splines.
        """
        return bspy._spline_domain.join(splineList)
       
    @staticmethod
    def least_squares(uValues, dataPoints, order = None, knots = None, compression = 0.0,
                      tolerance = None, fixEnds = False, metadata = {}):
        """
        Fit a spline to an array of data points using the method of least squares.

        Parameters
        ----------
        uValues : `iterable` of array-like values
            The values of the independent variables in each of the coordinate directions for
            each of the data points.  It's either a simple array of length nU or it is an array
            of length nInd of arrays with corresponding lengths nU, nV, . . ., nW, where each
            length indicates the number of data points in that independent variable.  For each
            independent variable, the values must be ordered, i.e. uValue[i] <= uValue[i+1].  The
            corresponding point in dataPoints will be an array of length nDep which represents the
            desired function values at that point.  If the values in uValues are repeated, then
            the corresponding data in dataPoints represents a derivative value.  Examples:
            
            If uValues == [0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0], then nInd == 1 and dataPoints
            contains
            [f(0.0), f(1.0), f'(1.0), f(2.0), f(3.0), f'(3.0), f''(3.0)]

            if uValues == [[0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 2.0, 3.0]], then nInd == 2 and
            dataPoints contains
            [[f(0,0), f(0,1), f(0,2), f_v(0,2), f(0,3)],
             [f(1,0), f(1,1), f(1,2), f_v(1,2), f(1,3)],
             [f_u(1,0), f_u(1,1), f_u(1,2), f_uv(1,2), f_u(1,3)],
             [f(2,0), f(2,1), f(2,2), f_v(2,2), f(2,3)]]

        dataPoints : `iterable` of array-like values
            A collection of data points.  The shape of this array mirrors the shape of
            Spline.coefs, i.e. it has shape (nDep, nU, nV, . . . , nW).  The indices into this
            array also index into uValues to retrieve the independent variable values.
            For example, if nInd == 2 and nDep == 3, then the shape of dataPoints is (3, nU, nV)
            and dataPoint[:,i,j] has independent variable values given by uValues[0,i] and
            uValues[1,j].

        order : `tuple`
            A tuple of length nInd where each integer entry represents the
            polynomial order of the spline in that variable.  If order == None, then
            order == [4] * nInd is assumed.

        knots : `list`, optional
            A list of the lists of the knots of the spline in each independent variable.  The
            length of knots should be nInd if it is specified, where each of the arrays specify
            the knots of the spline to use in that independent variable.  All of the dataPoints
            must lie within the domain specified by the knots.  If knots == None (the default),
            then the knots are automatically determined.
        
        compression : scalar, optional
            The desired compression of data used as a fraction of the number of data points, i.e
            compression is a number between 0 and 1.  This fraction is used to determine the
            total number of spline coefficients when knots are chosen automatically (it's if the
            knots are specified). The actual compression will be slightly less because the number
            of coefficients is rounded up. The default value is zero (interpolation, no compression).

        tolerance : scalar, optional
            If tolerance is specified, then a fit to tolerance is performed.  Using the given knots
            as the initial set of knots, additional knots are inserted until the 2-norm of the
            difference between each of the each given dataPoints and the vector given by the
            spline is no larger than the specified tolerance.
        
        fixEnds : Boolean, optional
            if fixEnds is True, then all of the dataPoints which lie on any of the domain
            boundaries of the spline become interpolation conditions, i.e. the spline will be
            fit in such a way that those boundary points are exactly satisfied by the computed
            spline.  Otherwise, least-squares fitting will be performed on all of the dataPoints.

        metadata : `dict`, optional
            A dictionary of ancillary data to store with the spline. Default is {}.

        Returns
        -------
        spline : `Spline`
            A spline curve which approximates the data points.

        Notes
        -----
        Uses `numpy.linalg.lstsq` to compute the least squares solution.
        """
        return bspy._spline_fitting.least_squares(uValues, dataPoints, order, knots, compression,
                                                  tolerance, fixEnds, metadata)

    @staticmethod
    def line(startPoint, endPoint):
        """
        Construct a line between two points or between two lines.

        Parameters
        ----------
        startPoint : any
            A scalar or a list or an array or an instance of a Spline.
        
        endPoint : any
            A scalar or a list or an array or an instance of a Spline.
        
        Returns
        -------
        spline : `Spline`
            The spline that interpolates the two inputs.
        
        See Also
        --------
        `ruled_surface` : Construct a ruled surface between two splines.

        Notes
        -----
        Construct splines out of the two points and then call ruled_surface.
        """
        return bspy._spline_fitting.line(startPoint, endPoint)
    
    @staticmethod
    def load(fileName):
        """
        Load splines in json format from the specified filename (full path).

        Parameters
        ----------
        fileName : `string`
            The full path to the file containing the spline. Can be a relative path.
        
        Returns
        -------
        splines : list of `Spline`
            The loaded splines.

        See Also
        --------
        `save` : Save a spline in json format to the specified filename (full path).
        """
        # First try numpy's load function for backward compatibility with original file format.
        try:
            kw = np.load(fileName)
            order = kw["order"]
            nInd = len(order)
            knots = []
            for i in range(nInd):
                knots.append(kw[f"knots{i}"])
            coefficients = kw["coefficients"]
            return [Spline(nInd, coefficients.shape[0], order, coefficients.shape[1:], knots, coefficients, metadata=dict(Path=path, Name=path.splitext(path.split(fileName)[1])[0]))]
        except:
            pass # Do nothing, an error is expected for json files
        
        # Load json file.
        with open(fileName, 'r', encoding='utf-8') as file:
            splineData = json.load(file)

        splines = []
        if isinstance(splineData, dict):
            splineData = [splineData]
        for splineDict in splineData:
            splines.append(Spline.from_dict(splineDict))
        return splines

    def multiply(self, other, indMap = None, productType = 'S'):
        """
        Multiply two splines (cross, dot, or scalar product).

        Parameters
        ----------
        other : `Spline`
            The spline to multiply by self.

        indMap : `iterable` or `None`, optional
            An iterable of indices or pairs of indices. Each index refers to an independent variable.
            Within the iterable, a single index, `n`, maps the nth independent variable of self to the same independent variable of other.
            A pair `(n, m)` maps the nth independent variable of self to the mth independent variable of other. 
            For example, if you wanted to compute `self(u, v, w) * other(u, w)`, you'd pass `[0, (2, 1)]` for `indMap`. 
            Unmapped independent variables remain independent (the default).
            The domains of mapped independent variables must match. 
            An independent variable can map to no more than one other independent variable.

        productType : {'C', 'D', 'S'}, optional
            The type of product to perform on the dependent variables (default is 'S').
                'C' is for a cross product, self x other (nDep must be 2 or 3).
                'D' is for a dot product (nDep must match).
                'S' is for a scalar product (nDep must match or be 1 for one of the splines).
        
        Returns
        -------
        spline : `Spline`
            The result of multiplying self and other.

        See Also
        --------
        `add` : Add two splines.
        `subtract` : Subtract two splines.
        `convolve` : Convolve two splines (cross, dot, or scalar product).

        Notes
        -----
        Equivalent to spline1 * spline2 with indMap = range(min(spline1.nInd, spline2.nInd)) and productType = 'S'.
        
        Taken in part from Lee, E. T. Y. "Computing a chain of blossoms, with application to products of splines." 
        Computer Aided Geometric Design 11, no. 6 (1994): 597-620.
        """
        if indMap is not None:
            indMap = [(mapping, mapping, False) if np.isscalar(mapping) else (*mapping, False) for mapping in indMap]
        return bspy._spline_operations.multiplyAndConvolve(self, other, indMap, productType)

    def normal(self, uvw, normalize=True, indices=None):
        """
        Compute the normal of the spline at given parameter values. The number of independent variables must be
        one different than the number of dependent variables.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).
        
        normalize : `boolean`, optional
            If True the returned normal will have unit length (the default). Otherwise, the normal's length will
            be the area of the tangent space (for two independent variables, its the length of the cross product of tangent vectors).
        
        indices : `iterable`, optional
            An iterable of normal indices to calculate. For example, `indices=(0, 3)` will return a vector of length 2
            with the first and fourth values of the normal. If `None`, all normal values are returned (the default).

        Returns
        -------
        normal : `numpy.array`
            The normal vector of the spline at the given parameter values.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at a given parameter value.
        `normal_spline` : Compute a spline that evaluates to the normal of the given spline (not normalized).

        Notes
        -----
        Attentive readers will notice that the number of independent variables could be one more than the number of 
        dependent variables (instead of one less, as is typical). In that case, the normal represents the null space of 
        the matrix formed by the tangents of the spline. If the null space is greater than one dimension, the normal will be zero.
        """
        return bspy._spline_evaluation.normal(self, uvw, normalize, indices)

    def normal_spline(self, indices=None):
        """
        Compute a spline that evaluates to the normal of the given spline. The length of the normal
        is the area of the tangent space (for two independent variables, its the length of the cross product of tangent vectors).
        The number of independent variables must be one different than the number of dependent variables.

        Parameters
        ----------
        indices : `iterable`, optional
            An iterable of normal indices to calculate. For example, `indices=(0, 3)` will make the returned spline compute a vector of length 2
            with the first and fourth values of the normal. If `None`, all normal values are returned (the default).

        Returns
        -------
        spline : `Spline`
            The spline that evaluates to the normal of the given spline.

        See Also
        --------
        `normal` : Compute the normal of the spline at given parameter values.
        `differentiate` : Differentiate a spline with respect to one of its independent variables, returning the resulting spline.

        Notes
        -----
        Attentive readers will notice that the number of independent variables could be one more than the number of 
        dependent variables (instead of one less, as is typical). In that case, the normal represents the null space of 
        the matrix formed by the tangents of the spline. If the null space is greater than one dimension, the normal will be zero.
        """
        return bspy._spline_operations.normal_spline(bspy.spline_block.SplineBlock(self), indices)

    @staticmethod
    def point(point):
        """
        Construct a spline which represents a point.

        Parameters
        ----------
        point : scalar, list, or array
            A scalar or a list or an array or an instance of a Spline.
        
        Returns
        -------
        spline : `Spline`
            The spline that represents the input point.
        
        See Also
        --------
        `line` : Construct a line between two points.

        Notes
        -----
        Construct a spline out of an input point.
        """
        return bspy._spline_fitting.point(point)
    
    def range_bounds(self):
        """
        Return the range of a spline as lower and upper bounds on each of the
        dependent variables.
        """
        return bspy._spline_evaluation.range_bounds(self)

    def range_dimension(self):
        """
        Return the range dimension of a spline (nDep).

        Returns
        -------
        dimension : `int`
            The dimension of the spline's range (nDep)
        """
        return self.nDep

    def remove_knot(self, iKnot, nLeft = 0, nRight = 0):
        """
        Remove a single knot from a univariate spline.
        
        Parameters
        ----------
        iKnot : integer
            Index of the knot to remove from the spline
        
        nLeft : integer
            Number of continuity conditions to enforce on the left end.  For example, if nLeft = 2,
            then remove_knot will leave the function value and first derivative unchanged on the
            left end of the domain.
        
        nRight : integer
            Number of continuity conditions to enforce on the right end.  For example, if
            nRight = 3, then remove_knot will leave the function value and first and second
            derivative values unchanged on the right end of the domain.
        
        Returns
        -------
        spline : `Spline`
            New spline with the specified knot removed
        residual : array-like
            A vector containing the least squares residuals for each dependent variable
        
        See Also
        --------
        `remove_knots`
        
        Notes
        -----
        Solves a simple least squares problem
        """
        return bspy._spline_domain.remove_knot(self, iKnot, nLeft, nRight)
    
    def remove_knots(self, tolerance = 1.0e-14, nLeft = 0, nRight = 0):
        """
        Remove interior knots from a spline.

        Parameters
        ----------
        tolerance : `float`, optional
            The maximum pointwise relative error permitted after removing all the knots. Knots will be
            removed until removal of the next knot would put the pointwise error above the threshold.
            The default is 1.0e-14 which is relative to the size of each of the dependent variables.

        nLeft : integer
            Number of continuity conditions to enforce on the left end.  For example, if nLeft = 2,
            then remove_knot will leave the function value and first derivative unchanged on the
            left end of the domain.
        
        nRight : integer
            Number of continuity conditions to enforce on the right end.  For example, if
            nRight = 3, then remove_knot will leave the function value and first and second
            derivative values unchanged on the right end of the domain.
        

        Returns
        -------
        spline : `Spline`
            A spline with the knots removed.

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `remove_knot` : Remove an indicated knot from a spline.
        `trim` : Trim the domain of a spline.

        Notes
        -----
        Calls `remove_knot` repeatedly until no longer possible.
        """
        return bspy._spline_domain.remove_knots(self, tolerance, nLeft, nRight)
    
    def reparametrize(self, newDomain):
        """
        Reparametrize a spline to match new domain bounds. The spline's number of knots and its coefficients remain unchanged.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new lower and upper bounds on each of the independent variables (same form as 
            returned from `domain`). If a bound pair is `None` then the original bound (and knots) are left unchanged. 
            For example, `[[0.0, 1.0], None]` will reparametrize the first independent variable and leave the second unchanged)

        Returns
        -------
        spline : `Spline`
            Reparametrized spline.

        See Also
        --------
        `domain` : Return the domain of a spline.
        """
        return bspy._spline_domain.reparametrize(self, newDomain)

    def reverse(self, variable = 0):
        """
        Reverse the direction of a spline along one of the independent variables

        Parameters
        ----------
        variable : integer
            index of the independent variable to reverse the direction of.
        
        Returns
        -------
        spline : `Spline`
            Reparametrized (i.e. reversed) spline.
        
        See Also
        --------
        `reparametrize` : Reparametrize a spline
        """
        return bspy._spline_domain.reverse(self, variable)

    def revolve(self, angle):
        """
        Revolve the spline to create a surface of revolution (nDep must equal 2, 
        first dimension provides the radius for x and y, second dimension provides the z).

        Parameters
        ----------
        angle : `float`
            The angle of the circular arc measured in degrees starting at the x-axis rotating counterclockwise.

        Returns
        -------
        spline : `Spline`
            Surface of revolution (first independent variable sweeps the arc from 0 to 1).

        See Also
        --------
        `circular_arc` : Create a 2D circular arc for a given radius and angle accurate to within a given tolerance.
        """
        return bspy._spline_fitting.revolve(self, angle)

    def rotate(self, vector, angle):
        """
        Rotate a spline around a vector by the specified angle.

        Parameters
        ----------
        vector : array-like
            A vector of length three about which to rotate the spline
        
        angle : scalar
            The number of degrees to rotate the spline around vector by
        
        Returns
        -------
        spline : `Spline`
            The rotated spline
        
        See Also
        --------
        `transform` : Multiply a spline by a given matrix.

        Notes
        -----
        Uses the well-known Rodrigues' rotation formula and mathematical operations
        on splines.
        """
        return bspy._spline_operations.rotate(self, vector, angle)
    
    @staticmethod
    def ruled_surface(spline1, spline2):
        """
        Create a ruled surface between two splines.

        Parameters
        ----------
        spline1 : `Spline`
            The first spline (must have the same nInd and nDep as the second spline).

        spline2 : `Spline`
            The second spline (must have the same nInd and nDep as the first spline).
        
        Returns
        -------
        spline : `Spline`
            Ruled surface between spline1 and spline2.
        """
        return bspy._spline_fitting.ruled_surface(spline1, spline2)
    
    def save(self, fileName, *additional_splines):
        """
        Save a spline in json format to the specified filename (full path).

        Parameters
        ----------
        fileName : `string`
            The full path to the file containing the spline. Can be a relative path.
        
        *additional_splines : `Spline`
            More splines to save in the same file.
        
        See Also
        --------
        `load` : Load splines in json format from the specified filename (full path).
        """
        class SplineEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, Spline):
                    return obj.to_dict()
                return super().default(obj)

        with open(fileName, 'w', encoding='utf-8') as file:
            if additional_splines:
                json.dump((self, *additional_splines), file, indent=4, cls=SplineEncoder)
            else:
                json.dump(self, file, indent=4, cls=SplineEncoder)

    def scale(self, multiplier):
        """
        Scale a spline by the given scalar or scale vector.

        Parameters
        ----------
        multiplier : scalar, array-like, or `Spline`
            A scalar, an array of length `nDep`, or spline that specifies the multiplier.

        Returns
        -------
        spline : `Spline`
            The scaled spline.

        See Also
        --------
        `multiply` : Multiply two splines (cross, dot, or scalar product).
        `transform` : Transform a spline by the given matrix.
        `translate` : Translate a spline by the given translation vector.

        Notes
        -----
        Equivalent to spline * multiplier.
        """
        return bspy._spline_operations.scale(self, multiplier)
    
    @staticmethod
    def section(xytk):
        """
        Fit a planar section to the list of 4-tuples of data.

        Parameters
        ----------
        xytk : array-like
            A list of points to fit an interpolating spline to.  Each point in the list contains
            four values.  The first two are x and y coordinates of the point.  The third value is the
            angle that the tangent makes as the desired section passes through that point (in degrees).
            The fourth value is the desired curvature at that point

        Returns
        -------
        spline : `Spline`
            A quartic spline which interpolates the given values.

        Notes
        -----
        The spline is shape-preserving.  Each consecutive pair of data points must describe a convex or
        concave curve.  In particular, if it is impossible for a differentiable curve to interpolate two
        consecutive data points without passing through an intermediate inflection point (i.e. a point which
        has zero curvature and at which the sign of the curvature changes), then this method will fail
        with an error.

        See Also
        --------
        `circular_arc` : Create a 2D circular arc for a given radius and angle accurate to within a given tolerance.
        """
        return bspy._spline_fitting.section(xytk)
    
    def solve_ode(self, nLeft, nRight, FAndF_u, tolerance = 1.0e-6, args = ()):
        """
        Numerically solve an ordinary differential equation with boundary conditions.

        Parameters
        ==========
        self : `Spline`
            The initial guess for the solution to the spline.  All of the spline parameters, e.g.
            order, boundary conditions, domain, etc. are all established by the initial guess.  The
            ODE itself must be of the form u^(nOrder)(t) = F(t, u, u', ... , u^(nOrder-1)).  self.nDep should
            be the same as the number of state variables in the ODE.
        
        nLeft : integer
            The number of boundary conditions to be imposed on the left side of the domain.  The
            order of the differential equation is assumed to be the sum of nLeft and nRight.
        
        nRight : integer
            The number of boundary conditions to be imposed on the right sie of the domain.  The
            order of the differential equation is assumed to be the sum of nLeft and nRight.
        
        FAndF_u : Python function
            FAndF_u must have exactly this calling sequence:  FAndF_u(t, uData, *args).  t is a scalar set
            to the desired value of the independent variable of the ODE.  uData will be a numpy matrix of shape
            (self.nDep, nOrder) whose columns are (u, ... , u^(nOrder - 1).  It must return a numpy
            vector of length self.nDep and a numpy array whose shape is (self.nDep, self.nDep, nOrder).
            The first output vector is the value of the forcing function F at (t, uData).  The numpy
            array is the array of partial derivatives with respect to all the numbers in uData.  Thus, if
            this array is called jacobian, then jacobian[:, i, j] is the gradient of the forcing function with
            respect to uData[i, j].
        
        tolerance : scalar
            The relative error to which the ODE should get solved.
        
        args : tuple
            Additional arguments to pass to the user-defined function FAndF_u.  For example, if FAndF_u has the
            FAndF_u(t, uData, a, b, c), then args must be a tuple of length 3.

        Notes
        =====
        This method uses B-splines as finite elements.  The ODE itself is discretized using
        collocation.
        """
        return bspy._spline_fitting.solve_ode(self, nLeft, nRight, FAndF_u, tolerance, args)

    @staticmethod
    def sphere(radius, tolerance = None):
        """
        Construct a sphere of the given radius.

        Parameters
        ----------
        radius : scalar
            The desired radius of the sphere
        
        tolerance : scalar, optional
            The desired absolute tolerance to which the sphere should be constructed.  Defaults
            to 1.0e-12 if tolerance == None.

        Returns
        -------
        spline : `Spline`
            A bi-quartic spline approximation to a sphere of the specified radius, accurate to
            the given tolerance.
        
        Notes
        -----
        The resulting mapping is defined over the unit square.  The first independent variable
        corresponds to the latitude of the sphere, with 0 mapping to the south pole, 1 mapping to
        the north pole, and 0.5 mapping to the equator.  The second independent variable is the
        longitude, with 0 mapping to the meridian which is coplanar with the positive x-axis.

        See Also
        --------
        `circular_arc` : Create a 2D circular arc for a given radius and angle accurate to within
                         a given tolerance.
        `multiply` : Multiply two splines together.
        """
        return bspy._spline_fitting.sphere(radius, tolerance)

    def split(self, minContinuity = 0, breaks = None):
        """
        Split a spline into separate spline pieces.

        Parameters
        ----------
        minContinuity : `int`, optional
            The minimum expected continuity of each spline piece. The default is zero, for C0 continuity.

        breaks : `iterable` of length `nInd` or `None`, optional
            An iterable that specifies the breaks at which to separate the spline. 
            len(breaks[ind]) == 0 if there the spline isn't separated for the `ind` independent variable. 
            If breaks is `None` (the default), the spline will only be separated at discontinuities.

        Returns
        -------
        splineArray : array of `Spline`
            A array of splines with nInd dimensions containing the spline pieces. 

        See Also
        --------
        `join` : Join a list of splines together into a single spline.
        """
        return bspy._spline_domain.split(self, minContinuity, breaks)
    
    def subtract(self, other, indMap = None):
        """
        Subtract two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to subtract from self. The number of dependent variables must match self.

        indMap : `iterable` or `None`, optional
            An iterable of indices or pairs of indices. Each index refers to an independent variable.
            Within the iterable, a single index, `n`, maps the nth independent variable of self to the same independent variable of other.
            A pair `(n, m)` maps the nth independent variable of self to the mth independent variable of other. 
            For example, if you wanted to compute `self(u, v, w) - other(u, w)`, you'd pass `[0, (2, 1)]` for `indMap`. 
            Unmapped independent variables remain independent (the default).
            The domains of mapped independent variables must match. 
            An independent variable can map to no more than one other independent variable.

        Returns
        -------
        spline : `Spline`
            The result of subtracting other from self.

        See Also
        --------
        `add` : Add two splines.
        `multiply` : Multiply two splines.
        `common_basis` : Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Notes
        -----
        Equivalent to spline1 - spline2 with indMap = range(min(spline1.nInd, spline2.nInd)).
        
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
        if indMap is not None:
            indMap = [(mapping, mapping) if np.isscalar(mapping) else mapping for mapping in indMap]
        return self.add(other.scale(-1.0), indMap)

    def tangent_space(self, uvw):
        """
        Return the tangent space of the spline. (Same as Jacobian.)

        Parameters
        ----------
        uvw : array-like
            The value at which to evaluate the tangent space.

        Returns
        -------
        tangentSpace : `numpy.array`
            The nDep x nInd matrix of tangent vectors (tangents are the columns).
        """
        return bspy._spline_evaluation.jacobian(self, uvw)

    def to_dict(self):
        """
        Return a `dict` with `Spline` data.

        Returns
        -------
        dictionary : `dict`

        See Also
        --------
        `from_dict` : Create a `Spline` from a data in a `dict`.
        """
        return {"type" : "Spline", "nInd" : self.nInd, "nDep" : self.nDep, "order" : self.order, "nCoef" : self.nCoef,
            "knots" : self.knots, "coefs" : self.coefs, "metadata" : self.metadata}

    @staticmethod
    def torus(innerRadius, outerRadius, tolerance = None):
        """
        Construct a torus of the given radii.

        Parameters
        ----------
        innerRadius : scalar
            The desired inner radius of the torus
        
        outerRadius : scalar
            The desired outer radius of the torus
        
        tolerance : scalar, optional
            The desired absolute tolerance to which the tolerance should be constructed.  Defaults
            to 1.0e-12 if tolerance == None.

        Returns
        -------
        spline : `Spline`
            A bi-quartic spline approximation to a torus of the specified radii, accurate to
            the given tolerance.
        
        Notes
        -----
        The resulting mapping is defined over the unit square.  The first independent variable
        controls the angular position of the cross sectional circles.  The second independent
        variable specifies the point on each cross section.

        See Also
        --------
        `circular_arc` : Create a 2D circular arc for a given radius and angle accurate to within
                         a given tolerance.
        `add` : Add two splines together.
        `multiply` : Multiply two splines together.
        """
        return bspy._spline_fitting.torus(innerRadius, outerRadius, tolerance)
    
    def transform(self, matrix, matrixInverseTranspose = None):
        """
        Transform a spline by the given matrix.

        Parameters
        ----------
        matrix : array-like
            An array of size `newNDep`x`nDep` that specifies the transform matrix.

        matrixInverseTranspose : `numpy.array`, optional
            The inverse transpose of matrix (not used for splines).

        Returns
        -------
        spline : `Spline`
            The transformed spline.

        See Also
        --------
        `scale` : Scale a spline by the given scalar or scale vector.
        `translate` : Translate a spline by the given translation vector.

        Notes
        -----
        Equivalent to spline @ matrix.
        """
        return bspy._spline_operations.transform(self, matrix)

    def translate(self, translationVector):
        """
        Translate a spline by the given translation vector.

        Parameters
        ----------
        translationVector : array-like
            An array of length `nDep` that specifies the translation vector.

        Returns
        -------
        spline : `Spline`
            The translated spline.

        See Also
        --------
        `scale` : Scale a spline by the given scalar or scale vector.
        `transform` : Transform a spline by the given matrix.
        
        Notes
        -----
        Equivalent to spline + translationVector.
        """
        return bspy._spline_operations.translate(self, translationVector)

    def transpose(self, axes=None):
        """
        Transpose the domain of a spline, changing the order of independent variables.

        Parameters
        ----------
        axes : iterable, optional
            If specified, it must be an iterable which contains a permutation of [0,1,…,nInd-1]. 
            The i'th independent variable of the returned spline will correspond to the axes[i] variable of the input. 
            If not specified, defaults to range(nInd)[::-1], which reverses the order of the independent variables.

        Returns
        -------
        spline : `Spline`
            Trimmed spline. If all the knots are unchanged, the original spline is returned.

        See Also
        --------
        `numpy.transpose` : Returns an array with axes transposed.

        Notes
        -----
        Use spline.transpose(argsort(axes)) to invert the transposition when using the axes argument.
        """
        return bspy._spline_domain.transpose(self, axes)

    def trim(self, newDomain):
        """
        Trim the domain of a spline.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new lower and upper bounds on each of the independent variables (same form as 
            returned from `domain`). If a bound is None or nan then the original bound (and knots) are left unchanged.

        Returns
        -------
        spline : `Spline`
            Trimmed spline. If all the knots are unchanged, the original spline is returned.

        See Also
        --------
        `domain` : Return the domain of a spline.
        `extrapolate` : Extrapolate a spline out to an extended domain maintaining a given order of continuity.
        """
        return bspy._spline_domain.trim(self, newDomain)

    def trimmed_range_bounds(self, domainBounds):
        """
        Return the trimmed range bounds for the spline.

        Parameters
        ----------
        domainBounds : array-like
            An array with shape (nInd, 2) of lower and upper and lower bounds on each independent variable.

        Returns
        -------
        trimmedSpline, rangeBounds : `Spline`, `np.array`
            A spline trimmed to the given domain bounds, and the range of the trimmed spline given as 
            lower and upper bounds on each dependent variable.

        See Also
        --------
        `trim` : Trim the domain of a spline.
        `range_bounds` : Return the range of a spline as lower and upper bounds on each of the
            dependent variables.
        """
        return bspy._spline_domain.trimmed_range_bounds(self, domainBounds)

    def unfold(self, foldedInd, coefficientlessSpline):
        """
        Unfold the coefficients of an original spline's indicated independent variables back into the spline, using the 
        indicated independent variables' knots and orders from a second spline with no coefficients.

        Parameters
        ----------
        foldedInd : `iterable`
            An iterable that specifies the independent variables whose coefficients should be unfolded.

        coefficientlessSpline : `Spline`
            The coefficientless spline that retains the indicated independent variables' knots and orders.

        Returns
        -------
        unfoldedSpline : `Spline`, `Spline`
            The unfolded spline.

        See Also
        --------
        `fold` : Fold the coefficients of a spline's indicated independent variables into the coefficients of the remaining independent variables.

        Notes
        -----
        Given a spline whose coefficients are an nDep x n0 x ... x nk array and a list of (ordered) indices which form a proper subset of {n0, ... , nk}, it should return 2 splines.
        The first one is a spline with k + 1 - length(index subset) independent variables where the knot sets of all the dimensions which have been removed have been removed.  However, 
        all of the coefficient data is still intact, so that the resulting coefficient array has shape (nDep nj0 nj1 ... njj) x nk0 x ... x nkk.  The second spline should be a spline 
        with 0 dependent variables which contains all the knot sequences that were removed from the spline.  The unfold method takes the two splines as input and reverses the process, 
        returning the original spline.

        Here's an example. Suppose spl is a spline with 3 independent variables and 3 dependent variables which has nCoef = [4, 5, 6] and knots = [knots0, knots1, knots2]. 
        Then spl.fold([0, 2]) should return a spline with 1 independent variable and 72 dependent variables.  It should have nCoef = [5], knots = [knots1], and its coefs array should have 
        shape (72, 5).  The other spline should have 0 dependent variables, 2 independent variables, and knots = [knots0, knots2].  How things get ordered in coefs probably doesn't matter 
        so long as unfold unscrambles things in the corresponding way.  The second spline is needed to hold the basis information that was dropped so that it can be undone.
        """
        return bspy._spline_domain.unfold(self, foldedInd, coefficientlessSpline)

    def zeros(self, epsilon=None):
        """
        Find the roots of a spline (nInd must match nDep).

        Parameters
        ----------
        epsilon : `float`, optional
            Tolerance for root precision. The root will be within epsilon of the actual root. 
            The default is the machine epsilon.

        Returns
        -------
        roots : `iterable`
            An iterable containing the roots of the spline. If the spline is 
            zero over an interval, that root will appear as a tuple of the interval. 
            For curves (nInd == 1), the roots are ordered.
        
        See Also
        --------
        `intersect` : Intersect two splines.
        `contours` : Find all the contour curves of a spline whose `nInd` is one larger than its `nDep`.

        Notes
        -----
        For curves (nInd == 1), it implements interval Newton's method from Grandine, Thomas A. "Computing zeroes of spline functions." 
        Computer Aided Geometric Design 6, no. 2 (1989): 129-136.
        For all higher dimensions, it implements the projected-polyhedron technique from Sherbrooke, Evan C., and Nicholas M. Patrikalakis. 
        "Computation of the solutions of nonlinear polynomial systems." Computer Aided Geometric Design 10, no. 5 (1993): 379-405.
        """
        if not(self.nInd == self.nDep): raise ValueError("The number of independent variables (nInd) must match the number of dependent variables (nDep).")
        if self.nInd <= 1:
            return bspy._spline_intersection.zeros_using_interval_newton(self)
        else:
            return bspy._spline_intersection.zeros_using_projected_polyhedron(bspy.spline_block.SplineBlock(self), epsilon)
 