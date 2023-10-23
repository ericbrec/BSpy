import numpy as np
from os import path
import bspy._spline_domain
import bspy._spline_evaluation
import bspy._spline_intersection
import bspy._spline_fitting
import bspy._spline_operations

def _isIterable(object):
    result = True
    try:
        iterator = iter(object)
    except TypeError:
        result = False
    return result

class Spline:
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
    
    accuracy : `float`, optional
        Each spline is presumed to be an approximation of something else. 
        The `accuracy` stores the infinity norm error of the difference between 
        the given spline and that something else. Default is zero.

    metadata : `dict`, optional
        A dictionary of ancillary data to store with the spline. Default is {}.
    """
    
    def __init__(self, nInd, nDep, order, nCoef, knots, coefs, accuracy = 0.0, metadata = {}):
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
        self.accuracy = accuracy
        self.metadata = dict(metadata)

    def __call__(self, uvw):
        return self.evaluate(uvw)

    def __repr__(self):
        return f"Spline({self.nInd}, {self.nDep}, {self.order}, " + \
               f"{self.nCoef}, {self.knots} {self.coefs}, {self.accuracy}, " + \
               f"{self.metadata})"

    def __add__(self, other):
        if isinstance(other, Spline):
            return self.add(other)
        elif _isIterable(other):
            return self.translate(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Spline):
            return other.add(self)
        elif _isIterable(other):
            return self.translate(other)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if _isIterable(other):
            if not isinstance(other, np.ndarray):
                other = np.array(other)
            if len(other.shape) == 2:
                return self.transform(other.T)
            else:
                return self.dot(other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if _isIterable(other):
            if not isinstance(other, np.ndarray):
                other = np.array(other)
            if len(other.shape) == 2:
                return self.transform(other)
            else:
                return self.dot(other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Spline):
            if self.nDep == 1 or other.nDep == 1:
                return self.multiply(other, None, 'S')
            else:
                return self.multiply(other, None, 'D')
        elif np.isscalar(other) or _isIterable(other):
            return self.scale(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Spline):
            if self.nDep == 1 or other.nDep == 1:
                return other.multiply(self, None, 'S')
            else:
                return other.multiply(self, None, 'D')
            return other.multiply(self)
        elif np.isscalar(other) or _isIterable(other):
            return self.scale(other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Spline):
            return self.subtract(other)
        elif _isIterable(other):
            return self.translate(-np.array(other))
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Spline):
            return other.subtract(self)
        elif _isIterable(other):
            spline = self.scale(-1.0)
            return spline.translate(other)
        else:
            return NotImplemented

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
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
        if indMap is not None:
            indMap = [mapping if _isIterable(mapping) else (mapping, mapping) for mapping in indMap]
        return bspy._spline_operations.add(self, other, indMap)

    def blossom(self, uvw):
        """
        Compute the blossom of the spline at given parameter values.

        Parameters
        ----------
        uvwValues : `iterable`
            An iterable of length `nInd` that specifies the degree-sized vectors of blossom parameters for each independent variable.

        Returns
        -------
        value : `numpy.array`
            The value of the spline's blossom at the given blossom parameters.

        See Also
        --------
        `evaluate` : Compute the value of the spline at a given parameter value.

        Notes
        -----
        Evaluates the blossom based on blossoming algorithm 1 found in Goldman, Ronald N. "Blossoming and knot insertion algorithms for B-spline curves." 
        Computer Aided Geometric Design 7, no. 1-4 (1990): 69-81.
        """
        return bspy._spline_evaluation.blossom(self, uvw)

    @staticmethod
    def bspline_values(knot, knots, splineOrder, u, derivativeOrder = 0, taylorCoefs = False):
        """
        Compute bspline values or their derivatives for a 1D bspline segment given the rightmost knot of the segment and a parameter value within that segment.

        Parameters
        ----------
        knot : `int`
            The rightmost knot in the bspline segment.

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

    def common_basis(self, splines, indMap):
        """
        Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Parameters
        ----------
        splines : `iterable`
            The collection of N - 1 splines to align (N total splines, including self).

        indMap : `iterable`
            The collection of independent variables to align. Since each spline can have multiple 
            independent variables, `indMap` is an `iterable` of `iterables` (like a list of lists). 
            Each collection of indices (i0, i1, .. iN) maps the i'th independent variable to each other. 
            The domains of mapped independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            If all the splines are curves (1 independent variable), then `indMap` is ((0, 0, .. 0),).

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
        return bspy._spline_domain.common_basis(self, splines, indMap)

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
            of zero. The default is square root of machine epsilon. The actual accuracy of the contour 
            is returned as `spline.accuracy`.
            
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
        return bspy._spline_intersection.contours(self)

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
                'S' is for a scalar product (nDep must be 1 for one of the splines).
        
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
            indMap = [(*(mapping if _isIterable(mapping) else (mapping, mapping)), True) for mapping in indMap]
        return bspy._spline_operations.multiplyAndConvolve(self, other, indMap, productType)

    def copy(self, metadata={}):
        """
        Create a copy of a spline.

        Parameters
        ----------
        metadata : `dict`, optional
            A dictionary of ancillary data to store with the spline. Default is {}.
        
        Returns
        -------
        spline : `Spline`
            The spline copy.
        """
        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, self.coefs, self.accuracy, metadata)

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

    def derivative(self, with_respect_to, uvw):
        """
        Compute the derivative of the spline at given parameter values.

        Parameters
        ----------
        with_respect_to : `iterable`
            An iterable of length `nInd` that specifies the integer order of derivative for each independent variable.
            A zero-order derivative just evaluates the spline normally.
        
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).

        Returns
        -------
        value : `numpy.array`
            The value of the derivative of the spline at the given parameter values.

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
        return bspy._spline_evaluation.derivative(self, with_respect_to, uvw)

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
            nInd x 2 array of the upper and lower bounds on each of the independent variables.

        See Also
        --------
        `reparametrize` : Reparametrize a spline to match new domain bounds
        `trim` : Trim the domain of a spline.
        """
        return bspy._spline_evaluation.domain(self)

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

    def evaluate(self, uvw):
        """
        Compute the value of the spline at given parameter values.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).

        Returns
        -------
        value : `numpy.array`
            The value of the spline at the given parameter values.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at given parameter values.

        Notes
        -----
        The evaluate method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline.  The non-zero B-splines are
        evaluated, then the dot product of those B-splines with the vector of
        B-spline coefficients is computed.
        """
        return bspy._spline_evaluation.evaluate(self, uvw)

    def extrapolate(self, newDomain, continuityOrder):
        """
        Extrapolate a spline out to an extended domain maintaining a given order of continuity.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new upper and lower bounds on each of the independent variables (same form as 
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
        value : `numpy.array`
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
        Intersect two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to intersect with self (`other.nDep` match match `self.nDep`).

        Returns
        -------
        intersection : `iterable` or `NotImplemented`
            If `self.nInd + other.nInd - self.nDep` is 0, returns an iterable of intersection points in the 
            parameter space of the two splines (a vector of size `self.nInd + other.nInd`).
            If `self.nInd + other.nInd - self.nDep` is 1, returns an iterable of `Spline` curves, each of whose domain is [0, 1] 
            and each of whose range is in the parameter space of the two splines (a vector of size `self.nInd + other.nInd`).
            If `self.nInd + other.nInd - self.nDep` is < 0 or > 1, `NotImplemented` is returned.
        
        See Also
        --------
        `zeros` : Find the roots of a spline (nInd must match nDep).
        `contours` : Find all the contour curves of a spline.

        Notes
        -----
        Uses `zeros` to find all intersection points and `contours` to find all the intersection curves.
        """
        if not(self.nDep == other.nDep): raise ValueError("The number of dependent variables for both splines much match.")
        freeParameters = self.nInd + other.nInd - self.nDep
        if freeParameters == 0:
            return (self - other).zeros()
        elif freeParameters == 1:
            return (self - other).contours()
        else:
            return NotImplemented

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
    def least_squares(nInd, nDep, order, dataPoints, knots = None, compression = 0, metadata = {}):
        """
        Fit a spline to an array of data points using the method of least squares.

        Parameters
        ----------
        nInd : `int`
            The number of independent variables of the spline.

        nDep : `int`
            The number of dependent variables of the spline.
        
        order : `tuple`
            A tuple of length nInd where each integer entry represents the
            polynomial order of the spline in that variable. When in doubt,
            use `[4] * nInd` (a cubic spline).

        dataPoints : `iterable` of array-like values
            A collection of data points. Each data point is an array of length `nInd + nDep`. 
            The first `nInd` values designate the independent values for the point.
            The next `nDep` values designate the dependent values for the point. 
            The data points need not form a regular mesh nor be in any particular order.
            In addition, each data point may instead have `nInd + nDep * (nInd + 1)` values, 
            the first `nInd` being independent and the next `nInd + 1` sets of `nDep` values designating 
            the dependent point and its first derivatives with respect to each independent variable.

        knots : `list`, optional
            A list of the lists of the knots of the spline in each independent variable.
            Default is `None`, in which case knots are chosen automatically.
        
        compression : `int`, optional
            The desired compression of data used as a percentage of the number of data points (0 - 99). 
            This percentage is used to determine the total number of spline coefficients when 
            knots are chosen automatically (it's ignored otherwise). The actual compression will be slightly less 
            because the number of coefficients is rounded up. The default value is zero (interpolation, no compression).

        metadata : `dict`, optional
            A dictionary of ancillary data to store with the spline. Default is {}.

        Returns
        -------
        spline : `Spline`
            A spline curve which approximates the data points.

        Notes
        -----
        Uses `numpy.linalg.lstsq` to compute the least squares solution. The returned spline.accuracy is computed 
        from the sum of the residual across dependent variables and the system epsilon. 
        The algorithm to choose knots automatically is from Piegl, Les A., and Wayne Tiller. 
        "Surface approximation to scanned data." The visual computer 16 (2000): 386-395.
        """
        return bspy._spline_fitting.least_squares(nInd, nDep, order, dataPoints, knots, compression, metadata)

    @staticmethod
    def load(fileName, splineType=None):
        """
        Load a spline from the specified filename (full path).

        Parameters
        ----------
        fileName : `string`
            The full path to the file containing the spline. Can be a relative path.
        
        splineType : `type`, optional
            The class type that should be created. It must be an instance of Spline (the default).
        
        Returns
        -------
        spline : `Spline`
            The loaded spline.

        See Also
        --------
        `save` : Save a spline to the specified filename (full path).

        Notes
        -----
        Uses numpy's load function.
        """
        kw = np.load(fileName)
        order = kw["order"]
        nInd = len(order)
        knots = []
        for i in range(nInd):
            knots.append(kw[f"knots{i}"])
        coefficients = kw["coefficients"]

        if splineType is None:
            splineType = Spline
        spline = splineType(nInd, coefficients.shape[0], order, coefficients.shape[1:], knots, coefficients, metadata=dict(Path=path, Name=path.splitext(path.split(fileName)[1])[0]))
        return spline

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
                'S' is for a scalar product (nDep must be 1 for one of the splines).
        
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
        Taken in part from Lee, E. T. Y. "Computing a chain of blossoms, with application to products of splines." 
        Computer Aided Geometric Design 11, no. 6 (1994): 597-620.
        """
        if indMap is not None:
            indMap = [(*(mapping if _isIterable(mapping) else (mapping, mapping)), False) for mapping in indMap]
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
        return bspy._spline_operations.normal_spline(self, indices)

    def range_bounds(self):
        """
        Return the range of a spline as lower and upper bounds on each of the
        dependent variables
        """
        return bspy._spline_evaluation.range_bounds(self)

    def remove_knots(self, oldKnots=((),), maxRemovalsPerKnot=0, tolerance=None):
        """
        Remove interior knots from a spline.

        Parameters
        ----------
        oldKnots : `iterable` of length `nInd`, optional
            An iterable that specifies the knots that can be removed from each independent variable's interior knots. 
            len(newKnots[ind]) == 0 if all interior knots can be removed for the `ind` independent variable (the default). 
            Knots that don't appear in the independent variable's interior knots are ignored.
        
        maxRemovalsPerKnot : `int`, optional
            A non-zero count of the largest number of times a knot can be removed. For example, one means that 
            only one instance of each knot can be removed. (Zero means each knot can be removed completely, 
            which is the default.)
        
        tolerance : `float` or `None`, optional
            The maximum residual error permitted after removing a knot. Knots will not be removed if the 
            resulting residual error is above this threshold. Default is `None`, meaning all specified knots 
            will be removed up to `maxRemovalsPerKnot`.

        Returns
        -------
        spline : `Spline`
            A spline with the knots removed.
        
        totalRemoved : `int`
            The total number of knots removed.
        
        residualError : `float`
            The residual error relative to the old spline. (The returned spline's accuracy is also adjusted accordingly.)

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `trim` : Trim the domain of a spline.

        Notes
        -----
        Implements a variation of the algorithms from Tiller, Wayne. "Knot-removal algorithms for NURBS curves and surfaces." Computer-Aided Design 24, no. 8 (1992): 445-453.
        """
        return bspy._spline_domain.remove_knots(self, oldKnots, maxRemovalsPerKnot, tolerance)

    def reparametrize(self, newDomain):
        """
        Reparametrize a spline to match new domain bounds. The spline's number of knots and its coefficients remain unchanged.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new upper and lower bounds on each of the independent variables (same form as 
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

    @staticmethod
    def ruled_surface(spline1, spline2):
       return bspy._spline_fitting.ruled_surface(spline1, spline2)
    
    def save(self, fileName):
        """
        Save a spline to the specified filename (full path).

        Parameters
        ----------
        fileName : `string`
            The full path to the file containing the spline. Can be a relative path.
        
        See Also
        --------
        `load` : Load a spline from the specified filename (full path).

        Notes
        -----
        Uses numpy's savez function. Accuracy and metadata are not saved.
        """
        kw = {}
        kw["order"] = order=np.array(self.order, np.int32)
        for i in range(len(self.knots)):
            kw[f"knots{i}"] = self.knots[i]
        kw["coefficients"] = self.coefs
        np.savez(fileName, **kw )

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
        """
        return bspy._spline_fitting.section(xytk)

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
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
        if indMap is not None:
            indMap = [mapping if _isIterable(mapping) else (mapping, mapping) for mapping in indMap]
        return self.add(other.scale(-1.0), indMap)

    def transform(self, matrix, maxSingularValue=None):
        """
        Transform a spline by the given matrix.

        Parameters
        ----------
        matrix : array-like
            An array of size `newNDep`x`nDep` that specifies the transform matrix.

        maxSingularValue : `float`, optional
            The largest singular value of `matrix`, used to update the accuracy of the spline. 
            If no value is provided (default), the largest singular value is computed.

        Returns
        -------
        spline : `Spline`
            The transformed spline.

        See Also
        --------
        `scale` : Scale a spline by the given scalar or scale vector.
        `translate` : Translate a spline by the given translation vector.
        """
        return bspy._spline_operations.transform(self, matrix, maxSingularValue)

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
        """
        return bspy._spline_operations.translate(self, translationVector)

    def trim(self, newDomain):
        """
        Trim the domain of a spline.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new upper and lower bounds on each of the independent variables (same form as 
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
            The default is the max of spline accuracy and machine epsilon.

        Returns
        -------
        roots : `iterable`
            An iterable containing the roots of the spline. If the spline is 
            zero over an interval, that root will appear as a tuple of the interval. 
            For curves (nInd == 1), the roots are ordered.
        
        See Also
        --------
        `intersect` : Intersect two splines.
        `contour` : Fit a spline to the contour defined by `F(x) = 0`.

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
            return bspy._spline_intersection.zeros_using_projected_polyhedron(self, epsilon)