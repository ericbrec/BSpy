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
    functions (spline functions) as linear combinations of B-splines. 

    Parameters
    ----------
    nInd : `int`
        The number of independent variables of the spline

    nDep : `int`
        The number of dependent variables of the spline
    
    order : `tuple`
        A tuple of length nInd where each integer entry represents the
        polynomial order of the function in that variable

    nCoef : `tuple`
        A tuple of length nInd where each integer entry represents the
        dimension (i.e. number of B-spline coefficients) of the function
        space in that variable

    knots : `list`
        A list of the lists of the knots of the spline in each independent variable

    coefs : array-like
        A list of the B-spline coefficients of the spline.
    
    accuracy : `float`, optional
        Each spline function is presumed to be an approximation of something else. 
        The `accuracy` stores the infinity norm error of the difference between 
        the given spline function and that something else. Default is zero.

    metadata : `dict`, optional
        A dictionary of ancillary data to store with the spline. Default is {}.
    """
    
    def __init__(self, nInd, nDep, order, nCoef, knots, coefs, accuracy = 0.0, metadata = {}):
        assert nInd >= 0, "nInd < 0"
        self.nInd = int(nInd)
        assert nDep >= 0, "nDep < 0"
        self.nDep = int(nDep)
        assert len(order) == self.nInd, "len(order) != nInd"
        self.order = tuple(int(x) for x in order)
        assert len(nCoef) == self.nInd, "len(nCoef) != nInd"
        self.nCoef = tuple(int(x) for x in nCoef)
        assert len(knots) == nInd, "len(knots) != nInd"
        for i in range(len(knots)):
            nKnots = self.order[i] + self.nCoef[i]
            assert len(knots[i]) == nKnots, \
                f"Knots array for variable {i} should have length {nKnots}"
        self.knots = tuple(np.array(kk) for kk in knots)
        for knots, order, nCoef in zip(self.knots, self.order, self.nCoef):
            for i in range(nCoef):
                assert knots[i] <= knots[i + 1] and knots[i] < knots[i + order],\
                       "Improperly ordered knot sequence"
        totalCoefs = 1
        for nCoef in self.nCoef:
            totalCoefs *= nCoef
        assert len(coefs) == totalCoefs or len(coefs) == self.nDep, \
            f"Length of coefs should be {totalCoefs} or {self.nDep}"
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

    def __rmatmul__ (self, other):
        if _isIterable(other):
            if isinstance(other, np.ndarray) and len(other.shape) == 2:
                return self.transform(other)
            else:
                return self.dot(other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other) or _isIterable(other):
            return self.scale(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other) or _isIterable(other):
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
            An iterable of pairs of indices. 
            Each pair (n, m) maps the mth independent variable of other to the nth independent variable of self. 
            The domains of the nth and mth independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            Unmapped independent variables remain independent (the default).

        Returns
        -------
        spline : `Spline`
            The result of adding other to self.

        See Also
        --------
        `subtract` : Subtract two splines.
        `multiply` : Multiply two splines.
        `common_basis : Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Notes
        -----
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
        return bspy._spline_operations.add(self, other, indMap)

    def blossom(self, uvw):
        """
        Compute the blossom of the spline at a given parameter values.

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

    def derivative(self, with_respect_to, uvw):
        """
        Compute the derivative of the spline at a given parameter value.

        Parameters
        ----------
        with_respect_to : `iterable`
            An iterable of length `nInd` that specifies the integer order of derivative for each independent variable.
            A zero-order derivative just evaluates the spline normally.
        
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter value).

        Returns
        -------
        value : `numpy.array`
            The value of the derivative of the spline at the given parameter value.

        See Also
        --------
        `evaluate` : Compute the value of the spline at a given parameter value.
        `differentiate` : Differentiate a spline with respect to one of its independent variables, returning the resulting spline.

        Notes
        -----
        The derivative method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline function.  The non-zero B-splines are
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
        vector : array-like
            An array of length `nDep` that specifies the vector.

        Returns
        -------
        spline : `Spline`
            The dotted spline.
        """
        return bspy._spline_evaluation.dot(self, vector)

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
        Compute the value of the spline at a given parameter value.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter value).

        Returns
        -------
        value : `numpy.array`
            The value of the spline at the given parameter value.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at a given parameter value.

        Notes
        -----
        The evaluate method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline function.  The non-zero B-splines are
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

    @staticmethod
    def least_squares(dataPoints):
        """
        Fit a curve to a string of data points using the method of least squares.

        Parameters
        ----------
        dataPoints : `iterable` containing the data points to fit.
            Each of the data points is of length nDep.

        Returns
        -------
        spline : `Spline`
            A spline curve which approximates the data points.
        """
        return bspy._spline_fitting.least_squares(dataPoints)

    @staticmethod
    def load(fileName, splineType=None):
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

    def multiply(self, other, indMap = None):
        """
        Multiply two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to multiply by self. The number of dependent variables must match self.

        indMap : `iterable` or `None`, optional
            An iterable of pairs of indices. 
            Each pair (n, m) maps the mth independent variable of other to the nth independent variable of self. 
            The domains of the nth and mth independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            Unmapped independent variables remain independent (the default).

        Returns
        -------
        spline : `Spline`
            The result of adding other to self.

        See Also
        --------
        `add` : Add two splines.
        `subtract` : Subtract two splines.
        `common_basis : Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Notes
        -----
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
        return bspy._spline_operations.multiply(self, other, indMap)

    def range_bounds(self):
        """
        Return the range of a spline as upper and lower bounds on each of the
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
            nInd x 2 array of the new upper and lower bounds on each of the independent variables. 
            Same form as returned from `domain`.

        Returns
        -------
        spline : `Spline`
            Reparametrized spline.

        See Also
        --------
        `domain` : Return the domain of a spline.
        """
        return bspy._spline_domain.reparametrize(self, newDomain)

    def save(self, fileName):
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
        multiplier : scalar or array-like
            A scalar or an array of length `nDep` that specifies the multiplier.

        Returns
        -------
        spline : `Spline`
            The scaled spline.

        See Also
        --------
        `transform` : Transform a spline by the given matrix.
        `translate` : Translate a spline by the given translation vector.
        """
        return bspy._spline_operations.scale(self, multiplier)

    def subtract(self, other, indMap = None):
        """
        Subtract two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to subtract from self. The number of dependent variables must match self.

        indMap : `iterable` or `None`, optional
            An iterable of pairs of indices. 
            Each pair (n, m) maps the mth independent variable of other to the nth independent variable of self. 
            The domains of the nth and mth independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            Unmapped independent variables remain independent (the default).

        Returns
        -------
        spline : `Spline`
            The result of subtracting other from self.

        See Also
        --------
        `add` : Add two splines.
        `multiply` : Multiply two splines.
        `common_basis : Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Notes
        -----
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
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
            An ordered iterable containing the roots of the spline. If the spline is 
            zero over an interval, that root will appear as a tuple of the interval.

        Notes
        -----
        Currently, the algorithm only works for nInd == 1. 
        Implements the algorithm from Grandine, Thomas A. "Computing zeroes of spline functions." Computer Aided Geometric Design 6, no. 2 (1989): 129-136.
        """
        return bspy._spline_intersection.zeros(self, epsilon)