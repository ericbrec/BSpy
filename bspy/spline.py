from turtle import width
import numpy as np
from os import path
from bspy.error import *

def isIterable(object):
    result = True
    try:
        iterator = iter(object)
    except TypeError:
        result = False
    return result

class Spline:
    """
    A class to model, represent, and process piecewise polynomial tensor product
    functions (spline functions) as linear combinations of B-splines.  The
    attributes of this class have the following meaning: 

    nInd - the number of independent variables of the spline
    nDep - the number of dependent variables of the spline
    order - a tuple of length nInd where each integer entry represents the
            polynomial order of the function in that variable
    nCoef - a tuple of length nInd where each integer entry represents the
            dimension (i.e. number of B-spline coefficients) of the function
            space in that variable
    knots - a list of the lists of the knots of the spline in each independent
            variable
    coefs - a list of the B-spline coefficients of the spline
    accuracy - each spline function is presumed to be an approximation of
               something else.  This attribute stores the infinity norm error of
               the difference between the given spline function and that
               something else
    metadata - a dictionary of ancillary data to store with the spline 
    """
    
    def __init__(self, nInd = 1, nDep = 1, order = [4], nCoef = [4],
                 knots = [[0, 0, 0, 0, 1, 1, 1, 1]],
                 coefs = [[0, 0, 0 ,1]], accuracy = 0.0, metadata = {}):
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
        self.metadata = metadata

    def __call__(self, uvw):
        return self.evaluate(uvw)

    def __repr__(self):
        return f"Spline({self.nInd}, {self.nDep}, {self.order}, " + \
               f"{self.nCoef}, {self.knots} {self.coefs}, {self.accuracy}, " + \
               f"{self.metadata})"

    def __add__(self, other):
        if isIterable(other):
            return self.translate(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isIterable(other):
            return self.translate(other)
        else:
            return NotImplemented

    def __rmatmul__ (self, other):
        if isIterable(other):
            if isinstance(other, np.ndarray) and len(other.shape) == 2:
                return self.transform(other)
            else:
                return self.dot(other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other) or isIterable(other):
            return self.scale(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other) or isIterable(other):
            return self.scale(other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isIterable(other):
            return self.translate(-np.array(other))
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isIterable(other):
            spline = self.scale(self.nDep * [-1.0])
            return spline.translate(other)
        else:
            return NotImplemented

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
        def b_spline_values(knot, knots, splineOrder, derivativeOrder, u):
            basis = np.zeros(splineOrder)
            basis[-1] = 1.0
            for degree in range(1, splineOrder - derivativeOrder):
                b = splineOrder - degree
                for i in range(knot - degree, knot):
                    alpha = (u - knots[i]) / (knots[i + degree] - knots[i])
                    basis[b - 1] += (1.0 - alpha) * basis[b]
                    basis[b] *= alpha
                    b += 1
            for degree in range(splineOrder - derivativeOrder, splineOrder):
                b = splineOrder - degree
                for i in range(knot - degree, knot):
                    alpha = degree / (knots[i + degree] - knots[i])
                    basis[b - 1] += -alpha * basis[b]
                    basis[b] *= alpha
                    b += 1
            return basis

        # Check for evaluation point inside domain
        dom = self.domain()
        for ix in range(self.nInd):
            if uvw[ix] < dom[ix][0] or uvw[ix] > dom[ix][1]:
                raise ArgumentOutsideDomainError(uvw)

        # Grab all of the appropriate coefficients
        mySection = [slice(0, self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(slice(ix - self.order[iv], ix))
        myCoefs = self.coefs[tuple(mySection)]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = b_spline_values(myIndices[iv], self.knots[iv], self.order[iv], with_respect_to[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs

    def differentiate(self, with_respect_to = 0):
        """
        Differentiate a spline with respect to one of its independent variables, returning the resulting spline.

        Parameters
        ----------
        with_respect_to : integer
            The number of the independent variable to differentiate.

        Returns
        -------
        spline : `Spline`
            The spline that results from differentiating the original spline with respect to the given independent variable.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at a given parameter value.
        """
        assert 0 <= with_respect_to < self.nInd
        assert self.order[with_respect_to] > 1

        order = [*self.order]
        order[with_respect_to] -= 1
        degree = order[with_respect_to] 

        nCoef = [*self.nCoef]
        nCoef[with_respect_to] -= 1

        dKnots = self.knots[with_respect_to][1:-1]
        knots = [self.knots]
        knots[with_respect_to] = dKnots

        coefs = np.delete(self.coefs, 0, axis=with_respect_to + 1) # first axis is the dependent variable
        sliceList = (self.nInd + 1) * [slice(None)]
        for i in range(nCoef[with_respect_to]):
            sliceList[with_respect_to + 1] = i
            sliceTuple = tuple(sliceList)
            alpha =  degree / (dKnots[i+degree] - dKnots[i])
            coefs[sliceTuple] = alpha * (coefs[sliceTuple] - self.coefs[sliceTuple])
        
        return type(self)(self.nInd, self.nDep, order, nCoef, knots, coefs, self.accuracy, self.metadata)

    def domain(self):
        """
        Return the domain of a spline as upper and lower bounds on each of the
        independent variables
        """
        dom = [[self.knots[i][self.order[i] - 1],
                self.knots[i][self.nCoef[i]]] for i in range(self.nInd)]
        return np.array(dom)

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
        assert len(vector) == self.nDep

        coefs = vector[0] * self.coefs[0]
        for i in range(1, self.nDep):
            coefs += vector[i] * self.coefs[i]
        return type(self)(self.nInd, 1, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

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
        def b_spline_values(knot, knots, order, u):
            basis = np.zeros(order)
            basis[-1] = 1.0
            for degree in range(1, order):
                b = order - degree
                for i in range(knot - degree, knot):
                    alpha = (u - knots[i]) / (knots[i + degree] - knots[i])
                    basis[b - 1] += (1.0 - alpha) * basis[b]
                    basis[b] *= alpha
                    b += 1
            return basis

        # Check for evaluation point inside domain
        dom = self.domain()
        for ix in range(self.nInd):
            if uvw[ix] < dom[ix][0] or uvw[ix] > dom[ix][1]:
                raise ArgumentOutsideDomainError(uvw)

        # Grab all of the appropriate coefficients
        mySection = [slice(0, self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(slice(ix - self.order[iv], ix))
        myCoefs = self.coefs[tuple(mySection)]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = b_spline_values(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs

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
        assert 0 < len(foldedInd) < self.nInd
        foldedOrder = []
        foldedNCoef = []
        foldedKnots = []

        coefficientlessOrder = []
        coefficientlessNCoef = []
        coefficientlessKnots = []

        coefficientMoveFrom = []
        foldedNDep = self.nDep

        for ind in range(self.nInd):
            if ind in foldedInd:
                coefficientlessOrder.append(self.order[ind])
                coefficientlessNCoef.append(self.nCoef[ind])
                coefficientlessKnots.append(self.knots[ind])
                coefficientMoveFrom.append(ind + 1)
                foldedNDep *= self.nCoef[ind]
            else:
                foldedOrder.append(self.order[ind])
                foldedNCoef.append(self.nCoef[ind])
                foldedKnots.append(self.knots[ind])

        coefficientMoveTo = range(1, len(coefficientMoveFrom) + 1)
        foldedCoefs = np.moveaxis(self.coefs, coefficientMoveFrom, coefficientMoveTo).reshape((foldedNDep, *foldedNCoef))
        coefficientlessCoefs = np.empty((0, *coefficientlessNCoef))

        foldedSpline = type(self)(len(foldedOrder), foldedNDep, foldedOrder, foldedNCoef, foldedKnots, foldedCoefs, self.accuracy, self.metadata)
        coefficientlessSpline = type(self)(len(coefficientlessOrder), 0, coefficientlessOrder, coefficientlessNCoef, coefficientlessKnots, coefficientlessCoefs, self.accuracy, self.metadata)
        return foldedSpline, coefficientlessSpline
    
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
            A spline with the new knots inserted.
        """
        assert len(newKnots) == self.nInd
        knots = list(self.knots)
        coefs = self.coefs
        for ind in range(self.nInd):
            for knot in newKnots[ind]:
                position = np.searchsorted(knots[ind], knot, 'right')
                newCoefs = np.insert(coefs, position - 1, 0.0, axis=ind + 1)
                sliceList = (self.nInd + 1) * [slice(None)]
                sliceList2 = (self.nInd + 1) * [slice(None)]
                for i in range(position - self.order[ind] + 1, position):
                    alpha = (knot - knots[ind][i]) / (knots[ind][i + self.order[ind] - 1] - knots[ind][i])
                    sliceList[ind + 1] = i - 1
                    sliceList2[ind + 1] = i
                    newCoefs[sliceList2] = (1.0 - alpha) * coefs[sliceList] + alpha * coefs[sliceList2]
                knots[ind] = np.insert(knots[ind], position, knot)
                coefs = newCoefs
        
        return type(self)(self.nInd, self.nDep, self.order, coefs.shape[1:], knots, coefs, self.accuracy, self.metadata)

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

    def range_bounds(self):
        """
        Return the range of a spline as upper and lower bounds on each of the
        dependent variables
        """
        # Assumes self.nDep is the first value in self.coefs.shape
        bounds = [[coefficient.min(), coefficient.max()] for coefficient in self.coefs]
        return np.array(bounds)

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
        assert np.isscalar(multiplier) or len(multiplier) == self.nDep

        if np.isscalar(multiplier):
            accuracy = multiplier * self.accuracy
            coefs = multiplier * self.coefs
        else:
            accuracy = np.linalg.norm(multiplier) * self.accuracy
            coefs = np.array(self.coefs)
            for i in range(self.nDep):
                coefs[i] *= multiplier[i]
        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, coefs, accuracy, self.metadata)

    def transform(self, matrix):
        """
        Transform a spline by the given matrix.

        Parameters
        ----------
        matrix : array-like
            An array of size `nDep`x`nDep` that specifies the transform matrix.

        Returns
        -------
        spline : `Spline`
            The transformed spline.

        See Also
        --------
        `scale` : Scale a spline by the given scalar or scale vector.
        `translate` : Translate a spline by the given translation vector.
        """
        assert matrix.shape == (self.nDep, self.nDep)

        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, matrix @ self.coefs, self.accuracy, self.metadata)

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
        assert len(translationVector) == self.nDep

        coefs = np.array(self.coefs)
        for i in range(self.nDep):
            coefs[i] += translationVector[i]
        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

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
        assert len(foldedInd) == coefficientlessSpline.nInd
        unfoldedOrder = []
        unfoldedNCoef = []
        unfoldedKnots = []
        
        coefficientMoveTo = []
        unfoldedNDep = self.nDep
        indFolded = 0
        indCoefficientless = 0

        for ind in range(self.nInd + coefficientlessSpline.nInd):
            if ind in foldedInd:
                unfoldedOrder.append(coefficientlessSpline.order[indCoefficientless])
                unfoldedNCoef.append(coefficientlessSpline.nCoef[indCoefficientless])
                unfoldedKnots.append(coefficientlessSpline.knots[indCoefficientless])
                unfoldedNDep //= coefficientlessSpline.nCoef[indCoefficientless]
                coefficientMoveTo.append(ind + 1)
                indCoefficientless += 1
            else:
                unfoldedOrder.append(self.order[indFolded])
                unfoldedNCoef.append(self.nCoef[indFolded])
                unfoldedKnots.append(self.knots[indFolded])
                indFolded += 1

        coefficientMoveFrom = range(1, coefficientlessSpline.nInd + 1)
        unfoldedCoefs = np.moveaxis(self.coefs.reshape(unfoldedNDep, *coefficientlessSpline.nCoef, *self.nCoef), coefficientMoveFrom, coefficientMoveTo)

        unfoldedSpline = type(self)(len(unfoldedOrder), unfoldedNDep, unfoldedOrder, unfoldedNCoef, unfoldedKnots, unfoldedCoefs, self.accuracy, self.metadata)
        return unfoldedSpline