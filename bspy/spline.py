from turtle import width
import numpy as np
from os import path
from bspy.error import *

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

    def derivative(self, with_respect_to, uvw):
        """
        Compute the derivative of the spline at a given parameter value.

        Parameters
        ----------
        with_respect_to : `iterable`
            An iterable of length nInd that specifies the integer order of derivative for each independent variable.
            A zero-order derivative just evaluates the spline normally.
        
        uvw : `iterable`
            An iterable of length nInd that specifies the values of each independent variable (the parameter value).

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
        mySection = [range(self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(range(ix - self.order[iv], ix))
        mySection = np.ix_(*mySection)
        myCoefs = self.coefs[mySection]
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

    def evaluate(self, uvw):
        """
        Compute the value of the spline at a given parameter value.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length nInd that specifies the values of each independent variable (the parameter value).

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
        mySection = [range(self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(range(ix - self.order[iv], ix))
        mySection = np.ix_(*mySection)
        myCoefs = self.coefs[mySection]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = b_spline_values(myIndices[iv], self.knots[iv],
                                     self.order[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs
    
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