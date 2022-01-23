import numpy as np
from error import *

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
    metadata - a pointer to any ancillary data to store with the spline 
    """
    
    def __init__(self, nInd = 1, nDep = 1, order = [4], nCoef = [4],
                 knots = [[0, 0, 0, 0, 1, 1, 1, 1]],
                 coefs = [[0, 0, 0 ,1]], accuracy = 0.0, metadata = None):
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
        self.knots = tuple(np.array([k for k in kk], float) for kk in knots)
        for knots, order, nCoef in zip(self.knots, self.order, self.nCoef):
            for i in range(nCoef):
                assert knots[i] <= knots[i + 1] and knots[i] < knots[i + order],\
                       "Improperly ordered knot sequence"
        totalCoefs = 1
        for nCoef in self.nCoef:
            totalCoefs *= nCoef
        assert len(coefs) == totalCoefs or len(coefs) == self.nDep, \
            f"Length of coefs should be {totalCoefs} or {self.nDep}"
        nCoef = list(self.nCoef)
        nCoef.reverse()
        if len(coefs) == totalCoefs:
            self.coefs = np.array(coefs, float).reshape(nCoef + [self.nDep]).T
        else:
            self.coefs = np.array([np.array(c, float) for c in coefs])
            if coefs.shape != [self.nDep] + nCoef:
                self.coefs = np.array([c.reshape(nCoef).T for c in coefs])
        self.accuracy = accuracy
        self.metadata = metadata

    def __call__(self, uvw):
        """
        The __call__ method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline function.  The non-zero B-splines are
        evaluated, then the dot product of those B-splines with the vector of
        B-spline coefficients is computed
        """
        def b_spline_values(knot, knots, order, u):
            basis = np.zeros(order)
            basis[-1] = 1.0
            for degree in range(1, order):
                b = order - degree
                for i in range(knot - degree, knot):
                    gap = knots[i + degree] - knots[i]
                    alpha = (u - knots[i]) / gap
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

        mysection = [range(self.nDep)]
        myix = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myix.append(ix)
            mysection.append(range(ix - self.order[iv], ix))
        mysection = np.ix_(*mysection)
        mycoefs = self.coefs[mysection]
        for iv in range(self.nInd - 1, -1, -1):
            bvals = b_spline_values(myix[iv], self.knots[iv],
                                     self.order[iv], uvw[iv])
            mycoefs = mycoefs @ bvals
        return mycoefs

    def __repr__(self):
        return f"Spline({self.nInd}, {self.nDep}, {self.order}, " + \
               f"{self.nCoef}, {self.knots} {self.coefs}, {self.accuracy}, " + \
               f"{self.metadata})"

    def domain(self):
        """
        Return the domain of a spline as upper and lower bounds on each of the
        independent variables
        """
        dom = [[self.knots[i][self.order[i] - 1],
                self.knots[i][self.nCoef[i]]] for i in range(self.nInd)]
        return np.array(dom)
