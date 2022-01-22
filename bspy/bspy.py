import numpy as np
import functools
from error import *

########################################################################
##
##     Header and constructor for the Spline class
##
########################################################################

class Spline:
    def __init__(self, nInd = 1, nDep = 1, order = [4], nCoef = [4],
                 knots = [[0, 0, 0, 0, 1, 1, 1, 1]],
                 coefs = [[0], [0], [0], [1]]):
        assert nInd >= 0, "nInd < 0"
        self.nInd = int(nInd)
        assert nDep >= 0, "nDep < 0"
        self.nDep = int(nDep)
        assert len(order) == self.nInd, "len(order) != nInd"
        self.order = [int(no) for no in order]
        assert len(nCoef) == self.nInd, "len(nCoef) != nInd"
        self.nCoef = [int(nc) for nc in nCoef]
        assert len(knots) == nInd, "len(knots) != nInd"
        for ix in range(len(knots)):
            knotlength = self.order[ix] + self.nCoef[ix]
            assert len(knots[ix]) == knotlength, \
                "Knots array for variable " + str(ix) + \
                " should have length " + str(knotlength)
        self.knots = [np.array([knot for knot in knotlist], float)
                      for knotlist in knots]
        for klist, no, nc in zip(self.knots, self.order, self.nCoef):
            for ix in range(nc):
                assert klist[ix] <= klist[ix + 1] and \
                       klist[ix] < klist[ix + no], \
                       "Improperly ordered knot sequence"
        nc = functools.reduce(lambda a, b: a * b, self.nCoef)
        assert len(coefs) == nc or len(coefs) == self.nDep, \
            "Length of coefs should be " + str(nc) + " or " + \
            str(self.nDep)
        ncoef = list(self.nCoef)
        ncoef.reverse()
        if len(coefs) == nc:
            self.coefs = np.array(coefs, float).reshape(
                ncoef + [self.nDep]).T
        else:
            self.coefs = [np.array(cl, float).reshape(ncoef).T
                          for cl in coefs]
            self.coefs = np.array(self.coefs)

########################################################################
##
##     The __call__ method uses the de Boor recurrence relations for a
##     B-spline series to evaluate a spline function.
##
########################################################################

    def __call__(self, uvw):
        def b_spline_values(knot, knots, order, u):
            basis = np.zeros(order)
            basis[-1] = 1.0
            for degree in range(1, order):
                b = order - degree
                for ix in range(knot - degree, knot):
                    gap = knots[ix + degree] - knots[ix]
                    alpha = (u - knots[ix]) / gap
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

########################################################################
##
##     Show actual spline definition when requested
##
########################################################################

    def __repr__(self):
        return "Spline" + str((self.nInd, self.nDep, self.order,
                               self.nCoef, list(self.knots), self.coefs))

########################################################################
##
##     Return the domain of a spline
##
########################################################################

    def domain(self):
        dom = []
        for ix in range(self.nInd):
            dom.append([self.knots[ix][self.order[ix] - 1],
                        self.knots[ix][self.nCoef[ix]]])
        return np.array(dom)
