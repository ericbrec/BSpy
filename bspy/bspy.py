import numpy as np
from error import *

class Spline:
    def __init__(self, nind = 1, ndep = 1, order = [4], ncoef = [4],
                 knots = [[0, 0, 0, 0, 1, 1, 1, 1]],
                 coefs = [[0], [0], [0], [1]]):
        if nind < 0:
            raise InvalidNindError(nind)
        self.nind = int(nind)
        if ndep < 0:
            raise InvalidNdepError(ndep)
        self.ndep = int(ndep)
        if len(order) != self.nind:
            raise InvalidNorderError(order)
        self.order = [int(no) for no in order]
        if len(ncoef) != self.nind:
            raise InvalidNcoefsError(ncoef)
        self.ncoef = [int(nc) for nc in ncoef]
        if len(knots) != nind:
            raise InvalidKnotsError(len(knots))
        for ix in range(len(knots)):
            if len(knots[ix]) != self.order[ix] + self.ncoef[ix]:
                raise InvalidKnotSequenceError(ix, knots[ix])
        self.knots = [np.array([float(knot) for knot in knotlist])
                      for knotlist in knots]
        for knotlist, order in zip(self.knots, self.order):
            for ix in range(self.ncoef[ix]):
                if knotlist[ix] > knotlist[ix + 1] or \
                   knotlist[ix] >= knotlist[ix + order]:
                    raise InvalidKnotSequenceError(ix, knotlist)
        nc = 1
        for ncoef in self.ncoef:
            nc *= ncoef
        if len(coefs) != nc:
            raise InvalidCoefficientError(coefs)
        for cpt in coefs:
            if len(cpt) != self.ndep:
                raise InvalidCoefficientError(cpt)
        self.coefs = np.array([[float(coef) for coef in cpt]
                               for cpt in coefs])

# Evaluate the spline when object is called

    def __call__(self, uvw):
        def bsvalues(ix, knots, order, value):
            bsvals = np.zeros(order)
            bsvals[-1] = 1.0
            for ir in range(1, order):
                for iy in range(ir):
                    alfa = (value - knots[ix - ir + iy]) / \
                           (knots[ix + iy] - knots[ix - ir + iy])
                    bsvals[-1 - ir + iy] += \
                        (1.0 - alfa) * bsvals[-ir + iy]
                    bsvals[-ir + iy] *= alfa
            return bsvals
        dom = self.domain()
        for ix in range(self.nind):
            if uvw[ix] < dom[ix][0] or uvw[ix] > dom[ix][1]:
                raise ArgumentOutsideDomainError(uvw)
        
        ix = np.searchsorted(self.knots[0], uvw[0], 'right')
        ix = min(ix, self.ncoef[0])
        bsvals = bsvalues(ix, self.knots[0], self.order[0], uvw[0])
        cpts = self.coefs[ix - self.order[0]:ix].T
        return cpts @ bsvals

# Show actual spline definition when requested

    def __repr__(self):
        return str((self.nind, self.ndep, self.order, self.ncoef,
                    self.knots, self.coefs))

# Return the domain of a spline

    def domain(self):
        dom = []
        for ix in range(self.nind):
            dom.append([self.knots[ix][self.order[ix] - 1],
                        self.knots[ix][self.ncoef[ix]]])
        return np.array(dom)
