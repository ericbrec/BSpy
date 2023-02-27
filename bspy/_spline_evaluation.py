import numpy as np
import bspy.spline

def blossom(self, uvw):
    def blossom_values(knot, knots, order, u):
        basis = np.zeros(order, knots.dtype)
        basis[-1] = 1.0
        for degree in range(1, order):
            b = order - degree
            for i in range(knot - degree, knot):
                alpha = (u[degree - 1] - knots[i]) / (knots[i + degree] - knots[i])
                basis[b - 1] += (1.0 - alpha) * basis[b]
                basis[b] *= alpha
                b += 1
        return basis

    # Check for evaluation point inside domain
    dom = self.domain()
    for ix in range(self.nInd):
        if uvw[ix][0] < dom[ix][0] or uvw[ix][self.order[ix]-2] > dom[ix][1]:
            raise ValueError(f"Spline evaluation outside domain: {uvw}")

    # Grab all of the appropriate coefficients
    mySection = [slice(0, self.nDep)]
    myIndices = []
    for iv in range(self.nInd):
        ix = np.searchsorted(self.knots[iv], uvw[iv][0], 'right')
        ix = min(ix, self.nCoef[iv])
        myIndices.append(ix)
        mySection.append(slice(ix - self.order[iv], ix))
    myCoefs = self.coefs[tuple(mySection)]
    for iv in range(self.nInd - 1, -1, -1):
        bValues = blossom_values(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv])
        myCoefs = myCoefs @ bValues
    return myCoefs

def bsplineValues(knot, knots, splineOrder, u, derivativeOrder = 0, taylorCoefs = False):
    basis = np.zeros(splineOrder, knots.dtype)
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
        derivativeAdjustment = degree / (splineOrder - degree if taylorCoefs else 1.0)
        for i in range(knot - degree, knot):
            alpha = derivativeAdjustment / (knots[i + degree] - knots[i])
            basis[b - 1] += -alpha * basis[b]
            basis[b] *= alpha
            b += 1
    return basis

def cross(self, vector):
    if isinstance(vector, bspy.spline.Spline):
        return self.multiply(vector, None, 'C')
    else:
        assert self.nDep == 3
        assert len(vector) == self.nDep

        coefs = np.empty(self.coefs.shape, self.coefs.dtype)
        coefs[0] = vector[2] * self.coefs[1] - vector[1] * self.coefs[2]
        coefs[1] = vector[0] * self.coefs[2] - vector[2] * self.coefs[0]
        coefs[2] = vector[1] * self.coefs[0] - vector[0] * self.coefs[1]
        return type(self)(self.nInd, 3, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

def derivative(self, with_respect_to, uvw):
    # Check for evaluation point inside domain
    dom = self.domain()
    for ix in range(self.nInd):
        if uvw[ix] < dom[ix][0] or uvw[ix] > dom[ix][1]:
            raise ValueError(f"Spline evaluation outside domain: {uvw}")

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
        bValues = bsplineValues(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv], with_respect_to[iv])
        myCoefs = myCoefs @ bValues
    return myCoefs

def domain(self):
    dom = [[self.knots[i][self.order[i] - 1],
            self.knots[i][self.nCoef[i]]] for i in range(self.nInd)]
    return np.array(dom)

def dot(self, vector):
    if isinstance(vector, bspy.spline.Spline):
        return self.multiply(vector, None, 'D')
    else:
        assert len(vector) == self.nDep

        coefs = vector[0] * self.coefs[0]
        for i in range(1, self.nDep):
            coefs += vector[i] * self.coefs[i]
        return type(self)(self.nInd, 1, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

def evaluate(self, uvw):
    # Check for evaluation point inside domain
    dom = self.domain()
    for ix in range(self.nInd):
        if uvw[ix] < dom[ix][0] or uvw[ix] > dom[ix][1]:
            raise ValueError(f"Spline evaluation outside domain: {uvw}")

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
        bValues = bsplineValues(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv])
        myCoefs = myCoefs @ bValues
    return myCoefs

def range_bounds(self):
    # Assumes self.nDep is the first value in self.coefs.shape
    bounds = [[coefficient.min(), coefficient.max()] for coefficient in self.coefs]
    return np.array(bounds, self.coefs.dtype)