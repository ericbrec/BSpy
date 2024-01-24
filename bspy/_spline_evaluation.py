import numpy as np

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

    # Make work for scalar valued functions
    uvw = np.atleast_1d(uvw)

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

def bspline_values(knot, knots, splineOrder, u, derivativeOrder = 0, taylorCoefs = False):
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

def curvature(self, uv):
    if self.nInd == 1:
        if self.nDep == 1:
            self = self.graph()
        fp = self.derivative([1], uv)
        fpp = self.derivative([2], uv)
        fpDotFp = fp @ fp
        fpDotFpp = fp @ fpp
        denom = fpDotFp ** 1.5
        if self.nDep == 2:
            numerator = fp[0] * fpp[1] - fp[1] * fpp[0]
        else:
            numerator = np.sqrt((fpp @ fpp) * fpDotFp - fpDotFpp ** 2)
        return numerator / denom 

def derivative(self, with_respect_to, uvw):
    # Make work for scalar valued functions
    uvw = np.atleast_1d(uvw)

    # Check for the correct number of independent variables
    if len(uvw) != self.nInd:
        raise ValueError(f"Incorrect number of parameter values: {len(uvw)}")

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
        bValues = bspline_values(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv], with_respect_to[iv])
        myCoefs = myCoefs @ bValues
    return myCoefs

def domain(self):
    dom = [[self.knots[i][self.order[i] - 1],
            self.knots[i][self.nCoef[i]]] for i in range(self.nInd)]
    return np.array(dom)

def evaluate(self, uvw):
    # Make work for scalar valued functions
    uvw = np.atleast_1d(uvw)

    # Check for the correct number of independent variables
    if len(uvw) != self.nInd:
        raise ValueError(f"Incorrect number of parameter values: {len(uvw)}")

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
        bValues = bspline_values(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv])
        myCoefs = myCoefs @ bValues
    return myCoefs

def integral(self, with_respect_to, uvw1, uvw2, returnSpline = False):
    # Make work for scalar valued functions
    uvw1 = np.atleast_1d(uvw1)
    uvw2 = np.atleast_1d(uvw2)

    # Check for evaluation point inside domain
    dom = self.domain()
    for ix in range(self.nInd):
        if uvw1[ix] < dom[ix][0] or uvw1[ix] > dom[ix][1]:
            raise ValueError(f"Spline evaluation outside domain: {uvw1}")
        if uvw2[ix] < dom[ix][0] or uvw2[ix] > dom[ix][1]:
            raise ValueError(f"Spline evaluation outside domain: {uvw2}")

    # Repeatedly integrate self
    spline = self
    for i in range(self.nInd):
        for j in range(with_respect_to[i]):
            spline = spline.integrate(i)

    value = spline(uvw2) - spline(uvw1)
    if returnSpline:
        return value, spline
    else:
        return value

def jacobian(self, uvw):
    value = np.empty((self.nDep, self.nInd), self.coefs.dtype)
    wrt = [0] * self.nInd
    for i in range(self.nInd):
        wrt[i] = 1
        value[:, i] = self.derivative(wrt, uvw)
        wrt[i] = 0
    
    return value

def normal(self, uvw, normalize=True, indices=None):
    # Make work for scalar valued functions
    uvw = np.atleast_1d(uvw)

    if abs(self.nInd - self.nDep) != 1: raise ValueError("The number of independent variables must be one different than the number of dependent variables.")

    # Evaluate the tangents at the point.
    tangentSpace = np.empty((self.nInd, self.nDep), self.coefs.dtype)
    with_respect_to = [0] * self.nInd
    for i in range(self.nInd):
        with_respect_to[i] = 1
        tangentSpace[i] = self.derivative(with_respect_to, uvw)
        with_respect_to[i] = 0
    
    # If self.nInd > self.nDep, transpose the tangent space and adjust the length of the normal.
    nDep = self.nDep
    if self.nInd > nDep:
        tangentSpace = tangentSpace.T
        nDep = self.nInd
    
    # Compute the normal using cofactors (determinants of subsets of the tangent space).
    sign = 1
    if indices is None:
        indices = range(nDep)
        normal = np.empty(nDep, self.coefs.dtype)
    else:
        normal = np.empty(len(indices), self.coefs.dtype)
    for i in indices:
        normal[i] = sign * np.linalg.det(np.delete(tangentSpace, i, 1))
        sign *= -1
    
    # Normalize the result as needed.
    if normalize:
        normal /= np.linalg.norm(normal)
    
    return normal

def range_bounds(self):
    # Assumes self.nDep is the first value in self.coefs.shape
    bounds = [[coefficient.min(), coefficient.max()] for coefficient in self.coefs]
    return np.array(bounds, self.coefs.dtype)