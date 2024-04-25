import numpy as np

def block_evaluate(self, uwv):
    value = np.zeros(self.nDep, self.coefsDtype)
    nDep = 0
    for row in self.block:
        nInd = 0
        for spline in row:
            value[nDep:nDep + spline.nDep] += spline(uwv[nInd:nInd + spline.nInd])
            nInd += spline.nInd
        nDep += spline.nDep
    return value

def block_jacobian(self, uwv):
    jacobian = np.zeros((self.nDep, self.nInd), self.coefsDtype)
    nDep = 0
    for row in self.block:
        nInd = 0
        for spline in row:
            jacobian[nDep:nDep + spline.nDep, nInd:nInd + spline.nInd] += spline.jacobian(uwv[nInd:nInd + spline.nInd])
            nInd += spline.nInd
        nDep += spline.nDep
    return jacobian

def block_normal(self, uvw, normalize=True, indices=None):
    # Make work for scalar valued functions
    uvw = np.atleast_1d(uvw)

    if abs(self.nInd - self.nDep) != 1: raise ValueError("The number of independent variables must be one different than the number of dependent variables.")

    # Evaluate the Jacobian at the point.
    tangentSpace = self.jacobian(uvw)
    
    # Record the larger dimension and ensure it comes first.
    if self.nInd > self.nDep:
        nDep = self.nInd
        tangentSpace = tangentSpace.T
    else:
        nDep = self.nDep
    
    # Compute the normal using cofactors (determinants of subsets of the tangent space).
    sign = -1 if self.metadata.get("flipNormal", False) else 1
    if indices is None:
        indices = range(nDep)
        normal = np.empty(nDep, self.coefs.dtype)
    else:
        normal = np.empty(len(indices), self.coefs.dtype)
    for i in indices:
        normal[i] = sign * np.linalg.det(tangentSpace[[j for j in range(nDep) if i != j]])
        sign *= -1
    
    # Normalize the result as needed.
    if normalize:
        normal /= np.linalg.norm(normal)
    
    return normal

def bspline_values(knot, knots, splineOrder, u, derivativeOrder = 0, taylorCoefs = False):
    basis = np.zeros(splineOrder, knots.dtype)
    if knot is None:
        knot = np.searchsorted(knots, u, side = 'right')
        knot = min(knot, len(knots) - splineOrder)
    if derivativeOrder >= splineOrder:
        return knot, basis
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
    return knot, basis

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
    bValues = []
    for iv in range(self.nInd):
        ix, indValues = bspline_values(None, self.knots[iv], self.order[iv], uvw[iv], with_respect_to[iv])
        bValues.append(indValues)
        mySection.append(slice(ix - self.order[iv], ix))
    myCoefs = self.coefs[tuple(mySection)]
    for iv in range(self.nInd - 1, -1, -1):
        myCoefs = myCoefs @ bValues[iv]
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
    bValues = []
    for iv in range(self.nInd):
        ix, indValues = bspline_values(None, self.knots[iv], self.order[iv], uvw[iv])
        bValues.append(indValues)
        mySection.append(slice(ix - self.order[iv], ix))
    myCoefs = self.coefs[tuple(mySection)]
    for iv in range(self.nInd - 1, -1, -1):
        myCoefs = myCoefs @ bValues[iv]
    return myCoefs

def greville(self, ind = 0):
    if ind < 0 or ind >= self.nInd:  raise ValueError("Invalid independent variable")
    myKnots = self.knots[ind]
    knotAverages = 0
    if self.order[ind] == 1:
        knotAverages = 0.5 * (myKnots[1:] + myKnots[:-1])
    else:
        for ix in range(1, self.order[ind]):
            knotAverages = knotAverages + myKnots[ix : ix + self.nCoef[ind]]
        knotAverages /= (self.order[ind] - 1)
    return knotAverages

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

    # Evaluate the Jacobian at the point.
    tangentSpace = self.jacobian(uvw)
    
    # Record the larger dimension and ensure it comes first.
    if self.nInd > self.nDep:
        nDep = self.nInd
        tangentSpace = tangentSpace.T
    else:
        nDep = self.nDep
    
    # Compute the normal using cofactors (determinants of subsets of the tangent space).
    sign = -1 if self.metadata.get("flipNormal", False) else 1
    if indices is None:
        indices = range(nDep)
        normal = np.empty(nDep, self.coefs.dtype)
    else:
        normal = np.empty(len(indices), self.coefs.dtype)
    for i in indices:
        normal[i] = sign * np.linalg.det(tangentSpace[[j for j in range(nDep) if i != j]])
        sign *= -1
    
    # Normalize the result as needed.
    if normalize:
        normal /= np.linalg.norm(normal)
    
    return normal

def range_bounds(self):
    # Assumes self.nDep is the first value in self.coefs.shape
    bounds = [[coefficient.min(), coefficient.max()] for coefficient in self.coefs]
    return np.array(bounds, self.coefs.dtype)