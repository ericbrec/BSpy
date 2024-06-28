import numpy as np
import scipy as sp

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

def composed_integral(self, integrand = None, domain = None):
    # Determine domain and check its validity
    actualDomain = self.domain()
    if domain is None:
        domain = actualDomain
    else:
        for iInd in range(self.nInd):
            if domain[iInd, 0] < actualDomain[iInd, 0] or \
               domain[iInd, 1] > actualDomain[iInd, 1]:
                raise ValueError("Can't integrate beyond the domain of the spline")

    # Determine breakpoints for quadrature intervals; require functions to be analytic

    uniqueKnots = []
    for iInd in range(self.nInd):
        iStart = np.searchsorted(self.knots[iInd], domain[iInd, 0], side = 'right')
        iEnd = np.searchsorted(self.knots[iInd], domain[iInd, 1], side = 'right')
        uniqueKnots.append(np.unique(np.insert(self.knots[iInd], [iStart, iEnd], domain[iInd])[iStart : iEnd + 2]))

    # Set integrand function if none is given
    if integrand is None:
        integrand = lambda x : 1.0

    # Set tolerance
    tolerance = 1.0e-13 / self.nInd

    # Establish the callback function
    def composedIntegrand(u, nIndSoFar, uValues):
        uValues[nIndSoFar] = u
        nIndSoFar += 1
        if self.nInd == nIndSoFar:
            total = integrand(self(uValues)) * \
                    np.prod(np.linalg.svd(self.jacobian(uValues), compute_uv = False))
        else:
            total = 0.0
            for ix in range(len(uniqueKnots[nIndSoFar]) - 1):
                value = sp.integrate.quad(composedIntegrand, uniqueKnots[nIndSoFar][ix],
                                          uniqueKnots[nIndSoFar][ix + 1], (nIndSoFar, uValues),
                                          epsabs = tolerance, epsrel = tolerance)
                total += value[0]
        return total    
    
    # Compute the value by calling the callback routine
    total = composedIntegrand(0.0, -1, self.nInd * [0.0])
    return total

def continuity(self):
    multiplicity = np.array([np.max(np.unique(knots, return_counts = True)[1][1 : -1]) for knots in self.knots])
    continuity = self.order - multiplicity - 1
    return continuity

def curvature(self, uv):
    if self.nDep == 1:
        self = self.graph()
    if self.nInd == 1:
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
    if self.nInd == 2:
        su = self.derivative([1, 0], uv)
        sv = self.derivative([0, 1], uv)
        normal = self.normal(uv)
        suu = self.derivative([2, 0], uv)
        suv = self.derivative([1, 1], uv)
        svv = self.derivative([0, 2], uv)
        E = su @ su
        F = su @ sv
        G = sv @ sv
        L = suu @ normal
        M = suv @ normal
        N = svv @ normal
        return (L * N - M ** 2) / (E * G - F ** 2)

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
    domain = self.domain()[ind]
    knotAverages = np.minimum(domain[1], np.maximum(domain[0], knotAverages))
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
    sign = -1 if hasattr(self, "metadata") and self.metadata.get("flipNormal", False) else 1
    dtype = self.coefs.dtype if hasattr(self, "coefs") else self.coefsDtype
    if indices is None:
        indices = range(nDep)
        normal = np.empty(nDep, dtype)
    else:
        normal = np.empty(len(indices), dtype)
    for ix, i in enumerate(indices):
        normal[ix] = sign * ((-1) ** i) * np.linalg.det(tangentSpace[[j for j in range(nDep) if i != j]])
    
    # Normalize the result as needed.
    if normalize:
        normal /= np.linalg.norm(normal)
    
    return normal

def range_bounds(self):
    # Assumes self.nDep is the first value in self.coefs.shape
    bounds = [[coefficient.min(), coefficient.max()] for coefficient in self.coefs]
    return np.array(bounds, self.coefs.dtype)