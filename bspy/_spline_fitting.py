import numpy as np
import scipy as sp
import bspy.spline
import bspy.spline_block
import math

def circular_arc(radius, angle, tolerance = None):
    if tolerance is None:
        tolerance = 1.0e-12
    if radius < 0.0 or angle < 0.0 or tolerance < 0.0: raise ValueError("The radius, angle, and tolerance must be positive.")

    samples = int(max(np.ceil(((1.1536e-5 * radius / tolerance)**(1/8)) * angle / 90), 2.0)) + 1
    return bspy.Spline.section([(radius * np.cos(u * angle * np.pi / 180), radius * np.sin(u * angle * np.pi / 180), 90 + u * angle, 1.0 / radius) for u in np.linspace(0.0, 1.0, samples)])

def composition(splines, tolerance):
    # Define the callback function
    def composition_of_splines(u):
        for f in splines[::-1]:
            u = f(u)
        return u
    
    # Approximate this composition
    return bspy.Spline.fit(splines[-1].domain(), composition_of_splines, tolerance = tolerance)

def cone(radius1, radius2, height, tolerance = None):
    if tolerance is None:
        tolerance = 1.0e-12
    bigRadius = max(radius1, radius2)
    radius1 /= bigRadius
    radius2 /= bigRadius
    bottom = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]] @ bspy.Spline.circular_arc(bigRadius, 360.0, tolerance)
    top = radius2 * bottom + [0.0, 0.0, height]
    bottom = radius1 * bottom
    return bspy.Spline.ruled_surface(bottom, top)

# Courtesy of Michael Epton - Translated from his F77 code lgnzro
def _legendre_polynomial_zeros(degree, mapToZeroOne = True):
    def legendre(degree, x):
        p = [1.0, x]
        pd = [0.0, 1.0]
        for n in range(2, degree + 1):
            alfa = (2 * n - 1) / n
            beta = (n - 1) / n
            pd.append(alfa * (p[-1] + x * pd[-1]) - beta * pd[-2])
            p.append(alfa * x * p[-1] - beta * p[-2])
        return p, pd
    zval = 1.0
    z = []
    zNegative = []
    for iRoot in range(degree // 2):
        done = False
        while True:
            p, pd = legendre(degree, zval)
            sum = 0.0
            for zRoot in z:
                sum += 1.0 / (zval - zRoot)
            dz = p[-1] / (pd[-1] - sum * p[-1])
            zval -= dz
            if done:
                break
            if dz < 1.0e-10:
                done = True
        z.append(zval)
        zNegative.append(-zval)
        zval -= 0.001
    if degree % 2 == 1:
        zNegative.append(0.0)
    z.reverse()
    z = np.array(zNegative + z)
    w = []
    for zval in z:
        p, pd = legendre(degree, zval)
        w.append(2.0 / ((1.0 - zval ** 2) * pd[-1] ** 2))
    w = np.array(w)
    if mapToZeroOne:
        z = 0.5 * (1.0 + z)
        w = 0.5 * w
    return z, w

def contour(F, knownXValues, dF = None, epsilon = None, metadata = {}):
    # Set up parameters for initial guess of x(t) and validate arguments.
    order = 4
    degree = order - 1
    gaussNodes, gaussWeights = _legendre_polynomial_zeros(degree - 1)

    if not(len(knownXValues) >= 2): raise ValueError("There must be at least 2 known x values.")
    m = len(knownXValues) - 1
    nCoef = m * (degree - 1) + 2
    nUnknownCoefs = nCoef - 2

    # Validate known x values and rescale them to [0, 1].
    knownXValues = np.array(knownXValues)
    contourDtype = knownXValues.dtype
    if epsilon is None:
        epsilon = math.sqrt(np.finfo(contourDtype).eps)
    evaluationEpsilon = np.sqrt(epsilon)
    nDep = knownXValues.shape[1]
    for knownXValue in knownXValues:
        FValues = F(knownXValue)
        if not(len(FValues) == nDep - 1 and np.linalg.norm(FValues) < evaluationEpsilon):
            raise ValueError(f"F(known x) must be a zero vector of length {nDep - 1}.")

    # Record domain of F and scaling of coefficients.
    if isinstance(F, (bspy.Spline, bspy.SplineBlock)):
        FDomain = F.domain().T
        coefsMin = FDomain[0]
        coefsMaxMinusMin = FDomain[1] - FDomain[0]
    else:
        FDomain = np.array(nDep * [[-np.inf, np.inf]]).T
        coefsMin = knownXValues.min(axis=0)
        coefsMaxMinusMin = knownXValues.max(axis=0) - coefsMin
        coefsMaxMinusMin = np.where(coefsMaxMinusMin < 1.0, 1.0, coefsMaxMinusMin)

    # Rescale known values.
    coefsMaxMinMinReciprocal = np.reciprocal(coefsMaxMinusMin)
    knownXValues = (knownXValues - coefsMin) * coefsMaxMinMinReciprocal # Rescale to [0 , 1]

    # Establish the Jacobian of F.
    if dF is None:
        if isinstance(F, (bspy.Spline, bspy.SplineBlock)):
            dF = F.jacobian
        else:
            def fJacobian(x):
                value = np.empty((nDep - 1, nDep), float)
                for i in range(nDep):
                    h = epsilon * (1.0 + abs(x[i]))
                    xShift = np.array(x, copy=True)
                    xShift[i] -= h
                    fLeft = np.array(F(xShift))
                    h2 = h * 2.0
                    xShift[i] += h2
                    value[:, i] = (np.array(F(xShift)) - fLeft) / h2
                return value
            dF = fJacobian
    elif not callable(dF):
        if not(len(dF) == nDep): raise ValueError(f"Must provide {nDep} first derivatives.")
        def fJacobian(x):
            value = np.empty((nDep - 1, nDep), float)
            for i in range(nDep):
                value[:, i] = dF[i]
            return value
        dF = fJacobian

    # Construct knots, t values, and GSamples.
    tValues = np.empty(nUnknownCoefs, contourDtype)
    GSamples = np.empty((nUnknownCoefs, nDep), contourDtype)
    t = 0.0 # t ranges from 0 to 1
    dt = 1.0 / m
    knots = [t] * order
    i = 0
    previousPoint = knownXValues[0]
    for point in knownXValues[1:]:
        for gaussNode in gaussNodes:
            tValues[i] = t + gaussNode * dt
            GSamples[i] = (1.0 - gaussNode) * previousPoint + gaussNode * point
            i += 1
        t += dt
        knots += [t] * (order - 2)
        previousPoint = point
    knots += [t] * 2 # Clamp last knot
    knots = np.array(knots, contourDtype)
    knots[nCoef:] = 1.0 # Ensure last knot is exactly 1.0
    assert i == nUnknownCoefs
    
    # Start subdivision loop.
    while True:
        # Define G(coefs) to be dGCoefs @ coefs - GSamples,
        # where dGCoefs and GSamples are the B-spline values and sample points, respectively, for x(t).
        # The dGCoefs matrix is banded due to B-spline local support, so initialize it to zero.
        # Solving for coefs provides us our initial coefficients of x(t).
        dGCoefs = np.zeros((nUnknownCoefs, nDep, nCoef, nDep), contourDtype)
        i = 0
        for i, t in enumerate(tValues):
            ix, bValues = bspy.Spline.bspline_values(None, knots, order, t)
            for j in range(nDep):
                dGCoefs[i, j, ix - order:ix, j] = bValues
        GSamples -= dGCoefs[:, :, 0, :] @ knownXValues[0] + dGCoefs[:, :, -1, :] @ knownXValues[-1]
        GSamples = GSamples.reshape(nUnknownCoefs * nDep)
        dGCoefs = dGCoefs[:, :, 1:-1, :].reshape(nUnknownCoefs * nDep, nUnknownCoefs * nDep)
        coefs = np.empty((nCoef, nDep), contourDtype)
        coefs[0, :] = knownXValues[0]
        coefs[-1, :] = knownXValues[-1]
        coefs[1:-1, :] = np.linalg.solve(dGCoefs, GSamples).reshape(nUnknownCoefs, nDep)

        # Array to hold the values of F and contour dot for each t, excluding endpoints.
        FSamples = np.empty((nUnknownCoefs, nDep), contourDtype)
        # Array to hold the Jacobian of the FSamples with respect to the coefficients.
        # The Jacobian is banded due to B-spline local support, so initialize it to zero.
        dFCoefs = np.zeros((nUnknownCoefs, nDep, nCoef, nDep), contourDtype)

        # Start Newton's method loop.
        previousFSamplesNorm = 0.0
        while True:
            FSamplesNorm = 0.0
            # Fill in FSamples and its Jacobian (dFCoefs) with respect to the coefficients of x(t).
            for i, t in enumerate(tValues):
                # Isolate coefficients and compute bspline values and their first two derivatives at t.
                ix, bValues = bspy.Spline.bspline_values(None, knots, order, t)
                ix, dValues = bspy.Spline.bspline_values(ix, knots, order, t, 1)
                ix, d2Values = bspy.Spline.bspline_values(ix, knots, order, t, 2)
                compactCoefs = coefs[ix - order:ix, :]

                # Compute the dot constraint for x(t) and check for divergence from solution.
                dotValues = np.dot(compactCoefs.T @ d2Values, compactCoefs.T @ dValues)
                FSamplesNorm = max(FSamplesNorm, abs(dotValues))
                if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                    break

                # Do the same for F(x(t)).
                x = coefsMin + (compactCoefs.T @ bValues) * coefsMaxMinusMin
                x = np.maximum(FDomain[0], np.minimum(FDomain[1], x))
                FValues = F(x)
                FSamplesNorm = max(FSamplesNorm, np.linalg.norm(FValues, np.inf))
                if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                    break

                # Record FSamples for t.
                FSamples[i, :-1] = FValues
                FSamples[i, -1] = dotValues

                # Compute the Jacobian of FSamples with respect to the coefficients of x(t).
                FValues = np.outer(dF(x) * coefsMaxMinusMin, bValues).reshape(nDep - 1, nDep, order).swapaxes(1, 2)
                dotValues = (np.outer(d2Values, compactCoefs.T @ dValues) + np.outer(dValues, compactCoefs.T @ d2Values)).reshape(order, nDep)
                dFCoefs[i, :-1, ix - order:ix, :] = FValues
                dFCoefs[i, -1, ix - order:ix, :] = dotValues
            
            # Check if we got closer to the solution.
            if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                # No we didn't, take a dampened step.
                coefDelta *= 0.5
                coefs[1:-1, :] += coefDelta # Don't update endpoints
            else:
                # Yes we did, rescale FSamples and its Jacobian.
                if FSamplesNorm >= evaluationEpsilon:
                    FSamples /= FSamplesNorm
                    dFCoefs /= FSamplesNorm
                
                # Perform a Newton iteration.
                n = nUnknownCoefs * nDep
                HSamples = FSamples.reshape(n)
                dHCoefs = dFCoefs[:, :, 1:-1, :].reshape((n, n))
                bandWidth = order * nDep - 1
                banded = np.zeros((2 * bandWidth + 1, n))
                for iDiagonal in range(min(bandWidth + 1, n)):
                    banded[bandWidth - iDiagonal, iDiagonal : n]= np.diagonal(dHCoefs, iDiagonal)
                    banded[bandWidth + iDiagonal, : n - iDiagonal] = np.diagonal(dHCoefs, -iDiagonal)
                coefDelta = sp.linalg.solve_banded((bandWidth, bandWidth), banded, HSamples).reshape(nUnknownCoefs, nDep)
                coefs[1:-1, :] -= coefDelta # Don't update endpoints

                # Record FSamples norm to ensure this Newton step is productive.
                previousFSamplesNorm = FSamplesNorm

            # Check for convergence of step size.
            if np.linalg.norm(coefDelta) < epsilon:
                # If step didn't improve the solution, remove it.
                if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                    coefs[1:-1, :] += coefDelta # Don't update endpoints
                break

        # Newton steps are done. Now, check if we need to subdivide.
        # We do this by building new subdivided knots, tValues, and GSamples (x(tValues)).
        # If F(GSamples) is close enough to zero, we're done.
        # Otherwise, re-run Newton's method using the new knots, tValues, and GSamples.
        nUnknownCoefs = 2 * (nCoef - 2)
        tValues = np.empty(nUnknownCoefs, contourDtype)
        GSamples = np.empty((nUnknownCoefs, nDep), contourDtype)
        previousKnot = knots[degree]
        newKnots = [previousKnot] * order
        FSamplesNorm = 0.0
        i = 0
        for ix in range(order, len(knots) - degree, order - 2):
            knot = knots[ix]
            compactCoefs = coefs[ix - order:ix, :]

            # New knots are at the midpoint between old knots.
            newKnot = 0.5 * (previousKnot + knot)

            # Place tValues at Gauss points for the intervals [previousKnot, newKnot] and [newKnot, knot].
            for knotInterval in [[previousKnot, newKnot], [newKnot, knot]]:
                for gaussNode in gaussNodes:
                    tValues[i] = t = (1.0 - gaussNode) * knotInterval[0] + gaussNode * knotInterval[1]
                    x = compactCoefs.T @ bspy.Spline.bspline_values(ix, knots, order, t)[1]
                    x = np.array([max(0.0, min(1.0, xi)) for xi in x])
                    GSamples[i] = x
                    FSamplesNorm = max(FSamplesNorm, np.linalg.norm(F(coefsMin + x * coefsMaxMinusMin), np.inf))
                    i += 1
            
            newKnots += [newKnot] * (order - 2) # C1 continuity
            newKnots += [knot] * (order - 2) # C1 continuity
            previousKnot = knot
        
        # Test if F(GSamples) is close enough to zero.
        if FSamplesNorm < evaluationEpsilon:
            break # We're done! Exit subdivision loop and return x(t).

        # Otherwise, update nCoef and knots array, and then re-run Newton's method.
        nCoef = nUnknownCoefs + 2
        newKnots += [knot] * 2 # Clamp last knot
        knots = np.array(newKnots, contourDtype)
        assert i == nUnknownCoefs
        assert len(knots) == nCoef + order

    # Rescale x(t) back to original data points.
    coefs = (coefsMin + coefs * coefsMaxMinusMin).T
    spline = bspy.Spline(1, nDep, (order,), (nCoef,), (knots,), coefs, metadata)
    if isinstance(F, (bspy.Spline, bspy.SplineBlock)):
        spline = spline.confine(F.domain())
    return spline

def cylinder(radius, height, tolerance = None):
    if tolerance is None:
        tolerance = 1.0e-12
    bottom = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]] @ bspy.Spline.circular_arc(radius, 360.0, tolerance)
    top = bottom + [0.0, 0.0, height]
    return bspy.Spline.ruled_surface(bottom, top)

def fit(domain, f, order = None, knots = None, tolerance = 1.0e-4):
    # Determine number of independent variables
    domain = np.array(domain)
    nInd = len(domain)
    midPoint = f(0.5 * (domain.T[0] + domain.T[1]))
    if not type(midPoint) is bspy.Spline:
        nDep = len(midPoint)

    # Make sure order and knots conform to this
    if order is None:
        order = nInd * [4]
    if len(order) != nInd:
        raise ValueError("Inconsistent number of independent variables")

    # Establish the initial knot sequence
    if knots is None:
        knots = np.array([order[iInd] * [domain[iInd, 0]] + order[iInd] * [domain[iInd, 1]] for iInd in range(nInd)])
    
    # Determine initial nCoef
    nCoef = [len(knotVector) - iOrder for iOrder, knotVector in zip(order, knots)]

    # Define function to insert midpoints
    def addMidPoints(u):
        newArray = np.empty([2, len(u)])
        newArray[0] = u
        newArray[1, :-1] = 0.5 * (u[1:] + u[:-1])
        return newArray.T.flatten()[:-1]
    
    # Track the current spline space we're fitting in
    currentSpace = bspy.Spline(nInd, 0, order, nCoef, knots, [])

    # Generate the Greville points for these knots
    uvw = [currentSpace.greville(iInd) for iInd in range(nInd)]

    # Enrich the sample points
    for iInd in range(nInd):
        uvw[iInd][0] = knots[iInd][order[iInd] - 1]
        uvw[iInd][-1] = knots[iInd][nCoef[iInd]]
        for iLevel in range(1):
            uvw[iInd] = addMidPoints(uvw[iInd])

    # Initialize the dictionary of function values

    fDictionary = {}

    # Keep looping until done
    while True:

        # Evaluate the function on this data set
        fValues = []
        indices = nInd * [0]
        iLast = nInd
        while iLast >= 0:
            uValue = tuple([uvw[i][indices[i]] for i in range(nInd)])
            if not uValue in fDictionary:
                fDictionary[uValue] = f(uValue)
            fValues.append(fDictionary[uValue])
            iLast = nInd - 1
            while iLast >= 0:
                indices[iLast] += 1
                if indices[iLast] < len(uvw[iLast]):
                    break
                indices[iLast] = 0
                iLast -= 1

        # Adjust the ordering
        pointShape = [len(uvw[i]) for i in range(nInd)]
        if type(midPoint) is bspy.Spline:
            fValues = np.array(fValues).reshape(pointShape)
        else:
            fValues = np.array(fValues).reshape(pointShape + [nDep]).transpose([nInd] + list(range(nInd)))

        # Call the least squares fitter on this data
        bestSoFar = bspy.Spline.least_squares(uvw, fValues, order, currentSpace.knots, fixEnds = True)

        # Determine the maximum error
        maxError = 0.0
        for key in fDictionary:
            if type(midPoint) is bspy.Spline:
                sampled = bestSoFar.contract(midPoint.nInd * [None] + list(key)).coefs
                trueCoefs = fDictionary[key].coefs
                thisError = np.max(np.linalg.norm(sampled - trueCoefs, axis = 0))
            else:
                thisError = np.linalg.norm(fDictionary[key] - bestSoFar(key))
            if thisError > maxError:
                maxError = thisError
                maxKey = key
        if maxError <= tolerance:
            break

        # Split the interval and try again
        maxGap = 0.0
        for iInd in range(nInd):
            insert = bspy.Spline.bspline_values(None, currentSpace.knots[iInd], order[iInd], maxKey[iInd])[0]
            leftKnot = currentSpace.knots[iInd][insert - 1]
            rightKnot = currentSpace.knots[iInd][insert]
            if rightKnot - leftKnot > maxGap:
                maxGap = rightKnot - leftKnot
                iFirst = np.searchsorted(uvw[iInd], leftKnot, side = 'right')
                iLast = np.searchsorted(uvw[iInd], rightKnot, side = 'right')
                maxLeft = leftKnot
                maxRight = rightKnot
                maxInd = iInd
        splitAt = 0.5 * (maxLeft + maxRight)
        newKnots = [[] for iInd in range(nInd)]
        newKnots[maxInd] = [splitAt]
        currentSpace = currentSpace.insert_knots(newKnots)

        # Add samples for the new knot
        uvw[maxInd] = np.array(list(uvw[maxInd][:iFirst - 1]) +
                               list(addMidPoints(uvw[maxInd][iFirst - 1:iLast])) +
                               list(uvw[maxInd][iLast:]))

    # Return the best spline found so far
    return bestSoFar

def four_sided_patch(bottom, right, top, left, surfParam = 0.5):
    if bottom.nInd != 1 or right.nInd != 1 or top.nInd != 1 or left.nInd != 1:
        raise ValueError("Input curves must have one independent variable")
    if bottom.nDep != right.nDep or bottom.nDep != top.nDep or bottom.nDep != left.nDep:
        raise ValueError("Input curves must all have the same number of dependent variables")
    
    # Make sure all curves are parametrized over [0, 1]

    bottom = bottom.reparametrize([[0.0, 1.0]])
    right = right.reparametrize([[0.0, 1.0]])
    top = top.reparametrize([[0.0, 1.0]])
    left = left.reparametrize([[0.0, 1.0]])

    # Make sure all curves and ordered and lined up in the best possible way

    matchTo = bottom(1.0)
    checkPoints = [right(1.0), top(0.0), top(1.0), left(0.0), left(1.0), right(0.0)]
    checkDists = [np.linalg.norm(matchTo - pt) for pt in checkPoints]
    iMin = np.argmin(checkDists)
    if iMin == 1 or iMin == 2:
        swap = right
        right = top
        top = swap
    if iMin == 3 or iMin == 4:
        swap = right
        right = left
        left = swap
    if iMin % 2 == 0:
        right = right.reverse()
    matchTo = right(1.0)
    checkPoints = [top(0.0), left(0.0), left(1.0), top(1.0)]
    checkDists = [np.linalg.norm(matchTo - pt) for pt in checkPoints]
    iMin = np.argmin(checkDists)
    if iMin == 1 or iMin == 2:
        swap = top
        top = left
        left = swap
    if iMin < 2:
        top = top.reverse()
    if np.linalg.norm(bottom(0.0) - left(1.0)) + np.linalg.norm(top(0.0) - left(0.0)) < \
       np.linalg.norm(bottom(0.0) - left(0.0)) + np.linalg.norm(top(0.0) - left(1.0)):
        left = left.reverse()

    # Construct the Coons patch for these four curves
    
    bottomTop = bspy.Spline.ruled_surface(bottom, top)
    leftRight = bspy.Spline.ruled_surface(left, right)
    bottomLine = bspy.Spline.line(0.5 * (bottom(0.0) + left(0.0)), 0.5 * (bottom(1.0) + right(0.0)))
    topLine = bspy.Spline.line(0.5 * (top(0.0) + left(1.0)), 0.5 * (top(1.0) + right(1.0)))
    biLinear = bspy.Spline.ruled_surface(bottomLine, topLine)
    coons = bottomTop.add(leftRight, ((0,1), (1,0))) - biLinear

    # Determine the Greville abscissae to use as collocation points

    uPts = coons.greville(0)[1 : -1]
    vPts = coons.greville(1)[1 : -1]
    nu = len(uPts)
    nv = len(vPts)
    
    # Set up the matrix and right-hand-side vectors for the problem
    
    laplace = coons.copy()
    zeroSpline = (laplace.nDep * [0]) @ laplace
    def laplacian(spline, uv):
        return spline.derivative([2, 0], uv) + spline.derivative([0, 2], uv)
    lapMat = np.zeros((nu, nv, nu, nv))
    rhs = np.zeros((nu, nv, laplace.nDep))
    for iv1 in range(nv):
        for iu1 in range(nu):
            rhs[iu1, iv1, :] = -laplacian(laplace, [uPts[iu1], vPts[iv1]])
            for iv2 in range(nv):
                for iu2 in range(nu):
                    zeroSpline.coefs[0, iu2 + 1, iv2 + 1] = 1.0
                    lapMat[iu1, iv1, iu2, iv2] = laplacian(zeroSpline, [uPts[iu1], vPts[iv1]])[0]
                    zeroSpline.coefs[0, iu2 + 1, iv2 + 1] = 0.0
    
    # Solve the linear system
    
    lapMat = np.reshape(lapMat, (nu * nv, nu * nv))
    rhs = np.reshape(rhs, (nu * nv, laplace.nDep))
    interiorCoefs = np.linalg.solve(lapMat, rhs)
    interiorCoefs = np.reshape(interiorCoefs, (nu, nv, laplace.nDep))
    for ix in range(laplace.nDep):
        laplace.coefs[ix, 1:-1, 1:-1] += interiorCoefs[:, :, ix]
    
    # Return the proper weighted average of the surfaces
    
    return (1.0 - surfParam) * coons + surfParam * laplace

def geodesic(self, uvStart, uvEnd, tolerance = 1.0e-5):
    # Check validity of input
    if self.nInd != 2:  raise ValueError("Surface must have two independent variables")
    if len(uvStart) != 2:  raise ValueError("uvStart must have two components")
    if len(uvEnd) != 2:  raise ValueError("uvEnd must have two components")
    uvDomain = self.domain()
    if uvStart[0] < uvDomain[0, 0] or uvStart[0] > uvDomain[0, 1] or \
       uvStart[1] < uvDomain[1, 0] or uvStart[1] > uvDomain[1, 1]:
        raise ValueError("uvStart is outside domain of the surface")
    if uvEnd[0] < uvDomain[0, 0] or uvEnd[0] > uvDomain[0, 1] or \
       uvEnd[1] < uvDomain[1, 0] or uvEnd[1] > uvDomain[1, 1]:
        raise ValueError("uvEnd is outside domain of the surface")
    
    # Define the callback function for the ODE solver
    def geodesicCallback(t, u, surface, uvDomain):
        # Evaluate the surface information needed for the Christoffel symbols
        u[:, 0] = np.maximum(uvDomain[:, 0], np.minimum(uvDomain[:, 1], u[:, 0]))
        su = surface.derivative([1, 0], u[:, 0])
        sv = surface.derivative([0, 1], u[:, 0])
        suu = surface.derivative([2, 0], u[:, 0])
        suv = surface.derivative([1, 1], u[:, 0])
        svv = surface.derivative([0, 2], u[:, 0])
        suuu = surface.derivative([3, 0], u[:, 0])
        suuv = surface.derivative([2, 1], u[:, 0])
        suvv = surface.derivative([1, 2], u[:, 0])
        svvv = surface.derivative([0, 3], u[:, 0])

        # Calculate inner products
        su_su = su @ su
        su_sv = su @ sv
        sv_sv = sv @ sv
        suu_su = suu @ su
        suu_sv = suu @ sv
        suv_su = suv @ su
        suv_sv = suv @ sv
        svv_su = svv @ su
        svv_sv = svv @ sv
        suu_suu = suu @ suu
        suu_suv = suu @ suv
        suu_svv = suu @ svv
        suv_suv = suv @ suv
        suv_svv = suv @ svv
        svv_svv = svv @ svv
        suuu_su = suuu @ su
        suuu_sv = suuu @ sv
        suuv_su = suuv @ su
        suuv_sv = suuv @ sv
        suvv_su = suvv @ su
        suvv_sv = suvv @ sv
        svvv_su = svvv @ su
        svvv_sv = svvv @ sv

        # Calculate the first fundamental form and derivatives
        E = su_su
        E_u = 2.0 * suu_su
        E_v = 2.0 * suv_su
        F = su_sv
        F_u = suu_sv + suv_su
        F_v = suv_sv + svv_su
        G = sv_sv
        G_u = 2.0 * suv_sv
        G_v = 2.0 * svv_sv
        A = np.array([[E, F], [F, G]])
        A_u = np.array([[E_u, F_u], [F_u, G_u]])
        A_v = np.array([[E_v, F_v], [F_v, G_v]])

        # Compute right hand side entries
        R = np.array([[suu_su, suv_su, svv_su], [suu_sv, suv_sv, svv_sv]])
        R_u = np.array([[suuu_su + suu_suu, suuv_su + suu_suv, suvv_su + suu_svv],
                        [suuu_sv + suu_suv, suuv_sv + suv_suv, suvv_sv + suv_svv]])
        R_v = np.array([[suuv_su + suu_suv, suvv_su + suv_suv, svvv_su + suv_svv],
                        [suuv_sv + suu_svv, suvv_sv + suv_svv, svvv_sv + svv_svv]])

        # Solve for the Christoffel symbols
        luAndPivot = sp.linalg.lu_factor(A)
        Gamma = sp.linalg.lu_solve(luAndPivot, R)
        Gamma_u = sp.linalg.lu_solve(luAndPivot, R_u - A_u @ Gamma)
        Gamma_v = sp.linalg.lu_solve(luAndPivot, R_v - A_v @ Gamma)

        # Compute the right hand side for the ODE
        rhs = -np.array([Gamma[0, 0] * u[0, 1] ** 2 + 2.0 * Gamma[0, 1] * u[0, 1] * u[1, 1] + Gamma[0, 2] * u[1, 1] ** 2,
                         Gamma[1, 0] * u[0, 1] ** 2 + 2.0 * Gamma[1, 1] * u[0, 1] * u[1, 1] + Gamma[1, 2] * u[1, 1] ** 2])

        # Compute the Jacobian matrix of the right hand side of the ODE
        jacobian = -np.array([[[Gamma_u[0, 0] * u[0, 1] ** 2 + 2.0 * Gamma_u[0, 1] * u[0, 1] * u[1, 1] + Gamma_u[0, 2] * u[1, 1] ** 2,
                                2.0 * Gamma[0, 0] * u[0, 1] + 2.0 * Gamma[0, 1] * u[1, 1]],
                               [Gamma_v[0, 0] * u[0, 1] ** 2 + 2.0 * Gamma_v[0, 1] * u[0, 1] * u[1, 1] + Gamma_v[0, 2] * u[1, 1] ** 2,
                                2.0 * Gamma[0, 1] * u[0, 1] + 2.0 * Gamma[0, 2] * u[1, 1]]],
                              [[Gamma_u[1, 0] * u[0, 1] ** 2 + 2.0 * Gamma_u[1, 1] * u[0, 1] * u[1, 1] + Gamma_u[1, 2] * u[1, 1] ** 2,
                                2.0 * Gamma[1, 0] * u[0, 1] + 2.0 * Gamma[1, 1] * u[1, 1]],
                               [Gamma_v[1, 0] * u[0, 1] ** 2 + 2.0 * Gamma_v[1, 1] * u[0, 1] * u[1, 1] + Gamma_v[1, 2] * u[1, 1] ** 2,
                                2.0 * Gamma[1, 1] * u[1, 1] + 2.0 * Gamma[1, 2] * u[1, 1]]]])
        return rhs, jacobian

    # Generate the initial guess for the contour
    initialGuess = line(uvStart, uvEnd).elevate([2])

    # Solve the ODE and return the geodesic
    solution = initialGuess.solve_ode(1, 1, geodesicCallback, tolerance, (self, uvDomain))
    return solution

def least_squares(uValues, dataPoints, order = None, knots = None, compression = 0.0,
                  tolerance = None, fixEnds = False, metadata = {}):

    # Preprocess all the input if everything is a spline
    
    dataPoints = np.array(dataPoints)
    flatView = np.ravel(dataPoints)
    splineInput = False
    if type(flatView[0]) is bspy.Spline:
        splineInput = True
        nInd = flatView[0].nInd
        nDep = flatView[0].nDep
        splineDomain = flatView[0].domain()
        for spline in flatView:
            if spline.nInd != nInd:  raise ValueError("Input splines have different number of independent variables")
            if spline.nDep != nDep:  raise ValueError("Input splines have different number of dependent variables")
            if not np.array_equal(spline.domain(), splineDomain):  raise ValueError("Input splines have different domains")
        commonSplines = bspy.Spline.common_basis(flatView)
        foldedData = []
        for spline in commonSplines:
            folded, unfoldInfo = spline.fold(list(range(nInd)))
            foldedData.append(folded())
        foldedData = np.array(foldedData).T
        dataPoints = np.reshape(foldedData, (foldedData.shape[0],) + dataPoints.shape)

    # Preprocess the parameters values for the points

    if np.isscalar(uValues[0]):
        nInd = 1
        uValues = [uValues]
    else:
        nInd = len(uValues)
    domain = []
    for iInd in range(nInd):
        uMin = np.min(uValues[iInd])
        uMax = np.max(uValues[iInd])
        domain.append([uMin, uMax])
        for i in range(len(uValues[iInd]) - 1):
            if uValues[iInd][i] > uValues[iInd][i + 1]:  raise ValueError("Independent variable values are out of order")
    
    # Preprocess the data points

    if len(dataPoints.shape) != nInd + 1:  raise ValueError("dataPoints has the wrong shape")
    nDep = dataPoints.shape[0]
    pointsPerDirection = list(dataPoints.shape)[1:]
    nPoints = 1
    for i, nu in enumerate(pointsPerDirection):
        if nu != len(uValues[i]):  raise ValueError("Wrong number of parameter values in one or more directions")
        nPoints *= nu

    # Make sure the order makes sense

    if order is None:
        order = [min(4, nu) for nu in pointsPerDirection]
    for nP, nOrder in zip(pointsPerDirection, order):
        if nP < nOrder:  raise ValueError("Not enough points in one or more directions")

    # Determine the (initial) knots array

    if not(0.0 <= compression <= 1.0):  raise ValueError("compression not between 0.0 and 1.0")
    if tolerance is None:
        tolerance = np.finfo(float).max
    else:
        compression = 1.0
    if knots is None:
        knots = [np.array(order[iInd] * [domain[iInd][0]] + order[iInd] * [domain[iInd][1]]) for iInd in range(nInd)]
        for iInd, (nP, nOrder) in enumerate(zip(pointsPerDirection, order)):
            knotsToAdd = int((nP - nOrder) * (1.0 - compression) + 0.9999999999)
            addSpots = np.linspace(0.0, nP - 1.0, knotsToAdd + 2)[1 : -1]
            newKnots = []
            for newSpot in addSpots:
                ix = int(newSpot)
                alpha = newSpot - ix
                newKnots.append((1.0 - alpha) * uValues[iInd][ix] + alpha * uValues[iInd][ix + 1])
            knots[iInd] = np.sort(np.append(knots[iInd], newKnots))
    else:
        knots = tuple(np.array(kk) for kk in knots)
    for iInd in range(nInd):
        if domain[iInd][0] < knots[iInd][order[iInd] - 1] or \
           domain[iInd][1] > knots[iInd][-order[iInd]]:  raise ValueError("One or more dataPoints are outside the domain of the spline")

    # Loop through each independent variable and fit all of the data
    
    for iInd in range(nInd):
        nRows = pointsPerDirection[iInd]
        b = np.swapaxes(dataPoints, 0, iInd + 1)
        loopDep = nDep * nPoints // nRows
        b = np.reshape(b, (nRows, loopDep))
        done = False
        while not done:
            nCols = len(knots[iInd]) - order[iInd]
            A = np.zeros((nRows, nCols))
            u = -np.finfo(float).max
            fixedRows = []
            for iRow in range(nRows):
                uNew = uValues[iInd][iRow]
                if uNew != u:
                    iDerivative = 0
                    u = uNew
                    ix = None
                else:
                    iDerivative += 1
                ix, row = bspy.Spline.bspline_values(ix, knots[iInd], order[iInd], u, iDerivative)
                A[iRow, ix - order[iInd] : ix] = row
                if fixEnds and (u == uValues[iInd][0] or u == uValues[iInd][-1]):
                    fixedRows.append(iRow)
            nInterpolationConditions = len(fixedRows)
            if nInterpolationConditions != 0:
                C = np.take(A, fixedRows, 0)
                d = np.take(b, fixedRows, 0)
                AUse = np.delete(A, fixedRows, 0)
                bUse = np.delete(b, fixedRows, 0)
                U, Sigma, VT = np.linalg.svd(C)
                d = U.T @ d
                for iRow in range(nInterpolationConditions):
                    d[iRow] = d[iRow] / Sigma[iRow]
                rangeCols = np.take(VT.T, range(nInterpolationConditions), 1)
                nullspace = np.delete(VT.T, range(nInterpolationConditions), 1)
                x1 = rangeCols @ d
                b1 = bUse - AUse @ x1
                xNullspace, residuals, rank, s = np.linalg.lstsq(AUse @ nullspace, b1, rcond = None)
                x = x1 + nullspace @ xNullspace
            else:
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond = None)
            residuals = b - A @ x
            maxError = 0.0
            for iRow in range(nRows):
                rowError = np.linalg.norm(residuals[iRow, :])
                if rowError > maxError:
                    maxError = rowError
                    maxRow = iRow
            if maxError <= tolerance / nInd:
                done = True
            else:
                ix = np.searchsorted(knots[iInd], uValues[iInd][maxRow], 'right')
                ix = min(ix, nCols)
                knots[iInd] = np.sort(np.append(knots[iInd], 0.5 * (knots[iInd][ix - 1] + knots[iInd][ix])))
        pointsPerDirection[iInd] = nDep
        x = np.reshape(x, [nCols] + pointsPerDirection)
        dataPoints = np.swapaxes(x, 0, iInd + 1)
        pointsPerDirection[iInd] = nCols
        nPoints = nCols * nPoints // nRows
    splineFit = bspy.Spline(nInd, nDep, order, pointsPerDirection, knots, dataPoints, metadata = metadata)
    if splineInput:
        splineFit = splineFit.unfold(range(unfoldInfo.nInd), unfoldInfo)
    return splineFit

def line(startPoint, endPoint):
    startPoint = bspy.Spline.point(startPoint)
    endPoint = bspy.Spline.point(endPoint)
    return bspy.Spline.ruled_surface(startPoint, endPoint)

def line_of_curvature(self, uvStart, is_max, tolerance = 1.0e-3):
    if self.nInd != 2:  raise ValueError("Surface must have two independent variables")
    if len(uvStart) != 2:  raise ValueError("uvStart must have two components")
    uvDomain = self.domain()
    if uvStart[0] < uvDomain[0, 0] or uvStart[0] > uvDomain[0, 1] or \
       uvStart[1] < uvDomain[1, 0] or uvStart[1] > uvDomain[1, 1]:
        raise ValueError("uvStart is outside domain of the surface")
    is_max = bool(is_max) # Ensure is_max is a boolean for XNOR operation
    
    # Define the callback function for the ODE solver
    def curvatureLineCallback(t, u):
        # Evaluate the surface information needed.
        uv = np.maximum(uvDomain[:, 0], np.minimum(uvDomain[:, 1], u[:, 0]))
        su = self.derivative((1, 0), uv)
        sv = self.derivative((0, 1), uv)
        suu = self.derivative((2, 0), uv)
        suv = self.derivative((1, 1), uv)
        svv = self.derivative((0, 2), uv)
        suuu = self.derivative((3, 0), uv)
        suuv = self.derivative((2, 1), uv)
        suvv = self.derivative((1, 2), uv)
        svvv = self.derivative((0, 3), uv)
        normal = self.normal(uv)

        # Calculate curvature matrix and its derivatives.
        sU = np.concatenate((su, sv)).reshape(2, -1)
        sUu = np.concatenate((suu, suv)).reshape(2, -1)
        sUv = np.concatenate((suv, svv)).reshape(2, -1)
        sUU = np.concatenate((suu, suv, suv, svv)).reshape(2, 2, -1)
        sUUu = np.concatenate((suuu, suuv, suuv, suvv)).reshape(2, 2, -1)
        sUUv = np.concatenate((suuv, suvv, suvv, svvv)).reshape(2, 2, -1)
        fffI = np.linalg.inv(sU @ sU.T) # Inverse of first fundamental form
        k = fffI @ (sUU @ normal) # Curvature matrix
        ku = fffI @ (sUUu @ normal - (sUu @ sU.T + sU @ sUu.T) @ k - sUU @ (sU.T @ k[:, 0]))
        kv = fffI @ (sUUv @ normal - (sUv @ sU.T + sU @ sUv.T) @ k - sUU @ (sU.T @ k[:, 1]))

        # Determine principle curvatures and directions, and assign new direction.
        curvatures, directions = np.linalg.eig(k)
        curvatureDelta = curvatures[1] - curvatures[0]
        if abs(curvatureDelta) < tolerance:
            # If we're at an umbilic, use the last direction (jacobian is zero at umbilic).
            direction = u[:, 1]
            jacobian = np.zeros((2,2,1), self.coefs.dtype)
        else:
            # Otherwise, compute the lhs inverse for the jacobian.
            directionsInverse = np.linalg.inv(directions)
            eigenIndex = 0 if bool(curvatures[0] > curvatures[1]) == is_max else 1
            direction = directions[:, eigenIndex]
            B = np.zeros((2, 2), self.coefs.dtype)
            B[0, 1 - eigenIndex] = np.dot(directions[:, 1], direction) / curvatureDelta
            B[1, 1 - eigenIndex] = -np.dot(directions[:, 0], direction) / curvatureDelta
            lhsInv =  directions @ B @ directionsInverse

            # Adjust the direction for consistency.
            if np.dot(direction, u[:, 1]) < -tolerance:
                direction *= -1

            # Compute the jacobian for the direction.
            jacobian = np.empty((2,2,1), self.coefs.dtype)
            jacobian[:,0,0] = lhsInv @ ku @ direction
            jacobian[:,1,0] = lhsInv @ kv @ direction

        return direction, jacobian

    # Generate the initial guess for the line of curvature.
    uvStart = np.atleast_1d(uvStart)
    direction = 0.5 * (uvDomain[:,0] + uvDomain[:,1]) - uvStart # Initial guess toward center
    distanceFromCenter = np.linalg.norm(direction)
    if distanceFromCenter < 10 * tolerance:
        # If we're at the center, just point to the far corner.
        direction = np.array((1.0, 1.0)) / np.sqrt(2)
    else:
        direction /= distanceFromCenter

    # Compute line of curvature direction at start.
    direction, jacobian = curvatureLineCallback(0.0, np.array(((uvStart[0], direction[0]), (uvStart[1], direction[1]))))

    # Calculate distance to the boundary in that direction.
    if direction[0] < -tolerance:
        uBoundaryDistance = (uvDomain[0, 0] - uvStart[0]) / direction[0]
    elif direction[0] > tolerance:
        uBoundaryDistance = (uvDomain[0, 1] - uvStart[0]) / direction[0]
    else:
        uBoundaryDistance = np.inf
    if direction[1] < -tolerance:
        vBoundaryDistance = (uvDomain[1, 0] - uvStart[1]) / direction[1]
    elif direction[1] > tolerance:
        vBoundaryDistance = (uvDomain[1, 1] - uvStart[1]) / direction[1]
    else:
        vBoundaryDistance = np.inf
    boundaryDistance = min(uBoundaryDistance, vBoundaryDistance)

    # Construct the initial guess from start point to boundary.
    initialGuess = line(uvStart, uvStart + boundaryDistance * direction).elevate([2])

    # Solve the ODE and return the line of curvature confined to the surface's domain.
    solution = initialGuess.solve_ode(1, 0, curvatureLineCallback, tolerance, includeEstimate = True)
    return solution.confine(uvDomain)

def offset(self, edgeRadius, bitRadius=None, angle=np.pi / 2.2, subtract=False, removeCusps=False, tolerance = 1.0e-4):
    if self.nDep < 2 or self.nDep > 3 or self.nDep - self.nInd != 1: raise ValueError("The offset is only defined for 2D curves and 3D surfaces with well-defined normals.")
    if edgeRadius < 0:
        raise ValueError("edgeRadius must be >= 0")
    elif edgeRadius == 0:
        return self
    if bitRadius is None:
        bitRadius = edgeRadius
    elif bitRadius < edgeRadius:
        raise ValueError("bitRadius must be >= edgeRadius")
    if angle < 0 or angle >= np.pi / 2: raise ValueError("angle must in the range [0, pi/2)")

    # Determine geometry of drill bit.
    if subtract:
        edgeRadius *= -1
        bitRadius *= -1
    w = bitRadius - edgeRadius
    h = w * np.tan(angle)
    bottom = np.sin(angle)
    bottomRadius = edgeRadius + h / bottom

    # Define drill bit function.
    if abs(w) < tolerance:
        def drillBit(uv):
            return self(uv) + edgeRadius * self.normal(uv)
    elif self.nDep == 2:
        def drillBit(u):
            xy = self(u)
            normal = self.normal(u)
            upward = np.sign(normal[1])
            if upward * normal[1] <= bottom:
                xy[0] += edgeRadius * normal[0] + w * np.sign(normal[0])
                xy[1] += edgeRadius * normal[1]
            else:
                xy[0] += bottomRadius * normal[0]
                xy[1] += bottomRadius * normal[1] - upward * h
            return xy
    elif self.nDep == 3:
        def drillBit(uv):
            xyz = self(uv)
            normal = self.normal(uv)
            upward = np.sign(normal[1])
            if upward * normal[1] <= bottom:
                norm = np.sqrt(normal[0] * normal[0] + normal[2] * normal[2])
                xyz[0] += edgeRadius * normal[0] + w * normal[0] / norm
                xyz[1] += edgeRadius * normal[1]
                xyz[2] += edgeRadius * normal[2] + w * normal[2] / norm
            else:
                xyz[0] += bottomRadius * normal[0]
                xyz[1] += bottomRadius * normal[1] - upward * h
                xyz[2] += bottomRadius * normal[2]
            return xyz

    # Compute new order and knots for offset (ensure order is at least 4).
    newOrder = []
    newKnots = []
    for order, knots in zip(self.order, self.knots):
        min4Order = max(order, 4)
        unique, count = np.unique(knots, return_counts=True)
        count += min4Order - order
        newOrder.append(min4Order)
        newKnots.append(np.repeat(unique, count))

    # Fit new spline to offset by drill bit.
    offset = fit(self.domain(), drillBit, newOrder, newKnots, tolerance)

    # Remove cusps as required (only applies to offset curves).
    if removeCusps and self.nInd == 1:
        # Find the cusps by checking for tangent direction reversal between the spline and offset.
        cusps = []
        previousKnot = None
        start = None
        for knot in offset.knots[0][offset.order[0]:offset.nCoef[0]]:
            flipped = np.dot(self.derivative((1,), knot), offset.derivative((1,), knot)) < 0
            if flipped and start is None:
                start = knot
            if not flipped and start is not None:
                cusps.append((start, previousKnot))
                start = None
            previousKnot = knot

        # Remove the cusps by intersecting the offset segments before and after each cusp.
        segmentList = []
        for cusp in cusps:
            domain = offset.domain()
            block = bspy.spline_block.SplineBlock([[offset.trim(((domain[0][0], cusp[0]),)), -offset.trim(((cusp[1], domain[0][1]),))]])
            intersections = block.zeros()
            for intersection in intersections:
                segmentList.append(offset.trim(((domain[0][0], intersection[0]),)))
                offset = offset.trim(((intersection[1], domain[0][1]),))
        segmentList.append(offset)
        offset = bspy.spline.Spline.join(segmentList)
    
    return offset

def point(point):
    point = np.atleast_1d(point)
    return bspy.Spline(0, len(point), [], [], [], point)

def revolve(self, angle):
    if self.nDep != 2: raise ValueError("Spline must have 2 dependent variables")

    maxRadius = max(abs(self.coefs[0].min()), self.coefs[0].max())
    arc = ((1.0 / maxRadius, 0.0),
            (0.0, 1.0 / maxRadius),
            (0.0, 0.0)) @ bspy.Spline.circular_arc(maxRadius, angle) + (0.0, 0.0, 1.0)
    radiusHeight = ((1.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0)) @ self
    return arc.multiply(radiusHeight)

def ruled_surface(curve1, curve2):
    # Ensure that the splines are compatible
    if curve1.nInd != curve2.nInd:  raise ValueError("Splines must have the same number of independent variables")
    if curve1.nDep != curve2.nDep:  raise ValueError("Splines must have the same number of dependent variables")
    [newCurve1, newCurve2] = bspy.Spline.common_basis([curve1, curve2])

    # Generate the ruled spline between them
    myCoefs1 = np.reshape(newCurve1.coefs, newCurve1.coefs.shape + (1,))
    myCoefs2 = np.reshape(newCurve2.coefs, newCurve2.coefs.shape + (1,))
    newCoefs = np.append(myCoefs1, myCoefs2, newCurve1.nInd + 1)
    return bspy.Spline(curve1.nInd + 1, curve1.nDep, list(newCurve1.order) + [2],
                       list(newCurve1.nCoef) + [2], list(newCurve1.knots) + [[0.0, 0.0, 1.0, 1.0]],
                       newCoefs)

def section(xytk):    
    def twoPointSection(startPointX, startPointY, startAngle, startKappa, endPointX, endPointY, endAngle, endKappa):
        # Check validity of curvatures
        if startKappa * endKappa < 0.0:  raise ValueError("startKappa and endKappa have different signs")

        # Compute unit tangent directions
        startRadians = math.pi * startAngle / 180.0
        startTangent = np.array([math.cos(startRadians), math.sin(startRadians)])
        endRadians = math.pi * endAngle / 180.0
        endTangent = np.array([math.cos(endRadians), math.sin(endRadians)])
        startPoint = np.array([startPointX, startPointY])
        endPoint = np.array([endPointX, endPointY])
        pointDiff = endPoint - startPoint
        crossTangents = startTangent[0] * endTangent[1] - startTangent[1] * endTangent[0]
        dotTangents = startTangent @ endTangent
        theta = math.atan2(crossTangents, dotTangents)

        # Make sure angle is less than 180 degrees
        if theta * startKappa < 0.0 or theta * endKappa < 0.0 or abs(theta) == math.pi:
            raise ValueError("Angle >= 180 degrees for two point section")
        
        # Check data consistency
        crossCheck = startTangent[0] * pointDiff[1] - startTangent[1] * pointDiff[0]
        if crossCheck * startKappa < 0.0 or crossCheck * endKappa < 0.0:  raise ValueError("Inconsistent start angle")
        if crossTangents * startKappa < 0.0 or crossTangents * endKappa < 0.0:  raise ValueError("Inconsistent tangent directions")
        crossCheck = pointDiff[0] * endTangent[1] - pointDiff[1] * endTangent[0]
        if crossCheck * startKappa < 0.0 or crossCheck * endKappa < 0.0:  raise ValueError("Inconsistent end angle")
 
        # Compute intersection point of tangent directions
        tangentDistances = np.linalg.solve(np.array([startTangent, endTangent]).T, pointDiff)
        frustum = startPoint + tangentDistances[0] * startTangent

        # Compute critical values for section algorithm
        onePlusCosTheta = 1.0 + math.cos(theta)
        r0 = 4.0 * startKappa * tangentDistances[0] ** 2 / (3.0 * tangentDistances[1] * crossTangents)
        r1 = 4.0 * endKappa * tangentDistances[1] ** 2 / (3.0 * tangentDistances[0] * crossTangents)
        if r0 != 0.0 or r1 != 0.0:
            rhoCrit = (math.sqrt(1.0 + 4.0 * (r0 + r1)) - 1.0) / (2.0 * (r0 + r1))
        else:
            rhoCrit = 1.0
        rhoCritOfTheta = 3.0 * (math.sqrt(1.0 + 32.0 / (3.0 * onePlusCosTheta)) - 1.0) * onePlusCosTheta / 16.0

        # Determine quadratic polynomial
        sqrtTerm = math.sqrt(2.0 * onePlusCosTheta)
        a = sqrtTerm + 8.0 / onePlusCosTheta
        b = -3.0 * sqrtTerm
        c = 9.0 * sqrtTerm / 4.0 - 3.0
        root1 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2.0 * a)
        root2 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2.0 * a)
        rhoOpt = max(root1, root2)

        # Determine optimal value of rho and values of alpha0 and alpha1
        rho = rhoCrit * rhoOpt / rhoCritOfTheta
        alpha0 = r1 * rho ** 2 / (1.0 - rho)
        alpha1 = r0 * rho ** 2 / (1.0 - rho)

        # Generate the quartic section which interpolates the data
        pt0 = startPoint
        pt1 = (1.0 - rho) * startPoint + rho * frustum
        pt3 = (1.0 - rho) * endPoint + rho * frustum
        pt4 = endPoint
        pt2 = alpha0 * pt1 + alpha1 * pt3 + (1.0 - alpha0 - alpha1) * frustum
        return bspy.Spline(1, 2, (5,), (5,), ((0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),), (pt0, pt1, pt2, pt3, pt4))

    # Check that the input data is the right size and shape
    if len(xytk) < 2:  raise ValueError("Must specify at least two points")
    for pt in xytk:
        if len(pt) != 4:  raise ValueError("Each point should be a 4-tuple:  (x, y, theta, kappa)")
    
    # Loop through each pair of points building a section
    mySections = []
    for i in range(len(xytk) - 1):
        mySections.append(twoPointSection(*xytk[i], *xytk[i + 1]))
    
    # Join the pieces together and return
    return bspy.Spline.join(mySections)

def solve_ode(self, nLeft, nRight, FAndF_u, tolerance = 1.0e-6, args = (), includeEstimate = False):
    # Ensure that the ODE is properly formulated

    if nLeft < 0:  raise ValueError("Invalid number of left hand boundary conditions")
    if nRight < 0:  raise ValueError("Invalid number of right hand boundary conditions")
    if self.nInd != 1:  raise ValueError("Initial guess must have exactly one independent variable")
    nOrder = nLeft + nRight

    # Make sure that there are full multiplicity knots on the ends

    currentGuess = self.copy().clamp((0,), (0,))
    scale = 1.0
    for lower, upper in currentGuess.range_bounds():
        scale = max(scale, upper - lower)
    nDep = currentGuess.nDep

    # Insert and remove knots so initial guess conforms to de Boor - Swartz setup

    uniqueKnots, indices = np.unique(currentGuess.knots[0], True)
    uniqueKnots = uniqueKnots[1 : -1]
    indices = indices[1:]
    knotsToAdd = []
    knotsToRemove = []
    for i, knot in enumerate(uniqueKnots):
        howMany = currentGuess.order[0] - indices[i + 1] + indices[i] - nOrder
        if howMany > 0:
            knotsToAdd.append((knot, howMany))
        if howMany < 0:
            knotsToRemove.append((knot, abs(howMany)))
    currentGuess = currentGuess.insert_knots([knotsToAdd])
    for (knot, howMany) in knotsToRemove:
        ix = np.searchsorted(currentGuess.knots[0], knot, side = 'left')
        for iy in range(howMany):
            currentGuess, residual = currentGuess.remove_knot(ix, nLeft, nRight)
    previousGuess = 0.0 * currentGuess
    bandWidth = nDep * currentGuess.order[0] - 1

    # Determine whether an initial value problem or boundary value problem

    IVP = nLeft == 0 or nRight == 0

    # Use the Gauss Legendre points as the collocation points

    perInterval = currentGuess.order[0] - nOrder
    gaussNodes, weights = _legendre_polynomial_zeros(perInterval)
    linear = False
    refine = True
    while refine:
        collocationPoints = []
        for knot0, knot1 in zip(currentGuess.knots[0][:-1], currentGuess.knots[0][1:]):
            if knot0 < knot1:
                for gaussNode in gaussNodes:
                    collocationPoints.append((1.0 - gaussNode) * knot0 + gaussNode * knot1)
        n = nDep * currentGuess.nCoef[0]
        bestGuess = np.reshape(currentGuess.coefs.T, (n,))

    # Set up for next pass at this refinement level

        nCollocation = len(collocationPoints)
        if IVP:
            n = nDep * currentGuess.order[0]
            if nLeft != 0:
                iFirstPoint = 0
                iNextPoint = perInterval
            else:
                iNextPoint = nCollocation
                iFirstPoint = iNextPoint - perInterval
        else:
            iFirstPoint = 0
            iNextPoint = nCollocation

    # Perform the loop through all the IVP intervals

        while True:
            done = linear
            continuation = 1.0
            bestContinuation = 0.0
            inCaseOfEmergency = bestGuess.copy()
            previous = 0.5 * np.finfo(bestGuess[0]).max
            iteration = 0
    
    # Perform nonlinear Newton iteration

            while True:
                iteration += 1
                collocationMatrix = np.zeros((2 * bandWidth + 1, n))
                residuals = np.array([])
                workingSpline = bspy.Spline(1, nDep, currentGuess.order, currentGuess.nCoef,
                                            currentGuess.knots, np.reshape(bestGuess, (currentGuess.nCoef[0], nDep)).T)
                residuals = np.append(residuals, np.zeros((nLeft * nDep,)))
                collocationMatrix[bandWidth, 0 : nLeft * nDep] = 1.0
                for iPoint, t in enumerate(collocationPoints[iFirstPoint : iNextPoint]):
                    uData = np.array([workingSpline.derivative([i], t) for i in range(nOrder + 1 if includeEstimate else nOrder)]).T
                    F, F_u = FAndF_u(t, uData, *args)
                    residuals = np.append(residuals, workingSpline.derivative([nOrder], t) - continuation * F)
                    ix = None
                    bValues = np.array([])
                    for iDerivative in range(nOrder + 1):
                        ix, iValues = bspy.Spline.bspline_values(ix, workingSpline.knots[0], workingSpline.order[0],
                                                                 t, derivativeOrder = iDerivative)
                        bValues = np.append(bValues, iValues)
                    bValues = np.reshape(bValues, (nOrder + 1, workingSpline.order[0]))
                    for iDep in range(nDep):
                        iRow = (nLeft + iPoint) * nDep + iDep
                        startSlice = nDep * (ix - workingSpline.order[0] - iFirstPoint)
                        endSlice = nDep * (ix - iFirstPoint)
                        indices = np.arange(startSlice + iDep, endSlice + iDep, nDep)
                        collocationMatrix[bandWidth + iRow - indices, indices] -= bValues[nOrder]
                        for iF_uRow in range(nDep):
                            indices = np.arange(startSlice + iF_uRow, endSlice + iF_uRow, nDep)
                            for iF_uColumn in range(nOrder):
                                collocationMatrix[bandWidth + iRow - indices, indices] += continuation * F_u[iDep, iF_uRow, iF_uColumn] * bValues[iF_uColumn]
                residuals = np.append(residuals, np.zeros((nRight * nDep,)))
                collocationMatrix[bandWidth, -1 : -(nRight * nDep + 1) : -1] = 1.0

    # Solve the collocation linear system

                update = sp.linalg.solve_banded((bandWidth, bandWidth), collocationMatrix, residuals)
                bestGuess[nDep * (iFirstPoint + nLeft) : nDep * (iNextPoint + nLeft)] += update[nDep * nLeft : nDep * (iNextPoint - iFirstPoint + nLeft)]
                updateSize = np.linalg.norm(update)
                if updateSize > 1.25 * previous and iteration >= 4 or \
                   updateSize > 0.01 and iteration > 50:
                    continuation = 0.5 * (continuation + bestContinuation)
                    bestGuess = inCaseOfEmergency.copy()
                    if continuation - bestContinuation < 0.01:
                        break
                    previous = 0.5 * np.finfo(bestGuess[0]).max
                    iteration = 0
                    continue
                previous = updateSize
                if done or iteration > 50:
                    break

    # Check to see if we're almost done

                if updateSize < math.sqrt(n) * scale * math.sqrt(np.finfo(update.dtype).eps):
                    if continuation < 1.0:
                        bestContinuation = continuation
                        inCaseOfEmergency = bestGuess.copy()
                        continuation = min(1.0, 1.2 * continuation)
                        previous = 0.5 * np.finfo(bestGuess[0]).max
                        iteration = 0
                    else:
                        done = True
    
    # Check to see if this is a linear problem

                if not linear and iteration == 2 and updateSize < 100.0 * scale * np.finfo(update[0]).eps:
                    linear = True
                    done = True

    # Set up for one more pass through an IVP

            if IVP:
                if nLeft != 0:
                    iFirstPoint = iNextPoint
                    iNextPoint += perInterval
                    if iFirstPoint < nCollocation:
                        continue
                else:
                    iNextPoint = iFirstPoint
                    iFirstPoint -= perInterval
                    if iFirstPoint >= 0:
                        continue
            break;

    # Is it time to give up?

        if (not done or continuation < 1.0) and n > 1000:
            raise RuntimeError("Can't find solution with given initial guess")

    # Estimate the error

        currentGuess = bspy.Spline(1, nDep, currentGuess.order, currentGuess.nCoef, currentGuess.knots,
                                   np.reshape(bestGuess, (currentGuess.nCoef[0], nDep)).T)
        errorRange = (previousGuess - currentGuess).range_bounds()
        refine = False
        for lower, upper in errorRange:
            if upper - lower > 2.0 * scale * tolerance:
                refine = True

    # Insert new knots if refinement is needed

        if refine:
            knotsToAdd = []
            for knot0, knot1 in zip(currentGuess.knots[0][:-1], currentGuess.knots[0][1:]):
                if knot0 < knot1:
                    knotsToAdd.append((0.5 * (knot0 + knot1), currentGuess.order[0] - nOrder))
            previousGuess = currentGuess
            currentGuess = currentGuess.insert_knots([knotsToAdd])

    # Simplify the result and return

    currentGuess = currentGuess.remove_knots(0.1 * scale * tolerance, nLeft, nRight)
    return currentGuess
    
def sphere(radius, tolerance = None):
    if radius <= 0.0:  raise ValueError("Radius must be positive")
    if tolerance == None:
        tolerance = 1.0e-12
    phiCirc = bspy.Spline.circular_arc(radius, 180.0, 0.5 * tolerance)
    thetaCirc = bspy.Spline.circular_arc(1.0, 360.0, 0.5 * tolerance / radius)
    phi3D = [[0, 1], [0, 1], [-1, 0]] @ phiCirc
    theta3D = [[1, 0], [0, 1], [0, 0]] @ thetaCirc + [0, 0, 1]
    return phi3D.multiply(theta3D)

def torus(innerRadius, outerRadius, tolerance = None):
    if innerRadius < 0.0:  raise ValueError("Inner radius must be positive")
    if outerRadius <= innerRadius:  raise ValueError("Outer radius must be larger than inner radius")
    if tolerance == None:
        tolerance = 1.0e-12
    bigRadius = 0.5 * (innerRadius + outerRadius)
    donutRadius = 0.5 * (outerRadius - innerRadius)
    bigCirc = bspy.Spline.circular_arc(bigRadius, 360.0, 0.25 * tolerance)
    donutCirc = bspy.Spline.circular_arc(donutRadius, 360.0, 0.25 * tolerance)
    bigCirc3D = [[1, 0], [0, 1], [0, 0]] @ bigCirc
    donutCircCos = [[1, 0]] @ donutCirc
    donutCirc3D = [[0, 0], [0, 0], [0, 1]] @ donutCirc
    torus = bigCirc3D.multiply(donutCircCos) / bigRadius
    torus = torus.add(bigCirc3D, [0]).add(donutCirc3D, [(1, 0)])
    return torus
