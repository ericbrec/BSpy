import numpy as np
import bspy.spline
import math

def circular_arc(radius, angle, tolerance = None):
    if tolerance is None:
        tolerance = 1.0e-12
    if radius < 0.0 or angle < 0.0 or tolerance < 0.0: raise ValueError("The radius, angle, and tolerance must be positive.")

    samples = int(max(np.ceil(((1.1536e-5 * radius / tolerance)**(1/8)) * angle / 90), 2.0)) + 1
    return bspy.Spline.section([(radius * np.cos(u * angle * np.pi / 180), radius * np.sin(u * angle * np.pi / 180), 90 + u * angle, 1.0 / radius) for u in np.linspace(0.0, 1.0, samples)])

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
def _legendre_polynomial_zeros(degree):
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
        zval -= 0.001
    if degree % 2 == 1:
        z.append(0.0)
    z.reverse()
    w = []
    for zval in z:
        p, pd = legendre(degree, zval)
        w.append(2.0 / ((1.0 - zval ** 2) * pd[-1] ** 2))
    return z, w

def contour(F, knownXValues, dF = None, epsilon = None, metadata = {}):
    # Set up parameters for initial guess of x(t) and validate arguments.
    order = 4
    degree = order - 1
    rhos, gaussWeights = _legendre_polynomial_zeros(degree - 1)
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
    coefsMin = knownXValues.min(axis=0)
    coefsMaxMinusMin = knownXValues.max(axis=0) - coefsMin
    coefsMaxMinusMin = np.where(coefsMaxMinusMin < 1.0, 1.0, coefsMaxMinusMin)
    coefsMaxMinMinReciprocal = np.reciprocal(coefsMaxMinusMin)
    knownXValues = (knownXValues - coefsMin) * coefsMaxMinMinReciprocal # Rescale to [0 , 1]

    # Establish the first derivatives of F.
    if dF is None:
        dF = []
        if isinstance(F, bspy.Spline):
            for i in range(nDep):
                def splineDerivative(x, i=i):
                    wrt = [0] * nDep
                    wrt[i] = 1
                    return F.derivative(wrt, x)
                dF.append(splineDerivative)
        else:
            for i in range(nDep):
                def fDerivative(x, i=i):
                    h = epsilon * (1.0 + abs(x[i]))
                    xShift = np.array(x, copy=True)
                    xShift[i] -= h
                    fLeft = np.array(F(xShift))
                    h2 = h * 2.0
                    xShift[i] += h2
                    return (np.array(F(xShift)) - fLeft) / h2
                dF.append(fDerivative)
    else:
        if not(len(dF) == nDep): raise ValueError(f"Must provide {nDep} first derivatives.")

    # Construct knots, t values, and GSamples.
    tValues = np.empty(nUnknownCoefs, contourDtype)
    GSamples = np.empty((nUnknownCoefs, nDep), contourDtype)
    t = 0.0 # We start with t measuring contour length.
    knots = [t] * order
    i = 0
    previousPoint = knownXValues[0]
    for point in knownXValues[1:]:
        dt = np.linalg.norm(point - previousPoint)
        if not(dt > epsilon): raise ValueError("Points must be separated by at least epsilon.")
        for rho in reversed(rhos):
            tValues[i] = t + 0.5 * dt * (1.0 - rho)
            GSamples[i] = 0.5 * (previousPoint + point - rho * (point - previousPoint))
            i += 1
        for rho in rhos[0 if degree % 2 == 1 else 1:]:
            tValues[i] = t + 0.5 * dt * (1.0 + rho)
            GSamples[i] = 0.5 * (previousPoint + point + rho * (point - previousPoint))
            i += 1
        t += dt
        knots += [t] * (order - 2)
        previousPoint = point
    knots += [t] * 2 # Clamp last knot
    knots = np.array(knots, contourDtype) / t # Rescale knots
    tValues /= t # Rescale t values
    assert i == nUnknownCoefs
    
    # Start subdivision loop.
    while True:
        # Define G(coefs) to be dGCoefs @ coefs - GSamples,
        # where dGCoefs and GSamples are the B-spline values and sample points, respectively, for x(t).
        # The dGCoefs matrix is banded due to B-spline local support, so initialize it to zero.
        # Solving for coefs provides us our initial coefficients of x(t).
        dGCoefs = np.zeros((nUnknownCoefs, nDep, nDep, nCoef), contourDtype)
        i = 0
        for t, i in zip(tValues, range(nUnknownCoefs)):
            ix = np.searchsorted(knots, t, 'right')
            ix = min(ix, nCoef)
            bValues = bspy.Spline.bspline_values(ix, knots, order, t)
            for j in range(nDep):
                dGCoefs[i, j, j, ix - order:ix] = bValues
        GSamples -= dGCoefs[:, :, :, 0] @ knownXValues[0] + dGCoefs[:, :, :, -1] @ knownXValues[-1]
        GSamples = GSamples.reshape(nUnknownCoefs * nDep)
        dGCoefs = dGCoefs[:, :, :, 1:-1].reshape(nUnknownCoefs * nDep, nDep * nUnknownCoefs)
        coefs = np.empty((nDep, nCoef), contourDtype)
        coefs[:, 0] = knownXValues[0]
        coefs[:, -1] = knownXValues[-1]
        coefs[:, 1:-1] = np.linalg.solve(dGCoefs, GSamples).reshape(nDep, nUnknownCoefs)

        # Array to hold the values of F and contour dot for each t, excluding endpoints.
        FSamples = np.empty((nUnknownCoefs, nDep), contourDtype)
        # Array to hold the Jacobian of the FSamples with respect to the coefficients.
        # The Jacobian is banded due to B-spline local support, so initialize it to zero.
        dFCoefs = np.zeros((nUnknownCoefs, nDep, nDep, nCoef), contourDtype)
        # Working array to hold the transpose of the Jacobian of F for a particular x(t).
        dFX = np.empty((nDep, nDep - 1), contourDtype)

        # Start Newton's method loop.
        previousFSamplesNorm = 0.0
        while True:
            FSamplesNorm = 0.0
            # Fill in FSamples and its Jacobian (dFCoefs) with respect to the coefficients of x(t).
            for t, i in zip(tValues, range(nUnknownCoefs)):
                # Isolate coefficients and compute bspline values and their first two derivatives at t.
                ix = np.searchsorted(knots, t, 'right')
                ix = min(ix, nCoef)
                compactCoefs = coefs[:, ix - order:ix]
                bValues = bspy.Spline.bspline_values(ix, knots, order, t)
                dValues = bspy.Spline.bspline_values(ix, knots, order, t, 1)
                d2Values = bspy.Spline.bspline_values(ix, knots, order, t, 2)

                # Compute the dot constraint for x(t) and check for divergence from solution.
                dotValues = np.dot(compactCoefs @ d2Values, compactCoefs @ dValues)
                FSamplesNorm = max(FSamplesNorm, abs(dotValues))
                if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                    break

                # Do the same for F(x(t)).
                x = coefsMin + (compactCoefs @ bValues) * coefsMaxMinusMin
                FValues = F(x)
                for FValue in FValues:
                    FSamplesNorm = max(FSamplesNorm, abs(FValue))
                if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                    break

                # Record FSamples for t.
                FSamples[i, :-1] = FValues
                FSamples[i, -1] = dotValues

                # Compute the Jacobian of FSamples with respect to the coefficients of x(t).
                for j in range(nDep):
                    dFX[j] = dF[j](x) * coefsMaxMinusMin[j]
                FValues = np.outer(dFX.T, bValues).reshape(nDep - 1, nDep, order)
                dotValues = (np.outer(compactCoefs @ dValues, d2Values) + np.outer(compactCoefs @ d2Values, dValues)).reshape(nDep, order)
                dFCoefs[i, :-1, :, ix - order:ix] = FValues
                dFCoefs[i, -1, :, ix - order:ix] = dotValues
            
            # Check if we got closer to the solution.
            if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                # No we didn't, take a dampened step.
                coefDelta *= 0.5
                coefs[:, 1:-1] += coefDelta # Don't update endpoints
            else:
                # Yes we did, rescale FSamples and its Jacobian.
                if FSamplesNorm >= evaluationEpsilon:
                    FSamples /= FSamplesNorm
                    dFCoefs /= FSamplesNorm
                
                # Perform a Newton iteration.
                HSamples = FSamples.reshape(nUnknownCoefs * nDep)
                dHCoefs = dFCoefs[:, :, :, 1:-1].reshape((nUnknownCoefs * nDep, nDep * nUnknownCoefs))
                coefDelta = np.linalg.solve(dHCoefs, HSamples).reshape(nDep, nUnknownCoefs)
                coefs[:, 1:-1] -= coefDelta # Don't update endpoints

                # Record FSamples norm to ensure this Newton step is productive.
                previousFSamplesNorm = FSamplesNorm

            # Check for convergence of step size.
            if np.linalg.norm(coefDelta) < epsilon:
                # If step didn't improve the solution, remove it.
                if previousFSamplesNorm > 0.0 and FSamplesNorm > previousFSamplesNorm * (1.0 - evaluationEpsilon):
                    coefs[:, 1:-1] += coefDelta # Don't update endpoints
                break

        # Newton steps are done. Now check if we need to subdivide.
        # TODO: This would be FSamplesNorm / dHCoefs norm, but dHCoefs was divided by FSamplesNorm earlier.
        if FSamplesNorm / np.linalg.norm(dHCoefs, np.inf) < epsilon:
            break # We're done!
        
        # We need to subdivide, so build new knots, tValues, and GSamples arrays.
        nCoef = 2 * (nCoef - 1)
        nUnknownCoefs = nCoef - 2
        tValues = np.empty(nUnknownCoefs, contourDtype)
        GSamples = np.empty((nUnknownCoefs, nDep), contourDtype)
        previousKnot = knots[degree]
        newKnots = [previousKnot] * order
        i = 0
        for ix in range(order, len(knots) - degree, order - 2):
            knot = knots[ix]
            compactCoefs = coefs[:, ix - order:ix]

            # New knots are at the midpoint between old knots.
            newKnot = 0.5 * (previousKnot + knot)

            # Place tValues at Gauss points for the intervals [previousKnot, newKnot] and [newKnot, knot].
            for rho in reversed(rhos):
                tValues[i] = t = 0.5 * (previousKnot + newKnot - rho * (newKnot - previousKnot))
                GSamples[i] = compactCoefs @ bspy.Spline.bspline_values(ix, knots, order, t)
                i += 1
            for rho in rhos[0 if degree % 2 == 1 else 1:]:
                tValues[i] = t = 0.5 * (previousKnot + newKnot + rho * (newKnot - previousKnot))
                GSamples[i] = compactCoefs @ bspy.Spline.bspline_values(ix, knots, order, t)
                i += 1
            for rho in reversed(rhos):
                tValues[i] = t = 0.5 * (newKnot + knot - rho * (knot - newKnot))
                GSamples[i] = compactCoefs @ bspy.Spline.bspline_values(ix, knots, order, t)
                i += 1
            for rho in rhos[0 if degree % 2 == 1 else 1:]:
                tValues[i] = t = 0.5 * (newKnot + knot + rho * (knot - newKnot))
                GSamples[i] = compactCoefs @ bspy.Spline.bspline_values(ix, knots, order, t)
                i += 1
            
            newKnots += [newKnot] * (order - 2) # C1 continuity
            newKnots += [knot] * (order - 2) # C1 continuity
            previousKnot = knot
        
        # Update knots array.
        newKnots += [knot] * 2 # Clamp last knot
        knots = np.array(newKnots, contourDtype)
        assert i == nUnknownCoefs
        assert len(knots) == nCoef + order

    # Rescale x(t) back to original data points.
    coefs = (coefsMin + coefs.T * coefsMaxMinusMin).T
    spline = bspy.Spline(1, nDep, (order,), (nCoef,), (knots,), coefs, epsilon, metadata)
    if isinstance(F, bspy.Spline):
        spline = spline.confine(F.domain())
    return spline

def cylinder(radius, height, tolerance = None):
    if tolerance is None:
        tolerance = 1.0e-12
    bottom = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]] @ bspy.Spline.circular_arc(radius, 360.0, tolerance)
    top = bottom + [0.0, 0.0, height]
    return bspy.Spline.ruled_surface(bottom, top)

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

    # Determine the Greville absiccae to use as collocation points

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
    for iInd in range(nInd):
        if domain[iInd][0] < knots[iInd][order[iInd] - 1] or \
           domain[iInd][1] > knots[iInd][-order[iInd]]:  raise ValueError("One or more dataPoints are outside the domain of the spline")

    # Loop through each independent variable and fit all of the data
    
    accuracy = np.finfo(float).eps
    for iInd in range(nInd):
        nRows = pointsPerDirection[iInd]
        b = np.swapaxes(dataPoints, 0, iInd + 1)
        loopDep = nDep * nPoints // nRows
        b = np.reshape(b, (nRows, loopDep))
        done = False
        while not done:
            loopAccuracy = accuracy
            nCols = len(knots[iInd]) - order[iInd]
            A = np.zeros((nRows, nCols))
            u = -np.finfo(float).max
            fixedRows = []
            for iRow in range(nRows):
                uNew = uValues[iInd][iRow]
                if uNew != u:
                    iDerivative = 0
                    u = uNew
                    ix = np.searchsorted(knots[iInd], u, 'right')
                    ix = min(ix, nCols)
                else:
                    iDerivative += 1
                row = bspy.Spline.bspline_values(ix, knots[iInd], order[iInd], u, iDerivative)
                A[iRow, ix - order[iInd] : ix] = row
                if fixEnds and (u == uValues[iInd][0] or u == uValues[iInd][-1]):
                    fixedRows.append(iRow)
            nInterp = len(fixedRows)
            if nInterp != 0:
                C = np.take(A, fixedRows, 0)
                d = np.take(b, fixedRows, 0)
                AUse = np.delete(A, fixedRows, 0)
                bUse = np.delete(b, fixedRows, 0)
                U, Sigma, VT = np.linalg.svd(C)
                d = U.T @ d
                for iRow in range(nInterp):
                    d[iRow] = d[iRow] / Sigma[iRow]
                loopAccuracy *= Sigma[0] / Sigma[-1]
                rangeCols = np.take(VT.T, range(nInterp), 1)
                nullspace = np.delete(VT.T, range(nInterp), 1)
                x1 = rangeCols @ d
                b1 = bUse - AUse @ x1
                xNullspace, residuals, rank, s = np.linalg.lstsq(AUse @ nullspace, b1, rcond = None)
                x = x1 + nullspace @ xNullspace
            else:
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond = None)
            loopAccuracy *= s[0] / s[rank - 1]
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
        accuracy = loopAccuracy
    splineFit = bspy.Spline(nInd, nDep, order, pointsPerDirection, knots, dataPoints,
                            accuracy = accuracy, metadata = metadata)
    if splineInput:
        splineFit = splineFit.unfold(range(unfoldInfo.nInd), unfoldInfo)
    return splineFit

def line(startPoint, endPoint):
    startPoint = bspy.Spline.point(startPoint)
    endPoint = bspy.Spline.point(endPoint)
    return bspy.Spline.ruled_surface(startPoint, endPoint)

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
                       newCoefs, accuracy = max(newCurve1.accuracy, newCurve2.accuracy))

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
        rhoCrit = (math.sqrt(1.0 + 4.0 * (r0 + r1)) - 1.0) / (2.0 * (r0 + r1))
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
