import numpy as np
import bspy.spline
import math

def circular_arc(radius, angle, tolerance):
    if radius < 0.0 or angle < 0.0 or tolerance < 0.0: raise ValueError("The radius, angle, and tolerance must be positive.")

    samples = int(max(np.ceil(((1.1536e-5 * radius / tolerance)**(1/8)) * angle / 90), 2.0)) + 1
    return bspy.Spline.section([(radius * np.cos(u * angle * np.pi / 180), radius * np.sin(u * angle * np.pi / 180), 90 + u * angle, 1.0 / radius) for u in np.linspace(0.0, 1.0, samples)])

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
        # where dGCoefs and GSamples are the b-spline values and sample points, respectively, for x(t).
        # The dGCoefs matrix is banded due to b-spline local support, so initialize it to zero.
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
        # The Jacobian is banded due to b-spline local support, so initialize it to zero.
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

def least_squares(nInd, nDep, order, dataPoints, knotList = None, compression = 0, metadata = {}):
    if not(nInd >= 0): raise ValueError("nInd < 0")
    if not(nDep >= 0): raise ValueError("nDep < 0")
    if not(len(order) == nInd): raise ValueError("len(order) != nInd")
    if not(0 <= compression < 100): raise ValueError("compression not between 0 and 99")
    totalOrder = 1
    for ord in order:
        totalOrder *= ord

    totalDataPoints = len(dataPoints)
    for point in dataPoints:
        if not(len(point) == nInd + nDep or len(point) == nInd + nDep * (nInd + 1)): raise ValueError(f"Data points do not have {nInd + nDep} values")
        if len(point) == nInd + nDep * (nInd + 1):
            totalDataPoints += nInd

    if knotList is None:
        # Compute the target number of coefficients and the actual number of samples in each independent variable.
        targetTotalCoef = len(dataPoints) * (100 - compression) / 100.0
        totalCoef = 1
        knotSamples = np.array([point[:nInd] for point in dataPoints], type(dataPoints[0][0])).T
        knotList = []
        for knotSample in knotSamples:
            knots = np.unique(knotSample)
            knotList.append(knots)
            totalCoef *= len(knots)
        
        # Scale the number of coefficients for each independent variable so that the total closely matches the target.
        scaling = min((targetTotalCoef / totalCoef) ** (1.0 / nInd), 1.0)
        nCoef = []
        totalCoef = 1
        for knots in knotList:
            nCf = int(math.ceil(len(knots) * scaling))
            nCoef.append(nCf)
            totalCoef *= nCf
        
        # Compute "ideal" knots for each independent variable, based on the number of coefficients and the sample values.
        # Piegl, Les A., and Wayne Tiller. "Surface approximation to scanned data." The visual computer 16 (2000): 386-395.
        newKnotList = []
        for iInd, ord, nCf, knots in zip(range(nInd), order, nCoef, knotList):
            degree = ord - 1
            newKnots = [knots[0]] * ord
            inc = len(knots)/nCf
            low = 0
            d = -1
            w = np.empty((nCf,), float)
            for i in range(nCf):
                d += inc
                high = int(d + 0.5 + 1) # Paper's algorithm sets high to d + 0.5, but only references high + 1
                w[i] = np.mean(knots[low:high])
                low = high
            for i in range(1, nCf - degree):
                newKnots.append(np.mean(w[i:i + degree]))
            newKnots += [knots[-1]] * ord
            newKnotList.append(np.array(newKnots, knots.dtype))
        knotList = newKnotList
    else:
        if not(len(knotList) == nInd): raise ValueError("len(knots) != nInd") # The documented interface uses the argument 'knots' instead of 'knotList'
        nCoef = [len(knotList[i]) - order[i] for i in range(nInd)]
        totalCoef = 1
        newKnotList = []
        for knots, ord, nCf in zip(knotList, order, nCoef):
            for i in range(nCf):
                if not(knots[i] <= knots[i + 1] and knots[i] < knots[i + ord]): raise ValueError("Improperly ordered knot sequence")
            totalCoef *= nCf
            newKnotList.append(np.array(knots))
        if not(totalCoef <= totalDataPoints): raise ValueError(f"Insufficient number of data points. You need at least {totalCoef}.")
        knotList = newKnotList
    
    # Initialize A and b from the likely overdetermined equation, A x = b, where A contains the bspline values at the independent variables,
    # b contains point values for the dependent variables, and the x contains the desired coefficients.
    A = np.zeros((totalDataPoints, totalCoef), type(dataPoints[0][0]))
    b = np.empty((totalDataPoints, nDep), A.dtype)

    # Fill in the bspline values in A and the dependent point values in b at row at a time.
    # Note that if a data point also specifies first derivatives, it fills out nInd + 1 rows (the point and its derivatives).
    row = 0
    for point in dataPoints:
        hasDerivatives = len(point) == nInd + nDep * (nInd + 1)

        # Compute the bspline values (and their first derivatives as needed).
        bValueData = []
        for knots, ord, nCf, u in zip(knotList, order, nCoef, point[:nInd]):
            ix = np.searchsorted(knots, u, 'right')
            ix = min(ix, nCf)
            bValueData.append((ix, bspy.Spline.bspline_values(ix, knots, ord, u), \
                bspy.Spline.bspline_values(ix, knots, ord, u, 1) if hasDerivatives else None))
        
        # Compute the values for the A array.
        # It's a little tricky because we have to multiply nInd different bspline arrays of different sizes
        # and index into flattened A array. The solution is to loop through the total number of entries
        # being changed (totalOrder), and compute the array indices via mods and multiplies.
        indices = [0] * nInd
        for i in range(totalOrder):
            column = 0
            bValues = np.ones((nInd + 1,), A.dtype)
            for j, ord, nCf, index, (ix, values, dValues) in zip(range(1, nInd + 1), order, nCoef, indices, bValueData):
                column = column * nCf + ix - ord + index
                # Compute the bspline value for this specific element of A.
                bValues[0] *= values[index]
                if hasDerivatives:
                    # Compute the first derivative values for each independent variable.
                    for k in range(1, nInd + 1):
                        bValues[k] *= dValues[index] if k == j else values[index]

            # Assign all the values and derivatives.
            A[row, column] = bValues[0]
            if hasDerivatives:
                for k in range(1, nInd + 1):
                    A[row + k, column] = bValues[k]

            # Increment the bspline indices.
            for j in range(nInd - 1, -1, -1):
                indices[j] = (indices[j] + 1) % order[j]
                if indices[j] > 0:
                    break

        # Assign values for the b array.
        b[row, :] = point[nInd:nInd + nDep]
        if hasDerivatives:
            for k in range(1, nInd + 1):
                b[row + k, :] = point[nInd + nDep * k:nInd + nDep * (k + 1)]

        # Increment the row before filling in the next data point
        row += nInd + 1 if hasDerivatives else 1
    
    # Yay, the A and b arrays are ready to solve.
    # Now, we call numpy's least squares solver.
    coefs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Reshape the coefs array to match nCoef (un-flatten) and move the dependent variables to the front.
    coefs = np.moveaxis(coefs.reshape((*nCoef, nDep)), -1, 0)

    # Return the resulting spline, computing the accuracy based on system epsilon and the norm of the residuals.
    maxError = np.finfo(coefs.dtype).eps
    if residuals.size > 0:
        maxError = max(maxError, residuals.sum())
    return bspy.Spline(nInd, nDep, order, nCoef, knotList, coefs, np.sqrt(maxError), metadata)

def ruled_surface(curve1, curve2):
    # Ensure that the splines are compatible
    if curve1.nInd != curve2.nInd:  raise ValueError("Splines must have the same number of independent variables")
    if curve1.nDep != curve2.nDep:  raise ValueError("Splines must have the same number of dependent variables")
    [newCurve1, newCurve2] = curve1.common_basis([curve2], ((0, 0),))

    # Generate the ruled spline between them
    return bspy.Spline(curve1.nInd + 1, curve1.nDep, list(newCurve1.order) + [2],
                       list(newCurve1.nCoef) + [2], list(newCurve1.knots) + [[0.0, 0.0, 1.0, 1.0]],
                       [np.array([coef1, coef2]).T for coef1, coef2 in zip(newCurve1.coefs, newCurve2.coefs)],
                       accuracy = max(newCurve1.accuracy, newCurve2.accuracy))

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

