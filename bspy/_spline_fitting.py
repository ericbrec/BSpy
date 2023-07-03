import numpy as np
import bspy.spline
import math

def least_squares(nInd, nDep, order, dataPoints, knotList = None, compression = 0, metadata = {}):
    assert nInd >= 0, "nInd < 0"
    assert nDep >= 0, "nDep < 0"
    assert len(order) == nInd, "len(order) != nInd"
    assert 0 <= compression < 100, "compression not between 0 and 99"
    totalOrder = 1
    for ord in order:
        totalOrder *= ord

    totalDataPoints = len(dataPoints)
    for point in dataPoints:
        assert len(point) == nInd + nDep or len(point) == nInd + nDep * (nInd + 1), f"Data points are not dimension {nInd + nDep}"
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
        assert len(knotList) == nInd, "len(knots) != nInd" # The documented interface uses the argument 'knots' instead of 'knotList'
        nCoef = [len(knotList[i]) - order[i] for i in range(nInd)]
        totalCoef = 1
        newKnotList = []
        for knots, ord, nCf in zip(knotList, order, nCoef):
            for i in range(nCf):
                assert knots[i] <= knots[i + 1] and knots[i] < knots[i + ord], "Improperly ordered knot sequence"
            totalCoef *= nCf
            newKnotList.append(np.array(knots))
        assert totalCoef <= totalDataPoints, f"Insufficient number of data points. You need at least {totalCoef}."
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

# From Lowan, Arnold N., Norman Davids, and Arthur Levenson. "Table of the zeros of the Legendre polynomials of 
# order 1-16 and the weight coefficients for Gauss' mechanical quadrature formula." (1942): 739-743.
_legendre_polynomial_zeros = [
    [0.000000000000000],
    [0.577350269189626],
    [0.000000000000000,0.774596669241483],
    [0.339981043584856,0.861136311594053],
    [0.000000000000000,0.538469310105683,0.906179845938664],
    [0.238619186083197,0.661209386466265,0.932469514203152],
    [0.000000000000000,0.405845151377397,0.741531185599394,0.949107912342759],
    [0.183434642495650,0.525532409916329,0.796666477413627,0.960289856497536],
    [0.000000000000000,0.324253423403809,0.613371432700590,0.836031107326636,0.968160239507626],
    [0.148874338981631,0.433395394129247,0.679409568299024,0.865063366688985,0.973906528517172],
    [0.000000000000000,0.269543155952345,0.519096129110681,0.730152005574049,0.887062599768095,0.978228658146057],
    [0.125333408511469,0.367831498918180,0.587317954286617,0.769902674194305,0.904117256370475,0.981560634246719],
    [0.000000000000000,0.230458315955135,0.448492751036447,0.642349339440340,0.801578090733310,0.917598399222978,0.984183054718588],
    [0.108054948707344,0.319112368927890,0.515248636358154,0.687292904811685,0.827201315069765,0.928434883663574,0.986283808696812],
    [0.000000000000000,0.201194093997435,0.394151347077563,0.570972172608539,0.724417731360170,0.848206583410427,0.937273392400706,0.987992518020485],
    [0.095012509837637,0.281603550779259,0.458016777657227,0.617876244402644,0.755404408355003,0.865631202387832,0.944575023073233,0.989400934991650],
    ]

def contour(F, knownXValues, dF = None, epsilon = None, metadata = {}):
    # Set up parameters for initial guess of x(t) and validate arguments.
    order = 4
    degree = order - 1
    rhos = _legendre_polynomial_zeros[degree - 1 - 1]
    assert len(knownXValues) >= 2, "There must be at least 2 known x values."
    m = len(knownXValues) - 1
    nCoef = m * (degree - 1) + 2
    nUnknownCoefs = nCoef - 2

    # Create the list of data points for an initial guess of x(t), rescaled to [0, 1].
    knownXValues = np.array(knownXValues)
    contourDtype = knownXValues.dtype
    if epsilon is None:
        epsilon = math.sqrt(np.finfo(contourDtype).eps)
    evaluationEpsilon = np.sqrt(epsilon)
    coefsMin = knownXValues.min(axis=0)
    coefsMaxMinusMin = knownXValues.max(axis=0) - coefsMin
    coefsMaxMinMin = np.where(coefsMaxMinusMin < 1.0, 1.0, coefsMaxMinusMin)
    coefsMaxMinMinReciprocal = np.reciprocal(coefsMaxMinusMin)
    nDep = knownXValues.shape[1]
    previousPoint = None
    t = 0.0 # We start with t measuring contour length and later rescale.
    for knownXValue in knownXValues:
        FValues = F(knownXValue)
        #assert len(FValues) == nDep - 1 and np.linalg.norm(FValues) < evaluationEpsilon, f"F(known x) must be a zero vector of length {nDep - 1}."
        x = (knownXValue - coefsMin) * coefsMaxMinMinReciprocal # rescale to [0 , 1]
        point = np.insert(x, 0, t) # full data point
        if previousPoint is None:
            knots = [t] * order
            dataPoints = [point]
        else:
            dt = np.linalg.norm(point[1:] - previousPoint[1:])
            if dt > epsilon: # Skip duplicates
                t += dt
                point[0] = t
                knots += [t] * (order - 2)
                for rho in reversed(rhos):
                    dataPoints.append(0.5 * (previousPoint + point - rho * (point - previousPoint)))
                for rho in rhos[0 if degree % 2 == 1 else 1:]:
                    dataPoints.append(0.5 * (previousPoint + point + rho * (point - previousPoint)))
        previousPoint = point
    knots += [t] * 2
    contourSpeedMin = contourSpeed = t * t
    dataPoints.append(point)

    # Rescale t values in knots and data points to be [0, 1], and collect ts.
    knots[:] /= t
    for point in dataPoints:
        point[0] /= t
    ts = [point[0] for point in dataPoints[1:-1]] # Exclude endpoints
    assert len(ts) == nUnknownCoefs

    # Compute initial guess for x(t).
    spline = least_squares(1, nDep, (order,), dataPoints, (knots,))
    knots = spline.knots[0]

    # Establish the first derivatives of F.
    if dF is None:
        dF = []
        if isinstance(F, bspy.Spline):
            for i in range(nDep):
                def splineDerivative(x, i=i):
                    return F.derivative((i,), x)
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
        assert len(dF) == nDep, f"Must provide {nDep} first derivatives."
    
    # Start subdivision loop.
    while True:
        # Array to hold the values of F and contour speed for each t, excluding endpoints.
        samples = np.empty(nDep * nUnknownCoefs + 1, contourDtype)
        # Array to hold the Jacobian base of the samples with respect to the coefficients.
        # We'll trim it and expand it to form the final Jacobian at the end.
        # The Jacobian is banded due to b-spline local support, so initialize it to zero.
        J = np.zeros((nDep * nUnknownCoefs + 1, nDep * nCoef), contourDtype)
        # Working array to hold the transpose of the Jacobian of F for a particular x(t).
        dFValues = np.empty((nDep, nDep - 1), contourDtype)
        # Working array to hold the last column of the final Jacobian (its always the same).
        JLastColumn = np.zeros(nDep * nUnknownCoefs + 1, contourDtype)
        iDotStart = (nDep - 1) * nUnknownCoefs
        JLastColumn[iDotStart:] = -1.0

        # Start Newton's method loop.
        previousSamplesNorm = 0.0
        while True:
            FScale = 0.0
            dotScale = 0.0
            samplesNorm = 0.0
            # Fill in samples and its Jacobian (J) with respect to the coefficients of x(t).
            for iF, iDot, t in zip(range(0, iDotStart, nDep - 1), range(iDotStart, nDep * nUnknownCoefs), ts):
                # Isolate coefficients and compute bspline values and their first two derivatives at t.
                ix = np.searchsorted(knots, t, 'right')
                ix = min(ix, nCoef)
                coefs = spline.coefs[:, ix - order:ix]
                bValues = spline.bspline_values(ix, knots, order, t)
                dValues = spline.bspline_values(ix, knots, order, t, 1)

                # Compute the speed constraint for x(t) and check from divergence from solution.
                xPrime = coefs @ dValues
                dotValues = np.dot(xPrime, xPrime) - contourSpeed
                dotScale = max(dotScale, abs(dotValues))
                samplesNorm = max(samplesNorm, dotScale)
                if previousSamplesNorm > 0.0 and samplesNorm > previousSamplesNorm * (1.0 - evaluationEpsilon):
                    break

                # Do the same for F(x(t)).
                x = coefsMin + (coefs @ bValues) * coefsMaxMinusMin
                FValues = F(x)
                for FValue in FValues:
                    FScale = max(FScale, abs(FValue))
                samplesNorm = max(samplesNorm, FScale)
                if previousSamplesNorm > 0.0 and samplesNorm > previousSamplesNorm * (1.0 - evaluationEpsilon):
                    break

                # Record samples for t.
                samples[iF:iF + nDep - 1] = FValues
                samples[iDot] = dotValues

                # Compute the Jacobian of samples with respect to the coefficients of x(t).
                for j in range(nDep):
                    dFValues[j] = dF[j](x) * coefsMaxMinusMin[j]
                FValues = np.outer(dFValues.T, bValues).reshape(nDep - 1, nDep * order)
                dotValues = (2.0 * np.outer(xPrime, dValues)).reshape(nDep * order)
                J[iF:iF + nDep - 1, (ix - order) * nDep:ix * nDep] = FValues
                J[iDot, (ix - order) * nDep:ix * nDep] = dotValues
            
            # Check for convergence of residual.
            if samplesNorm < evaluationEpsilon:
                break
            
            # Check if we got closer to the solution.
            if previousSamplesNorm > 0.0 and samplesNorm > previousSamplesNorm * (1.0 - evaluationEpsilon):
                # No we didn't, take a dampened step.
                coefDelta *= 0.5
                spline.coefs[:, 1:-1] += coefDelta[:-1].reshape(nDep, nCoef - 2) # Don't update endpoints
                contourSpeed += coefDelta[-1]
            else:
                # Yes we did, construct the final sample and Jacobian.
                dValues = spline.bspline_values(ix, knots, order, 1.0, 1) # Using 1.0 for the final value (ix and coefs are aligned to it)
                xPrime = coefs @ dValues
                dotValues = np.dot(xPrime, xPrime) - contourSpeed
                dotScale = max(dotScale, abs(dotValues))
                samplesNorm = max(samplesNorm, dotScale)
                samples[-1] = dotValues
                dotValues = (2.0 * np.outer(xPrime, dValues)).reshape(nDep * order)
                J[-1, (ix - order) * nDep:ix * nDep] = dotValues
                JFinal = np.c_[J[:, nDep:-nDep], JLastColumn]

                # Rescale samples and its Jacobian.
                if FScale >= evaluationEpsilon:
                    samples[:iDotStart] /= FScale
                    JFinal[:iDotStart] /= FScale
                if dotScale >= evaluationEpsilon:
                    samples[iDotStart:] /= dotScale 
                    JFinal[iDotStart:] /= dotScale
                
                # Perform a Newton iteration.
                coefDelta = np.linalg.solve(JFinal, samples)

                # Dampen the iteration to a reasonable size within [0, 1].
                deltaMax = abs(coefDelta[:-1].max())
                if deltaMax > 0.5:
                    coefDelta *= 0.5 / deltaMax
                spline.coefs[:, 1:-1] -= coefDelta[:-1].reshape(nDep, nCoef - 2) # Don't update endpoints
                contourSpeed -= coefDelta[-1]
                contourSpeed = max(contourSpeed, contourSpeedMin)

                # Record samples norm to ensure this Newton step is productive.
                previousSamplesNorm = samplesNorm

            # Check for convergence of Newton step size.
            if np.linalg.norm(coefDelta) < epsilon:
                break

        # Newton steps are done. Now check if we need to subdivide.
        if samplesNorm / np.linalg.norm(JFinal, np.inf) < epsilon:
            break # We're done!
        
        # We need to subdivide, so build new knots array and new ts array.
        newKnots = []
        ts = []
        knots = np.unique(knots)
        previousKnot = knots[0]
        for knot in knots[1:]:
            # New knots are at the midpoint between old knots.
            newKnot = 0.5 * (previousKnot + knot)
            newKnots += [newKnot] * (order - 2) # C1 continuity

            # Place ts at Gauss points for the intervals [previousKnot, newKnot] and [newKnot, knot].
            for rho in reversed(rhos):
                ts.append(0.5 * (previousKnot + newKnot - rho * (newKnot - previousKnot)))
            for rho in rhos[0 if degree % 2 == 1 else 1:]:
                ts.append(0.5 * (previousKnot + newKnot + rho * (newKnot - previousKnot)))
            for rho in reversed(rhos):
                ts.append(0.5 * (newKnot + knot - rho * (knot - newKnot)))
            for rho in rhos[0 if degree % 2 == 1 else 1:]:
                ts.append(0.5 * (newKnot + knot + rho * (knot - newKnot)))
            
            previousKnot = knot
        
        # Add new knots to spline and update knots, nCoef, and nUnknownCoefs.
        spline = spline.elevate_and_insert_knots((0,), (newKnots,))
        knots = spline.knots[0]
        nCoef = spline.nCoef[0]
        nUnknownCoefs = nCoef - 2
        assert len(ts) == nUnknownCoefs

    # Rescale x(t) back to original data points.
    spline.coefs = (coefsMin + spline.coefs.T * coefsMaxMinusMin).T
    return spline