import numpy as np
import bspy.spline

def least_squares(nInd, nDep, order, dataPoints, knotList = None, accuracy = 0.0, metadata = {}):
    assert nInd >= 0, "nInd < 0"
    assert nDep >= 0, "nDep < 0"
    assert len(order) == nInd, "len(order) != nInd"
    totalOrder = 1
    for ord in order:
        totalOrder *= ord

    totalDataPoints = len(dataPoints)
    for point in dataPoints:
        assert len(point) == nInd + nDep or len(point) == nInd + nDep * (nInd + 1), f"Data points are not dimension {nInd + nDep}"
        if len(point) == nInd + nDep * (nInd + 1):
            totalDataPoints += nInd

    if knotList is None:
        return NotImplemented
    else:
        assert len(knotList) == nInd, "len(knots) != nInd" # The documented interface uses the argument 'knots' instead of 'knotList'
        nCoef = [len(knotList[i]) - order[i] for i in range(nInd)]
        totalCoef = 1
        for knots, ord, nCf in zip(knotList, order, nCoef):
            for i in range(nCf):
                assert knots[i] <= knots[i + 1] and knots[i] < knots[i + ord], "Improperly ordered knot sequence"
            totalCoef *= nCf
        assert totalCoef <= totalDataPoints, f"Insufficient number of data points. You need at least {totalCoef}."
    
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
    coefs, residuals, rank, s = np.linalg.lstsq(A, b)

    # Reshape the coefs array to match nCoef (un-flatten) and move the dependent variables to the front.
    coefs = np.moveaxis(coefs.reshape((*nCoef, nDep)), -1, 0)

    # Return the resulting spline, computing the accuracy based on system epsilon and the norm of the residuals.
    maxError = np.finfo(coefs.dtype).eps
    if residuals.size > 0:
        maxError = max(maxError, np.linalg.norm(residuals))
    return bspy.Spline(nInd, nDep, order, nCoef, knotList, coefs, np.sqrt(maxError), metadata)