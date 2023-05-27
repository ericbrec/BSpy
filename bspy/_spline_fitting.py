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
        assert len(point) == nInd + nDep or len(point) == nInd + nDep * (nInd + 1), f"Data points are not dimension {nInd + nDep}"
        totalCoef = 1
        for knots, ord, nCf in zip(knotList, order, nCoef):
            for i in range(nCf):
                assert knots[i] <= knots[i + 1] and knots[i] < knots[i + ord], "Improperly ordered knot sequence"
            totalCoef *= nCf
        assert totalCoef <= totalDataPoints, f"Insufficient number of data points. You need at least {totalCoef}."
    
    A = np.zeros((totalDataPoints, totalCoef), type(dataPoints[0]))
    b = np.empty((totalDataPoints, nDep))
    for point, ARow, bRow in zip(dataPoints, A, b):
        bValues = []
        for knots, ord, nCf, value in zip(knotList, order, nCoef, point[:nInd]):
            ix = np.searchsorted(knots, value, 'right')
            ix = min(ix, nCf)
            bValues.append((ix, bspy.Spline.bspline_values(ix, knots, ord, value)))
        
        for i in range(totalOrder):
            pass
