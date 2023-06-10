import numpy as np
from collections import namedtuple

def zeros_using_interval_newton(self, epsilon=None):
    assert self.nInd == self.nDep
    assert self.nInd == 1
    machineEpsilon = np.finfo(self.knots[0].dtype).eps
    if epsilon is None:
        epsilon = max(self.accuracy, machineEpsilon)
    roots = []
    # Set initial spline, domain, and interval.
    spline = self
    (domain,) = spline.domain()
    Interval = namedtuple('Interval', ('spline', 'slope', 'intercept', 'atMachineEpsilon'))
    intervalStack = [Interval(spline.trim((domain,)).reparametrize(((0.0, 1.0),)), domain[1] - domain[0], domain[0], False)]

    def test_and_add_domain():
        """Macro to perform common operations when considering a domain as a new interval."""
        if domain[0] <= 1.0 and domain[1] >= 0.0:
            width = domain[1] - domain[0]
            if width >= 0.0:
                slope = width * interval.slope
                intercept = domain[0] * interval.slope + interval.intercept
                # Iteration is complete if the interval actual width (slope) is either
                # one iteration past being less than sqrt(machineEpsilon) or simply less than epsilon.
                if interval.atMachineEpsilon or slope < epsilon:
                    root = intercept + 0.5 * slope
                    # Double-check that we're at an actual zero (avoids boundary case).
                    if self((root,)) < epsilon:
                        # Check for duplicate root. We test for a distance between roots of 2*epsilon to account for a left vs. right sided limit.
                        if roots and abs(root - roots[-1]) < 2.0 * epsilon:
                            # For a duplicate root, return the average value.
                            roots[-1] = 0.5 * (roots[-1] + root)
                        else:
                            roots.append(root)
                else:
                    intervalStack.append(Interval(spline.trim((domain,)).reparametrize(((0.0, 1.0),)), slope, intercept, slope * slope < machineEpsilon))

    # Process intervals until none remain
    while intervalStack:
        interval = intervalStack.pop()
        range = interval.spline.range_bounds()
        scale = np.abs(range).max(axis=1)
        if scale < epsilon:
            roots.append((interval.intercept, interval.slope + interval.intercept))
        else:
            spline = interval.spline.scale(1.0 / scale)
            mValue = spline((0.5,))
            derivativeRange = spline.differentiate().range_bounds()
            if derivativeRange[0, 0] * derivativeRange[0, 1] <= 0.0:
                # Derivative range contains zero, so consider two intervals.
                leftIndex = 0 if mValue > 0.0 else 1
                domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                domain[1] = 1.0
                test_and_add_domain()
                domain[0] = 0.0
                domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                test_and_add_domain()
            else:
                leftIndex = 0 if mValue > 0.0 else 1
                domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                test_and_add_domain()
    
    return roots

def _convex_hull_2D(xData, yData, xInterval = None):
    # Allow xData to be repeated for longer yData, but only if yData is a multiple.
    assert len(yData) % len(xData) == 0

    # Assign p0 to the leftmost lowest point. Also compute xMin, xMax, and yMax.
    xMin = xMax = x0 = xData[0]
    yMax = y0 = yData[0]
    xIter = iter(xData[1:])
    for y in yData[1:]:
        x = next(xIter, None)
        if x is None:
            xIter = iter(xData)
            x = next(xIter)
            
        if y < y0 or (y == y0 and x < x0):
            (x0, y0) = (x, y)
        xMin = min(xMin, x)
        xMax = max(xMax, x)
        yMax = max(yMax, y)

    # Only return convex null if it contains y = 0 and x within xInterval.
    if xInterval is not None and (y0 > 0.0 or yMax < 0.0 or xMin > xInterval[1] or xMax < xInterval[0]):
        return None

    # Sort points by angle around p0.
    sortedPoints = []
    xIter = iter(xData)
    for y in yData:
        x = next(xIter, None)
        if x is None:
            xIter = iter(xData)
            x = next(xIter)
        sortedPoints.append((math.atan2(y - y0, x - x0), x, y))
    sortedPoints.sort()

    # Trim away points with the same angle (keep furthest point from p0), and then remove angle.
    trimmedPoints = [sortedPoints[0][1:]] # Ensure we keep the first point
    previousPoint = None
    previousDistance = -1.0
    for point in sortedPoints[1:]:
        if previousPoint is not None and abs(previousPoint[0] - point[0]) < 1.0e-8:
            if previousDistance < 0.0:
                previousDistance = (previousPoint[1] - x0) ** 2 + (previousPoint[2] - y0) ** 2
            distance = (point[1] - x0) ** 2 + (point[2] - y0) ** 2
            if distance > previousDistance:
                trimmedPoints[-1] = point[1:]
                previousPoint = point
                previousDistance = distance
        else:
            trimmedPoints.append(point[1:])
            previousPoint = point
            previousDistance = -1.0

    # Build the convex hull by moving counterclockwise around trimmed sorted points.
    hullPoints = []
    for point in trimmedPoints:
        while len(hullPoints) > 1 and \
            (hullPoints[-1][0] - hullPoints[-2][0]) * (point[1] - hullPoints[-2][1]) - \
            (hullPoints[-1][1] - hullPoints[-2][1]) * (point[0] - hullPoints[-2][0]) <= 0.0:
            hullPoints.pop()
        hullPoints.append(point)

    return hullPoints

def _intersect_convex_hull_with_x_interval(hullPoints, xInterval):
    xMin = xInterval[1] + 1.0e-8
    xMax = xInterval[0] - 1.0e-8
    previousPoint = hullPoints[-1]
    for point in hullPoints:
        # Check for intersection with x axis.
        if previousPoint[1] * point[1] <= 1.0e-8:
            determinant = point[1] - previousPoint[1]
            if abs(determinant) > 1.0e-8:
                # Crosses x axis, determine intersection.
                x = previousPoint[0] - previousPoint[1] * (point[0] - previousPoint[0]) / determinant
                xMin = min(xMin, x)
                xMax = max(xMax, x)
            elif abs(point[1]) < 1.0e-8:
                # Touches at endpoint. (Previous point is checked earlier.)
                xMin = min(xMin, point[0])
                xMax = max(xMax, point[0])
        previousPoint = point

    if xMin > xInterval[1] or xMax < xInterval[0]:
        return None
    else:
        return (max(xMin, xInterval[0]), min(xMax, xInterval[1]))

def zeros_using_projected_polyhedron(self, epsilon=None):
    assert self.nInd == self.nDep
    machineEpsilon = np.finfo(self.knots[0].dtype).eps
    if epsilon is None:
        epsilon = max(self.accuracy, machineEpsilon)
    roots = []

    # Set initial spline, domain, and interval.
    spline = self
    domain = spline.domain().T
    Interval = namedtuple('Interval', ('spline', 'slope', 'intercept', 'atMachineEpsilon'))
    intervalStack = [Interval(spline.trim(domain.T).reparametrize(((0.0, 1.0),) * spline.nInd), domain[1] - domain[0], domain[0], False)]

    def test_and_add_domain():
        """Macro to perform common operations when considering a domain as a new interval."""
        if domain[0] <= 1.0 and domain[1] >= 0.0:
            width = domain[1] - domain[0]
            if width >= 0.0:
                slope = width * interval.slope
                intercept = domain[0] * interval.slope + interval.intercept
                # Iteration is complete if the interval actual width (slope) is either
                # one iteration past being less than sqrt(machineEpsilon) or simply less than epsilon.
                if interval.atMachineEpsilon or slope < epsilon:
                    root = intercept + 0.5 * slope
                    # Double-check that we're at an actual zero (avoids boundary case).
                    if self((root,)) < epsilon:
                        # Check for duplicate root. We test for a distance between roots of 2*epsilon to account for a left vs. right sided limit.
                        if roots and abs(root - roots[-1]) < 2.0 * epsilon:
                            # For a duplicate root, return the average value.
                            roots[-1] = 0.5 * (roots[-1] + root)
                        else:
                            roots.append(root)
                else:
                    intervalStack.append(Interval(spline.trim((domain,)).reparametrize(((0.0, 1.0),)), slope, intercept, slope * slope < machineEpsilon))

    # Process intervals until none remain
    while intervalStack:
        interval = intervalStack.pop()
        range = interval.spline.range_bounds()
        scale = np.abs(range).max(axis=1)
        if scale < epsilon:
            roots.append((interval.intercept, interval.slope + interval.intercept))
        else:
            spline = interval.spline.scale(1.0 / scale)
            mValue = spline((0.5,))
            derivativeRange = spline.differentiate().range_bounds()
            if derivativeRange[0, 0] * derivativeRange[0, 1] <= 0.0:
                # Derivative range contains zero, so consider two intervals.
                leftIndex = 0 if mValue > 0.0 else 1
                domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                domain[1] = 1.0
                test_and_add_domain()
                domain[0] = 0.0
                domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                test_and_add_domain()
            else:
                leftIndex = 0 if mValue > 0.0 else 1
                domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                test_and_add_domain()
    
    return roots

def zeros(self, epsilon=None):
    assert self.nInd == self.nDep
    return zeros_using_interval_newton(self, epsilon)