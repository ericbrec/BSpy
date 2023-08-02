import math
import numpy as np
from collections import namedtuple

def zeros_using_interval_newton(self):
    assert self.nInd == self.nDep, "The number of independent variables (nInd) must match the number of dependent variables (nDep)."
    assert self.nInd == 1, "Only works for curves (nInd == 1)."
    epsilon = np.finfo(self.knots[0].dtype).eps

    # Set initial spline and domain
 
    spline = self
    (domain,) = spline.domain()

    # Make sure this works for weird intervals of definition

    if domain[0] != 0.0 or domain[1] != 1.0:
        spline = spline.reparametrize([[0.0, 1.0]])
        myZeros = spline.zeros()
        fixedZeros = []
        for thisZero in myZeros:
            if type(thisZero) is type((1.0, 2.0),):
                newLeft = (1.0 - thisZero[0]) * domain[0] + thisZero[0] * domain[1]
                newRight = (1.0 - thisZero[1]) * domain[0] + thisZero[1] * domain[1]
                fixedZeros += [(newLeft, newRight),]
            else:
                fixedZeros += [(1.0 - thisZero) * domain[0] + thisZero * domain[1]]
        return fixedZeros

    # Perform an interval Newton step

    def refine(spline, intervalSize, functionMax):
        (boundingBox,) = spline.range_bounds()
        if boundingBox[0] * boundingBox[1] > epsilon:
            return []
        scaleFactor = max(abs(boundingBox[0]), abs(boundingBox[1]))
        scaledSpline = spline.scale(1.0 / scaleFactor)
        (myDomain,) = scaledSpline.domain()
        intervalSize *= myDomain[1] - myDomain[0]
        functionMax *= scaleFactor
        mySpline = scaledSpline.reparametrize([[0.0, 1.0]])
        midPoint = 0.5
        [functionValue] = mySpline([midPoint])

        # Root found

        if intervalSize < epsilon or abs(functionValue) * functionMax < epsilon:
            if intervalSize < epsilon ** 0.25:
                return [0.5 * (myDomain[0] + myDomain[1])]
            else:
                myZeros = refine(mySpline.trim(((0.0, midPoint - np.sqrt(epsilon)),)), intervalSize, functionMax)
                myZeros += [0.5 * (myDomain[0] + myDomain[1])]
                myZeros += refine(mySpline.trim(((midPoint + np.sqrt(epsilon), 1.0),)), intervalSize, functionMax)
                return myZeros

        # Calculate Newton update

        (derivativeBounds,) = mySpline.differentiate().range_bounds()
        if derivativeBounds[0] == 0.0:
            derivativeBounds[0] = epsilon
        if derivativeBounds[1] == 0.0:
            derivativeBounds[1] = -epsilon
        leftNewtonStep = midPoint - functionValue / derivativeBounds[0]
        rightNewtonStep = midPoint - functionValue / derivativeBounds[1]
        adjustedLeftStep = min(leftNewtonStep, rightNewtonStep) - 0.5 * epsilon
        adjustedRightStep = max(leftNewtonStep, rightNewtonStep) + 0.5 * epsilon
        if derivativeBounds[0] * derivativeBounds[1] >= 0.0:    # Refine interval
           projectedLeftStep = max(0.0, adjustedLeftStep)
           projectedRightStep = min(1.0, adjustedRightStep)
           if projectedLeftStep <= projectedRightStep:
               trimmedSpline = mySpline.trim(((projectedLeftStep, projectedRightStep),))
               myZeros = refine(trimmedSpline, intervalSize, functionMax)
           else:
               return []
        else:                           # . . . or split as needed
            myZeros = []
            if adjustedLeftStep > 0.0:
                trimmedSpline = mySpline.trim(((0.0, adjustedLeftStep),))
                myZeros += refine(trimmedSpline, intervalSize, functionMax)
            if adjustedRightStep < 1.0:
                trimmedSpline = mySpline.trim(((adjustedRightStep, 1.0),))
                myZeros += refine(trimmedSpline, intervalSize, functionMax)
        return [(1.0 - thisZero) * myDomain[0] + thisZero * myDomain[1] for thisZero in myZeros]

    # See if there are any zero intervals

    (boundingBox,) = spline.range_bounds()
    scaleFactor = max(abs(boundingBox[0]), abs(boundingBox[1]))
    mySolution = []
    for interval in range(spline.nCoef[0] - spline.order[0] + 1):
        functionMax = max(np.abs(spline.coefs[0][interval:interval + spline.order[0]]))
        if functionMax < scaleFactor * epsilon: # Found an interval of zeros
            intervalExtend = spline.nCoef[0] - spline.order[0] - interval
            for ix in range(intervalExtend):    # Attempt to extend the interval to more than one polynomial piece
                if abs(spline.coefs[0][interval + ix + spline.order[0]]) >= scaleFactor * epsilon:
                    intervalExtend = ix
                    break
            leftEnd = spline.knots[0][interval + spline.order[0] - 1]
            rightEnd = spline.knots[0][interval + spline.order[0] + intervalExtend]
            if domain[0] != leftEnd:            # Compute zeros from left of the interval
                mySolution = refine(spline.trim(((domain[0], leftEnd - np.sqrt(epsilon)),)),
                                    max (1.0, 1.0 / (leftEnd - domain[0])), 1.0)
            mySolution += [(leftEnd, rightEnd)] # Add the interval of zeros
            if rightEnd != domain[1]:           # Add the zeros from right of the interval
                mySolution += spline.trim(((rightEnd + np.sqrt(epsilon), domain[1]),)).zeros()
            return mySolution
    return refine(spline, 1.0, 1.0)

def _convex_hull_2D(xData, yData, epsilon = 1.0e-8, xInterval = None):
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
    if xInterval is not None and (y0 > epsilon or yMax < -epsilon or xMin > xInterval[1] or xMax < xInterval[0]):
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
        if previousPoint is not None and abs(previousPoint[0] - point[0]) < epsilon:
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

def _intersect_convex_hull_with_x_interval(hullPoints, epsilon, xInterval):
    xMin = xInterval[1] + epsilon
    xMax = xInterval[0] - epsilon
    previousPoint = hullPoints[-1]
    for point in hullPoints:
        # Check for intersection with x axis.
        if previousPoint[1] * point[1] <= epsilon:
            determinant = point[1] - previousPoint[1]
            if abs(determinant) > epsilon:
                # Crosses x axis, determine intersection.
                x = previousPoint[0] - previousPoint[1] * (point[0] - previousPoint[0]) / determinant
                xMin = min(xMin, x)
                xMax = max(xMax, x)
            elif abs(point[1]) < epsilon:
                # Touches at endpoint. (Previous point is checked earlier.)
                xMin = min(xMin, point[0])
                xMax = max(xMax, point[0])
        previousPoint = point

    if xMin > xInterval[1] or xMax < xInterval[0]:
        return None
    else:
        return (max(xMin, xInterval[0]), min(xMax, xInterval[1]))

def zeros_using_projected_polyhedron(self, epsilon=None):
    assert self.nInd == self.nDep, "The number of independent variables (nInd) must match the number of dependent variables (nDep)."
    machineEpsilon = np.finfo(self.knots[0].dtype).eps
    if epsilon is None:
        epsilon = max(self.accuracy, machineEpsilon)
    Crit = 0.85 # Required percentage decrease in domain per iteration.
    evaluationEpsilon = np.sqrt(epsilon)
    roots = []

    # Set initial spline, domain, and interval.
    spline = self
    domain = spline.domain().T
    Interval = namedtuple('Interval', ('spline', 'slope', 'intercept', 'atMachineEpsilon'))
    intervalStack = [Interval(spline.trim(domain.T).reparametrize(((0.0, 1.0),) * spline.nInd), domain[1] - domain[0], domain[0], False)]

    # Process intervals until none remain
    while intervalStack:
        interval = intervalStack.pop()
        scale = np.abs(interval.spline.range_bounds()).max()
        if scale < epsilon:
            # Return the bounds of the interval within which the spline is zero.
            roots.append((interval.intercept, interval.slope + interval.intercept))
        else:
            # Rescale the spline to max 1.0.
            spline = interval.spline.scale(1.0 / scale)
            # Loop through each independent variable to determine a tighter domain around roots.
            domain = []
            for nInd, order, knots, nCoef, slope in zip(range(spline.nInd), spline.order, spline.knots, spline.nCoef, interval.slope):
                # Start with the current interval for this independent variable.
                if slope < epsilon:
                    # If the slope for this independent variable is less than epsilon, 
                    # then we've isolated its value and should leave its interval unchanged.
                    domain.append(spline.domain()[nInd])
                else:
                    # Move independent variable to the last (fastest) axis, adding 1 to account for the dependent variables.
                    coefs = np.moveaxis(spline.coefs, nInd + 1, -1)

                    # Compute the coefficients for f(x) = x for the independent variable and its knots.
                    degree = order - 1
                    knotCoefs = np.empty((nCoef,), knots.dtype)
                    knotCoefs[0] = knots[1]
                    for i in range(1, nCoef):
                        knotCoefs[i] = knotCoefs[i - 1] + (knots[i + degree] - knots[i])/degree
                    
                    # Loop through each dependent variable to compute the interval containing the root for this independent variable.
                    xInterval = (0.0, 1.0)
                    for nDep in range(spline.nDep):
                        # Compute the 2D convex hull of the knot coefficients and the spline's coefficients
                        hull = _convex_hull_2D(knotCoefs, coefs[nDep].flatten(), epsilon, xInterval)
                        if hull is None:
                            xInterval = None
                            break
                        
                        # Intersect the convex hull with the xInterval along the x axis (the knot coefficients axis).
                        xInterval = _intersect_convex_hull_with_x_interval(hull, epsilon, xInterval)
                        if xInterval is None:
                            break
                    
                    # Add valid xInterval to domain.
                    if xInterval is None:
                        domain = None
                        break
                    domain.append(xInterval)
            
            if domain is not None:
                domain = np.array(domain).T
                width = domain[1] - domain[0]
                slope = np.multiply(width, interval.slope)
                # Iteration is complete if the interval actual width (slope) is either
                # one iteration past being less than sqrt(machineEpsilon) or simply less than epsilon.
                if interval.atMachineEpsilon or slope.max() < epsilon:
                    intercept = np.multiply(domain[0], interval.slope) + interval.intercept
                    root = intercept + 0.5 * slope
                    # Double-check that we're at an actual zero (avoids boundary case).
                    if np.linalg.norm(self(root)) < evaluationEpsilon:
                        # Check for duplicate root. We test for a distance between roots of 2*epsilon to account for a left vs. right sided limit.
                        foundDuplicate = False
                        for i, oldRoot in zip(range(len(roots)), roots):
                            if np.linalg.norm(root - oldRoot) < 2.0 * epsilon:
                                # For a duplicate root, return the average value.
                                roots[i] = 0.5 * (oldRoot + root)
                                foundDuplicate = True
                                break
                        if not foundDuplicate:
                            roots.append(root)
                else:
                    # Split domain in dimensions that aren't decreasing in width sufficiently.
                    domains = [domain]
                    for nInd, w, s in zip(range(spline.nInd), width, slope):
                        if s >= epsilon and w > Crit:
                            # Not close to root and didn't get the required decrease in with, so split the domain.
                            domainCount = len(domains) # Cache the domain list size, since we're increasing it mid loop
                            w *= 0.5 # Halve the domain width for this independent variable
                            for i in range(domainCount):
                                leftDomain = domains[i]
                                rightDomain = leftDomain.copy()
                                leftDomain[1][nInd] -= w # Alters domain in domains list
                                rightDomain[0][nInd] += w
                                domains.append(rightDomain)
                    
                    # Add new intervals to interval stack.
                    for domain in domains:
                        width = domain[1] - domain[0]
                        slope = np.multiply(width, interval.slope)
                        intercept = np.multiply(domain[0], interval.slope) + interval.intercept
                        newDomain = [None if s < epsilon else (0.0, 1.0) for s in slope]
                        intervalStack.append(Interval(spline.trim(domain.T).reparametrize(newDomain), slope, intercept, np.dot(slope, slope) < machineEpsilon))

    if self.nInd == 1:
        roots.sort(key=lambda root: root[0] if type(root) is tuple else root)
    return roots