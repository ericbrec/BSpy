import math
import numpy as np
import bspy.spline
from collections import namedtuple
from multiprocessing import Pool

def zeros_using_interval_newton(self):
    if not(self.nInd == self.nDep): raise ValueError("The number of independent variables (nInd) must match the number of dependent variables (nDep).")
    if not(self.nInd == 1): raise ValueError("Only works for curves (nInd == 1).")
    epsilon = np.finfo(self.coefs.dtype).eps

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
                if projectedRightStep - projectedLeftStep <= epsilon:
                    myZeros = [0.5 * (projectedLeftStep + projectedRightStep)]
                else:
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

def _convex_hull_2D(xData, yData, epsilon = 1.0e-8, evaluationEpsilon = 1.0e-4, xInterval = None):
    # Allow xData to be repeated for longer yData, but only if yData is a multiple.
    if not(len(yData) % len(xData) == 0): raise ValueError("Size of xData does not divide evenly in size of yData")

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
    if xInterval is not None and (y0 > evaluationEpsilon or yMax < -evaluationEpsilon or xMin > xInterval[1] or xMax < xInterval[0]):
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

Interval = namedtuple('Interval', ('spline', 'scale', 'slope', 'intercept', 'epsilon', 'atMachineEpsilon'))

# We use multiprocessing.Pool to call this function in parallel, so it cannot be nested and must take a single argument.
def _refine_projected_polyhedron(interval):
    Crit = 0.85 # Required percentage decrease in domain per iteration.
    epsilon = interval.epsilon
    evaluationEpsilon = np.sqrt(epsilon)
    machineEpsilon = np.finfo(interval.spline.coefs.dtype).eps
    root = None
    intervals = []
    scale = 0.0
    # Go through each nDep of the spline, checking bounds.
    for coefs in interval.spline.coefs:
        coefsMin = coefs.min()
        coefsMax = coefs.max()
        if coefsMax < -evaluationEpsilon or coefsMin > evaluationEpsilon:
            # No roots in this interval.
            return root, intervals
        scale = max(scale, abs(coefsMin), abs(coefsMax))
    newScale = scale * interval.scale

    if newScale < evaluationEpsilon:
        # Return the bounds of the interval within which the spline is zero.
        root = (interval.intercept, interval.slope + interval.intercept)
    else:
        # Rescale the spline to max 1.0.
        spline = interval.spline.scale(1.0 / scale)
        # Loop through each independent variable to determine a tighter domain around roots.
        domain = []
        for nInd, order, knots, nCoef, s in zip(range(spline.nInd), spline.order, spline.knots, spline.nCoef, interval.slope):
            # Start with the current interval for this independent variable.
            if s < epsilon:
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
                    hull = _convex_hull_2D(knotCoefs, coefs[nDep].ravel(), epsilon, evaluationEpsilon, xInterval)
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
            newSlope = np.multiply(width, interval.slope)
            # Iteration is complete if the interval actual width (slope) is either
            # one iteration past being less than sqrt(machineEpsilon) or simply less than epsilon.
            if interval.atMachineEpsilon or newSlope.max() < epsilon:
                newIntercept = np.multiply(domain[0], interval.slope) + interval.intercept
                # Double-check that we're at an actual zero (avoids boundary case).
                if newScale * np.linalg.norm(spline(0.5 * (domain[0] + domain[1]))) < evaluationEpsilon:
                    root = (newIntercept, newIntercept + newSlope)
            else:
                # Split domain in dimensions that aren't decreasing in width sufficiently.
                domains = [domain]
                for nInd, w, s in zip(range(spline.nInd), width, newSlope):
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
                    newSlope = np.multiply(width, interval.slope)
                    newIntercept = np.multiply(domain[0], interval.slope) + interval.intercept
                    for i, w in zip(range(spline.nInd), width):
                        if w < machineEpsilon:
                            if domain[0, i] > machineEpsilon:
                                domain[0, i] = domain[1, i] - machineEpsilon
                            else:
                                domain[1, i] = domain[0, i] + machineEpsilon
                    newDomain = [None if s < epsilon else (0.0, 1.0) for s in newSlope]
                    intervals.append(Interval(spline.trim(domain.T).reparametrize(newDomain), newScale, newSlope, newIntercept, epsilon, np.dot(newSlope, newSlope) < machineEpsilon))
  
    return (root, intervals)

def zeros_using_projected_polyhedron(self, epsilon=None):
    if not(self.nInd == self.nDep): raise ValueError("The number of independent variables (nInd) must match the number of dependent variables (nDep).")
    machineEpsilon = np.finfo(self.coefs.dtype).eps
    if epsilon is None:
        epsilon = self.accuracy
    epsilon = max(epsilon, np.sqrt(machineEpsilon))
    roots = []

    # Set initial spline, domain, and interval.
    domain = self.domain().T
    intervals = [Interval(self.trim(domain.T).reparametrize(((0.0, 1.0),) * self.nInd), 1.0, domain[1] - domain[0], domain[0], epsilon, False)]
    chunkSize = 8
    pool = Pool() # Pool size matches CPU count

    while intervals:
        nextIntervals = []
        if len(intervals) > chunkSize:
            for (root, newIntervals) in pool.imap_unordered(_refine_projected_polyhedron, intervals, chunkSize):
                if root is not None:
                    roots.append(root)
                nextIntervals += newIntervals
        else:
            for (root, newIntervals) in map(_refine_projected_polyhedron, intervals):
                if root is not None:
                    roots.append(root)
                nextIntervals += newIntervals
        intervals = nextIntervals
            
    # Connect intervals of zeros that overlap.
    gotOverlap = True
    while gotOverlap:
        gotOverlap = False
        i = 0
        rootCount = len(roots)
        while i < rootCount:
            iRoot = roots[i]
            root = (iRoot[0].copy(), iRoot[1].copy()) # Temporary storage for expanded interval
            j = i + 1
            # Check for overlap with other intervals.
            while j < rootCount:
                jRoot = roots[j]
                overlapped = True
                for d in range(self.nInd):
                    if iRoot[0][d] < jRoot[1][d] + epsilon and jRoot[0][d] < iRoot[1][d] + epsilon:
                        root[0][d] = min(iRoot[0][d], jRoot[0][d])
                        root[1][d] = max(iRoot[1][d], jRoot[1][d])
                    else:
                        overlapped = False
                        break
                if overlapped:
                    # For an overlapped interval, expand original interval and toss overlapping interval.
                    iRoot[0][:] = root[0]
                    iRoot[1][:] = root[1]
                    roots.pop(j)
                    rootCount -= 1
                    gotOverlap = True
                else:
                    j += 1
            i += 1
    
    # Collapse intervals to points as appropriate.
    for i in range(len(roots)):
        iRoot = roots[i]
        # If interval is small, just return a single value (not an interval).
        if True or np.linalg.norm(iRoot[1] - iRoot[0]) < 2.0 * epsilon:
            roots[i] = 0.5 * (iRoot[0] + iRoot[1])

    # Sort roots if there's only one dimension.
    if self.nInd == 1:
        roots.sort(key=lambda root: root[0] if isinstance(root, tuple) else root)

    return roots

def contours(self):
    if self.nInd - self.nDep != 1: raise ValueError("The number of free variables (self.nInd - self.nDep) must be one.")

    Point = namedtuple('Point', ('d', 'det', 'onBoundary', 'uvw'))
    epsilon = np.sqrt(np.finfo(self.coefs.dtype).eps)
    evaluationEpsilon = np.sqrt(epsilon)

    # Go through each nDep of the spline, checking bounds.
    for coefs in self.coefs:
        coefsMin = coefs.min()
        coefsMax = coefs.max()
        if coefsMax < -evaluationEpsilon or coefsMin > evaluationEpsilon:
            # No contours for this spline.
            return []

    # Record self's original domain and then reparametrize self's domain to [0, 1]^nInd.
    domain = self.domain().T
    self = self.reparametrize(((0.0, 1.0),) * self.nInd)
    
    # Construct self's tangents and normal.
    tangents = []
    for nInd in range(self.nInd):
        tangents.append(self.differentiate(nInd))
    normal = self.normal_spline((0, 1)) # We only need the first two indices

    theta = np.sqrt(2) # Arbitrary starting value for theta (picked one in [0, pi/2] unlikely to be a stationary point)
    # Try different theta values until no border or turning points are degenerate.
    while True:
        points = []
        theta *= 0.77
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        abort = False

        # Construct the turning point determinant.
        turningPointDeterminant = normal.dot((cosTheta, sinTheta))

        # Find intersections with u and v boundaries.
        def uvIntersections(nInd, boundary):
            zeros = self.contract([None] * nInd + [boundary] + [None] * (self.nInd - nInd - 1)).zeros()
            abort = False
            for zero in zeros:
                if isinstance(zero, tuple):
                    abort = True
                    break
                uvw = np.insert(np.array(zero), nInd, boundary)
                d = uvw[0] * cosTheta + uvw[1] * sinTheta
                det = (0.5 - boundary) * normal(uvw)[nInd] * turningPointDeterminant(uvw)
                if abs(det) < epsilon:
                    abort = True
                    break
                points.append(Point(d, det, True, uvw))
            return abort
        for nInd in range(2):
            abort = uvIntersections(nInd, 0.0)
            if abort:
                break # Try a different theta
            abort = uvIntersections(nInd, 1.0)
            if abort:
                break # Try a different theta
        if abort:
            continue # Try a different theta

        # Find intersections with other boundaries.
        def otherIntersections(nInd, boundary):
            zeros = self.contract([None] * nInd + [boundary] + [None] * (self.nInd - nInd - 1)).zeros()
            abort = False
            for zero in zeros:
                if isinstance(zero, tuple):
                    abort = True
                    break
                uvw = np.insert(np.array(zero), nInd, boundary)
                d = uvw[0] * cosTheta + uvw[1] * sinTheta
                columns = np.empty((self.nDep, self.nInd - 1))
                i = 0
                for j in range(self.nInd):
                    if j != nInd:
                        columns[:, i] = tangents[j](uvw)
                        i += 1
                duv = np.solve(columns, -tangents[nInd](uvw))
                det = np.arctan2((0.5 - boundary) * (duv[0] * cosTheta + duv[1] * sinTheta), (0.5 - boundary) * (duv[0] * cosTheta - duv[1] * sinTheta))
                if abs(det) < epsilon:
                    abort = True
                    break
                points.append(Point(d, det, True, uvw))
            return abort
        for nInd in range(2, self.nInd):
            abort = otherIntersections(nInd, 0.0)
            if abort:
                break # Try a different theta
            abort = otherIntersections(nInd, 1.0)
            if abort:
                break # Try a different theta
        if abort:
            continue # Try a different theta

        # Find turning points by combining self and turningPointDeterminant into a system and processing its zeros.
        systemSelf, systemTurningPointDeterminant = self.common_basis((turningPointDeterminant,), [(nInd, nInd) for nInd in range(self.nInd)])
        system = type(systemSelf)(self.nInd, self.nInd, systemSelf.order, systemSelf.nCoef, systemSelf.knots, \
            np.concatenate((systemSelf.coefs, systemTurningPointDeterminant.coefs)), systemSelf.accuracy, systemSelf.metadata)
        zeros = system.zeros()
        for uvw in zeros:
            if isinstance(uvw, tuple):
                abort = True
                break
            d = uvw[0] * cosTheta + uvw[1] * sinTheta 
            n = self.normal(uvw, False) # Computing all indices of the normal this time
            wrt = [0] * self.nInd
            det = 0.0
            for nInd in range(self.nInd):
                wrt[nInd] = 1
                det += turningPointDeterminant.derivative(wrt, uvw) * n[nInd]
                wrt[nInd] = 0
            if abs(det) < epsilon:
                abort = True
                break
            points.append(Point(d, det, False, uvw))
        if not abort:
            break # We're done!
    
    # We've got all the contour points, now we bucket them into individual contours using the algorithm 
    # from Grandine, Thomas A., and Frederick W. Klein IV. "A new approach to the surface intersection problem." 
    # Computer Aided Geometric Design 14, no. 2 (1997): 111-134.

    # Before we sort, we're going to need a system to find all the contour points on 
    # a panel boundary: u * cosTheta + v * sinTheta = d. Basically, we add this panel boundary plane
    # to the contour condition. We'll define it for d = 0, and add the actual d later.
    # We didn't construct the panel system earlier, because we didn't have theta.
    panelCoefs = np.empty((self.nDep + 1, *self.coefs.shape[1:]), self.coefs.dtype) # Note that self.nDep + 1 == self.nInd
    panelCoefs[:self.nDep] = self.coefs
    # The following value should be -d. We're setting it for d = 0 to start.
    panelCoefs[self.nDep, 0, 0] = 0.0 
    degree = self.order[0] - 1
    for i in range(1, self.nCoef[0]):
        panelCoefs[self.nDep, i, 0] = panelCoefs[self.nDep, i - 1, 0] + ((self.knots[0][degree + i] - self.knots[0][i]) / degree) * cosTheta
    degree = self.order[1] - 1
    for i in range(1, self.nCoef[1]):
        panelCoefs[self.nDep, :, i] = panelCoefs[self.nDep, :, i - 1] + ((self.knots[1][degree + i] - self.knots[1][i]) / degree) * sinTheta
    panel = type(self)(self.nInd, self.nInd, self.order, self.nCoef, self.knots, panelCoefs, self.accuracy, self.metadata)

    # Okay, we have everything we need to determine the contour topology and points along each contour.
    # We've done the first two steps of Grandine and Klein's algorithm:
    # (1) Choose theta and find all solutions to (1.6) (system)
    # (2) Find all zeros of f on the boundary of [0, 1]^2

    # Next, sort the edge and turning points by panel distance (d) and then by the determinant (det)
    # (3) Take all the points found in Step (1) and Step (2) and order them by distance in the theta direction from the origin.
    points.sort()

    # Extra step not in the paper: Add a panel between two consecutive turning points to uniquely determine contours between them.
    if len(points) > 1:
        i = 0
        previousPoint = points[i]
        while i < len(points) - 1:
            i += 1
            point = points[i]
            if not previousPoint.onBoundary and not point.onBoundary and point.d - previousPoint.d > epsilon:
                # We have two consecutive turning points on separate panels.
                # Insert a panel in between them, with the uvw value of None, since there is no zero associated.
                points.insert(i, Point(0.5 * (previousPoint.d + point.d), 0.0, False, None))
                i += 1
            previousPoint = point

    # (4) Initialize an ordered list of contours. No contours will be on the list at first.
    currentContourPoints = [] # Holds contours currently being identified
    contourPoints = [] # Hold contours already identified

    # (5) If no points remain to be processed, stop. Otherwise, take the next closest point.
    for point in points:
        # If it is a boundary point, go to Step (6). Otherwise, go to Step (7).
        if point.onBoundary:
            # (6) Determine whether the point corresponds to a contour which is starting or ending
            # at the given point. A point corresponds to a starting contour if it continues in the
            # increasing panel direction, and it corresponds to an ending contour if it continues
            # in the decreasing panel direction. If it is starting and the point is on the v = 0
            # or u = 1 edge, add a new contour to the front of the ordered list of contours
            # with the given point as an endpoint. If it is starting and the point is on the u = 0
            # or v = 1 edge, add a new contour to the end of the ordered list. If it is an
            # ending point, then delete a contour from either the beginning or the end of the
            # list, depending upon which edge the point is on. Go back to Step (5).
            if point.det > 0.0:
                # Starting point
                if abs(point.uvw[0] - 1.0) < epsilon or abs(point.uvw[1]) < epsilon:
                    currentContourPoints.insert(0, [True, point.uvw]) # True indicates end point
                else:
                    currentContourPoints.append([True, point.uvw]) # True indicates end point
            else:
                # Ending point
                if abs(point.uvw[0] - 1.0) < epsilon or abs(point.uvw[1]) < epsilon:
                    i = 0
                else:
                    i = -1
                fullList = currentContourPoints.pop(i) + [point.uvw]
                endPoint = fullList[0]
                if endPoint:
                    contourPoints.append(fullList)
                else:
                    fullList.reverse() # The last two values will be a repeat turning point and the endPoint flag, we remove them on the next line
                    currentContourPoints[i] = [True] + fullList[:-2] + currentContourPoints[i][1:]
        else:
            # (7) Determine whether two contours start or two contours end
            # at the turning point. Locate the two contours in the list of contours by finding
            # all points which lie on both the panel boundary and on the contour. The turning
            # point will be one of these, and it will be well ordered with respect to the other
            # points. Either insert two new contours in the list or delete two existing ones from
            # the list. Go back to Step (5).
            # First, construct panel, whose zeros lie along the panel boundary, u * cosTheta + v * sinTheta - d = 0.
            panel.coefs[self.nDep] -= point.d

            if point.uvw is None:
                # For an inserted panel between two consecutive turning points, just find zeros along the panel.
                panelPoints = panel.zeros()
            else:
                # Split panel below and above the known zero point.
                # This avoids extra computation and the high-zero at the known zero point.
                panelPoints = [point.uvw]
                # To split the panel, we need to determine the offset from the point.
                # Since the objective function (self) is zero and its derivative is zero at the point,
                # we use second derivatives to determine when the objective function will likely grow 
                # evaluationEpsilon above zero again.
                wrt = [0] * self.nInd
                wrt[0] = 2
                selfUU = self.derivative(wrt, point.uvw)
                wrt[0] = 1
                wrt[1] = 1
                selfUV = self.derivative(wrt, point.uvw)
                wrt[0] = 0
                wrt[1] = 2
                selfVV = self.derivative(wrt, point.uvw)
                offset = np.sqrt(2.0 * evaluationEpsilon / \
                    np.linalg.norm(selfUU * sinTheta * sinTheta - 2.0 * selfUV * sinTheta * cosTheta + selfVV * cosTheta * cosTheta))
                # Now, we can find the zeros of the split panel, checking to ensure each panel is within bounds first.
                if point.uvw[0] + sinTheta * offset < 1.0 - epsilon and epsilon < point.uvw[1] - cosTheta * offset:
                    panelPoints += panel.trim(((point.uvw[0] + sinTheta * offset, 1.0), (0.0, point.uvw[1] - cosTheta * offset)) + ((None, None),) * (self.nInd - 2)).zeros()
                if epsilon < point.uvw[0] - sinTheta * offset and point.uvw[1] + cosTheta * offset < 1.0 - epsilon:
                    panelPoints += panel.trim(((0.0, point.uvw[0] - sinTheta * offset), (point.uvw[1] + cosTheta * offset, 1.0)) + ((None, None),) * (self.nInd - 2)).zeros()
            # Sort zero points by their position along the panel boundary (using vector orthogonal to its normal).
            panelPoints.sort(key=lambda uvw: uvw[1] * cosTheta - uvw[0] * sinTheta)
            # Add d back to prepare for next turning point.
            panel.coefs[self.nDep] += point.d
            # Go through panel points, adding them to existing contours, creating new ones, or closing old ones.
            adjustment = 0 # Adjust index after a contour point is added or removed.
            for i, uvw in zip(range(len(panelPoints)), panelPoints):
                if point.uvw is not None and np.allclose(point.uvw, uvw):
                    if point.det > 0.0:
                        # Insert the turning point twice (second one appears before the first one in the points list).
                        currentContourPoints.insert(i, [True, point.uvw]) # True indicates end point
                        currentContourPoints.insert(i, [False, point.uvw]) # False indicates continuation point
                        adjustment = 1
                    else:
                        secondHalf = currentContourPoints.pop(i + 1)
                        endPoint = secondHalf.pop(0)
                        secondHalf.reverse()
                        fullList = currentContourPoints.pop(i) + [point.uvw] + secondHalf
                        if endPoint:
                            contourPoints.append(fullList)
                        else:
                            currentContourPoints[i] = fullList + currentContourPoints[i][1:]
                        adjustment = -1
                else:
                    currentContourPoints[i + adjustment].append(uvw)

    # We've determined a bunch of points along all the contours, including starting and ending points.
    # Now we just need to create splines for those contours using the Spline.contour method.
    splineContours = []
    for points in contourPoints:
        contour = bspy.spline.Spline.contour(self, points[1:]) # Skip endPoint boolean at start of points list
        # Transform the contour to self's original domain.
        contour.coefs = (contour.coefs.T * (domain[1] - domain[0]) + domain[0]).T
        splineContours.append(contour)
    
    return splineContours