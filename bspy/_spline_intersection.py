import logging
import math
import numpy as np
from bspy.manifold import Manifold
from bspy.hyperplane import Hyperplane
import bspy.spline
from bspy.solid import Solid, Boundary
from collections import namedtuple
from multiprocessing import Pool

def zeros_using_interval_newton(self):
    if not(self.nInd == self.nDep): raise ValueError("The number of independent variables (nInd) must match the number of dependent variables (nDep).")
    if not(self.nInd == 1): raise ValueError("Only works for curves (nInd == 1).")
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
        midPoint = 0.5 * (myDomain[0] + myDomain[1])
        [functionValue] = scaledSpline(midPoint)

        # Root found

        if intervalSize < epsilon or abs(functionValue) * functionMax < epsilon:
            if intervalSize < epsilon ** 0.25:
                return [midPoint]
            else:
                mySpline = scaledSpline.reparametrize([[0.0, 1.0]])
                myZeros = refine(mySpline.trim(((0.0, 0.5 - np.sqrt(epsilon)),)), intervalSize, functionMax)
                myZeros.append(midPoint)
                myZeros += refine(mySpline.trim(((0.5 + np.sqrt(epsilon), 1.0),)), intervalSize, functionMax)
                return myZeros

        # Calculate Newton update

        mySpline = scaledSpline.reparametrize([[0.0, 1.0]])
        midPoint = 0.5
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

def _convex_hull_2D(xData, yData, yBounds, epsilon = 1.0e-8):
    # Allow xData to be repeated for longer yData, but only if yData is a multiple.
    if not(yData.shape[0] % xData.shape[0] == 0): raise ValueError("Size of xData does not divide evenly in size of yData")

    # Assign (x0, y0) to the lowest point.
    yMinIndex = np.argmin(yData)
    x0 = xData[yMinIndex % xData.shape[0]]
    y0 = yData[yMinIndex]

    # Calculate y adjustment as needed for values close to zero
    yAdjustment = -yBounds[0] if yBounds[0] > 0.0 else -yBounds[1] if yBounds[1] < 0.0 else 0.0
    y0 += yAdjustment

    # Sort points by angle around p0.
    sortedPoints = []
    xIter = iter(xData)
    for y in yData:
        y += yAdjustment
        x = next(xIter, None)
        if x is None:
            xIter = iter(xData)
            x = next(xIter)
        sortedPoints.append((math.atan2(y - y0, x - x0), x, y))
    sortedPoints.sort()

    # Trim away points with the same angle (keep furthest point from p0), removing the angle from the list.
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

    if xMin - epsilon > xInterval[1] or xMax + epsilon < xInterval[0]:
        return None
    else:
        return (min(max(xMin, xInterval[0]), xInterval[1]), max(min(xMax, xInterval[1]), xInterval[0]))

Interval = namedtuple('Interval', ('spline', 'unknowns', 'scale', 'slope', 'intercept', 'epsilon', 'atMachineEpsilon'))

# We use multiprocessing.Pool to call this function in parallel, so it cannot be nested and must take a single argument.
def _refine_projected_polyhedron(interval):
    Crit = 0.85 # Required percentage decrease in domain per iteration.
    epsilon = interval.epsilon
    evaluationEpsilon = np.sqrt(epsilon)
    machineEpsilon = np.finfo(interval.spline.coefs.dtype).eps
    roots = []
    intervals = []

    # Remove dependent variables that are near zero and compute newScale.
    spline = interval.spline.copy()
    bounds = spline.range_bounds()
    keepDep = []
    for nDep, (coefsMin, coefsMax) in enumerate(bounds * interval.scale):
        if coefsMax < -epsilon or coefsMin > epsilon:
            # No roots in this interval.
            return roots, intervals
        if coefsMin < -epsilon or coefsMax > epsilon:
            # Dependent variable not near zero for entire interval.
            keepDep.append(nDep)

    spline.nDep = len(keepDep)
    if spline.nDep == 0:
        # Return the interval center and radius.
        roots.append((interval.intercept + 0.5 * interval.slope, 0.5 * np.linalg.norm(interval.slope)))
        return roots, intervals

    # Rescale remaining spline coefficients to max 1.0.
    bounds = bounds[keepDep]
    newScale = np.abs(bounds).max()
    spline.coefs = spline.coefs[keepDep]
    spline.coefs *= 1.0 / newScale
    bounds *= 1.0 / newScale
    newScale *= interval.scale
    
    # Loop through each independent variable to determine a tighter domain around roots.
    domain = []
    coefs = spline.coefs
    for nInd, order, knots, nCoef, s in zip(range(spline.nInd), spline.order, spline.knots, spline.nCoef, interval.slope):
        # Move independent variable to the last (fastest) axis, adding 1 to account for the dependent variables.
        coefs = np.moveaxis(spline.coefs, nInd + 1, -1)

        # Compute the coefficients for f(x) = x for the independent variable and its knots.
        degree = order - 1
        xData = np.empty((nCoef,), knots.dtype)
        xData[0] = knots[1]
        for i in range(1, nCoef):
            xData[i] = xData[i - 1] + (knots[i + degree] - knots[i])/degree
        
        # Loop through each dependent variable to compute the interval containing the root for this independent variable.
        xInterval = (0.0, 1.0)
        for yData, yBounds in zip(coefs, bounds):
            # Compute the 2D convex hull of the knot coefficients and the spline's coefficients
            hull = _convex_hull_2D(xData, yData.ravel(), yBounds, epsilon)
            if hull is None:
                return roots, intervals
            
            # Intersect the convex hull with the xInterval along the x axis (the knot coefficients axis).
            xInterval = _intersect_convex_hull_with_x_interval(hull, epsilon, xInterval)
            if xInterval is None:
                return roots, intervals
        
        domain.append(xInterval)
    
    # Compute new slope, intercept, and unknowns.
    domain = np.array(domain, spline.knots[0].dtype).T
    width = domain[1] - domain[0]
    newSlope = interval.slope.copy()
    newIntercept = interval.intercept.copy()
    newUnknowns = []
    newDomain = domain.copy()
    uvw = []
    nInd = 0
    for i, w, d in zip(interval.unknowns, width, domain.T):
        newSlope[i] = w * interval.slope[i]
        newIntercept[i] = d[0] * interval.slope[i] + interval.intercept[i]
        if newSlope[i] < epsilon:
            uvw.append(0.5 * (d[0] + d[1]))
            newDomain = np.delete(newDomain, nInd, axis=1)
        else:
            newUnknowns.append(i)
            uvw.append(None)
            nInd += 1

    # Iteration is complete if the interval actual width (slope) is either
    # one iteration past being less than sqrt(machineEpsilon) or there are no remaining unknowns.
    if interval.atMachineEpsilon or len(newUnknowns) == 0:
        # Return the interval center and radius.
        roots.append((newIntercept + 0.5 * newSlope, epsilon))
        return roots, intervals

    # Contract spline as needed.
    spline = spline.contract(uvw)

    # Use interval newton for one-dimensional splines.
    if spline.nInd == 1 and spline.nDep == 1:
        i = newUnknowns[0]
        for root in zeros_using_interval_newton(spline):
            if not isinstance(root, tuple):
                root = (root, root)
            w = root[1] - root[0]
            slope = newSlope.copy()
            intercept = newIntercept.copy()
            slope[i] = w * interval.slope[i]
            intercept[i] = root[0] * interval.slope[i] + interval.intercept[i]
            # Return the interval center and radius.
            roots.append((intercept + 0.5 * slope, epsilon))
        
        return roots, intervals

    # Split domain in dimensions that aren't decreasing in width sufficiently.
    width = newDomain[1] - newDomain[0]
    domains = [newDomain]
    for nInd, w in zip(range(spline.nInd), width):
        if w > Crit:
            # Didn't get the required decrease in width, so split the domain.
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
        splitSlope = newSlope.copy()
        splitIntercept = newIntercept.copy()
        for i, w, d in zip(newUnknowns, width, domain.T):
            splitSlope[i] = w * interval.slope[i]
            splitIntercept[i] = d[0] * interval.slope[i] + interval.intercept[i]
        intervals.append(Interval(spline.trim(domain.T).reparametrize(((0.0, 1.0),) * spline.nInd), newUnknowns, newScale, splitSlope, splitIntercept, epsilon, np.dot(splitSlope, splitSlope) < machineEpsilon))
  
    return roots, intervals

class _Region:
    def __init__(self, center, radius, count):
        self.center = center
        self.radius = radius
        self.count = count

def zeros_using_projected_polyhedron(self, epsilon=None):
    if not(self.nInd == self.nDep): raise ValueError("The number of independent variables (nInd) must match the number of dependent variables (nDep).")
    machineEpsilon = np.finfo(self.knots[0].dtype).eps
    if epsilon is None:
        epsilon = 0.0
    epsilon = max(epsilon, np.sqrt(machineEpsilon))
    evaluationEpsilon = np.sqrt(epsilon)
    roots = []

    # Set initial spline, domain, and interval.
    domain = self.domain().T
    intervals = [Interval(self.trim(domain.T).reparametrize(((0.0, 1.0),) * self.nInd), [*range(self.nInd)], 1.0, domain[1] - domain[0], domain[0], epsilon, False)]
    chunkSize = 8
    #pool = Pool() # Pool size matches CPU count

    # Refine all the intervals, collecting roots as we go.
    while intervals:
        nextIntervals = []
        if False and len(intervals) > chunkSize:
            for (newRoots, newIntervals) in pool.imap_unordered(_refine_projected_polyhedron, intervals, chunkSize):
                roots += newRoots
                nextIntervals += newIntervals
        else:
            for (newRoots, newIntervals) in map(_refine_projected_polyhedron, intervals):
                roots += newRoots
                nextIntervals += newIntervals
        intervals = nextIntervals

    # Combine overlapping roots into regions.
    regions = []
    roots.sort(key=lambda root: -root[1]) # Sort widest roots to the front
    for root in roots:
        rootCenter = root[0]
        rootRadius = root[1]

        # Ensure we have a real root (not a boundary special case).
        if np.linalg.norm(self(rootCenter)) >= evaluationEpsilon:
            continue

        # Expand the radius of the root based on the approximate distance from the center needed
        # to raise the value of the spline above evaluationEpsilon.
        jacobian = self.jacobian(rootCenter)
        minEigenvalue = np.sqrt(np.linalg.eigvalsh(jacobian.T @ jacobian)[0])
        if minEigenvalue > epsilon:
            rootRadius = max(rootRadius, evaluationEpsilon / minEigenvalue)
        
        # Intersect this root with the existing regions, expanding and combining them as appropriate.
        firstRegion = None
        for region in regions:
            if region.count == 0:
                continue
            separation = np.linalg.norm(rootCenter - region.center)
            if separation < rootRadius + region.radius + epsilon:
                if firstRegion is None:
                    firstRegion = region
                    firstRegion.center = (region.count * region.center + rootCenter) / (region.count + 1)
                    firstRegion.radius = max(region.radius + separation, \
                        rootRadius + separation * region.count) / (region.count + 1)
                    firstRegion.count += 1
                else:
                    separation = np.linalg.norm(firstRegion.center - region.center)
                    firstRegion.center = (firstRegion.count * firstRegion.center + region.count * region.center) / (firstRegion.count + region.count)
                    firstRegion.radius = max(region.radius + separation * firstRegion.count, \
                        firstRegion.radius + separation * region.count) / (firstRegion.count + region.count)
                    firstRegion.count += region.count
                    region.count = 0
        if firstRegion is None:
            regions.append(_Region(rootCenter, rootRadius, 1))

    # Reconstitute the list of roots from the remaining region centers.
    roots = [region.center for region in regions if region.count > 0]

    # Sort roots if there's only one dimension.
    if self.nInd == 1:
        roots.sort()

    return roots

def contours(self):
    if self.nInd - self.nDep != 1: raise ValueError("The number of free variables (self.nInd - self.nDep) must be one.")

    Point = namedtuple('Point', ('d', 'det', 'onUVBoundary', 'turningPoint', 'uvw'))
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
    # Try different theta values until no border or turning points are degenerate or we run out of attempts.
    attempts = 3
    while attempts > 0:
        points = []
        theta *= 0.607
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        abort = False
        attempts -=1

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
                points.append(Point(d, det, True, False, uvw))
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
                duv = np.linalg.solve(columns, -tangents[nInd](uvw))
                det = np.arctan2((0.5 - boundary) * (duv[0] * cosTheta + duv[1] * sinTheta), (0.5 - boundary) * (duv[0] * cosTheta - duv[1] * sinTheta))
                if abs(det) < epsilon:
                    abort = True
                    break
                points.append(Point(d, det, False, False, uvw))
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
        systemSelf, systemTurningPointDeterminant = bspy.Spline.common_basis((self, turningPointDeterminant))
        system = type(systemSelf)(self.nInd, self.nInd, systemSelf.order, systemSelf.nCoef, systemSelf.knots, \
            np.concatenate((systemSelf.coefs, systemTurningPointDeterminant.coefs)), systemSelf.metadata)
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
            points.append(Point(d, det, False, True, uvw))
        if not abort:
            break # We're done!
    
    if attempts <= 0: raise ValueError("No contours. Degenerate equations.")

    if not points:
        return [] # No contours
    
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
    panel = type(self)(self.nInd, self.nInd, self.order, self.nCoef, self.knots, panelCoefs, self.metadata)

    # Okay, we have everything we need to determine the contour topology and points along each contour.
    # We've done the first two steps of Grandine and Klein's algorithm:
    # (1) Choose theta and find all solutions to (1.6) (system)
    # (2) Find all zeros of f on the boundary of [0, 1]^2

    # Next, sort the edge and turning points by panel distance (d) and then by the determinant (det)
    # (3) Take all the points found in Step (1) and Step (2) and order them by distance in the theta direction from the origin.
    points.sort()

    # Extra step not in paper.
    # Run a checksum on the points, ensuring starting and ending points balance.
    # Start by flipping endpoints as needed, since we can miss turning points near endpoints.
    if points[0].det < 0.0:
        point = points[0]
        points[0] = Point(point.d, -point.det, point.onUVBoundary, point.turningPoint, point.uvw)
    if points[-1].det > 0.0:
        point = points[-1]
        points[-1] = Point(point.d, -point.det, point.onUVBoundary, point.turningPoint, point.uvw)
    checksum = 0
    for i, point in enumerate(points): # Ensure checksum stays non-negative front to back
        checksum += (1 if point.det > 0 else -1) * (2 if point.turningPoint else 1)
    if checksum != 0: raise ValueError("No contours. Inconsistent contour topology.")

    # Extra step not in the paper:
    # Add a panel between two consecutive open/close turning points to uniquely determine contours between them.
    if len(points) > 1:
        i = 0
        previousPoint = points[i]
        while i < len(points) - 1:
            i += 1
            point = points[i]
            if previousPoint.turningPoint and previousPoint.det > 0.0 and \
                point.turningPoint and point.det < 0.0 and \
                point.d - previousPoint.d > epsilon:
                # We have two consecutive open/close turning points on separate panels.
                # Insert a panel in between them, with the uvw value of None, since there is no zero associated.
                points.insert(i, Point(0.5 * (previousPoint.d + point.d), 0.0, False, True, None))
                i += 1
            previousPoint = point

    # (4) Initialize an ordered list of contours. No contours will be on the list at first.
    currentContourPoints = [] # Holds contours (point lists) currently being identified
    contourPoints = [] # Hold contours (point lists) already identified

    # (5) If no points remain to be processed, stop. Otherwise, take the next closest point.
    for point in points:
        # If it is a boundary point, go to Step (6). Otherwise, go to Step (7).
        if point.onUVBoundary:
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
                    currentContourPoints.insert(0, [0, point.uvw]) # 0 indicates no connected contours
                else:
                    currentContourPoints.append([0, point.uvw]) # 0 indicates no connected contours
            else:
                # Ending point
                if abs(point.uvw[0] - 1.0) < epsilon or abs(point.uvw[1]) < epsilon:
                    i = 0
                else:
                    i = len(currentContourPoints) - 1 # Can't use -1, because we manipulate the index below
                fullList = currentContourPoints.pop(i) + [point.uvw]
                connection = fullList.pop(0)
                if connection == 0:
                    contourPoints.append(fullList)
                else:
                    index = i if connection == -1 else i - 1
                    fullList.reverse()
                    currentContourPoints[index] = [0] + fullList + currentContourPoints[index][2:]
        else:
            # (7) Determine whether two contours start or two contours end
            # at the turning point. Locate the two contours in the list of contours by finding
            # all points which lie on both the panel boundary and on the contour. The turning
            # point will be one of these, and it will be well ordered with respect to the other
            # points. Either insert two new contours in the list or delete two existing ones from
            # the list. Go back to Step (5).
            # First, construct panel, whose zeros lie along the panel boundary, u * cosTheta + v * sinTheta - d = 0.
            panel.coefs[self.nDep] -= point.d

            if point.turningPoint and point.uvw is None:
                # For an inserted panel between two consecutive turning points, just find zeros along the panel.
                panelPoints = panel.zeros()
            elif point.turningPoint:
                # Split panel below and above the known zero point.
                # This avoids extra computation and the high-zero at the known zero point, while ensuring we match the turning point.
                panelPoints = [point.uvw]
                # Only split the panel looking for other points if any are expected (> 0 for starting turning point, > 2 for ending one).
                expectedPanelPoints = len(currentContourPoints) - (0 if point.det > 0.0 else 2)
                if expectedPanelPoints > 0:
                    # To split the panel, we need to determine the offset from the point.
                    # Since the objective function (self) is zero and its derivative is zero at the point,
                    # we use second derivatives to determine when the objective function will likely grow 
                    # evaluationEpsilon above zero again.
                    wrt = [0] * self.nInd; wrt[0] = 2
                    selfUU = self.derivative(wrt, point.uvw)
                    wrt[0] = 1; wrt[1] = 1
                    selfUV = self.derivative(wrt, point.uvw)
                    wrt[0] = 0; wrt[1] = 2
                    selfVV = self.derivative(wrt, point.uvw)
                    offset = np.sqrt(2.0 * evaluationEpsilon / \
                        np.linalg.norm(selfUU * sinTheta * sinTheta - 2.0 * selfUV * sinTheta * cosTheta + selfVV * cosTheta * cosTheta))
                    # Now, we can find the zeros of the split panel, checking to ensure each panel is within bounds first.
                    if point.uvw[0] + sinTheta * offset < 1.0 - epsilon and epsilon < point.uvw[1] - cosTheta * offset:
                        panelPoints += panel.trim(((point.uvw[0] + sinTheta * offset, 1.0), (0.0, point.uvw[1] - cosTheta * offset)) + ((None, None),) * (self.nInd - 2)).zeros()
                        expectedPanelPoints -= len(panelPoints) - 1 # Discount the turning point itself
                    if expectedPanelPoints > 0 and epsilon < point.uvw[0] - sinTheta * offset and point.uvw[1] + cosTheta * offset < 1.0 - epsilon:
                        panelPoints += panel.trim(((0.0, point.uvw[0] - sinTheta * offset), (point.uvw[1] + cosTheta * offset, 1.0)) + ((None, None),) * (self.nInd - 2)).zeros()
            else: # It's an other-boundary point.
                # Only find extra zeros along the panel if any are expected (> 0 for starting point, > 1 for ending one).
                expectedPanelPoints = len(currentContourPoints) - (0 if point.det > 0.0 else 1)
                if expectedPanelPoints > 0:
                    panelPoints = panel.zeros()
                    panelPoints.sort(key=lambda uvw: np.linalg.norm(point.uvw - uvw)) # Sort by distance from boundary point
                    while len(panelPoints) > expectedPanelPoints:
                        panelPoints.pop(0) # Drop points closest to the boundary point
                    panelPoints.append(point.uvw)
                else:
                    panelPoints = [point.uvw]

            # Add d back to prepare for next turning point.
            panel.coefs[self.nDep] += point.d
            # Sort zero points by their position along the panel boundary (using vector orthogonal to its normal).
            panelPoints.sort(key=lambda uvw: uvw[1] * cosTheta - uvw[0] * sinTheta)
            # Go through panel points, adding them to existing contours, creating new ones, or closing old ones.
            adjustment = 0 # Adjust index after a contour point is added or removed.
            for i, uvw in zip(range(len(panelPoints)), panelPoints):
                if point.uvw is not None and np.allclose(point.uvw, uvw):
                    if point.det > 0.0:
                        if point.turningPoint:
                            # Insert the turning point twice (second one appears before the first one in the points list).
                            currentContourPoints.insert(i, [1, point.uvw]) # 1 indicates higher connection point
                            currentContourPoints.insert(i, [-1, point.uvw]) # -1 indicates lower connection point
                            adjustment = 1
                        else:
                            # Insert the other-boundary point once.
                            currentContourPoints.insert(i, [0, point.uvw]) # 0 indicates no connected contours
                    else:
                        if point.turningPoint:
                            # Join contours that connect through the turning point.
                            upperHalf = currentContourPoints.pop(i + 1)
                            upperConnection = upperHalf.pop(0)
                            lowerHalf = currentContourPoints.pop(i)
                            lowerConnection = lowerHalf.pop(0)
                            adjustment = -1
                            # Handle all the shape possibilities.
                            if upperConnection == 0 and lowerConnection == 0:
                                # U shape rotated left 90 degrees.
                                upperHalf.reverse()
                                contourPoints.append(lowerHalf + [point.uvw] + upperHalf)
                            elif upperConnection == 0 and lowerConnection != 0:
                                # 2 shape, upper portion.
                                assert lowerConnection == 1
                                index = i if lowerConnection == -1 else i - 1
                                lowerHalf.reverse()
                                currentContourPoints[index] = [upperConnection] + upperHalf + [point.uvw] + lowerHalf + currentContourPoints[index][2:]
                            elif upperConnection != 0 and lowerConnection == 0:
                                # S shape, lower portion.
                                assert upperConnection == -1
                                index = i if upperConnection == -1 else i - 1
                                upperHalf.reverse()
                                currentContourPoints[index] = [lowerConnection] + lowerHalf + [point.uvw] + upperHalf + currentContourPoints[index][2:]
                            elif upperConnection == 1 and lowerConnection == -1:
                                # O shape.
                                upperHalf.reverse()
                                contourPoints.append(lowerHalf + [point.uvw] + upperHalf)
                            elif upperConnection == 1 and lowerConnection == 1:
                                # C shape, upper portion.
                                index = i if lowerConnection == -1 else i - 1
                                lowerHalf.reverse()
                                currentContourPoints[index] = [upperConnection] + upperHalf + [point.uvw] + lowerHalf + currentContourPoints[index][2:]
                            elif upperConnection == -1 and lowerConnection == -1:
                                # C shape, lower portion.
                                index = i if upperConnection == -1 else i - 1
                                upperHalf.reverse()
                                currentContourPoints[index] = [lowerConnection] + lowerHalf + [point.uvw] + upperHalf + currentContourPoints[index][2:]
                            else: # upperConnection == -1 and lowerConnection == 1
                                # M shape rotated left 90 degrees
                                assert upperConnection == -1
                                assert lowerConnection == 1
                                index = i if lowerConnection == -1 else i - 1
                                lowerHalf.reverse()
                                currentContourPoints[index] = [upperConnection] + upperHalf + [point.uvw] + lowerHalf + currentContourPoints[index][2:]
                        else: 
                            # It's an ending point on an other boundary (same steps as uv boundary).
                            adjustment = -1
                            fullList = currentContourPoints.pop(i) + [point.uvw]
                            connection = fullList.pop(0)
                            if connection == 0:
                                contourPoints.append(fullList)
                            else:
                                index = i if connection == -1 else i - 1
                                fullList.reverse()
                                currentContourPoints[index] = [0] + fullList + currentContourPoints[index][2:]
                else:
                    currentContourPoints[i + adjustment].append(uvw)

    # We've determined a bunch of points along all the contours, including starting and ending points.
    # Now we just need to create splines for those contours using the bspy.Spline.contour method.
    splineContours = []
    for points in contourPoints:
        contour = bspy.Spline.contour(self, points)
        # Transform the contour to self's original domain.
        contour.coefs = (contour.coefs.T * (domain[1] - domain[0]) + domain[0]).T
        splineContours.append(contour)
    
    return splineContours

def intersect(self, other):
    intersections = []
    nDep = self.nInd # The dimension of the intersection's range

    # Spline-Hyperplane intersection.
    if isinstance(other, Hyperplane):
        # Compute the projection onto the hyperplane to map Spline-Hyperplane intersection points to the domain of the Hyperplane.
        projection = np.linalg.inv(other._tangentSpace.T @ other._tangentSpace) @ other._tangentSpace.T
        # Construct a new spline that represents the intersection.
        spline = self.dot(other._normal) - np.atleast_1d(np.dot(other._normal, other._point))

        # Curve-Line intersection.
        if nDep == 1:
            # Find the intersection points and intervals.
            zeros = spline.zeros()
            # Convert each intersection point into a Manifold.Crossing and each intersection interval into a Manifold.Coincidence.
            for zero in zeros:
                if isinstance(zero, tuple):
                    # Intersection is an interval, so create a Manifold.Coincidence.
                    planeBounds = (projection @ (self((zero[0],)) - other._point), projection @ (self((zero[1],)) - other._point))

                    # First, check for crossings at the boundaries of the coincidence, since splines can have discontinuous tangents.
                    # We do this first because later we may change the order of the plane bounds.
                    (bounds,) = self.domain()
                    epsilon = 0.1 * Manifold.minSeparation
                    if zero[0] - epsilon > bounds[0]:
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[0] - epsilon, 0.0), Hyperplane(1.0, planeBounds[0], 0.0)))
                    if zero[1] + epsilon < bounds[1]:
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[1] + epsilon, 0.0), Hyperplane(1.0, planeBounds[1], 0.0)))

                    # Now, create the coincidence.
                    left = Solid(nDep, False)
                    left.add_boundary(Boundary(Hyperplane(-1.0, zero[0], 0.0), Solid(0, True)))
                    left.add_boundary(Boundary(Hyperplane(1.0, zero[1], 0.0), Solid(0, True)))
                    right = Solid(nDep, False)
                    if planeBounds[0] > planeBounds[1]:
                        planeBounds = (planeBounds[1], planeBounds[0])
                    right.add_boundary(Boundary(Hyperplane(-1.0, planeBounds[0], 0.0), Solid(0, True)))
                    right.add_boundary(Boundary(Hyperplane(1.0, planeBounds[1], 0.0), Solid(0, True)))
                    alignment = np.dot(self.normal((zero[0],)), other._normal) # Use the first zero, since B-splines are closed on the left
                    width = zero[1] - zero[0]
                    transform = (planeBounds[1] - planeBounds[0]) / width
                    translation = (planeBounds[0] * zero[1] - planeBounds[1] * zero[0]) / width
                    intersections.append(Manifold.Coincidence(left, right, alignment, np.atleast_2d(transform), np.atleast_2d(1.0 / transform), np.atleast_1d(translation)))
                else:
                    # Intersection is a point, so create a Manifold.Crossing.
                    intersections.append(Manifold.Crossing(Hyperplane(1.0, zero, 0.0), Hyperplane(1.0, projection @ (self((zero,)) - other._point), 0.0)))

        # Surface-Plane intersection.
        elif nDep == 2:
            # Find the intersection contours, which are returned as splines.
            contours = spline.contours()
            # Convert each contour into a Manifold.Crossing.
            for contour in contours:
                # The left portion is the contour returned for the spline-plane intersection. 
                left = contour
                # The right portion is the contour projected onto the plane's domain, which we compute with samples and a least squares fit.
                tValues = np.linspace(0.0, 1.0, contour.nCoef[0] + 5) # Over-sample a bit to reduce the condition number and avoid singular matrix
                points = []
                for t in tValues:
                    zero = contour((t,))
                    points.append(projection @ (self(zero) - other._point))
                right = bspy.Spline.least_squares(tValues, np.array(points).T, contour.order, contour.knots)
                intersections.append(Manifold.Crossing(left, right))
        else:
            return NotImplemented
    
    # Spline-Spline intersection.
    elif isinstance(other, bspy.Spline):
        # Construct a new spline that represents the intersection.
        spline = self.subtract(other)

        # Curve-Curve intersection.
        if nDep == 1:
            # Find the intersection points and intervals.
            zeros = spline.zeros()
            # Convert each intersection point into a Manifold.Crossing and each intersection interval into a Manifold.Coincidence.
            for zero in zeros:
                if isinstance(zero, tuple):
                    # Intersection is an interval, so create a Manifold.Coincidence.

                    # First, check for crossings at the boundaries of the coincidence, since splines can have discontinuous tangents.
                    # We do this first to match the approach for Curve-Line intersections.
                    (boundsSelf,) = self.domain()
                    (boundsOther,) = other.domain()
                    epsilon = 0.1 * Manifold.minSeparation
                    if zero[0][0] - epsilon > boundsSelf[0]:
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[0][0] - epsilon, 0.0), Hyperplane(1.0, zero[0][1], 0.0)))
                    elif zero[0][1] - epsilon > boundsOther[0]:
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[0][0], 0.0), Hyperplane(1.0, zero[0][1] - epsilon, 0.0)))
                    if zero[1][0] + epsilon < boundsSelf[1]:
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[1][0] + epsilon, 0.0), Hyperplane(1.0, zero[1][1], 0.0)))
                    elif zero[1][1] + epsilon < boundsOther[1]:
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[1][0], 0.0), Hyperplane(1.0, zero[1][1] + epsilon, 0.0)))

                    # Now, create the coincidence.
                    left = Solid(nDep, False)
                    left.add_boundary(Boundary(Hyperplane(-1.0, zero[0][0], 0.0), Solid(0, True)))
                    left.add_boundary(Boundary(Hyperplane(1.0, zero[1][0], 0.0), Solid(0, True)))
                    right = Solid(nDep, False)
                    right.add_boundary(Boundary(Hyperplane(-1.0, zero[0][1], 0.0), Solid(0, True)))
                    right.add_boundary(Boundary(Hyperplane(1.0, zero[1][1], 0.0), Solid(0, True)))
                    alignment = np.dot(self.normal(zero[0][0]), other.normal(zero[0][1])) # Use the first zeros, since B-splines are closed on the left
                    width = zero[1][0] - zero[0][0]
                    transform = (zero[1][1] - zero[0][1]) / width
                    translation = (zero[0][1] * zero[1][0] - zero[1][1] * zero[0][0]) / width
                    intersections.append(Manifold.Coincidence(left, right, alignment, np.atleast_2d(transform), np.atleast_2d(1.0 / transform), np.atleast_1d(translation)))
                else:
                    # Intersection is a point, so create a Manifold.Crossing.
                    intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[:nDep], 0.0), Hyperplane(1.0, zero[nDep:], 0.0)))
        
        # Surface-Surface intersection.
        elif nDep == 2:
            logging.info(f"intersect({self.metadata['Name']}, {other.metadata['Name']})")
            # Find the intersection contours, which are returned as splines.
            swap = False
            try:
                # First try the intersection as is.
                contours = spline.contours()
            except ValueError:
                # If that fails, swap the manifolds. Worth a shot since intersections are touchy.
                swap = True

            # Convert each contour into a Manifold.Crossing.
            if swap:
                spline = other.subtract(self)
                logging.info(f"intersect({other.metadata['Name']}, {self.metadata['Name']})")
                contours = spline.contours()
                for contour in contours:
                    # Swap left and right, compared to not swapped.
                    left = bspy.Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.metadata)
                    right = bspy.Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.metadata)
                    intersections.append(Manifold.Crossing(left, right))
            else:
                for contour in contours:
                    left = bspy.Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.metadata)
                    right = bspy.Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.metadata)
                    intersections.append(Manifold.Crossing(left, right))
        else:
            return NotImplemented
    else:
        return NotImplemented

    # Ensure the normals point outwards for both Manifolds in each crossing intersection.
    # Note that evaluating left and right at 0.5 is always valid because either they are points or curves with [0.0, 1.0] domains.
    domainPoint = np.atleast_1d(0.5)
    for i, intersection in enumerate(intersections):
        if isinstance(intersection, Manifold.Crossing):
            left = intersection.left
            right = intersection.right
            if np.dot(self.tangent_space(left.evaluate(domainPoint)) @ left.normal(domainPoint), other.normal(right.evaluate(domainPoint))) < 0.0:
                left = left.flip_normal()
            if np.dot(other.tangent_space(right.evaluate(domainPoint)) @ right.normal(domainPoint), self.normal(left.evaluate(domainPoint))) < 0.0:
                right = right.flip_normal()
            intersections[i] = Manifold.Crossing(left, right)

    return intersections

def establish_domain_bounds(domain, bounds):
    """
    Establish the outer bounds of a spline's domain (creates a hypercube based on the spline's bounds).

    Parameters
    ----------
    domain : `solid.Solid`
        The domain of the spline into which boundaries should be added based on the spline's bounds.

    bounds : array-like
        nInd x 2 array of the lower and upper bounds on each of the independent variables.

    See Also
    --------
    `solid.Solid.slice` : slice the solid by a manifold.
    `complete_slice` : Add any missing inherent (implicit) boundaries of this manifold's domain to the given slice.
    """
    dimension = len(bounds)
    assert len(bounds[0]) == 2
    assert domain.dimension == dimension
    domain.containsInfinity = False
    for i in range(dimension):
        if dimension > 1:
            domainDomain1 = Solid(dimension - 1, False)
            establish_domain_bounds(domainDomain1, np.delete(bounds, i, axis=0))
            domainDomain2 = Solid(dimension - 1, False)
            establish_domain_bounds(domainDomain2, np.delete(bounds, i, axis=0))
        else:
            domainDomain1 = Solid(0, True)
            domainDomain2 = Solid(0, True)
        diagonal = np.identity(dimension)
        unitVector = diagonal[i]
        if dimension > 1:
            tangentSpace = np.delete(diagonal, i, axis=1)
        else:
            tangentSpace = np.array([0.0])
        hyperplane = Hyperplane(-unitVector, bounds[i][0] * unitVector, tangentSpace)
        domain.add_boundary(Boundary(hyperplane, domainDomain1))
        hyperplane = Hyperplane(unitVector, bounds[i][1] * unitVector, tangentSpace)
        domain.add_boundary(Boundary(hyperplane, domainDomain2))

def complete_slice(self, slice, solid):
    # Spline manifold domains have finite bounds.
    slice.containsInfinity = False
    bounds = self.domain()

    # If manifold (self) has no intersections with solid, just check containment.
    if not slice.boundaries:
        if slice.dimension == 2:
            logging.info(f"check containment: {self.metadata['Name']}")
        domain = bounds.T
        if solid.contains_point(self(0.5 * (domain[0] + domain[1]))):
            for boundary in Hyperplane.create_hypercube(bounds).boundaries:
                slice.add_boundary(boundary)
        return

    # For curves, add domain bounds as needed.
    if slice.dimension == 1:
        slice.boundaries.sort(key=lambda b: (b.manifold.evaluate(0.0), b.manifold.normal(0.0)))
        # First, check right end since we add new boundary to the end.
        if abs(slice.boundaries[-1].manifold._point - bounds[0][1]) >= Manifold.minSeparation and \
            slice.boundaries[-1].manifold._normal < 0.0:
            slice.add_boundary(Boundary(Hyperplane(-slice.boundaries[-1].manifold._normal, bounds[0][1], 0.0), Solid(0, True)))
        # Next, check left end since it's still untouched.
        if abs(slice.boundaries[0].manifold._point - bounds[0][0]) >= Manifold.minSeparation and \
            slice.boundaries[0].manifold._normal > 0.0:
            slice.add_boundary(Boundary(Hyperplane(-slice.boundaries[0].manifold._normal, bounds[0][0], 0.0), Solid(0, True)))

    # For surfaces, add bounding box for domain and intersect it with existing slice boundaries.
    if slice.dimension == 2:
        boundaryCount = len(slice.boundaries) # Keep track of existing slice boundaries
        establish_domain_bounds(slice, bounds) # Add bounding box boundaries to slice boundaries
        for boundary in slice.boundaries[boundaryCount:]: # Mark bounding box boundaries as untouched
            boundary.touched = False

        # Define function for adding slice points to new bounding box boundaries.
        def process_domain_point(boundary, domainPoint):
            point = boundary.manifold.evaluate(domainPoint)
            # See if and where point touches bounding box of slice.
            for newBoundary in slice.boundaries[boundaryCount:]:
                vector = point - newBoundary.manifold._point
                if abs(np.dot(newBoundary.manifold._normal, vector)) < Manifold.minSeparation:
                    # Add the point onto the new boundary.
                    normal = np.sign(newBoundary.manifold._tangentSpace.T @ boundary.manifold.normal(domainPoint))
                    newBoundary.domain.add_boundary(Boundary(Hyperplane(normal, newBoundary.manifold._tangentSpace.T @ vector, 0.0), Solid(0, True)))
                    newBoundary.touched = True
                    break

        # Go through existing boundaries and check if either of their endpoints lies on the spline's bounds.
        for boundary in slice.boundaries[:boundaryCount]:
            domainBoundaries = boundary.domain.boundaries
            domainBoundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), boundary.manifold.normal(0.0)))
            process_domain_point(boundary, domainBoundaries[0].manifold._point)
            if len(domainBoundaries) > 1:
                process_domain_point(boundary, domainBoundaries[-1].manifold._point)
        
        # For touched boundaries, remove domain bounds that aren't needed.
        boundaryWasTouched = False
        for newBoundary in slice.boundaries[boundaryCount:]:
            if newBoundary.touched:
                boundaryWasTouched = True
                domainBoundaries = newBoundary.domain.boundaries
                assert len(domainBoundaries) > 2
                domainBoundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), boundary.manifold.normal(0.0)))
                # Ensure domain endpoints don't overlap and their normals are consistent.
                if abs(domainBoundaries[0].manifold._point - domainBoundaries[1].manifold._point) < Manifold.minSeparation or \
                    domainBoundaries[1].manifold._normal < 0.0:
                    del domainBoundaries[0]
                if abs(domainBoundaries[-1].manifold._point - domainBoundaries[-2].manifold._point) < Manifold.minSeparation or \
                    domainBoundaries[-2].manifold._normal > 0.0:
                    del domainBoundaries[-1]
        
        if boundaryWasTouched:
            # Touch untouched boundaries that are connected to touched boundary endpoints.
            boundaryMap = ((2, 3, 0), (2, 3, -1), (0, 1, 0), (0, 1, -1)) # Map of which bounding box boundaries touch each other
            while True:
                noTouches = True
                for map, newBoundary, bound in zip(boundaryMap, slice.boundaries[boundaryCount:], bounds.flatten()):
                    if not newBoundary.touched:
                        leftBoundary = slice.boundaries[boundaryCount + map[0]]
                        rightBoundary = slice.boundaries[boundaryCount + map[1]]
                        if leftBoundary.touched and abs(leftBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                            newBoundary.touched = True
                            noTouches = False
                        elif rightBoundary.touched and abs(rightBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                            newBoundary.touched = True
                            noTouches = False
                if noTouches:
                    break
            
            # Remove untouched boundaries.
            i = boundaryCount
            while i < len(slice.boundaries):
                if not slice.boundaries[i].touched:
                    del slice.boundaries[i]
                else:
                    i += 1
        else:
            # No slice boundaries touched the bounding box, so remove bounding box if it's not contained in the solid.
            if not solid.contains_point(self.evaluate(bounds[:,0])):
                slice.boundaries = slice.boundaries[:boundaryCount]

def full_domain(self):
    return Hyperplane.create_hypercube(self.domain())