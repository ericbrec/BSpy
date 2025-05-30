import logging
import math
import numpy as np
from bspy.manifold import Manifold
from bspy.hyperplane import Hyperplane
import bspy.spline
import bspy.spline_block
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
            provisionalZero = [0.5 * (projectedLeftStep + projectedRightStep)]
            if projectedLeftStep <= projectedRightStep:
                if projectedRightStep - projectedLeftStep <= epsilon:
                    myZeros = provisionalZero
                else:
                    trimmedSpline = mySpline.trim(((projectedLeftStep, projectedRightStep),))
                    myZeros = refine(trimmedSpline, intervalSize, functionMax)
                    if len(myZeros) == 0 and mySpline.order[0] == mySpline.nCoef[0] and \
                       mySpline.coefs[0][0] * mySpline.coefs[0][-1] < 0.0:
                        myZeros = provisionalZero
            else:
               if mySpline.order[0] == mySpline.nCoef[0] and \
                  mySpline.coefs[0][0] * mySpline.coefs[0][-1] < 0.0:
                   myZeros = provisionalZero
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

def _convex_hull_2D(xData, yData, yBounds, yOtherBounds):
    # Allow xData to be repeated for longer yData, but only if yData is a multiple.
    if not(yData.shape[0] % xData.shape[0] == 0): raise ValueError("Size of xData does not divide evenly in size of yData")
    yData = np.reshape(yData, (yData.shape[0] // xData.shape[0], xData.shape[0]))

    # Calculate y adjustment as needed for values close to zero
    yMinAdjustment = -yBounds[0] if yBounds[0] > 0.0 else 0.0
    yMaxAdjustment = -yBounds[1] if yBounds[1] < 0.0 else 0.0
    yMinAdjustment += yOtherBounds[0]
    yMaxAdjustment += yOtherBounds[1]

    # Calculate the yMin and yMax arrays corresponding to xData
    yMin = np.min(yData, axis = 0) + yMinAdjustment
    yMax = np.max(yData, axis = 0) + yMaxAdjustment

    # Initialize lower and upper hulls
    lowerHull = [[xData[0], yMin[0]], [xData[1], yMin[1]]]
    upperHull = [[xData[0], yMax[0]], [xData[1], yMax[1]]]

    # Add additional lower points one at a time, throwing out intermediates if necessary
    for xNext, yNext in zip(xData[2:], yMin[2:]):
        lowerHull.append([xNext, yNext])
        while len(lowerHull) > 2 and \
                (lowerHull[-2][0] - lowerHull[-3][0]) * (lowerHull[-1][1] - lowerHull[-2][1]) <= \
                (lowerHull[-1][0] - lowerHull[-2][0]) * (lowerHull[-2][1] - lowerHull[-3][1]):
            del lowerHull[-2]

    # Do the same for the upper points
    for xNext, yNext in zip(xData[2:], yMax[2:]):
        upperHull.append([xNext, yNext])
        while len(upperHull) > 2 and \
                (upperHull[-2][0] - upperHull[-3][0]) * (upperHull[-1][1] - upperHull[-2][1]) >= \
                (upperHull[-1][0] - upperHull[-2][0]) * (upperHull[-2][1] - upperHull[-3][1]):
            del upperHull[-2]

    # Return the two hulls
    return lowerHull, upperHull

def _intersect_convex_hull_with_x_interval(lowerHull, upperHull, epsilon, xInterval):
    xMin = xInterval[0]
    xMax = xInterval[1]
    sign = -1.0
    for hull in [lowerHull, upperHull]:
        sign = -sign
        p0 = hull[0]
        for p1 in hull[1:]:
            yDelta = p0[1] - p1[1]
            if p0[1] * p1[1] <= 0.0 and yDelta != 0.0:
                yDelta = p0[1] - p1[1]
                alpha = p0[1] / yDelta
                xNew = p0[0] * (1.0 - alpha) + p1[0] * alpha
                if sign * yDelta > 0.0:
                    xMin = max(xMin, xNew - epsilon)
                else:
                    xMax = min(xMax, xNew + epsilon)
            p0 = p1
    if xMin > xMax:
        return None
    else:
        return [xMin, xMax]

Interval = namedtuple('Interval', ('block', 'active', 'split', 'scale', 'bounds', 'xLeft', 'xRight', 'epsilon', 'atMachineEpsilon'))

def _create_interval(block, active, split, scale, xLeft, xRight, epsilon):
    nDep = 0
    nInd = len(scale)
    bounds = np.zeros((nInd, 2), scale.dtype)
    newScale = np.empty_like(scale)
    newBlock = []
    for row in block:
        newRow = []
        # Reparametrize splines and sum bounds
        for map, spline in row:
            spline = spline.reparametrize(((0.0, 1.0),) * spline.nInd)
            bounds[nDep:nDep + spline.nDep] += spline.range_bounds()
            newRow.append((map, spline))
        newBlock.append(newRow)

        # Check row bounds for potential roots.
        for dep in range(spline.nDep):
            coefsMin = bounds[nDep, 0] * scale[nDep]
            coefsMax = bounds[nDep, 1] * scale[nDep]
            if coefsMax < -epsilon or coefsMin > epsilon:
                # No roots in this interval.
                return None
            newScale[nDep] = max(-coefsMin, coefsMax)
            # Rescale spline coefficients to max 1.0.
            rescale = 1.0 / max(-bounds[nDep, 0], bounds[nDep, 1])
            for map, spline in newRow:
                spline.coefs[dep] *= rescale
            bounds[nDep] *= rescale
            nDep += 1

    for iInd in range(nInd):
        newSplit = (split + iInd + 1) % nInd
        if active[newSplit]:
            return Interval(newBlock, active, newSplit, newScale, bounds, xLeft, xRight, epsilon, np.dot(xRight - xLeft, xRight - xLeft) < np.finfo(xLeft.dtype).eps)

    # No active variables left
    return None

# We use multiprocessing.Pool to call this function in parallel, so it cannot be nested and must take a single argument.
def _refine_projected_polyhedron(interval):
    Crit = 0.85 # Required percentage decrease in domain per iteration.
    epsilon = interval.epsilon
    roots = []
    intervals = []
    
    # Explore given independent variable to determine a tighter domain around roots.
    xInterval = [0.0, 1.0]
    iInd = interval.split
    nDep = 0
    for row in interval.block:
        order = 0
        for map, spline in row:
            if iInd in map:
                ind = map.index(iInd)
                order = spline.order[ind]
                # Move independent variable to the last (fastest) axis, adding 1 to account for the dependent variables.
                coefs = np.moveaxis(spline.coefs, ind + 1, -1)
                break
            
        # Skip this row if it doesn't contain this independent variable.
        if order < 1:
            nDep += spline.nDep # Assumes there is at least one spline per block row
            continue

        # Compute the coefficients for f(x) = x for the independent variable and its knots.
        xData = spline.greville(ind)
        
        # Loop through each dependent variable in this row to refine the interval containing the root for this independent variable.
        for yData, ySplineBounds, yBounds in zip(coefs, spline.range_bounds(),
                                                 interval.bounds[nDep:nDep + spline.nDep]):
            # Compute the 2D convex hull of the knot coefficients and the spline's coefficients
            lowerHull, upperHull = _convex_hull_2D(xData, yData.ravel(), yBounds, yBounds - ySplineBounds)
            if lowerHull is None or upperHull is None:
                return roots, intervals
            
            # Intersect the convex hull with the xInterval along the x axis (the knot coefficients axis).
            xInterval = _intersect_convex_hull_with_x_interval(lowerHull, upperHull, epsilon, xInterval)
            if xInterval is None:
                return roots, intervals
            
        nDep += spline.nDep
    
    # Compute new interval bounds.

    xNewLeft = interval.xLeft.copy()
    xNewRight = interval.xRight.copy()
    xNewLeft[iInd] = (1.0 - xInterval[0]) * interval.xLeft[iInd] + xInterval[0] * interval.xRight[iInd]
    xNewRight[iInd] = (1.0 - xInterval[1]) * interval.xLeft[iInd] + xInterval[1] * interval.xRight[iInd]
    newActive = interval.active.copy()
    newActive[iInd] = (xNewRight[iInd] - xNewLeft[iInd] >= epsilon)
    nInd = 0
    for active in newActive:
        if active:
            nInd += 1

    # Iteration is complete if the interval actual width is either
    # one iteration past being less than sqrt(machineEpsilon) or there are no remaining independent variables.
    if interval.atMachineEpsilon or nInd == 0:
        # Return the interval center and radius.
        roots.append((0.5 * (xNewLeft + xNewRight), epsilon))
        return roots, intervals

    # Split domain if not sufficient decrease in width
    width = xInterval[1] - xInterval[0]
    domains = [xInterval]
    if width > Crit:
        # Didn't get the required decrease in width, so split the domain.
        leftDomain = xInterval
        rightDomain = xInterval.copy()
        leftDomain[1] = 0.5 * (leftDomain[0] + leftDomain[1])
        rightDomain[0] = leftDomain[1]
        domains = [leftDomain, rightDomain]

    # Add new intervals to interval stack.
    for domain in domains:
        xSplitLeft = xNewLeft.copy()
        xSplitRight = xNewRight.copy()
        xSplitLeft[iInd] = (1.0 - domain[0]) * interval.xLeft[iInd] + domain[0] * interval.xRight[iInd]
        xSplitRight[iInd] = (1.0 - domain[1]) * interval.xLeft[iInd] + domain[1] * interval.xRight[iInd]
        newBlock = []
        for row in interval.block:
            newRow = []
            # Trim splines
            for map, spline in row:
                trimRegion = [(0.0, 1.0) for i in range(spline.nInd)]
                if iInd in map:
                    ind = map.index(iInd)
                    trimRegion[ind] = domain
                spline = spline.trim(trimRegion)
                newRow.append((map, spline))
            newBlock.append(newRow)
        newInterval = _create_interval(newBlock, newActive, iInd,
                                       interval.scale, xSplitLeft, xSplitRight, epsilon)
        if newInterval:
            if newInterval.block:
                intervals.append(newInterval)
            else:
                roots.append((0.5 * (newInterval.xLeft + newInterval.xRight),
                              0.5 * np.linalg.norm(newInterval.xRight - newInterval.xLeft)))
  
    return roots, intervals

class _Region:
    def __init__(self, center, radius, count):
        self.center = center
        self.radius = radius
        self.count = count

def zeros_using_projected_polyhedron(self, epsilon=None, initialScale=None):
    if self.nInd != self.nDep: raise ValueError("The number of independent variables (nInd) must match the number of dependent variables (nDep).")

    # Determine epsilon and initialize roots.
    machineEpsilon = np.finfo(self.knotsDtype).eps
    if epsilon is None:
        epsilon = 0.0
    epsilon = max(epsilon, np.sqrt(machineEpsilon)) if epsilon else np.sqrt(machineEpsilon)
    evaluationEpsilon = max(np.sqrt(epsilon), np.finfo(self.coefsDtype).eps ** 0.25)
    intervals = []
    roots = []

    # Set initial interval.
    domain = self.domain().T
    initialScale = np.full(self.nDep, 1.0, self.coefsDtype) if initialScale is None else np.array(initialScale, self.coefsDtype)
    newInterval = _create_interval(self.block, self.nInd * [True], -1, initialScale,
                                   domain[0], domain[1], epsilon)
    if newInterval:
        if newInterval.block:
            intervals.append(newInterval)
        else:
            roots.append(0.5 * (newInterval.xLeft + newInterval.xRight),
                         0.5 * np.linalg.norm(newInterval.xRight - newInterval.xLeft))

    # Refine all the intervals, collecting roots as we go.
    while intervals:
        interval = intervals.pop()
        newRoots, newIntervals = _refine_projected_polyhedron(interval)
        roots += newRoots
        newIntervals.reverse()
        intervals += newIntervals

    # Combine overlapping roots into regions.
    regions = []
    roots.sort(key=lambda root: -root[1]) # Sort widest roots to the front
    for root in roots:
        rootCenter = root[0]
        rootRadius = root[1]

        # Take one Newton step on each root
        value = self.evaluate(rootCenter)
        residualNorm = np.linalg.norm(value)
        try:
            update = np.linalg.solve(self.jacobian(rootCenter), value)
            if np.linalg.norm(update) < rootRadius:
                rootCenter -= update
        except:
            pass

        # Project back onto spline domain
        selfDomain = self.domain()
        rootCenter = np.maximum(np.minimum(rootCenter, selfDomain.T[1]), selfDomain.T[0])
        value = self.evaluate(rootCenter)
        newResidualNorm = np.linalg.norm(value)
        rootRadius *= newResidualNorm / residualNorm
        residualNorm = newResidualNorm

        # Ensure we have a real root (not a boundary special case).
        if residualNorm >= evaluationEpsilon:
            continue

        # Expand the radius of the root based on the approximate distance from the center needed
        # to raise the value of the spline above evaluationEpsilon.
        minSingularValue = np.linalg.svd(self.jacobian(rootCenter), False, False)[-1]
        if minSingularValue > epsilon:
            rootRadius = max(rootRadius, evaluationEpsilon / minSingularValue)
        
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

def _turning_point_determinant(self, uvw, cosTheta, sinTheta):
    sign = -1 if hasattr(self, "metadata") and self.metadata.get("flipNormal", False) else 1
    tangentSpace = self.jacobian(uvw).T
    return cosTheta * sign * np.linalg.det(tangentSpace[[j for j in range(self.nInd) if j != 0]]) - \
        sinTheta * sign * np.linalg.det(tangentSpace[[j for j in range(self.nInd) if j != 1]])

def _turning_point_determinant_gradient(self, uvw, cosTheta, sinTheta):
    dtype = self.coefs.dtype if hasattr(self, "coefs") else self.coefsDtype
    gradient = np.zeros(self.nInd, dtype)

    sign = -1 if hasattr(self, "metadata") and self.metadata.get("flipNormal", False) else 1
    tangentSpace = self.jacobian(uvw).T
    dTangentSpace = tangentSpace.copy()

    wrt = [0] * self.nInd
    for i in range(self.nInd):
        wrt[i] = 1
        for j in range(self.nInd):
            wrt[j] = 1 if i != j else 2
            dTangentSpace[j, :] = self.derivative(wrt, uvw) # tangentSpace and dTangentSpace are the transpose of the jacobian
            gradient[i] += cosTheta * sign * np.linalg.det(dTangentSpace[[k for k in range(self.nInd) if k != 0]]) - \
                sinTheta * sign * np.linalg.det(dTangentSpace[[k for k in range(self.nInd) if k != 1]])
            dTangentSpace[j, :] = tangentSpace[j, :] # tangentSpace and dTangentSpace are the transpose of the jacobian
            wrt[j] = 0 if i != j else 1
        wrt[i] = 0
    
    return gradient

def _contours_of_C1_spline_block(self, epsilon, evaluationEpsilon):
    Point = namedtuple('Point', ('d', 'det', 'onUVBoundary', 'turningPoint', 'uvw'))

    # Go through each nDep of the spline block, checking bounds.
    bounds = self.range_bounds()
    for bound in bounds:
        if bound[1] < -evaluationEpsilon or bound[0] > evaluationEpsilon:
            # No contours for this spline.
            return []

    # Record self's original domain and then reparametrize self's domain to [0, 1]^nInd.
    domain = self.domain().T
    self = self.reparametrize(((0.0, 1.0),) * self.nInd)

    # Rescale self in all dimensions.
    initialScale = np.max(np.abs(bounds), axis=1)
    rescale = np.reciprocal(initialScale)
    nDep = 0
    for row in self.block:
        for map, spline in row:
            for coefs, scale in zip(spline.coefs, rescale[nDep:nDep + spline.nDep]):
                coefs *= scale
        nDep += spline.nDep

    # Try arbitrary values for theta between [0, pi/2] that are unlikely to be a stationary points.
    for theta in (1.0 / np.sqrt(2), np.pi / 6.0, 1.0/ np.e):
        points = []
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        abort = False

        # Find intersections with u and v boundaries.
        def uvIntersections(nInd, boundary):
            zeros = self.contract([None] * nInd + [boundary] + [None] * (self.nInd - nInd - 1)).zeros(epsilon, initialScale)
            abort = False
            for zero in zeros:
                if isinstance(zero, tuple):
                    abort = True
                    break
                uvw = np.insert(np.array(zero), nInd, boundary)
                d = uvw[0] * cosTheta + uvw[1] * sinTheta
                n = self.normal(uvw, False, (0, 1))
                tpd = _turning_point_determinant(self, uvw, cosTheta, sinTheta)
                det = (0.5 - boundary) * n[nInd] * tpd
                if abs(det) < epsilon:
                    abort = True
                    break
                # Check for literal corner case.
                otherInd = 1 - nInd
                otherValue = uvw[otherInd]
                if otherValue < epsilon or otherValue + epsilon > 1.0:
                    otherDet = (0.5 - otherValue) * n[otherInd] * tpd
                    if det * otherDet < 0.0:
                        continue # Corner that starts and ends, ignore it
                    elif max(otherValue, boundary) < epsilon and det < 0.0:
                        continue # End point at (0, 0), ignore it
                    elif min(otherValue, boundary) + epsilon > 1.0 and det > 0.0:
                        continue # Start point at (1, 1), ignore it
                # Append boundary point.
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
            zeros = self.contract([None] * nInd + [boundary] + [None] * (self.nInd - nInd - 1)).zeros(epsilon, initialScale)
            abort = False
            for zero in zeros:
                if isinstance(zero, tuple):
                    abort = True
                    break
                uvw = np.insert(np.array(zero), nInd, boundary)
                d = uvw[0] * cosTheta + uvw[1] * sinTheta
                columns = np.empty((self.nDep, self.nInd - 1))
                tangents = self.jacobian(uvw).T
                i = 0
                for j in range(self.nInd):
                    if j != nInd:
                        columns[:, i] = tangents[j]
                        i += 1
                duv = np.linalg.solve(columns, -tangents[nInd])
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

        # First, add the null space constraint to the system: dot(self's gradient, (r * sinTheta, -r * cosTheta, c, d, ...) = 0.
        # This introduces self.nInd - 1 new independent variables: r, c, d, ...
        turningPointBlock = self.block.copy()
        if self.nInd > 2:
            rSpline = bspy.Spline(1, 1, (2,), (2,), ((0.0, 0.0, 1.0, 1.0),), ((0.0, 1.0),))
        else:
            rSpline = bspy.Spline.point([1.0])
        otherSpline = bspy.Spline(1, 1, (2,), (2,), ((-1.0, -1.0, 1.0, 1.0),), ((-1.0, 1.0),))
        # Track indices of other independent variables (c, d, ...).
        otherNInd = self.nInd + 1 # Add one since r is always the first new variable (index for r is self.nInd)
        otherDictionary = {}
        # Go through each row building the null space constraint.
        for row in self.block:
            newRow = []
            for map, spline in row:
                newSpline = None # The spline's portion of the null space constraint starts with None
                newMap = map.copy() # The map for spline's contribution to the null space constraint starts with its existing map
                # Create addition indMap with existing independent variables for use in summing the dot product.
                indMapForAdd = [(index, index) for index in range(spline.nInd)]
                rIndex = None # Index of r in newSpline, which we need to track since rSpline may be added twice

                # Add each term of spline's contribution to dot(self's gradient, (r * sinTheta, -r * cosTheta, c, d, ...). 
                for i in range(spline.nInd):
                    dSpline = spline.differentiate(i)
                    nInd = map[i]
                    if nInd < 2:
                        factor = sinTheta if nInd == 0 else -cosTheta
                        term = dSpline.multiply(factor * rSpline)
                        if rIndex is None:
                            # Adding rSpline for the first time, so add r to newMap and track its index.
                            newMap.append(self.nInd)
                            newSpline = term if newSpline is None else newSpline.add(term, indMapForAdd)
                            rIndex = newSpline.nInd - 1
                        else:
                            # The same rSpline is being added again, so enhance the indMapForAdd to associate the two rSplines.
                            newSpline = newSpline.add(term, indMapForAdd + [(rIndex, term.nInd - 1)])
                    else:
                        if nInd not in otherDictionary:
                            otherDictionary[nInd] = otherNInd
                            otherNInd += 1
                        newMap.append(otherDictionary[nInd])
                        term = dSpline.multiply(otherSpline)
                        newSpline = term if newSpline is None else newSpline.add(term, indMapForAdd)

                newMap = newMap[:newSpline.nInd]
                newRow.append((newMap, newSpline))
            turningPointBlock.append(newRow)

        # Second, add unit vector constrain to the system.
        # r^2 + c^2 + d^2 + ... = 1
        rSquaredMinus1 = bspy.Spline(1, 1, (3,), (3,), ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),), ((-1.0, -1.0, 0.0),))
        otherSquared = bspy.Spline(1, 1, (3,), (3,), ((-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),), ((1.0, -1.0, 1.0),))
        newRow = [((self.nInd,), rSquaredMinus1)]
        assert otherNInd == 2 * self.nInd - 1
        for nInd in range(self.nInd + 1, otherNInd):
            newRow.append(((nInd,), otherSquared))
        if self.nInd > 2:
            turningPointBlock.append(newRow)
        if self.nDep > 1:
            turningPointInitialScale = np.append(initialScale, (1.0,) * (self.nDep + 1))
        else:
            turningPointInitialScale = np.append(initialScale, (1.0,))
        
        # Finally, find the zeros of the system (only the first self.nInd values are of interest).
        zeros = bspy.spline_block.SplineBlock(turningPointBlock).zeros(epsilon, turningPointInitialScale)
        for uvw in zeros:
            if isinstance(uvw, tuple):
                abort = True
                break
            uvw = uvw[:self.nInd] # Remove any new independent variables added by the turning point system
            d = uvw[0] * cosTheta + uvw[1] * sinTheta 
            det = np.dot(self.normal(uvw, False), _turning_point_determinant_gradient(self, uvw, cosTheta, sinTheta))
            if abs(det) < epsilon:
                abort = True
                break
            points.append(Point(d, det, False, True, uvw))
        if not abort:
            break # We're done!
    
    if abort: raise ValueError("No contours. Degenerate equations.")

    if not points:
        return [] # No contours
    
    # We've got all the contour points, now we bucket them into individual contours using the algorithm 
    # from Grandine, Thomas A., and Frederick W. Klein IV. "A new approach to the surface intersection problem." 
    # Computer Aided Geometric Design 14, no. 2 (1997): 111-134.

    # Before we sort, we're going to need a system to find all the contour points on 
    # a panel boundary: u * cosTheta + v * sinTheta = d. Basically, we add this panel boundary plane
    # to the contour condition. We'll define it for d = 0, and add the actual d later.
    # We didn't construct the panel system earlier, because we didn't have theta.
    panelCoefs = np.array((((0.0, sinTheta), (cosTheta, cosTheta + sinTheta)),), self.coefsDtype)
    panelSpline = bspy.spline.Spline(2, 1, (2, 2), (2, 2), 
        (np.array((0.0, 0.0, 1.0, 1.0), self.knotsDtype), np.array((0.0, 0.0, 1.0, 1.0), self.knotsDtype)), 
        panelCoefs)
    panelBlock = self.block.copy()
    panelBlock.append([panelSpline])
    panel = bspy.spline_block.SplineBlock(panelBlock)
    panelInitialScale = np.append(initialScale, 1.0)

    # Okay, we have everything we need to determine the contour topology and points along each contour.
    # We've done the first two steps of Grandine and Klein's algorithm:
    # (1) Choose theta and find all solutions to (1.6) (system)
    # (2) Find all zeros of f on the boundary of [0, 1]^2

    # Next, sort the edge and turning points by panel distance (d) and then by the determinant (det)
    # (3) Take all the points found in Step (1) and Step (2) and order them by distance in the theta direction from the origin.
    points.sort()

    # Extra step not in paper.
    # Remove duplicate points (typically appear at corners).
    i = 0
    while i < len(points):
        previousPoint = points[i]
        j = i + 1
        while j < len(points):
            point = points[j]
            if point.d < previousPoint.d + epsilon:
                if np.linalg.norm(point.uvw - previousPoint.uvw) < epsilon:
                    del points[j]
                else:
                    j += 1
            else:
                break
        i += 1

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
            panelSpline.coefs = panelCoefs - point.d

            if point.turningPoint and point.uvw is None:
                # For an inserted panel between two consecutive turning points, just find zeros along the panel.
                panelPoints = panel.zeros(epsilon, panelInitialScale)
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
                        panelPoints += panel.trim(((point.uvw[0] + sinTheta * offset, 1.0), (0.0, point.uvw[1] - cosTheta * offset)) + ((None, None),) * (self.nInd - 2)).zeros(epsilon, panelInitialScale)
                        expectedPanelPoints -= len(panelPoints) - 1 # Discount the turning point itself
                    if expectedPanelPoints > 0 and epsilon < point.uvw[0] - sinTheta * offset and point.uvw[1] + cosTheta * offset < 1.0 - epsilon:
                        panelPoints += panel.trim(((0.0, point.uvw[0] - sinTheta * offset), (point.uvw[1] + cosTheta * offset, 1.0)) + ((None, None),) * (self.nInd - 2)).zeros(epsilon, panelInitialScale)
            else: # It's an other-boundary point.
                # Only find extra zeros along the panel if any are expected (> 0 for starting point, > 1 for ending one).
                expectedPanelPoints = len(currentContourPoints) - (0 if point.det > 0.0 else 1)
                if expectedPanelPoints > 0:
                    panelPoints = panel.zeros(epsilon, panelInitialScale)
                    panelPoints.sort(key=lambda uvw: np.linalg.norm(point.uvw - uvw)) # Sort by distance from boundary point
                    while len(panelPoints) > expectedPanelPoints:
                        panelPoints.pop(0) # Drop points closest to the boundary point
                    panelPoints.append(point.uvw)
                else:
                    panelPoints = [point.uvw]

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

def contours(self):
    if self.nInd - self.nDep != 1: raise ValueError("The number of free variables (self.nInd - self.nDep) must be one.")
    epsilon = np.sqrt(np.finfo(self.knotsDtype).eps)
    evaluationEpsilon = max(np.sqrt(epsilon), np.finfo(self.coefsDtype).eps ** 0.25)

    # Split the splines in the block to ensure C1 continuity within each block
    blocks = self.split(minContinuity=1).ravel()

    # For each block, find its contours and join them to the contours from previous blocks.
    contours = []
    for block in blocks:
        splineContours = _contours_of_C1_spline_block(block, epsilon, evaluationEpsilon)
        for newContour in splineContours:
            newStart = newContour(0.0)
            newFinish = newContour(1.0)
            joined = False
            for i, oldContour in enumerate(contours):
                oldStart = oldContour(0.0)
                oldFinish = oldContour(1.0)
                if np.linalg.norm(newStart - oldFinish) < evaluationEpsilon:
                    contours[i] = bspy.Spline.join((oldContour, newContour))
                    joined = True
                    break
                if np.linalg.norm(newStart - oldStart) < evaluationEpsilon:
                    contours[i] = bspy.Spline.join((oldContour, newContour.reverse()))
                    joined = True
                    break
                if np.linalg.norm(newFinish - oldStart) < evaluationEpsilon:
                    contours[i] = bspy.Spline.join((newContour, oldContour))
                    joined = True
                    break
                if np.linalg.norm(newFinish - oldFinish) < evaluationEpsilon:
                    contours[i] = bspy.Spline.join((newContour, oldContour.reverse()))
                    joined = True
                    break
            if not joined:
                contours.append(newContour)
    return contours

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
                if isinstance(zero, tuple) and zero[1] - zero[0] < Manifold.minSeparation:
                    zero = 0.5 * (zero[0] + zero[1])
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
        # Construct a spline block that represents the intersection.
        block = bspy.spline_block.SplineBlock([[self, -other]])

        # Curve-Curve intersection.
        if nDep == 1:
            # Find the intersection points and intervals.
            zeros = block.zeros()
            # Convert each intersection point into a Manifold.Crossing and each intersection interval into a Manifold.Coincidence.
            for zero in zeros:
                if isinstance(zero, tuple) and zero[1] - zero[0] < Manifold.minSeparation:
                    zero = 0.5 * (zero[0] + zero[1])
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
            if "Name" in self.metadata and "Name" in other.metadata:
                logging.info(f"intersect:{self.metadata['Name']}:{other.metadata['Name']}")
            # Find the intersection contours, which are returned as splines.
            swap = False
            try:
                # First try the intersection as is.
                contours = block.contours()
            except ValueError as e:
                logging.info(e)
                # If that fails, swap the manifolds. Worth a shot since intersections are touchy.
                block = bspy.spline_block.SplineBlock([[other, -self]])
                if "Name" in self.metadata and "Name" in other.metadata:
                    logging.info(f"intersect:{other.metadata['Name']}:{self.metadata['Name']}")
                contours = block.contours()
                # Convert each contour into a Manifold.Crossing, swapping the manifolds back.
                for contour in contours:
                    # Swap left and right, compared to not swapped.
                    left = bspy.Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.metadata)
                    right = bspy.Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.metadata)
                    intersections.append(Manifold.Crossing(left, right))
            else:
                # Convert each contour into a Manifold.Crossing.
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

def complete_slice(self, slice, solid):
    # Spline manifold domains have finite bounds.
    slice.containsInfinity = False
    bounds = self.domain()

    # If manifold (self) has no intersections with solid, just check containment.
    if not slice.boundaries:
        if slice.dimension == 2:
            if "Name" in self.metadata:
                logging.info(f"check containment:{self.metadata['Name']}")
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

    # For surfaces, intersect full spline domain with existing slice boundaries.
    if slice.dimension == 2:
        fullDomain = Hyperplane.create_hypercube(bounds)
        for newBoundary in fullDomain.boundaries: # Mark full domain boundaries as untouched
            newBoundary.touched = False

        # Define function for adding slice points to full domain boundaries.
        def process_domain_point(boundary, domainPoint, adjustment):
            point = boundary.manifold.evaluate(domainPoint)
            # See if and where point touches full domain.
            for newBoundary in fullDomain.boundaries:
                vector = point - newBoundary.manifold._point
                if abs(np.dot(newBoundary.manifold._normal, vector)) < Manifold.minSeparation:
                    # Add the point onto the new boundary (adjust normal evaluation point to move away from boundary).
                    normal = np.sign(newBoundary.manifold._tangentSpace.T @ boundary.manifold.normal(domainPoint + adjustment))
                    newBoundary.domain.add_boundary(Boundary(Hyperplane(normal, newBoundary.manifold._tangentSpace.T @ vector, 0.0), Solid(0, True)))
                    newBoundary.touched = True
                    break

        # Go through existing boundaries and check if either of their endpoints lies on the spline's bounds.
        for boundary in slice.boundaries:
            domainBoundaries = boundary.domain.boundaries
            domainBoundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), boundary.manifold.normal(0.0)))
            process_domain_point(boundary, domainBoundaries[0].manifold._point, Manifold.minSeparation)
            if len(domainBoundaries) > 1:
                process_domain_point(boundary, domainBoundaries[-1].manifold._point, -Manifold.minSeparation)
        
        # For touched boundaries, remove domain bounds that aren't needed and then add boundary to slice.
        boundaryWasTouched = False
        for newBoundary in fullDomain.boundaries:
            if newBoundary.touched:
                boundaryWasTouched = True
                domainBoundaries = newBoundary.domain.boundaries
                domainBoundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), boundary.manifold.normal(0.0)))
                # Ensure domain endpoints don't overlap and their normals are consistent.
                if abs(domainBoundaries[0].manifold._point - domainBoundaries[1].manifold._point) < Manifold.minSeparation or \
                    domainBoundaries[1].manifold._normal < 0.0:
                    del domainBoundaries[0]
                if abs(domainBoundaries[-1].manifold._point - domainBoundaries[-2].manifold._point) < Manifold.minSeparation or \
                    domainBoundaries[-2].manifold._normal > 0.0:
                    del domainBoundaries[-1]
                slice.add_boundary(newBoundary)
        
        if boundaryWasTouched:
            # Touch untouched boundaries that are connected to touched boundary endpoints and add them to slice.
            boundaryMap = ((2, 3, 0), (2, 3, -1), (0, 1, 0), (0, 1, -1)) # Map of which full domain boundaries touch each other
            while True:
                noTouches = True
                for map, newBoundary, bound in zip(boundaryMap, fullDomain.boundaries, bounds.flatten()):
                    if not newBoundary.touched:
                        leftBoundary = fullDomain.boundaries[map[0]]
                        rightBoundary = fullDomain.boundaries[map[1]]
                        if leftBoundary.touched and abs(leftBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                            newBoundary.touched = True
                            slice.add_boundary(newBoundary)
                            noTouches = False
                        elif rightBoundary.touched and abs(rightBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                            newBoundary.touched = True
                            slice.add_boundary(newBoundary)
                            noTouches = False
                if noTouches:
                    break
        else:
            # No slice boundaries touched the full domain (a hole), so only add full domain if it is contained in the solid.
            if solid.contains_point(self.evaluate(bounds[:,0])):
                for newBoundary in fullDomain.boundaries:
                    slice.add_boundary(newBoundary)

def full_domain(self):
    return Hyperplane.create_hypercube(self.domain())