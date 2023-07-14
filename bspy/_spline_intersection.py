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

    # Perform an interval Newton step

    def refine(spline, intervalSize, maxFunc):
        (bbox,) = spline.range_bounds()
        if bbox[0] * bbox[1] > epsilon:
            return []
        scalefactor = max(abs(bbox[0]), abs(bbox[1]))
        scalespl = spline.scale(1.0 / scalefactor)
        (mydomain,) = scalespl.domain()
        intervalSize *= mydomain[1] - mydomain[0]
        maxFunc *= scalefactor
        myspline = scalespl.reparametrize([[0.0, 1.0]])
        midPoint = 0.5
        [fval] = myspline([midPoint])

        # Root found

        if intervalSize < epsilon or abs(fval) * maxFunc < epsilon:
            return [0.5 * (mydomain[0] + mydomain[1])]
    
        # Calculate Newton update

        (fder,) = myspline.differentiate().range_bounds()
        if fder[0] == 0.0:
            fder[0] = epsilon
        if fder[1] == 0.0:
            fder[1] = -epsilon
        xleft = midPoint - fval / fder[0]
        xright = midPoint - fval / fder[1]
        dleft = min(xleft, xright) - 0.5 * epsilon
        dright = max(xleft, xright) + 0.5 * epsilon
        if fder[0] * fder[1] >= 0.0:    # Refine interval
           xnewleft = max(0.0, dleft)
           xnewright = min(1.0, dright)
           if xnewleft <= xnewright:
               trimspl = myspline.trim(((xnewleft, xnewright),))
               myzeros = refine(trimspl, intervalSize, maxFunc)
           else:
               return []
        else:                           # . . . or split as needed
            myzeros = []
            if dleft > 0.0:
                trimspl = myspline.trim(((0.0, dleft),))
                myzeros += refine(trimspl, intervalSize, maxFunc)
            if dright < 1.0:
                trimspl = myspline.trim(((dright, 1.0),))
                myzeros += refine(trimspl, intervalSize, maxFunc)
        return [(1.0 - thiszero) * mydomain[0] + thiszero * mydomain[1] for thiszero in myzeros]

    # See if there are any zero intervals

    (bbox,) = spline.range_bounds()
    scalefactor = max(abs(bbox[0]), abs(bbox[1]))
    mysolution = []
    for interval in range(spline.nCoef[0] - spline.order[0] + 1):
        maxFunc = max(np.abs(spline.coefs[0][interval:interval + spline.order[0]]))
        if maxFunc < scalefactor * epsilon:     # Found an interval of zeros
            intExtend = spline.nCoef[0] - spline.order[0] - interval
            for ix in range(intExtend):         # Attempt to extend the interval to more than one polynomial piece
                if abs(spline.coefs[0][interval + ix + spline.order[0]]) >= scalefactor * epsilon:
                    intExtend = ix
                    break
            leftend = spline.knots[0][interval + spline.order[0] - 1]
            rightend = spline.knots[0][interval + spline.order[0] + intExtend]
            if domain[0] != leftend:            # Compute zeros from left of the interval
                mysolution = refine(spline.trim(((domain[0], leftend - np.sqrt(epsilon)),)),
                                    max (1.0, 1.0 / (leftend - domain[0])), 1.0)
            mysolution += [(leftend, rightend)] # Add the interval of zeros
            if rightend != domain[1]:           # Add the zeros from right of the interval
                mysolution += spline.trim(((rightend + np.sqrt(epsilon), domain[1]),)).zeros()
            return mysolution
    return refine(spline, max (1.0, 1.0 / (domain[1] - domain[0])), 1.0)

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
        epsilon = self.accuracy
    epsilon = max(epsilon, np.sqrt(machineEpsilon))
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
                        hull = _convex_hull_2D(knotCoefs, coefs[nDep].ravel(), epsilon, xInterval)
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

def intersection_curves(self, other):
    assert self.nDep == other.nDep, "The number of dependent variables for both splines much match."
    assert self.nInd + other.nInd - self.nDep == 1, "The number of free variables (self.nInd + other.nInd - self.nDep) must be one."
    assert self.nInd == 2, "Only surfaces are supported."
    assert other.nInd == 2, "Only surfaces are supported."

    FMinusG = self - other
    Fu = self.differentiate(0)
    Fv = self.differentiate(1)
    Gs = other.differentiate(0)
    Gt = other.differentiate(1)
    GCross = Gs.multiply(Gt, (0, 1), 'C') # Map s and t to each other for Gs and Gt
    FuDotGCross = Fu.dot(GCross) # Fu and GCross don't share variables, so no mapping needed
    FvDotGCross = Fv.dot(GCross) # Fv and GCross don't share variables, so no mapping needed
    Point = namedtuple('Point', ('d', 'det', 'onBoundary', 'uvst',))
    theta = np.sqrt(2) # Arbitrary starting value for theta (picked one unlikely to be a stationary point)
    # Try different theta values until no border or turning points are degenerate.
    while True:
        points = []
        theta *= 0.7
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        abort = False

        # Construct the turning point determinant, mapping u and v in FuDotGCross and FvDotGCross.
        turningPointDeterminant = (sinTheta * FuDotGCross).add(-cosTheta * FvDotGCross, (0, 1))

        # Find intersections with boundaries, starting with u = 0.
        zeros = FMinusG.contract((0.0, None, None, None)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((0.0, zero[0], zero[1], zero[2]))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            det = (0.5 - uvst[1]) * FuDotGCross(uvst) * turningPointDeterminant(uvst)
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find intersections with u = 1.
        zeros = FMinusG.contract((0.0, None, None, None)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((1.0, zero[0], zero[1], zero[2]))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            det = (0.5 - uvst[1]) * FuDotGCross(uvst) * turningPointDeterminant(uvst)
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find intersections with v = 0.
        zeros = FMinusG.contract((None, 0.0, None, None)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((zero[0], 0.0, zero[1], zero[2]))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            det = (uvst[0] - 0.5) * FvDotGCross(uvst) * turningPointDeterminant(uvst)
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find intersections with v = 1.
        zeros = FMinusG.contract((None, 1.0, None, None)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((zero[0], 1.0, zero[1], zero[2]))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            det = (uvst[0] - 0.5) * FvDotGCross(uvst) * turningPointDeterminant(uvst)
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find intersections with s = 0.
        zeros = FMinusG.contract((None, None, 0.0, None)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((zero[0], zero[1], 0.0, zero[2]))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            duv = np.solve(np.column_stack(Fu(uvst[:2]), Fv(uvst[:2]), -Gt(uvst[2:])), Gs(uvst[2:]))
            det = np.arctan2((0.5 - uvst[2]) * (duv[0] * cosTheta + duv[1] * sinTheta), (0.5 - uvst[2]) * (duv[0] * cosTheta - duv[1] * sinTheta))
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find intersections with s = 1.
        zeros = FMinusG.contract((None, None, 1.0, None)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((zero[0], zero[1], 1.0, zero[2]))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            duv = np.solve(np.column_stack(Fu(uvst[:2]), Fv(uvst[:2]), -Gt(uvst[2:])), Gs(uvst[2:]))
            det = np.arctan2((0.5 - uvst[2]) * (duv[0] * cosTheta + duv[1] * sinTheta), (0.5 - uvst[2]) * (duv[0] * cosTheta - duv[1] * sinTheta))
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find intersections with t = 0.
        zeros = FMinusG.contract((None, None, None, 0.0)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((zero[0], zero[1], zero[2], 0.0))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            duv = np.solve(np.column_stack(Fu(uvst[:2]), Fv(uvst[:2]), -Gs(uvst[2:])), Gt(uvst[2:]))
            det = np.arctan2((0.5 - uvst[3]) * (duv[0] * cosTheta + duv[1] * sinTheta), (0.5 - uvst[2]) * (duv[0] * cosTheta - duv[1] * sinTheta))
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find intersections with t = 1.
        zeros = FMinusG.contract((None, None, None, 1.0)).zeros()
        for zero in zeros:
            if not isinstance(zero, np.ndarray):
                abort = True
                break
            uvst = np.array((zero[0], zero[1], zero[2], 1.0))
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            duv = np.solve(np.column_stack(Fu(uvst[:2]), Fv(uvst[:2]), -Gs(uvst[2:])), Gt(uvst[2:]))
            det = np.arctan2((0.5 - uvst[3]) * (duv[0] * cosTheta + duv[1] * sinTheta), (0.5 - uvst[2]) * (duv[0] * cosTheta - duv[1] * sinTheta))
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, True, uvst))
        if abort:
            continue # Try a different theta

        # Find turning points by combining FMinusG and turningPointDeterminant into a system and processing its zeros.
        systemFMinusG, systemTurningPointDeterminant = FMinusG.common_basis((turningPointDeterminant,), ((0,0), (1,1), (2,2), (3,3)))
        system = type(systemFMinusG)(4, 4, systemFMinusG.order, systemFMinusG.nCoef, systemFMinusG.knots, \
            np.concatenate((systemFMinusG.coef, systemTurningPointDeterminant.coef)), systemFMinusG.accuracy, systemFMinusG.metadata)
        zeros = system.zeros()
        for uvst in zeros:
            if not isinstance(uvst, np.ndarray):
                abort = True
                break
            d = cosTheta * uvst[0] + sinTheta * uvst[1]
            uv = uvst[:2]
            st = uvst[2:]
            gCross = GCross(st)
            fuDotGCross = FuDotGCross(uvst)
            fvDotGCross = FvDotGCross(uvst)
            fCross = np.cross(Fu(uv), Fv(uv))
            gsDotFCross = np.dot(Gs(st), fCross)
            gtDotFCross = np.dot(Gt(st), fCross)
            gamma = np.dot(self.derivative((2, 0), uv), gCross) * fvDotGCross * fvDotGCross \
                - 2.0 * np.dot(self.derivative((1, 1), uv), gCross) * fuDotGCross * fvDotGCross \
                + np.dot(self.derivative((0, 2), uv), gCross) * fuDotGCross * fuDotGCross \
                - np.dot(other.derivative((2, 0), st), gCross) * gtDotFCross * gtDotFCross \
                + 2.0 * np.dot(other.derivative((1, 1), st), gCross) * gsDotFCross * gtDotFCross \
                - np.dot(other.derivative((0, 2), st), gCross) * gsDotFCross * gsDotFCross
            alpha = cosTheta * fuDotGCross + sinTheta * fvDotGCross
            det = alpha * gamma
            if abs(det) < 1.0e-8:
                about = True
                break
            points.append(Point(d, det, False, uvst))
        if not abort:
            break # We're done!
    
    # We've got all the contour points, now we bucket them into individual contours using the algorithm 
    # from Grandine, Thomas A., and Frederick W. Klein IV. "A new approach to the surface intersection problem." 
    # Computer Aided Geometric Design 14, no. 2 (1997): 111-134.

    # Before we sort, we're going to need a system to find all the contour points on 
    # a panel boundary: u * cosTheta + v * sinTheta = d. Basically, we add this panel boundary plane
    # to the FMinusG contour condition. We'll define it for d = 0, and add the actual d later.
    # We didn't construct the panel system earlier, because we didn't have theta.
    panelCoefs = np.empty((4, *FMinusG.coefs.shape[1:]), FMinusG.coefs.dtype)
    panelCoefs[:3] = FMinusG.coefs
    # The following value should be -d. We're setting it for d = 0 to start.
    panelCoefs[3, 0, 0, :, :] = 0.0 
    degree = FMinusG.order[0] - 1
    for i in range(1, FMinusG.nCoef[0]):
        panelCoefs[3, i, 0, :, :] = panelCoefs[3, i - 1, 0, :, :] + ((FMinusG.knots[0][degree + i] - FMinusG.knots[0][i]) / degree) * cosTheta
    degree = FMinusG.order[1] - 1
    for i in range(1, FMinusG.nCoef[1]):
        panelCoefs[3, :, i, :, :] = panelCoefs[3, :, i - 1, :, :] + ((FMinusG.knots[1][degree + i] - FMinusG.knots[1][i]) / degree) * sinTheta
    panelFMinusG = type(FMinusG)(4, 4, FMinusG.order, FMinusG.nCoef, FMinusG.knots, panelCoefs, FMinusG.accuracy, FMinusG.metadata)

    # Okay, we have everything we need to determine the contour topology and points along each contour.
    # We've done the first two steps of Grandine and Klein's algorithm:
    # (1) Choose theta and find all solutions to (1.6) (system)
    # (2) Find all zeros of f on the boundary of [0, 1]^2

    # Next, sort the edge and turning points by panel distance (d) and then by the determinant (det)
    # (3) Take all the points found in Step (1) and Step (2) and order them by distance in the theta direction from the origin.
    points.sort()

    # (4) Initialize an ordered list of contours. No contours will be on the list at first.
    currentContourPoints = [] # Holds contours currently being identified
    contourPoints = [] # Hold contours already identified

    # (5) If no points remain to be processed, stop. Otherwise, take the next closest point.
    while points:
        point = points.pop(0)
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
            pass
        else:
            # (7) Determine whether the point is a turning point or a critical point. For now, we
            # will assume that the point is a turning point and defer the discussion of critical
            # points to Section 2. Determine whether two contours start or two contours end
            # at the turning point. Locate the two contours in the list of contours by finding
            # all points which lie on both the panel boundary and on the contour. The turning
            # point will be one of these, and it will be well ordered with respect to the other
            # points. Either insert two new contours in the list or delete two existing ones from
            # the list. Go back to Step (5).
