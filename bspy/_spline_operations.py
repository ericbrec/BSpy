import numpy as np
import bspy.spline
from collections import namedtuple
from enum import Enum
from math import comb

def _shiftPolynomial(polynomial, delta):
    if abs(delta) > np.finfo(polynomial.dtype).eps:
        length = len(polynomial)
        for j in range(1, length):
            for i in range(length - 2, j - 2, -1):
                polynomial[i] += delta * polynomial[i + 1]

def add(self, other, indMap = None):
    if not(self.nDep == other.nDep): raise ValueError("self and other must have same nDep")
    selfMapped = []
    otherMapped = []
    otherToSelf = {}
    if indMap is not None:
        (self, other) = bspy.Spline.common_basis((self, other), indMap)
        for map in indMap:
            selfMapped.append(map[0])
            otherMapped.append(map[1])
            otherToSelf[map[1]] = map[0]

    # Construct new spline parameters.
    # We index backwards because we're adding transposed coefficients (see below). 
    nInd = self.nInd
    order = [*self.order]
    nCoef = [*self.nCoef]
    knots = list(self.knots)
    permutation = [] # Used to transpose coefs to match other.coefs.T.
    for i in range(self.nInd - 1, -1, -1):
        if i not in selfMapped:
            permutation.append(i + 1) # Add 1 to account for dependent variables.
    for i in range(other.nInd - 1, -1, -1):
        if i not in otherMapped:
            order.append(other.order[other.nInd - 1 - i])
            nCoef.append(other.nCoef[other.nInd - 1 - i])
            knots.append(other.knots[other.nInd - 1 - i])
            permutation.append(self.nInd + i + 1) # Add 1 to account for dependent variables.
            nInd += 1
        else:
            permutation.append(otherToSelf[i] + 1) # Add 1 to account for dependent variables.
    permutation.append(0) # Account for dependent variables.
    permutation = np.array(permutation)
    coefs = np.zeros((self.nDep, *nCoef), self.coefs.dtype)

    # Build coefs array by transposing the changing coefficients to the end, including the dependent variables.
    # First, add in self.coefs.
    coefs = coefs.T
    coefs += self.coefs.T
    # Permutation for other.coefs.T accounts for coefs being transposed by subtracting permutation from ndim - 1.
    coefs = coefs.transpose((coefs.ndim - 1) - permutation)
    # Add in other.coefs. 
    coefs += other.coefs.T
    # Reverse the permutation.
    coefs = coefs.transpose(np.argsort(permutation)) 
    
    return type(self)(nInd, self.nDep, order, nCoef, knots, coefs, self.metadata)

def confine(self, range_bounds):
    if self.nInd != 1: raise ValueError("Confine only works on curves (nInd == 1)")
    if len(range_bounds) != self.nDep: raise ValueError("len(range_bounds) must equal nDep")
    spline = self.clamp((0,), (0,))
    order = spline.order[0]
    degree = order - 1
    domain = spline.domain()
    unique, counts = np.unique(spline.knots[0], return_counts=True)
    machineEpsilon = np.finfo(self.coefs.dtype).eps
    epsilon = np.sqrt(machineEpsilon)
    intersections = [] # List of tuples (u, boundaryPoint, headingOutside)

    def addIntersection(u, headedOutside = False):
        boundaryPoint = spline(np.atleast_1d(u))
        for i in range(spline.nDep):
            if boundaryPoint[i] < range_bounds[i][0]:
                headedOutside = True if boundaryPoint[i] < range_bounds[i][0] - epsilon else headedOutside
                boundaryPoint[i] = range_bounds[i][0]
            if boundaryPoint[i] > range_bounds[i][1]:
                headedOutside = True if boundaryPoint[i] > range_bounds[i][1] + epsilon else headedOutside
                boundaryPoint[i] = range_bounds[i][1]
        intersections.append((u, boundaryPoint, headedOutside))

    def intersectBoundary(i, j):
        zeros = type(spline)(1, 1, spline.order, spline.nCoef, spline.knots, (spline.coefs[i] - range_bounds[i][j],)).zeros()
        for zero in zeros:
            if isinstance(zero, tuple):
                headedOutside = (-1 if j == 0 else 1) * spline.derivative((1,), np.atleast_1d(zero[0]))[i] > epsilon
                addIntersection(zero[0], headedOutside)
                headedOutside = (-1 if j == 0 else 1) * spline.derivative((1,), np.atleast_1d(zero[1]))[i] > epsilon
                addIntersection(zero[1], headedOutside)
            else:
                headedOutside = (-1 if j == 0 else 1) * spline.derivative((1,), np.atleast_1d(zero))[i] > epsilon
                addIntersection(zero, headedOutside)

    addIntersection(domain[0][0]) # Confine starting point
    addIntersection(domain[0][1]) # Confine ending point
    # Confine points between start and end.
    for i in range(spline.nDep):
        intersectBoundary(i, 0)
        intersectBoundary(i, 1)
    
    # Put the intersection points in order.
    intersections.sort(key=lambda intersection: intersection[0])

    # Remove repeat points at start and end.
    if intersections[1][0] - intersections[0][0] < epsilon:
        del intersections[1]
    if intersections[-1][0] - intersections[-2][0] < epsilon:
        del intersections[-2]

    # Insert order-1 knots at each intersection point.
    for (knot, boundaryPoint, headedOutside) in intersections:
        ix = np.searchsorted(unique, knot)
        if unique[ix] == knot:
            count = (order - 1) - counts[ix]
            if count > 0:
                spline = spline.insert_knots(([knot] * count,))
        else:
            spline = spline.insert_knots(([knot] * (order - 1),))

    # Go through the boundary points, assigning boundary coefficients, interpolating between boundary points, 
    # and removing knots and coefficients where the curve stalls.
    nCoef = spline.nCoef[0]
    knots = spline.knots[0]
    coefs = spline.coefs
    previousKnot, previousBoundaryPoint, previousHeadedOutside = intersections[0]
    previousIx = 0
    coefs[:, previousIx] = previousBoundaryPoint
    knotAdjustment = 0.0
    for knot, boundaryPoint, headedOutside in intersections[1:]:
        knot += knotAdjustment
        ix = np.searchsorted(knots, knot, 'right') - order
        ix = min(ix, nCoef - 1)
        coefs[:, ix] = boundaryPoint # Assign boundary coefficients
        if previousHeadedOutside and np.linalg.norm(boundaryPoint - previousBoundaryPoint) < epsilon:
            # Curve has stalled, so remove intervening knots and coefficients, and adjust knot values.
            nCoef -= ix - previousIx
            knots = np.delete(knots, slice(previousIx + 1, ix + 1))
            knots[previousIx + 1:] -= knot - previousKnot
            knotAdjustment -= knot - previousKnot
            coefs = np.delete(coefs, slice(previousIx, ix), axis=1)
            previousHeadedOutside = headedOutside # The previous knot is unchanged, but inherits the new headedOutside value
        else:
            if previousHeadedOutside:
                # If we were outside, linearly interpolate between the previous and current boundary points.
                slope = (boundaryPoint - previousBoundaryPoint) / (knot - previousKnot)
                for i in range(previousIx + 1, ix):
                    coefs[:, i] = coefs[:, i - 1] + ((knots[i + degree] - knots[i]) / degree) * slope

            # Update previous knot
            previousKnot = knot
            previousBoundaryPoint = boundaryPoint
            previousHeadedOutside = headedOutside
            previousIx = ix
    
    spline.nCoef = (nCoef,)
    spline.knots = (knots,)
    spline.coefs = coefs
    return spline.reparametrize(domain) # Return the spline adjusted back to the original domain

def contract(self, uvw):
    domain = self.domain()
    section = [slice(None)]
    bValues = []
    contracting = False
    for iv in range(self.nInd):
        if uvw[iv] is not None:
            if uvw[iv] < domain[iv][0] or uvw[iv] > domain[iv][1]:
                raise ValueError(f"Spline evaluation outside domain: {uvw}")

            # Grab all of the appropriate coefficients
            ix, indValues = bspy.Spline.bspline_values(None, self.knots[iv], self.order[iv], uvw[iv])
            bValues.append(indValues)
            section.append(slice(ix - self.order[iv], ix))
            contracting = True
        else:
            bValues.append([])
            section.append(slice(None))

    if not contracting:
        return self

    nInd = self.nInd
    order = [*self.order]
    nCoef = [*self.nCoef]
    knots = [*self.knots]
    coefs = self.coefs[tuple(section)]
    ix = 0
    for iv in range(self.nInd):
        if uvw[iv] is not None:
            nInd -= 1
            del order[ix]
            del nCoef[ix]
            del knots[ix]
            coefs = np.moveaxis(coefs, ix + 1, -1)
            coefs = coefs @ bValues[iv]
        else:
            ix += 1
    
    return type(self)(nInd, self.nDep, order, nCoef, knots, coefs, self.metadata)

def cross(self, vector):
    if isinstance(vector, bspy.Spline):
        return self.multiply(vector, [(ix, ix) for ix in range(min(self.nInd, vector.nInd))], 'C')
    elif self.nDep == 3:
        if not(len(vector) == self.nDep): raise ValueError("Invalid vector")

        coefs = np.empty(self.coefs.shape, self.coefs.dtype)
        coefs[0] = vector[2] * self.coefs[1] - vector[1] * self.coefs[2]
        coefs[1] = vector[0] * self.coefs[2] - vector[2] * self.coefs[0]
        coefs[2] = vector[1] * self.coefs[0] - vector[0] * self.coefs[1]
        return type(self)(self.nInd, 3, self.order, self.nCoef, self.knots, coefs, self.metadata)
    else:
        if not(self.nDep == 2): raise ValueError("Invalid nDep")
        if not(len(vector) == self.nDep): raise ValueError("Invalid vector")

        coefs = np.empty((1, *self.coefs.shape[1:]), self.coefs.dtype)
        coefs[0] = vector[1] * self.coefs[0] - vector[0] * self.coefs[1]
        return type(self)(self.nInd, 3, self.order, self.nCoef, self.knots, coefs, self.metadata)

def differentiate(self, with_respect_to = 0):
    if not(0 <= with_respect_to < self.nInd): raise ValueError("Invalid with_respect_to")
    if not(self.order[with_respect_to] > 1): raise ValueError("Invalid with_respect_to")

    order = [*self.order]
    order[with_respect_to] -= 1
    degree = order[with_respect_to] 

    nCoef = [*self.nCoef]
    nCoef[with_respect_to] -= 1

    dKnots = self.knots[with_respect_to][1:-1]
    knots = list(self.knots)
    knots[with_respect_to] = dKnots

    # Swap dependent variable axis with specified independent variable and remove first row.
    oldCoefs = self.coefs.swapaxes(0, with_respect_to + 1)
    newCoefs = np.delete(oldCoefs, 0, axis=0) 
    for i in range(nCoef[with_respect_to]):
        alpha =  degree / (dKnots[i+degree] - dKnots[i])
        newCoefs[i] = alpha * (newCoefs[i] - oldCoefs[i])
    
    return type(self)(self.nInd, self.nDep, order, nCoef, knots, newCoefs.swapaxes(0, with_respect_to + 1), self.metadata)

def dot(self, vector):
    if isinstance(vector, bspy.Spline):
        return self.multiply(vector, [(ix, ix) for ix in range(min(self.nInd, vector.nInd))], 'D')
    else:
        if not(len(vector) == self.nDep): raise ValueError("Invalid vector")

        coefs = vector[0] * self.coefs[0]
        for i in range(1, self.nDep):
            coefs += vector[i] * self.coefs[i]
        if len(coefs.shape) == len(self.coefs.shape) - 1:
            coefs = coefs.reshape(1, *coefs.shape)
        return type(self)(self.nInd, 1, self.order, self.nCoef, self.knots, coefs, self.metadata)

def graph(self):
    self = self.clamp(range(self.nInd), range(self.nInd))
    coefs = np.insert(self.coefs, self.nInd * (0,), 0.0, axis = 0)
    for nInd in range(self.nInd):
        dep = np.swapaxes(coefs, nInd + 1, 1)[nInd] # Returns a view, so changes to dep make changes to coefs
        for i, knotAverage in enumerate(self.greville(nInd)):
            dep[i] = knotAverage
    return type(self)(self.nInd, self.nInd + self.nDep, self.order, self.nCoef, self.knots, coefs, self.metadata)

def integrate(self, with_respect_to = 0):
    if not(0 <= with_respect_to < self.nInd): raise ValueError("Invalid with_respect_to")

    order = [*self.order]
    degree = order[with_respect_to]
    order[with_respect_to] += 1

    nCoef = [*self.nCoef]
    nCoef[with_respect_to] += 1

    iKnots = np.empty(len(self.knots[with_respect_to]) + 2, self.knots[with_respect_to].dtype)
    iKnots[1:-1] = self.knots[with_respect_to]
    iKnots[0] = iKnots[1]
    iKnots[-1] = iKnots[-2]
    knots = list(self.knots)
    knots[with_respect_to] = iKnots

    # Swap dependent variable axis with specified independent variable.
    oldCoefs = self.coefs.swapaxes(0, with_respect_to + 1)
    newCoefs = np.empty((self.nDep, *nCoef), self.coefs.dtype).swapaxes(0, with_respect_to + 1)

    # Compute new coefficients.
    newCoefs[0] = 0.0
    for i in range(1, nCoef[with_respect_to]):
        newCoefs[i] = newCoefs[i - 1] + ((iKnots[degree + i] - iKnots[i]) / degree) * oldCoefs[i - 1]

    return type(self)(self.nInd, self.nDep, order, nCoef, knots, newCoefs.swapaxes(0, with_respect_to + 1), self.metadata)

def multiplyAndConvolve(self, other, indMap = None, productType = 'S'):
    if not(productType == 'C' or productType == 'D' or productType == 'S'): raise ValueError("productType must be 'C', 'D' or 'S'")

    if not(productType != 'D' or self.nDep == other.nDep): raise ValueError("Mismatched dimensions")
    if not(productType != 'C' or (self.nDep == other.nDep and 2 <= self.nDep <= 3)): raise ValueError("Mismatched dimensions")
    if not(productType != 'S' or self.nDep == 1 or other.nDep == 1 or self.nDep == other.nDep): raise ValueError("Mismatched dimensions")

    # Eliminate case tough case for data handling
    if other.nDep == 1 and self.nDep > 1:
        other = (self.nDep * [[1]]) @ other

    # Construct new spline parameters.
    nInd = self.nInd + other.nInd
    nDep = other.nDep
    if productType == 'C' and nDep == 2:
        nDep = 1
    order = [*self.order, *other.order]
    nCoef = [*self.nCoef, *other.nCoef]
    knots = [*self.knots, *other.knots]

    # Multiply the coefs arrays as if the independent variables from both splines are unrelated.
    outer = np.outer(self.coefs, other.coefs).reshape((*self.coefs.shape, *other.coefs.shape))
    # Move the other spline's dependent variable next to this spline's. 
    outer = np.moveaxis(outer, self.nInd + 1, 1)

    # Combine dependent variables based on type of product.
    if productType == 'C': # Cross product
        if self.nDep == 3:
            coefs = np.empty(outer[0].shape, outer.dtype)
            coefs[0] = outer[1,2] - outer[2,1]
            coefs[1] = outer[2,0] - outer[0,2]
            coefs[2] = outer[0,1] - outer[1,0]
        else: # self.nDep == 2
            coefs = np.empty((1, *outer.shape[2:]), outer.dtype)
            coefs[0] = outer[0,1] - outer[1,0]
    elif productType == 'D': # Dot product
        coefs = outer[0,0]
        for i in range(1, self.nDep):
            coefs += outer[i,i]
        coefs = np.expand_dims(coefs, axis=0)
        nDep = 1
    else: # Scalar product
        coefs = outer
        for i in range(1, self.nDep):
            coefs[0,i] = coefs[i,i]
        coefs = coefs[0]

    if indMap is not None:
        indMap = indMap.copy() # Make a copy, since we change the list as we combine independent variables
        while indMap:
            (ind1, ind2, isConvolve) = indMap.pop()

            # First, get multiplicities of the knots for each independent variable.
            order1 = self.order[ind1]
            knots1, multiplicities1 = np.unique(self.knots[ind1][order1-1:self.nCoef[ind1]+1], return_counts=True)
            multiplicities1[0] = multiplicities1[-1] = order1
            order2 = other.order[ind2]
            knots2, multiplicities2 = np.unique(other.knots[ind2][order2-1:other.nCoef[ind2]+1], return_counts=True)
            multiplicities2[0] = multiplicities2[-1] = order2

            if isConvolve:
                # For convolve, we need to convolve matching independent variables (variables mapped to each other).
                # We can't do this directly with B-spline coefficients, but we can indirectly with the following steps:
                #   1) Determine the knots of the convolved spline from the knots of the matching independent variables.
                #   2) Use the knots to determine segments and integration intervals that compute integral of self(x - y) * other(y) dy.
                #   3) For each segment:
                #       a) For each interval:
                #           i) Convert each spline interval into a polynomial (Taylor series).
                #           ii) Separate the variables for the self spline interval: self(x - y) = tensor product of selfX(x) * selfY(y).
                #           iii) Shift selfY(y) Taylor series to be about the same point as other(y).
                #           iv) Multiply selfY(y) times other(y) by summing coefficients of matching polynomial degree.
                #           v) Integrate the result (trivial for polynomials, just increase the order by one and divide by the increased order).
                #           vi) Evaluate the integral at the interval endpoints (which are linear functions of x), shifting the Taylor series to be about the same point as selfX(x).
                #           vii) Multiply selfX(x) times the result by summing coefficients of matching polynomial degree.
                #           viii) Accumulate the integral.
                #       b) Use blossoms to compute the spline segment coefficients from the polynomial segment (uses the raceme function from E.T.Y. Lee).

                # 1) Determine the knots of the convolved spline from the knots of the matching independent variables.

                # Create a list of all the knot intervals.
                IntervalKind = Enum('IntervalKind', ('Start', 'End', 'EndPoint'))
                IntervalInfo = namedtuple('IntervalInfo', ('kind', 'knot', 'multiplicity', 'index1', 'index2'))
                intervalInfoList = []
                for knotNumber1 in range(len(knots1)):
                    for knotNumber2 in range(len(knots2)):
                        knot = knots1[knotNumber1] + knots2[knotNumber2]
                        multiplicity = max(multiplicities1[knotNumber1] + order2 - 1, multiplicities2[knotNumber2] + order1 - 1)
                        if knotNumber1 < len(knots1) - 1 and knotNumber2 < len(knots2) - 1:
                            intervalInfoList.append(IntervalInfo(IntervalKind.Start, knot, multiplicity, knotNumber1, knotNumber2)) # Start an interval
                        if knotNumber1 > 0 and knotNumber2 > 0:
                            intervalInfoList.append(IntervalInfo(IntervalKind.End, knot, multiplicity, knotNumber1 - 1, knotNumber2 - 1)) # End a previous interval
                        if (knotNumber1 == 0 and knotNumber2 == len(knots2) - 1) or (knotNumber1 == len(knots1) - 1 and knotNumber2 == 0):
                            intervalInfoList.append(IntervalInfo(IntervalKind.EndPoint, knot, multiplicity, knotNumber1, knotNumber2)) # EndPoint knot

                # Sort the list of intervals.
                intervalInfoList.sort(key=lambda intervalInfo: intervalInfo.knot)

                # 2) Use the knots to determine segments and integration intervals that compute integral of self(x - y) * other(y) dy.
                KnotInfo = namedtuple('KnotInfo', ('knot', 'multiplicity', 'intervals'))
                atol = 1.0e-8
                intervals = []
                knotInfoList = []
                knotInfo = None
                for intervalInfo in intervalInfoList:
                    if intervalInfo.kind == IntervalKind.Start:
                        intervals.append((intervalInfo.index1, intervalInfo.index2))
                    elif intervalInfo.kind == IntervalKind.End:
                        intervals.remove((intervalInfo.index1, intervalInfo.index2))
                    intervals.sort(key=lambda interval: (-interval[0], interval[1]))
                    # Update previous knot or add a new knot
                    if knotInfo and np.isclose(knotInfo.knot, intervalInfo.knot, atol=atol):
                        knotInfoList[-1] = KnotInfo(knotInfo.knot, max(knotInfo.multiplicity, intervalInfo.multiplicity), list(intervals))
                    else:
                        knotInfo = KnotInfo(intervalInfo.knot, intervalInfo.multiplicity, list(intervals))
                        knotInfoList.append(knotInfo)

                # Compute the new order of the combined spline and its new knots array.
                newOrder = order1 + order2
                newKnots = [knotInfoList[0].knot] * newOrder
                newMultiplicities = [newOrder]
                for knotInfo in knotInfoList[1:-1]:
                    newKnots += [knotInfo.knot] * knotInfo.multiplicity
                    newMultiplicities.append(knotInfo.multiplicity)
                newKnots += [knotInfoList[-1].knot] * newOrder
                newMultiplicities.append(newOrder)
            else:
                # For multiply, we need to combine like terms for matching independent variables (variables mapped to each other).
                # We can't do this directly with B-spline coefficients, but we can indirectly with the following steps:
                #   1) Use the combined knots from matching independent variables to divide the spline into segments.
                #   2) For each segment, convert the splines into polynomials (Taylor series).
                #   3) Sum coefficients of matching polynomial degree (the coefficients have already been multiplied together).
                #   4) Use blossoms to compute the spline segment coefficients from the polynomial segment (uses the raceme function from E.T.Y. Lee).

                # 1) Use the combined knots from matching independent variables to divide the spline into segments.

                # Compute the new order of the combined spline and its new knots array.
                if not(knots1[0] == knots2[0] and knots1[-1] == knots2[-1]): raise ValueError(f"self[{ind1}] domain doesn't match other[{ind2}]")
                newOrder = order1 + order2 - 1
                newKnots = [knots1[0]] * newOrder
                newMultiplicities = [newOrder]
                i1 = i2 = 0
                while i1 + 1 < len(knots1) and i2 + 1 < len(knots2):
                    if knots1[i1 + 1] < knots2[i2 + 1]:
                        i1 += 1
                        knot = knots1[i1]
                        multiplicity = multiplicities1[i1] + order2 - 1
                    elif knots1[i1 + 1] > knots2[i2 + 1]:
                        i2 += 1
                        knot = knots2[i2]
                        multiplicity = multiplicities2[i2] + order1 - 1
                    else:
                        i1 += 1
                        i2 += 1
                        knot = knots1[i1]
                        multiplicity = max(multiplicities1[i1] + order2 - 1, multiplicities2[i2] + order1 - 1)
                    newKnots += [knot] * multiplicity
                    newMultiplicities.append(multiplicity)

            # Update nInd, order, nCoef, overall knots, and indMap
            nInd -= 1
            del order[self.nInd + ind2]
            order[ind1] = newOrder
            del nCoef[self.nInd + ind2]
            nCoef[ind1] = len(newKnots) - newOrder
            del knots[self.nInd + ind2]
            knots[ind1] = np.array(newKnots, knots1.dtype)
            for i in range(len(indMap)):
                i2 = indMap[i][1]
                if not(i2 != ind2): raise ValueError("You can't map the same independent variable to multiple others.")
                if i2 > ind2:
                    indMap[i] = (indMap[i][0], i2 - 1)

            # Compute segments (uses the III algorithm from E.T.Y. Lee)
            i = 0 # knot index into full knot list
            j = 0 # knot index into unique knot list
            Segment = namedtuple('Segment', ('knot','unique'))
            segments = [Segment(i, j)]
            sigma = newMultiplicities[j]
            while i < nCoef[ind1]:
                while sigma <= segments[-1].knot + newOrder:
                    j += 1
                    i = sigma
                    sigma += newMultiplicities[j]
                segments.append(Segment(i, j))

            # Move the two independent variables to the left side of the coefficients array in prep for computing Taylor coefficients,
            #   and initialize new coefficients array.
            coefs = np.moveaxis(coefs, (ind1+1, self.nInd+1 + ind2), (0, 1))
            newCoefs = np.empty(((len(segments) - 1) * newOrder, *coefs.shape[2:]), coefs.dtype)

            # Loop through the segments
            segmentStart = segments[0]
            for segmentEnd in segments[1:]:
                # Initialize segment coefficients.
                a = newCoefs[segmentStart.knot:segmentStart.knot + newOrder]
                a.fill(0.0)
                knot = newKnots[segmentStart.knot]

                if isConvolve:
                    # Next step in convolve:
                    # a) For each interval:
                    for interval in knotInfoList[segmentStart.unique].intervals:
                        # i) Convert each spline interval into a polynomial (Taylor series).

                        # Isolate the appropriate interval coefficients
                        xLeft = knots1[interval[0]]
                        xRight = knots1[interval[0] + 1]
                        ix1 = np.searchsorted(self.knots[ind1], xLeft, 'right')
                        ix1 = min(ix1, self.nCoef[ind1])
                        yLeft = knots2[interval[1]]
                        yRight = knots2[interval[1] + 1]
                        ix2 = np.searchsorted(other.knots[ind2], yLeft, 'right')
                        ix2 = min(ix2, other.nCoef[ind2])

                        # Compute taylor coefficients for the interval.
                        # Expand self(x) about knot.
                        # Expand other(y) about yLeft.
                        taylorCoefs = (coefs[ix1 - order1:ix1, ix2 - order2:ix2]).T # Transpose so we multiply on the left (due to matmul rules)
                        bValues = np.empty((order1, order1), knots1.dtype)
                        for derivativeOrder in range(order1):
                            ix1, bValues[:,derivativeOrder] = bspy.Spline.bspline_values(ix1, self.knots[ind1], order1, knot, derivativeOrder, True)
                        taylorCoefs = taylorCoefs @ bValues
                        taylorCoefs = np.moveaxis(taylorCoefs, -1, 0) # Move ind1's taylor coefficients to the left side so we can compute ind2's
                        bValues = np.empty((order2, order2), knots2.dtype)
                        for derivativeOrder in range(order2):
                            ix2, bValues[:,derivativeOrder] = bspy.Spline.bspline_values(ix2, other.knots[ind2], order2, yLeft, derivativeOrder, True)
                        taylorCoefs = taylorCoefs @ bValues
                        taylorCoefs = (np.moveaxis(taylorCoefs, 0, -1)).T # Move ind1's taylor coefficients back to the right side, and re-transpose

                        # ii) Separate the variables for the self spline interval: self(x - y) = tensor product of selfX(x) * selfY(y).
                        # Initialize coefficients to zero (separatedTaylorCoefs[i, j] = 0 for i + j >= order1).
                        separatedTaylorCoefs = np.zeros((order1, *taylorCoefs.shape), taylorCoefs.dtype)
                        for i in range(order1):
                            for j in range(order1 - i):
                                separatedTaylorCoefs[i, j] = comb(i + j, i) * ((-1) ** j) * taylorCoefs[i + j]
                        # Move selfX(x) to the right side, so we can focus our operations on selfY * other(y)
                        separatedTaylorCoefs = np.moveaxis(separatedTaylorCoefs, 0, -1)

                        # iii) Shift selfY(y) Taylor series to be about the same point as other(y).
                        _shiftPolynomial(separatedTaylorCoefs, yLeft)

                        # iv) Multiply selfY(y) times other(y) by summing coefficients of matching polynomial degree.
                        # v) Integrate the result (trivial for polynomials, just increase the order by one and divide by the increased order).
                        integratedTaylorCoefs = np.zeros((newOrder, *separatedTaylorCoefs.shape[2:]), separatedTaylorCoefs.dtype)
                        for i2 in range(order2):
                            for i1 in range(order1):
                                integratedTaylorCoefs[i1 + i2 + 1] += separatedTaylorCoefs[i1, i2] / (i1 + i2 + 1)

                        # vi) Evaluate the integral at the interval bounds (which are linear functions of x), shifting the Taylor series to be about the same point as selfX(x).
                        integral = integratedTaylorCoefs.copy() # A polynomial of the form ai * (y - yLeft)^i

                        # First, we determine the lower bound of the integral for the interval.
                        # It's yLeft if xRight >= knotRight - yLeft.
                        # Otherwise, it's x - xRight.
                        boundIsY = yLeft + xRight >= knotInfoList[segmentStart.unique + 1].knot - atol
                        
                        if boundIsY:
                            # The constant lower bound is yLeft, thus evaluating the integral at the lower bound give you zero.
                            integral.fill(0.0)
                        else:
                            # The variable lower bound is of the form (x - xRight).
                            # That makes integral evaluated at the lower bound of the form ai * (x - (xRight + yLeft))^i.
                            # Shift the integral to be about knot instead to match selfX(x).
                            _shiftPolynomial(integral, knot - (xRight + yLeft))
                            integral *= -1.0 # Subtract the lower bound

                        # Next, we determine the upper bound of the integral for the interval.
                        # It's yRight if xLeft <= knotLeft - yRight.
                        # Otherwise, it's x - xLeft.
                        boundIsY = yRight + xLeft <= knotInfoList[segmentStart.unique].knot + atol
                        
                        if boundIsY:
                            # The constant upper bound is yRight.
                            # Evaluate the integral at the upper bound.
                            base = yRight - yLeft
                            value = integratedTaylorCoefs[-1]
                            for i in range(1, newOrder):
                                value = integratedTaylorCoefs[-1 - i] + base * value
                            integral[0] += value
                        else:
                            # The variable upper bound is of the form (x - xLeft).
                            # That makes integral evaluated at the upper bound of the form ai * (x - (xLeft + yLeft))^i.
                            # Shift the integral polynomial to be about knot instead to match selfX(x).
                            _shiftPolynomial(integratedTaylorCoefs, knot - (xLeft + yLeft))
                            integral += integratedTaylorCoefs

                        # vii) Multiply selfX(x) times the result by summing coefficients of matching polynomial degree.
                        # viii) Accumulate the integral.
                        integral = np.moveaxis(integral, -1, 0) # Move selfX(x) back to the left side
                        for i1 in range(order1):
                            for i2 in range(newOrder - i1): # Coefficients with indices >= newOrder - i1 are zero by construction of the separation of variables
                                a[i1 + i2] += integral[i1, i2]
                else:
                    # Next step in multiply:
                    # 2) Convert each spline segment into a polynomial (Taylor series).
                    # Isolate the appropriate segment coefficients
                    ix1 = None
                    ix2 = None

                    # Compute taylor coefficients for the segment
                    bValues1 = np.empty((order1, order1), knots1.dtype)
                    for derivativeOrder in range(order1):
                        ix1, bValues1[:,derivativeOrder] = bspy.Spline.bspline_values(ix1, self.knots[ind1], order1, knot, derivativeOrder, True)
                    bValues2 = np.empty((order2, order2), knots2.dtype)
                    for derivativeOrder in range(order2):
                        ix2, bValues2[:,derivativeOrder] = bspy.Spline.bspline_values(ix2, other.knots[ind2], order2, knot, derivativeOrder, True)
                    taylorCoefs = (coefs[ix1 - order1:ix1, ix2 - order2:ix2]).T # Transpose so we multiply on the left (due to matmul rules)
                    taylorCoefs = taylorCoefs @ bValues1
                    taylorCoefs = np.moveaxis(taylorCoefs, -1, 0) # Move ind1's taylor coefficients to the left side so we can compute ind2's
                    taylorCoefs = taylorCoefs @ bValues2
                    taylorCoefs = (np.moveaxis(taylorCoefs, 0, -1)).T # Move ind1's taylor coefficients back to the right side, and re-transpose

                    # 3) Sum coefficients of matching polynomial degree (the coefficients have already been multiplied together by the outer product).
                    for i2 in range(order2):
                        for i1 in range(order1):
                            a[i1 + i2] += taylorCoefs[i1, i2]

                # 4) Use blossoms to compute the spline segment coefficients from the polynomial segment (uses the raceme function from E.T.Y. Lee).
                m = newOrder - 1
                rho = segmentEnd.knot - segmentStart.knot
                for j in range(m):
                    for i in range(min(newOrder, m - j)):
                        a[i] = (1 - i/(m - j)) * a[i] + ((i + 1)/(m - j)) * (newKnots[segmentStart.knot + m - j] - knot) * a[i + 1]
                for j in range(rho - 1):
                    for i in range(min(newOrder + j, rho - 1), j, -1):
                        a[i] = a[i - 1] + (newKnots[segmentStart.knot + m + j + 1] - newKnots[segmentStart.knot + i]) * a[i]
                
                # Move to next segment
                segmentStart = segmentEnd

            # All the segment coefficients are computed.
            # Now move combined independent variable back to its original axis.
            coefs = np.moveaxis(newCoefs[:nCoef[ind1]], 0, ind1 + 1)

    return type(self)(nInd, nDep, order, nCoef, knots, coefs, self.metadata)

def normal_spline(splineMatrix, indices=None):
    # Compute and validate nInd, nDep, knots dtype, and coefs dtype for splineMatrix.
    # Also, construct order, nCoef, knots, and sample values for generalized cross product of the tangent space.
    nInd = 0
    nDep = 0
    knotsDtype = coefsDtype = None
    newOrder = []
    newKnots = []
    uvwValues = []
    nCoefs = []
    totalCoefs = [1]
    # Loop through each independent variable.
    while True:
        knots = None
        counts = None
        order = 0
        for row in splineMatrix:
            rowInd = 0
            rowDep = 0
            for spline in row:
                if rowDep == 0:
                    rowDep = spline.nDep
                elif rowDep != spline.nDep:
                    raise ValueError("All splines in the same row must have the same nDep")
                if knotsDtype == None:
                    knotsDtype = spline.knots[0].dtype
                    coefsDtype = spline.coefs.dtype
                if (rowInd <= nInd < rowInd + spline.nInd) and \
                    (indices is None or nInd in indices):
                    ind = nInd - rowInd
                    ord = spline.order[ind]
                    fullOrd = (ord - 1) * spline.nInd
                    k, c = np.unique(spline.knots[ind][ord-1:spline.nCoef[ind]+1], return_counts=True)
                    if knots:
                        if knots[0] != k[0] or knots[-1] != k[-1]: raise ValueError("Domains of independent variables must match")
                        if order < ord:
                            counts += ord - order
                            order = ord
                        for knot, count in zip(k[1:-1], c[1:-1]):
                            ix = np.searchsorted(knots, knot)
                            if knots[ix] == knot:
                                counts[ix] = max(counts[ix], count + order - ord)
                            else:
                                knots = np.insert(knots, ix, knot)
                                counts = np.insert(counts, ix, count + order - ord)
                    else:
                        knots = k
                        counts = c
                        order = ord

                rowInd += spline.nInd
            nInd = max(nInd, rowInd)
            nDep += rowDep

    if abs(nInd - nDep) != 1: raise ValueError("The number of independent variables must be different than the number of dependent variables.")

    # Construct order and knots for generalized cross product of the tangent space.
    newOrder = []
    newKnots = []
    uvwValues = []
    nCoefs = []
    totalCoefs = [1]
    for i, (order, knots) in enumerate(zip(self.order, self.knots)):
        # First, calculate the order of the normal for this independent variable.
        # Note that the total order will be one less than usual, because one of 
        # the tangents is the derivative with respect to that independent variable.
        newOrd = 0
        if self.nInd < self.nDep:
            # If this normal involves all tangents, simply add the degree of each,
            # so long as that tangent contains the independent variable.  
            for j in range(self.nInd):
                newOrd += order - 1 if ccm[i, j] else 0
        else:
            # If this normal doesn't involve all tangents, find the max order of
            # each returned combination (as defined by the indices).
            for index in range(self.nInd) if indices is None else indices:
                # The order will be one larger if this independent variable's tangent is excluded by the index.
                ord = 0 if index != i else 1
                # Add the degree of each tangent, so long as that tangent contains the 
                # independent variable and is not excluded by the index.  
                for j in range(self.nInd):
                    ord += order - 1 if ccm[i, j] and index != j else 0
                newOrd = max(newOrd, ord)
        newOrder.append(newOrd)
        uniqueKnots, counts = np.unique(knots[order - 1:self.nCoef[i] + 1], return_counts=True)
        counts += newOrd - order + 1 # Because we're multiplying all the tangents, the knot elevation is one more
        counts[0] = newOrd # But not at the endpoints, which are full order as usual
        counts[-1] = newOrd # But not at the endpoints, which are full order as usual
        newKnots.append(np.repeat(uniqueKnots, counts))
        # Also calculate the total number of coefficients, capturing how it progressively increases, and
        # using that calculation to span uvw from the starting knot to the end for each variable.
        nCoef = len(newKnots[-1]) - newOrder[-1]
        totalCoefs.append(totalCoefs[-1] * nCoef)
        knotAverages = bspy.Spline(1, 0, [newOrd], [nCoef], [newKnots[-1]], []).greville()
        for iKnot in range(1, len(knotAverages) - 1):
            if knotAverages[iKnot] == knotAverages[iKnot + 1]:
                knotAverages[iKnot] = 0.5 * (knotAverages[iKnot - 1] + knotAverages[iKnot])
                knotAverages[iKnot + 1] = 0.5 * (knotAverages[iKnot + 1] + knotAverages[iKnot + 2])
        uvwValues.append(knotAverages)
        nCoefs.append(nCoef)
    points = []
    ijk = [0 for order in self.order]
    for i in range(totalCoefs[-1]):
        uvw = [uvwValues[j][k] for j, k in enumerate(ijk)]
        points.append(self.normal(uvw, False, indices))
        for j, nCoef in enumerate(totalCoefs[:-1]):
            if (i + 1) % nCoef == 0:
                ijk[j] += 1
                if j > 0:
                    ijk[j - 1] = 0
    points = np.array(points).T
    nDep = max(self.nInd, self.nDep) if indices is None else len(indices)
    nCoefs.reverse()
    points = np.reshape(points, [nDep] + nCoefs)
    points = np.transpose(points, [0] + list(range(self.nInd, 0, -1)))
    return bspy.Spline.least_squares(uvwValues, points, order = newOrder, knots = newKnots, metadata = self.metadata)

def rotate(self, vector, angle):
    vector = np.atleast_1d(vector)
    vector = vector / np.linalg.norm(vector)
    if len(vector) != 3:  raise ValueError("Rotation vector must have 3 components")
    if self.nDep != 3:  raise ValueError("Spline must have exactly 3 dependent variables")
    radians = np.pi * angle / 180.0
    cost = np.cos(radians)
    sint = np.sin(radians)
    kMat = np.array([[0.0, -vector[2], vector[1]],
                     [vector[2], 0.0, -vector[0]],
                     [-vector[1], vector[0], 0.0]])
    rotMat = np.identity(3) + sint * kMat + (1.0 - cost) * kMat @ kMat
    return list(rotMat) @ self

def scale(self, multiplier):
    if isinstance(multiplier, bspy.Spline):
        return self.multiply(multiplier, [(ix, ix) for ix in range(min(self.nInd, multiplier.nInd))], 'S')
    else:
        if np.isscalar(multiplier):
            nDep = self.nDep
            coefs = multiplier * self.coefs
        elif len(multiplier) == self.nDep:
            nDep = self.nDep
            coefs = np.array(self.coefs)
            for i in range(nDep):
                coefs[i] *= multiplier[i]
        elif self.nDep == 1:
            nDep = len(multiplier)
            coefs = np.empty((nDep, *self.coefs.shape[1:]), self.coefs.dtype)
            for i in range(nDep):
                coefs[i] = multiplier[i] * self.coefs[0]
        else:
            raise ValueError("Invalid multiplier")
        return type(self)(self.nInd, nDep, self.order, self.nCoef, self.knots, coefs, self.metadata)

def transform(self, matrix):
    if not(matrix.ndim == 2 and matrix.shape[1] == self.nDep): raise ValueError("Invalid matrix")

    swapped = np.swapaxes(self.coefs, 0, -2)
    newCoefs = np.swapaxes(matrix @ swapped, 0, -2)
    return type(self)(self.nInd, matrix.shape[0], self.order, self.nCoef, self.knots, newCoefs, self.metadata)

def translate(self, translationVector):
    translationVector = np.atleast_1d(translationVector)
    if not(len(translationVector) == self.nDep): raise ValueError("Invalid translationVector")

    coefs = np.array(self.coefs)
    for i in range(self.nDep):
        coefs[i] += translationVector[i]
    return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, coefs, self.metadata)