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
        (self, other) = self.common_basis((other,), indMap)
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
            order.append(other.order[i])
            nCoef.append(other.nCoef[i])
            knots.append(other.knots[i])
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
    
    return type(self)(nInd, self.nDep, order, nCoef, knots, coefs, self.accuracy + other.accuracy, self.metadata)

def contract(self, uvw):
    nInd = self.nInd
    order = [*self.order]
    nCoef = [*self.nCoef]
    knots = [*self.knots]
    domain = self.domain()
    section = [slice(None)]
    indices = []
    for iv in range(self.nInd):
        if uvw[iv] is not None:
            if uvw[iv] < domain[iv][0] or uvw[iv] > domain[iv][1]:
                raise ValueError(f"Spline evaluation outside domain: {uvw}")

            # Grab all of the appropriate coefficients
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            indices.append(ix)
            section.append(slice(ix - self.order[iv], ix))
        else:
            section.append(slice(None))

    coefs = self.coefs[tuple(section)]
    ix = 0
    for iv in range(self.nInd):
        if uvw[iv] is not None:
            nInd -= 1
            del order[ix]
            del nCoef[ix]
            del knots[ix]
            bValues = bspy.Spline.bspline_values(indices.pop(0), self.knots[iv], self.order[iv], uvw[iv])
            coefs = np.moveaxis(coefs, ix + 1, -1)
            coefs = coefs @ bValues
        else:
            ix += 1
    
    return type(self)(nInd, self.nDep, order, nCoef, knots, coefs, self.accuracy, self.metadata)

def cross(self, vector):
    if isinstance(vector, bspy.Spline):
        return self.multiply(vector, None, 'C')
    elif self.nDep == 3:
        if not(len(vector) == self.nDep): raise ValueError("Invalid vector")

        coefs = np.empty(self.coefs.shape, self.coefs.dtype)
        coefs[0] = vector[2] * self.coefs[1] - vector[1] * self.coefs[2]
        coefs[1] = vector[0] * self.coefs[2] - vector[2] * self.coefs[0]
        coefs[2] = vector[1] * self.coefs[0] - vector[0] * self.coefs[1]
        return type(self)(self.nInd, 3, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)
    else:
        if not(self.nDep == 2): raise ValueError("Invalid nDep")
        if not(len(vector) == self.nDep): raise ValueError("Invalid vector")

        coefs = np.empty((1, *self.coefs.shape[1:]), self.coefs.dtype)
        coefs[0] = vector[1] * self.coefs[0] - vector[0] * self.coefs[1]
        return type(self)(self.nInd, 3, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

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
    
    return type(self)(self.nInd, self.nDep, order, nCoef, knots, newCoefs.swapaxes(0, with_respect_to + 1), self.accuracy, self.metadata)

def dot(self, vector):
    if isinstance(vector, bspy.Spline):
        return self.multiply(vector, None, 'D')
    else:
        if not(len(vector) == self.nDep): raise ValueError("Invalid vector")

        coefs = vector[0] * self.coefs[0]
        for i in range(1, self.nDep):
            coefs += vector[i] * self.coefs[i]
        return type(self)(self.nInd, 1, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

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

    return type(self)(self.nInd, self.nDep, order, nCoef, knots, newCoefs.swapaxes(0, with_respect_to + 1), self.accuracy, self.metadata)

def multiplyAndConvolve(self, other, indMap = None, productType = 'S'):
    if not(productType == 'C' or productType == 'D' or productType == 'S'): raise ValueError("productType must be 'C', 'D' or 'S'")

    if not(productType != 'D' or self.nDep == other.nDep): raise ValueError("Mismatched dimensions")
    if not(productType != 'C' or (self.nDep == other.nDep and 2 <= self.nDep <= 3)): raise ValueError("Mismatched dimensions")
    if not(productType != 'S' or self.nDep == 1 or other.nDep == 1): raise ValueError("Mismatched dimensions")

    # Ensure scalar spline (if any) comes first (simplifies array processing).
    if other.nDep == 1 and self.nDep > 1:
        temp = self
        self = other
        other = temp

    # Construct new spline parameters.
    nInd = self.nInd + other.nInd
    nDep = other.nDep
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
    else: # Scalar product, where self is the scalar
        coefs = outer[0]

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
                # We can't do this directly with b-spline coefficients, but we can indirectly with the following steps:
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
                # We can't do this directly with b-spline coefficients, but we can indirectly with the following steps:
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
                            bValues[:,derivativeOrder] = bspy.Spline.bspline_values(ix1, self.knots[ind1], order1, knot, derivativeOrder, True)
                        taylorCoefs = taylorCoefs @ bValues
                        taylorCoefs = np.moveaxis(taylorCoefs, -1, 0) # Move ind1's taylor coefficients to the left side so we can compute ind2's
                        bValues = np.empty((order2, order2), knots2.dtype)
                        for derivativeOrder in range(order2):
                            bValues[:,derivativeOrder] = bspy.Spline.bspline_values(ix2, other.knots[ind2], order2, yLeft, derivativeOrder, True)
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
                    ix1 = np.searchsorted(self.knots[ind1], knot, 'right')
                    ix1 = min(ix1, self.nCoef[ind1])
                    ix2 = np.searchsorted(other.knots[ind2], knot, 'right')
                    ix2 = min(ix2, other.nCoef[ind2])

                    # Compute taylor coefficients for the segment
                    taylorCoefs = (coefs[ix1 - order1:ix1, ix2 - order2:ix2]).T # Transpose so we multiply on the left (due to matmul rules)
                    bValues = np.empty((order1, order1), knots1.dtype)
                    for derivativeOrder in range(order1):
                        bValues[:,derivativeOrder] = bspy.Spline.bspline_values(ix1, self.knots[ind1], order1, knot, derivativeOrder, True)
                    taylorCoefs = taylorCoefs @ bValues
                    taylorCoefs = np.moveaxis(taylorCoefs, -1, 0) # Move ind1's taylor coefficients to the left side so we can compute ind2's
                    bValues = np.empty((order2, order2), knots2.dtype)
                    for derivativeOrder in range(order2):
                        bValues[:,derivativeOrder] = bspy.Spline.bspline_values(ix2, other.knots[ind2], order2, knot, derivativeOrder, True)
                    taylorCoefs = taylorCoefs @ bValues
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

    return type(self)(nInd, nDep, order, nCoef, knots, coefs, self.accuracy + other.accuracy, self.metadata)

def scale(self, multiplier):
    if isinstance(multiplier, bspy.Spline):
        return self.multiply(multiplier, None, 'S')
    else:
        if not(np.isscalar(multiplier) or len(multiplier) == self.nDep): raise ValueError("Invalid multiplier")

        if np.isscalar(multiplier):
            accuracy = multiplier * self.accuracy
            coefs = multiplier * self.coefs
        else:
            accuracy = np.linalg.norm(multiplier) * self.accuracy
            coefs = np.array(self.coefs)
            for i in range(self.nDep):
                coefs[i] *= multiplier[i]
        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, coefs, accuracy, self.metadata)

def transform(self, matrix, maxSingularValue=None):
    if not(matrix.ndim == 2 and matrix.shape[1] == self.nDep): raise ValueError("Invalid matrix")

    if maxSingularValue is None:
        maxSingularValue = np.linalg.svd(matrix, compute_uv=False)[0]

    return type(self)(self.nInd, matrix.shape[0], self.order, self.nCoef, self.knots, matrix @ self.coefs, maxSingularValue * self.accuracy, self.metadata)

def translate(self, translationVector):
    if not(len(translationVector) == self.nDep): raise ValueError("Invalid translationVector")

    coefs = np.array(self.coefs)
    for i in range(self.nDep):
        coefs[i] += translationVector[i]
    return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)