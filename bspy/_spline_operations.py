import numpy as np
import bspy.spline

def add(self, other, indMap = None):
    assert self.nDep == other.nDep
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
            permutation.append(nInd + 1) # Add 1 to account for dependent variables.
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

def differentiate(self, with_respect_to = 0):
    assert 0 <= with_respect_to < self.nInd
    assert self.order[with_respect_to] > 1

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

def cross(self, vector):
    if isinstance(vector, bspy.spline.Spline):
        return self.multiply(vector, None, 'C')
    elif self.nDep == 3:
        assert len(vector) == self.nDep

        coefs = np.empty(self.coefs.shape, self.coefs.dtype)
        coefs[0] = vector[2] * self.coefs[1] - vector[1] * self.coefs[2]
        coefs[1] = vector[0] * self.coefs[2] - vector[2] * self.coefs[0]
        coefs[2] = vector[1] * self.coefs[0] - vector[0] * self.coefs[1]
        return type(self)(self.nInd, 3, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)
    else:
        assert self.nDep == 2
        assert len(vector) == self.nDep

        coefs = np.empty((1, *self.coefs.shape[1:]), self.coefs.dtype)
        coefs[0] = vector[1] * self.coefs[0] - vector[0] * self.coefs[1]
        return type(self)(self.nInd, 3, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

def dot(self, vector):
    if isinstance(vector, bspy.spline.Spline):
        return self.multiply(vector, None, 'D')
    else:
        assert len(vector) == self.nDep

        coefs = vector[0] * self.coefs[0]
        for i in range(1, self.nDep):
            coefs += vector[i] * self.coefs[i]
        return type(self)(self.nInd, 1, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

def multiply(self, other, indMap = None, productType = 'S'):
    assert productType == 'C' or productType == 'D' or productType == 'S', "productType must be 'C', 'D' or 'S'"

    assert productType != 'D' or self.nDep == other.nDep, "Mismatched dimensions"
    assert productType != 'C' or (self.nDep == other.nDep and 2 <= self.nDep <= 3), "Mismatched dimensions"
    assert productType != 'S' or self.nDep == 1 or other.nDep == 1, "Mismatched dimensions"

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
        # Now we need to combine like terms for matching independent variables (variables mapped to each other).
        # We can't do this directly with b-spline coefficients, but we can indirectly with the following steps:
        #   1) Use the combined knots from matching independent variables to divide the spline into segments.
        #   2) Convert each spline segment into a polynomial (Taylor series).
        #   3) Sum coefficients of matching polynomial degree (the coefficients have already been multiplied together).
        #   4) Use blossoms to compute the spline segment coefficients from the polynomial segment (uses the raceme function from E.T.Y. Lee).

        for (ind1, ind2) in indMap:
            # 1) Use the combined knots from matching independent variables to divide the spline into segments.

            # First, get multiplicities of the knots for each independent variable.
            order1 = self.order[ind1]
            knots1, multiplicities1 = np.unique(self.knots[ind1][order1-1:self.nCoef[ind1]+1], return_counts=True)
            multiplicities1[0] = multiplicities1[-1] = order1
            order2 = other.order[ind2]
            knots2, multiplicities2 = np.unique(other.knots[ind2][order2-1:other.nCoef[ind2]+1], return_counts=True)
            multiplicities2[0] = multiplicities2[-1] = order2
            assert knots1[0] == knots2[0] and knots1[-1] == knots2[-1], f"self[{ind1}] domain doesn't match other[{ind2}]"

            # Compute the new order of the combined spline and its new knots array.
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

            # Update nInd, order, nCoef, and overall knots
            nInd -= 1
            del order[self.nInd + ind2]
            order[ind1] = newOrder
            del nCoef[self.nInd + ind2]
            nCoef[ind1] = len(newKnots) - newOrder
            del knots[self.nInd + ind2]
            knots[ind1] = np.array(newKnots, knots1.dtype)

            # Compute segments (uses the III algorithm from E.T.Y. Lee)
            i = 0
            segments = [i]
            j = 0
            sigma = newMultiplicities[j]
            while i < nCoef[ind1]:
                while sigma <= segments[-1] + newOrder:
                    j += 1
                    i = sigma
                    sigma += newMultiplicities[j]
                segments.append(i)

            # Move the two independent variables to the left side of the coefficients array in prep for computing Taylor coefficients,
            #   and initialize new coefficients array.
            coefs = np.moveaxis(coefs, (ind1+1, self.nInd+1 + ind2), (0, 1))
            newCoefs = np.empty(((len(segments) - 1) * newOrder, *coefs.shape[2:]), coefs.dtype)

            # Loop through the segments
            segmentStart = segments[0]
            for segmentEnd in segments[1:]:
                # 2) Convert each spline segment into a polynomial (Taylor series).

                # Isolate the appropriate segment coefficients
                knot = newKnots[segmentStart + 1]
                ix1 = np.searchsorted(self.knots[ind1], knot, 'right')
                ix1 = min(ix1, self.nCoef[ind1])
                ix2 = np.searchsorted(other.knots[ind2], knot, 'right')
                ix2 = min(ix2, other.nCoef[ind2])
                taylorCoefs = (coefs[ix1 - order1:ix1, ix2 - order2:ix2]).T # Transpose so we multiply on the left (due to matmul rules)

                # Compute taylor coefficients for the segment
                bValues = np.empty((order1, order1), knots1.dtype)
                for derivativeOrder in range(order1):
                    bValues[:,derivativeOrder] = bspy.Spline.bsplineValues(ix1, self.knots[ind1], order1, knot, derivativeOrder, True)
                taylorCoefs = taylorCoefs @ bValues
                taylorCoefs = np.moveaxis(taylorCoefs, -1, 0) # Move ind1's taylor coefficients to the left side so we can compute ind2's
                bValues = np.empty((order2, order2), knots2.dtype)
                for derivativeOrder in range(order2):
                    bValues[:,derivativeOrder] = bspy.Spline.bsplineValues(ix2, other.knots[ind2], order2, knot, derivativeOrder, True)
                taylorCoefs = taylorCoefs @ bValues
                taylorCoefs = (np.moveaxis(taylorCoefs, 0, -1)).T # Move ind1's taylor coefficients back to the right side, and re-transpose

                # 3) Sum coefficients of matching polynomial degree (the coefficients have already been multiplied together by the outer product).
                a = newCoefs[segmentStart:segmentStart + newOrder]
                a.fill(0.0)
                for i2 in range(order2):
                    for i1 in range(order1):
                        a[i1 + i2] += taylorCoefs[i1, i2]

                # 4) Use blossoms to compute the spline segment coefficients from the polynomial segment (uses the raceme function from E.T.Y. Lee).
                m = newOrder - 1
                rho = segmentEnd - segmentStart
                for j in range(m):
                    for i in range(min(newOrder, m - j)):
                        a[i] = (1 - i/(m - j)) * a[i] + ((i + 1)/(m - j)) * (newKnots[segmentStart + m - j] - knot) * a[i + 1]
                for j in range(rho - 1):
                    for i in range(min(newOrder + j, rho - 1), j, -1):
                        a[i] = a[i - 1] + (newKnots[segmentStart + m + j + 1] - newKnots[segmentStart + i]) * a[i]
                
                # Move to next segment
                segmentStart = segmentEnd

            # All the segment coefficients are computed.
            # Now move combined independent variable back to its original axis.
            coefs = np.moveaxis(newCoefs[:nCoef[ind1]], 0, ind1 + 1)

    return type(self)(nInd, nDep, order, nCoef, knots, coefs, self.accuracy + other.accuracy, self.metadata)

def scale(self, multiplier):
    if isinstance(multiplier, bspy.spline.Spline):
        return self.multiply(multiplier, None, 'S')
    else:
        assert np.isscalar(multiplier) or len(multiplier) == self.nDep

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
    assert matrix.ndim == 2 and matrix.shape[1] == self.nDep

    if maxSingularValue is None:
        maxSingularValue = np.linalg.svd(matrix, compute_uv=False)[0]

    return type(self)(self.nInd, matrix.shape[0], self.order, self.nCoef, self.knots, matrix @ self.coefs, maxSingularValue * self.accuracy, self.metadata)

def translate(self, translationVector):
    assert len(translationVector) == self.nDep

    coefs = np.array(self.coefs)
    for i in range(self.nDep):
        coefs[i] += translationVector[i]
    return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)