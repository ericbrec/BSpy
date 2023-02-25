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

def multiply(self, other, indMap = None, productType = 'S'):
    assert productType == 'C' or productType == 'D' or productType == 'S', "productType must be 'C', 'D' or 'S'"

    assert productType != 'D' or self.nDep == other.nDep, "Mismatched dimensions"
    assert productType != 'C' or (self.nDep == 3 and other.nDep == 3), "Mismatched dimensions"
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
        coefs = np.empty(outer[0].shape, outer.dtype)
        coefs[0] = outer[1,2] - outer[2,1]
        coefs[1] = outer[2,0] - outer[0,2]
        coefs[2] = outer[0,1] - outer[1,0]
    elif productType == 'D': # Dot product
        coefs = outer[0,0]
        for i in range(1, self.nDep):
            coefs += outer[i,i]
        coefs = np.expand_dims(coefs, axis=0)
        nDep = 1
    else: # Scalar product, where self is the scalar
        coefs = outer[0]

    # Construct the new spline (= self * other).
    spline = type(self)(nInd, nDep, order, nCoef, knots, coefs, max(self.accuracy, other.accuracy), self.metadata)

    if indMap is not None:
        # Now we need to combine like terms for matching independent variables (variables mapped to each other).
        # We can't do this directly with b-spline coefficients, but we can indirectly with the following steps:
        #   1) Use the combined knots from matching independent variables to divide the spline into segments.
        #   2) Convert each spline segment into a polynomial (Taylor series).
        #   3) Sum coefficients of matching polynomial degree (the coefficients have already been multiplied together).
        #   4) Use blossoms to compute the coefficients from the resulting polynomial (uses the raceme function from E.T.Y. Lee).

        for (ind1, ind2) in indMap:
            # 1) Use the combined knots from matching independent variables to divide the spline into segments.
            # First, get multiplicities of the knots for each independent variable.
            order1 = self.order[ind1]
            knots1, counts1 = np.unique(self.knots[ind1][order1-1:self.nCoef[ind1]+1], return_counts=True)
            counts1[0] = counts1[-1] = order1
            order2 = other.order[ind2]
            knots2, counts2 = np.unique(other.knots[ind2][order2-1:other.nCoef[ind2]+1], return_counts=True)
            counts2[0] = counts2[-1] = order2
            assert knots1[0] == knots2[0] and knots1[-1] == knots2[-1], f"self[{ind1}] domain doesn't match other[{ind2}]"

            # Move the two independent variables to the right side of the coefficients array in prep for computing Taylor coefficients.
            coefs = np.moveaxis(coefs, (ind1+1, self.nInd+1 + ind2+1), (-2, -1))

            # Compute the new order of the combined spline and its new knots
            newOrder = order1 + order2 - 1
            newKnots = []
            i1 = i2 = 0
            while i1 < len(knots1) and i2 < len(knots2):
                if knots1[i1] < knots2[i2]:
                    knot = knots1[i1]
                    count = counts1[i1] + order2 - 1
                    i1 += 1
                elif knots[i1] > knots2[i2]:
                    knot = knots2[i2]
                    count = counts2[i2] + order1 - 1
                    i2 += 1
                else:
                    knot = knots1[i1]
                    count = max(counts1[i1] + order2 - 1, counts2[i2] + order1 - 1)
                    i1 += 1
                    i2 += 1
                newKnots += [knot] * count

                # 2) Convert each spline segment into a polynomial (Taylor series).
                taylorCoefs = spline.derivatives((order1 - 1, order2 - 1),(knot, knot), taylorCoefs = True)


            
            # 3) Sum coefficients of matching polynomial degree (the coefficients have already been multiplied together).
            # 4) Use blossoms to compute the coefficients from the resulting polynomial (uses the raceme function from E.T.Y. Lee).


    return spline

def scale(self, multiplier):
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