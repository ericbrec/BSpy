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

def multiply(self, other, indMap = None):
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