import numpy as np
from os import path
from collections import namedtuple
from bspy.error import *

def _isIterable(object):
    result = True
    try:
        iterator = iter(object)
    except TypeError:
        result = False
    return result

class Spline:
    """
    A class to model, represent, and process piecewise polynomial tensor product
    functions (spline functions) as linear combinations of B-splines.  The
    attributes of this class have the following meaning: 

    nInd - the number of independent variables of the spline
    nDep - the number of dependent variables of the spline
    order - a tuple of length nInd where each integer entry represents the
            polynomial order of the function in that variable
    nCoef - a tuple of length nInd where each integer entry represents the
            dimension (i.e. number of B-spline coefficients) of the function
            space in that variable
    knots - a list of the lists of the knots of the spline in each independent
            variable
    coefs - a list of the B-spline coefficients of the spline
    accuracy - each spline function is presumed to be an approximation of
               something else.  This attribute stores the infinity norm error of
               the difference between the given spline function and that
               something else
    metadata - a dictionary of ancillary data to store with the spline 
    """
    
    def __init__(self, nInd = 1, nDep = 1, order = [4], nCoef = [4],
                 knots = [[0, 0, 0, 0, 1, 1, 1, 1]],
                 coefs = [[0, 0, 0 ,1]], accuracy = 0.0, metadata = {}):
        assert nInd >= 0, "nInd < 0"
        self.nInd = int(nInd)
        assert nDep >= 0, "nDep < 0"
        self.nDep = int(nDep)
        assert len(order) == self.nInd, "len(order) != nInd"
        self.order = tuple(int(x) for x in order)
        assert len(nCoef) == self.nInd, "len(nCoef) != nInd"
        self.nCoef = tuple(int(x) for x in nCoef)
        assert len(knots) == nInd, "len(knots) != nInd"
        for i in range(len(knots)):
            nKnots = self.order[i] + self.nCoef[i]
            assert len(knots[i]) == nKnots, \
                f"Knots array for variable {i} should have length {nKnots}"
        self.knots = tuple(np.array(kk) for kk in knots)
        for knots, order, nCoef in zip(self.knots, self.order, self.nCoef):
            for i in range(nCoef):
                assert knots[i] <= knots[i + 1] and knots[i] < knots[i + order],\
                       "Improperly ordered knot sequence"
        totalCoefs = 1
        for nCoef in self.nCoef:
            totalCoefs *= nCoef
        assert len(coefs) == totalCoefs or len(coefs) == self.nDep, \
            f"Length of coefs should be {totalCoefs} or {self.nDep}"
        self.coefs = np.array(coefs)
        if self.coefs.shape != (self.nDep, *self.nCoef):
            if len(self.coefs) == totalCoefs:
                self.coefs = self.coefs.reshape((*self.nCoef[::-1], self.nDep)).T
            else:
                self.coefs = np.array([c.T for c in self.coefs]).reshape((self.nDep, *self.nCoef))
        self.accuracy = accuracy
        self.metadata = metadata

    def __call__(self, uvw):
        return self.evaluate(uvw)

    def __repr__(self):
        return f"Spline({self.nInd}, {self.nDep}, {self.order}, " + \
               f"{self.nCoef}, {self.knots} {self.coefs}, {self.accuracy}, " + \
               f"{self.metadata})"

    def __add__(self, other):
        if isinstance(other, Spline):
            return self.add(other)
        elif _isIterable(other):
            return self.translate(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Spline):
            return other.add(self)
        elif _isIterable(other):
            return self.translate(other)
        else:
            return NotImplemented

    def __rmatmul__ (self, other):
        if _isIterable(other):
            if isinstance(other, np.ndarray) and len(other.shape) == 2:
                return self.transform(other)
            else:
                return self.dot(other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other) or _isIterable(other):
            return self.scale(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other) or _isIterable(other):
            return self.scale(other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Spline):
            return self.subtract(other)
        elif _isIterable(other):
            return self.translate(-np.array(other))
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Spline):
            return other.subtract(self)
        elif _isIterable(other):
            spline = self.scale(-1.0)
            return spline.translate(other)
        else:
            return NotImplemented

    def add(self, other, indMap = None):
        """
        Add two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to add to self. The number of dependent variables must match self.

        indMap : `iterable` or `None` (default)
            An iterable of pairs of indices. 
            Each pair (n, m) maps the mth independent variable of other to the nth independent variable of self. 
            The domains of the nth and mth independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            Unmapped independent variables remain independent (the default).

        Returns
        -------
        spline : `Spline`
            The result of adding other to self.

        See Also
        --------
        `subtract` : Subtract two splines.
        `common_basis : Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Notes
        -----
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
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

    def clamp(self, left, right):
        """
        Ensure the leftmost/rightmost knot has a full order multiplicity, clamping the spline's 
        value at the first/last knot to its first/last coefficient.

        Parameters
        ----------
        left : `iterable`
            An iterable of independent variables to clamp on the left side.

        right : `iterable`
            An iterable of independent variables to clamp on the right side.

        Returns
        -------
        spline : `Spline`
            The clamped spline. If the spline was already clamped, the original spline is returned.

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        """
        newKnots = self.nInd * [[]]

        for ind in left:
            insertions = self.order[ind]
            for i in range(len(self.knots[ind])):
                if self.knots[ind][i] > self.knots[ind][0]:
                    break
                else:
                    insertions -= 1
            newKnots[ind] += insertions * [self.knots[ind][0]]

        for ind in right:
            insertions = self.order[ind]
            for i in range(len(self.knots[ind])):
                if self.knots[ind][-i - 1] < self.knots[ind][-1]:
                    break
                else:
                    insertions -= 1
            newKnots[ind] += insertions * [self.knots[ind][-1]]

        return self.insert_knots(newKnots)

    def common_basis(self, splines, indMap):
        """
        Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Parameters
        ----------
        splines : `iterable`
            The collection of N - 1 splines to align (N total splines, including self).

        indMap : `iterable`
            The collection of independent variables to align. Since each spline can have multiple 
            independent variables, `indMap` is an `iterable` of `iterables` (like a list of lists). 
            Each collection of indices (i0, i1, .. iN) maps the i'th independent variable to each other. 
            The domains of mapped independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            If all the splines are curves (1 independent variable), then `indMap` is ((0, 0, .. 0),).

        Returns
        -------
        splines : `tuple`
            The aligned collection of N splines.

        See Also
        --------
        `elevate_and_insert_knots` : Elevate a spline and insert new knots.

        Notes
        -----
        Uses `elevate_and_insert_knots` to ensure mapped variables share the same order and knots. 
        """
        # Step 1: Compute the order for each aligned independent variable.
        orders = []
        splines = (self, *splines)
        for map in indMap:
            assert len(map) == len(splines)
            order = 0
            for (spline, ind) in zip(splines, map):
                order = max(order, spline.order[ind])
            orders.append(order)
        
        # Step 2: Compute knot multiplicities for each aligned independent variable.
        knots = []
        for (map, order) in zip(indMap, orders):
            multiplicities = []
            leftKnot = splines[0].knots[map[0]][0]
            rightKnot = splines[0].knots[map[0]][-1]
            for (spline, ind) in zip(splines, map):
                assert spline.knots[ind][0] == leftKnot
                assert spline.knots[ind][-1] == rightKnot
                uniqueKnots, counts = np.unique(spline.knots[ind], return_counts=True)
                match = 0
                for (knot, count) in zip(uniqueKnots, counts):
                    while match < len(multiplicities) and knot > multiplicities[match][0]:
                        match += 1
                    if match == len(multiplicities) or knot < multiplicities[match][0]:
                        # Account for elevation increase in knot count.
                        multiplicities.insert(match, [knot, count + order - spline.order[ind]])
                    else:
                        # Account for elevation increase in knot count.
                        multiplicities[match][1] = max(multiplicities[match][1], count + order - spline.order[ind])
                        match += 1
            knots.append(multiplicities)

        # Step 3: Elevate and insert missing knots to each spline accordingly.
        alignedSplines = []
        for (spline, i) in zip(splines, range(len(splines))):
            m = spline.nInd * [0]
            newKnots = spline.nInd * [[]]
            for (map, order, multiplicities) in zip(indMap, orders, knots):
                ind = map[i]
                m[ind] = order - spline.order[ind]
                uniqueKnots, counts = np.unique(spline.knots[ind], return_counts=True)
                match = 0
                for (knot, count) in zip(uniqueKnots, counts):
                    while match < len(multiplicities) and knot > multiplicities[match][0]:
                        newKnots[ind] += multiplicities[match][1] * [multiplicities[match][0]]
                        match += 1
                    assert knot == multiplicities[match][0]
                    newKnots[ind] += (multiplicities[match][1] - count) * [knot]
                    match += 1
                while match < len(multiplicities) and knot > multiplicities[match][0]:
                    newKnots[ind] += multiplicities[match][1] * [multiplicities[match][0]]
                    match += 1
            spline = spline.elevate_and_insert_knots(m, newKnots)
            alignedSplines.append(spline)

        return tuple(alignedSplines)

    def derivative(self, with_respect_to, uvw):
        """
        Compute the derivative of the spline at a given parameter value.

        Parameters
        ----------
        with_respect_to : `iterable`
            An iterable of length `nInd` that specifies the integer order of derivative for each independent variable.
            A zero-order derivative just evaluates the spline normally.
        
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter value).

        Returns
        -------
        value : `numpy.array`
            The value of the derivative of the spline at the given parameter value.

        See Also
        --------
        `evaluate` : Compute the value of the spline at a given parameter value.
        `differentiate` : Differentiate a spline with respect to one of its independent variables, returning the resulting spline.

        Notes
        -----
        The derivative method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline function.  The non-zero B-splines are
        evaluated, then the dot product of those B-splines with the vector of
        B-spline coefficients is computed.
        """
        def b_spline_values(knot, knots, splineOrder, derivativeOrder, u):
            basis = np.zeros(splineOrder, knots.dtype)
            basis[-1] = 1.0
            for degree in range(1, splineOrder - derivativeOrder):
                b = splineOrder - degree
                for i in range(knot - degree, knot):
                    alpha = (u - knots[i]) / (knots[i + degree] - knots[i])
                    basis[b - 1] += (1.0 - alpha) * basis[b]
                    basis[b] *= alpha
                    b += 1
            for degree in range(splineOrder - derivativeOrder, splineOrder):
                b = splineOrder - degree
                for i in range(knot - degree, knot):
                    alpha = degree / (knots[i + degree] - knots[i])
                    basis[b - 1] += -alpha * basis[b]
                    basis[b] *= alpha
                    b += 1
            return basis

        # Check for evaluation point inside domain
        dom = self.domain()
        for ix in range(self.nInd):
            if uvw[ix] < dom[ix][0] or uvw[ix] > dom[ix][1]:
                raise ArgumentOutsideDomainError(uvw)

        # Grab all of the appropriate coefficients
        mySection = [slice(0, self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(slice(ix - self.order[iv], ix))
        myCoefs = self.coefs[mySection]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = b_spline_values(myIndices[iv], self.knots[iv], self.order[iv], with_respect_to[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs

    def differentiate(self, with_respect_to = 0):
        """
        Differentiate a spline with respect to one of its independent variables, returning the resulting spline.

        Parameters
        ----------
        with_respect_to : integer
            The number of the independent variable to differentiate.

        Returns
        -------
        spline : `Spline`
            The spline that results from differentiating the original spline with respect to the given independent variable.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at a given parameter value.
        """
        assert 0 <= with_respect_to < self.nInd
        assert self.order[with_respect_to] > 1

        order = [*self.order]
        order[with_respect_to] -= 1
        degree = order[with_respect_to] 

        nCoef = [*self.nCoef]
        nCoef[with_respect_to] -= 1

        dKnots = self.knots[with_respect_to][1:-1]
        knots = [self.knots]
        knots[with_respect_to] = dKnots

        coefs = np.delete(self.coefs, 0, axis=with_respect_to + 1) # first axis is the dependent variable
        sliceI = (self.nInd + 1) * [slice(None)]
        for i in range(nCoef[with_respect_to]):
            sliceI[with_respect_to + 1] = i
            alpha =  degree / (dKnots[i+degree] - dKnots[i])
            coefs[sliceI] = alpha * (coefs[sliceI] - self.coefs[sliceI])
        
        return type(self)(self.nInd, self.nDep, order, nCoef, knots, coefs, self.accuracy, self.metadata)

    def domain(self):
        """
        Return the domain of a spline.

        Returns
        -------
        bounds : `numpy.array`
            nInd x 2 array of the upper and lower bounds on each of the independent variables.

        See Also
        --------
        `reparametrize` : Reparametrize a spline to match new domain bounds
        `trim` : Trim the domain of a spline.
        """
        dom = [[self.knots[i][self.order[i] - 1],
                self.knots[i][self.nCoef[i]]] for i in range(self.nInd)]
        return np.array(dom)

    def dot(self, vector):
        """
        Dot product a spline by the given vector.

        Parameters
        ----------
        vector : array-like
            An array of length `nDep` that specifies the vector.

        Returns
        -------
        spline : `Spline`
            The dotted spline.
        """
        assert len(vector) == self.nDep

        coefs = vector[0] * self.coefs[0]
        for i in range(1, self.nDep):
            coefs += vector[i] * self.coefs[i]
        return type(self)(self.nInd, 1, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

    def evaluate(self, uvw):
        """
        Compute the value of the spline at a given parameter value.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter value).

        Returns
        -------
        value : `numpy.array`
            The value of the spline at the given parameter value.

        See Also
        --------
        `derivative` : Compute the derivative of the spline at a given parameter value.

        Notes
        -----
        The evaluate method uses the de Boor recurrence relations for a B-spline
        series to evaluate a spline function.  The non-zero B-splines are
        evaluated, then the dot product of those B-splines with the vector of
        B-spline coefficients is computed.
        """
        def b_spline_values(knot, knots, order, u):
            basis = np.zeros(order, knots.dtype)
            basis[-1] = 1.0
            for degree in range(1, order):
                b = order - degree
                for i in range(knot - degree, knot):
                    alpha = (u - knots[i]) / (knots[i + degree] - knots[i])
                    basis[b - 1] += (1.0 - alpha) * basis[b]
                    basis[b] *= alpha
                    b += 1
            return basis

        # Check for evaluation point inside domain
        dom = self.domain()
        for ix in range(self.nInd):
            if uvw[ix] < dom[ix][0] or uvw[ix] > dom[ix][1]:
                raise ArgumentOutsideDomainError(uvw)

        # Grab all of the appropriate coefficients
        mySection = [slice(0, self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(slice(ix - self.order[iv], ix))
        myCoefs = self.coefs[mySection]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = b_spline_values(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs

    def elevate(self, m):
        """
        Elevate a spline, increasing its order by `m`.

        Parameters
        ----------
        m : `iterable` of length `nInd`
            An iterable that specifies the non-negative integer amount to increase the order 
            for each independent variable of the spline.

        Returns
        -------
        spline : `Spline`
            A spline with the order of the current spline plus `m`.

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `elevate_and_insert_knots` : Elevate a spline and insert new knots.

        Notes
        -----
        Implements the algorithm from Huang, Qi-Xing, Shi-Min Hu, and Ralph R. Martin. 
        "Fast degree elevation and knot insertion for B-spline curves." Computer Aided Geometric Design 22, no. 2 (2005): 183-197.
        """
        return self.elevate_and_insert_knots(m, self.nInd * [[]])

    def elevate_and_insert_knots(self, m, newKnots):
        """
        Elevate a spline and insert new knots.

        Parameters
        ----------
        m : `iterable` of length `nInd`
            An iterable that specifies the non-negative integer amount to increase the order 
            for each independent variable of the spline.

        newKnots : `iterable` of length `nInd`
            An iterable that specifies the knots to be added to each independent variable's knots. 
            len(newKnots[ind]) == 0 if no knots are to be added for the `ind` independent variable.

        Returns
        -------
        spline : `Spline`
            A spline with the order of the current spline plus `m` that includes the new knots.

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `clamp` : Clamp the left and/or right side of a spline.
        `elevate` : Elevate a spline, increasing its order by `m`.

        Notes
        -----
        Implements the algorithm from Huang, Qi-Xing, Shi-Min Hu, and Ralph R. Martin. 
        "Fast degree elevation and knot insertion for B-spline curves." Computer Aided Geometric Design 22, no. 2 (2005): 183-197.
        """
        assert len(m) == self.nInd
        assert len(newKnots) == self.nInd

        # Check if any elevation or insertion is needed. If none, return self unchanged.
        impactedInd = []
        for ind in range(self.nInd):
            if m[ind] + len(newKnots[ind]) > 0:
                impactedInd.append(ind)
        if len(impactedInd) == 0:
            return self

        # Ensure the spline is clamped on the left side.
        self = self.clamp(impactedInd, [])

        # Initialize new order, nCoef, knots, coefs, and working slices and indices.
        order = [*self.order]
        nCoef = [*self.nCoef]
        knots = list(self.knots)
        coefs = self.coefs
        fullSlice = slice(None)
        sliceJI = (self.nInd + 2) * [fullSlice]
        sliceJm1Ip1 = (self.nInd + 2) * [fullSlice]
        sliceJm1I = (self.nInd + 2) * [fullSlice]
        index = (self.nInd + 2) * [0]
        for ind in range(self.nInd):
            # Step 1: Compute derivatives of coefficients at original knots, adjusted for elevated spline.

            # Step 1.1: Set zeroth derivative to original coefficients.
            assert m[ind] >= 0
            k = order[ind] # Use k for the original order to better match the paper's algorithm
            sliceJI[0] = 0
            sliceJI[ind + 2] = fullSlice
            dCoefs = np.full((k, *coefs.shape), np.nan, coefs.dtype)
            dCoefs[sliceJI] = coefs[sliceJI[1:]]

            # Step 1.2: Compute original derivative coefficients, adjusted for elevated spline.
            for j in range(1, k):
                sliceJI[0] = j
                sliceJm1Ip1[0] = j - 1
                sliceJm1I[0] = j - 1
                derivativeAdjustment = (k - j) / (k + m[ind] - j)
                for i in range(0, nCoef[ind] - j):
                    sliceJI[ind + 2] = i
                    sliceJm1Ip1[ind + 2] = i + 1
                    sliceJm1I[ind + 2] = i
                    gap = knots[ind][i + k] - knots[ind][i + j]
                    alpha = derivativeAdjustment / gap if gap > 0.0 else 0.0
                    dCoefs[sliceJI] = alpha * (dCoefs[sliceJm1Ip1] - dCoefs[sliceJm1I])
 
            # Step 2: Construct elevated order, nCoef, knots, and coefs.
            u, z = np.unique(knots[ind], return_counts=True)
            assert z[0] == k
            order[ind] += m[ind]
            uBar, zBar = np.unique(np.append(knots[ind], newKnots[ind]), return_counts=True)
            i = 0
            for iBar in range(zBar.shape[0]):
                if u[i] == uBar[iBar]:
                    zBar[iBar] = min(max(zBar[iBar], z[i] + m[ind]), order[ind])
                    i += 1
            assert zBar[0] == order[ind]
            knots[ind] = np.repeat(uBar, zBar)
            nCoef[ind] = knots[ind].shape[0] - order[ind]
            coefs = np.full((k, self.nDep, *nCoef), np.nan, coefs.dtype)
            
            # Step 3: Initialize known elevated coefficients at beta values.
            beta = -k
            betaBarL = -order[ind]
            betaBar = betaBarL
            i = 0
            for iBar in range(zBar.shape[0] - 1):
                betaBar += zBar[iBar]
                if u[i] == uBar[iBar]:
                    beta += z[i]
                    betaBarL += z[i] + m[ind]
                    # Reusing sliceJI to mean sliceJBeta, forgive me.
                    sliceJI[0] = slice(k - z[i], k)
                    sliceJI[ind + 2] = beta
                    # Reusing sliceJm1I to mean sliceJBetaBarL, forgive me.
                    sliceJm1I[0] = sliceJI[0]
                    sliceJm1I[ind + 2] = betaBarL
                    coefs[sliceJm1I] = dCoefs[sliceJI]
                    i += 1
                betaBarL = betaBar
            # Spread known <k-1>th derivatives across coefficients, since kth derivatives are zero.
            sliceJm1I[0] = k - 1
            sliceJm1Ip1[0] = k - 1
            index[0] = k - 1
            for i in range(0, nCoef[ind] - k):
                index[ind + 2] = i + 1
                if np.isnan(coefs[tuple(index)]):
                    sliceJm1I[ind + 2] = i
                    sliceJm1Ip1[ind + 2] = i + 1
                    coefs[sliceJm1Ip1] = coefs[sliceJm1I]

            # Step 4: Compute remaining derivative coefficients at elevated and new knots.
            for j in range(k - 1, 0, -1):
                sliceJI[0] = j
                sliceJm1Ip1[0] = j - 1
                sliceJm1I[0] = j - 1
                index[0] = j - 1
                for i in range(0, nCoef[ind] - j):
                    index[ind + 2] = i + 1
                    if np.isnan(coefs[tuple(index)]):
                        sliceJI[ind + 2] = i
                        sliceJm1Ip1[ind + 2] = i + 1
                        sliceJm1I[ind + 2] = i
                        coefs[sliceJm1Ip1] = coefs[sliceJm1I] + (knots[ind][i + order[ind]] - knots[ind][i + j]) * coefs[sliceJI]
            
            # Set new coefs to the elevated zeroth derivative coefficients and reset slices.
            coefs = coefs[0]
            sliceJI[ind + 2] = fullSlice
            sliceJm1Ip1[ind + 2] = fullSlice
            sliceJm1I[ind + 2] = fullSlice
            index[ind + 2] = 0
        
        return type(self)(self.nInd, self.nDep, order, nCoef, knots, coefs, self.accuracy, self.metadata)

    def find_roots(self, epsilon=1.0e-6):
        """
        Find the roots of a spline (nInd must match nDep).

        Parameters
        ----------
        epsilon : `float`
            Optional tolerance for root precision. The root will be within epsilon of the actual root. 
            The default is 1.0e-6.
        Returns
        -------
        roots : `iterable`
            An ordered iterable containing the roots of the spline. If the spline is 
            zero over an interval, that root will appear as a tuple of the interval.

        Notes
        -----
        Currently, the algorithm only works for nInd == 1. 
        Implements the algorithm from Grandine, Thomas A. "Computing zeroes of spline functions." Computer Aided Geometric Design 6, no. 2 (1989): 129-136.
        """
        assert self.nInd == self.nDep
        assert self.nInd == 1
        roots = []
        # Set initial spline, domain, and interval.
        spline = self
        domain = spline.domain()
        Interval = namedtuple('Interval', ('spline', 'slope', 'intercept'))
        intervalStack = [Interval(spline.trim(domain).reparametrize(((0.0, 1.0),)), domain[0, 1] - domain[0, 0], domain[0, 0])]

        def test_and_add_domain():
            """Macro to perform common operations when considering a domain as a new interval."""
            if domain[0, 0] <= 1.0 and domain[0, 1] >= 0.0:
                width = domain[0, 1] - domain[0, 0]
                if width >= 0.0:
                    slope = width * interval.slope
                    intercept = domain[0, 0] * interval.slope + interval.intercept
                    if slope < epsilon:
                        #if spline((domain[0, 0] + 0.5 * width,)) < epsilon:
                        roots.append(intercept + 0.5 * slope)
                    else:
                        intervalStack.append(Interval(spline.trim(domain).reparametrize(((0.0, 1.0),)), slope, intercept))

        # Process intervals until none remain
        while intervalStack:
            interval = intervalStack.pop()
            range = interval.spline.range_bounds()
            scale = np.abs(range).max(axis=1)
            if scale < epsilon:
                roots.append((interval.intercept, interval.slope + interval.intercept))
            else:
                spline = interval.spline.scale(1.0 / scale)
                mValue = spline((0.5,))
                derivativeRange = spline.differentiate().range_bounds()
                if derivativeRange[0, 0] * derivativeRange[0, 1] < 0.0:
                    # Derivative range contains zero, so consider two intervals.
                    leftIndex = 0 if mValue > 0.0 else 1
                    domain[0, 0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                    domain[0, 1] = 1.0
                    test_and_add_domain()
                    domain[0, 0] = 0.0
                    domain[0, 1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                    test_and_add_domain()
                else:
                    leftIndex = 0 if mValue > 0.0 else 1
                    domain[0, 0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                    domain[0, 1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                    test_and_add_domain()
        
        return roots

    def fold(self, foldedInd):
        """
        Fold the coefficients of a spline's indicated independent variables into the coefficients of the remaining independent variables, retaining the 
        indicated independent variables' knots and orders in a second spline with no coefficients.

        Parameters
        ----------
        foldedInd : `iterable`
            An iterable that specifies the independent variables whose coefficients should be folded.

        Returns
        -------
        foldedSpline, coefficientlessSpline : `Spline`, `Spline`
            The folded spline and the coefficientless spline that retains the indicated independent variables' knots and orders.

        See Also
        --------
        `unfold` : Unfold the coefficients of an original spline's indicated independent variables back into the spline.

        Notes
        -----
        Given a spline whose coefficients are an nDep x n0 x ... x nk array and a list of (ordered) indices which form a proper subset of {n0, ... , nk}, it should return 2 splines.
        The first one is a spline with k + 1 - length(index subset) independent variables where the knot sets of all the dimensions which have been removed have been removed.  However, 
        all of the coefficient data is still intact, so that the resulting coefficient array has shape (nDep nj0 nj1 ... njj) x nk0 x ... x nkk.  The second spline should be a spline 
        with 0 dependent variables which contains all the knot sequences that were removed from the spline.  The unfold method takes the two splines as input and reverses the process, 
        returning the original spline.

        Here's an example. Suppose spl is a spline with 3 independent variables and 3 dependent variables which has nCoef = [4, 5, 6] and knots = [knots0, knots1, knots2]. 
        Then spl.fold([0, 2]) should return a spline with 1 independent variable and 72 dependent variables.  It should have nCoef = [5], knots = [knots1], and its coefs array should have 
        shape (72, 5).  The other spline should have 0 dependent variables, 2 independent variables, and knots = [knots0, knots2].  How things get ordered in coefs probably doesn't matter 
        so long as unfold unscrambles things in the corresponding way.  The second spline is needed to hold the basis information that was dropped so that it can be undone.
        """
        assert 0 < len(foldedInd) < self.nInd
        foldedOrder = []
        foldedNCoef = []
        foldedKnots = []

        coefficientlessOrder = []
        coefficientlessNCoef = []
        coefficientlessKnots = []

        coefficientMoveFrom = []
        foldedNDep = self.nDep

        for ind in range(self.nInd):
            if ind in foldedInd:
                coefficientlessOrder.append(self.order[ind])
                coefficientlessNCoef.append(self.nCoef[ind])
                coefficientlessKnots.append(self.knots[ind])
                coefficientMoveFrom.append(ind + 1)
                foldedNDep *= self.nCoef[ind]
            else:
                foldedOrder.append(self.order[ind])
                foldedNCoef.append(self.nCoef[ind])
                foldedKnots.append(self.knots[ind])

        coefficientMoveTo = range(1, len(coefficientMoveFrom) + 1)
        foldedCoefs = np.moveaxis(self.coefs, coefficientMoveFrom, coefficientMoveTo).reshape((foldedNDep, *foldedNCoef))
        coefficientlessCoefs = np.empty((0, *coefficientlessNCoef), self.coefs.dtype)

        foldedSpline = type(self)(len(foldedOrder), foldedNDep, foldedOrder, foldedNCoef, foldedKnots, foldedCoefs, self.accuracy, self.metadata)
        coefficientlessSpline = type(self)(len(coefficientlessOrder), 0, coefficientlessOrder, coefficientlessNCoef, coefficientlessKnots, coefficientlessCoefs, self.accuracy, self.metadata)
        return foldedSpline, coefficientlessSpline
    
    def insert_knots(self, newKnots):
        """
        Insert new knots into a spline.

        Parameters
        ----------
        newKnots : `iterable` of length `nInd`
            An iterable that specifies the knots to be added to each independent variable's knots. 
            len(newKnots[ind]) == 0 if no knots are to be added for the `ind` independent variable.

        Returns
        -------
        spline : `Spline`
            A spline with the new knots inserted. If no knots were inserted, the original spline is returned.

        See Also
        --------
        `clamp` : Clamp the left and/or right side of a spline.

        Notes
        -----
        Implements Boehm's standard knot insertion algorithm.
        """
        assert len(newKnots) == self.nInd
        knots = list(self.knots)
        coefs = self.coefs
        fullSlice = slice(None)
        sliceIm1 = (self.nInd + 1) * [fullSlice]
        sliceI = (self.nInd + 1) * [fullSlice]
        for ind in range(self.nInd):
            for knot in newKnots[ind]:
                if knot < knots[ind][0] or knot > knots[ind][-self.order[ind]]:
                    raise ArgumentOutsideDomainError(knot)
                if knot == knots[ind][-self.order[ind]]:
                    position = len(knots[ind]) - self.order[ind]
                else:
                    position = np.searchsorted(knots[ind], knot, 'right')
                newCoefs = np.insert(coefs, position - 1, 0.0, axis=ind + 1)
                for i in range(position - self.order[ind] + 1, position):
                    alpha = (knot - knots[ind][i]) / (knots[ind][i + self.order[ind] - 1] - knots[ind][i])
                    sliceIm1[ind + 1] = i - 1
                    sliceI[ind + 1] = i
                    newCoefs[sliceI] = (1.0 - alpha) * coefs[sliceIm1] + alpha * coefs[sliceI]
                knots[ind] = np.insert(knots[ind], position, knot)
                coefs = newCoefs
            sliceIm1[ind + 1] = fullSlice
            sliceI[ind + 1] = fullSlice

        if self.coefs is coefs:
            return self
        else: 
            return type(self)(self.nInd, self.nDep, self.order, coefs.shape[1:], knots, coefs, self.accuracy, self.metadata)

    @staticmethod
    def load(fileName, splineType=None):
        kw = np.load(fileName)
        order = kw["order"]
        nInd = len(order)
        knots = []
        for i in range(nInd):
            knots.append(kw[f"knots{i}"])
        coefficients = kw["coefficients"]

        if splineType is None:
            splineType = Spline
        spline = splineType(nInd, coefficients.shape[0], order, coefficients.shape[1:], knots, coefficients, metadata=dict(Path=path, Name=path.splitext(path.split(fileName)[1])[0]))
        return spline

    def range_bounds(self):
        """
        Return the range of a spline as upper and lower bounds on each of the
        dependent variables
        """
        # Assumes self.nDep is the first value in self.coefs.shape
        bounds = [[coefficient.min(), coefficient.max()] for coefficient in self.coefs]
        return np.array(bounds, self.coefs.dtype)

    def reparametrize(self, newDomain):
        """
        Reparametrize a spline to match new domain bounds. The spline's number of knots and its coefficients remain unchanged.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new upper and lower bounds on each of the independent variables. 
            Same form as returned from `domain`.

        Returns
        -------
        spline : `Spline`
            Reparametrized spline.

        See Also
        --------
        `domain` : Return the domain of a spline.
        """
        assert len(newDomain) == self.nInd
        domain = self.domain()
        knotList = []
        for order, knots, d, nD in zip(self.order, self.knots, domain, newDomain):
            divisor = d[1] - d[0]
            assert abs(divisor) > 0.0
            slope = (nD[1] - nD[0]) / divisor
            assert abs(slope) > 0.0
            intercept = (nD[0] * d[1] - nD[1] * d[0]) / divisor
            knots = knots * slope + intercept
            # Force domain to match exactly at its ends and knots to be non-decreasing.
            knots[order-1] = nD[0]
            for i in range(0, order-1):
                knots[i] = min(knots[i], nD[0])
            knots[-order] = nD[1]
            for i in range(1-order, 0):
                knots[i] = max(knots[i], nD[1])
            knotList.append(knots)
        
        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, knotList, self.coefs, self.accuracy, self.metadata)   

    def save(self, fileName):
        kw = {}
        kw["order"] = order=np.array(self.order, np.int32)
        for i in range(len(self.knots)):
            kw[f"knots{i}"] = self.knots[i]
        kw["coefficients"] = self.coefs
        np.savez(fileName, **kw )

    def scale(self, multiplier):
        """
        Scale a spline by the given scalar or scale vector.

        Parameters
        ----------
        multiplier : scalar or array-like
            A scalar or an array of length `nDep` that specifies the multiplier.

        Returns
        -------
        spline : `Spline`
            The scaled spline.

        See Also
        --------
        `transform` : Transform a spline by the given matrix.
        `translate` : Translate a spline by the given translation vector.
        """
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

    def subtract(self, other, indMap = None):
        """
        Subtract two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to subtract from self. The number of dependent variables must match self.

        indMap : `iterable` or `None` (default)
            An iterable of pairs of indices. 
            Each pair (n, m) maps the mth independent variable of other to the nth independent variable of self. 
            The domains of the nth and mth independent variables must match. 
            An independent variable can map to no more than one other independent variable.
            Unmapped independent variables remain independent (the default).

        Returns
        -------
        spline : `Spline`
            The result of subtracting other from self.

        See Also
        --------
        `add` : Add two splines.
        `common_basis : Align a collection of splines to a common basis, elevating the order and adding knots as needed.

        Notes
        -----
        Uses `common_basis` to ensure mapped variables share the same order and knots. 
        """
        return self.add(other.scale(-1.0), indMap)

    def transform(self, matrix, maxSingularValue=None):
        """
        Transform a spline by the given matrix.

        Parameters
        ----------
        matrix : array-like
            An array of size `newNDep`x`nDep` that specifies the transform matrix.

        maxSingularValue : `float`
            The largest singular value of `matrix`, used to update the accuracy of the spline. 
            If no value is provided (default), the largest singular value is computed.

        Returns
        -------
        spline : `Spline`
            The transformed spline.

        See Also
        --------
        `scale` : Scale a spline by the given scalar or scale vector.
        `translate` : Translate a spline by the given translation vector.
        """
        assert matrix.ndim == 2 and matrix.shape[1] == self.nDep

        if maxSingularValue is None:
            maxSingularValue = np.linalg.svd(matrix, compute_uv=False)[0]

        return type(self)(self.nInd, matrix.shape[0], self.order, self.nCoef, self.knots, matrix @ self.coefs, maxSingularValue * self.accuracy, self.metadata)

    def translate(self, translationVector):
        """
        Translate a spline by the given translation vector.

        Parameters
        ----------
        translationVector : array-like
            An array of length `nDep` that specifies the translation vector.

        Returns
        -------
        spline : `Spline`
            The translated spline.

        See Also
        --------
        `scale` : Scale a spline by the given scalar or scale vector.
        `transform` : Transform a spline by the given matrix.
        """
        assert len(translationVector) == self.nDep

        coefs = np.array(self.coefs)
        for i in range(self.nDep):
            coefs[i] += translationVector[i]
        return type(self)(self.nInd, self.nDep, self.order, self.nCoef, self.knots, coefs, self.accuracy, self.metadata)

    def trim(self, newDomain):
        """
        Trim the domain of a spline.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new upper and lower bounds on each of the independent variables. 
            Same form as returned from `domain`.

        Returns
        -------
        spline : `Spline`
            Trimmed spline.

        See Also
        --------
        `domain` : Return the domain of a spline.
        """
        assert len(newDomain) == self.nInd

        # Step 1: Determine the knots to insert at the new domain bounds.
        newKnotsList = []
        for (order, knots, bounds) in zip(self.order, self.knots, newDomain):
            assert len(bounds) == 2
            assert knots[order - 1] <= bounds[0] < bounds[1] <= knots[-order]
            unique, counts = np.unique(knots, return_counts=True)

            multiplicity = order
            i = np.searchsorted(unique, bounds[0])
            if unique[i] == bounds[0]:
                multiplicity -= counts[i]
            newKnots = multiplicity * [bounds[0]]

            multiplicity = order
            i = np.searchsorted(unique, bounds[1])
            if unique[i] == bounds[1]:
                multiplicity -= counts[i]
            newKnots += multiplicity * [bounds[1]]

            newKnotsList.append(newKnots)
        
        # Step 2: Insert the knots.
        spline = self.insert_knots(newKnotsList)

        # Step 3: Trim the knots and coefficients.
        knotsList = []
        coefIndex = [slice(None)] # First index is for nDep
        for (order, knots, bounds) in zip(spline.order, spline.knots, newDomain):
            leftIndex = np.searchsorted(knots, bounds[0])
            rightIndex = np.searchsorted(knots, bounds[1])
            knotsList.append(knots[leftIndex:rightIndex + order])
            coefIndex.append(slice(leftIndex, rightIndex))
        coefs = spline.coefs[coefIndex]

        return type(spline)(spline.nInd, spline.nDep, spline.order, coefs.shape[1:], knotsList, coefs, spline.accuracy, spline.metadata)

    def unfold(self, foldedInd, coefficientlessSpline):
        """
        Unfold the coefficients of an original spline's indicated independent variables back into the spline, using the 
        indicated independent variables' knots and orders from a second spline with no coefficients.

        Parameters
        ----------
        foldedInd : `iterable`
            An iterable that specifies the independent variables whose coefficients should be unfolded.

        coefficientlessSpline : `Spline`
            The coefficientless spline that retains the indicated independent variables' knots and orders.

        Returns
        -------
        unfoldedSpline : `Spline`, `Spline`
            The unfolded spline.

        See Also
        --------
        `fold` : Fold the coefficients of a spline's indicated independent variables into the coefficients of the remaining independent variables.

        Notes
        -----
        Given a spline whose coefficients are an nDep x n0 x ... x nk array and a list of (ordered) indices which form a proper subset of {n0, ... , nk}, it should return 2 splines.
        The first one is a spline with k + 1 - length(index subset) independent variables where the knot sets of all the dimensions which have been removed have been removed.  However, 
        all of the coefficient data is still intact, so that the resulting coefficient array has shape (nDep nj0 nj1 ... njj) x nk0 x ... x nkk.  The second spline should be a spline 
        with 0 dependent variables which contains all the knot sequences that were removed from the spline.  The unfold method takes the two splines as input and reverses the process, 
        returning the original spline.

        Here's an example. Suppose spl is a spline with 3 independent variables and 3 dependent variables which has nCoef = [4, 5, 6] and knots = [knots0, knots1, knots2]. 
        Then spl.fold([0, 2]) should return a spline with 1 independent variable and 72 dependent variables.  It should have nCoef = [5], knots = [knots1], and its coefs array should have 
        shape (72, 5).  The other spline should have 0 dependent variables, 2 independent variables, and knots = [knots0, knots2].  How things get ordered in coefs probably doesn't matter 
        so long as unfold unscrambles things in the corresponding way.  The second spline is needed to hold the basis information that was dropped so that it can be undone.
        """
        assert len(foldedInd) == coefficientlessSpline.nInd
        unfoldedOrder = []
        unfoldedNCoef = []
        unfoldedKnots = []
        
        coefficientMoveTo = []
        unfoldedNDep = self.nDep
        indFolded = 0
        indCoefficientless = 0

        for ind in range(self.nInd + coefficientlessSpline.nInd):
            if ind in foldedInd:
                unfoldedOrder.append(coefficientlessSpline.order[indCoefficientless])
                unfoldedNCoef.append(coefficientlessSpline.nCoef[indCoefficientless])
                unfoldedKnots.append(coefficientlessSpline.knots[indCoefficientless])
                unfoldedNDep //= coefficientlessSpline.nCoef[indCoefficientless]
                coefficientMoveTo.append(ind + 1)
                indCoefficientless += 1
            else:
                unfoldedOrder.append(self.order[indFolded])
                unfoldedNCoef.append(self.nCoef[indFolded])
                unfoldedKnots.append(self.knots[indFolded])
                indFolded += 1

        coefficientMoveFrom = range(1, coefficientlessSpline.nInd + 1)
        unfoldedCoefs = np.moveaxis(self.coefs.reshape(unfoldedNDep, *coefficientlessSpline.nCoef, *self.nCoef), coefficientMoveFrom, coefficientMoveTo)

        unfoldedSpline = type(self)(len(unfoldedOrder), unfoldedNDep, unfoldedOrder, unfoldedNCoef, unfoldedKnots, unfoldedCoefs, self.accuracy, self.metadata)
        return unfoldedSpline