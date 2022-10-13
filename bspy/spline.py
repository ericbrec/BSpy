import numpy as np
from os import path
from collections import namedtuple

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
    functions (spline functions) as linear combinations of B-splines. 

    Parameters
    ----------
    nInd : `int`
        The number of independent variables of the spline

    nDep : `int`
        The number of dependent variables of the spline
    
    order : `tuple`
        A tuple of length nInd where each integer entry represents the
        polynomial order of the function in that variable

    nCoef : `tuple`
        A tuple of length nInd where each integer entry represents the
        dimension (i.e. number of B-spline coefficients) of the function
        space in that variable

    knots : `list`
        A list of the lists of the knots of the spline in each independent variable

    coefs : array-like
        A list of the B-spline coefficients of the spline.
    
    accuracy : `float`, optional
        Each spline function is presumed to be an approximation of something else. 
        The `accuracy` stores the infinity norm error of the difference between 
        the given spline function and that something else. Default is zero.

    metadata : `dict`, optional
        A dictionary of ancillary data to store with the spline. Default is {}.
    """
    
    def __init__(self, nInd, nDep, order, nCoef, knots, coefs, accuracy = 0.0, metadata = {}):
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
        self.metadata = dict(metadata)

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

        indMap : `iterable` or `None`, optional
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
        `multiply` : Multiply two splines.
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

    def blossom(self, uvw):
        """
        Compute the blossom of the spline at a given parameter values.

        Parameters
        ----------
        uvwValues : `iterable`
            An iterable of length `nInd` that specifies the degree-sized vectors of blossom parameters for each independent variable.

        Returns
        -------
        value : `numpy.array`
            The value of the spline's blossom at the given blossom parameters.

        See Also
        --------
        `evaluate` : Compute the value of the spline at a given parameter value.

        Notes
        -----
        Evaluates the blossom based on blossoming algorithm 1 found in Goldman, Ronald N. "Blossoming and knot insertion algorithms for B-spline curves." 
        Computer Aided Geometric Design 7, no. 1-4 (1990): 69-81.
        """
        def blossom_values(knot, knots, order, u):
            basis = np.zeros(order, knots.dtype)
            basis[-1] = 1.0
            for degree in range(1, order):
                b = order - degree
                for i in range(knot - degree, knot):
                    alpha = (u[degree - 1] - knots[i]) / (knots[i + degree] - knots[i])
                    basis[b - 1] += (1.0 - alpha) * basis[b]
                    basis[b] *= alpha
                    b += 1
            return basis

        # Check for evaluation point inside domain
        dom = self.domain()
        for ix in range(self.nInd):
            if uvw[ix][0] < dom[ix][0] or uvw[ix][self.order[ix]-2] > dom[ix][1]:
                raise ValueError(f"Spline evaluation outside domain: {uvw}")

        # Grab all of the appropriate coefficients
        mySection = [slice(0, self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv][0], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(slice(ix - self.order[iv], ix))
        myCoefs = self.coefs[tuple(mySection)]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = blossom_values(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs

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
        `trim` : Trim the domain of a spline.
        """
        bounds = self.nInd * [[None, None]]

        for ind in left:
            bounds[ind][0] = self.knots[ind][self.order[ind]-1]

        for ind in right:
            bounds[ind][1] = self.knots[ind][self.nCoef[ind]]

        return self.trim(bounds)

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
            multiplicities = [] # List of shared knots and their multiplicities: [[knot0, multiplicity0], [knot1, multiplicity1], ...]
            ind = map[0]
            leftKnot = splines[0].knots[ind][splines[0].order[ind]-1]
            rightKnot = splines[0].knots[ind][splines[0].nCoef[ind]]
            for (spline, ind) in zip(splines, map):
                assert spline.knots[ind][spline.order[ind]-1] == leftKnot
                assert spline.knots[ind][spline.nCoef[ind]] == rightKnot
                uniqueKnots, counts = np.unique(spline.knots[ind][spline.order[ind]-1:spline.nCoef[ind]+1], return_counts=True)
                match = 0 # Index of matching knot in the multiplicities list
                for (knot, count) in zip(uniqueKnots, counts):
                    while match < len(multiplicities) and knot > multiplicities[match][0]:
                        match += 1
                    if match == len(multiplicities) or knot < multiplicities[match][0]:
                        # No matching knot, so add the current knot to the multiplicities list.
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
                uniqueKnots, counts = np.unique(spline.knots[ind][spline.order[ind]-1:spline.nCoef[ind]+1], return_counts=True)
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
                raise ValueError(f"Spline evaluation outside domain: {uvw}")

        # Grab all of the appropriate coefficients
        mySection = [slice(0, self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(slice(ix - self.order[iv], ix))
        myCoefs = self.coefs[tuple(mySection)]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = b_spline_values(myIndices[iv], self.knots[iv], self.order[iv], with_respect_to[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs

    def differentiate(self, with_respect_to = 0):
        """
        Differentiate a spline with respect to one of its independent variables, returning the resulting spline.

        Parameters
        ----------
        with_respect_to : integer, optional
            The number of the independent variable to differentiate. Default is zero.

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
        knots = list(self.knots)
        knots[with_respect_to] = dKnots

        # Swap dependent variable axis with specified independent variable and remove first row.
        oldCoefs = self.coefs.swapaxes(0, with_respect_to + 1)
        newCoefs = np.delete(oldCoefs, 0, axis=0) 
        for i in range(nCoef[with_respect_to]):
            alpha =  degree / (dKnots[i+degree] - dKnots[i])
            newCoefs[i] = alpha * (newCoefs[i] - oldCoefs[i])
        
        return type(self)(self.nInd, self.nDep, order, nCoef, knots, newCoefs.swapaxes(0, with_respect_to + 1), self.accuracy, self.metadata)

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
            assert m[ind] >= 0
            if m[ind] + len(newKnots[ind]) > 0:
                impactedInd.append(ind)
        if len(impactedInd) == 0:
            return self

        # Ensure the spline is clamped on the left side.
        self = self.clamp(impactedInd, [])

        # Initialize new order, nCoef, knots, coefs, and indices.
        order = [*self.order]
        nCoef = [*self.nCoef]
        knots = list(self.knots)
        coefs = self.coefs
        index = (self.nInd + 2) * [0]
        for ind in range(self.nInd):
            if m[ind] + len(newKnots[ind]) == 0:
                continue

            # Step 1: Compute derivatives of coefficients at original knots, adjusted for elevated spline.

            # Step 1.1: Set zeroth derivative to original coefficients.
            k = order[ind] # Use k for the original order to better match the paper's algorithm
            coefs = coefs.swapaxes(0, ind + 1) # Swap the dependent variable with the independent variable (swap back later)
            dCoefs = np.full((k, *coefs.shape), np.nan, coefs.dtype)
            dCoefs[0] = coefs

            # Step 1.2: Compute original derivative coefficients, adjusted for elevated spline.
            for j in range(1, k):
                derivativeAdjustment = (k - j) / (k + m[ind] - j)
                for i in range(0, nCoef[ind] - j):
                    gap = knots[ind][i + k] - knots[ind][i + j]
                    alpha = derivativeAdjustment / gap if gap > 0.0 else 0.0
                    dCoefs[j, i] = alpha * (dCoefs[j - 1, i + 1] - dCoefs[j - 1, i])
 
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
            coefs = np.full((k, self.nDep, *nCoef), np.nan, coefs.dtype).swapaxes(1, ind + 2)
            
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
                    coefSlice = slice(k - z[i], k)
                    coefs[coefSlice, betaBarL] = dCoefs[coefSlice, beta]
                    i += 1
                betaBarL = betaBar
            # Spread known <k-1>th derivatives across coefficients, since kth derivatives are zero.
            index[0] = k - 1
            for i in range(0, nCoef[ind] - k):
                index[1] = i + 1
                if np.isnan(coefs[tuple(index)]):
                    coefs[k - 1, i + 1] = coefs[k - 1, i]

            # Step 4: Compute remaining derivative coefficients at elevated and new knots.
            for j in range(k - 1, 0, -1):
                index[0] = j - 1
                for i in range(0, nCoef[ind] - j):
                    index[1] = i + 1
                    if np.isnan(coefs[tuple(index)]):
                        coefs[j - 1, i + 1] = coefs[j - 1, i] + (knots[ind][i + order[ind]] - knots[ind][i + j]) * coefs[j, i]
            
            # Set new coefs to the elevated zeroth derivative coefficients with variables swapped back.
            coefs = coefs[0].swapaxes(0, ind + 1)
        
        return type(self)(self.nInd, self.nDep, order, nCoef, knots, coefs, self.accuracy, self.metadata)

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
                raise ValueError(f"Spline evaluation outside domain: {uvw}")

        # Grab all of the appropriate coefficients
        mySection = [slice(0, self.nDep)]
        myIndices = []
        for iv in range(self.nInd):
            ix = np.searchsorted(self.knots[iv], uvw[iv], 'right')
            ix = min(ix, self.nCoef[iv])
            myIndices.append(ix)
            mySection.append(slice(ix - self.order[iv], ix))
        myCoefs = self.coefs[tuple(mySection)]
        for iv in range(self.nInd - 1, -1, -1):
            bValues = b_spline_values(myIndices[iv], self.knots[iv], self.order[iv], uvw[iv])
            myCoefs = myCoefs @ bValues
        return myCoefs

    def extrapolate(self, newDomain, continuityOrder):
        """
        Extrapolate a spline out to an extended domain maintaining a given order of continuity.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new upper and lower bounds on each of the independent variables (same form as 
            returned from `domain`). If a bound is None or nan then the original bound (and knots) are left unchanged.

        continuityOrder : `int`
            The order of continuity of the extrapolation (the number of derivatives that match at the endpoints). 
            A continuity order of zero means the extrapolation just matches the spline value at the endpoints. 
            The continuity order is automatically limited to one less than the degree of the spline.

        Returns
        -------
        spline : `Spline`
            Extrapolated spline. If all the knots are unchanged, the original spline is returned.

        See Also
        --------
        `domain` : Return the domain of a spline.
        `trim` : Trim the domain of a spline.
        """
        assert len(newDomain) == self.nInd
        assert continuityOrder >= 0

        # Check if any extrapolation is needed. If none, return self unchanged.
        # Also compute the new nCoef, new knots, coefficient slices, and left/right clamp variables.
        nCoef = [*self.nCoef]
        knots = list(self.knots)
        coefSlices = [0, slice(None)]
        leftInd = []
        rightInd = []
        for ind, bounds in zip(range(self.nInd), newDomain):
            order = self.order[ind]
            degree = order - 1
            assert len(bounds) == 2
            # Add new knots to end first, so indexing isn't messed up at the beginning.
            if bounds[1] is not None and not np.isnan(bounds[1]):
                oldBound = self.knots[ind][self.nCoef[ind]]
                assert bounds[1] > oldBound
                knots[ind] = np.append(knots[ind], degree * [bounds[1]])
                nCoef[ind] += degree
                for i in range(self.nCoef[ind] + 1, self.nCoef[ind] + order):
                    knots[ind][i] = oldBound # Reflect upcoming clamp
                rightInd.append(ind)
            # Next, add knots to the beginning and set coefficient slice.
            if bounds[0] is not None and not np.isnan(bounds[0]):
                oldBound = self.knots[ind][degree]
                assert bounds[0] < oldBound
                knots[ind] = np.insert(knots[ind], 0, degree * [bounds[0]])
                nCoef[ind] += degree
                for i in range(degree, 2 * degree):
                    knots[ind][i] = oldBound # Reflect upcoming clamp
                leftInd.append(ind)
                coefSlice = slice(degree, degree + self.nCoef[ind])
            else:
                coefSlice = slice(0, self.nCoef[ind])
            coefSlices.append(coefSlice)

        if len(leftInd) + len(rightInd) == 0:
            return self

        # Ensure the spline is clamped on the sides being extrapolated.
        self = self.clamp(leftInd, rightInd)

        # Initialize dCoefs working array and working spline.
        dCoefs = np.empty((continuityOrder + 1, self.nDep, *nCoef), self.coefs.dtype)
        dCoefs[tuple(coefSlices)] = self.coefs

        for ind, bounds in zip(range(self.nInd), newDomain):
            order = self.order[ind]
            coefSlice = coefSlices[ind + 2]
            continuity = min(continuityOrder, order - 2)
            dCoefs = dCoefs.swapaxes(1, ind + 2) # Swap dependent and independent variables (swap back later).
            
            if bounds[0] is not None and not np.isnan(bounds[0]):
                # Compute derivatives of coefficients at interior knots.
                for j in range(1, continuity + 1):
                    for i in range(coefSlice.start, coefSlice.start + continuity - j + 1):
                        gap = knots[ind][i + order] - knots[ind][i + j]
                        alpha = (order - j) / gap if gap > 0.0 else 0.0
                        dCoefs[j, i] = alpha * (dCoefs[j - 1, i + 1] - dCoefs[j - 1, i])

                # Extrapolate spline values out to new bounds by Taylor series for each derivative.
                gap = knots[ind][0] - knots[ind][coefSlice.start]
                for j in range(continuity, -1, -1):
                    dCoefs[j, 0] = dCoefs[continuity, coefSlice.start]
                    for i in range(continuity - 1, j - 1, -1):
                        dCoefs[j, 0] = dCoefs[i, coefSlice.start] + (gap / (i + 1 - j)) * dCoefs[j, 0]

                # Convert new bound to full multiplicity and old bound to interior knot.
                knots[ind][degree] = knots[ind][0]

                # Backfill coefficients by integrating out from extrapolated spline values.
                for j in range(continuity, 0, -1):
                    for i in range(0, degree - j):
                        dCoefs[j - 1, i + 1] = dCoefs[j - 1, i] + ((knots[ind][i + order] - knots[ind][i + j]) / (order - j)) * dCoefs[j, i if j < continuity else 0]

            if bounds[1] is not None and not np.isnan(bounds[1]):
                # Compute derivatives of coefficients at interior knots.
                for j in range(1, continuity + 1):
                    for i in range(coefSlice.stop + j - continuity - 1, coefSlice.stop):
                        gap = knots[ind][i + order - j] - knots[ind][i]
                        alpha = (order - j) / gap if gap > 0.0 else 0.0
                        dCoefs[j, i] = alpha * (dCoefs[j - 1, i] - dCoefs[j - 1, i - 1])

                # Extrapolate spline values out to new bounds by Taylor series for each derivative.
                gap = knots[ind][coefSlice.stop + order] - knots[ind][coefSlice.stop]
                lastOldCoef = coefSlice.stop - 1
                lastNewCoef = nCoef[ind] - 1
                for j in range(continuity, -1, -1):
                    dCoefs[j, lastNewCoef] = dCoefs[continuity, lastOldCoef]
                    for i in range(continuity - 1, j - 1, -1):
                        dCoefs[j, lastNewCoef] = dCoefs[i, lastOldCoef] + (gap / (i + 1 - j)) * dCoefs[j, lastNewCoef]

                # Convert new bound to full multiplicity and old bound to interior knot.
                knots[ind][coefSlice.stop + degree] = knots[ind][coefSlice.stop + order]

                # Backfill coefficients by integrating out from extrapolated spline values.
                for j in range(continuity, 0, -1):
                    for i in range(lastNewCoef, lastOldCoef + j, -1):
                        dCoefs[j - 1, i - 1] = dCoefs[j - 1, i] - ((knots[ind][i + order - j] - knots[ind][i]) / (order - j)) * dCoefs[j, i if j < continuity else lastNewCoef]

            # Swap dependent and independent variables back.
            dCoefs = dCoefs.swapaxes(1, ind + 2) 

        return type(self)(self.nInd, self.nDep, self.order, nCoef, knots, dCoefs[0], self.accuracy, self.metadata)

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
        for ind in range(self.nInd):
            # We can't reference self.nCoef[ind] in this loop because we are expanding the knots and coefs arrays.
            for knot in newKnots[ind]:
                if knot < knots[ind][self.order[ind]-1] or knot > knots[ind][-self.order[ind]]:
                    raise ValueError(f"Knot insertion outside domain: {knot}")
                if knot == knots[ind][-self.order[ind]]:
                    position = len(knots[ind]) - self.order[ind]
                else:
                    position = np.searchsorted(knots[ind], knot, 'right')
                coefs = coefs.swapaxes(0, ind + 1) # Swap dependend and independent variable (swap back later)
                newCoefs = np.insert(coefs, position - 1, 0.0, axis=0)
                for i in range(position - self.order[ind] + 1, position):
                    alpha = (knot - knots[ind][i]) / (knots[ind][i + self.order[ind] - 1] - knots[ind][i])
                    newCoefs[i] = (1.0 - alpha) * coefs[i - 1] + alpha * coefs[i]
                knots[ind] = np.insert(knots[ind], position, knot)
                coefs = newCoefs.swapaxes(0, ind + 1)

        if self.coefs is coefs:
            return self
        else: 
            return type(self)(self.nInd, self.nDep, self.order, coefs.shape[1:], knots, coefs, self.accuracy, self.metadata)

    @staticmethod
    def least_squares(dataPoints):
        """
        Fit a curve to a string of data points using the method of least squares.

        Parameters
        ----------
        dataPoints : `iterable` containing the data points to fit.
            Each of the data points is of length nDep.

        Returns
        -------
        spline : `Spline`
            A spline curve which approximates the data points.
        """
        rhsPoints = []
        uValues = []
        for dp in list(dataPoints):
            uValues.append(dp[0])
            rhsPoints.append(dp[1:])

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

    def multiply(self, other, indMap = None):
        """
        Multiply two splines.

        Parameters
        ----------
        other : `Spline`
            The spline to multiply by self. The number of dependent variables must match self.

        indMap : `iterable` or `None`, optional
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
        `add` : Add two splines.
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

    def range_bounds(self):
        """
        Return the range of a spline as upper and lower bounds on each of the
        dependent variables
        """
        # Assumes self.nDep is the first value in self.coefs.shape
        bounds = [[coefficient.min(), coefficient.max()] for coefficient in self.coefs]
        return np.array(bounds, self.coefs.dtype)

    def remove_knots(self, oldKnots=((),), maxRemovalsPerKnot=0, tolerance=None):
        """
        Remove interior knots from a spline.

        Parameters
        ----------
        oldKnots : `iterable` of length `nInd`, optional
            An iterable that specifies the knots that can be removed from each independent variable's interior knots. 
            len(newKnots[ind]) == 0 if all interior knots can be removed for the `ind` independent variable (the default). 
            Knots that don't appear in the independent variable's interior knots are ignored.
        
        maxRemovalsPerKnot : `int`, optional
            A non-zero count of the largest number of times a knot can be removed. For example, one means that 
            only one instance of each knot can be removed. (Zero means each knot can be removed completely, 
            which is the default.)
        
        tolerance : `float` or `None`, optional
            The maximum residual error permitted after removing a knot. Knots will not be removed if the 
            resulting residual error is above this threshold. Default is `None`, meaning all specified knots 
            will be removed up to `maxRemovalsPerKnot`.

        Returns
        -------
        spline : `Spline`
            A spline with the knots removed.
        
        totalRemoved : `int`
            The total number of knots removed.
        
        residualError : `float`
            The residual error relative to the old spline. (The returned spline's accuracy is also adjusted accordinly.)

        See Also
        --------
        `insert_knots` : Insert new knots into a spline.
        `trim` : Trim the domain of a spline.

        Notes
        -----
        Implements a variation of the algorithms from Tiller, Wayne. "Knot-removal algorithms for NURBS curves and surfaces." Computer-Aided Design 24, no. 8 (1992): 445-453.
        """
        assert len(oldKnots) == self.nInd
        nCoef = [*self.nCoef]
        knotList = list(self.knots)
        coefs = self.coefs.copy()
        temp = np.empty_like(coefs)
        totalRemoved = 0
        maxResidualError = 0.0
        for ind in range(self.nInd):
            order = self.order[ind]
            highSpan = nCoef[ind] - 1
            if highSpan < order:
                continue # no interior knots

            removeAll = len(oldKnots[ind]) == 0
            degree = order - 1
            knots = knotList[ind].copy()
            highU = knots[highSpan]
            gap = 0 # size of the gap
            u = knots[order]
            knotIndex = order
            while u == knots[knotIndex + 1]:
                knotIndex += 1
            multiplicity = knotIndex - degree
            firstCoefOut = (2 * knotIndex - degree - multiplicity) // 2 # first control point out
            last = knotIndex - multiplicity
            first = multiplicity
            beforeGap = knotIndex # control-point index before gap
            afterGap = beforeGap + 1 # control-point index after gap
            # Move the independent variable to the front of coefs and temp. We'll move it back at later.
            coefs = coefs.swapaxes(0, ind + 1)
            temp = temp.swapaxes(0, ind + 1)

            # Loop thru knots, stop after we process highU.
            while True: 
                # Compute how many times to remove knot.
                removed = 0
                if removeAll or u in oldKnots[ind]:
                    if maxRemovalsPerKnot > 0:
                        maxRemovals = min(multiplicity, maxRemovalsPerKnot)
                    else:
                        maxRemovals = multiplicity
                else:
                    maxRemovals = 0

                while removed < maxRemovals:
                    offset = first - 1  # diff in index of temp and coefs
                    temp[0] = coefs[offset]
                    temp[last + 1 - offset] = coefs[last + 1]
                    i = first
                    j = last
                    ii = first - offset
                    jj = last - offset

                    # Compute new coefficients for 1 removal step.
                    while j - i > removed:
                        alphaI = (u - knots[i]) / (knots[i + order + gap + removed] - knots[i])
                        alphaJ = (u - knots[j - removed]) / (knots[j + order + gap] - knots[j - removed])
                        temp[ii] = (coefs[i] - (1.0 - alphaI) * temp[ii - 1]) / alphaI
                        temp[jj] = (coefs[j] - alphaJ * temp[jj + 1])/ (1.0 - alphaJ)
                        i += 1
                        ii += 1
                        j -= 1
                        j -= 1

                    # Compute residual error.
                    if j - i < removed:
                        residualError = np.linalg.norm(temp[ii - 1] - temp[jj + 1], axis=ind).max()
                    else:
                        alphaI = (u - knots[i]) / (knots[i + order + gap + removed] - knots[i])
                        residualError = np.linalg.norm(alphaI * temp[ii + removed + 1] + (1.0 - alphaI) * temp[ii - 1] - coefs[i], axis=ind).max()

                    # Check if knot is removable.
                    if tolerance is None or residualError <= tolerance:
                        # Successful removal. Save new coefficients.
                        maxResidualError = max(residualError, maxResidualError)
                        i = first
                        j = last
                        while j - i > removed:
                            coefs[i] = temp[i - offset]
                            coefs[j] = temp[j - offset]
                            i += 1
                            j -= 1
                    else:
                        break # Get out of removed < maxRemovals while-loop
                    
                    first -= 1
                    last += 1
                    removed += 1
                    # End of removed < maxRemovals while-loop.
                
                if removed > 0:
                    # Knots removed. Shift coefficients down.
                    j = firstCoefOut
                    i = j
                    # Pj thru Pi will be overwritten.
                    for k in range(1, removed):
                        if k % 2 == 1:
                            i += 1
                        else:
                            j -= 1
                    for k in range(i + 1, beforeGap):
                        coefs[j] = coefs[k] # shift
                        j += 1
                else:
                    j = beforeGap + 1

                if u >= highU:
                    gap += removed # No more knots, get out of endless while-loop
                    break
                else:
                    # Go to next knot, shift knots and coefficients down, and reset gaps.
                    k1 = knotIndex - removed + 1
                    k = knotIndex + gap + 1
                    i = k1
                    u = knots[k]
                    while u == knots[k]:
                        knots[i] = knots[k]
                        i += 1
                        k += 1
                    multiplicity = i - k1
                    knotIndex = i - 1
                    gap += removed
                    for k in range(0, multiplicity):
                        coefs[j] = coefs[afterGap]
                        j += 1
                        afterGap += 1
                    beforeGap = j - 1
                    firstCoefOut = (2 * knotIndex - degree - multiplicity) // 2
                    last = knotIndex - multiplicity
                    first = knotIndex - degree
                # End of endless while-loop

            # Shift remaining knots.
            i = highSpan + 1
            k = i - gap
            for j in range(1, order + 1):
                knots[k] = knots[i] 
                k += 1
                i += 1
            
            # Update totalRemoved, nCoef, knots, and coefs.
            totalRemoved += gap
            nCoef[ind] -= gap
            knotList[ind] = knots[:order + nCoef[ind]]
            coefs = coefs[:nCoef[ind]]
            coefs = coefs.swapaxes(0, ind + 1)
            temp = temp.swapaxes(0, ind + 1)
            # End of ind loop
        
        spline = type(self)(self.nInd, self.nDep, self.order, nCoef, knotList, coefs, self.accuracy + maxResidualError, self.metadata)   
        return spline, totalRemoved, maxResidualError

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

        indMap : `iterable` or `None`, optional
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
        `multiply` : Multiply two splines.
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

        maxSingularValue : `float`, optional
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
            nInd x 2 array of the new upper and lower bounds on each of the independent variables (same form as 
            returned from `domain`). If a bound is None or nan then the original bound (and knots) are left unchanged.

        Returns
        -------
        spline : `Spline`
            Trimmed spline. If all the knots are unchanged, the original spline is returned.

        See Also
        --------
        `domain` : Return the domain of a spline.
        `extrapolate` : Extrapolate a spline out to an extended domain maintaining a given order of continuity.
        """
        assert len(newDomain) == self.nInd

        # Step 1: Determine the knots to insert at the new domain bounds.
        newKnotsList = []
        for (order, knots, bounds) in zip(self.order, self.knots, newDomain):
            assert len(bounds) == 2
            unique, counts = np.unique(knots, return_counts=True)
            leftBound = False # Do we have a left bound?
            newKnots = []

            if bounds[0] is not None and not np.isnan(bounds[0]):
                assert knots[order - 1] <= bounds[0] <= knots[-order]
                leftBound = True
                multiplicity = order
                i = np.searchsorted(unique, bounds[0])
                if unique[i] == bounds[0]:
                    multiplicity -= counts[i]
                newKnots += multiplicity * [bounds[0]]

            if bounds[1] is not None and not np.isnan(bounds[1]):
                assert knots[order - 1] <= bounds[1] <= knots[-order]
                if leftBound:
                    assert bounds[0] < bounds[1]
                multiplicity = order
                i = np.searchsorted(unique, bounds[1])
                if unique[i] == bounds[1]:
                    multiplicity -= counts[i]
                newKnots += multiplicity * [bounds[1]]

            newKnotsList.append(newKnots)
        
        # Step 2: Insert the knots.
        spline = self.insert_knots(newKnotsList)
        if spline is self:
            return spline

        # Step 3: Trim the knots and coefficients.
        knotsList = []
        coefIndex = [slice(None)] # First index is for nDep
        for (order, knots, bounds) in zip(spline.order, spline.knots, newDomain):
            leftIndex = order - 1 if bounds[0] is None or np.isnan(bounds[0]) else np.searchsorted(knots, bounds[0])
            rightIndex = len(knots) - order if bounds[1] is None or np.isnan(bounds[1]) else np.searchsorted(knots, bounds[1])
            knotsList.append(knots[leftIndex:rightIndex + order])
            coefIndex.append(slice(leftIndex, rightIndex))
        coefs = spline.coefs[tuple(coefIndex)]

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

    def zeros(self, epsilon=None):
        """
        Find the roots of a spline (nInd must match nDep).

        Parameters
        ----------
        epsilon : `float`, optional
            Tolerance for root precision. The root will be within epsilon of the actual root. 
            The default is the max of spline accuracy and machine epsilon.

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
        machineEpsilon = np.finfo(self.knots[0].dtype).eps
        if epsilon is None:
            epsilon = max(self.accuracy, machineEpsilon)
        roots = []
        # Set initial spline, domain, and interval.
        spline = self
        (domain,) = spline.domain()
        Interval = namedtuple('Interval', ('spline', 'slope', 'intercept', 'atMachineEpsilon'))
        intervalStack = [Interval(spline.trim((domain,)).reparametrize(((0.0, 1.0),)), domain[1] - domain[0], domain[0], False)]

        def test_and_add_domain():
            """Macro to perform common operations when considering a domain as a new interval."""
            if domain[0] <= 1.0 and domain[1] >= 0.0:
                width = domain[1] - domain[0]
                if width >= 0.0:
                    slope = width * interval.slope
                    intercept = domain[0] * interval.slope + interval.intercept
                    # Iteration is complete if the interval actual width (slope) is either
                    # one iteration past being less than sqrt(machineEpsilon) or simply less than epsilon.
                    if interval.atMachineEpsilon or slope < epsilon:
                        root = intercept + 0.5 * slope
                        # Double-check that we're at an actual zero (avoids boundary case).
                        if self((root,)) < epsilon:
                            # Check for duplicate root. We test for a distance between roots of 2*epsilon to account for a left vs. right sided limit.
                            if roots and abs(root - roots[-1]) < 2.0 * epsilon:
                                # For a duplicate root, return the average value.
                                roots[-1] = 0.5 * (roots[-1] + root)
                            else:
                                roots.append(root)
                    else:
                        intervalStack.append(Interval(spline.trim((domain,)).reparametrize(((0.0, 1.0),)), slope, intercept, slope * slope < machineEpsilon))

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
                if derivativeRange[0, 0] * derivativeRange[0, 1] <= 0.0:
                    # Derivative range contains zero, so consider two intervals.
                    leftIndex = 0 if mValue > 0.0 else 1
                    domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                    domain[1] = 1.0
                    test_and_add_domain()
                    domain[0] = 0.0
                    domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                    test_and_add_domain()
                else:
                    leftIndex = 0 if mValue > 0.0 else 1
                    domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                    domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                    test_and_add_domain()
        
        return roots