import numpy as np
from bspy.manifold import Manifold

def clamp(self, left, right):
    bounds = [[None, None] for i in range(self.nInd)]

    for ind in left:
        bounds[ind][0] = self.knots[ind][self.order[ind]-1]

    for ind in right:
        bounds[ind][1] = self.knots[ind][self.nCoef[ind]]

    return self.trim(bounds)

def common_basis(splines, indMap):
    # Fill out the default indMap.
    if indMap is None:
        indMap = [len(splines) * [iInd] for iInd in range(splines[0].nInd)]
    
    # Ensure all splines are clamped at both ends.
    splines = [spline.clamp(tuple(range(spline.nInd)), tuple(range(spline.nInd))) for spline in splines]

    # Step 1: Compute the order for each aligned independent variable.
    orders = []
    for map in indMap:
        if not(len(map) == len(splines)): raise ValueError("Invalid map")
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
            if not(spline.knots[ind][spline.order[ind]-1] == leftKnot): raise ValueError("Spline domains don't match")
            if not(spline.knots[ind][spline.nCoef[ind]] == rightKnot): raise ValueError("Spline domains don't match")
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
    for i, spline in enumerate(splines):
        m = spline.nInd * [0]
        newKnots = [[] for ix in spline.order]
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

def elevate_and_insert_knots(self, m, newKnots):
    if not(len(m) == self.nInd): raise ValueError("Invalid m")
    if not(len(newKnots) == self.nInd): raise ValueError("Invalid newKnots")

    # Check if any elevation or insertion is needed. If none, return self unchanged.
    impactedInd = []
    for ind in range(self.nInd):
        if not(m[ind] >= 0): raise ValueError("Invalid m")
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
    
    return type(self)(self.nInd, self.nDep, order, nCoef, knots, coefs, self.metadata)

def extrapolate(self, newDomain, continuityOrder):
    if not(len(newDomain) == self.nInd): raise ValueError("Invalid newDomain")
    if not(continuityOrder >= 0): raise ValueError("Invalid continuityOrder")

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
        if not(len(bounds) == 2): raise ValueError("Invalid bounds")
        # Add new knots to end first, so indexing isn't messed up at the beginning.
        if bounds[1] is not None and not np.isnan(bounds[1]):
            oldBound = self.knots[ind][self.nCoef[ind]]
            if not(bounds[1] > oldBound): raise ValueError("Invalid bounds")
            knots[ind] = np.append(knots[ind], degree * [bounds[1]])
            nCoef[ind] += degree
            for i in range(self.nCoef[ind] + 1, self.nCoef[ind] + order):
                knots[ind][i] = oldBound # Reflect upcoming clamp
            rightInd.append(ind)
        # Next, add knots to the beginning and set coefficient slice.
        if bounds[0] is not None and not np.isnan(bounds[0]):
            oldBound = self.knots[ind][degree]
            if not(bounds[0] < oldBound): raise ValueError("Invalid bounds")
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

    return type(self)(self.nInd, self.nDep, self.order, nCoef, knots, dCoefs[0], self.metadata).remove_knots()

def fold(self, foldedInd):
    if not(0 <= len(foldedInd) <= self.nInd): raise ValueError("Invalid foldedInd")
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

    foldedSpline = type(self)(len(foldedOrder), foldedNDep, foldedOrder, foldedNCoef, foldedKnots, foldedCoefs, self.metadata)
    coefficientlessSpline = type(self)(len(coefficientlessOrder), 0, coefficientlessOrder, coefficientlessNCoef, coefficientlessKnots, coefficientlessCoefs, self.metadata)
    return foldedSpline, coefficientlessSpline

def insert_knots(self, newKnots):
    if not(len(newKnots) == self.nInd): raise ValueError("Invalid newKnots")
    knotsList = list(self.knots)
    coefs = self.coefs
    for ind, (order, knots) in enumerate(zip(self.order, self.knots)):
        # We can't reference self.nCoef[ind] in this loop because we are expanding the knots and coefs arrays.
        for knot in newKnots[ind]:
            if knot < knots[order-1] or knot > knots[-order]:
                raise ValueError(f"Knot insertion outside domain: {knot}")
            if knot == knots[-order]:
                position = len(knots) - order
            else:
                position = np.searchsorted(knots, knot, 'right')
            coefs = coefs.swapaxes(0, ind + 1) # Swap dependent and independent variable (swap back later)
            newCoefs = np.insert(coefs, position - 1, 0.0, axis=0)
            for i in range(position - order + 1, position):
                alpha = (knot - knots[i]) / (knots[i + order - 1] - knots[i])
                newCoefs[i] = (1.0 - alpha) * coefs[i - 1] + alpha * coefs[i]
            knotsList[ind] = knots = np.insert(knots, position, knot)
            coefs = newCoefs.swapaxes(0, ind + 1)

    if self.coefs is coefs:
        return self
    else: 
        return type(self)(self.nInd, self.nDep, self.order, coefs.shape[1:], knotsList, coefs, self.metadata)

def join(splineList):
    # Make sure all the splines in the list are curves
    if len(splineList) < 1:  raise ValueError("Not enough splines in list")
    for spl in splineList:
        if spl.nInd != 1:  raise NotImplementedError("Method needs to be extended to multivariate splines")
    
    # Make sure all splines have the same number of dependent variables
    numDep = splineList[0].nDep
    for spl in splineList[1:]:
        if spl.nDep != numDep:  raise ValueError("Inconsistent number of dependent variables")
    
    # Go through all of the splines in turn
    workingSpline = splineList[0]
    for spl in splineList[1:]:
        workingDomain = workingSpline.domain()[0]
        start1 = workingSpline(workingDomain[0])
        end1 = workingSpline(workingDomain[1])
        splDomain = spl.domain()[0]
        start2 = spl(splDomain[0])
        end2 = spl(splDomain[1])
        gaps = [np.linalg.norm(vecDiff) for vecDiff in [start1 - start2, start1 - end2, end1 - start2, end1 - end2]]
        minDist = min(*gaps)
        if minDist == gaps[0] or minDist == gaps[1]:
            workingSpline = workingSpline.reverse()
        if minDist == gaps[1] or minDist == gaps[3]:
            spl = spl.reverse()
        maxOrder = max(workingSpline.order[0], spl.order[0])
        workingSpline = workingSpline.elevate([maxOrder - workingSpline.order[0]])
        spl = spl.elevate([maxOrder - spl.order[0]])
        speed1 = np.linalg.norm(workingSpline.derivative([1], workingDomain[1]))
        speed2 = np.linalg.norm(spl.derivative([1], splDomain[0]))
        spl = spl.reparametrize([[workingDomain[1], workingDomain[1] + speed2 * (splDomain[1] - splDomain[0]) / speed1]])
        newKnots = [list(workingSpline.knots[0]) + list(spl.knots[0][maxOrder:])]
        newCoefs = [list(workingCoefs) + list(splCoefs) for workingCoefs, splCoefs in zip(workingSpline.coefs, spl.coefs)]
        workingSpline = type(workingSpline)(1, numDep, workingSpline.order, [workingSpline.nCoef[0] + spl.nCoef[0]],
                                            newKnots, newCoefs, workingSpline.metadata)
    return workingSpline.reparametrize([[0.0, 1.0]]).remove_knots()

def remove_knot(self, iKnot, nLeft = 0, nRight = 0):
    if self.nInd != 1:  raise ValueError("Must have one independent variable")
    myOrder = self.order[0]
    if iKnot < myOrder or iKnot >= self.nCoef[0]:  raise ValueError("Must specify interior knots for removal")
    diag0 = []
    diag1 = [1.0]
    rhs = [np.ndarray.copy(self.coefs[:, iKnot - myOrder])]
    myKnots = self.knots[0]
    thisKnot = myKnots[iKnot]

    # Form the bi-diagonal system
    for ix in range(1, myOrder):
        alpha = (myKnots[iKnot + ix] - thisKnot) / (myKnots[iKnot + ix] - myKnots[iKnot + ix - myOrder])
        diag0.append(alpha)
        diag1.append(1.0 - alpha)
        rhs.append(np.ndarray.copy(self.coefs[:, iKnot - myOrder + ix]))
    diag0.append(1.0)
    diag1.append(0.0)
    rhs.append(np.ndarray.copy(self.coefs[:, iKnot]))
    rhs = np.array(rhs)

    # Take care of the extra known conditions on the left
    extraLeft = max(0, nLeft - iKnot + myOrder)
    for ix in range(extraLeft):
        rhs[ix] /= diag1[ix]
        rhs[ix + 1] -= diag0[ix] * rhs[ix]
    
    # Take care of the extra known conditions on the right
    extraRight = max(0, nRight - self.nCoef[0] + iKnot + 1)
    for ix in range(extraRight):
        rhs[-1 - ix] /= diag0[-1 - ix]
        rhs[-2 - ix] -= diag1[-2 - ix] * rhs[-1 - ix]

    # Use Givens rotations to factor the matrix and track right hand side
    for ix in range(extraLeft, myOrder - extraRight):
        cos = diag1[ix]
        sin = diag0[ix]
        denom = np.sqrt(cos ** 2 + sin ** 2)
        cos /= denom
        sin /= denom
        diag1[ix] = denom
        diag0[ix] = sin * diag1[ix + 1]
        diag1[ix + 1] *= cos
        tempRow = cos * rhs[ix] + sin * rhs[ix + 1]
        rhs[ix + 1] = cos * rhs[ix + 1] - sin * rhs[ix]
        rhs[ix] = tempRow
    
    # Perform back substitution
    for ix in range(1 + extraRight, myOrder - extraLeft):
        rhs[-1 - ix] /= diag1[-1 - ix]
        rhs[-2 - ix] -= diag0[-1 - ix] * rhs[-1 - ix]
    rhs[-1 - myOrder + extraLeft] /= diag1[-1 - myOrder + extraLeft]
    
    # Save residual and adjust solution
    residual = abs(rhs[-1 - extraRight])
    for ix in range(extraRight):
        rhs[-1 - extraRight+ ix] = rhs[-extraRight + ix]
    
    # Create new spline
    newNCoef = [self.nCoef[0] - 1]
    newKnots = [np.delete(self.knots[0], iKnot)]
    newCoefs = np.delete(self.coefs, iKnot - self.order[0] + 1, 1)
    newCoefs[: , iKnot - self.order[0] : iKnot] = rhs[: -1].T
    withoutKnot = type(self)(self.nInd, self.nDep, self.order, newNCoef, newKnots, newCoefs)
    return withoutKnot, residual

def remove_knots(self, tolerance, nLeft = 0, nRight = 0):
    scaleDep = [max(np.abs(bound[0]), np.abs(bound[1])) for bound in self.range_bounds()]
    scaleDep = [1.0 if factor == 0.0 else factor for factor in scaleDep]
    rScaleDep = np.array([1.0 / factor for factor in scaleDep])
    # Remove knots one at a time until done
    currentSpline = self.scale(rScaleDep)
    truthSpline = currentSpline
    indIndex = list(range(currentSpline.nInd))
    for id in indIndex:
        foldedIndices = list(filter(lambda x: x != id, indIndex))
        currentFold, foldedBasis = currentSpline.fold(foldedIndices)
        while True:
            bestError = np.finfo(scaleDep[0].dtype).max
            bestSpline = currentFold
            ix = currentFold.order[0]
            while ix < currentFold.nCoef[0]:
                newSpline, residual = currentFold.remove_knot(ix, nLeft, nRight)
                error = np.max(residual)
                if error < 0.001 * tolerance:
                    currentFold = newSpline
                    continue
                if error < bestError:
                    bestError = error
                    bestSpline = newSpline
                ix += 1
            if currentFold.nCoef[0] < bestSpline.nCoef[0]:
                continue
            if bestError > tolerance:
                break
            errorSpline = truthSpline - bestSpline.unfold(foldedIndices, foldedBasis)
            maxError = [max(np.abs(bound[0]), np.abs(bound[1])) for bound in errorSpline.range_bounds()]
            if np.max(maxError) > tolerance:
                break
            else:
                currentFold = bestSpline
        currentSpline = currentFold.unfold(foldedIndices, foldedBasis)
    return currentSpline.scale(scaleDep)

def reparametrize(self, newDomain):
    if not(len(newDomain) == self.nInd): raise ValueError("Invalid newDomain")
    knotList = []
    for order, knots, d, nD in zip(self.order, self.knots, self.domain(), newDomain):
        if nD is not None:
            divisor = d[1] - d[0]
            if not(divisor > np.finfo(self.knots[0].dtype).eps): raise ValueError("Invalid spline domain")
            slope = (nD[1] - nD[0]) / divisor
            if not(abs(slope) > 0.0): raise ValueError("Invalid newDomain")
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
    return type(self)(self.nInd, self.nDep, self.order, self.nCoef, knotList, self.coefs, self.metadata)   

def reverse(self, variable = 0):
    # Check to make sure variable is in range
    if variable < 0 or variable >= self.nInd:  raise ValueError("Improper independent variable index")

    # Create a new spline with only the indicated independent variable
    myIndices = [ix for ix in range(self.nInd) if ix != variable ]
    folded, basisInfo = self.fold(myIndices)

    # Create a new spline with the parametrization reversed
    knots = folded.knots[0]
    firstKnot = knots[0]
    lastKnot = knots[-1]
    newKnots = [firstKnot + lastKnot - knot for knot in knots[::-1]]
    newCoefs = []
    for iDep in range(folded.nDep):
        newCoefs.append(self.coefs[iDep][::-1])
    newFolded = type(self)(folded.nInd, folded.nDep, folded.order, folded.nCoef, (newKnots,), newCoefs, folded.metadata)
    return newFolded.unfold(myIndices, basisInfo)

def split(self, minContinuity = 0, breaks = None):
    if minContinuity < 0: raise ValueError("minContinuity must be >= 0")
    if breaks is not None and len(breaks) != self.nInd: raise ValueError("Invalid breaks")
    if self.nInd < 1: return self

    # Step 1: Determine the knots to insert.
    newKnotsList = []
    for i, order, knots in zip(range(self.nInd), self.order, self.knots):
        unique, counts = np.unique(knots, return_counts=True)
        newKnots = []
        for knot, count in zip(unique, counts):
            assert count <= order
            if count > order - 1 - minContinuity:
                newKnots += [knot] * (order - count)
        if breaks is not None:
            for knot in breaks[i]:
                if knot not in unique:
                    newKnots += [knot] * order
        newKnotsList.append(newKnots)    
    
    # Step 2: Insert the knots.
    spline = self.insert_knots(newKnotsList)
    if spline is self:
        return np.full((1,) * spline.nInd, spline)

    # Step 3: Store the indices of the full order knots.
    indexList = []
    splineCount = []
    totalSplineCount = 1
    for order, knots in zip(spline.order, spline.knots):
        unique, counts = np.unique(knots, return_counts=True)
        indices = np.searchsorted(knots, unique)
        fullOrder = []
        for ix, count in zip(indices, counts):
            if count == order:
                fullOrder.append(ix)
        indexList.append(fullOrder)
        splines = len(fullOrder) - 1
        splineCount.append(splines)
        totalSplineCount *= splines

    # Step 4: Slice up the spline.
    splineArray = np.empty(totalSplineCount, object)
    for i in range(totalSplineCount):
        knotsList = []
        coefIndex = [slice(None)] # First index is for nDep
        ix = i
        for order, knots, splines, indices in zip(spline.order, spline.knots, splineCount, indexList):
            j = ix % splines
            ix = ix // splines
            leftIndex = indices[j]
            rightIndex = indices[j + 1]
            knotsList.append(knots[leftIndex:rightIndex + order])
            coefIndex.append(slice(leftIndex, rightIndex))
        coefs = spline.coefs[tuple(coefIndex)]
        splineArray[i] = type(spline)(spline.nInd, spline.nDep, spline.order, coefs.shape[1:], knotsList, coefs, spline.metadata)

    # Return the transpose because we put the splines into splineArray dimensions in reverse order.
    return splineArray.reshape(tuple(reversed(splineCount))).T

def transpose(self, axes=None):
    if axes is None:
        axes = range(self.nInd)[::-1]
    order = []
    nCoef = []
    knots = []
    coefAxes = [0]
    for axis in axes:
        order.append(self.order[axis])
        nCoef.append(self.nCoef[axis])
        knots.append(self.knots[axis])
        coefAxes.append(axis + 1)
    return type(self)(self.nInd, self.nDep, order, nCoef, knots, np.transpose(self.coefs, coefAxes), self.metadata)

def trim(self, newDomain):
    if not(len(newDomain) == self.nInd): raise ValueError("Invalid newDomain")
    if self.nInd < 1: return self
    newDomain = np.array(newDomain, self.knots[0].dtype, copy=True) # Force dtype and convert None to nan
    epsilon = np.finfo(newDomain.dtype).eps

    # Step 1: Determine the knots to insert at the new domain bounds.
    newKnotsList = []
    for (order, knots, bounds) in zip(self.order, self.knots, newDomain):
        if not(len(bounds) == 2): raise ValueError("Invalid newDomain")
        unique, counts = np.unique(knots, return_counts=True)
        leftBound = False # Do we have a left bound?
        newKnots = []

        if not np.isnan(bounds[0]):
            if not(knots[order - 1] <= bounds[0] <= knots[-order]): raise ValueError("Invalid newDomain")
            leftBound = True
            i = np.searchsorted(unique, bounds[0])
            if unique[i] - bounds[0] < epsilon:
                bounds[0] = unique[i]
                multiplicity = order - counts[i]
            elif i > 0 and bounds[0] - unique[i - 1] < epsilon:
                bounds[0] = unique[i - 1]
                multiplicity = order - counts[i - 1]
            else:
                multiplicity = order
        
            newKnots += multiplicity * [bounds[0]]

        if not np.isnan(bounds[1]):
            if not(knots[order - 1] <= bounds[1] <= knots[-order]): raise ValueError("Invalid newDomain")
            if leftBound:
                if not(bounds[0] < bounds[1]): raise ValueError("Invalid newDomain")
            i = np.searchsorted(unique, bounds[1])
            if unique[i] - bounds[1] < epsilon:
                bounds[1] = unique[i]
                multiplicity = order - counts[i]
            elif i > 0 and bounds[1] - unique[i - 1] < epsilon:
                bounds[1] = unique[i - 1]
                multiplicity = order - counts[i - i]
            else:
                multiplicity = order
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
        leftIndex = 0 if np.isnan(bounds[0]) else np.searchsorted(knots, bounds[0])
        rightIndex = len(knots) - order if np.isnan(bounds[1]) else np.searchsorted(knots, bounds[1])
        knotsList.append(knots[leftIndex:rightIndex + order])
        coefIndex.append(slice(leftIndex, rightIndex))
    coefs = spline.coefs[tuple(coefIndex)]

    return type(spline)(spline.nInd, spline.nDep, spline.order, coefs.shape[1:], knotsList, coefs, spline.metadata)

def trimmed_range_bounds(self, domainBounds):
    domainBounds = np.array(domainBounds, copy=True)
    for original, trim in zip(self.domain(), domainBounds):
        trim[0] = max(original[0], trim[0] - Manifold.minSeparation)
        trim[1] = min(original[1], trim[1] + Manifold.minSeparation)
    trimmedSpline = self.trim(domainBounds)
    return trimmedSpline, trimmedSpline.range_bounds()

def unfold(self, foldedInd, coefficientlessSpline):
    if not(len(foldedInd) == coefficientlessSpline.nInd): raise ValueError("Invalid coefficientlessSpline")
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

    unfoldedSpline = type(self)(len(unfoldedOrder), unfoldedNDep, unfoldedOrder, unfoldedNCoef, unfoldedKnots, unfoldedCoefs, self.metadata)
    return unfoldedSpline