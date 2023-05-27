import numpy as np

def clamp(self, left, right):
    bounds = self.nInd * [[None, None]]

    for ind in left:
        bounds[ind][0] = self.knots[ind][self.order[ind]-1]

    for ind in right:
        bounds[ind][1] = self.knots[ind][self.nCoef[ind]]

    return self.trim(bounds)

def common_basis(self, splines, indMap):
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

def elevate_and_insert_knots(self, m, newKnots):
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

def extrapolate(self, newDomain, continuityOrder):
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
            coefs = coefs.swapaxes(0, ind + 1) # Swap dependent and independent variable (swap back later)
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

def remove_knots(self, oldKnots=((),), maxRemovalsPerKnot=0, tolerance=None):
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

def trim(self, newDomain):
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