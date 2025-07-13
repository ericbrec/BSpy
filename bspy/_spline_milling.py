import numpy as np
import bspy.spline
import bspy.spline_block
from collections import namedtuple

def line_of_curvature(self, uvStart, is_max, tolerance = 1.0e-3):
    if self.nInd != 2:  raise ValueError("Surface must have two independent variables")
    if len(uvStart) != 2:  raise ValueError("uvStart must have two components")
    uvDomain = self.domain()
    if uvStart[0] < uvDomain[0, 0] or uvStart[0] > uvDomain[0, 1] or \
       uvStart[1] < uvDomain[1, 0] or uvStart[1] > uvDomain[1, 1]:
        raise ValueError("uvStart is outside domain of the surface")
    is_max = bool(is_max) # Ensure is_max is a boolean for XNOR operation
    
    # Define the callback function for the ODE solver
    def curvatureLineCallback(t, u):
        # Evaluate the surface information needed.
        uv = np.maximum(uvDomain[:, 0], np.minimum(uvDomain[:, 1], u[:, 0]))
        su = self.derivative((1, 0), uv)
        sv = self.derivative((0, 1), uv)
        suu = self.derivative((2, 0), uv)
        suv = self.derivative((1, 1), uv)
        svv = self.derivative((0, 2), uv)
        suuu = self.derivative((3, 0), uv)
        suuv = self.derivative((2, 1), uv)
        suvv = self.derivative((1, 2), uv)
        svvv = self.derivative((0, 3), uv)
        normal = self.normal(uv)

        # Calculate curvature matrix and its derivatives.
        sU = np.concatenate((su, sv)).reshape(2, -1)
        sUu = np.concatenate((suu, suv)).reshape(2, -1)
        sUv = np.concatenate((suv, svv)).reshape(2, -1)
        sUU = np.concatenate((suu, suv, suv, svv)).reshape(2, 2, -1)
        sUUu = np.concatenate((suuu, suuv, suuv, suvv)).reshape(2, 2, -1)
        sUUv = np.concatenate((suuv, suvv, suvv, svvv)).reshape(2, 2, -1)
        fffI = np.linalg.inv(sU @ sU.T) # Inverse of first fundamental form
        k = fffI @ (sUU @ normal) # Curvature matrix
        ku = fffI @ (sUUu @ normal - (sUu @ sU.T + sU @ sUu.T) @ k - sUU @ (sU.T @ k[:, 0]))
        kv = fffI @ (sUUv @ normal - (sUv @ sU.T + sU @ sUv.T) @ k - sUU @ (sU.T @ k[:, 1]))

        # Determine principle curvatures and directions, and assign new direction.
        curvatures, directions = np.linalg.eig(k)
        curvatureDelta = curvatures[1] - curvatures[0]
        if abs(curvatureDelta) < tolerance:
            # If we're at an umbilic, use the last direction (jacobian is zero at umbilic).
            direction = u[:, 1]
            jacobian = np.zeros((2,2,1), self.coefs.dtype)
        else:
            # Otherwise, compute the lhs inverse for the jacobian.
            directionsInverse = np.linalg.inv(directions)
            eigenIndex = 0 if bool(curvatures[0] > curvatures[1]) == is_max else 1
            direction = directions[:, eigenIndex]
            B = np.zeros((2, 2), self.coefs.dtype)
            B[0, 1 - eigenIndex] = np.dot(directions[:, 1], direction) / curvatureDelta
            B[1, 1 - eigenIndex] = -np.dot(directions[:, 0], direction) / curvatureDelta
            lhsInv =  directions @ B @ directionsInverse

            # Adjust the direction for consistency.
            if np.dot(direction, u[:, 1]) < -tolerance:
                direction *= -1

            # Compute the jacobian for the direction.
            jacobian = np.empty((2,2,1), self.coefs.dtype)
            jacobian[:,0,0] = lhsInv @ ku @ direction
            jacobian[:,1,0] = lhsInv @ kv @ direction

        return direction, jacobian

    # Generate the initial guess for the line of curvature.
    uvStart = np.atleast_1d(uvStart)
    direction = 0.5 * (uvDomain[:,0] + uvDomain[:,1]) - uvStart # Initial guess toward center
    distanceFromCenter = np.linalg.norm(direction)
    if distanceFromCenter < 10 * tolerance:
        # If we're at the center, just point to the far corner.
        direction = np.array((1.0, 1.0)) / np.sqrt(2)
    else:
        direction /= distanceFromCenter

    # Compute line of curvature direction at start.
    direction, jacobian = curvatureLineCallback(0.0, np.array(((uvStart[0], direction[0]), (uvStart[1], direction[1]))))

    # Calculate distance to the boundary in that direction.
    if direction[0] < -tolerance:
        uBoundaryDistance = (uvDomain[0, 0] - uvStart[0]) / direction[0]
    elif direction[0] > tolerance:
        uBoundaryDistance = (uvDomain[0, 1] - uvStart[0]) / direction[0]
    else:
        uBoundaryDistance = np.inf
    if direction[1] < -tolerance:
        vBoundaryDistance = (uvDomain[1, 0] - uvStart[1]) / direction[1]
    elif direction[1] > tolerance:
        vBoundaryDistance = (uvDomain[1, 1] - uvStart[1]) / direction[1]
    else:
        vBoundaryDistance = np.inf
    boundaryDistance = min(uBoundaryDistance, vBoundaryDistance)

    # Construct the initial guess from start point to boundary.
    initialGuess = bspy.spline.Spline.line(uvStart, uvStart + boundaryDistance * direction).elevate([2])

    # Solve the ODE and return the line of curvature confined to the surface's domain.
    solution = initialGuess.solve_ode(1, 0, curvatureLineCallback, tolerance, includeEstimate = True)
    return solution.confine(uvDomain)

def offset(self, edgeRadius, bitRadius=None, angle=np.pi / 2.2, path=None, subtract=False, removeCusps=False, tolerance = 1.0e-4):
    if self.nDep < 2 or self.nDep > 3 or self.nDep - self.nInd != 1: raise ValueError("The offset is only defined for 2D curves and 3D surfaces with well-defined normals.")
    if edgeRadius < 0:
        raise ValueError("edgeRadius must be >= 0")
    elif edgeRadius == 0:
        return self
    if bitRadius is None:
        bitRadius = edgeRadius
    elif bitRadius < edgeRadius:
        raise ValueError("bitRadius must be >= edgeRadius")
    if angle < 0 or angle >= np.pi / 2: raise ValueError("angle must in the range [0, pi/2)")
    if path is not None and (path.nInd != 1 or path.nDep != 2 or self.nInd != 2):
        raise ValueError("path must be a 2D curve and self must be a 3D surface")

    # Compute new order, knots, and fillets for offset (ensure order is at least 4).
    Fillet = namedtuple('Fillet', ('adjustment', 'isFillet', 'point', 'n0', 'n1'))
    newOrder = []
    newKnotList = []
    newUniqueList = []
    filletList = []
    for order, knots in zip(self.order, self.knots):
        min4Order = max(order, 4)
        unique, counts = np.unique(knots, return_counts=True)
        counts += min4Order - order # Ensure order is at least 4
        newOrder.append(min4Order)
        adjustment = 0
        epsilon = np.finfo(knots.dtype).eps

        # Add first knot.
        fillets = [Fillet(adjustment, False, None, None, None)]
        newKnots = [unique[0]] * counts[0]
        newUnique = [unique[0]]

        # Add internal knots, checking for C1 discontinuities needing fillets.
        for knot, count in zip(unique[1:-1], counts[1:-1]):
            fillets.append(Fillet(adjustment, False, None, None, None))
            newKnots += [knot + adjustment] * count
            newUnique.append(knot + adjustment)
            # Check for lack of C1 continuity (need for a fillet)
            if count >= min4Order - 1:
                fillets.append(Fillet(adjustment, True, self(knot), self.normal(knot - epsilon), self.normal(knot + epsilon)))
                adjustment += 1
                newKnots += [knot + adjustment] * (min4Order - 1)
                newUnique.append(knot + adjustment)

        # Add last knot.
        fillets.append(Fillet(adjustment, False, None, None, None))
        newKnots += [unique[-1] + adjustment] * counts[-1]
        newUnique.append(unique[-1] + adjustment)

        # Build fillet and knot lists.
        filletList.append(fillets)
        newKnotList.append(np.array(newKnots, knots.dtype))
        newUniqueList.append(np.array(newUnique, knots.dtype))

    # Determine geometry of drill bit.
    if subtract:
        edgeRadius *= -1
        bitRadius *= -1
    w = bitRadius - edgeRadius
    h = w * np.tan(angle)
    bottom = np.sin(angle)
    bottomRadius = edgeRadius + h / bottom

    # Define drill bit function.
    if abs(w) < tolerance and path is None: # Simple offset curve or surface
        def drillBit(uv):
            return self(uv) + edgeRadius * self.normal(uv)
    elif self.nDep == 2: # General offset curve
        def drillBit(u):
            xy = self(u)
            normal = self.normal(u)
            upward = np.sign(normal[1])
            if upward * normal[1] <= bottom:
                xy[0] += edgeRadius * normal[0] + w * np.sign(normal[0])
                xy[1] += edgeRadius * normal[1]
            else:
                xy[0] += bottomRadius * normal[0]
                xy[1] += bottomRadius * normal[1] - upward * h
            return xy
    elif self.nDep == 3 and path is None: # General offset surface
        def drillBit(uv):
            xyz = self(uv)
            normal = self.normal(uv)
            upward = np.sign(normal[1])
            if upward * normal[1] <= bottom:
                norm = np.sqrt(normal[0] * normal[0] + normal[2] * normal[2])
                xyz[0] += edgeRadius * normal[0] + w * normal[0] / norm
                xyz[1] += edgeRadius * normal[1]
                xyz[2] += edgeRadius * normal[2] + w * normal[2] / norm
            else:
                xyz[0] += bottomRadius * normal[0]
                xyz[1] += bottomRadius * normal[1] - upward * h
                xyz[2] += bottomRadius * normal[2]
            return xyz
    elif self.nDep == 3: # General offset of a given path along a surface
        surface = self
        self = path # Redefine self to be the path (used below for fitting)
        def drillBit(u):
            uv = self(u)
            xyz = surface(uv)
            normal = surface.normal(uv)
            upward = np.sign(normal[1])
            if upward * normal[1] <= bottom:
                norm = np.sqrt(normal[0] * normal[0] + normal[2] * normal[2])
                xyz[0] += edgeRadius * normal[0] + w * normal[0] / norm
                xyz[1] += edgeRadius * normal[1]
                xyz[2] += edgeRadius * normal[2] + w * normal[2] / norm
            else:
                xyz[0] += bottomRadius * normal[0]
                xyz[1] += bottomRadius * normal[1] - upward * h
                xyz[2] += bottomRadius * normal[2]
            return xyz
    else: # Should never get here (exception raised earlier)
        raise ValueError("The offset is only defined for 2D curves and 3D surfaces with well-defined normals.")

    # Fit new spline to offset by drill bit.
    offset = bspy.spline.Spline.fit(self.domain(), drillBit, newOrder, newKnots, tolerance)

    # Remove cusps as required (only applies to offset curves).
    if removeCusps and self.nInd == 1:
        # Find the cusps by checking for tangent direction reversal between the spline and offset.
        cusps = []
        previousKnot = None
        start = None
        for knot in np.unique(offset.knots[0][offset.order[0]:offset.nCoef[0]]):
            tangent = self.derivative((1,), knot)
            if path is not None:
                tangent = surface.jacobian(path(knot)) @ tangent
            flipped = np.dot(tangent, offset.derivative((1,), knot)) < 0
            if flipped and start is None:
                start = knot
            if not flipped and start is not None:
                cusps.append((start, previousKnot))
                start = None
            previousKnot = knot

        # Remove the cusps by intersecting the offset segments before and after each cusp.
        segmentList = []
        for cusp in cusps:
            domain = offset.domain()
            before = offset.trim(((domain[0][0], cusp[0]),))
            after = -offset.trim(((cusp[1], domain[0][1]),))
            if path is not None:
                # Project before and after onto a 2D plane defined by the offset tangent 
                # and the surface normal at the start of the cusp.
                # This is necessary to find the intersection point (2 equations, 2 unknowns).
                tangent = offset.derivative((1,), cusp[0])
                projection = np.concatenate((tangent / np.linalg.norm(tangent),
                    surface.normal(path(cusp[0])))).reshape((2,3))
                before = before.transform(projection)
                after = after.transform(projection)
            block = bspy.spline_block.SplineBlock([[before, after]])
            intersections = block.zeros()
            for intersection in intersections:
                segmentList.append(offset.trim(((domain[0][0], intersection[0]),)))
                offset = offset.trim(((intersection[1], domain[0][1]),))
        segmentList.append(offset)
        offset = bspy.spline.Spline.join(segmentList)
    
    return offset
