import numpy as np
from bspy import Solid, Boundary, Manifold, Hyperplane, Spline

def solid_edges(solid, subdivide = False):
    """
    A generator for edges of the solid.

    Yields
    -------
    (point1, point2, normal) : `tuple(numpy.array, numpy.array, numpy.array)`
        Starting point, ending point, and normal for an edge of the solid.


    Notes
    -----
    The edges are not guaranteed to be connected or in any particular order, and typically aren't.

    If the solid is a number line (dimension 1), the generator yields a tuple with two scalar values (start, end).
    """
    if solid.dimension > 1:
        for boundary in solid.boundaries:
            for domainEdge in solid_edges(boundary.domain, subdivide or not isinstance(boundary.manifold, Hyperplane)):
                yield (boundary.manifold.evaluate(domainEdge[0]), boundary.manifold.evaluate(domainEdge[1]), boundary.manifold.normal(domainEdge[0]))
    else:
        solid.boundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), -boundary.manifold.normal(0.0)))
        leftB = 0
        rightB = 0
        while leftB < len(solid.boundaries):
            if solid.boundaries[leftB].manifold.normal(0.0) < 0.0:
                leftPoint = solid.boundaries[leftB].manifold.evaluate(0.0)
                while rightB < len(solid.boundaries):
                    rightPoint = solid.boundaries[rightB].manifold.evaluate(0.0)
                    if leftPoint - Manifold.minSeparation < rightPoint and solid.boundaries[rightB].manifold.normal(0.0) > 0.0:
                        if subdivide:
                            dt = 0.1
                            t = leftPoint.copy()
                            while t + dt < rightPoint:
                                yield (t, t + dt)
                                t += dt
                            yield (t, rightPoint)
                        else:
                            yield (leftPoint, rightPoint)
                        leftB = rightB
                        rightB += 1
                        break
                    rightB += 1
            leftB += 1


def create_segments_from_solid(solid, includeManifold=False):
    segments = []

    if includeManifold and solid.dimension == 3:
        for boundary in solid.boundaries:
            if isinstance(boundary.manifold, Spline):
                spline = boundary.manifold.spline
                domain = spline.domain()
                segment = []
                for u in np.linspace(domain[0][0], domain[0][1], 5):
                    for v in np.linspace(domain[1][0], domain[1][1], 10):
                        segment.append(spline((u, v)))
                segments.append(segment)
                segment = []
                for v in np.linspace(domain[1][0], domain[1][1], 5):
                    for u in np.linspace(domain[0][0], domain[0][1], 10):
                        segment.append(spline((u, v)))
                segments.append(segment)
    
    for edge in solid_edges(solid):
        middle = 0.5 * (edge[0] + edge[1])
        normal = middle + 0.1 * edge[2]
        segments.append((edge[0], edge[1]))
        segments.append((middle, normal))
    
    return segments

def hyperplane_1D(normal, offset):
    assert np.isscalar(normal) or len(normal) == 1
    normalizedNormal = np.atleast_1d(normal)
    normalizedNormal = normalizedNormal / np.linalg.norm(normalizedNormal)
    return Hyperplane(normalizedNormal, offset * normalizedNormal, 0.0)

def hyperplane_2D(normal, offset):
    assert len(normal) == 2
    normalizedNormal = np.atleast_1d(normal)
    normalizedNormal = normalizedNormal / np.linalg.norm(normalizedNormal)
    return Hyperplane(normalizedNormal, offset * normalizedNormal, np.transpose(np.array([[normal[1], -normal[0]]])))

def hyperplane_domain_from_point(hyperplane, point):
    tangentSpaceTranspose = np.transpose(hyperplane._tangentSpace)
    return np.linalg.inv(tangentSpaceTranspose @ hyperplane._tangentSpace) @ tangentSpaceTranspose @ (point - hyperplane._point)

def create_faceted_solid_from_points(dimension, points, containsInfinity = False):
    # create_faceted_solid_from_points only works for dimension 2 so far.
    assert dimension == 2
    assert len(points) > 2
    assert len(points[0]) == dimension

    solid = Solid(dimension, containsInfinity)

    previousPoint = np.array(points[len(points)-1])
    for point in points:
        point = np.array(point)
        vector = point - previousPoint
        normal = np.array([-vector[1], vector[0]])
        normal = normal / np.linalg.norm(normal)
        hyperplane = hyperplane_2D(normal,np.dot(normal,point))
        domain = Solid(dimension-1, False)
        previousPointDomain = hyperplane_domain_from_point(hyperplane, previousPoint)
        pointDomain = hyperplane_domain_from_point(hyperplane, point)
        if previousPointDomain < pointDomain:
            domain.boundaries.append(Boundary(hyperplane_1D(-1.0, -previousPointDomain), Solid(dimension-2, True)))
            domain.boundaries.append(Boundary(hyperplane_1D(1.0, pointDomain), Solid(dimension-2, True)))
        else:
            domain.boundaries.append(Boundary(hyperplane_1D(-1.0, -pointDomain), Solid(dimension-2, True)))
            domain.boundaries.append(Boundary(hyperplane_1D(1.0, previousPointDomain), Solid(dimension-2, True)))
        solid.boundaries.append(Boundary(hyperplane, domain))
        previousPoint = point

    return solid

def create_smooth_solid_from_points(dimension, points, containsInfinity = False):
    # create_smooth_solid_from_points only works for dimension 2 so far.
    assert dimension == 2
    assert len(points) > 2
    assert len(points[0]) == dimension

    solid = Solid(dimension, containsInfinity)

    t = 0.0
    uValues = [t]
    dataPoints = np.array(points, np.float64)
    previousPoint = dataPoints[0]
    for point in dataPoints[1:]:
        t += np.linalg.norm(point - previousPoint)
        uValues.append(t)
        previousPoint = point
    t += np.linalg.norm(dataPoints[0] - previousPoint)
    uValues.append(t)
    dataPoints = np.append(dataPoints, (dataPoints[0],), axis=0)

    spline = Spline.least_squares(uValues, dataPoints.T, (4,) * (dimension - 1), tolerance = 0.1)
    domain = Solid(dimension-1, False)
    domain.boundaries.append(Boundary(hyperplane_1D(-1.0, 0.0), Solid(dimension-2, True)))
    domain.boundaries.append(Boundary(hyperplane_1D(1.0, t), Solid(dimension-2, True)))
    solid.boundaries.append(Boundary(spline, domain))

    return solid

def create_star(radius, center, angle, smooth = False):
    points = 5
    vertices = []

    if smooth:
        dAngle = 6.2832 / points
        for i in range(points):
            vertices.append([radius*np.cos(angle + i*dAngle) + center[0], radius*np.sin(angle + i*dAngle) + center[1]])
            vertices.append([0.5*radius*np.cos(angle + (i + 0.5)*dAngle) + center[0], 0.5*radius*np.sin(angle + (i + 0.5)*dAngle) + center[1]])

        star = create_smooth_solid_from_points(2, vertices)
    else:
        for i in range(points):
            vertices.append([radius*np.cos(angle - ((2*i)%points)*6.2832/points) + center[0], radius*np.sin(angle - ((2*i)%points)*6.2832/points) + center[1]])

        star = create_faceted_solid_from_points(2, vertices)

        nt = (vertices[1][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[1][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0])
        u = ((vertices[3][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[3][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0]))/nt
        for boundary in star.boundaries:
            u0 = boundary.domain.boundaries[0].manifold._point[0]
            u1 = boundary.domain.boundaries[1].manifold._point[0]
            boundary.domain.boundaries.append(Boundary(hyperplane_1D(1.0, u0 + (1.0 - u)*(u1 - u0)), Solid(0, True)))
            boundary.domain.boundaries.append(Boundary(hyperplane_1D(-1.0, -(u0 + u*(u1 - u0))), Solid(0, True)))

    return star

def extrude_solid(solid, path):
    assert len(path) > 1
    assert solid.dimension+1 == len(path[0])
    
    extrusion = Solid(solid.dimension+1, False)

    # Extrude boundaries along the path
    point = None
    for nextPoint in path:
        nextPoint = np.atleast_1d(nextPoint)
        if point is None:
            point = nextPoint
            continue
        tangent = nextPoint - point
        extent = tangent[solid.dimension]
        tangent = tangent / extent
        # Extrude each boundary
        for boundary in solid.boundaries:
            # Construct a normal orthogonal to both the boundary tangent space and the path tangent
            extruded_normal = np.full((extrusion.dimension), 0.0)
            extruded_normal[0:solid.dimension] = boundary.manifold._normal[:]
            extruded_normal[solid.dimension] = -np.dot(boundary.manifold._normal, tangent[0:solid.dimension])
            extruded_normal = extruded_normal / np.linalg.norm(extruded_normal)
            # Construct a point that adds the boundary point to the path point
            extruded_point = np.full((extrusion.dimension), 0.0)
            extruded_point[0:solid.dimension] = boundary.manifold._point[:]
            extruded_point += point
            # Combine the boundary tangent space and the path tangent
            extruded_tangentSpace = np.full((extrusion.dimension, solid.dimension), 0.0)
            if solid.dimension > 1:
                extruded_tangentSpace[0:solid.dimension, 0:solid.dimension-1] = boundary.manifold._tangentSpace[:,:]
            extruded_tangentSpace[:, solid.dimension-1] = tangent[:]
            extrudedHyperplane = Hyperplane(extruded_normal, extruded_point, extruded_tangentSpace)
            # Construct a domain for the extruded boundary
            if boundary.domain.dimension > 0:
                # Extrude the boundary's domain to include path domain
                domainPath = []
                domainPoint = np.full((solid.dimension), 0.0)
                domainPath.append(domainPoint)
                domainPoint = np.full((solid.dimension), 0.0)
                domainPoint[solid.dimension-1] = extent
                domainPath.append(domainPoint)
                extrudedDomain = extrude_solid(boundary.domain, domainPath)
            else:
                extrudedDomain = Solid(solid.dimension, False)
                extrudedDomain.boundaries.append(Boundary(hyperplane_1D(-1.0, 0.0), Solid(0, True)))
                extrudedDomain.boundaries.append(Boundary(hyperplane_1D(1.0, extent), Solid(0, True)))
            # Add extruded boundary
            extrusion.boundaries.append(Boundary(extrudedHyperplane, extrudedDomain))
        
        # Move onto the next point
        point = nextPoint

    # Add end cap boundaries
    extrudedHyperplane = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, 0.0, True)
    extrudedHyperplane = extrudedHyperplane.translate(path[0])
    extrusion.boundaries.append(Boundary(extrudedHyperplane, solid))
    extrudedHyperplane = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, 0.0, False)
    extrudedHyperplane = extrudedHyperplane.translate(path[-1])
    extrusion.boundaries.append(Boundary(extrudedHyperplane, solid))

    return extrusion

def find_boundary(solid, name):
    for boundary in solid.boundaries:
        if isinstance(boundary.manifold, Spline) and \
            "Name" in boundary.manifold.spline.metadata and \
            boundary.manifold.spline.metadata["Name"] == name:
            return boundary
    return None