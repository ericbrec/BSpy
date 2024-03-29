import numpy as np
from bspy import Spline, Viewer

def CreateSplineFromMesh(xRange, zRange, yFunction):
    order = (3, 3)
    coefficients = np.zeros((3, xRange[2], zRange[2]))
    knots = (np.zeros(xRange[2] + order[0]), np.zeros(zRange[2] + order[1]))
    knots[0][0] = xRange[0]
    knots[0][1:xRange[2]+1] = np.linspace(xRange[0], xRange[1], xRange[2])[:]
    knots[0][xRange[2]+1:] = xRange[1]
    knots[1][0] = zRange[0]
    knots[1][1:zRange[2]+1] = np.linspace(zRange[0], zRange[1], zRange[2])[:]
    knots[1][zRange[2]+1:] = zRange[1]
    for i in range(xRange[2]):
        for j in range(zRange[2]):
            coefficients[0, i, j] = knots[0][i]
            coefficients[1, i, j] = yFunction(knots[0][i], knots[1][j])
            coefficients[2, i, j] = knots[1][j]
    
    return Spline(2, 3, order, (xRange[2], zRange[2]), knots, coefficients)

def Create4Sphere():
    innerSphere = Spline.sphere(1.0)
    outerSphere = 10.0 * innerSphere
    return Spline.ruled_surface(innerSphere, outerSphere)

if __name__=='__main__':
    viewer = Viewer()
    ds1 = CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: np.sin(4*np.sqrt(x*x + y*y)))
    ds2 = CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x + y*y - 1)
    ds3 = CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x - y*y)
    viewer.list(ds1, "Surface1")
    viewer.list(ds2, "Surface2")
    viewer.list(ds3, "Surface3")
    coefs = np.zeros((6, *ds1.nCoef), ds1.coefs.dtype)
    coefs[:3] = ds1.coefs
    coefs[3:6] = ds1.coefs
    cs1 = Spline(2, 6, ds1.order, ds1.nCoef, ds1.knots, coefs)
    viewer.list(cs1, "Surface1 6D")
    coefs = np.zeros((5, *ds2.nCoef), ds2.coefs.dtype)
    coefs[:3] = ds2.coefs
    coefs[3] = ds2.coefs[0]
    coefs[4] = ds2.coefs[2]
    cs2 = Spline(2, 5, ds2.order, ds2.nCoef, ds2.knots, coefs)
    viewer.list(cs2, "Surface2 5D")
    coefs = np.zeros((4, *ds3.nCoef), ds3.coefs.dtype)
    coefs[:3] = ds3.coefs
    coefs[3] = ds3.coefs[0]
    cs3 = Spline(2, 4, ds3.order, ds3.nCoef, ds3.knots, coefs)
    viewer.list(cs3, "Surface3 4D")
    for i in range(8):
        viewer.list(Spline(1, 1, (3,), (5,), (np.array((-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0)),), np.array((0, i/8.0, 0, -i/8.0, 0))))
    for spline in Spline.load("examples/TomsNasty.json"):
        viewer.list(spline)
    viewer.list(Create4Sphere(), "Solid1")
    viewer.mainloop()