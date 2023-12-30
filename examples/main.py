import numpy as np
from bspy import Spline, DrawableSpline, bspyApp

def CreateSplineFromMesh(xRange, zRange, yFunction):
    order = (3, 3)
    coefficients = np.zeros((1, xRange[2], zRange[2]))
    knots = (np.zeros(xRange[2] + order[0]), np.zeros(zRange[2] + order[1]))
    knots[0][0] = xRange[0]
    knots[0][1:xRange[2]+1] = np.linspace(xRange[0], xRange[1], xRange[2])[:]
    knots[0][xRange[2]+1:] = xRange[1]
    knots[1][0] = zRange[0]
    knots[1][1:zRange[2]+1] = np.linspace(zRange[0], zRange[1], zRange[2])[:]
    knots[1][zRange[2]+1:] = zRange[1]
    for i in range(xRange[2]):
        for j in range(zRange[2]):
            coefficients[0, i, j] = yFunction(knots[0][i], knots[1][j])
    
    return Spline(2, 1, order, (xRange[2], zRange[2]), knots, coefficients)

def Create4Sphere(samples):
    values = []
    for radius in np.linspace(1.0, 10.0, samples):
        for theta in np.linspace(0.0, 1.0, samples):
            for phi in np.linspace(0.1, 0.9, samples):
                values.append((radius, theta, phi, 
                    radius * np.cos(theta * 2 * np.pi) * np.sin(phi * np.pi),
                    radius * np.cos(phi * np.pi),
                    radius * np.sin(theta * 2 * np.pi) * np.sin(phi * np.pi)))
    return Spline.least_squares(3, 3, (4, 4, 4), values)

if __name__=='__main__':
    app = bspyApp()
    ds1 = DrawableSpline.make_drawable(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: np.sin(4*np.sqrt(x*x + y*y))))
    ds2 = DrawableSpline.make_drawable(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x + y*y - 1))
    ds3 = DrawableSpline.make_drawable(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x - y*y))
    app.show(ds1, "Surface1")
    app.show(ds2, "Surface2")
    app.show(ds3, "Surface3")
    coefs = np.zeros((6, *ds1.nCoef), ds1.coefs.dtype)
    coefs[:3] = ds1.coefs
    coefs[3:6] = ds1.coefs
    cs1 = Spline(2, 6, ds1.order, ds1.nCoef, ds1.knots, coefs)
    app.show(cs1, "Surface1 6D")
    coefs = np.zeros((5, *ds2.nCoef), ds2.coefs.dtype)
    coefs[:3] = ds2.coefs
    coefs[3] = ds2.coefs[0]
    coefs[4] = ds2.coefs[2]
    cs2 = Spline(2, 5, ds2.order, ds2.nCoef, ds2.knots, coefs)
    app.show(cs2, "Surface2 5D")
    coefs = np.zeros((4, *ds3.nCoef), ds3.coefs.dtype)
    coefs[:3] = ds3.coefs
    coefs[3] = ds3.coefs[0]
    cs3 = Spline(2, 4, ds3.order, ds3.nCoef, ds3.knots, coefs)
    app.show(cs3, "Surface3 4D")
    for i in range(8):
        app.show(Spline(1, 1, (3,), (5,), (np.array((-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.0, 1.0)),), np.array((0, i/8.0, 0, -i/8.0, 0))))
    app.show(Spline.load("C:/Users/ericb/OneDrive/Desktop/TomsNasty.npz", DrawableSpline))
    app.show(Create4Sphere(5), "Solid1")
    app.mainloop()