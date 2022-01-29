import numpy as np
from bspy import Spline, DrawableSpline, bspyApp

def CreateSplineFromMesh(xRange, zRange, yFunction):
    order = (3, 3)
    coefficients = np.zeros((4, xRange[2], zRange[2]), np.float32)
    knots = (np.zeros(xRange[2] + order[0], np.float32), np.zeros(zRange[2] + order[1], np.float32))
    knots[0][0] = xRange[0]
    knots[0][1:xRange[2]+1] = np.linspace(xRange[0], xRange[1], xRange[2], dtype=np.float32)[:]
    knots[0][xRange[2]+1:] = xRange[1]
    knots[1][0] = zRange[0]
    knots[1][1:zRange[2]+1] = np.linspace(zRange[0], zRange[1], zRange[2], dtype=np.float32)[:]
    knots[1][zRange[2]+1:] = zRange[1]
    for i in range(xRange[2]):
        for j in range(zRange[2]):
            coefficients[0, i, j] = knots[0][i]
            coefficients[1, i, j] = yFunction(knots[0][i], knots[1][j])
            coefficients[2, i, j] = knots[1][j]
            coefficients[3, i, j] = 1.0
    
    return DrawableSpline(2, 4, order, (xRange[2], zRange[2]), knots, coefficients)

if __name__=='__main__':
    app = bspyApp()
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: np.sin(4*np.sqrt(x*x + y*y))))
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x + y*y - 1))
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x - y*y))
    for i in range(8):
        app.AddSpline(DrawableSpline(1, 4, (3,), (5,), (np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5], np.float32),), np.array([[-1, 0, 0, 1], [-0.5, i/8.0, 0, 1], [0,0,0,1], [0.5, -i/8.0, 0, 1], [1,0,0,1]], np.float32)))
    app.AddSpline(Spline.load("C:/Users/ericb/OneDrive/Desktop/TomsNasty.npz", DrawableSpline))
    app.mainloop()