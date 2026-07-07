import numpy as np
from bspy import Solid, Boundary, Hyperplane, Spline, Viewer
import solidUtils as utils

if __name__ == "__main__":
    cubeA = Hyperplane.create_hypercube([[-2.5, 0.5]] * 3)
    print(cubeA.volume_integral(lambda x: 1.0), 3.0*3.0*3.0)
    print(cubeA.surface_integral(lambda x, n: n), 3.0*3.0*6.0)
    print(cubeA.winding_number([-1,-1,0]))
    print(cubeA.winding_number([4,1,0]))
    cubeB = Hyperplane.create_hypercube([[-0.5, 1.5]] * 3)
    print(cubeB.volume_integral(lambda x: 1.0), 2.0*2.0*2.0)
    print(cubeB.surface_integral(lambda x, n: n), 2.0*2.0*6.0)
    
    viewer = Viewer()

    case = "pipes"
    if case == "pipes":
        square = Hyperplane.create_hypercube([[-1, 1]] * 2)
        star = utils.create_star(2.0, [0.0, 0.0], 90.0*6.28/360.0)
        extrudedSquare = utils.extrude_solid(square,[[-2,2,-4],[2,-2,4]])
        extrudedStar = utils.extrude_solid(star,[[-2,-2,-4],[2,2,4]])
        combined = extrudedStar.union(extrudedSquare)
        viewer.list(extrudedSquare, "Square pipe")
        viewer.list(extrudedStar, "Star pipe")
        viewer.draw(combined, "Union")
    if case == "cutout":
        viewer.frame.SetBackgroundColor(1.0, 1.0, 1.0)
        sphere = Hyperplane.create_hypercube([[-1, 1]] * 3)
        #sphere = Solid(3, False)
        #sphere.add_boundary(Boundary(BSpline(Spline.sphere(1.0, 0.001))))
        #sphere.add_boundary(Boundary(BSpline(Spline.cone(2.0, 0.01, 3.0, 0.001) + (0.0, 0.0, -1.5)))
        viewer.draw(sphere, "cube", np.array((.4, .6, 1, 1),np.float32))
        endCurve = [[1, 0], [0, 0], [0, 1]] @ Spline(1, 1, (3,), (5,), (np.array((-3.0, -3.0, -3.0, -0.6, 0.6, 3.0, 3.0, 3.0)),), np.array((0, 3.0/8.0, 0, -4.0/8.0, 0))).graph()
        spline = Spline.ruled_surface(endCurve + (0.0, -2.0, 0.0), endCurve + (0.0, 2.0, 0.0))
        halfSpace = Solid(3, False)
        halfSpace.add_boundary(Boundary(spline, Hyperplane.create_hypercube([[-3.0, 3.0], [0.0, 1.0]])))
        viewer.draw(halfSpace, "halfSpace", np.array((0, 1, 0, 1),np.float32))
        difference = sphere - halfSpace
        viewer.draw(difference, "difference")
        viewer.mainloop()
    if case == "paraboloids":
        order = 3
        knots = [0.0] * order + [1.0] * order
        nCoef = len(knots) - order
        spline = Spline(2, 3, (order, order), (nCoef, nCoef), (knots, knots), \
            (((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), \
            ((1.0, 0.0, 1.0), (0.0, -5.0, 0.0), (1.0, 0.0, 1.0)), \
            ((-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0))))
        cap = Hyperplane.create_axis_aligned(3, 1, 0.7, False)
        paraboloid = Solid(3, False)
        paraboloid.add_boundary(Boundary(spline))
        paraboloid.add_boundary(Boundary(cap, Hyperplane.create_hypercube([[-1.0, 1.0]] * 2)))
        viewer.list(paraboloid, "paraboloid", np.array((.4, .6, 1, 1),np.float32))

        spline = spline.copy()
        cap = Hyperplane.create_axis_aligned(3, 1, 0.7, False)
        paraboloid2 = Solid(3,False)
        paraboloid2.add_boundary(Boundary(spline))
        paraboloid2.add_boundary(Boundary(cap, Hyperplane.create_hypercube([[-1.0, 1.0]] * 2)))
        paraboloid2 = paraboloid2.translate(np.array((0.0, 0.5, 0.55)))
        viewer.list(paraboloid2, "paraboloid2", np.array((0, 1, 0, 1),np.float32))

        paraboloid3 = paraboloid + paraboloid2
        viewer.draw(paraboloid3, "p + p2")
    
    viewer.mainloop()