import math
import numpy as np
import bspy


airfoilData = [[1.0, 0.00322, [-0.756, 0.655], 0.0677],
               [0.552, 0.337, [-0.8577, 0.5141], 0.311],
               [0.220, 0.468, [-1.0, 0.0], 4.32],
               [0.0617,0.350, [-0.433,-0.901], 3.16],
               [0.0, 0.0, [0.0, -1.0], 0.391],
               [0.0676, -0.403, [0.537, -0.844], 4.73],
               [0.316, -0.539, [1.0, 0.0], 5.43],
               [0.514, -0.451, [0.796, 0.605], 1.19],
               [0.728, -0.243, [0.676, 0.737], 0.00],
               [1.0, 0.0, [0.942, 0.336], -5.23]]
airfoilData = [[x, y, 180.0 * math.atan2(dir[1], dir[0]) / math.pi, kappa] for [x, y, dir, kappa] in airfoilData] 
airfoilCurve = bspy.Spline.section(airfoilData)
rootFoil = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.15]] @ airfoilCurve
tipFoil = rootFoil.scale(0.2) + [0.85, 3.0, 0.0]
wingLoft = bspy.Spline.ruled_surface(rootFoil, tipFoil)
rootFoil.metadata = dict(Name = 'root airfoil')
tipFoil.metadata = dict(Name = 'tip airfoil')
wingLoft.metadata = dict(Name = 'wing')

sqrthalf = math.sqrt(0.5)
road1 = bspy.Spline(1, 2, [2], [2], [[0.0, 0, 1, 1]], [[0.0, 0], [-1.0, 0]])
road2 = bspy.Spline.section([[0, 0, 90, 1], [sqrthalf - 1.0, sqrthalf, 135, 1],
                             [-1, 1, 180, 1]])
road3 = bspy.Spline(1, 2, [2], [2], [[0.0, 0, 1, 1]], [[-1.0, -2.0], [1.0, 1.0]])
road = bspy.Spline.join([road1, road2, road3])
road.metadata = dict(Name = 'road')

controlPgon = bspy.Spline(1, 2, [2], [4], [[0.0, 0.0, 0.3, 0.7, 1, 1]],
                          [[0.0, 0.3, 0.7, 1.0], [0.0, 0.5, 0.5, 0.0]])
controlPgon.metadata = dict(Name = 'control_polygon')
cubic = bspy.Spline(1, 2, [4], [4], [[0.0, 0.0, 0, 0, 1, 1, 1, 1]],
                          [[0.0, 0.3, 0.7, 1.0], [0.0, 0.5, 0.5, 0.0]])
cubic.metadata = dict(Name = 'cubic_curve')

if __name__=='__main__':
    viewer = bspy.Viewer()
    viewer.show(wingLoft)
    viewer.show([[1.0, 0, 0], [0.0, 0, 1]] @ rootFoil)
    viewer.show(tipFoil)
    viewer.show(road)
    viewer.show(controlPgon)
    viewer.show(cubic)
    viewer.mainloop()
