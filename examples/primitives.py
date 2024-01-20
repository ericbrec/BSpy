import math
import numpy as np
import bspy
#from bspy import DrawableSpline
#from bspy import bspyApp

mySphere = bspy.Spline.sphere(1.0, 1.0e-8)
myTorus = bspy.Spline.torus(1.0, 2.0, 1.0e-8)
crv1 = bspy.Spline(1, 3, [4], [4], [[0.0, 0, 0, 0, 1, 1, 1, 1]],
                   [[0.0, 0.3, 0.7, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.5, 0.0]])
crv2 = bspy.Spline.line([0.0, 1.0, 0.0], [1.0, 1.0, 1.0])
crv3 = bspy.Spline.line([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
crv4 = [[0, 0], [1, 0], [0, 1]] @ bspy.Spline.section([[0.0, 0.0, 90.0, -0.7], [1.0, 1.0, -10.0, -0.7]]) + [1, 0, 0]
patch4m1 = bspy.Spline.four_sided_patch(crv1, crv2, crv3, crv4, -1.0)
patch42 = bspy.Spline.four_sided_patch(crv1, crv2, crv3, crv4, 2.0)
patch4 = bspy.Spline.ruled_surface(patch4m1, patch42)
bottomSurf = bspy.Spline(2, 3, [4, 4], [4, 4], 2 * [[0.0, 0, 0, 0, 1, 1, 1, 1]],
                                         [4 * [0.0, 0.3, 0.7, 1],
                                          [0.0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3,
                                           0.7, 0.7, 0.7, 0.7, 1, 1, 1, 1],
                                          [0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0,
                                           0.2, 0.3, 0.7, 0.2, 0.0, 0.6, 0.0, 0.0]])
topSurf = bottomSurf + [0, 0, 1]
mySolid = bspy.Spline.ruled_surface(bottomSurf, topSurf)
myCylinder = bspy.Spline.cylinder(1.0, 5.0)
rotCylinder = myCylinder.rotate([1.0, 1.0, 0.0], 45.0)
myCone = bspy.Spline.cone(1.0, 0.2, 2.0)
crv1 = bspy.Spline.line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
crv2 = bspy.Spline.section([[0.0, 0.0, 45.0, -0.5], [1.0, 0.0, -45.0, -0.5]])
crv2 = [[1, 0], [0, 0], [0, 1]] @ crv2 + [0.0, 0.5, 0.0]
crv3 = bspy.Spline.line([0.0, 1.0, 0.0], [1.0, 1.0, 0.0])
surfThroughCurves = bspy.Spline.least_squares([0.0, 0.5, 1.0], [crv1, crv2, crv3])

if __name__=='__main__':
    app = bspy.bspyApp()
    app.list(mySphere, 'mySphere')
    app.list(myTorus, 'myTorus')
    app.list(patch4, 'patch4')
    app.list(mySolid, 'mySolid')
    app.list(myCylinder, 'myCylinder')
    app.list(rotCylinder, 'rotCylinder')
    app.list(myCone, 'myCone')
    app.list(surfThroughCurves, 'surfThroughCurves')
    app.mainloop()