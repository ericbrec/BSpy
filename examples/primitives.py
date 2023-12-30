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
patch40 = bspy.Spline.four_sided_patch(crv1, crv2, crv3, crv4, 0.0)
patch405 = bspy.Spline.four_sided_patch(crv1, crv2, crv3, crv4, 0.5)
patch41 = bspy.Spline.four_sided_patch(crv1, crv2, crv3, crv4, 1.0)
patch40 = bspy.DrawableSpline.make_drawable(patch40)
patch40.set_fill_color(0.0, 0.8, 0.2, 0.6)
patch405 = bspy.DrawableSpline.make_drawable(patch405)
patch405.set_fill_color(0.0, 0.6, 0.4, 0.8)
patch41 = bspy.DrawableSpline.make_drawable(patch41)
patch41.set_fill_color(0.0, 0.4, 0.6, 1.0)
bottomSurf = bspy.Spline(2, 3, [4, 4], [4, 4], 2 * [[0.0, 0, 0, 0, 1, 1, 1, 1]],
                                         [4 * [0.0, 0.3, 0.7, 1],
                                          [0.0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3,
                                           0.7, 0.7, 0.7, 0.7, 1, 1, 1, 1],
                                          [0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0,
                                           0.2, 0.3, 0.7, 0.2, 0.0, 0.6, 0.0, 0.0]])
topSurf = bottomSurf + [0, 0, 1]
mySolid = bspy.Spline.ruled_surface(bottomSurf, topSurf)

if __name__=='__main__':
    app = bspy.bspyApp()
    app.list(mySphere, 'mySphere')
    app.list(myTorus, 'myTorus')
    app.draw(patch40, 'patch40')
    app.draw(patch405, 'patch405')
    app.draw(patch41, 'patch41')
    app.list(mySolid, 'mySolid')
    app.mainloop()