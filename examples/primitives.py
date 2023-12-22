import math
import numpy as np
import bspy
#from bspy import DrawableSpline
#from bspy import bspyApp

mySphere = bspy.Spline.sphere(1.0, 1.0e-8)
mySphere.metadata = dict(Name = 'mySphere')
myTorus = bspy.Spline.torus(1.0, 2.0, 1.0e-8)
myTorus.metadata = dict(Name = 'myTorus')
crv1 = bspy.Spline(1, 3, [4], [4], [[0.0, 0, 0, 0, 1, 1, 1, 1]],
                   [[0.0, 0.3, 0.7, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.5, 0.0]])
crv2 = bspy.Spline.line([0.0, 1.0, 0.0], [1.0, 1.0, 1.0])
crv3 = bspy.Spline.line([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
crv4 = [[0, 0], [1, 0], [0, 1]] @ bspy.Spline.section([[0.0, 0.0, 90.0, -0.7], [1.0, 1.0, -10.0, -0.7]]) + [1, 0, 0]
patch4, rule1, rule2, topline, bottomline, bilinear = bspy.Spline.four_sided_patch(crv1, crv2, crv3, crv4)
patch4.metadata = dict(Name = 'patch4')

if __name__=='__main__':
    app = bspy.bspyApp()
    app.show(mySphere)
    app.show(myTorus)
    app.show(patch4)
    app.show(rule1)
    app.show(rule2)
    app.show(topline)
    app.show(bottomline)
    app.show(bilinear)
    app.mainloop()
