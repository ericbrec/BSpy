import math
import numpy as np
import bspy
#from bspy import DrawableSpline
#from bspy import bspyApp

mySphere = bspy.Spline.sphere(1.0, 1.0e-8)
mySphere.metadata = dict(Name = 'mySphere')

if __name__=='__main__':
    app = bspy.bspyApp()
    app.show(mySphere)
    app.mainloop()
