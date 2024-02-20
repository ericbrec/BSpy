"""
BSpy is a python library for manipulating and rendering non-uniform B-splines.

Available subpackages
---------------------
`bspy.spline` : Provides the `Spline` class that models, represents, and processes piecewise polynomial tensor product
    functions (spline functions) as linear combinations of B-splines.

`bspy.splineOpenGLFrame` : Provides the `SplineOpenGLFrame` class, a tkinter `OpenGLFrame` with shaders to display splines.

`bspy.viewer` : Provides the `Viewer` tkinter app (`tkinter.Tk`) that hosts a `SplineOpenGLFrame`, a listbox full of 
    splines, and a set of controls to adjust and view the selected splines. It also provides the `Graphics` engine that creates 
    an associated `Viewer`, allowing you to script splines and display them in the viewer."""
from bspy.spline import Spline
from bspy.splineOpenGLFrame import SplineOpenGLFrame
from bspy.viewer import Viewer, Graphics