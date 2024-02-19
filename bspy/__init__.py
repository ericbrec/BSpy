"""
bspy is a python library for manipulating and rendering non-uniform B-splines.

Available subpackages
---------------------
`bspy.spline` : A class to model, represent, and process piecewise polynomial tensor product
    functions (spline functions) as linear combinations of B-splines.

`bspy.splineOpenGLFrame` : A tkinter `OpenGLFrame` with shaders to display a `Spline` list.

`bspy.viewer` : A tkinter app (`tkinter.Tk`) that hosts a `SplineOpenGLFrame`, a listbox full of 
    splines, and a set of controls to adjust and view the selected splines.
"""
from bspy.spline import Spline
from bspy.splineOpenGLFrame import SplineOpenGLFrame
from bspy.viewer import Viewer, Graphics