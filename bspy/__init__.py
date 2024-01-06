"""
bspy is a python library for manipulating and rendering non-uniform B-splines.

Available subpackages
---------------------
`bspy.spline` : A class to model, represent, and process piecewise polynomial tensor product
    functions (spline functions) as linear combinations of B-splines.

`bspy.drawableSpline` : A `Spline` that can be drawn within a `SplineOpenGLFrame`.

`bspy.splineOpenGLFrame` : A tkinter `OpenGLFrame` with shaders to display a `DrawableSpline` list.

`bspy.bspyApp` : A tkinter app (`tkinter.Tk`) that hosts a `SplineOpenGLFrame`, a listbox full of 
    splines, and a set of controls to adjust and view the selected splines.
"""
from bspy.spline import Spline
from bspy.drawableSpline import DrawableSpline
from bspy.splineOpenGLFrame import SplineOpenGLFrame
from bspy.bspyApp import bspyApp, bspyGraphics