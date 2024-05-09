"""
BSpy is a python library for manipulating and rendering non-uniform B-splines.

Available subpackages
---------------------
`bspy.solid` : Provides the `Solid` and `Boundary` classes that model solids.

`bspy.manifold` : Provides the `Manifold` base class for manifolds.

`bspy.hyperplane` : Provides the `Hyperplane` subclass of `Manifold` that models hyperplanes.

`bspy.spline` : Provides the `Spline` subclass of `Manifold` that models, represents, and processes 
    piecewise polynomial tensor product functions (spline functions) as linear combinations of B-splines.

`bspy.spline_block` : Provides the `SplineBlock` class that represents and processes an array-like collection of splines.

`bspy.splineOpenGLFrame` : Provides the `SplineOpenGLFrame` class, a tkinter `OpenGLFrame` with shaders to display splines.

`bspy.viewer` : Provides the `Viewer` tkinter app (`tkinter.Tk`) that hosts a `SplineOpenGLFrame`, a listbox full of 
    splines, and a set of controls to adjust and view the selected splines. It also provides the `Graphics` engine that creates 
    an associated `Viewer`, allowing you to script splines and display them in the viewer.
"""
from bspy.solid import Solid, Boundary
from bspy.manifold import Manifold
from bspy.hyperplane import Hyperplane
from bspy.spline import Spline
from bspy.spline_block import SplineBlock
from bspy.splineOpenGLFrame import SplineOpenGLFrame
from bspy.viewer import Viewer, Graphics