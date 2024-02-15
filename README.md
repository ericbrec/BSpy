# BSpy
Library for manipulating and rendering B-spline curves, surfaces, and multidimensional manifolds with non-uniform knots in each dimension.

The [Spline](https://ericbrec.github.io/BSpy/bspy/spline.html) class has a method to fit multidimensional data for 
scalar and vector functions of single and multiple variables. It also has methods to create points, lines, circular arcs, spheres, cones, cylinders, tori, ruled surfaces, surfaces of revolution, and four-sided patches. 
Other methods add, subtract, and multiply splines, as well as confine spline curves to a given range. 
There are methods to evaluate spline values, derivatives, integrals, normals, curvature, and the Jacobian, as well as methods that return spline representations of derivatives, normals, integrals, graphs, and convolutions. In addition, there are methods to manipulate the domain of splines, including trim, join, reparametrize, transpose, reverse, add and remove knots, elevate and extrapolate, and fold and unfold. There are methods to manipulate the range of splines, including dot product, cross product, translate, rotate, scale, and transform. Finally, there are methods to compute the zeros and contours of a spline and to intersect two splines.

The [SplineOpenGLFrame](https://ericbrec.github.io/BSpy/bspy/splineOpenGLFrame.html) class is an 
[OpenGLFrame](https://pypi.org/project/pyopengltk/) with custom shaders to render spline curves and surfaces. Only tested on Windows systems.

The [DrawableSpline](https://ericbrec.github.io/BSpy/bspy/drawableSpline.html) class converts a 
[Spline](https://ericbrec.github.io/BSpy/bspy/spline.html) to a curve, surface, or solid that can be drawn in a 
[SplineOpenGLFrame](https://ericbrec.github.io/BSpy/bspy/splineOpenGLFrame.html). Only 1D, 2D, and 3D splines can be converted. 
Spline surfaces and solids with more than 3 dependent variables will have their added dimensions rendered as colors 
(up to 6 dependent variables are supported).

The [bspyApp](https://ericbrec.github.io/BSpy/bspy/bspyApp.html) class is a 
[tkinter.Tk](https://docs.python.org/3/library/tkinter.html) app that hosts a 
[SplineOpenGLFrame](https://ericbrec.github.io/BSpy/bspy/splineOpenGLFrame.html), 
a listbox full of splines, and a set of controls to adjust and view the selected splines. Only tested on Windows systems.

The [bspyGraphics](https://ericbrec.github.io/BSpy/bspy/bspyApp.html#bspyGraphics) class is a graphics engine to display splines.
It launches a [bspyApp](https://ericbrec.github.io/BSpy/bspy/bspyApp.html) and issues commands to the app for use 
in [jupyter](https://jupyter.org/) notebooks and other scripting environments. Only tested on Windows systems.

![bspyApp rendering the Utah teapot](https://ericbrec.github.io/BSpy/bspyApp.png "bspyApp rendering the Utah teapot")

The full documentation for BSpy can be found [here](https://ericbrec.github.io/BSpy), its GitHub project can be found 
[here](https://github.com/ericbrec/BSpy), a test suite can be found [here](https://github.com/ericbrec/BSpy/tree/main/tests), and
a set of examples, including a jupyter notebook, can be found [here](https://github.com/ericbrec/BSpy/tree/main/examples).

### Release 3.0 breaking changes
* Removed accuracy as a member of Spline
* Spline.common_basis is now a static method (see documentation for details)
* Spline.least_squares changed arguments (see documentation for details)
* Spline.load always returns a list of splines
