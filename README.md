# BSpy
Library for manipulating and rendering B-spline curves, surfaces, and multidimensional manifolds with non-uniform knots in each dimension.

The [Manifold](https://ericbrec.github.io/BSpy/bspy/manifold.html) abstract base class for [Hyperplane](https://ericbrec.github.io/BSpy/bspy/hyperplane.html) and [Spline](https://ericbrec.github.io/BSpy/bspy/spline.html).

The [Spline](https://ericbrec.github.io/BSpy/bspy/spline.html) class has a method to fit multidimensional data for scalar and vector functions of single and multiple variables. It also can fit splines to functions, to solutions for ordinary differential equations (ODEs), and to geodesics. 
Spline has methods to create points, lines, circular arcs, spheres, cones, cylinders, tori, ruled surfaces, surfaces of revolution, and four-sided patches. 
Other methods add, subtract, and multiply splines, as well as confine spline curves to a given range. 
There are methods to evaluate spline values, derivatives, normals, integrals, continuity, curvature, and the Jacobian, as well as methods that return spline representations of derivatives, normals, integrals, graphs, and convolutions. 
In addition, there are methods to manipulate the domain of splines, including trim, join, split, reparametrize, transpose, reverse, add and remove knots, elevate and extrapolate, and fold and unfold. 
There are methods to manipulate the range of splines, including dot product, cross product, translate, rotate, scale, and transform. 
Finally, there are methods to compute the zeros and contours of a spline and to intersect two splines. 
Splines can be saved and loaded in json format.

The [Hyperplane](https://ericbrec.github.io/BSpy/bspy/hyperplane.html) class has methods to create individual hyperplanes in any dimension, along with axis-aligned hyperplanes and hypercubes.

The [Solid](https://ericbrec.github.io/BSpy/bspy/solid.html) class has methods to construct n-dimensional solids from trimmed [Manifold](https://ericbrec.github.io/BSpy/bspy/manifold.html) boundaries. Each solid consists of a list of boundaries and a Boolean value that indicates if the solid contains infinity. Each [Boundary](https://ericbrec.github.io/BSpy/bspy/solid.html) consists of a manifold (currently a [Hyperplane](https://ericbrec.github.io/BSpy/bspy/hyperplane.html) or [Spline](https://ericbrec.github.io/BSpy/bspy/spline.html)) and a domain solid that trims the manifold. Solids have methods to form the intersection, union, difference, and complement of solids. There are methods to compute point containment, winding numbers, surface integrals, and volume integrals. There are also methods to translate, transform, and slice solids. Solids can be saved and loaded in json format.

The [SplineBlock](https://ericbrec.github.io/BSpy/bspy/spline_block.html) class has methods to process an array-like collection of splines that represent a system of equations. There are highly-optimized methods to compute the contours and zeros of a spline block, as well as a variety of methods to manipulate and evaluate a spline block and its derivatives.

The [BSpyConvert](https://pypi.org/project/BSpyConvert/) package converts BSpy splines and solid models to and from [OpenCascade (OCCT)](https://dev.opencascade.org/) equivalents and a variety of geometry and CAD file formats, including STEP, IGES, and STL.

The [SplineOpenGLFrame](https://ericbrec.github.io/BSpy/bspy/splineOpenGLFrame.html) class is an 
[OpenGLFrame](https://pypi.org/project/pyopengltk/) with custom shaders to render spline curves and surfaces. Spline surfaces with more 
than 3 dependent variables will have their added dimensions rendered as colors (up to 6 dependent variables are supported). Only tested on Windows systems.

The [Viewer](https://ericbrec.github.io/BSpy/bspy/viewer.html) class is a 
[tkinter.Tk](https://docs.python.org/3/library/tkinter.html) app that hosts a 
[SplineOpenGLFrame](https://ericbrec.github.io/BSpy/bspy/splineOpenGLFrame.html), 
a tree view full of solids and splines, and a set of controls to adjust and view the selected solids and splines. Only tested on Windows systems.

The [Graphics](https://ericbrec.github.io/BSpy/bspy/viewer.html#Graphics) class is a graphics engine to display splines.
It launches a [Viewer](https://ericbrec.github.io/BSpy/bspy/viewer.html) and issues commands to the viewer for use 
in [jupyter](https://jupyter.org/) notebooks and other scripting environments. Only tested on Windows systems.

![Viewer rendering the Utah teapot](https://ericbrec.github.io/BSpy/viewer.png "Viewer rendering the Utah teapot")

The full documentation for BSpy can be found [here](https://ericbrec.github.io/BSpy), its GitHub project can be found 
[here](https://github.com/ericbrec/BSpy), a test suite can be found [here](https://github.com/ericbrec/BSpy/tree/main/tests), and
a set of examples, including a jupyter notebook, can be found [here](https://github.com/ericbrec/BSpy/tree/main/examples).

### Release 3.0 breaking changes
* Removed accuracy as a member of Spline
* Spline.common_basis is now a static method (see documentation for details)
* Spline.least_squares changed arguments (see documentation for details)
* Spline.load always returns a list of splines

### Release 4.0 breaking changes
* Removed Spline blossom method
* Removed DrawableSpline class
* Changed bspyApp class name to Viewer
* Changed Viewer listbox to use extended selection (shift and ctrl keys)
* Changed bspyGraphics class name to Graphics
* Moved DrawableSpine methods for adjusting spline appearance to Viewer (see documentation for details)
* Spline.bspline_values changed arguments (see documentation for details)
* Spline.intersect changed return values (see documentation for details)
