import numpy as np
from OpenGL.GL import *
from bspy import Spline

def _set_color(r, g=None, b=None, a=None):
    """
    Return an array with the specified color.

    Parameters
    ----------
    r : `float`, `int` or array-like of floats or ints
        The red value [0, 1] as a float, [0, 255] as an int, or the rgb or rgba value as floats or ints (default).
    
    g: `float` or `int`
        The green value [0, 1] as a float or [0, 255] as an int.
    
    b: `float` or `int`
        The blue value [0, 1] as a float or [0, 255] as an int.
    
    a: `float`, `int`, or None
        The alpha value [0, 1] as a float or [0, 255] as an int. If `None` then alpha is set to 1.

    Returns
    -------
    color : `numpy.array`
        The specified color as an array of 4 float32 values between 0 and 1.
    """
    if isinstance(r, (int, np.integer)):
        red = float(r) / 255.0
        green = red
        blue = red
        alpha = 1.0
    elif np.isscalar(r):
        red = r
        green = red
        blue = red
        alpha = 1.0
    elif isinstance(r[0], (int, np.integer)):
        red = float(r[0]) / 255.0
        green = float(r[1]) / 255.0
        blue = float(r[2]) / 255.0
        alpha = float(r[3]) / 255.0 if len(r) >= 4 else 1.0
    else:
        red = r[0]
        green = r[1]
        blue = r[2]
        alpha = r[3] if len(r) >= 4 else 1.0

    if isinstance(g, (int, np.integer)):
        green = float(g) / 255.0
    elif np.isscalar(g):
        green = g

    if isinstance(b, (int, np.integer)):
        blue = float(b) / 255.0
    elif np.isscalar(b):
        blue = b

    if isinstance(a, (int, np.integer)):
        alpha = float(a) / 255.0
    elif np.isscalar(a):
        alpha = a
    
    return np.array((red, green, blue, alpha), np.float32)

class DrawableSpline(Spline):
    """
    A `Spline` that can be drawn within a `SplineOpenGLFrame`.

    Parameters
    ----------
    spline or nInd : `Spline` or `int`
        An existing spline that needs to become drawable (using `DrawableSpline.make_drawable`), or the number of independent variables of the new spline. 
        If it is an existing spline, the remaining parameters are optional and ignored.

    nDep : `int`
        The number of dependent variables of the spline
    
    order : `tuple`
        A tuple of length nInd where each integer entry represents the
        polynomial order of the function in that variable

    nCoef : `tuple`
        A tuple of length nInd where each integer entry represents the
        dimension (i.e. number of B-spline coefficients) of the function
        space in that variable

    knots : `list`
        A list of the lists of the knots of the spline in each independent variable

    coefs : array-like
        A list of the B-spline coefficients of the spline.
    
    accuracy : `float`
        Each spline function is presumed to be an approximation of something else. 
        The `accuracy` stores the infinity norm error of the difference between 
        the given spline function and that something else.

    metadata : `dict`
        A dictionary of ancillary data to store with the spline

    See Also
    --------
    `bspy.spline` : A class to model, represent, and process piecewise polynomial tensor product
        functions (spline functions) as linear combinations of B-splines.
    
    `make_drawable` : Convert a `Spline` into a `DrawableSpline` that can be drawn in a `SplineOpenGLFrame`. Converts 
        1D splines into 3D curves and 2D splines into surfaces (y-axis hold amplitude).
    """

    maxOrder = 9
    """Maximum order for drawable splines."""
    maxCoefficients = 120
    """Maximum number of coefficients for drawable splines."""
    maxKnots = maxCoefficients + maxOrder
    """Maximum number of knots for drawable splines."""
    _maxFloats = 4 + 2 * maxKnots + 4 * maxCoefficients * maxCoefficients
    """Maximum total number of floats for drawable splines."""

    HULL = (1 << 0)
    """Option to draw the convex hull of the spline (the coefficients). Off by default."""
    SHADED = (1 << 1)
    """Option to draw the spline shaded (only useful for nInd >= 2). On by default."""
    BOUNDARY = (1 << 2)
    """Option to draw the boundary of the spline in the line color (only useful for nInd >= 2). On by default."""
    ISOPARMS = (1 << 3)
    """Option to draw the lines of constant knot values of the spline in the line color (only useful for nInd >= 2). Off by default."""

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], Spline):
            spline = DrawableSpline.make_drawable(args[0])
            self.nInd = spline.nInd
            self.nDep = spline.nDep
            self.order = spline.order
            self.nCoef = spline.nCoef
            self.knots = spline.knots
            self.coefs = spline.coefs
            self.accuracy = spline.accuracy
            self.metadata = spline.metadata
            self.fillColor = spline.fillColor
            self.lineColor = spline.lineColor
            self.options = spline.options
        else:
            Spline.__init__(self, *args, **kwargs)

            floatCount = 0
            coefficientCount = 1
            for i in range(self.nInd):
                assert self.order[i] <= self.maxOrder
                floatCount += 2 + self.order[i] + self.nCoef[i]
                coefficientCount *= self.nCoef[i]
            assert self.nDep == 4 # Coefficients are all 4-vectors (homogeneous coordinates)
            assert floatCount + 4 * coefficientCount <= self._maxFloats
            for knotArray in self.knots:
                assert knotArray.dtype == np.float32
            assert self.coefs.dtype == np.float32

            self.fillColor = np.array((0.0, 1.0, 0.0, 1.0), np.float32)
            self.lineColor = np.array((0.0, 0.0, 0.0, 1.0) if self.nInd > 1 else (1.0, 1.0, 1.0, 1.0), np.float32)
            self.options = self.SHADED | self.BOUNDARY

    def __str__(self):
        return self.metadata.get("Name", "[{0}, {1}]".format(self.coefs[0], self.coefs[1]))

    @staticmethod
    def make_drawable(spline):
        """
        Convert a `Spline` into a `DrawableSpline` that can be drawn in a `SplineOpenGLFrame`. Converts 
        1D splines into 3D curves and 2D splines into surfaces (y-axis hold amplitude).
        """
        if isinstance(spline, DrawableSpline):
            return spline
        assert isinstance(spline, Spline)
        
        knotList = [knots.astype(np.float32, copy=False) for knots in spline.knots]
        coefs = np.zeros((4, *spline.nCoef), np.float32)
        coefs[3,...] = 1.0
        if spline.nInd == 1 and spline.nDep == 1:
            coefs[0] = np.linspace(knotList[0][spline.order[0] - 1], knotList[0][spline.nCoef[0]], spline.nCoef[0], dtype=np.float32)
            coefs[1] = spline.coefs[0]
        elif spline.nInd == 1 and spline.nDep <= 3:
            coefs[0:spline.nDep] = spline.coefs
        elif spline.nInd == 2 and spline.nDep == 1:
            xValues = np.linspace(knotList[0][spline.order[0] - 1], knotList[0][spline.nCoef[0]], spline.nCoef[0], dtype=np.float32)[:]
            zValues = np.linspace(knotList[1][spline.order[1] - 1], knotList[1][spline.nCoef[1]], spline.nCoef[1], dtype=np.float32)[:]
            xMesh, zMesh = np.meshgrid(xValues, zValues)
            coefs[0] = xMesh.T
            coefs[1] = spline.coefs[0]
            coefs[2] = zMesh.T
        elif spline.nInd == 2 and spline.nDep == 3:
            coefs[0:3] = spline.coefs
        else:
            raise ValueError("Can't convert to drawable spline.")
        
        return DrawableSpline(spline.nInd, 4, spline.order, spline.nCoef, knotList, coefs, spline.accuracy, spline.metadata)

    def _DrawPoints(self, frame, drawCoefficients):
        """
        Draw spline points for an order 1 spline within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        glColor4fv(self.lineColor)
        glBegin(GL_POINTS)
        for point in drawCoefficients:
            glVertex4fv(point)
        glEnd()

    def _DrawCurve(self, frame, drawCoefficients):
        """
        Draw a spline curve (nInd == 1) within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        if self.options & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            glBegin(GL_LINE_STRIP)
            for point in drawCoefficients:
                glVertex3f(point[0], point[1], point[2])
            glEnd()

        glUseProgram(frame.curveProgram)
        glUniform4fv(frame.uCurveLineColor, 1, self.lineColor)
        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 2
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((self.order[0], drawCoefficients.shape[0]), np.float32))
        offset += size
        size = 4 * len(self.knots[0])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[0])
        offset += size
        size = 4 * 4 * len(drawCoefficients)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)
        glEnableVertexAttribArray(frame.aCurveParameters)
        glPatchParameteri(GL_PATCH_VERTICES, 1)
        glDrawArraysInstanced(GL_PATCHES, 0, 1, drawCoefficients.shape[0] - self.order[0] + 1)
        glDisableVertexAttribArray(frame.aCurveParameters)
        glUseProgram(0)

    def _DrawSurface(self, frame, drawCoefficients):
        """
        Draw a spline surface (nInd == 2) within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        if self.options & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            for pointList in drawCoefficients:
                glBegin(GL_LINE_STRIP)
                for point in pointList:
                    glVertex3f(point[0], point[1], point[2])
                glEnd()

        glUseProgram(frame.surfaceProgram)
        glUniform4fv(frame.uSurfaceFillColor, 1, self.fillColor)
        glUniform4fv(frame.uSurfaceLineColor, 1, self.lineColor)
        glUniform1i(frame.uSurfaceOptions, self.options)
        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 4
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((self.order[0], self.order[1], drawCoefficients.shape[1], drawCoefficients.shape[0]), np.float32))
        offset += size
        size = 4 * len(self.knots[0])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[0])
        offset += size
        size = 4 * len(self.knots[1])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[1])
        offset += size
        size = 4 * 4 * drawCoefficients.shape[1] * drawCoefficients.shape[0]
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)
        glEnableVertexAttribArray(frame.aSurfaceParameters)
        glPatchParameteri(GL_PATCH_VERTICES, 1)
        glDrawArraysInstanced(GL_PATCHES, 0, 1, (drawCoefficients.shape[1] - self.order[0] + 1) * (drawCoefficients.shape[0] - self.order[1] + 1))
        glDisableVertexAttribArray(frame.aSurfaceParameters)
        glUseProgram(0)

    def _Draw(self, frame, transform):
        """
        Draw a spline  within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        drawCoefficients = self.coefs.T @ transform
        if self.order[0] == 1:
            self._DrawPoints(frame, drawCoefficients)
        elif self.nInd == 1:
            self._DrawCurve(frame, drawCoefficients)
        elif self.nInd == 2:
            self._DrawSurface(frame, drawCoefficients)
    
    def set_fill_color(self, r, g=None, b=None, a=None):
        """
        Set the fill color of the spline (only useful for nInd >= 2).

        Parameters
        ----------
        r : `float`, `int` or array-like of floats or ints
            The red value [0, 1] as a float, [0, 255] as an int, or the rgb or rgba value as floats or ints (default).
        
        g: `float` or `int`
            The green value [0, 1] as a float or [0, 255] as an int.
        
        b: `float` or `int`
            The blue value [0, 1] as a float or [0, 255] as an int.
        
        a: `float`, `int`, or None
            The alpha value [0, 1] as a float or [0, 255] as an int. If `None` then alpha is set to 1.
        """
        self.fillColor = _set_color(r, g, b, a)

    def set_line_color(self, r, g=None, b=None, a=None):
        """
        Set the line color of the spline.

        Parameters
        ----------
        r : `float`, `int` or array-like of floats or ints
            The red value [0, 1] as a float, [0, 255] as an int, or the rgb or rgba value as floats or ints (default).
        
        g: `float` or `int`
            The green value [0, 1] as a float or [0, 255] as an int.
        
        b: `float` or `int`
            The blue value [0, 1] as a float or [0, 255] as an int.
        
        a: `float`, `int`, or None
            The alpha value [0, 1] as a float or [0, 255] as an int. If `None` then alpha is set to 1.
        """
        self.lineColor = _set_color(r, g, b, a)

    def set_options(self, options):
        """
        Set the draw options for the spline.

        Parameters
        ----------
        options : `int` bitwise or (`|`) of zero or more of the following values:
            * `DrawableSpline.HULL` Draw the convex hull of the spline (the coefficients). Off by default.
            * `DrawableSpline.SHADED` Draw the spline shaded (only useful for nInd >= 2). On by default.
            * `DrawableSpline.BOUNDARY` Draw the boundary of the spline in the line color (only useful for nInd >= 2). On by default.
            * `DrawableSpline.ISOPARMS` Draw the lines of constant knot values of the spline in the line color (only useful for nInd >= 2). Off by default.
        """
        self.options = options
