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
        1D splines into 3D curves, 2D splines into surfaces (y-axis hold amplitude), and 3D splines into solids.
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
        else:
            Spline.__init__(self, *args, **kwargs)
            if self.nInd > 3: raise ValueError("nInd must be no more than 3")
            if self.nDep < 3: raise ValueError("nDep must be at least 3")
            if self.coefs.dtype != np.float32: raise ValueError("Must use 32-bit floats")
            for knotArray in self.knots:
                if knotArray.dtype != np.float32: raise ValueError("Must use 32-bit floats")
            floatCount = 0
            coefficientCount = 1
            for i in range(self.nInd):
                if self.order[i] > self.maxOrder: raise ValueError(f"order larger than {self.maxOrder}")
                floatCount += 2 + self.order[i] + self.nCoef[i]
                coefficientCount *= self.nCoef[i]
            if not(floatCount + self.nDep * coefficientCount <= self._maxFloats): raise ValueError("Spline to large to draw")
            self.metadata["fillColor"] = np.array((0.0, 1.0, 0.0, 1.0), np.float32)
            self.metadata["lineColor"] = np.array((0.0, 0.0, 0.0, 1.0) if self.nInd > 1 else (1.0, 1.0, 1.0, 1.0), np.float32)
            self.metadata["options"] = self.SHADED | self.BOUNDARY

    def __str__(self):
        return self.metadata.get("Name", "[{0}, {1}]".format(self.coefs[0], self.coefs[1]))

    @staticmethod
    def make_drawable(spline):
        """
        Convert a `Spline` into a `DrawableSpline` that can be drawn in a `SplineOpenGLFrame`. Converts 
        1D splines into 3D curves, 2D splines into surfaces (y-axis hold amplitude), and 3D splines into solids.

        Spline surfaces and solids with more than 3 dependent variables will have their added dimensions rendered 
        as colors (up to 6 dependent variables are supported).

        The drawable spline will share the original spline's metadata (metadata changes are shared).
        """
        if isinstance(spline, DrawableSpline):
            return spline
        if not(isinstance(spline, Spline)): raise ValueError("Invalid spline")
        if spline.nInd > 3: raise ValueError("Spline must have no more than 3 independent variables")
        if spline.nDep > 6: raise ValueError("Spline must have no more than 6 dependent variables")

        nDep = 3
        if spline.nInd >= 2 and spline.nDep > 3:
            nDep = 4 if spline.nDep == 4 else 6 # No nDep of 5
        
        knotList = [knots.astype(np.float32, copy=False) for knots in spline.knots]
        coefs = np.zeros((nDep, *spline.nCoef), np.float32)
        if spline.nInd == 1:
            if spline.nDep == 1:
                coefs[0] = np.linspace(knotList[0][spline.order[0] - 1], knotList[0][spline.nCoef[0]], spline.nCoef[0], dtype=np.float32)
                coefs[1] = spline.coefs[0]
            else:
                coefs[:min(spline.nDep, 3)] = spline.coefs[:min(spline.nDep, 3)]
        elif spline.nInd == 2:
            if spline.nDep == 1:
                xValues = np.linspace(knotList[0][spline.order[0] - 1], knotList[0][spline.nCoef[0]], spline.nCoef[0], dtype=np.float32)[:]
                zValues = np.linspace(knotList[1][spline.order[1] - 1], knotList[1][spline.nCoef[1]], spline.nCoef[1], dtype=np.float32)[:]
                xMesh, zMesh = np.meshgrid(xValues, zValues)
                coefs[0] = xMesh.T
                coefs[1] = spline.coefs[0]
                coefs[2] = zMesh.T
            else:
                coefs[:spline.nDep] = spline.coefs
                # For dimensions above three, rescale dependent variables to [0, 1].
                for i in range(3, spline.nDep):
                    minCoef = coefs[i].min()
                    rangeCoef = coefs[i].max() - minCoef
                    if rangeCoef > 1.0e-8:
                        coefs[i] = (coefs[i] - minCoef) / rangeCoef
                    else:
                        coefs[i] = 1.0
        elif spline.nInd == 3:
            if spline.nDep == 1:
                xValues = np.linspace(knotList[0][spline.order[0] - 1], knotList[0][spline.nCoef[0]], spline.nCoef[0], dtype=np.float32)[:]
                zValues = np.linspace(knotList[1][spline.order[1] - 1], knotList[1][spline.nCoef[1]], spline.nCoef[1], dtype=np.float32)[:]
                wValues = np.linspace(knotList[2][spline.order[1] - 1], knotList[2][spline.nCoef[1]], spline.nCoef[2], dtype=np.float32)[:]
                xMesh, zMesh, wMesh = np.meshgrid(xValues, zValues, wValues)
                coefs[0] = xMesh.T
                coefs[1] = spline.coefs[0]
                coefs[2] = zMesh.T
                coefs[3] = wMesh.T
            else:
                coefs[:spline.nDep] = spline.coefs
                # For dimensions above three, rescale dependent variables to [0, 1].
                for i in range(3, spline.nDep):
                    minCoef = coefs[i].min()
                    rangeCoef = coefs[i].max() - minCoef
                    if rangeCoef > 1.0e-8:
                        coefs[i] = (coefs[i] - minCoef) / rangeCoef
                    else:
                        coefs[i] = 1.0
        else:
            raise ValueError("Can't convert to drawable spline.")
        
        drawable = DrawableSpline(spline.nInd, nDep, spline.order, spline.nCoef, knotList, coefs, spline.accuracy)
        drawable.metadata = spline.metadata # Make the original spline share its metadata with its drawable spline
        if not "fillColor" in drawable.metadata:
            drawable.metadata["fillColor"] = np.array((0.0, 1.0, 0.0, 1.0), np.float32)
        if not "lineColor" in drawable.metadata:
            drawable.metadata["lineColor"] = np.array((0.0, 0.0, 0.0, 1.0) if drawable.nInd > 1 else (1.0, 1.0, 1.0, 1.0), np.float32)
        if not "options" in drawable.metadata:
            drawable.metadata["options"] = drawable.SHADED | drawable.BOUNDARY
        return drawable

    def _DrawPoints(self, frame, drawCoefficients):
        """
        Draw spline points for an order 1 spline within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        glColor4fv(self.get_line_color())
        glBegin(GL_POINTS)
        for point in drawCoefficients:
            glVertex3fv(point)
        glEnd()

    def _DrawCurve(self, frame, drawCoefficients):
        """
        Draw a spline curve (nInd == 1) within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        if self.get_options() & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            glBegin(GL_LINE_STRIP)
            for point in drawCoefficients:
                glVertex3f(point[0], point[1], point[2])
            glEnd()

        program = frame.curveProgram
        glUseProgram(program.curveProgram)
        glUniform4fv(program.uCurveLineColor, 1, self.get_line_color())
        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 2
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((self.order[0], self.nCoef[0]), np.float32))
        offset += size
        size = 4 * len(self.knots[0])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[0])
        offset += size
        size = 3 * 4 * self.nCoef[0]
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)
        glEnableVertexAttribArray(program.aCurveParameters)
        if frame.tessellationEnabled:
            glPatchParameteri(GL_PATCH_VERTICES, 1)
            glDrawArraysInstanced(GL_PATCHES, 0, 1, self.nCoef[0] - self.order[0] + 1)
        else:
            glDrawArraysInstanced(GL_POINTS, 0, 1, self.nCoef[0] - self.order[0] + 1)
            glFlush() # Old graphics card
        glDisableVertexAttribArray(program.aCurveParameters)
        glUseProgram(0)

    @staticmethod
    def _ConvertRGBToHSV(r, g, b, a):
        # Taken from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
        K = 0.0
        if g < b:
            tmp = g
            g = b
            b = tmp
            K = -1.0
        if r < g:
            tmp = r
            r = g
            g = tmp
            K = -2.0 / 6.0 - K
        chroma = r - min(g, b)
        return np.array((abs(K + (g - b) / (6.0 * chroma + 1e-20)), chroma / (r + 1e-20), r, a), np.float32)
    
    def _DrawSurface(self, frame, drawCoefficients):
        """
        Draw a spline surface (nInd == 2) within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        if self.get_options() & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            for pointList in drawCoefficients:
                glBegin(GL_LINE_STRIP)
                for point in pointList:
                    glVertex3f(point[0], point[1], point[2])
                glEnd()

        fillColor = self.get_fill_color()
        if self.nDep == 3:
            program = frame.surface3Program
        elif self.nDep == 4:
            program = frame.surface4Program
            fillColor = self._ConvertRGBToHSV(fillColor[0], fillColor[1], fillColor[2], fillColor[3])
        elif self.nDep == 6:
            program = frame.surface6Program
        else:
            raise ValueError("Can't draw surface.")
        
        glUseProgram(program.surfaceProgram)
        glUniform4fv(program.uSurfaceFillColor, 1, fillColor)
        glUniform4fv(program.uSurfaceLineColor, 1, self.get_line_color())
        glUniform1i(program.uSurfaceOptions, self.get_options())
        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 4
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((self.order[0], self.order[1], self.nCoef[0], self.nCoef[1]), np.float32))
        offset += size
        size = 4 * len(self.knots[0])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[0])
        offset += size
        size = 4 * len(self.knots[1])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[1])
        offset += size
        size = self.nDep * 4 * self.nCoef[0] * self.nCoef[1]
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)
        glEnableVertexAttribArray(program.aSurfaceParameters)
        if frame.tessellationEnabled:
            glPatchParameteri(GL_PATCH_VERTICES, 1)
            glDrawArraysInstanced(GL_PATCHES, 0, 1, (self.nCoef[0] - self.order[0] + 1) * (self.nCoef[1] - self.order[1] + 1))
        else:
            glDrawArraysInstanced(GL_POINTS, 0, 1, (self.nCoef[0] - self.order[0] + 1) * (self.nCoef[1] - self.order[1] + 1))
            glFlush() # Old graphics card
        glDisableVertexAttribArray(program.aSurfaceParameters)
        glUseProgram(0)
    
    def _DrawSolid(self, frame, drawCoefficients):
        """
        Draw a spline solid (nInd == 3) within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        if self.get_options() & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            for pointSet in drawCoefficients:
                for pointList in pointSet:
                    glBegin(GL_LINE_STRIP)
                    for point in pointList:
                        glVertex3f(point[0], point[1], point[2])
                    glEnd()

        fillColor = self.get_fill_color().copy()
        lineColor = self.get_line_color().copy()
        if self.nDep == 3:
            program = frame.surface3Program
        elif self.nDep == 4:
            program = frame.surface4Program
            fillColor = self._ConvertRGBToHSV(fillColor[0], fillColor[1], fillColor[2], fillColor[3])
        elif self.nDep == 6:
            program = frame.surface6Program
        else:
            raise ValueError("Can't draw surface.")
        fillColor[3] *= 0.5
        lineColor[3] *= 0.5

        def _DrawBoundarySurface(axis, index):
            fullSlice = slice(None)
            if axis == 0:
                i1 = 1
                i2 = 2
                coefSlice = (index, fullSlice, fullSlice, fullSlice)
            elif axis == 1:
                i1 = 0
                i2 = 2
                coefSlice = (fullSlice, index, fullSlice, fullSlice)
            else:
                i1 = 0
                i2 = 1
                coefSlice = (fullSlice, fullSlice, index, fullSlice)

            glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
            offset = 0
            size = 4 * 4
            glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((self.order[i1], self.order[i2], self.nCoef[i1], self.nCoef[i2]), np.float32))
            offset += size
            size = 4 * len(self.knots[i1])
            glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[i1])
            offset += size
            size = 4 * len(self.knots[i2])
            glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[i2])
            offset += size
            size = self.nDep * 4 * self.nCoef[i1] * self.nCoef[i2]
            glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients[coefSlice])
            glEnableVertexAttribArray(program.aSurfaceParameters)
            if frame.tessellationEnabled:
                glPatchParameteri(GL_PATCH_VERTICES, 1)
                glDrawArraysInstanced(GL_PATCHES, 0, 1, (self.nCoef[i1] - self.order[i1] + 1) * (self.nCoef[i2] - self.order[i2] + 1))
            else:
                glDrawArraysInstanced(GL_POINTS, 0, 1, (self.nCoef[i1] - self.order[i1] + 1) * (self.nCoef[i2] - self.order[i2] + 1))
                glFlush() # Old graphics card
            glDisableVertexAttribArray(program.aSurfaceParameters)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glDisable( GL_DEPTH_TEST )
        glUseProgram(program.surfaceProgram)
        glUniform4fv(program.uSurfaceFillColor, 1, fillColor)
        glUniform4fv(program.uSurfaceLineColor, 1, lineColor)
        glUniform1i(program.uSurfaceOptions, self.get_options())

        _DrawBoundarySurface(0, 0)
        _DrawBoundarySurface(0, -1)
        _DrawBoundarySurface(1, 0)
        _DrawBoundarySurface(1, -1)
        _DrawBoundarySurface(2, 0)
        _DrawBoundarySurface(2, -1)

        glUseProgram(0)
        glDisable( GL_BLEND )
        glEnable( GL_DEPTH_TEST )

    def _Draw(self, frame, transform):
        """
        Draw a spline  within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        coefs = self.coefs.T
        drawCoefficients = np.empty(coefs.shape, np.float32)
        drawCoefficients[..., :3] = coefs[..., :3] @ transform[:3,:3] + transform[3,:3]
        drawCoefficients[..., 3:] = coefs[..., 3:]
        if self.order[0] == 1:
            self._DrawPoints(frame, drawCoefficients)
        elif self.nInd == 1:
            self._DrawCurve(frame, drawCoefficients)
        elif self.nInd == 2:
            self._DrawSurface(frame, drawCoefficients)
        elif self.nInd == 3:
            self._DrawSolid(frame, drawCoefficients)
    
    def get_fill_color(self):
        """
        Gets the fill color of the spline (only useful for nInd >= 2).

        Returns
        -------
        fillColor : `numpy.array`
            Array of four floats (r, g, b, a) in the range [0, 1].
        """
        return self.metadata["fillColor"]

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
        self.metadata["fillColor"] = _set_color(r, g, b, a)
    
    def get_line_color(self):
        """
        Gets the line color of the spline.

        Returns
        -------
        lineColor : `numpy.array`
            Array of four floats (r, g, b, a) in the range [0, 1].
        """
        return self.metadata["lineColor"]

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
        self.metadata["lineColor"] = _set_color(r, g, b, a)
    
    def get_options(self):
        """
        Gets the draw options for the spline.

        Returns
        -------
        options : `int` bitwise or (`|`) of zero or more of the following values:
            * `DrawableSpline.HULL` Draw the convex hull of the spline (the coefficients). Off by default.
            * `DrawableSpline.SHADED` Draw the spline shaded (only useful for nInd >= 2). On by default.
            * `DrawableSpline.BOUNDARY` Draw the boundary of the spline in the line color (only useful for nInd >= 2). On by default.
            * `DrawableSpline.ISOPARMS` Draw the lines of constant knot values of the spline in the line color (only useful for nInd >= 2). Off by default.
        """
        return self.metadata["options"]

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
        self.metadata["options"] = options
