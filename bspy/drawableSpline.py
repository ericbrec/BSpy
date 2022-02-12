import numpy as np
from OpenGL.GL import *
from bspy import Spline

class DrawableSpline(Spline):
    """
    A `Spline` that can be drawn within a `SplineOpenGLFrame`.
    """

    maxOrder = 9
    """Maximum order for drawable splines."""
    maxCoefficients = 120
    """Maximum number of coefficients for drawable splines."""
    maxKnots = maxCoefficients + maxOrder
    """Maximum number of knots for drawable splines."""
    maxFloats = 4 + 2 * maxKnots + 4 * maxCoefficients * maxCoefficients
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
        Spline.__init__(self, *args, **kwargs)

        floatCount = 0
        coefficientCount = 1
        for i in range(self.nInd):
            assert self.order[i] <= self.maxOrder
            floatCount += 2 + self.order[i] + self.nCoef[i]
            coefficientCount *= self.nCoef[i]
        assert self.nDep == 4 # Coefficients are all 4-vectors (homogeneous coordinates)
        assert floatCount + 4 * coefficientCount <= self.maxFloats
        for knotArray in self.knots:
            assert knotArray.dtype == np.float32
        assert self.coefs.dtype == np.float32

        self.fillColor = np.array((0.0, 1.0, 0.0, 1.0), np.float32)
        self.lineColor = np.array((0.0, 0.0, 0.0, 1.0) if self.nInd > 1 else (1.0, 1.0, 1.0, 1.0), np.float32)
        self.options = self.SHADED | self.BOUNDARY

    def __str__(self):
        return self.metadata.get("Name", "[{0}, {1}]".format(self.coefs[0], self.coefs[1]))
    
    def DrawCurve(self, frame, drawCoefficients):
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

    def DrawSurface(self, frame, drawCoefficients):
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

    def Draw(self, frame, transform):
        """
        Draw a spline  within a `SplineOpenGLFrame`. The frame will call this method for you.
        """
        drawCoefficients = self.coefs.T @ transform
        if len(self.order) == 1:
            self.DrawCurve(frame, drawCoefficients)
        elif len(self.order) == 2:
            self.DrawSurface(frame, drawCoefficients)
    
    def SetFillColor(self, r, g=None, b=None, a=None):
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
        
        self.fillColor = np.array((red, green, blue, alpha))

    def SetLineColor(self, r, g=None, b=None, a=None):
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
        
        self.lineColor = np.array((red, green, blue, alpha))

    def SetOptions(self, options):
        """
        Set the draw options for the spline.

        Parameters
        ----------
        options : `int` bitwise or (`|`) of zero or more of the following values:
            `DrawableSpline.HULL` Draw the convex hull of the spline (the coefficients). Off by default.
            `DrawableSpline.SHADED` Draw the spline shaded (only useful for nInd >= 2). On by default.
            `DrawableSpline.BOUNDARY` Draw the boundary of the spline in the line color (only useful for nInd >= 2). On by default.
            `DrawableSpline.ISOPARMS` Draw the lines of constant knot values of the spline in the line color (only useful for nInd >= 2). Off by default.
        """
        self.options = options