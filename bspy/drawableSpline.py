import numpy as np
from OpenGL.GL import *
from bspy import Spline

class DrawableSpline(Spline):

    maxOrder = 9
    maxCoefficients = 120
    maxKnots = maxCoefficients + maxOrder
    maxFloats = 4 + 2 * maxKnots + 4 * maxCoefficients * maxCoefficients

    HULL = (1 << 0)
    SHADED = (1 << 1)
    BOUNDARY = (1 << 2)
    ISOPARMS = (1 << 3)

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
        self.lineColor = np.array((0.0, 0.0, 0.0, 1.0), np.float32)
        self.options = self.SHADED | self.BOUNDARY

    def __str__(self):
        return self.metadata.get("Name", "[{0}, {1}]".format(self.coefs[0], self.coefs[1]))
    
    def DrawCurve(self, frame, drawCoefficients):
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
        drawCoefficients = self.coefs.T @ transform
        if len(self.order) == 1:
            self.DrawCurve(frame, drawCoefficients)
        elif len(self.order) == 2:
            self.DrawSurface(frame, drawCoefficients)