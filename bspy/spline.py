import numpy as np
from OpenGL.GL import *

class Spline:

    maxOrder = 9
    maxCoefficients = 100
    maxKnots = maxCoefficients + maxOrder

    def __init__(self, order, knots, coefficients):
        for i in range(len(order)):
            assert len(knots[i]) == order[i] + coefficients.shape[i]
        assert coefficients.shape[len(order)] == 4 # Coefficients are all 4-vectors (homogeneous coordinates)
        for knotArray in knots:
            assert knotArray.dtype == np.float32
        assert coefficients.dtype == np.float32
        self.order = order
        self.knots = knots
        self.coefficients = coefficients

    def __str__(self):
        return "[{0}, {1}]".format(self.coefficients[0], self.coefficients[1])
    
    def DrawCurve(self, frame, drawCoefficients):
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINE_STRIP)
        for point in drawCoefficients:
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        glVertex3f(-0.01, -0.01, 0.0)
        glVertex3f(-0.01, 0.01 + 5.0/6.0, 0.0)
        glVertex3f(1.01, 0.01 + 5.0/6.0, 0.0)
        glVertex3f(1.01, -0.01, 0.0)
        glEnd()

        glUseProgram(frame.curveProgram)
        glUniform3f(frame.uCurveSplineColor, 0.0, 1.0, 0.0)
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
        #glDrawArraysInstanced(GL_POINTS, 0, 1, drawCoefficients.shape[0] - self.order[0] + 1)
        glPatchParameteri(GL_PATCH_VERTICES, 1)
        glDrawArraysInstanced(GL_PATCHES, 0, 1, 1)
        glDisableVertexAttribArray(frame.aCurveParameters)
        glUseProgram(0)

    def DrawSurface(self, frame, drawCoefficients):
        glColor3f(0.0, 0.0, 1.0)
        for pointList in drawCoefficients:
            glBegin(GL_LINE_STRIP)
            for point in pointList:
                glVertex3f(point[0], point[1], point[2])
            glEnd()

        glUseProgram(frame.surfaceProgram)
        glUniform3f(frame.uSurfaceSplineColor, 0.0, 1.0, 0.0)
        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 4
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((self.order[0], self.order[1], drawCoefficients.shape[0], drawCoefficients.shape[1]), np.float32))
        offset += size
        size = 4 * len(self.knots[0])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[0])
        offset += size
        size = 4 * len(self.knots[1])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots[1])
        offset += size
        size = 4 * 4 * drawCoefficients.shape[0] * drawCoefficients.shape[1]
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)
        glEnableVertexAttribArray(frame.aSurfaceParameters)
        glDrawArraysInstanced(GL_POINTS, 0, 1, (drawCoefficients.shape[0] - self.order[0] + 1) * (drawCoefficients.shape[1] - self.order[1] + 1))
        glDisableVertexAttribArray(frame.aSurfaceParameters)
        glUseProgram(0)

    def Draw(self, frame, transform):
        drawCoefficients = self.coefficients @ transform
        if len(self.order) == 1:
            self.DrawCurve(frame, drawCoefficients)
        elif len(self.order) == 2:
            self.DrawSurface(frame, drawCoefficients)