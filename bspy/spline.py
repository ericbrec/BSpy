import numpy as np
from OpenGL.GL import *

class Spline:

    maxOrder = 9
    maxCoefficients = 100
    maxKnots = maxCoefficients + maxOrder

    def __init__(self, order, knots, coefficients):
        for i in range(len(order)):
            assert len(knots[i]) == order[i] + coefficients.shape[i]
        self.order = order
        self.knots = knots
        self.coefficients = coefficients

    def __str__(self):
        return "[{0}, {1}]".format(self.coefficients[0], self.coefficients[1])

    def ComputeBasis(self, dimension, m, u):
        uOrder = self.order[dimension]
        uKnots = self.knots[dimension]
        uBasis = np.zeros(uOrder + 1, np.float32)
        duBasis = np.zeros(uOrder + 1, np.float32)
        du2Basis = np.zeros(uOrder + 1, np.float32)
        uBasis[uOrder-1] = 1.0
        for degree in range(1, uOrder):
            b = uOrder - degree - 1
            for n in range(m-degree, m+1):
                gap0 = uKnots[n+degree] - uKnots[n]
                gap1 = uKnots[n+degree+1] - uKnots[n+1]
                gap0 = 0.0 if gap0 < 1.0e-8 else 1.0 / gap0
                gap1 = 0.0 if gap1 < 1.0e-8 else 1.0 / gap1
                val0 = (u - uKnots[n]) * gap0
                val1 = (uKnots[n+degree+1] - u) * gap1
                if degree == uOrder - 2:
                    d0 = degree * gap0
                    d1 = -degree * gap1
                    du2Basis[b] = uBasis[b] * d0 + uBasis[b+1] * d1
                elif degree == uOrder - 1:
                    d0 = degree * gap0
                    d1 = -degree * gap1
                    duBasis[b] = uBasis[b] * d0 + uBasis[b+1] * d1
                    du2Basis[b] = du2Basis[b] * d0 + du2Basis[b+1] * d1
                uBasis[b] = uBasis[b] * val0 + uBasis[b+1] * val1
                b += 1
        
        return uBasis, duBasis, du2Basis
    
    def ComputeDelta(self, screenScale, point, dPoint, d2Point, delta):
        if screenScale[2] > 1.0:
            zScale = 1.0 / (screenScale[2] - point[2])
            zScale2 = zScale * zScale
            zScale3 = zScale2 * zScale
            d2Point[0] = screenScale[0] * (d2Point[0] * zScale - 2.0 * dPoint[0] * dPoint[2] * zScale2 + \
                point[0] * (2.0 * dPoint[2] * dPoint[2] * zScale3 - d2Point[2] * zScale2))
            d2Point[1] = screenScale[1] * (d2Point[1] * zScale - 2.0 * dPoint[1] * dPoint[2] * zScale2 + \
                point[1] * (2.0 * dPoint[2] * dPoint[2] * zScale3 - d2Point[2] * zScale2))
        else:
            d2Point[0] *= screenScale[0]
            d2Point[1] *= screenScale[1]
        
        sqrtLength = (d2Point[0]*d2Point[0] + d2Point[1]*d2Point[1])**0.25
        return delta if sqrtLength < 1.0e-8 else 1.0 / sqrtLength

    def DrawCurvePoints(self, screenScale, drawCoefficients, m, u, deltaU):
        uOrder = self.order[0]
        uBasis, duBasis, du2Basis = self.ComputeBasis(0, m, u)

        point = np.zeros(4, np.float32)
        dPoint = np.zeros(4, np.float32)
        d2Point = np.zeros(4, np.float32)
        b = 0
        for n in range(m+1-uOrder, m+1):
            point += uBasis[b] * drawCoefficients[n]
            dPoint += duBasis[b] * drawCoefficients[n]
            d2Point += du2Basis[b] * drawCoefficients[n]
            b += 1
        
        glVertex3f(point[0], point[1], point[2])
        glVertex3f(point[0] - 0.01*dPoint[1], point[1] + 0.01*dPoint[0], point[2])
        glVertex3f(point[0], point[1], point[2])
        
        deltaU = self.ComputeDelta(screenScale, point, dPoint, d2Point, deltaU)
        return deltaU
    
    def DrawCurve(self, frame, drawCoefficients):
        uOrder = self.order[0]
        uKnots = self.knots[0]

        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINE_STRIP)
        for point in drawCoefficients:
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        glColor3f(1.0, 0.0, 0.0)
        for m in range(0): #uOrder-1, len(uKnots)-uOrder):
            u = uKnots[m]
            deltaU = 0.5 * (uKnots[m+1] - u)
            vertices = 0
            glBegin(GL_LINE_STRIP)
            while u < uKnots[m+1] and vertices <= frame.maxVertices - 6: # Save room for the vertices at u and lastU
                deltaU = self.DrawCurvePoints(frame.screenScale, drawCoefficients, m, u, deltaU)
                u += deltaU
                vertices += 3
            self.DrawCurvePoints(frame.screenScale, drawCoefficients, m, uKnots[m+1], deltaU)
            vertices += 3
            glEnd()

        glUseProgram(frame.curveProgram)

        glUniform3f(frame.uCurveSplineColor, 0.0, 1.0, 0.0)

        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 2
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((uOrder, drawCoefficients.shape[0]), np.float32))
        offset += size
        size = 4 * len(uKnots)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, uKnots)
        offset += size
        size = 4 * 4 * len(drawCoefficients)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)

        glEnableVertexAttribArray(frame.aCurveParameters)
        glDrawArraysInstanced(GL_POINTS, 0, 1, drawCoefficients.shape[0] - uOrder + 1)
        glDisableVertexAttribArray(frame.aCurveParameters)

        glUseProgram(0)
    
    def DrawSurfacePoints(self, screenScale, drawCoefficients, uM, uBasis, duBasis, uBasisNext, duBasisNext, vM, v, deltaV):
        uOrder = self.order[0]
        vOrder = self.order[1]
        vBasis, dvBasis, dv2Basis = self.ComputeBasis(1, vM, v)

        point = np.zeros(4, np.float32)
        duPoint = np.zeros(4, np.float32)
        dvPoint = np.zeros(4, np.float32)
        dv2Point = np.zeros(4, np.float32)
        vB = 0
        for vN in range(vM+1-vOrder, vM+1):
            uB = 0
            for uN in range(uM+1-uOrder, uM+1):
                point += uBasis[uB] * vBasis[vB] * drawCoefficients[uN, vN]
                duPoint += duBasis[uB] * vBasis[vB] * drawCoefficients[uN, vN]
                dvPoint += uBasis[uB] * dvBasis[vB] * drawCoefficients[uN, vN]
                dv2Point += uBasis[uB] * dv2Basis[vB] * drawCoefficients[uN, vN]
                uB += 1
            vB += 1

        glVertex3f(point[0], point[1], point[2])

        deltaV = self.ComputeDelta(screenScale, point, dvPoint, dv2Point, deltaV)

        point = np.zeros(4, np.float32)
        duPoint = np.zeros(4, np.float32)
        dvPoint = np.zeros(4, np.float32)
        dv2Point = np.zeros(4, np.float32)
        vB = 0
        for vN in range(vM+1-vOrder, vM+1):
            uB = 0
            for uN in range(uM+1-uOrder, uM+1):
                point += uBasisNext[uB] * vBasis[vB] * drawCoefficients[uN, vN]
                duPoint += duBasisNext[uB] * vBasis[vB] * drawCoefficients[uN, vN]
                dvPoint += uBasisNext[uB] * dvBasis[vB] * drawCoefficients[uN, vN]
                dv2Point += uBasisNext[uB] * dv2Basis[vB] * drawCoefficients[uN, vN]
                uB += 1
            vB += 1

        glVertex3f(point[0], point[1], point[2])

        newDeltaV = self.ComputeDelta(screenScale, point, dvPoint, dv2Point, deltaV)
        deltaV = min(newDeltaV, deltaV)
        return deltaV

    def DrawSurface(self, frame, drawCoefficients):
        uOrder = self.order[0]
        uKnots = self.knots[0]
        vOrder = self.order[1]
        vKnots = self.knots[1]

        glColor3f(0.0, 0.0, 1.0)
        for pointList in drawCoefficients:
            glBegin(GL_LINE_STRIP)
            for point in pointList:
                glVertex3f(point[0], point[1], point[2])
            glEnd()
        
        glColor3f(1.0, 0.0, 0.0)
        for instance in range(0): #(drawCoefficients.shape[0] - uOrder + 1) * (drawCoefficients.shape[1] - vOrder + 1)):
            uM = uOrder - 1 + instance // (drawCoefficients.shape[1] - vOrder + 1)
            vM = vOrder - 1 + instance % (drawCoefficients.shape[1] - vOrder + 1)
            u = uKnots[uM]
            uBasis, duBasis, du2Basis = self.ComputeBasis(0, uM, u)
            vertices = 0
            verticesPerU = 0
            while u < uKnots[uM+1]:
                # Calculate deltaU (min deltaU value over the comvex hull of v values)
                deltaU = 0.5 * (uKnots[uM+1] - uKnots[uM])
                for vN in range(vM+1-vOrder, vM+1):
                    point = np.zeros(4, np.float32)
                    duPoint = np.zeros(4, np.float32)
                    du2Point = np.zeros(4, np.float32)
                    uB = 0
                    for uN in range(uM+1-uOrder, uM+1):
                        point += uBasis[uB] * drawCoefficients[uN, vN]
                        duPoint += duBasis[uB] * drawCoefficients[uN, vN]
                        du2Point += du2Basis[uB] * drawCoefficients[uN, vN]
                        uB += 1
                    newDeltaU = self.ComputeDelta(frame.screenScale, point, duPoint, du2Point, deltaU)
                    deltaU = min(newDeltaU, deltaU)

                # If there's less than 8 vertices for the last row, force this to be the last row.
                verticesPerU = verticesPerU if verticesPerU > 0 else 2 * int((uKnots[uM+1] - u) / deltaU)
                deltaU = deltaU if vertices + verticesPerU + 8 <= frame.maxVertices else 2.0 * (uKnots[uM+1] - u)

                u = min(u + deltaU, uKnots[uM+1])
                uBasisNext, duBasisNext, du2BasisNext = self.ComputeBasis(0, uM, u)

                v = vKnots[vM]
                deltaV = 0.5 * (vKnots[vM+1] - v)
                verticesPerU = 0
                glBegin(GL_LINE_STRIP)
                while v < vKnots[vM+1] and vertices <= frame.maxVertices - 4: # Save room for the vertices at v and lastV
                    deltaV = self.DrawSurfacePoints(frame.screenScale, drawCoefficients, uM, uBasis, duBasis, uBasisNext, duBasisNext, vM, v, deltaV)
                    v += deltaV
                    vertices += 2
                    verticesPerU += 2
                self.DrawSurfacePoints(frame.screenScale, drawCoefficients, uM, uBasis, duBasis, uBasisNext, duBasisNext, vM, vKnots[vM+1], deltaV)
                vertices += 2
                verticesPerU += 2
                glEnd()
                uBasis = uBasisNext
                duBasis = duBasisNext
                du2Basis = du2BasisNext

        glUseProgram(frame.surfaceProgram)

        glUniform3f(frame.uSurfaceSplineColor, 0.0, 1.0, 0.0)

        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 4
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((uOrder, vOrder, drawCoefficients.shape[0], drawCoefficients.shape[1]), np.float32))
        offset += size
        size = 4 * len(uKnots)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, uKnots)
        offset += size
        size = 4 * len(vKnots)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, vKnots)
        offset += size
        size = 4 * 4 * drawCoefficients.shape[0] * drawCoefficients.shape[1]
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)

        glEnableVertexAttribArray(frame.aSurfaceParameters)
        glDrawArraysInstanced(GL_POINTS, 0, 1, (drawCoefficients.shape[0] - uOrder + 1) * (drawCoefficients.shape[1] - vOrder + 1))
        glDisableVertexAttribArray(frame.aSurfaceParameters)

        glUseProgram(0)

    def Draw(self, frame, transform):
        drawCoefficients = self.coefficients @ transform
        if len(self.order) == 1:
            self.DrawCurve(frame, drawCoefficients)
        elif len(self.order) == 2:
            self.DrawSurface(frame, drawCoefficients)