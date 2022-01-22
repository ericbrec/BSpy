import numpy as np
import quaternion as quat
import tkinter as tk
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from pyopengltk import OpenGLFrame

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

    def ComputeBSpline(self, dimension, knot, u):
        uOrder = self.order[dimension]
        uKnots = self.knots[dimension]
        uBSpline = np.zeros(uOrder + 1, np.float32)
        duBSpline = np.zeros(uOrder + 1, np.float32)
        du2BSpline = np.zeros(uOrder + 1, np.float32)
        uBSpline[uOrder-1] = 1.0
        for degree in range(1, uOrder):
            b = uOrder - degree - 1
            for n in range(knot-degree, knot+1):
                gap0 = uKnots[n+degree] - uKnots[n]
                gap1 = uKnots[n+degree+1] - uKnots[n+1]
                gap0 = 1.0 / gap0 if gap0 > 0.0 else 0.0
                gap1 = 1.0 / gap1 if gap1 > 0.0 else 0.0
                val0 = (u - uKnots[n]) * gap0
                val1 = (uKnots[n+degree+1] - u) * gap1
                if degree == uOrder - 2:
                    d0 = degree * gap0
                    d1 = -degree * gap1
                    du2BSpline[b] = uBSpline[b] * d0 + uBSpline[b+1] * d1
                elif degree == uOrder - 1:
                    d0 = degree * gap0
                    d1 = -degree * gap1
                    duBSpline[b] = uBSpline[b] * d0 + uBSpline[b+1] * d1
                    du2BSpline[b] = du2BSpline[b] * d0 + du2BSpline[b+1] * d1
                uBSpline[b] = uBSpline[b] * val0 + uBSpline[b+1] * val1
                b += 1
        
        return uBSpline, duBSpline, du2BSpline
    
    def ComputeDelta(self, screenScale, point, dPoint, d2Point, delta):
        d2Screen = np.empty(2, np.float32)
        if screenScale[2] > 1.0:
            zScale = 1.0 / (screenScale[2] - point[2])
            zScale2 = zScale * zScale
            zScale3 = zScale2 * zScale
            d2Screen[0] = screenScale[0] * (d2Point[0] * zScale - 2.0 * dPoint[0] * dPoint[2] * zScale2 + \
                point[0] * (2.0 * dPoint[2] * dPoint[2] * zScale3 - d2Point[2] * zScale2))
            d2Screen[1] = screenScale[1] * (d2Point[1] * zScale - 2.0 * dPoint[1] * dPoint[2] * zScale2 + \
                point[1] * (2.0 * dPoint[2] * dPoint[2] * zScale3 - d2Point[2] * zScale2))
        else:
            d2Screen[0] = screenScale[0] * d2Point[0]
            d2Screen[1] = screenScale[1] * d2Point[1]
        
        sqrtLength = (d2Screen[0]*d2Screen[0] + d2Screen[1]*d2Screen[1])**0.25
        return delta if sqrtLength < 1.0e-8 else 1.0 / sqrtLength

    def ComputeCurveDelta(self, screenScale, drawCoefficients, uKnot):
        uOrder = self.order[0]
        uKnots = self.knots[0]
        uInterval = uKnots[uKnot+1] - uKnots[uKnot]

        deltaU = max(uInterval, 1.0e-8)
        for uN in range(uKnot+1-uOrder, uKnot-1):
            gap = uKnots[uN+uOrder] - uKnots[uN+1]
            gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
            dPoint0 = (uOrder - 1) * (drawCoefficients[uN+1] - drawCoefficients[uN]) * gap
            gap = uKnots[uN+1+uOrder] - uKnots[uN+2]
            gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
            dPoint1 = (uOrder - 1) * (drawCoefficients[uN+2] - drawCoefficients[uN+1]) * gap
            gap = uKnots[uN+uOrder] - uKnots[uN+2]
            gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
            d2Point = (uOrder - 2) * (dPoint1 - dPoint0) * gap
            deltaU = min(deltaU, self.ComputeDelta(screenScale, drawCoefficients[uN], dPoint0, d2Point, deltaU))
            deltaU = min(deltaU, self.ComputeDelta(screenScale, drawCoefficients[uN+1], dPoint0, d2Point, deltaU))
            deltaU = min(deltaU, self.ComputeDelta(screenScale, drawCoefficients[uN+1], dPoint1, d2Point, deltaU))
            deltaU = min(deltaU, self.ComputeDelta(screenScale, drawCoefficients[uN+2], dPoint1, d2Point, deltaU))

        self.samples = max(self.samples, uInterval / deltaU)
        return deltaU

    def DrawCurvePoints(self, screenScale, drawCoefficients, knot, u, deltaU):
        uOrder = self.order[0]
        uBSpline, duBSpline, du2BSpline = self.ComputeBSpline(0, knot, u)

        point = np.zeros(4, np.float32)
        dPoint = np.zeros(4, np.float32)
        d2Point = np.zeros(4, np.float32)
        b = 0
        for n in range(knot+1-uOrder, knot+1):
            point += uBSpline[b] * drawCoefficients[n]
            dPoint += duBSpline[b] * drawCoefficients[n]
            d2Point += du2BSpline[b] * drawCoefficients[n]
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
        self.samples = 0.0
        for knot in range(uOrder-1, len(uKnots)-uOrder):
            minDeltaU = self.ComputeCurveDelta(frame.screenScale, drawCoefficients, knot)
            u = uKnots[knot]
            deltaU = 0.5 * (uKnots[knot+1] - u)
            vertices = 0
            glBegin(GL_LINE_STRIP)
            while u < uKnots[knot+1] and vertices <= frame.maxVertices - 6: # Save room for the vertices at u and lastU
                deltaU = self.DrawCurvePoints(frame.screenScale, drawCoefficients, knot, u, deltaU)
                u += deltaU
                vertices += 3
            self.DrawCurvePoints(frame.screenScale, drawCoefficients, knot, uKnots[knot+1], deltaU)
            vertices += 3
            glEnd()
        print(self.samples)

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

    def ComputeSurfaceDeltas(self, screenScale, drawCoefficients, uKnot, vKnot):
        uOrder = self.order[0]
        uKnots = self.knots[0]
        vOrder = self.order[1]
        vKnots = self.knots[1]
        uInterval = uKnots[uKnot+1] - uKnots[uKnot]
        vInterval = vKnots[vKnot+1] - vKnots[vKnot]

        deltaU = max(uInterval, 1.0e-8)
        delta = deltaU
        deltaULeft = np.full(uOrder, deltaU, np.float32)
        deltaURight = np.full(uOrder, deltaU, np.float32)
        for vN in range(vKnot+1-vOrder, vKnot+1):
            for uN in range(max(uKnot-uOrder, 0), min(uKnot, drawCoefficients.shape[0]-2)):
                gap = uKnots[uN+uOrder] - uKnots[uN+1]
                gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
                dPoint0 = (uOrder - 1) * (drawCoefficients[uN+1, vN] - drawCoefficients[uN, vN]) * gap
                gap = uKnots[uN+1+uOrder] - uKnots[uN+2]
                gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
                dPoint1 = (uOrder - 1) * (drawCoefficients[uN+2, vN] - drawCoefficients[uN+1, vN]) * gap
                gap = uKnots[uN+uOrder] - uKnots[uN+2]
                gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
                d2Point = (uOrder - 2) * (dPoint1 - dPoint0) * gap
                if uKnot+1-uOrder <= uN < uKnot-1:
                    delta = min(delta, self.ComputeDelta(screenScale, drawCoefficients[uN, vN], dPoint0, d2Point, delta))
                    delta = min(delta, self.ComputeDelta(screenScale, drawCoefficients[uN+1, vN], dPoint0, d2Point, delta))
                    delta = min(delta, self.ComputeDelta(screenScale, drawCoefficients[uN+1, vN], dPoint1, d2Point, delta))
                    delta = min(delta, self.ComputeDelta(screenScale, drawCoefficients[uN+2, vN], dPoint1, d2Point, delta))
                k = uN - uKnot + uOrder
                deltaULeft[k] = min(deltaULeft[k], self.ComputeDelta(screenScale, drawCoefficients[uN, vN], dPoint0, d2Point, deltaULeft[k]))
                deltaULeft[k] = min(deltaULeft[k], self.ComputeDelta(screenScale, drawCoefficients[uN+1, vN], dPoint0, d2Point, deltaULeft[k]))
                deltaURight[k] = min(deltaURight[k], self.ComputeDelta(screenScale, drawCoefficients[uN+1, vN], dPoint1, d2Point, deltaURight[k]))
                deltaURight[k] = min(deltaURight[k], self.ComputeDelta(screenScale, drawCoefficients[uN+2, vN], dPoint1, d2Point, deltaURight[k]))
        deltaUm = deltaU
        deltaUm1 = deltaU
        for k in range(1, uOrder-1):
            deltaUm = min(deltaUm, deltaURight[k-1], deltaULeft[k])
            deltaU = min(deltaU, deltaULeft[k], deltaURight[k])
            deltaUm1 = min(deltaUm1, deltaURight[k], deltaULeft[k+1])

        deltaV = max(vInterval, 1.0e-8)
        for uN in range(uKnot+1-uOrder, uKnot+1):
            for vN in range(vKnot+1-vOrder, vKnot-1):
                gap = vKnots[vN+vOrder] - vKnots[vN+1]
                gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
                dPoint0 = (vOrder - 1) * (drawCoefficients[uN, vN+1] - drawCoefficients[uN, vN]) * gap
                gap = vKnots[vN+1+vOrder] - vKnots[vN+2]
                gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
                dPoint1 = (vOrder - 1) * (drawCoefficients[uN, vN+2] - drawCoefficients[uN, vN+1]) * gap
                gap = vKnots[vN+vOrder] - vKnots[vN+2]
                gap = 0.0 if gap < 1.0e-8 else 1.0 / gap
                d2Point = (vOrder - 2) * (dPoint1 - dPoint0) * gap
                deltaV = min(deltaV, self.ComputeDelta(screenScale, drawCoefficients[uN, vN], dPoint0, d2Point, deltaV))
                deltaV = min(deltaV, self.ComputeDelta(screenScale, drawCoefficients[uN, vN+1], dPoint0, d2Point, deltaV))
                deltaV = min(deltaV, self.ComputeDelta(screenScale, drawCoefficients[uN, vN+1], dPoint1, d2Point, deltaV))
                deltaV = min(deltaV, self.ComputeDelta(screenScale, drawCoefficients[uN, vN+2], dPoint1, d2Point, deltaV))

        self.samples = max(self.samples, uInterval / deltaU, vInterval / deltaV)
        return deltaU, deltaV

    def DrawSurfacePoints(self, screenScale, drawCoefficients, uKnot, uBSpline, duBSpline, uBSplineNext, duBSplineNext, vKnot, v, deltaV):
        uOrder = self.order[0]
        vOrder = self.order[1]
        vBSpline, dvBSpline, dv2BSpline = self.ComputeBSpline(1, vKnot, v)

        point = np.zeros(4, np.float32)
        duPoint = np.zeros(4, np.float32)
        dvPoint = np.zeros(4, np.float32)
        dv2Point = np.zeros(4, np.float32)
        vB = 0
        for vN in range(vKnot+1-vOrder, vKnot+1):
            uB = 0
            for uN in range(uKnot+1-uOrder, uKnot+1):
                point += uBSpline[uB] * vBSpline[vB] * drawCoefficients[uN, vN]
                duPoint += duBSpline[uB] * vBSpline[vB] * drawCoefficients[uN, vN]
                dvPoint += uBSpline[uB] * dvBSpline[vB] * drawCoefficients[uN, vN]
                dv2Point += uBSpline[uB] * dv2BSpline[vB] * drawCoefficients[uN, vN]
                uB += 1
            vB += 1

        glVertex3f(point[0], point[1], point[2])

        deltaV = self.ComputeDelta(screenScale, point, dvPoint, dv2Point, deltaV)

        point = np.zeros(4, np.float32)
        duPoint = np.zeros(4, np.float32)
        dvPoint = np.zeros(4, np.float32)
        dv2Point = np.zeros(4, np.float32)
        vB = 0
        for vN in range(vKnot+1-vOrder, vKnot+1):
            uB = 0
            for uN in range(uKnot+1-uOrder, uKnot+1):
                point += uBSplineNext[uB] * vBSpline[vB] * drawCoefficients[uN, vN]
                duPoint += duBSplineNext[uB] * vBSpline[vB] * drawCoefficients[uN, vN]
                dvPoint += uBSplineNext[uB] * dvBSpline[vB] * drawCoefficients[uN, vN]
                dv2Point += uBSplineNext[uB] * dv2BSpline[vB] * drawCoefficients[uN, vN]
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
        self.samples = 0.0
        for instance in range((drawCoefficients.shape[0] - uOrder + 1) * (drawCoefficients.shape[1] - vOrder + 1)):
            uKnot = uOrder - 1 + instance // (drawCoefficients.shape[1] - vOrder + 1)
            vKnot = vOrder - 1 + instance % (drawCoefficients.shape[1] - vOrder + 1)
            minDeltaU, minDeltaV = self.ComputeSurfaceDeltas(frame.screenScale, drawCoefficients, uKnot, vKnot)
            u = uKnots[uKnot]
            uBSpline, duBSpline, du2BSpline = self.ComputeBSpline(0, uKnot, u)
            vertices = 0
            verticesPerU = 0
            while u < uKnots[uKnot+1]:
                # Calculate deltaU (min deltaU value over the comvex hull of v values)
                deltaU = 0.5 * (uKnots[uKnot+1] - uKnots[uKnot])
                for vN in range(vKnot+1-vOrder, vKnot+1):
                    point = np.zeros(4, np.float32)
                    duPoint = np.zeros(4, np.float32)
                    du2Point = np.zeros(4, np.float32)
                    uB = 0
                    for uN in range(uKnot+1-uOrder, uKnot+1):
                        point += uBSpline[uB] * drawCoefficients[uN, vN]
                        duPoint += duBSpline[uB] * drawCoefficients[uN, vN]
                        du2Point += du2BSpline[uB] * drawCoefficients[uN, vN]
                        uB += 1
                    newDeltaU = self.ComputeDelta(frame.screenScale, point, duPoint, du2Point, deltaU)
                    deltaU = min(newDeltaU, deltaU)

                # If there's less than 8 vertices for the last row, force this to be the last row.
                verticesPerU = verticesPerU if verticesPerU > 0 else 2 * int((uKnots[uKnot+1] - u) / deltaU)
                #deltaU = deltaU if vertices + verticesPerU + 8 <= frame.maxVertices else 2.0 * (uKnots[uKnot+1] - u)

                u = min(u + deltaU, uKnots[uKnot+1])
                uBSplineNext, duBSplineNext, du2BSplineNext = self.ComputeBSpline(0, uKnot, u)

                v = vKnots[vKnot]
                deltaV = 0.5 * (vKnots[vKnot+1] - v)
                verticesPerU = 0
                glBegin(GL_LINE_STRIP)
                while v < vKnots[vKnot+1] and vertices <= frame.maxVertices - 4: # Save room for the vertices at v and lastV
                    deltaV = self.DrawSurfacePoints(frame.screenScale, drawCoefficients, uKnot, uBSpline, duBSpline, uBSplineNext, duBSplineNext, vKnot, v, deltaV)
                    v += deltaV
                    vertices += 2
                    verticesPerU += 2
                self.DrawSurfacePoints(frame.screenScale, drawCoefficients, uKnot, uBSpline, duBSpline, uBSplineNext, duBSplineNext, vKnot, vKnots[vKnot+1], deltaV)
                vertices += 2
                verticesPerU += 2
                glEnd()
                uBSpline = uBSplineNext
                duBSpline = duBSplineNext
                du2BSpline = du2BSplineNext
        print(self.samples)

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
        #glDrawArraysInstanced(GL_POINTS, 0, 1, (drawCoefficients.shape[0] - uOrder + 1) * (drawCoefficients.shape[1] - vOrder + 1))
        glDisableVertexAttribArray(frame.aSurfaceParameters)

        glUseProgram(0)

    def Draw(self, frame, transform):
        drawCoefficients = self.coefficients @ transform
        if len(self.order) == 1:
            self.DrawCurve(frame, drawCoefficients)
        elif len(self.order) == 2:
            self.DrawSurface(frame, drawCoefficients)

class SplineOpenGLFrame(OpenGLFrame):

    computeBSplineCode = """
        void ComputeBSpline(in int offset, in int order, in int n, in int knot, in float u, 
            out float uBSpline[{maxBSpline}], out float duBSpline[{maxBSpline}], out float du2BSpline[{maxBSpline}]) {{
            int degree = 1;

            uBSpline = float[{maxBSpline}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            duBSpline = float[{maxBSpline}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            du2BSpline = float[{maxBSpline}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            uBSpline[order-1] = 1.0;

            while (degree < order - 2) {{
                int b = order - degree - 1;
                for (int i = knot - degree; i < knot + 1; i++) {{
                    float gap0 = texelFetch(uSplineData, offset + i + degree).x - texelFetch(uSplineData, offset + i).x; // knots[i+degree] - knots[i]
                    float gap1 = texelFetch(uSplineData, offset + i + degree + 1).x - texelFetch(uSplineData, offset + i + 1).x; // knots[i+degree+1] - knots[i+1]
                    gap0 = gap0 < 1.0e-8 ? 0.0 : 1.0 / gap0;
                    gap1 = gap1 < 1.0e-8 ? 0.0 : 1.0 / gap1;

                    float val0 = (u - texelFetch(uSplineData, offset + i).x) * gap0; // (u - knots[i]) * gap0;
                    float val1 = (texelFetch(uSplineData, offset + i + degree + 1).x - u) * gap1; // (knots[i+degree+1] - u) * gap1;
                    uBSpline[b] = uBSpline[b] * val0 + uBSpline[b+1] * val1;
                    b++;
                }}
                degree++;
            }}
            while (degree < order - 1) {{
                int b = order - degree - 1;
                for (int i = knot - degree; i < knot + 1; i++) {{
                    float gap0 = texelFetch(uSplineData, offset + i + degree).x - texelFetch(uSplineData, offset + i).x; // knots[i+degree] - knots[i]
                    float gap1 = texelFetch(uSplineData, offset + i + degree + 1).x - texelFetch(uSplineData, offset + i + 1).x; // knots[i+degree+1] - knots[i+1]
                    gap0 = gap0 < 1.0e-8 ? 0.0 : 1.0 / gap0;
                    gap1 = gap1 < 1.0e-8 ? 0.0 : 1.0 / gap1;

                    float d0 = degree * gap0;
                    float d1 = -degree * gap1;
                    du2BSpline[b] = uBSpline[b] * d0 + uBSpline[b+1] * d1;
                    float val0 = (u - texelFetch(uSplineData, offset + i).x) * gap0; // (u - knots[i]) * gap0;
                    float val1 = (texelFetch(uSplineData, offset + i + degree + 1).x - u) * gap1; // (knots[i+degree+1] - u) * gap1;
                    uBSpline[b] = uBSpline[b] * val0 + uBSpline[b+1] * val1;
                    b++;
                }}
                degree++;
            }}
            while (degree < order) {{
                int b = order - degree - 1;
                for (int i = knot - degree; i < knot + 1; i++) {{
                    float gap0 = texelFetch(uSplineData, offset + i + degree).x - texelFetch(uSplineData, offset + i).x; // knots[i+degree] - knots[i]
                    float gap1 = texelFetch(uSplineData, offset + i + degree + 1).x - texelFetch(uSplineData, offset + i + 1).x; // knots[i+degree+1] - knots[i+1]
                    gap0 = gap0 < 1.0e-8 ? 0.0 : 1.0 / gap0;
                    gap1 = gap1 < 1.0e-8 ? 0.0 : 1.0 / gap1;

                    float d0 = degree * gap0;
                    float d1 = -degree * gap1;
                    duBSpline[b] = uBSpline[b] * d0 + uBSpline[b+1] * d1;
                    du2BSpline[b] = du2BSpline[b] * d0 + du2BSpline[b+1] * d1;
                    float val0 = (u - texelFetch(uSplineData, offset + i).x) * gap0; // (u - knots[i]) * gap0;
                    float val1 = (texelFetch(uSplineData, offset + i + degree + 1).x - u) * gap1; // (knots[i+degree+1] - u) * gap1;
                    uBSpline[b] = uBSpline[b] * val0 + uBSpline[b+1] * val1;
                    b++;
                }}
                degree++;
            }}
        }}
    """

    computeDeltaCode = """
        void ComputeDelta(in vec4 point, in vec3 dPoint, in vec3 d2Point, inout float delta) {
            float zScale = 1.0 / (uScreenScale.z - point.z);
            float zScale2 = zScale * zScale;
            float zScale3 = zScale2 * zScale;
            vec2 projection = uScreenScale.z > 1.0 ? 
                vec2(uScreenScale.x * (d2Point.x * zScale - 2.0 * dPoint.x * dPoint.z * zScale2 +
                    point.x * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2)),
                    uScreenScale.y * (d2Point.y * zScale - 2.0 * dPoint.y * dPoint.z * zScale2 +
                    point.y * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2)))
                : vec2(uScreenScale.x * d2Point.x, uScreenScale.y * d2Point.y);
            float projectionLength = length(projection);
            delta = projectionLength < 1.0e-8 ? delta : 1.0 / sqrt(projectionLength);
        }
    """

    curveVertexShaderCode = """
        #version 330 core
     
        attribute vec4 aParameters;

        uniform samplerBuffer uSplineData;

        out KnotIndices {
            int knot;
        } outData;

        void main() {
            int order = int(texelFetch(uSplineData, 0).x);
            int n = int(texelFetch(uSplineData, 1).x);

            outData.knot = min(gl_InstanceID + order - 1, n - 1);
            gl_Position = aParameters;
        }
    """

    curveGeometryShaderCode = """
        #version 330 core

        layout( points ) in;
        layout( line_strip, max_vertices = {maxVertices} ) out;

        const int header = 2;

        in KnotIndices {{
            int knot;
        }} inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec3 uSplineColor;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor;

        {computeBSplineCode}

        {computeDeltaCode}

        void DrawCurvePoints(in int order, in int n, in int knot, in float u, inout float deltaU) {{
            float uBSpline[{maxBSpline}];
            float duBSpline[{maxBSpline}];
            float du2BSpline[{maxBSpline}];

            ComputeBSpline(header, order, n, knot, u, uBSpline, duBSpline, du2BSpline);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            vec3 dPoint = vec3(0.0, 0.0, 0.0);
            vec3 d2Point = vec3(0.0, 0.0, 0.0);
            int i = header + order + n + 4 * (knot + 1 - order);
            for (int b = 0; b < order; b++) {{ // loop from coefficient[knot+1-order] to coefficient[knot+1]
                point.x += uBSpline[b] * texelFetch(uSplineData, i).x;
                point.y += uBSpline[b] * texelFetch(uSplineData, i+1).x;
                point.z += uBSpline[b] * texelFetch(uSplineData, i+2).x;
                point.w += uBSpline[b] * texelFetch(uSplineData, i+3).x;
                dPoint.x += duBSpline[b] * texelFetch(uSplineData, i).x;
                dPoint.y += duBSpline[b] * texelFetch(uSplineData, i+1).x;
                dPoint.z += duBSpline[b] * texelFetch(uSplineData, i+2).x;
                d2Point.x += du2BSpline[b] * texelFetch(uSplineData, i).x;
                d2Point.y += du2BSpline[b] * texelFetch(uSplineData, i+1).x;
                d2Point.z += du2BSpline[b] * texelFetch(uSplineData, i+2).x;
                i += 4;
            }}

            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            gl_Position = uProjectionMatrix * vec4(point.x - 0.01*dPoint.y, point.y + 0.01*dPoint.x, point.z, point.w);
            EmitVertex();
            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            
            ComputeDelta(point, dPoint, d2Point, deltaU);
        }}

        void main() {{
            int order = int(texelFetch(uSplineData, 0).x);
            int n = int(texelFetch(uSplineData, 1).x);
            int knot = inData[0].knot;
            float u = texelFetch(uSplineData, header + knot).x; // knots[knot]
            float lastU = texelFetch(uSplineData, header + knot + 1).x; // knots[knot+1]
            float deltaU = 0.5 * (lastU - u);
            int vertices = 0;

            splineColor = uSplineColor;
            while (u < lastU && vertices <= {maxVertices} - 6) {{ // Save room for the vertices at u and lastU
                DrawCurvePoints(order, n, knot, u, deltaU);
                u += deltaU;
                vertices += 3;
            }}
            DrawCurvePoints(order, n, knot, lastU, deltaU);
            vertices += 3;
            EndPrimitive();
        }}
    """

    surfaceVertexShaderCode = """
        #version 330 core
     
        attribute vec4 aParameters;

        uniform samplerBuffer uSplineData;

        out KnotIndices {
            int uKnot;
            int vKnot;
        } outData;

        void main() {
            int uOrder = int(texelFetch(uSplineData, 0).x);
            int vOrder = int(texelFetch(uSplineData, 1).x);
            int uN = int(texelFetch(uSplineData, 2).x);
            int vN = int(texelFetch(uSplineData, 3).x);
            int stride = vN - vOrder + 1;

            outData.uKnot = min(int(gl_InstanceID / stride) + uOrder - 1, uN - 1);
            outData.vKnot = min(int(mod(gl_InstanceID, stride)) + vOrder - 1, vN - 1);
            gl_Position = aParameters;
        }
    """

    surfaceGeometryShaderCode = """
        #version 330 core

        layout( points ) in;
        layout( triangle_strip, max_vertices = {maxVertices} ) out;

        const int header = 4;

        in KnotIndices {{
            int uKnot;
            int vKnot;
        }} inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec3 uSplineColor;
        uniform vec3 uLightDirection;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor;

        {computeBSplineCode}

        {computeDeltaCode}

        void DrawSurfacePoints(in int uOrder, in int uN, in int uKnot,
            in float uBSpline[{maxBSpline}], in float duBSpline[{maxBSpline}],
            in float uBSplineNext[{maxBSpline}], in float duBSplineNext[{maxBSpline}],
            in int vOrder, in int vN, in int vKnot, in float v, inout float deltaV) {{
            float vBSpline[{maxBSpline}];
            float dvBSpline[{maxBSpline}];
            float dv2BSpline[{maxBSpline}];

            ComputeBSpline(header + uOrder + uN, vOrder, vN, vKnot, v, vBSpline, dvBSpline, dv2BSpline);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            vec3 duPoint = vec3(0.0, 0.0, 0.0);
            vec3 dvPoint = vec3(0.0, 0.0, 0.0);
            vec3 dv2Point = vec3(0.0, 0.0, 0.0);
            int j = header + uOrder + uN + vOrder + vN + (vKnot + 1 - vOrder) * 4;
            for (int vB = 0; vB < vOrder; vB++) {{
                int i = j + (uKnot + 1 - uOrder) * vN * 4;
                for (int uB = 0; uB < uOrder; uB++) {{
                    point.x += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                    point.y += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    point.z += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    point.w += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+3).x;
                    duPoint.x += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                    duPoint.y += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    duPoint.z += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    dvPoint.x += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i).x;
                    dvPoint.y += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    dvPoint.z += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    dv2Point.x += uBSpline[uB] * dv2BSpline[vB] * texelFetch(uSplineData, i).x;
                    dv2Point.y += uBSpline[uB] * dv2BSpline[vB] * texelFetch(uSplineData, i+1).x;
                    dv2Point.z += uBSpline[uB] * dv2BSpline[vB] * texelFetch(uSplineData, i+2).x;
                    i += vN * 4;
                }}
                j += 4;
            }}

            vec3 normal = normalize(cross(duPoint, dvPoint));
            float intensity = dot(normal, uLightDirection);
            splineColor = (0.3 + 0.7 * abs(intensity)) * uSplineColor;
            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            
            ComputeDelta(point, dvPoint, dv2Point, deltaV);
            
            point = vec4(0.0, 0.0, 0.0, 0.0);
            duPoint = vec3(0.0, 0.0, 0.0);
            dvPoint = vec3(0.0, 0.0, 0.0);
            dv2Point = vec3(0.0, 0.0, 0.0);
            j = header + uOrder + uN + vOrder + vN + (vKnot + 1 - vOrder) * 4;
            for (int vB = 0; vB < vOrder; vB++) {{
                int i = j + (uKnot + 1 - uOrder) * vN * 4;
                for (int uB = 0; uB < uOrder; uB++) {{
                    point.x += uBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                    point.y += uBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    point.z += uBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    point.w += uBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+3).x;
                    duPoint.x += duBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                    duPoint.y += duBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    duPoint.z += duBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    dvPoint.x += uBSplineNext[uB] * dvBSpline[vB] * texelFetch(uSplineData, i).x;
                    dvPoint.y += uBSplineNext[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    dvPoint.z += uBSplineNext[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    dv2Point.x += uBSplineNext[uB] * dv2BSpline[vB] * texelFetch(uSplineData, i).x;
                    dv2Point.y += uBSplineNext[uB] * dv2BSpline[vB] * texelFetch(uSplineData, i+1).x;
                    dv2Point.z += uBSplineNext[uB] * dv2BSpline[vB] * texelFetch(uSplineData, i+2).x;
                    i += vN * 4;
                }}
                j += 4;
            }}

            normal = normalize(cross(duPoint, dvPoint));
            intensity = dot(normal, uLightDirection);
            splineColor = (0.3 + 0.7 * abs(intensity)) * uSplineColor;
            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            
            float newDeltaV = deltaV;
            ComputeDelta(point, dvPoint, dv2Point, newDeltaV);
            deltaV = min(newDeltaV, deltaV);
        }}

        void main() {{
            int uOrder = int(texelFetch(uSplineData, 0).x);
            int vOrder = int(texelFetch(uSplineData, 1).x);
            int uN = int(texelFetch(uSplineData, 2).x);
            int vN = int(texelFetch(uSplineData, 3).x);
            int uKnot = inData[0].uKnot;
            int vKnot = inData[0].vKnot;
            float firstU = texelFetch(uSplineData, header + uKnot).x; // uKnots[uKnot]
            float lastU = texelFetch(uSplineData, header + uKnot + 1).x; // uKnots[uKnot+1]
            float u = firstU;
            int vertices = 0;
            int verticesPerU = 0;
            float uBSpline[{maxBSpline}];
            float duBSpline[{maxBSpline}];
            float du2BSpline[{maxBSpline}];

            splineColor = uSplineColor;
            ComputeBSpline(header, uOrder, uN, uKnot, u, uBSpline, duBSpline, du2BSpline);
            while (u < lastU && vertices < {maxVertices}) {{
                // Calculate deltaU (min deltaU value over the comvex hull of v values)
                float deltaU = 0.5 * (lastU - firstU);
                int j = header + uOrder + uN + vOrder + vN + (vKnot + 1 - vOrder) * 4;
                for (int vB = 0; vB < vOrder; vB++) {{
                    vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
                    vec3 duPoint = vec3(0.0, 0.0, 0.0);
                    vec3 du2Point = vec3(0.0, 0.0, 0.0);
                    int i = j + (uKnot + 1 - uOrder) * vN * 4;
                    for (int uB = 0; uB < uOrder; uB++) {{
                        point.x += uBSpline[uB] * texelFetch(uSplineData, i).x;
                        point.y += uBSpline[uB] * texelFetch(uSplineData, i+1).x;
                        point.z += uBSpline[uB] * texelFetch(uSplineData, i+2).x;
                        point.w += uBSpline[uB] * texelFetch(uSplineData, i+3).x;
                        duPoint.x += duBSpline[uB] * texelFetch(uSplineData, i).x;
                        duPoint.y += duBSpline[uB] * texelFetch(uSplineData, i+1).x;
                        duPoint.z += duBSpline[uB] * texelFetch(uSplineData, i+2).x;
                        du2Point.x += du2BSpline[uB] * texelFetch(uSplineData, i).x;
                        du2Point.y += du2BSpline[uB] * texelFetch(uSplineData, i+1).x;
                        du2Point.z += du2BSpline[uB] * texelFetch(uSplineData, i+2).x;
                        i += vN * 4;
                    }}
                    float newDeltaU = deltaU;
                    ComputeDelta(point, duPoint, du2Point, newDeltaU);
                    deltaU = min(newDeltaU, deltaU);
                    j += 4;
                }}

                // If there's less than 8 vertices for the last row, force this to be the last row.
                verticesPerU = verticesPerU > 0 ? verticesPerU : 2 * int((lastU - u) / deltaU);
                //deltaU = vertices + verticesPerU + 8 <= {maxVertices} ? deltaU : 2.0 * (lastU - u);

                u = min(u + deltaU, lastU);
                float uBSplineNext[{maxBSpline}];
                float duBSplineNext[{maxBSpline}];
                float du2BSplineNext[{maxBSpline}];
                ComputeBSpline(header, uOrder, uN, uKnot, u, uBSplineNext, duBSplineNext, du2BSplineNext);

                float v = texelFetch(uSplineData, header + uOrder + uN + vKnot).x; // vKnots[vKnot]
                float lastV = texelFetch(uSplineData, header + uOrder + uN + vKnot + 1).x; // vKnots[vKnot+1]
                float deltaV = 0.5 * (lastV - v);
                verticesPerU = 0;
                while (v < lastV && vertices <= {maxVertices} - 4) {{ // Save room for the vertices at v and lastV
                    DrawSurfacePoints(uOrder, uN, uKnot, uBSpline, duBSpline, uBSplineNext, duBSplineNext,
                        vOrder, vN, vKnot, v, deltaV);
                    v = min(v + deltaV, lastV);
                    vertices += 2;
                    verticesPerU += 2;
                }}
                DrawSurfacePoints(uOrder, uN, uKnot, uBSpline, duBSpline, uBSplineNext, duBSplineNext,
                    vOrder, vN, vKnot, v, deltaV); //lastV, deltaV);
                vertices += 2;
                verticesPerU += 2;
                EndPrimitive();
                uBSpline = uBSplineNext;
                duBSpline = duBSplineNext;
                du2BSpline = du2BSplineNext;
            }}
        }}
    """

    fragmentShaderCode = """
        #version 330 core
     
        in vec3 splineColor;
        out vec3 color;
     
        void main() {
            color = splineColor;
        }
    """
 
    def __init__(self, *args, **kw):
        OpenGLFrame.__init__(self, *args, **kw)
        self.animate = 0 # Set to number of milliseconds before showing next frame (0 means no animation)
        self.splineDrawList = []
        self.currentQ = quat.one
        self.lastQ = quat.one
        self.origin = None
        self.bind("<ButtonPress-3>", self.Home)
        self.bind("<ButtonPress-1>", self.RotateStartHandler)
        self.bind("<ButtonRelease-1>", self.RotateEndHandler)
        self.bind("<B1-Motion>", self.RotateDragHandler)
        self.glInitialized = False

    def initgl(self):
        if not self.glInitialized:
            self.maxVertices = glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES)
            print("GL_VERSION: ", glGetString(GL_VERSION))
            print("GL_SHADING_LANGUAGE_VERSION: ", glGetString(GL_SHADING_LANGUAGE_VERSION))
            print("GL_MAX_GEOMETRY_OUTPUT_VERTICES: ", glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES))
            print("GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS: ", glGetIntegerv(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS))
            print("GL_MAX_TESS_GEN_LEVEL: ", glGetIntegerv(GL_MAX_TESS_GEN_LEVEL))

            try:
                self.computeBSplineCode = self.computeBSplineCode.format(maxBSpline=Spline.maxOrder+1)

                self.curveGeometryShaderCode = self.curveGeometryShaderCode.format(maxVertices=self.maxVertices,
                    computeBSplineCode=self.computeBSplineCode,
                    computeDeltaCode=self.computeDeltaCode,
                    maxBSpline=Spline.maxOrder+1)
                self.curveProgram = shaders.compileProgram(
                    shaders.compileShader(self.curveVertexShaderCode, GL_VERTEX_SHADER), 
                    shaders.compileShader(self.curveGeometryShaderCode, GL_GEOMETRY_SHADER), 
                    shaders.compileShader(self.fragmentShaderCode, GL_FRAGMENT_SHADER))
                
                self.surfaceGeometryShaderCode = self.surfaceGeometryShaderCode.format(maxVertices=self.maxVertices,
                    computeBSplineCode=self.computeBSplineCode,
                    computeDeltaCode=self.computeDeltaCode,
                    maxBSpline=Spline.maxOrder+1)
                self.surfaceProgram = shaders.compileProgram(
                    shaders.compileShader(self.surfaceVertexShaderCode, GL_VERTEX_SHADER), 
                    shaders.compileShader(self.surfaceGeometryShaderCode, GL_GEOMETRY_SHADER), 
                    shaders.compileShader(self.fragmentShaderCode, GL_FRAGMENT_SHADER))
            except shaders.ShaderCompilationError as exception:
                error = exception.args[0]
                lineNumber = error.split(":")[3]
                source = exception.args[1][0]
                badLine = source.split(b"\n")[int(lineNumber)-1]
                shaderType = exception.args[2]
                print(shaderType, error)
                print(badLine)
                quit()

            self.splineDataBuffer = glGenBuffers(1)
            self.splineTextureBuffer = glGenTextures(1)
            glBindBuffer(GL_TEXTURE_BUFFER, self.splineDataBuffer)
            glBindTexture(GL_TEXTURE_BUFFER, self.splineTextureBuffer)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.splineDataBuffer)
            maxFloats = 4 + 2 * Spline.maxKnots + 4 * Spline.maxCoefficients * Spline.maxCoefficients
            glBufferData(GL_TEXTURE_BUFFER, 4 * maxFloats, None, GL_STATIC_READ)

            self.parameterBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glBufferData(GL_ARRAY_BUFFER, 4 * 4, np.array([0,0,0,0], np.float32), GL_STATIC_DRAW)

            glUseProgram(self.curveProgram)
            self.aCurveParameters = glGetAttribLocation(self.curveProgram, "aParameters")
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glVertexAttribPointer(self.aCurveParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uCurveProjectionMatrix = glGetUniformLocation(self.curveProgram, 'uProjectionMatrix')
            self.uCurveScreenScale = glGetUniformLocation(self.curveProgram, 'uScreenScale')
            self.uCurveSplineColor = glGetUniformLocation(self.curveProgram, 'uSplineColor')
            self.uCurveSplineData = glGetUniformLocation(self.curveProgram, 'uSplineData')
            glUniform1i(self.uCurveSplineData, 0) # 0 is the active texture (default is 0)

            glUseProgram(self.surfaceProgram)
            self.aSurfaceParameters = glGetAttribLocation(self.surfaceProgram, "aParameters")
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glVertexAttribPointer(self.aSurfaceParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uSurfaceProjectionMatrix = glGetUniformLocation(self.surfaceProgram, 'uProjectionMatrix')
            self.uSurfaceScreenScale = glGetUniformLocation(self.surfaceProgram, 'uScreenScale')
            self.uSurfaceSplineColor = glGetUniformLocation(self.surfaceProgram, 'uSplineColor')
            self.uSurfaceLightDirection = glGetUniformLocation(self.surfaceProgram, 'uLightDirection')
            self.lightDirection = np.array((0.6, 0.3, 1), np.float32)
            self.lightDirection = self.lightDirection / np.linalg.norm(self.lightDirection)
            glUniform3fv(self.uSurfaceLightDirection, 1, self.lightDirection)
            self.uSurfaceSplineData = glGetUniformLocation(self.surfaceProgram, 'uSplineData')
            glUniform1i(self.uSurfaceSplineData, 0) # 0 is the active texture (default is 0)

            glUseProgram(0)

            glEnable( GL_DEPTH_TEST )
            glClearColor(1.0, 1.0, 1.0, 0.0)

            self.glInitialized = True

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        xExtent = self.width / self.height
        clipDistance = np.sqrt(3.0)
        zDropoff = 3.0
        near = zDropoff - clipDistance
        far = zDropoff + clipDistance
        top = clipDistance * near / zDropoff # Choose frustum that displays [-clipDistance,clipDistance] in y for z = -zDropoff
        glFrustum(-top*xExtent, top*xExtent, -top, top, near, far)
        glTranslate(0.0, 0.0, -zDropoff)
        #glOrtho(-xExtent, xExtent, -1.0, 1.0, -1.0, 1.0)

        self.projection = glGetFloatv(GL_PROJECTION_MATRIX)
        self.screenScale = np.array((0.5 * self.height * self.projection[0,0], 0.5 * self.height * self.projection[1,1], self.projection[3,3]), np.float32)
        glUseProgram(self.curveProgram)
        glUniformMatrix4fv(self.uCurveProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uCurveScreenScale, 1, self.screenScale)
        glUseProgram(self.surfaceProgram)
        glUniformMatrix4fv(self.uSurfaceProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uSurfaceScreenScale, 1, self.screenScale)
        glUseProgram(0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def redraw(self):

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glLoadIdentity()
        rotation33 = quat.as_rotation_matrix(self.currentQ * self.lastQ)
        rotation44 = np.identity(4, np.float32)
        rotation44[0:3,0:3] = rotation33.T # Transpose to match OpenGL format in numpy
        transform = rotation44

        for spline in self.splineDrawList:
            spline.Draw(self, transform)

        glFlush()

    def Home(self, event):
        self.lastQ = quat.one
        self.currentQ = quat.one
        self.tkExpose(None)

    def ProjectToSphere(self, point):
        length = np.linalg.norm(point)
        if length <= 0.7071: # 1/sqrt(2)
            projection = np.array((point[0], point[1], np.sqrt(1.0 - length * length)), np.float32)
        else:
            projection = np.array((point[0], point[1], 0.5 / length), np.float32)
            projection = projection / np.linalg.norm(projection)
        return projection

    def RotateStartHandler(self, event):
        self.origin = np.array(((2.0 * event.x - self.width)/self.height, (self.height - 2.0 * event.y)/self.height), np.float32)

    def RotateDragHandler(self, event):
        if self.origin is not None:
            point = np.array(((2.0 * event.x - self.width)/self.height, (self.height - 2.0 * event.y)/self.height), np.float32)
            a = self.ProjectToSphere(self.origin)
            b = self.ProjectToSphere(point)
            dot = np.dot(a, b)
            halfCosine = np.sqrt(0.5 * (1.0 + dot))
            halfSine = np.sqrt(0.5 * (1.0 - dot))
            n = np.cross(a,b)
            if halfSine > 1.0e-8:
                n = (halfSine / np.linalg.norm(n)) * n
            self.currentQ = quat.from_float_array((halfCosine, n[0], n[1], n[2]))
            self.tkExpose(None)

    def RotateEndHandler(self, event):
        if self.origin is not None:
            self.lastQ = self.currentQ * self.lastQ
            self.currentQ = quat.one
            self.origin = None

class bspyApp(tk.Tk):
    def __init__(self, *args, **kw):
        tk.Tk.__init__(self, *args, **kw)
        self.title('bspy')

        self.listBox = tk.Listbox(self, selectmode=tk.MULTIPLE)
        self.listBox.pack(side=tk.LEFT, fill=tk.Y)
        self.listBox.bind('<<ListboxSelect>>', self.ListChanged)

        self.verticalScroll = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.verticalScroll.pack(side=tk.LEFT, fill=tk.Y)
        self.listBox.configure(yscrollcommand=self.verticalScroll.set)
        self.verticalScroll.config(command=self.listBox.yview)

        self.frame = SplineOpenGLFrame(self, width=500, height=500)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        self.splineList = []

    def AddSpline(self, spline):
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)

    def ListChanged(self, event):
        self.frame.splineDrawList = []
        for item in self.listBox.curselection():
            self.frame.splineDrawList.append(self.splineList[item])
        self.frame.tkExpose(None)

def CreateSplineFromMesh(xRange, zRange, yFunction):
    order = (3, 3)
    coefficients = np.zeros((xRange[2], zRange[2], 4), np.float32)
    knots = (np.zeros(xRange[2] + order[0], np.float32), np.zeros(zRange[2] + order[1], np.float32))
    knots[0][:xRange[2]] = np.linspace(xRange[0], xRange[1], xRange[2], dtype=np.float32)[:]
    knots[0][xRange[2]:] = xRange[1]
    knots[1][:zRange[2]] = np.linspace(zRange[0], zRange[1], zRange[2], dtype=np.float32)[:]
    knots[1][zRange[2]:] = zRange[1]
    for i in range(xRange[2]):
        for j in range(zRange[2]):
            coefficients[i, j, 0] = knots[0][i]
            coefficients[i, j, 1] = yFunction(knots[0][i], knots[1][j])
            coefficients[i, j, 2] = knots[1][j]
            coefficients[i, j, 3] = 1.0
    
    return Spline(order, knots, coefficients)

if __name__=='__main__':
    app = bspyApp()
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: np.sin(4*np.sqrt(x*x + y*y))))
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x + y*y - 1))
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x - y*y))
    for i in range(16):
        app.AddSpline(Spline((3,), (np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5], np.float32),), np.array([[-1, 0, 0, 1], [-0.5, i/16.0, 0, 1], [0,0,0,1], [0.5, -i/16.0, 0, 1], [1,0,0,1]], np.float32)))
    app.mainloop()