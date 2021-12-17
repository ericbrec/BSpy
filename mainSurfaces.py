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

    def DrawCurvePoint(self, screenScale, drawCoefficients, m, u, deltaU):
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

        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for point in drawCoefficients:
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        glColor3f(0.0, 1.0, 0.0)
        for m in range(uOrder-1, len(uKnots)-uOrder):
            u = uKnots[m]
            deltaU = 0.5 * (uKnots[m+1] - u)
            vertices = 0
            glBegin(GL_LINE_STRIP)
            while u < uKnots[m+1] and vertices < frame.maxVertices - 3:
                deltaU = self.DrawCurvePoint(frame.screenScale, drawCoefficients, m, u, deltaU)
                u += deltaU
                vertices += 3
            self.DrawCurvePoint(frame.screenScale, drawCoefficients, m, uKnots[m+1], deltaU)
            glEnd()

        glUseProgram(frame.curveProgram)

        glUniform3f(frame.uCurveSplineColor, 0.0, 0.0, 1.0)

        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 2
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((uOrder, len(drawCoefficients)), np.float32))
        offset += size
        size = 4 * len(uKnots)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, uKnots)
        offset += size
        size = 4 * 4 * len(drawCoefficients)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)

        glEnableVertexAttribArray(frame.aCurveParameters)
        glDrawArraysInstanced(GL_POINTS, 0, 1, len(drawCoefficients) - uOrder + 1)
        glDisableVertexAttribArray(frame.aCurveParameters)

        glUseProgram(0)

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
        for uM in range(uOrder-1, len(uKnots)-uOrder):
            for vM in range(vOrder-1, len(vKnots)-vOrder):
                u = uKnots[uM]
                uBasis, duBasis, du2Basis = self.ComputeBasis(0, uM, u)
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

                    u = min(u + deltaU, uKnots[uM+1])
                    uBasisNext, duBasisNext, du2BasisNext = self.ComputeBasis(0, uM, u)
    
                    v = vKnots[vM]
                    deltaV = 0.5 * (vKnots[vM+1] - v)
                    vertices = 0
                    glBegin(GL_LINE_STRIP)
                    while v < vKnots[vM+1] and vertices < frame.maxVertices - 2:
                    
                        # TODO: Move this to DrawSurfacePoint()
                    
                        vBasis, dvBasis, dv2Basis = self.ComputeBasis(0, vM, v)
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
                        deltaV = self.ComputeDelta(frame.screenScale, point, dvPoint, dv2Point, deltaV)
                        glVertex3f(point[0], point[1], point[2])

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
                        newDeltaV = self.ComputeDelta(frame.screenScale, point, dvPoint, dv2Point, deltaV)
                        deltaV = min(newDeltaV, deltaV)
                        glVertex3f(point[0], point[1], point[2])

                        v += deltaV
                        vertices += 2
                    
                    # TODO: DrawSurfacePoint for vKnots[vM+1]
                    glEnd()
                    uBasis = uBasisNext
                    duBasis = duBasisNext
                    du2Basis = du2BasisNext
    
    def Draw(self, frame, transform):
        drawCoefficients = self.coefficients @ transform
        if len(self.order) == 1:
            self.DrawCurve(frame, drawCoefficients)
        elif len(self.order) == 2:
            self.DrawSurface(frame, drawCoefficients)

class SplineOpenGLFrame(OpenGLFrame):

    computeBasisCode = """
        void ComputeBasis(in int offset, in int order, in int n, in int m, in float u, 
            out float uBasis[{maxBasis}], out float duBasis[{maxBasis}], out float du2Basis[{maxBasis}]) {{
            uBasis = float[{maxBasis}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            duBasis = float[{maxBasis}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            du2Basis = float[{maxBasis}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

            int b;
            uBasis[order-1] = 1.0;
            for (int degree = 1; degree < order; degree++) {{
                b = order - degree - 1;
                for (int i = m - degree; i < m + 1; i++) {{
                    float gap0 = texelFetch(uSplineData, offset + i + degree).x - texelFetch(uSplineData, offset + i).x; // knots[i+degree] - knots[i]
                    float gap1 = texelFetch(uSplineData, offset + i + degree + 1).x - texelFetch(uSplineData, offset + i + 1).x; // knots[i+degree+1] - knots[i+1]
                    float val0 = 0.0;
                    float val1 = 0.0;
                    float d0 = 0.0;
                    float d1 = 0.0;

                    gap0 = gap0 < 1.0e-8 ? 0.0 : 1.0 / gap0;
                    gap1 = gap1 < 1.0e-8 ? 0.0 : 1.0 / gap1;
                    val0 = (u - texelFetch(uSplineData, offset + i).x) * gap0; // (u - knots[i]) * gap0
                    val1 = (texelFetch(uSplineData, offset + i + degree + 1).x - u) * gap1; // (knots[i+degree+1] - u) * gap1
                    if (degree == order - 2) {{
                        d0 = degree * gap0;
                        d1 = -degree * gap1;
                        du2Basis[b] = uBasis[b] * d0 + uBasis[b+1] * d1;
                    }}
                    else if (degree == order - 1) {{
                        d0 = degree * gap0;
                        d1 = -degree * gap1;
                        duBasis[b] = uBasis[b] * d0 + uBasis[b+1] * d1;
                        du2Basis[b] = du2Basis[b] * d0 + du2Basis[b+1] * d1;
                    }}
                    uBasis[b] = uBasis[b] * val0 + uBasis[b+1] * val1;
                    b++;
                }}
            }}
        }}
    """

    computeDeltaCode = """
        void ComputeDelta(in vec4 point, in vec3 dPoint, in vec3 d2Point, inout float delta) {
            float zScale;
            float zScale2;
            float zScale3;
            if (uScreenScale.z > 1.0) {
                zScale = 1.0 / (uScreenScale.z - point.z);
                zScale2 = zScale * zScale;
                zScale3 = zScale2 * zScale;
                d2Point.x = uScreenScale.x * (d2Point.x * zScale - 2.0 * dPoint.x * dPoint.z * zScale2 +
                    point.x * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2));
                d2Point.y = uScreenScale.y * (d2Point.y * zScale - 2.0 * dPoint.y * dPoint.z * zScale2 +
                    point.y * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2));
            }
            else {
                d2Point.x *= uScreenScale.x;
                d2Point.y *= uScreenScale.y;
            }
            float sqrtLength = pow(d2Point.x*d2Point.x + d2Point.y*d2Point.y, 0.25);
            delta = sqrtLength < 1.0e-8 ? delta : 1.0 / sqrtLength;
        }
    """

    curveVertexShaderCode = """
        #version 330 core
     
        attribute vec4 aParameters;

        uniform samplerBuffer uSplineData;

        out KnotIndices {
            int u;
        } outData;

        void main() {
            int order;
            int n;

            order = int(texelFetch(uSplineData, 0));
            n = int(texelFetch(uSplineData, 1));

            outData.u = min(gl_InstanceID, n - 1);
            gl_Position = aParameters;
        }
    """

    curveGeometryShaderCode = """
        #version 330 core

        layout( points ) in;
        layout( line_strip, max_vertices = {maxVertices} ) out;

        const int header = 2;

        in KnotIndices {{
            int u;
        }} inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec3 uSplineColor;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor;

        {computeBasisCode}

        {computeDeltaCode}

        void DrawPoint(in int order, in int n, in int m, in float u, inout float deltaU) {{
            float uBasis[{maxBasis}];
            float duBasis[{maxBasis}];
            float du2Basis[{maxBasis}];
            int b;

            ComputeBasis(header, order, n, m, u, uBasis, duBasis, du2Basis);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            vec3 dPoint = vec3(0.0, 0.0, 0.0);
            vec3 d2Point = vec3(0.0, 0.0, 0.0);
            int lastI = header + order + n + 4 * (m + 1);
            b = 0;
            for (int i = header + order + n + 4 * (m + 1 - order); i < lastI; i += 4) {{ // loop from coefficient[m+1-order] to coefficient[m+1]
                point.x += uBasis[b] * texelFetch(uSplineData, i).x;
                point.y += uBasis[b] * texelFetch(uSplineData, i+1).x;
                point.z += uBasis[b] * texelFetch(uSplineData, i+2).x;
                point.w += uBasis[b] * texelFetch(uSplineData, i+3).x;
                dPoint.x += duBasis[b] * texelFetch(uSplineData, i).x;
                dPoint.y += duBasis[b] * texelFetch(uSplineData, i+1).x;
                dPoint.z += duBasis[b] * texelFetch(uSplineData, i+2).x;
                d2Point.x += du2Basis[b] * texelFetch(uSplineData, i).x;
                d2Point.y += du2Basis[b] * texelFetch(uSplineData, i+1).x;
                d2Point.z += du2Basis[b] * texelFetch(uSplineData, i+2).x;
                b++;
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
            int m = inData[0].u + order - 1;
            float u = texelFetch(uSplineData, header + m).x; // knots[m]
            float lastU = texelFetch(uSplineData, header + m + 1).x; // knots[m+1]
            float deltaU = 0.5 * (lastU - u);
            int vertices = 0;

            splineColor = uSplineColor;
            while (u < lastU && vertices < {maxVertices} - 3) {{
                DrawPoint(order, n, m, u, deltaU);
                u += deltaU;
                vertices += 3;
            }}
            DrawPoint(order, n, m, lastU, deltaU);
            EndPrimitive();
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
        self.bind("<ButtonPress-1>", self.RotateStartHandler)
        self.bind("<ButtonRelease-1>", self.RotateEndHandler)
        self.bind("<B1-Motion>", self.RotateDragHandler)
        self.glInitialized = False

    def initgl(self):
        if not self.glInitialized:
            self.maxVertices = min(256, GL_MAX_GEOMETRY_OUTPUT_VERTICES // 2) # Divide by two because each vertex also includes color

            self.computeBasisCode = self.computeBasisCode.format(maxBasis=Spline.maxOrder+1)
            self.curveGeometryShaderCode = self.curveGeometryShaderCode.format(maxVertices=self.maxVertices,
                computeBasisCode=self.computeBasisCode,
                computeDeltaCode=self.computeDeltaCode,
                maxBasis=Spline.maxOrder+1)
            self.curveProgram = shaders.compileProgram(
                shaders.compileShader(self.curveVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(self.curveGeometryShaderCode, GL_GEOMETRY_SHADER), 
                shaders.compileShader(self.fragmentShaderCode, GL_FRAGMENT_SHADER))

            self.splineDataBuffer = glGenBuffers(1)
            self.splineTextureBuffer = glGenTextures(1)
            glBindBuffer(GL_TEXTURE_BUFFER, self.splineDataBuffer)
            glBindTexture(GL_TEXTURE_BUFFER, self.splineTextureBuffer)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.splineDataBuffer)
            maxFloats = 2 + 2 * Spline.maxKnots + 4 * Spline.maxCoefficients * Spline.maxCoefficients
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
            glUseProgram(0)

            #glEnable( GL_DEPTH_TEST )
            glClearColor(1.0, 1.0, 1.0, 0.0)

            self.glInitialized = True

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        xExtent = self.width / self.height
        near = 2.0 - np.sqrt(2)
        far = 2.0 + 2.0 - near
        top = near / 2.0 # Choose frustum that displays [-1,1] in y for z = -2.0
        glFrustum(-top*xExtent, top*xExtent, -top, top, near, far)
        glTranslate(0.0, 0.0, -2.0)
        #glOrtho(-xExtent, xExtent, -1.0, 1.0, -1.0, 1.0)

        self.projection = glGetFloatv(GL_PROJECTION_MATRIX)
        self.screenScale = np.array((0.5 * self.height * self.projection[0,0], 0.5 * self.height * self.projection[1,1], self.projection[3,3]), np.float32)
        glUseProgram(self.curveProgram)
        glUniformMatrix4fv(self.uCurveProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uCurveScreenScale, 1, self.screenScale)
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

def CreateSplineFromMesh(xRange, yRange, zFunction):
    order = (3, 3)
    coefficients = np.zeros((xRange[2], yRange[2], 4), np.float32)
    knots = (np.zeros(xRange[2] + order[0], np.float32), np.zeros(yRange[2] + order[1], np.float32))
    knots[0][:xRange[2]] = np.linspace(xRange[0], xRange[1], xRange[2], dtype=np.float32)[:]
    knots[0][xRange[2]:] = xRange[1]
    knots[1][:yRange[2]] = np.linspace(yRange[0], yRange[1], yRange[2], dtype=np.float32)[:]
    knots[1][yRange[2]:] = yRange[1]
    for i in range(xRange[2]):
        for j in range(yRange[2]):
            coefficients[i, j, 0] = knots[0][i]
            coefficients[i, j, 1] = knots[1][j]
            coefficients[i, j, 2] = zFunction(knots[0][i], knots[1][j])
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