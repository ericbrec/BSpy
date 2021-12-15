import numpy as np
import quaternion as quat
import scipy.interpolate as scispline
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

    def DrawPoint(self, screenScale, drawCoefficients, m, x, deltaX):
        basis = np.zeros(self.order + 1, np.float32)
        dBasis = np.zeros(self.order + 1, np.float32)
        d2Basis = np.zeros(self.order + 1, np.float32)
        basis[self.order-1] = 1.0
        for degree in range(1, self.order):
            b = self.order - degree - 1
            for n in range(m-degree, m+1):
                gap0 = self.knots[n+degree] - self.knots[n]
                gap1 = self.knots[n+degree+1] - self.knots[n+1]
                gap0 = 0.0 if gap0 < 1.0e-8 else 1.0 / gap0
                gap1 = 0.0 if gap1 < 1.0e-8 else 1.0 / gap1
                val0 = (x - self.knots[n]) * gap0
                val1 = (self.knots[n+degree+1] - x) * gap1
                if degree == self.order - 2:
                    d0 = degree * gap0
                    d1 = -degree * gap1
                    d2Basis[b] = basis[b] * d0 + basis[b+1] * d1
                elif degree == self.order - 1:
                    d0 = degree * gap0
                    d1 = -degree * gap1
                    dBasis[b] = basis[b] * d0 + basis[b+1] * d1
                    d2Basis[b] = d2Basis[b] * d0 + d2Basis[b+1] * d1
                basis[b] = basis[b] * val0 + basis[b+1] * val1
                b += 1
        
        point = np.zeros(4, np.float32)
        dPoint = np.zeros(4, np.float32)
        d2Point = np.zeros(4, np.float32)
        b = 0
        for n in range(m+1-self.order, m+1):
            point += basis[b] * drawCoefficients[n]
            dPoint += dBasis[b] * drawCoefficients[n]
            d2Point += d2Basis[b] * drawCoefficients[n]
            b += 1
        
        glVertex3f(point[0], point[1], point[2])
        glVertex3f(point[0] - 0.01*dPoint[1], point[1] + 0.01*dPoint[0], point[2])
        glVertex3f(point[0], point[1], point[2])
        
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
        deltaX = deltaX if sqrtLength < 1.0e-8 else 1.0 / sqrtLength

        return deltaX
    
    def Draw(self, frame, transform):
        drawCoefficients = np.zeros((len(self.coefficients),4), np.float32)
        drawCoefficients[:,:self.coefficients.shape[1]] = self.coefficients[:,:]
        drawCoefficients[:,3] = 1.0
        drawCoefficients = drawCoefficients @ transform

        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINE_STRIP)
        for point in drawCoefficients:
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        tck = (self.knots, drawCoefficients.T, self.order-1)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for i in range(100):
            x = self.knots[self.order-1] + i * (self.knots[-self.order] - self.knots[self.order-1]) / 99.0
            values = scispline.spalde(x, tck)
            glVertex3f(values[0][0], values[1][0], values[2][0])
        glEnd()

        glColor3f(0.0, 1.0, 0.0)
        for m in range(self.order-1, len(self.knots)-self.order):
            x = self.knots[m]
            deltaX = 0.5 * (self.knots[m+1] - x)
            vertices = 0
            glBegin(GL_LINE_STRIP)
            while x < self.knots[m+1] and vertices < GL_MAX_GEOMETRY_OUTPUT_VERTICES - 1:
                deltaX = self.DrawPoint(frame.screenScale, drawCoefficients, m, x, deltaX)
                x += deltaX
                vertices += 1
            self.DrawPoint(frame.screenScale, drawCoefficients, m, self.knots[m+1], deltaX)
            glEnd()

        glUseProgram(frame.program)

        glUniform3f(frame.uSplineColor, 1.0, 0.0, 1.0)

        glBindBuffer(GL_TEXTURE_BUFFER, frame.splineDataBuffer)
        offset = 0
        size = 4 * 2
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((self.order, len(drawCoefficients)), np.float32))
        offset += size
        size = 4 * len(self.knots)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, self.knots)
        offset += size
        size = 4 * 4 * len(drawCoefficients)
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, drawCoefficients)

        glEnableVertexAttribArray(frame.aParameters)
        glDrawArraysInstanced(GL_POINTS, 0, 1, len(drawCoefficients) - self.order + 1)
        glDisableVertexAttribArray(frame.aParameters)

        glUseProgram(0)

class SplineOpenGLFrame(OpenGLFrame):

    vertexShaderCode = """
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

    geometryShaderCode = """
        #version 330 core
        
        layout( points ) in;
        layout( line_strip, max_vertices = 128 ) out; // 0.5 * GL_MAX_GEOMETRY_OUTPUT_VERTICES (vertex + color)

        const int header = 2;

        in KnotIndices {
            int u;
        } inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec3 uSplineColor;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor;

        void DrawPoint(in int order, in int n, in int m, in float x, inout float deltaX) {
            float basis[10] = float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            float dBasis[10] = float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            float d2Basis[10] = float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            int b;

            basis[order-1] = 1.0;
            for (int degree = 1; degree < order; degree++) {
                b = order - degree - 1;
                for (int i = m - degree; i < m + 1; i++) {
                    float gap0 = texelFetch(uSplineData, header + i + degree).x - texelFetch(uSplineData, header + i).x; // knots[i+degree] - knots[i]
                    float gap1 = texelFetch(uSplineData, header + i + degree + 1).x - texelFetch(uSplineData, header + i + 1).x; // knots[i+degree+1] - knots[i+1]
                    float val0 = 0.0;
                    float val1 = 0.0;
                    float d0 = 0.0;
                    float d1 = 0.0;

                    gap0 = gap0 < 1.0e-8 ? 0.0 : 1.0 / gap0;
                    gap1 = gap1 < 1.0e-8 ? 0.0 : 1.0 / gap1;
                    val0 = (x - texelFetch(uSplineData, header + i).x) * gap0; // (x - knots[i]) * gap0
                    val1 = (texelFetch(uSplineData, header + i + degree + 1).x - x) * gap1; // (knots[i+degree+1] - x) * gap1
                    if (degree == order - 2) {
                        d0 = degree * gap0;
                        d1 = -degree * gap1;
                        d2Basis[b] = basis[b] * d0 + basis[b+1] * d1;
                    }
                    else if (degree == order - 1) {
                        d0 = degree * gap0;
                        d1 = -degree * gap1;
                        dBasis[b] = basis[b] * d0 + basis[b+1] * d1;
                        d2Basis[b] = d2Basis[b] * d0 + d2Basis[b+1] * d1;
                    }
                    basis[b] = basis[b] * val0 + basis[b+1] * val1;
                    b++;
                }
            }
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            vec4 dPoint = vec4(0.0, 0.0, 0.0, 0.0);
            vec4 d2Point = vec4(0.0, 0.0, 0.0, 0.0);
            int lastI = header + order + n + 4 * (m + 1);
            b = 0;
            for (int i = header + order + n + 4 * (m + 1 - order); i < lastI; i += 4) { // loop from coefficient[m+1-order] to coefficient[m+1]
                point.x += basis[b] * texelFetch(uSplineData, i).x;
                point.y += basis[b] * texelFetch(uSplineData, i+1).x;
                point.z += basis[b] * texelFetch(uSplineData, i+2).x;
                point.w += basis[b] * texelFetch(uSplineData, i+3).x;
                dPoint.x += dBasis[b] * texelFetch(uSplineData, i).x;
                dPoint.y += dBasis[b] * texelFetch(uSplineData, i+1).x;
                dPoint.z += dBasis[b] * texelFetch(uSplineData, i+2).x;
                dPoint.w += dBasis[b] * texelFetch(uSplineData, i+3).x;
                d2Point.x += d2Basis[b] * texelFetch(uSplineData, i).x;
                d2Point.y += d2Basis[b] * texelFetch(uSplineData, i+1).x;
                d2Point.z += d2Basis[b] * texelFetch(uSplineData, i+2).x;
                d2Point.w += d2Basis[b] * texelFetch(uSplineData, i+3).x;
                b++;
            }

            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            gl_Position = uProjectionMatrix * vec4(point.x - 0.01*dPoint.y, point.y + 0.01*dPoint.x, point.z, point.w);
            EmitVertex();
            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            
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
            deltaX = sqrtLength < 1.0e-8 ? deltaX : 1.0 / sqrtLength;
        }

        void main() {
            int order = int(texelFetch(uSplineData, 0).x);
            int n = int(texelFetch(uSplineData, 1).x);
            int m = inData[0].u + order - 1;
            float x = texelFetch(uSplineData, header + m).x; // knots[m]
            float lastX = texelFetch(uSplineData, header + m + 1).x; // knots[m+1]
            float deltaX = 0.5 * (lastX - x);
            int vertices = 0;

            splineColor = uSplineColor;
            while (x < lastX && vertices < 127) {
                DrawPoint(order, n, m, x, deltaX);
                x += deltaX;
                vertices += 1;
            }
            DrawPoint(order, n, m, lastX, deltaX);
            EndPrimitive();
        }
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
        self.initialized = False

    def initgl(self):
        if not self.initialized:
            self.bind("<ButtonPress-1>", self.RotateStartHandler)
            self.bind("<ButtonRelease-1>", self.RotateEndHandler)
            self.bind("<B1-Motion>", self.RotateDragHandler)
            
            print(glGetString(GL_VERSION))
            print(glGetString(GL_SHADING_LANGUAGE_VERSION))

            self.vertexShader = shaders.compileShader(self.vertexShaderCode, GL_VERTEX_SHADER)
            self.geometryShader = shaders.compileShader(self.geometryShaderCode, GL_GEOMETRY_SHADER)
            self.fragmentShader = shaders.compileShader(self.fragmentShaderCode, GL_FRAGMENT_SHADER)
            self.program = shaders.compileProgram(self.vertexShader, self.geometryShader, self.fragmentShader)

            self.splineDataBuffer = glGenBuffers(1)
            self.splineTextureBuffer = glGenTextures(1)
            glBindBuffer(GL_TEXTURE_BUFFER, self.splineDataBuffer)
            glBindTexture(GL_TEXTURE_BUFFER, self.splineTextureBuffer)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.splineDataBuffer)
            maxFloats = 2 + 2 * Spline.maxKnots + 4 * Spline.maxCoefficients * Spline.maxCoefficients
            glBufferData(GL_TEXTURE_BUFFER, 4 * maxFloats, None, GL_STATIC_READ)

            glUseProgram(self.program)
            self.aParameters = glGetAttribLocation(self.program, "aParameters")
            self.parameterBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glBufferData(GL_ARRAY_BUFFER, 4 * 4, np.array([0,0,0,0], np.float32), GL_STATIC_DRAW)
            glVertexAttribPointer(self.aParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uProjectionMatrix = glGetUniformLocation(self.program, 'uProjectionMatrix')
            self.uScreenScale = glGetUniformLocation(self.program, 'uScreenScale')
            self.uSplineColor = glGetUniformLocation(self.program, 'uSplineColor')
            self.uSplineData = glGetUniformLocation(self.program, 'uSplineData')
            glUniform1i(self.uSplineData, 0) # 0 is the active texture (default is 0)
            glUseProgram(0)

            glEnable( GL_DEPTH_TEST )
            glClearColor(1.0, 1.0, 1.0, 0.0)
            self.initialized = True

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        xExtent = self.width / self.height
        glFrustum(-0.5*xExtent, 0.5*xExtent, -0.5, 0.5, 1.0, 3.0)
        glTranslate(0.0, 0.0, -2.0)
        #glOrtho(-xExtent, xExtent, -1.0, 1.0, -1.0, 1.0)

        self.projection = glGetFloatv(GL_PROJECTION_MATRIX)
        self.screenScale = np.array((0.5 * self.height * self.projection[0,0], 0.5 * self.height * self.projection[1,1], self.projection[3,3]), np.float32)
        glUseProgram(self.program)
        glUniformMatrix4fv(self.uProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uScreenScale, 1, self.screenScale)
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
    spline = CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: np.sin(4*np.sqrt(x*x + y*y)))
    app.AddSpline(spline)
    spline = CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x + y*y - 1)
    app.AddSpline(spline)
    #for i in range(16):
        #app.AddSpline(Spline(3, [0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6], [[-1, 0], [-0.5, i/16.0], [0.5, -i/16.0], [1,0]]))
    app.mainloop()