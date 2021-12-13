import numpy as np
import quaternion as quat
import scipy.interpolate as scispline
import tkinter as tk
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from pyopengltk import OpenGLFrame

class Spline:
    def __init__(self, order, knots, coefficients):
        assert len(knots) == order + len(coefficients)
        self.order = order
        self.knots = np.array(knots, np.float32)
        self.coefficients = np.array(coefficients, np.float32)

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
                val0 = 0.0 if gap0 < 1.0e-8 else (x - self.knots[n]) / gap0
                val1 = 0.0 if gap1 < 1.0e-8 else (self.knots[n+degree+1] - x) / gap1
                if degree == self.order - 2:
                    d0 = 0.0 if gap0 < 1.0e-8 else degree / gap0
                    d1 = 0.0 if gap1 < 1.0e-8 else -degree / gap1
                    d2Basis[b] = basis[b] * d0 + basis[b+1] * d1
                elif degree == self.order - 1:
                    d0 = 0.0 if gap0 < 1.0e-8 else degree / gap0
                    d1 = 0.0 if gap1 < 1.0e-8 else -degree / gap1
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
        glBufferSubData(GL_TEXTURE_BUFFER, 0, 4 * 2, np.array((self.order, len(drawCoefficients)), np.float32))
        glBufferSubData(GL_TEXTURE_BUFFER, 4 * 2, 4 * 4*len(drawCoefficients), drawCoefficients)

        glEnableVertexAttribArray(frame.aParameters)
        glDrawArrays(GL_POINTS, 0, 3)
        glDisableVertexAttribArray(frame.aParameters)

        glUseProgram(0)

class SplineOpenGLFrame(OpenGLFrame):

    vertexShaderCode = """
        #version 330 core
     
        attribute vec4 aParameters;
     
        void main() {
            gl_Position = aParameters;
        }
    """
    geometryShaderCode = """
        #version 330 core
        
        layout( points ) in;
        layout( line_strip, max_vertices = 2 ) out;

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec3 uSplineColor;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor;
     
        void main() {
            int order;
            int n;
            int index;
            vec4 coefficient;

            splineColor = uScreenScale;
            splineColor = uSplineColor;
            order = int(texelFetch(uSplineData, 0));
            n = int(texelFetch(uSplineData, 1));
            index = 2 + int(gl_in[0].gl_Position.x) * 4;
            coefficient = vec4(texelFetch(uSplineData, index).x, texelFetch(uSplineData, index+1).x, texelFetch(uSplineData, index+2).x, texelFetch(uSplineData, index+3).x);
            gl_Position = uProjectionMatrix * coefficient;
            EmitVertex();
            gl_Position.x = gl_Position.x + 0.5;
            EmitVertex();
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
            glBufferData(GL_TEXTURE_BUFFER, 4 * (2 + 4*100), None, GL_STATIC_READ)

            glUseProgram(self.program)
            self.aParameters = glGetAttribLocation(self.program, "aParameters")
            self.parameterBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glBufferData(GL_ARRAY_BUFFER, 4 * 4 * 3, np.array([[0,0,0,0], [1,0,0,0], [3,0,0,0]], np.float32), GL_STATIC_DRAW)
            glVertexAttribPointer(self.aParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uProjectionMatrix = glGetUniformLocation(self.program, 'uProjectionMatrix')
            self.uScreenScale = glGetUniformLocation(self.program, 'uScreenScale')
            self.uSplineColor = glGetUniformLocation(self.program, 'uSplineColor')
            self.uSplineData = glGetUniformLocation(self.program, 'uSplineData')
            glUniform1i(self.uSplineData, 0) # 0 is the active texture (default is 0)
            glUseProgram(0)

            self.initialized = True
        
        glClearColor(1.0, 1.0, 1.0, 0.0)

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

        glClear(GL_COLOR_BUFFER_BIT)
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

class PyNubApp(tk.Tk):
    def __init__(self, *args, **kw):
        tk.Tk.__init__(self, *args, **kw)
        self.title('PyNub')

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

if __name__=='__main__':
    app = PyNubApp()
    for i in range(16):
        app.AddSpline(Spline(3, [0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6], [[-1, 0], [-0.5, i/16.0], [0.5, -i/16.0], [1,0]]))
    app.mainloop()