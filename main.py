import numpy as np
import quaternion as quat
import scipy.interpolate as scispline
import tkinter as tk
from OpenGL.GL import *
from pyopengltk import OpenGLFrame

class Spline:
    def __init__(self, order, knots, coefficients):
        assert len(knots) == order + len(coefficients)
        self.order = order
        self.knots = np.array(knots)
        self.coefficients = np.array(coefficients)

    def __str__(self):
        return "[{0}, {1}]".format(self.coefficients[0], self.coefficients[1])

    def DrawPoint(self, frame, m, x, deltaX):
        basis = np.zeros(self.order + 1)
        dBasis = np.zeros(self.order + 1)
        d2Basis = np.zeros(self.order + 1)
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
        
        point = np.zeros(2)
        dPoint = np.zeros(2)
        d2Point = np.zeros(2)
        b = 0
        for n in range(m+1-self.order, m+1):
            point += basis[b] * self.coefficients[n]
            dPoint += dBasis[b] * self.coefficients[n]
            d2Point += d2Basis[b] * self.coefficients[n]
            b += 1
        
        glVertex2f(point[0], point[1])
        glVertex2f(point[0] - 0.01*dPoint[1], point[1] + 0.01*dPoint[0])
        glVertex2f(point[0], point[1])
        
        dPoint[0] *= 0.5 * frame.width
        dPoint[1] *= 0.5 * frame.height
        length = np.linalg.norm(dPoint)
        #deltaX = deltaX if length < 1.0e-8 else 3.0 / length

        d2Point[0] *= 0.5 * frame.width
        d2Point[1] *= 0.5 * frame.height
        length = np.linalg.norm(d2Point)
        deltaX = deltaX if length < 1.0e-8 else 1.0 / np.sqrt(length)

        return deltaX
    
    def Draw(self, frame):
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINE_STRIP)
        for point in self.coefficients:
            glVertex2f(point[0], point[1])
        glEnd()

        tck = (self.knots, self.coefficients.T, self.order-1)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for i in range(100):
            x = self.knots[self.order-1] + i * (self.knots[-self.order] - self.knots[self.order-1]) / 99.0
            values = scispline.spalde(x, tck)
            glVertex2f(values[0][0], values[1][0])
        glEnd()

        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for m in range(self.order-1, len(self.knots)-self.order):
            x = self.knots[m]
            deltaX = 0.5 * (self.knots[m+1] - x)
            vertices = 0
            while x < self.knots[m+1] and vertices < GL_MAX_GEOMETRY_OUTPUT_VERTICES:
                deltaX = self.DrawPoint(frame, m, x, deltaX)
                x += deltaX
                vertices += 1
        self.DrawPoint(frame, m, self.knots[m+1], deltaX)
        glEnd()

class SplineOpenGLFrame(OpenGLFrame):

    def __init__(self, *args, **kw):
        OpenGLFrame.__init__(self, *args, **kw)
        self.animate = 0 # Set to number of milliseconds before showing next frame (0 means no animation)
        self.splineDrawList = []
        self.currentQ = quat.one
        self.lastQ = quat.one
        self.origin = None

    def initgl(self):
        glViewport(0, 0, self.width, self.height)
        glClearColor(1.0, 1.0, 1.0, 0.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.bind("<ButtonPress-1>", self.RotateStartHandler)
        self.bind("<ButtonRelease-1>", self.RotateEndHandler)
        self.bind("<B1-Motion>", self.RotateDragHandler)
        
        print(glGetString(GL_VERSION))
        print(glGetString(GL_SHADING_LANGUAGE_VERSION))
        
    def redraw(self):

        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        rotation33 = quat.as_rotation_matrix(self.currentQ * self.lastQ)
        rotation44 = np.identity(4)
        rotation44[0:3,0:3] = rotation33
        glMultMatrixf(rotation44)

        for spline in self.splineDrawList:
            spline.Draw(self)

        glFlush()

    def ProjectToSphere(self, point):
        length = np.linalg.norm(point)
        if length <= 0.7071: # 1/sqrt(2)
            projection = np.array((point[0], point[1], np.sqrt(1.0 - length * length)))
        else:
            projection = np.array((point[0], point[1], 0.5 / length))
            projection = projection / np.linalg.norm(projection)
        return projection

    def RotateStartHandler(self, event):
        self.origin = np.array(((2.0 * event.x - self.width)/self.height, (self.height - 2.0 * event.y)/self.height))

    def RotateDragHandler(self, event):
        if self.origin is not None:
            point = np.array(((2.0 * event.x - self.width)/self.height, (self.height - 2.0 * event.y)/self.height))
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