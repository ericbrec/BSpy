import numpy as np
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
    
    def Draw(self):
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for point in self.coefficients:
            glVertex2f(point[0], point[1])
        glEnd()

        tck = (self.knots, self.coefficients.T, self.order-1)
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINE_STRIP)
        for i in range(100):
            x = self.knots[self.order-1] + i * (self.knots[-self.order] - self.knots[self.order-1]) / 99.0
            values = scispline.spalde(x, tck)
            glVertex2f(values[0][0], values[1][0])
        glEnd()

class SplineOpenGLFrame(OpenGLFrame):

    def __init__(self, *args, **kw):
        OpenGLFrame.__init__(self, *args, **kw)
        self.animate = 0 # Set to number of milliseconds before showing next frame (0 means no animation)
        self.splineDrawList = []

    def initgl(self):
        glViewport(0, 0, self.width, self.height)
        glClearColor(1.0, 1.0, 1.0, 0.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
    def redraw(self):

        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        for spline in self.splineDrawList:
            spline.Draw()

        glFlush()

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
        app.AddSpline(Spline(3, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [[-1, 0], [-0.5, i/16.0], [0.5, -i/16.0], [1,0]]))
    app.mainloop()