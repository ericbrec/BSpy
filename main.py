import numpy as np
import tkinter as tk
from OpenGL.GL import *
from pyopengltk import OpenGLFrame

class Spline:
    def __init__(self, point):
        self.point = np.array(point)

    def __str__(self):
        return "[{0}, {1}]".format(self.point[0], self.point[1])
    
    def Draw(self):
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for point in self.point:
            glVertex2f(point[0], point[1])
        glEnd()

class SplineOpenGLFrame(OpenGLFrame):

    def __init__(self, *args, **kw):
        OpenGLFrame.__init__(self, *args, **kw)
        self.animate = 0 # Set to number of milliseconds before showing next frame (0 means no animation)
        self.splineDrawList = []

    def initgl(self):
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)

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
        app.AddSpline(Spline([[-1, i/16.0], [0, -i/16.0], [1,0]]))
    app.mainloop()