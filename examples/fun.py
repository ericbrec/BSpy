import numpy as np
from bspy.spline import Spline
import tkinter as tk
from examples.funOpenGLFrame import FunOpenGLFrame

class FunApp(tk.Tk):
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

        self.frame = FunOpenGLFrame(self, width=500, height=500)
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
    app = FunApp()
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: np.sin(4*np.sqrt(x*x + y*y))))
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x + y*y - 1))
    app.AddSpline(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x - y*y))
    for i in range(16):
        app.AddSpline(Spline((3,), (np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5], np.float32),), np.array([[-1, 0, 0, 1], [-0.5, i/16.0, 0, 1], [0,0,0,1], [0.5, -i/16.0, 0, 1], [1,0,0,1]], np.float32)))
    app.mainloop()