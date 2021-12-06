import numpy as np
import tkinter as tk
from OpenGL.GL import *
from pyopengltk import OpenGLFrame

class Spline:
    def __init__(self, point):
        self.point = np.array(point)

    def __str__(self):
        return "[{0}, {1}, z: {2}]".format(self.point[0], self.point[1], self.point[2])

class SplineOpenGLFrame(OpenGLFrame):

    def initgl(self):
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.0,1.0,0.0,0.0)

        # setup projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        # setup identity model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        
    def redraw(self):

        glClear(GL_COLOR_BUFFER_BIT)

        glLoadIdentity()

        glBegin(GL_LINES)
        glColor3f(1.0,0.0,3.0)
        glVertex2f(200,100)
        glVertex2f(100,100)
        glVertex2f(100,100)
        glVertex2f(200,200)
        glEnd()
        glFlush()

def ListChanged(event):
    splineList = event.widget
    spline = splineList.get(splineList.curselection())
    print(spline)

if __name__=='__main__':

    root = tk.Tk()
    root.title('PyNub')
    splineList = tk.Listbox(root, height=20)
    splineList.grid(row=0, column=0, sticky=tk.NS)
    splineList.bind('<<ListboxSelect>>', ListChanged)
    splineList.insert(tk.END, Spline([1,2,3]))
    splineList.insert(tk.END, Spline([-1,-2,-3]))
    for i in range(30):
        splineList.insert(tk.END, Spline([i,0,-1]))
    verticalScroll = tk.Scrollbar(root, orient=tk.VERTICAL)
    verticalScroll.grid(row=0, column=1, sticky=tk.NS)
    splineList.configure(yscrollcommand=verticalScroll.set)
    verticalScroll.config(command=splineList.yview)
    frame = SplineOpenGLFrame(root, width=500, height=500)
    frame.grid(row=0, column=2)
    root.mainloop()