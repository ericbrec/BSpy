import tkinter as tk
from OpenGL.GL import *
from pyopengltk import OpenGLFrame

class frame(OpenGLFrame):

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

if __name__=='__main__':

    root = tk.Tk()
    app = frame(root, width=500, height=500)
    app.pack(fill=tk.BOTH, expand=tk.YES)
    app.mainloop()