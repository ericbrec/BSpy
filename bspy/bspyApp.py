import numpy as np
import tkinter as tk
from tkinter.colorchooser import askcolor
from bspy.splineOpenGLFrame import SplineOpenGLFrame

class bspyApp(tk.Tk):
    def __init__(self, *args, **kw):
        tk.Tk.__init__(self, *args, **kw)
        self.title('bspy')

        controls = tk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        adjustButton = tk.Button(controls, text='Adjust', command=self.Adjust)
        adjustButton.pack(side=tk.BOTTOM, fill=tk.X)

        self.listBox = tk.Listbox(controls, selectmode=tk.MULTIPLE)
        self.listBox.pack(side=tk.LEFT, fill=tk.Y)
        self.listBox.bind('<<ListboxSelect>>', self.ListChanged)

        verticalScroll = tk.Scrollbar(controls, orient=tk.VERTICAL)
        verticalScroll.pack(side=tk.LEFT, fill=tk.Y)
        self.listBox.configure(yscrollcommand=verticalScroll.set)
        verticalScroll.config(command=self.listBox.yview)

        self.frame = SplineOpenGLFrame(self, width=500, height=500)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)

        self.splineList = []

    def AddSpline(self, spline):
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)

    def ListChanged(self, event):
        self.frame.splineDrawList = []
        for item in self.listBox.curselection():
            self.frame.splineDrawList.append(self.splineList[item])
        self.frame.tkExpose(None)

    def Adjust(self):
        if len(self.listBox.curselection()) > 0:
            oldColor = 255.0 * self.splineList[self.listBox.curselection()[0]].color
            newColor = askcolor(title="Set spline color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for item in self.listBox.curselection():
                    self.splineList[item].color = np.array(newColor[0], np.float32) / 255.0
                self.frame.tkExpose(None)