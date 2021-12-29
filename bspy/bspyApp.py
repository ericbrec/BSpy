import numpy as np
import tkinter as tk
from tkinter.colorchooser import askcolor
from bspy.splineOpenGLFrame import SplineOpenGLFrame

class bspyApp(tk.Tk):
    def __init__(self, *args, **kw):
        tk.Tk.__init__(self, *args, **kw)
        self.title('bspy')

        # Controls on the left
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

        # OpenGL display frame
        self.frame = SplineOpenGLFrame(self, width=500, height=500)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)

        # Adjustment dialog
        self.adjust = tk.Toplevel(self)
        self.adjust.withdraw()
        self.adjust.title("Adjust spline")

        tk.Checkbutton(self.adjust, text="Points", anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        tk.Checkbutton(self.adjust, text="Lines", anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        tk.Checkbutton(self.adjust, text="Shaded", anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        tk.Checkbutton(self.adjust, text="Symbol", anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        tk.Checkbutton(self.adjust, text="Boundary", anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        tk.Checkbutton(self.adjust, text="Isoparms", anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        tk.Checkbutton(self.adjust, text="Label", anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        tk.Button(self.adjust, text='Color', command=self.ColorChange).pack(side=tk.TOP, fill=tk.X)
        tk.Button(self.adjust, text='Done', command=self.adjust.withdraw).pack(side=tk.TOP, fill=tk.X)

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
        self.adjust.deiconify()

    def ColorChange(self):
        if len(self.listBox.curselection()) > 0:
            oldColor = 255.0 * self.splineList[self.listBox.curselection()[0]].color
            newColor = askcolor(title="Set spline color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for item in self.listBox.curselection():
                    self.splineList[item].color = np.array(newColor[0], np.float32) / 255.0
                self.frame.tkExpose(None)