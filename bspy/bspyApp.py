import numpy as np
import tkinter as tk
from tkinter.colorchooser import askcolor
from bspy.splineOpenGLFrame import SplineOpenGLFrame
from bspy.spline import Spline

class BitCheckbutton(tk.Checkbutton):
    def __init__(self, parent, bitNumber, **kw):
        self.bitNumber = bitNumber
        self.variable = kw.get("variable")
        self.command = kw.get("command")
        self.var = tk.IntVar()
        self.var.set(1 if self.variable.get() & (1 << self.bitNumber) else 0)
        kw["variable"] = self.var
        kw["onvalue"] = 1
        kw["offvalue"] = 0
        kw["command"] = self.Command
        tk.Checkbutton.__init__(self, parent, **kw)
    
    def Command(self):
        if self.var.get() == 1:
            self.variable.set(self.variable.get() | (1 << self.bitNumber))
        else:
            self.variable.set(self.variable.get() & ~(1 << self.bitNumber))
        self.command(self.variable.get())

    def Update(self):
        self.var.set(1 if self.variable.get() & (1 << self.bitNumber) else 0)

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

        self.splineList = []
        self.adjust = None

    def AddSpline(self, spline):
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)

    def ListChanged(self, event):
        self.frame.splineDrawList = []
        for item in self.listBox.curselection():
            self.frame.splineDrawList.append(self.splineList[item])

        if self.adjust is not None:
            if self.frame.splineDrawList: 
                self.bits.set(self.frame.splineDrawList[0].options)
            else:
                self.bits.set(0)
            for button in self.checkButtons.winfo_children():
                button.Update()

        self.frame.tkExpose(None)

    def Adjust(self):
        if self.adjust is None:
            self.adjust = tk.Toplevel()
            self.adjust.title("Adjust")

            self.checkButtons = tk.LabelFrame(self.adjust, text="Style")
            self.checkButtons.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

            self.bits = tk.IntVar()
            if self.frame.splineDrawList:
                self.bits.set(self.frame.splineDrawList[0].options)
            else:
                self.bits.set(0)
            BitCheckbutton(self.checkButtons, Spline.POINTS, text="Points", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, Spline.LINES, text="Lines", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, Spline.SHADED, text="Shaded", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, Spline.SYMBOL, text="Symbol", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, Spline.BOUNDARY, text="Boundary", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, Spline.ISOPARMS, text="Isoparms", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, Spline.CONTOUR, text="Contour", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, Spline.LABEL, text="Label", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)

            buttons = tk.LabelFrame(self.adjust, text="Color")
            buttons.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
            tk.Button(buttons, text='Fill color', command=self.FillColorChange).pack(side=tk.TOP, fill=tk.X)
            tk.Button(buttons, text='Line color', command=self.LineColorChange).pack(side=tk.TOP, fill=tk.X)
            tk.Button(buttons, text='Dismiss', command=self.adjust.withdraw).pack(side=tk.TOP, fill=tk.X)

            self.adjust.minsize(205, self.adjust.winfo_height())
            self.adjust.bind('<Destroy>', self.AdjustDestroy)

        else:
            self.adjust.deiconify()

    def AdjustDestroy(self, event):
        self.adjust = None
        self.checkButtons = None
    
    def ChangeOptions(self, options):
        for spline in self.frame.splineDrawList:
            spline.options = options
        self.frame.tkExpose(None)

    def FillColorChange(self):
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.splineList[0].fillColor
            newColor = askcolor(title="Set spline color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.fillColor = np.array(newColor[0], np.float32) / 255.0
                self.frame.tkExpose(None)

    def LineColorChange(self):
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.splineList[0].lineColor
            newColor = askcolor(title="Set spline color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.lineColor = np.array(newColor[0], np.float32) / 255.0
                self.frame.tkExpose(None)