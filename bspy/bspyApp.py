import numpy as np
import tkinter as tk
from tkinter.colorchooser import askcolor
from bspy import SplineOpenGLFrame
from bspy import DrawableSpline

class BitCheckbutton(tk.Checkbutton):
    """A tkinter `CheckButton` that gets/sets its variable based on a given `bitmask`."""
    def __init__(self, parent, bitMask, **kw):
        self.bitMask = bitMask
        self.variable = kw.get("variable")
        self.command = kw.get("command")
        self.var = tk.IntVar()
        self.var.set(1 if self.variable.get() & self.bitMask else 0)
        kw["variable"] = self.var
        kw["onvalue"] = 1
        kw["offvalue"] = 0
        kw["command"] = self.Command
        tk.Checkbutton.__init__(self, parent, **kw)
    
    def Command(self):
        """
        Handles when the checkbutton is pushed, updating the variable based on the 
        bitmask and then calling the checkbutton command.
        """
        if self.var.get() == 1:
            self.variable.set(self.variable.get() | self.bitMask)
        else:
            self.variable.set(self.variable.get() & ~self.bitMask)
        self.command(self.variable.get())

    def Update(self):
        """Updates checkbutton state."""
        self.var.set(1 if self.variable.get() & self.bitMask else 0)

class bspyApp(tk.Tk):
    """
    A tkinter app (`tkinter.Tk`) that hosts a `SplineOpenGLFrame`, a listbox full of 
    splines, and a set of controls to adjust and view the selected splines.

    Example
    -------
    >>> app = bspyApp()
    >>> app.AddSpline(spline1)
    >>> app.AddSpline(spline2)
    >>> app.AddSpline(spline3)
    >>> app.mainloop()
    """
    def __init__(self, *args, SplineOpenGLFrame=SplineOpenGLFrame, **kw):
        tk.Tk.__init__(self, *args, **kw)
        self.title('bspy')
        self.geometry('600x500')

        # Controls on the left
        controls = tk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        adjustButton = tk.Button(controls, text='Adjust Splines', command=self.Adjust)
        adjustButton.pack(side=tk.BOTTOM, fill=tk.X)

        self.listBox = tk.Listbox(controls, selectmode=tk.MULTIPLE)
        self.listBox.pack(side=tk.LEFT, fill=tk.Y)
        self.listBox.bind('<<ListboxSelect>>', self.ListChanged)

        verticalScroll = tk.Scrollbar(controls, orient=tk.VERTICAL)
        verticalScroll.pack(side=tk.LEFT, fill=tk.Y)
        self.listBox.configure(yscrollcommand=verticalScroll.set)
        verticalScroll.config(command=self.listBox.yview)

        # Controls on the right
        controls = tk.Frame(self)
        controls.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)

        self.frame = SplineOpenGLFrame(controls)
        self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        buttons = tk.Frame(controls)
        buttons.pack(side=tk.BOTTOM)
        tk.Button(buttons, text='Reset View', command=self.frame.Reset).pack(side=tk.LEFT)
        self.frameMode = tk.IntVar()
        tk.Radiobutton(buttons, text='Rotate', variable=self.frameMode, value=SplineOpenGLFrame.ROTATE, command=self.ChangeFrameMode).pack(side=tk.LEFT)
        tk.Radiobutton(buttons, text='Pan', variable=self.frameMode, value=SplineOpenGLFrame.PAN, command=self.ChangeFrameMode).pack(side=tk.LEFT)
        tk.Radiobutton(buttons, text='Fly', variable=self.frameMode, value=SplineOpenGLFrame.FLY, command=self.ChangeFrameMode).pack(side=tk.LEFT)
        self.frameMode.set(SplineOpenGLFrame.ROTATE)
        self.scale = tk.Scale(buttons, orient=tk.HORIZONTAL, from_=0, to=1, resolution=0.1, showvalue=0, command=self.frame.SetScale)
        self.scale.pack(side=tk.LEFT)
        self.scale.set(0.5)

        self.splineList = []
        self.adjust = None

    def AddSpline(self, spline):
        """Add a `DrawableSpline` to the listbox."""
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)

    def RemoveSpline(self, spline):
        """Remove a `DrawableSpline` from the listbox."""
        self.splineList.remove(spline)
        self.listBox.delete(self.listBox.get(0, tk.END).index(spline))

    def ListChanged(self, event):
        """Handle when the listbox selection has changed."""
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

        self.frame.Update()
    
    def ChangeFrameMode(self):
        """Handle when the view mode has changed."""
        self.frame.SetMode(self.frameMode.get())

    def Adjust(self):
        """Handle when the Adjust button is pressed."""
        if self.adjust is None:
            self.adjust = tk.Toplevel()
            self.adjust.title("Adjust")
            self.adjust.bind('<Destroy>', self.AdjustDestroy)

            self.checkButtons = tk.LabelFrame(self.adjust, text="Style")
            self.checkButtons.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

            self.bits = tk.IntVar()
            if self.frame.splineDrawList:
                self.bits.set(self.frame.splineDrawList[0].options)
            else:
                self.bits.set(0)
            BitCheckbutton(self.checkButtons, DrawableSpline.SHADED, text="Shaded", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, DrawableSpline.BOUNDARY, text="Boundary", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, DrawableSpline.ISOPARMS, text="Isoparms", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            BitCheckbutton(self.checkButtons, DrawableSpline.HULL, text="Hull", anchor=tk.W, variable=self.bits, command=self.ChangeOptions).pack(side=tk.TOP, fill=tk.X)

            buttons = tk.LabelFrame(self.adjust, text="Color")
            buttons.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
            tk.Button(buttons, text='Fill color', command=self.FillColorChange).pack(side=tk.TOP, fill=tk.X)
            tk.Button(buttons, text='Line color', command=self.LineColorChange).pack(side=tk.TOP, fill=tk.X)
            tk.Button(buttons, text='Dismiss', command=self.adjust.withdraw).pack(side=tk.TOP, fill=tk.X)

            self.adjust.update()
            self.adjust.resizable(False, False)
        else:
            self.adjust.deiconify()

        if self.winfo_x() + self.winfo_width() + 205 <= self.winfo_screenwidth():
            self.adjust.geometry("{width}x{height}+{x}+{y}".format(width=205, height=self.adjust.winfo_height(), x=self.winfo_x() + self.winfo_width(), y=self.winfo_y()))
        else:
            self.adjust.geometry("{width}x{height}+{x}+{y}".format(width=205, height=self.adjust.winfo_height(), x=self.winfo_screenwidth() - 205, y=self.winfo_y()))

    def AdjustDestroy(self, event):
        """Handle when the adjust dialog is destroyed."""
        self.adjust = None
        self.checkButtons = None
    
    def ChangeOptions(self, options):
        """Handle when the spline options are changed."""
        for spline in self.frame.splineDrawList:
            spline.SetOptions(options)
        self.frame.Update()

    def FillColorChange(self):
        """Handle when the fill color changed."""
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.frame.splineDrawList[0].fillColor
            newColor = askcolor(title="Set spline fill color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.SetFillColor(newColor[0])
                self.frame.Update()

    def LineColorChange(self):
        """Handle when the line color changed."""
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.frame.splineDrawList[0].lineColor
            newColor = askcolor(title="Set spline line color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.SetLineColor(newColor[0])
                self.frame.Update()