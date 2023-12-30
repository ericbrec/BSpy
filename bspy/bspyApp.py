import numpy as np
import tkinter as tk
from tkinter.colorchooser import askcolor
import queue, threading
from bspy import SplineOpenGLFrame
from bspy import DrawableSpline

class _BitCheckbutton(tk.Checkbutton):
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

    See Also
    --------
    `bspyGraphics` : A graphics engine to display splines. It launches a `bspyApp` and issues commands to the app.

    Examples
    --------
    Creates a bspyApp, show some splines, and launches the app (blocks on the main loop).
    >>> app = bspyApp()
    >>> app.show(spline1)
    >>> app.show(spline2)
    >>> app.show(spline3)
    >>> app.mainloop()
    """

    def __init__(self, *args, SplineOpenGLFrame=SplineOpenGLFrame, workQueue=None, **kw):
        tk.Tk.__init__(self, *args, **kw)
        self.title('bspy')
        self.geometry('600x500')

        # Controls on the left
        controls = tk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(controls, text='Adjust Splines', command=self._Adjust).pack(side=tk.BOTTOM, fill=tk.X)
        #tk.Button(controls, text='Empty Splines', command=self.empty).pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(controls, text='Erase Splines', command=self.erase_all).pack(side=tk.BOTTOM, fill=tk.X)

        self.listBox = tk.Listbox(controls, selectmode=tk.MULTIPLE)
        self.listBox.pack(side=tk.LEFT, fill=tk.Y)
        self.listBox.bind('<<ListboxSelect>>', self._ListSelectionChanged)

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
        tk.Radiobutton(buttons, text='Rotate', variable=self.frameMode, value=SplineOpenGLFrame.ROTATE, command=self._ChangeFrameMode).pack(side=tk.LEFT)
        tk.Radiobutton(buttons, text='Pan', variable=self.frameMode, value=SplineOpenGLFrame.PAN, command=self._ChangeFrameMode).pack(side=tk.LEFT)
        tk.Radiobutton(buttons, text='Fly', variable=self.frameMode, value=SplineOpenGLFrame.FLY, command=self._ChangeFrameMode).pack(side=tk.LEFT)
        self.frameMode.set(SplineOpenGLFrame.ROTATE)
        self.scale = tk.Scale(buttons, orient=tk.HORIZONTAL, from_=0, to=1, resolution=0.1, showvalue=0, command=self.frame.SetScale)
        self.scale.pack(side=tk.LEFT)
        self.scale.set(0.5)

        self.splineList = []
        self.splineRadius = 0.0
        self.adjust = None
        self.workQueue = workQueue
        if self.workQueue is not None:
            self.work = {
                "show" : self.show,
                "draw" : self.draw,
                "erase_all" : self.erase_all,
                "empty" : self.empty,
                "set_background_color" : self.set_background_color,
                "update" : self.update
            }
            self.after(1000, self._check_for_work)
    
    def _check_for_work(self):
        """Check queue for calls to make."""
        while not self.workQueue.empty():
            try:
                work = self.workQueue.get_nowait()
            except queue.Empty:
                break
            else:
                if work[0] in self.work:
                    self.work[work[0]](*work[1])
        self.after(200, self._check_for_work)

    def list(self, spline, name = None):
        """List a `Spline` in the listbox. Can be called before app is running."""
        spline = DrawableSpline.make_drawable(spline)
        if name is not None:
            spline.metadata["Name"] = name
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)

    def show(self, spline, name = None):
        """Show a `Spline` in the listbox (calls list method, kept for compatibility). Can be called before app is running."""
        self.list(spline, name)
        
    def draw(self, spline, name = None):
        """Add a `Spline` to the listbox and draw it. Can only be called after the app is running."""
        spline = DrawableSpline.make_drawable(spline)
        if name is not None:
            spline.metadata["Name"] = name
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)
        self.listBox.selection_set(self.listBox.size() - 1)
        self.update()

    def erase_all(self):
        """Stop drawing all splines. Splines remain in the listbox."""
        self.listBox.selection_clear(0, self.listBox.size() - 1)
        self.splineRadius = 0.0
        self.frame.ResetView()
        self.update()

    def empty(self):
        """Stop drawing all splines and remove them from the listbox."""
        self.splineList = []
        self.listBox.delete(0, self.listBox.size() - 1)
        self.splineRadius = 0.0
        self.frame.ResetView()
        self.update()

    def set_background_color(self, r, g=None, b=None, a=None):
        """
        Set the background color.

        Parameters
        ----------
        r : `float`, `int` or array-like of floats or ints
            The red value [0, 1] as a float, [0, 255] as an int, or the rgb or rgba value as floats or ints (default).
        
        g: `float` or `int`
            The green value [0, 1] as a float or [0, 255] as an int.
        
        b: `float` or `int`
            The blue value [0, 1] as a float or [0, 255] as an int.
        
        a: `float`, `int`, or None
            The alpha value [0, 1] as a float or [0, 255] as an int. If `None` then alpha is set to 1.
        """
        self.frame.SetBackgroundColor(r, g, b, a)
        self.frame.Update()

    def update(self):
        """Update the spline draw list, set the default view, reset the bounds, and refresh the frame."""
        self.frame.splineDrawList = []
        gotOne = False
        for item in self.listBox.curselection():
            spline = self.splineList[item]
            coefsAxis = tuple(range(1,spline.nInd + 1))
            if gotOne:
                splineMin = np.minimum(splineMin, spline.coefs[:3].min(axis=coefsAxis))
                splineMax = np.maximum(splineMax, spline.coefs[:3].max(axis=coefsAxis))
            else:
                splineMin = spline.coefs[:3].min(axis=coefsAxis)
                splineMax = spline.coefs[:3].max(axis=coefsAxis)
                gotOne = True
            self.frame.splineDrawList.append(spline)

        if gotOne:
            newRadius = 0.5 * np.max(splineMax - splineMin)
            if newRadius > self.splineRadius:
                self.splineRadius = newRadius
                atDefaultEye = np.allclose(self.frame.eye, self.frame.defaultEye)
                center = 0.5 * (splineMax + splineMin)
                self.frame.SetDefaultView(center + (0.0, 0.0, 3.0 * newRadius), center, (0.0, 1.0, 0.0))
                self.frame.ResetBounds()
                if atDefaultEye:
                    self.frame.ResetView()

        if self.adjust is not None:
            if self.frame.splineDrawList: 
                self.bits.set(self.frame.splineDrawList[0].get_options())
            else:
                self.bits.set(0)
            for button in self.checkButtons.winfo_children():
                button.Update()

        self.frame.Update()

    def _ListSelectionChanged(self, event):
        """Handle when the listbox selection has changed."""
        self.update()
    
    def _ChangeFrameMode(self):
        """Handle when the view mode has changed."""
        self.frame.SetMode(self.frameMode.get())

    def _Adjust(self):
        """Handle when the Adjust button is pressed."""
        if self.adjust is None:
            self.adjust = tk.Toplevel()
            self.adjust.title("Adjust")
            self.adjust.bind('<Destroy>', self._AdjustDestroy)

            self.checkButtons = tk.LabelFrame(self.adjust, text="Style")
            self.checkButtons.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

            self.bits = tk.IntVar()
            if self.frame.splineDrawList:
                self.bits.set(self.frame.splineDrawList[0].get_options())
            else:
                self.bits.set(0)
            _BitCheckbutton(self.checkButtons, DrawableSpline.SHADED, text="Shaded", anchor=tk.W, variable=self.bits, command=self._ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            _BitCheckbutton(self.checkButtons, DrawableSpline.BOUNDARY, text="Boundary", anchor=tk.W, variable=self.bits, command=self._ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            _BitCheckbutton(self.checkButtons, DrawableSpline.ISOPARMS, text="Isoparms", anchor=tk.W, variable=self.bits, command=self._ChangeOptions).pack(side=tk.TOP, fill=tk.X)
            _BitCheckbutton(self.checkButtons, DrawableSpline.HULL, text="Hull", anchor=tk.W, variable=self.bits, command=self._ChangeOptions).pack(side=tk.TOP, fill=tk.X)

            buttons = tk.LabelFrame(self.adjust, text="Color")
            buttons.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
            tk.Button(buttons, text='Fill color', command=self._FillColorChange).pack(side=tk.TOP, fill=tk.X)
            tk.Button(buttons, text='Line color', command=self._LineColorChange).pack(side=tk.TOP, fill=tk.X)
            tk.Button(buttons, text='Dismiss', command=self.adjust.withdraw).pack(side=tk.TOP, fill=tk.X)

            self.adjust.update()
            self.adjust.resizable(False, False)
        else:
            self.adjust.deiconify()

        if self.winfo_x() + self.winfo_width() + 205 <= self.winfo_screenwidth():
            self.adjust.geometry("{width}x{height}+{x}+{y}".format(width=205, height=self.adjust.winfo_height(), x=self.winfo_x() + self.winfo_width(), y=self.winfo_y()))
        else:
            self.adjust.geometry("{width}x{height}+{x}+{y}".format(width=205, height=self.adjust.winfo_height(), x=self.winfo_screenwidth() - 205, y=self.winfo_y()))

    def _AdjustDestroy(self, event):
        """Handle when the adjust dialog is destroyed."""
        self.adjust = None
        self.checkButtons = None
    
    def _ChangeOptions(self, options):
        """Handle when the spline options are changed."""
        for spline in self.frame.splineDrawList:
            spline.set_options(options)
        self.frame.Update()

    def _FillColorChange(self):
        """Handle when the fill color changed."""
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.frame.splineDrawList[0].get_fill_color()
            newColor = askcolor(title="Set spline fill color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.set_fill_color(newColor[0])
                self.frame.Update()

    def _LineColorChange(self):
        """Handle when the line color changed."""
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.frame.splineDrawList[0].get_line_color()
            newColor = askcolor(title="Set spline line color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.set_line_color(newColor[0])
                self.frame.Update()
    
class bspyGraphics:
    """
    A graphics engine to display splines. It launches a `bspyApp` and issues commands to the app.

    Parameters
    ----------
    variableDictionary : `dict`
        A dictionary of variable names, typically `locals()`, used to assign names to splines.

    See Also
    --------
    `bspyApp` : A tkinter app (`tkinter.Tk`) that hosts a `SplineOpenGLFrame`, a listbox full of 
        splines, and a set of controls to adjust and view the selected splines.

    Examples
    --------
    Launch a bspyApp and tell it to draw some splines.
    >>> graphics = bspyGraphics(locals())
    >>> graphics.draw(spline1)
    >>> graphics.draw(spline2)
    >>> graphics.draw(spline3)
    """

    def __init__(self, variableDictionary):
        self.workQueue = queue.Queue()
        self.appThread = threading.Thread(target=self._app_thread)
        self.appThread.start()
        self.variableDictionary = variableDictionary
            
    def _app_thread(self):
        app = bspyApp(workQueue=self.workQueue)
        app.mainloop()        

    def list(self, spline, name = None):
        """Add a `Spline` to the listbox."""
        if name is not None:
            spline.metadata["Name"] = name
        elif "Name" not in spline.metadata:
            for name, value in self.variableDictionary.items():
                if value is spline:
                    spline.metadata["Name"] = name
                    break
        self.workQueue.put(("show", (spline,)))

    def show(self, spline, name = None):
        """Show a `Spline` in the listbox (calls list method, kept for compatibility)."""
        self.list(spline, name)

    def draw(self, spline, name = None):
        """Add a `Spline` to the listbox and draw it."""
        if name is not None:
            spline.metadata["Name"] = name
        elif "Name" not in spline.metadata:
            for name, value in self.variableDictionary.items():
                if value is spline:
                    spline.metadata["Name"] = name
                    break
        self.workQueue.put(("draw", (spline,)))

    def erase_all(self):
        """Stop drawing all splines. Splines remain in the listbox."""
        self.workQueue.put(("erase_all", ()))

    def empty(self):
        """Stop drawing all splines and remove them from the listbox."""
        self.workQueue.put(("empty", ()))

    def set_background_color(self, r, g=None, b=None, a=None):
        """
        Set the background color.

        Parameters
        ----------
        r : `float`, `int` or array-like of floats or ints
            The red value [0, 1] as a float, [0, 255] as an int, or the rgb or rgba value as floats or ints (default).
        
        g: `float` or `int`
            The green value [0, 1] as a float or [0, 255] as an int.
        
        b: `float` or `int`
            The blue value [0, 1] as a float or [0, 255] as an int.
        
        a: `float`, `int`, or None
            The alpha value [0, 1] as a float or [0, 255] as an int. If `None` then alpha is set to 1.
        """
        self.workQueue.put(("set_background_color", (r, g, b, a)))

    def update(self):
        """Update the spline draw list and refresh the frame."""
        self.workQueue.put(("update", ()))
