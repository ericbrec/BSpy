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
    Creates a bspyApp, adds some splines, and launches the app (blocks on the main loop).
    >>> app = bspyApp()
    >>> app.add_spline(spline1)
    >>> app.add_spline(spline2)
    >>> app.add_spline(spline3)
    >>> app.mainloop()
    """

    def __init__(self, *args, SplineOpenGLFrame=SplineOpenGLFrame, workQueue=None, **kw):
        tk.Tk.__init__(self, *args, **kw)
        self.title('bspy')
        self.geometry('600x500')

        # Controls on the left
        controls = tk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        #tk.Button(controls, text='Empty Splines', command=self.empty).pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(controls, text='Erase Splines', command=self.erase_all).pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(controls, text='Adjust Splines', command=self._Adjust).pack(side=tk.BOTTOM, fill=tk.X)

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
        self.adjust = None
        self.workQueue = workQueue
        if self.workQueue is not None:
            self.work = {
                "add_spline" : self.add_spline,
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

    def add_spline(self, spline):
        """Add a `Spline` to the listbox. Can be called before app is running."""
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)
        
    def draw(self, spline):
        """Add a `Spline` to the listbox and draw it. Can only be called after the app is running."""
        self.splineList.append(spline)
        self.listBox.insert(tk.END, spline)
        self.listBox.selection_set(self.listBox.size() - 1)
        self.update()

    def erase_all(self):
        """Stop drawing all splines. Splines remain in the listbox."""
        self.listBox.selection_clear(0, self.listBox.size() - 1)
        self.update()

    def empty(self):
        """Stop drawing all splines and remove them from the listbox."""
        self.splineList = []
        self.listBox.delete(0, self.listBox.size() - 1)
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
        """Update the spline draw list and refresh the frame."""
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
                self.bits.set(self.frame.splineDrawList[0].options)
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
            spline.SetOptions(options)
        self.frame.Update()

    def _FillColorChange(self):
        """Handle when the fill color changed."""
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.frame.splineDrawList[0].fillColor
            newColor = askcolor(title="Set spline fill color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.SetFillColor(newColor[0])
                self.frame.Update()

    def _LineColorChange(self):
        """Handle when the line color changed."""
        if self.frame.splineDrawList:
            oldColor = 255.0 * self.frame.splineDrawList[0].lineColor
            newColor = askcolor(title="Set spline line color", color="#%02x%02x%02x" % (int(oldColor[0]), int(oldColor[1]), int(oldColor[2])))
            if newColor[0] is not None:
                for spline in self.frame.splineDrawList:
                    spline.SetLineColor(newColor[0])
                self.frame.Update()
    
class bspyGraphics:
    """
    A graphics engine to display splines. It launches a `bspyApp` and issues commands to the app.

    See Also
    --------
    `bspyApp` : A tkinter app (`tkinter.Tk`) that hosts a `SplineOpenGLFrame`, a listbox full of 
        splines, and a set of controls to adjust and view the selected splines.

    Examples
    --------
    Launch a bspyApp and tell it to draw some splines.
    >>> graphics = bspyGraphics()
    >>> graphics.draw(spline1)
    >>> graphics.draw(spline2)
    >>> graphics.draw(spline3)
    """

    def __init__(self):
        self.workQueue = queue.Queue()
        self.appThread = threading.Thread(target=self._app_thread)
        self.appThread.start()
            
    def _app_thread(self):
        app = bspyApp(workQueue=self.workQueue)
        app.mainloop()        

    def add_spline(self, spline):
        """Add a `Spline` to the listbox."""
        self.workQueue.put(("add_spline", (spline,)))

    def draw(self, spline):
        """Add a `Spline` to the listbox and draw it."""
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
