import tkinter as tk
from splineOpenGLFrame import SplineOpenGLFrame

class bspyApp(tk.Tk):
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

        self.frame = SplineOpenGLFrame(self, width=500, height=500)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        self.splineList = []