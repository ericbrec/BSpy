# For new contributors,
 
Tom Grandine and I are using VSCode as our IDE. It’s great, but if you’d prefer a different Python IDE any should work. We’re working with Python 3.0 or higher. We install Python from the Microsoft Store (search for Python).
 
If you use VSCode, here are some extensions you might consider.
* Python essentials:
	* Pylance
	* Python
* Pull requests:
	* GitHub Pull Requests and Issues (Always make changes on a feature branch off main and submit a pull request when you're ready to release them.)
* Jupyter support:
	* Jupyter (Tom loves Jupyter notebooks)
	* Jupyter Keymap (Creates keymaps for notebooks.)
	* Jupyter Notebook Renderers (Visualization for Jupyter.)
* Fun addins:
	* Code Spell Checker (I’ve got this. Fast, easy, custom dictionaries)
	* Markdown All in One (I’ve got this. Useful for authoring .md files.)
	* Awesome Emacs Keymap (VSCode emacs emulation.)

You'll likely want to use a virtual environment (.venv) for the BSpy project so that you don't need worry about version conflicts or machine changes impacting BSpy and your development environment. This is easy in VSCode.
* From the Help menu, select Show All Commands
	* Type: `Python: Create Environment…` (you don't need to type the whole thing, just select it when it comes up)
	* Select `Venv`
	* Select the Microsoft store version of Python

BSpy depends on a few Python packages (found on PyPi.org). You’ll want to pip install them.

    pip install numpy scipy PyOpenGL pyopengltk
 
We also use the following Python packages for testing.

    pip install pytest

Finally, we use the following standard Python packages for building and deploying documentation and PyPI packages. You probably won't do that, so **installing them is optional**.

    pip install pdoc build twine

The repo directory structure is as follows:
* bspy (the library code)
	* `__init__.py` (bspy configuration)
	* `manifold.py` (primary manifold file that contains Manifold and Hyperplane classes)
	* `_spline_domain.py` (methods that change the domain of splines)
	* `_spline_evaluation.py` (methods that evaluate splines)
	* `_spline_fitting.py` (methods that create splines from input data)
	* `_spline_intersection.py` (methods that intersect splines and find their zeros)
	* `_spline_operations.py` (methods that perform spline operations, such as addition and differentiation)
	* `spline.py` (primary spline file that also holds all spline documentation)
	* `viewer,py` (UI app for displaying splines)
	* `splineOpenGLFrame.py` (rendering and shading frame for splines)
* docs (the automatically generated documentation)
* examples (the top-level programs that use bspy)
	* `main.py` (the primary program, if you want to select Tom’s nasty spline, you’ll need to use TomsNasty.py to generate the file and update the path to the file in main.py)
	* `Notebook.py` (a Jupyter notebook that uses bspy)
	* `teapot.py` (a small program to use bspy to draw the Utah teapot)
	* `TomsNasty.py` (a small program to save Tom’s nasty test spline as a file on your desktop—you need to change the path at the bottom of the file)
	* `fun.py` (a silly version of main that might not work)
* test
	* Test functions for spline.py methods (use pytest to run them)

We use [GitHub Flow](https://githubflow.github.io/) as our branching and submission strategy. Always make your changes in a feature branch off main and then submit a pull request when you're ready to release your improvements.

I’m happy to help you with any issues you have getting BSpy working, so I can update these instructions for future collaborators. 😊
 
Thanks,
 
Eric Brechner.
