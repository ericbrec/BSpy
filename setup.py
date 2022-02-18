from setuptools import setup

setup(
    name='bspy',
    version='0.0.1',
    author='Eric Brechner',
    author_email='ericbrec@outlook.com',
    url='http://github.com/ericbrec/bspy',
    license='MIT',
    description="Library for manipulating and rendering non-uniform b-splines",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=['bspy'],
    install_requires=['numpy','pyopengl','tk','pyopengltk'],
    keywords=['opengl', 'bspline', 'b-spline', 'nub', 'tkinter'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Environment :: Win32 (MS Windows)',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)