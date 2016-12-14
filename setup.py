import glob
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='maze',
    ext_modules=cythonize(glob.glob('maze/*.pyx')),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'PyQt5',
        'Cython',
        'NumPy',
        'bresenham',
        'docutils',
    ],
    license='GPLv3+',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    ]
)
