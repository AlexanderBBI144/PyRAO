from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='cinterp1d',
      ext_modules=cythonize("cinterp1d.pyx"),
      include_dirs=[numpy.get_include()])