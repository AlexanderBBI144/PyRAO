from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='cinterp1d',
      ext_modules=cythonize("cinterp1d.pyx",
                            include_path=[numpy.get_include()]))
