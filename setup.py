"""Setup.py module."""
from setuptools import setup, find_packages, Extension
try:
    import numpy as np
except ImportError:
    import pip
    if hasattr(pip, 'main'):
        pip.main(['install', 'numpy'])
    else:
        import pip._internal as internal
        internal.main(['install', 'numpy'])

setup(
   name='pyrao',
   version='0.9',
   description='Toolkit designed to integrate BSA structures \
                with the most recent world astronomic practices.',
   license="GNUv3",
   author='Alexander',
   # author_email='',
   # url='',
   packages=find_packages(),
   install_requires=['numpy', 'pandas', 'matplotlib', 'angles', 'astropy',
                     'scipy'],
   ext_modules=[Extension("pyrao.integration.cinterp1d",
                          ["pyrao/integration/cinterp1d.c"],
                          include_dirs=[np.get_include()],
                          build_dir="pyrao/integration")]
)
