"""Setup.py module."""
from setuptools import setup, find_packages, Extension
import platform
import warnings

try:
    import numpy as np
except ImportError:
    import pip
    if hasattr(pip, 'main'):
        pip.main(['install', 'numpy'])
    else:
        import pip._internal as internal
        internal.main(['install', 'numpy'])
        import numpy as np

configuration = dict(
    name='pyrao',
    version='0.9',
    description='Toolkit designed to integrate BSA structures \
                 with the most recent world astronomic practices.',
    license="GNUv3",
    author='Alexander Somov',
    # author_email='',
    url='https://github.com/AlexanderBBI144/PyRAO/',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'angles', 'astropy',
                      'scipy'],
    zip_safe=False,
)

from distutils.core import setup, Extension
if platform.system() == "Windows":
    configuration['include_package_data'] = True
else:
    configuration['packages'] = find_packages(exclude=['cinterp1d.py'])
    configuration['ext_modules'] = [Extension("pyrao.integration.cinterp1d",
                                              ["pyrao/integration/cinterp1d.c"],
                                                include_dirs=[np.get_include()],
                                                build_dir="pyrao/integration")]
print(configuration)

setup(**configuration)
