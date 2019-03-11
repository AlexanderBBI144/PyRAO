"""Setup.py module."""
import sys
from setuptools import setup, find_packages, Extension, command
from setuptools.command.install import install

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

class InstallCommand(install):
    user_options = install.user_options + [
        ('no-cython-build', None, None),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.no_cython_build = None

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global no_cython_build
        no_cython_build = self.no_cython_build
        install.run(self)

configuration = dict(
    name='pyrao',
    version='1.1',
    description='Toolkit designed to integrate BSA structures '\
                'with the most recent world astronomic practices.',
    license="GNUv3",
    author='Alexander S.',
    # author_email='',
    url='https://github.com/AlexanderBBI144/PyRAO/',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'angles', 'astropy',
                      'scipy', 'plotly', 'dash', 'flask_caching'],
    zip_safe=False,
    cmdclass={
        'install': InstallCommand,
    }
)

if '--no-cython-build' in sys.argv:
    print('Installing for win')
    configuration['include_package_data'] = True
else:
    print('Installing for all')
    exclude = ['*cinterp1d']
    configuration['ext_modules'] = [Extension("pyrao.integration.cinterp1d",
                                              ["pyrao/integration/cinterp1d.c"],
                                              include_dirs=[np.get_include()],
                                              build_dir="pyrao/integration")]

setup(**configuration)
