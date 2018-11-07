"""Setup.py module."""
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='pyrao',
   version='0.1',
   description='Toolkit designed to integrate BSA structures \
                with the most recent world astronomic practices.',
   license="GNUv3",
   long_description=long_description,
   author='Alexander',
   # author_email='',
   # url='',
   packages=['pyrao'],
   install_requires=['numpy', 'pandas', 'matplotlib', 'angles', 'astropy',
                     'scipy']
)
