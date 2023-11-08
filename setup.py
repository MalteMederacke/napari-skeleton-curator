#!/usr/bin/env python
from setuptools import setup


setup(
   name='napari-skeleton-curator',
   version='0.1.0',
   author='Kevin Yamauchi, Malte Mederacke',
   author_email='malte.mederacke@bsse.ethz.ch',
#    packages=['napari-skeleton-curator'],
   scripts=[],
   url='',
   license='LICENSE.txt',
   description='Parse graphs from networkx to napari and modify them.',
   long_description=open('README.txt').read(),
   install_requires=[
       "napari",
       "numpy",
       "networkx",
       "pandas",
       "morphosamplers",
       'scikit-image',
       "scipy"


   ],
)