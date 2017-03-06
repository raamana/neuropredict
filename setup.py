#!/usr/bin/env python

from setuptools import setup

setup(name='neuropredict',
      version='0.1',
      description='Neuroimaging Predictive Analysis',
      long_description=open('README.md').read(),
      author='Pradeep Reddy Raamana',
      author_email='raamana@gmail.com',
      url='https://github.com/raamana/neuropredict',
      packages=['neuropredict'],
      install_requires=['numpy', 'sklearn'],
     )