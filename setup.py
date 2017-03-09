#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='neuropredict',
      version='0.1',
      description='Neuroimaging Predictive Analysis',
      long_description=open('README.md').read(),
      author='Pradeep Reddy Raamana',
      author_email='raamana@gmail.com',
      url='https://github.com/raamana/neuropredict',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]), # ['neuropredict'],
      install_requires=['numpy', 'sklearn', 'pyradigm', 'nibabel'],
      entry_points={
                'console_scripts': [ 'neuropredict=neuropredict:run' ],
            },
     )