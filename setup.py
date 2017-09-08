#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='neuropredict',
      version='0.2.5',
      description='Neuroimaging Predictive Analysis',
      long_description='Neuroimaging Predictive Analysis; neuropredict',
      author='Pradeep Reddy Raamana',
      author_email='raamana@gmail.com',
      url='https://github.com/raamana/neuropredict',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]), # ['neuropredict'],
      install_requires=['numpy', 'scikit-learn', 'pyradigm', 'nibabel', 'scipy', 'matplotlib', 'setuptools'],
      classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 2.7',
          ],
      entry_points={
          "console_scripts": [
              "neuropredict=neuropredict.__main__:main",
          ]
      }

     )
