#!/usr/bin/env python
"""
volumetricinterp interpolates scalar quantities within a 3D
  AMISR field of view and provides estimates of the scalar quantity
  anywhere in the field of view based on volumetric imaging point
  measurements
The full license can be found in LICENSE.txt
"""

import os
import re
import sys
import subprocess
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the package requirements
REQSFILE = os.path.join(here, 'requirements.txt')
with open(REQSFILE, 'r') as f:
    REQUIREMENTS = f.readlines()
REQUIREMENTS = '\n'.join(REQUIREMENTS)

# Do some nice things to help users install on conda.
if sys.version_info[:2] < (3, 0):
    EXCEPTION = OSError
else:
    EXCEPTION = subprocess.builtins.FileNotFoundError
try:
    subprocess.call(['conda', 'install', ' '.join(REQUIREMENTS)])
    REQUIREMENTS = []
except EXCEPTION:
    pass

# Get the readme text
README = os.path.join(here, 'README.rst')
with open(README, 'r') as f:
    READMETXT = f.readlines()
READMETXT = '\n'.join(READMETXT)

# Get version number from __init__.py
regex = "(?<=__version__..\s)\S+"
with open(os.path.join(here,'volumetricinterp/__init__.py'),'r', encoding='utf-8') as f:
    text = f.read()
match = re.findall(regex,text)
version = match[0].strip("'")

# Package description
DESC = "Tool for interpolating 3D scalar parameters from from AMISR data"

#############################################################################
# First, check to make sure we are executing
# 'python setup.py install' from the same directory
# as setup.py (root directory)
#############################################################################
PATH = os.getcwd()
assert('setup.py' in os.listdir(PATH)), \
       "You must execute 'python setup.py install' from within the \
repo root directory."


#############################################################################
# Now execute the setup
#############################################################################
setup(name='volumetricinterp',
      install_requires=REQUIREMENTS,
      setup_requires=REQUIREMENTS,
      version=version,
      description=DESC,
      author="AMISR",
      author_email="leslie.lamarche@sri.com",
      url="https://github.com/amisr/volumetricinterp",
      download_url="https://github.com/amisr/volumetricinterp",
      packages=find_packages(),
      long_description=READMETXT,
      zip_safe=False,
      py_modules=['volumetricinterp'],
      classifiers=["Development Status :: 1.0.0 - Release",
                   "Topic :: Scientific/Engineering",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License (GPL)",
                   "Natural Language :: English",
                   "Programming Language :: Python",
                  ],
      entry_points={
          'console_scripts': [
              'volumetricinterp=volumetricinterp.run_volumetricinterp:main',
        ],
}
      )
