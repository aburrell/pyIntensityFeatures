.. _install:

Installation
============

The following instructions will allow you to install pyIntensityFeatures


.. _install-prereq:

Prerequisites
-------------

pyIntensityFeatures uses common Python modules, as well as modules developed by
and for the Space Physics community.  This module officially supports
Python 3.6+.

 ============== =================
 Common modules Community modules
 ============== =================
  numpy         aacgmv2
  pandas        apexpy (optional)
  scipy
  xarray
 ============== =================


.. _install-opt:


Installation Options
--------------------


.. _install-opt-pip:

PyPi
^^^^
All public pyIntensityFeatures releases will be made available through the
PyPi package manager at a future time.
::


   pip install pyIntensityFeatures



.. _install-opt-git:

Git Repository
^^^^^^^^^^^^^^
You can keep up to date with the latest changes at the git repository.

1. Clone the git repository
::


   git clone https://github.com/PACKAGE_REPOSITORY


2. Install pyIntensityFeatures:
   Change directories into the repository folder and run the pyproject.toml
   file. There are a few ways you can do this:

   A. Install on the system (will install locally without root privileges)::


        python -m build
	pip install .

   B. Install with the intent to develop locally::


        python -m build
	pip install -e .
