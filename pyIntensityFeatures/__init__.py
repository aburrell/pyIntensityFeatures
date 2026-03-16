#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.15102100
# Full license can be found in License.md
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Package to identify intensity features in imager data."""

import logging
from importlib import metadata

# Define a logger object to allow easier log handling
logging.raiseExceptions = False
logger = logging.getLogger('pyIntensityFeatures_logger')

# Import the package modules and top-level classes
from pyIntensityFeatures import _main # noqa F401
from pyIntensityFeatures import instruments  # noqa F401
from pyIntensityFeatures import proc  # noqa F401
from pyIntensityFeatures import utils  # noqa F401

from pyIntensityFeatures._main import AuroralBounds  # noqa F401

# Define the global variables
__version__ = metadata.version('pyIntensityFeatures')
