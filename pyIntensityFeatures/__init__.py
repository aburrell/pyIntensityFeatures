#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Package to identify intensity features in imager data."""

import logging

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

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
