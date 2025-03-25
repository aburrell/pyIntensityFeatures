#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Main classes and functions for the package."""

import datetime as dt
from functools import partial
import numpy as np
import pandas as pds
import xarray as xr

from pyIntensityFeatures import logger
from pyIntensityFeatures import proc
from pyIntensityFeatures.instruments import satellites
from pyIntensityFeatures import utils


class AuroralBounds(object):
    """Manage intensity data and obtain auroral luminosity boundaries.

    Parameters
    ----------
    inst_data : object
        Instrument data of any type (e.g., pysat.Instrument, xr.Dataset,
        pds.DataFrame, dict of arrays) with appropriate observations.
    time_var : str or int
        Time key, attribute, variable name, or index
    glon_var : str
        Geographic longitude key, attribute, variable name, or index
    glat_var : str
        Geographic latitude key, attribute, variable name, or index
    intensity_var : str
        Observed intensity key, attribute, variable name, or index
    alt : float or int
        Altitude in km at which the aurora is observed
    transpose : bool or dict
        Flag that indicates whether or not the 2D data accessed by `glon_var`,
        `glat_var`, and `intensity_var` needs to be transposed.  If a dict is
        supplied, the keys should correspond the the different variable names
        and allows different flags for each set of data. (default=False)
    opt_coords : dict or NoneType
        Dict of coordinates to include in `boundaries` or None to only have
        the required coordinates (default=None)
    hemipshere : int
        Integer denoting hemisphere, 1 for North and -1 for South (default=1)
    stime : object or NoneType
        Time object or None to use earliest time from `inst_data`
        (default=None)
    etime : object or NoneType
        Time object or None to use latest time from `inst_data` (default=None)
    slice_func : function
        Appropriate function to retrieve an auroral image slice
        (default=pyIntensityFeatures.instruments.satellites.get_auroral_slice)
    slice_kwargs : dict or NoneType
        Kwargs to be used by `slice_func` (default=None)
    clean_func : function or NoneType
        Appropriate function to clean the intensity data (default=None)
    clean_kwargs : dict or NoneType
        Kwargs to be used by `clean_func` (default=None)

    Attributes
    ----------
    inst_data
    time_var
    glon_var
    glat_var
    intensity_var
    alt
    opt_coords
    hemipshere
    stime
    etime
    slice_func
    clean_func
    boundaries : xr.Dataset or NoneType
        Dataset with Auroral Luminosity Boundaries or NoneType if not run

    Methods
    -------
    set_boundaries
        Set the `boundaries` attribute using the assigned data.
    update_times
        Update `stime` and `etime` attributes to reflect current data in
        `inst_data`, if available.

    """

    def __init__(self, inst_data, time_var, glon_var, glat_var, intensity_var,
                 alt, transpose=False, opt_coords=None, hemisphere=1,
                 stime=None, etime=None,
                 slice_func=satellites.get_auroral_slice, slice_kwargs=None,
                 clean_func=None, clean_kwargs=None):
        """Initalize the class attributes."""

        # Set the data object and altitude
        self.inst_data = inst_data
        self.alt = alt

        # Set the data access variables
        self.time_var = time_var
        self.glon_var = glon_var
        self.glat_var = glat_var
        self.intensity_var = intensity_var

        # Set the transpose flag for the intensity/location data
        if hasattr(transpose, 'keys'):
            self.transpose = transpose
            trans_default = False  # Set the default for unspecified variables
        else:
            # All variables will be set to the same flag value
            self.transpose = {}
            trans_default = transpose

        for tkey in [self.glon_var, self.glat_var, self.intensity_var]:
            if tkey not in self.transpose.keys():
                # Update the unset variables
                self.transpose[tkey] = trans_default

        # Set the hemisphere and coordinates
        self.hemisphere = hemisphere

        if opt_coords is None:
            self.opt_coords = {'hemisphere': self.hemisphere}
        else:
            self.opt_coords = opt_coords
            self.opt_coords['hemisphere'] = self.hemisphere

        # Set the starting and ending times
        self.stime = stime
        self.etime = etime

        # Set the instrument-specific functions
        if slice_kwargs is None:
            slice_kwargs = {}

        if slice_func is None:
            logger.info('slice function not provided, data must be pre-sliced.')
            self.slice_func = None
        else:
            self.slice_func = partial(slice_func, **slice_kwargs)

        if clean_kwargs is None:
            clean_kwargs = {}

        if clean_func is None:
            self.clean_func = None
        else:
            self.clean_func = partial(clean_func, **clean_kwargs)

        # Initalize the boundary object
        self.boundaries = None

        return

    def __eq__(self, other):
        """Perform equality check.

        Parameters
        ----------
        other : any
            Other object to compare for equality

        Returns
        -------
        bool
            True if objects are identical, False if they are not.

        """
        # Check if other is the same class
        if not isinstance(other, self.__class__):
            logger.info("wrong class")
            return False

        # Check to see if other has the same and equal attributes
        for attr in self.__dict__.keys():
            attr_flag = False
            if hasattr(other, attr):
                self_val = getattr(self, attr)
                other_val = getattr(other, attr)

                if attr.find('inst_data') == 0:
                    # This is the Instrument data, different comparisons are
                    # needed for known supported data types
                    if isinstance(self_val, xr.Dataset) or isinstance(
                            self_val, pds.DataFrame):
                        attr_flag = self_val.equals(other_val)
                    else:
                        attr_flag = np.all(self_val == other_val)
                elif attr.find('func') < 0:
                    # This is an integer, boolean or string
                    attr_flag = self_val == other_val
                else:
                    # This is a partial function, ensure it has all the
                    # necessary attributes
                    attr_flag = str(self_val) == str(other_val)
            else:
                logger.info("object is missing attribute: {:}".format(attr))
                return attr_flag

            if not attr_flag:
                # Exit upon first inequality
                logger.info("attribute {:} differs".format(attr))
                return attr_flag

        # Confirm that other object doesn't have extra terms
        for attr in other.__dict__.keys():
            if attr not in self.__dict__.keys():
                logger.info("object contains extra attribute: {:}".format(attr))
                return False

        return True

    def __repr__(self):
        """Print basic class properties."""
        out_str = "".join(["pyIntensityFeatures.AuroralBounds(",
                           ", ".join([repr(self.inst_data),
                                      repr(self.time_var),
                                      repr(self.glon_var), repr(self.glat_var),
                                      repr(self.intensity_var),
                                      repr(self.alt)]),
                           ", transpose=", repr(self.transpose),
                           ", opt_coords=", repr(self.opt_coords),
                           ", hemisphere=", repr(self.hemisphere), ", stime=",
                           repr(self.stime), ", etime=", repr(self.etime),
                           ", slice_func=", repr(self.slice_func),
                           ", clean_func=", repr(self.clean_func), ")"])
        return out_str

    def __str__(self):
        """Descriptively print the class properties."""

        out_str = "\n".join(["Auroral Boundary object",
                             "=======================",
                             "Instrument Data: {:}".format(self.inst_data),
                             "\nData Variables",
                             "--------------",
                             "Time: {:s}".format(self.time_var),
                             "Geo Lon: {:s}".format(self.glon_var),
                             "Geo Lat: {:s}".format(self.glat_var),
                             "Intensity: {:s}".format(self.intensity_var),
                             "Transpose: {:}".format(self.transpose),
                             "\nCoordinate Attributes",
                             "---------------------",
                             "Hemisphere: {:d}".format(self.hemisphere),
                             "Altitude: {:.2f} km".format(self.alt),
                             "Optional coords: {:}".format(self.opt_coords),
                             "Start time: {:}".format(self.stime),
                             "End time: {:}".format(self.etime),
                             "\nInstrument Functions",
                             "--------------------",
                             "Slicing: {:}".format(self.slice_func),
                             "Cleaning: {:}".format(self.clean_func)])

        return out_str

    def _get_variable(self, var):
        """Get the data variable data as an array from the data object.

        Parameters
        ----------
        var : str
            Data variable name

        Returns
        -------
        ret_data :array-like
            numpy array of the desired data.

        Notes
        -----
        Uses the `transpose` attribute to determine whether or not returned
        data will be transposed.

        """

        if hasattr(self.inst_data, str(var)):
            # Retrieve the time attribute and cast as a numpy array
            ret_data = np.array(getattr(self.inst_data, str(var)))
        else:
            try:
                # Retrieve the data variable as a key
                ret_data = np.array(self.inst_data[var])
            except (TypeError, KeyError, ValueError, IndexError):
                logger.warning("".join(["unable to retrieve ", repr(var), " ",
                                        "from `inst_data`, data may be empty"]))
                ret_data = None

        # Transpose the data if desired
        if var in self.transpose.keys() and ret_data is not None:
            if self.transpose[var]:
                ret_data = ret_data.transpose()

        return ret_data

    @property
    def alt(self):
        """Altitude in km at which the intensity data is located."""
        return self._alt

    @alt.setter
    def alt(self, new_alt=-1):
        self._alt = new_alt

    @property
    def hemisphere(self):
        """Hemisphere for data processing (1 for North, -1 for South)."""
        return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, new_hemisphere=1):
        # Also update the `opt_coords` attribute dict, if it exists and
        # contains hemisphere information.
        if hasattr(self, 'opt_coords'):
            opt_dict = getattr(self, 'opt_coords')

            if 'hemisphere' in opt_dict.keys():
                opt_dict['hemisphere'] = new_hemisphere
                setattr(self, 'opt_coords', opt_dict)

        self._hemisphere = new_hemisphere

    @property
    def stime(self):
        """Starting time for loaded data."""
        return self._stime

    @stime.setter
    def stime(self, new_stime=None):
        if new_stime is None:
            self._stime = self._get_variable(self.time_var)

            if self._stime is not None:
                self._stime = utils.coords.as_datetime(
                    self._stime.flatten().min())
        else:
            self._stime = utils.coords.as_datetime(new_stime)

    @property
    def etime(self):
        """Ending time for loaded data."""
        return self._etime

    @etime.setter
    def etime(self, new_etime=None):
        if new_etime is None:
            self._etime = self._get_variable(self.time_var)

            if self._etime is not None:
                self._etime = utils.coords.as_datetime(
                    self._etime.flatten().max())
        else:
            self._etime = utils.coords.as_datetime(new_etime)

    def set_boundaries(self, min_mlat_base=59.0, mag_method='ALLOWTRACE',
                       mlat_inc=1.0, mlt_inc=0.5, un_threshold=1.25,
                       dayglow_threshold=300.0, strict_fit=False,
                       lt_out_bin=5.0, max_iqr=1.5):
        """Set `boundaries` with auroral boundaries from intensity data.

        Parameters
        ----------
        min_mlat_base : float
            Base minimum co-latitude for intensity profiles. (default=59.0)
        method : str
            Method for converting between geographic and magnetic coordinates.
            (default='ALLOWTRACE')
        mlat_inc : float
            Magnetic latitude increment for gridding intensity. (default=1.0)
        mlt_inc : float
            Magnetic local time increment for gridding intensity. (default=0.5)
        un_threshold : float
            Maximum acceptable uncertainty value in degrees (default=1.25)
        dayglow_threshold : float
            Minimum allowable background intensity value in Rayleighs
            (default=300.0)
        strict_fit : bool
            Enforce positive values for the x-offsets in quadratic-Gaussian fits
            (default=False)
        lt_out_bin : float
            Size of local time bin in hours over which outliers in the data
            will be identified (default=5.0)
        max_iqr : float
            Maximum multiplier for the interquartile range (IQR) used to
            identify outliers above or below the upper or lower quartile
            (default=1.5)

        See Also
        --------
        utils.coords.convert_geo_to_mag

        Notes
        -----
        Luminosity boundary identification finding and verification based on,
        but not identical to, the method developed by Longden, et al. (2010).

        References
        ----------
        Longden, N. S., et al. (2010) Estimating the location of the open-closed
        magnetic field line boundary from auroral images, 28 (9), p 1659-1678,
        doi:10.5194/angeo-28-1659-2010.

        """
        # Get the time, location, and intensity data
        time_data = self._get_variable(self.time_var)
        glat_data = self._get_variable(self.glat_var)
        glon_data = self._get_variable(self.glon_var)
        intensity_data = self._get_variable(self.intensity_var)

        for name, var in [(self.time_var, time_data),
                          (self.glat_var, glat_data),
                          (self.glon_var, glon_data),
                          (self.intensity_var, intensity_data)]:
            if var is None:
                logger.info("Missing {:} data, cannot set boundaries".format(
                    name))
                return

        # Initalize the dicts for holding coordinate and boundary data
        attr_dict = {'min_mlat_base': min_mlat_base, 'mag_method': mag_method,
                     'mlat_inc': mlat_inc, 'mlt_inc': mlt_inc,
                     'un_threshold': un_threshold, 'max_iqr': max_iqr,
                     'strict_fit': int(strict_fit), 'lt_out_bin': lt_out_bin}
        coord_dict, data_dict = utils.output.init_boundary_dicts(
            opt_coords=self.opt_coords)
        max_coeff = 0

        # Clean the intensity data
        if self.clean_func is not None:
            clean_mask = self.clean_func(self.inst_data)
        else:
            clean_mask = None

        # Cycle through the desired date range
        ctime = utils.coords.as_datetime(self.stime)
        while ctime < utils.coords.as_datetime(self.etime):
            # Get the next auroral intensity slice
            if self.slice_func is not None:
                # Use provided function to select desired data
                intensity, glat, glon, sweep_times = self.slice_func(
                    time_data, glat_data, glon_data, intensity_data,
                    clean_mask=clean_mask, start_time=ctime,
                    hemisphere=self.hemisphere)
            else:
                # No slicing function provided, assume data is pre-sliced
                intensity = intensity_data
                glat = glat_data
                glon = glon_data
                sweep_times = [time_data[0], time_data[-1]]

            # Get the next auroral slice and boundaries
            (sweep_end, sweep_data,
             max_coeff) = proc.intensity.find_intensity_boundaries(
                 intensity, glat, glon, sweep_times, self.alt,
                 min_mlat_base, max_coeff, method=mag_method,
                 mlat_inc=mlat_inc, mlt_inc=mlt_inc, un_threshold=un_threshold,
                 dayglow_threshold=dayglow_threshold, strict_fit=strict_fit)

            logger.info("Auroral slice at {:} {:s} data".format(
                sweep_end, "without" if sweep_data is None else "with"))

            if sweep_end < ctime:
                # We've reached the end of the data, but it's not triggering
                # the escape condition
                break

            # Save the output for this sweep
            utils.output.update_boundary_dicts(sweep_data, coord_dict,
                                               data_dict)

            # Cycle the time to start looking for the next slice
            ctime = utils.coords.as_datetime(sweep_end) + dt.timedelta(
                seconds=1)

        # Reshape the data and cast as xarray
        self.boundaries = utils.output.convert_boundary_dict(
            coord_dict, data_dict, max_coeff, attr_dict=attr_dict)

        # If there is data, update the boundaries, otherwise inform the user
        if len(self.boundaries) > 0:
            # Remove boundary outliers from each slice
            logger.info("Removing boundary outliers from {:} to {:}.".format(
                self.stime, self.etime))
            self.boundaries = utils.checks.evaluate_boundary_in_mlt(
                self.boundaries, "eq_bounds", "po_bounds", "mlt",
                "sweep_start", lt_bin=lt_out_bin, max_iqr=max_iqr)
        else:
            logger.info("No boundary data found from {:} to {:}.".format(
                self.stime, self.etime))

        return

    def update_times(self):
        """Update `stime` and `etime` based on potentially updated data."""
        # Setting to NoneType triggers the "get value from data" feature
        self.stime = None
        self.etime = None
        return
