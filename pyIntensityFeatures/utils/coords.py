#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
"""Coordinate handling and conversion functions."""

import datetime as dt
import numpy as np
import pandas as pds

import aacgmv2

try:
    import apexpy
except ImportError:
    apexpy = None


def convert_geo_to_mag(ctime, glat, glon, alt, method='ALLOWTRACE'):
    """Convert and reshape GUVI location output to magnetic coordinates.

    Parameters
    ----------
    ctime : dt.datetime
        Conversion time
    glat : array-like
        2D array of geodetic latitudes
    glon : array-like
        2D array of geographic longitudes
    alt : float
        Altitude of the auroral data in km
    method : str
        Magnetic method to use, expects one of the AACGMV2 codes ('TRACE',
        'ALLOWTRACE', 'BADIDEA', 'GEOCENTRIC'), 'apex', or 'qd'. The last two
        methods correspond to apex and quasi-dipole coordiantes, and only work
        if apexpy is available. (default='ALLOWTRACE')

    Returns
    -------
    mlat : array-like
        2D array of magnetic latitudes
    mlon : array-like
        2D array of magnetic longitudes
    mlt : array-like
        2D array of magnetic local times

    Raises
    ------
    ValueError
        If 'apex' or 'qd' is requested, but apexpy is not available.

    """
    if method.lower() in ['apex', 'qd']:
        if apexpy is None:
            raise ValueError('apexpy is not available.')

        # Convert the data, requires 1D arrays
        apex = apexpy.Apex(date=ctime, refh=alt)
        mlat, mlon = apex.convert(glat.flatten(), glon.flatten(), source='geo',
                                  dest=method, height=alt)
        mlt = apex.mlon2mlt(mlon, dtime=ctime)
    else:
        # Convert the data, requires 1D arrays
        mlat, mlon, mlt = aacgmv2.get_aacgm_coord_arr(glat.flatten(),
                                                      glon.flatten(), alt,
                                                      ctime, method=method)

    # Reshape the output
    mlat = mlat.reshape(glat.shape)
    mlon = mlon.reshape(glon.shape)
    mlt = mlt.reshape(glon.shape)

    return mlat, mlon, mlt


def get_slice_mlat_max_min(num_samples, mlat_bins, mlt_bins, mlat_inc=1.0):
    """Get latitude range of the auroral intensity slice.

    Parameters
    ----------
    num_samples : array-like
        2D array with number of samples in each mlat/mlt bin
    mlat_bins : np.array
        Magnetic latitude of output bin centres
    mlt_bins : np.array
        Magnetic local time of output bin centres
    mlat_inc : float
        Magnetic latitude increment for output (default=1.0)

    Returns
    -------
    mlat_min : float
        Minimum magnetic latitude along MLT slice that also contains the
        magnetic latitude maximum in degrees
    mlat_max : float
        Maximum magnetic latitude of slice in degrees

    """
    # Initialize output
    mlat_max = 0.0

    # Find the maximum magnetic latitude
    for ilat in np.arange(mlat_bins.shape[0] - 1, -1, -1):
        if num_samples[ilat, :].max() > 0:
            mlat_max = mlat_bins[ilat] + 0.5 * mlat_inc
            break

    # If a maximum was found, get the minimum at that MLT
    mlat_min = mlat_max
    if mlat_max != 0.0:
        # Get the local times with data at the maximum latitude
        ilts = np.where(num_samples[ilat, :] > 0)[0]

        # Find the lowest latitude where all these local times have data
        mmin = mlat_max
        for jlat in range(ilat):
            if np.all(num_samples[jlat, ilts] > 0):
                mlat_min = mlat_bins[jlat] - 0.5 * mlat_inc
                break
            else:
                if mlat_bins[jlat] - 0.5 * mlat_inc < mmin:
                    mmin = mlat_bins[jlat] - 0.5 * mlat_inc

        # There may not be a slice where this works
        if mlat_max == mlat_min:
            mlat_min = mmin

    return mlat_max, mlat_min


def as_datetime(time_val):
    """Ensure a time value is cast as datetime without timezone information.

    Parameters
    ----------
    time_val : object
        Date and time object that may be np.datetime64, pds.datetime,
        dt.datetime, or dt.date

    Returns
    -------
    out_time : dt.datetime
        desired casting of the datetime object

    """
    if isinstance(time_val, dt.datetime):
        out_time = dt.datetime(time_val.year, time_val.month, time_val.day,
                               time_val.hour, time_val.minute, time_val.second,
                               time_val.microsecond)
    elif isinstance(time_val, dt.date):
        out_time = dt.datetime(time_val.year, time_val.month, time_val.day)
    else:
        out_time = pds.to_datetime(time_val).to_pydatetime()

    return out_time
