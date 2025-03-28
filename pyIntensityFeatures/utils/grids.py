#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Functions to support, create, and process grids."""

import numpy as np


def unique(vals, decimals=3, **kwargs):
    """Identify the unique values to the desired significance.

    Parameters
    ----------
    vals : array-like
        Input array that will be flattened unless `axis` is specified
    decimals : int
        Number of decimal places to round to. If the number is negative, it
        specifies the number of positions to the left of the decimal point.
        (default=3)
    **kwargs : dict
        Keyword arguements supported by numpy's unique function.

    Returns
    -------
    uvals : array-like
        Sorted array of unique values.

    See Also
    --------
    np.unique

    """
    # Use numpy unique with rounded values
    uvals = np.unique(np.round(vals, decimals=decimals), **kwargs)

    return uvals


def grid_intensity(intensity, mlat, mlt, eq_mlat=45.0, mlat_inc=1.0,
                   mlt_inc=0.5):
    """Create an intensity grid, using the mean and standard deviation.

    Parameters
    ----------
    intensity : np.array
        Input intensity array with NaN fill values
    mlat : np.array
        Magnetic latitude at intensity locations
    mlt : np.array
        Magnetic local time at intensity locations
    eq_mlat : float
        Most equatorward magnetic latitude magnitude, do not account for the
        hemisphere (defaut=45.0)
    mlat_inc : float
        Magnetic latitude increment for output (default=1.0)
    mlt_inc : float
        Magnetic local time increment for output (default=0.5)

    Returns
    -------
    mean_intensity : np.array
       2D array of mean intensity values
    std_intensity : np.array
       2D array of intensity standard deviations
    num_intensity : np.array
       2D array of intensity counts per bin
    mlat_bins : np.array
        Magnetic latitude of output bin centres
    mlt_bins : np.array
        Magnetic local time of output bin centres

    Notes
    -----
    The hemisphere is determined using the sign of the data in `mlat`

    """

    # Set the bin centres
    hemisphere = np.sign(np.nanmax(mlat))
    mlat_bins = np.arange(hemisphere * (eq_mlat + 0.5 * mlat_inc),
                          hemisphere * 90.0, hemisphere * mlat_inc)
    mlt_bins = np.arange(0.0 + 0.5 * mlt_inc, 24.0, mlt_inc)

    # Initalize the mean and standard deviation output
    mean_intensity = np.full(shape=(mlat_bins.shape[0], mlt_bins.shape[0]),
                             fill_value=np.nan)
    std_intensity = np.full(shape=(mlat_bins.shape[0], mlt_bins.shape[0]),
                            fill_value=np.nan)
    num_intensity = np.zeros(shape=(mlat_bins.shape[0], mlt_bins.shape[0]))

    # Cycle through each latititude and local time, finding appropriate data
    for ilat, mlat_cent in enumerate(mlat_bins):
        min_lat = mlat_cent - 0.5 * mlat_inc
        max_lat = mlat_cent + 0.5 * mlat_inc
        for ilt, mlt_cent in enumerate(mlt_bins):
            min_lt = mlt_cent - 0.5 * mlt_inc
            max_lt = mlt_cent + 0.5 * mlt_inc

            # Find the data indexes
            ind = np.where((mlt >= min_lt) & (mlt < max_lt) & (mlat >= min_lat)
                           & (mlat < max_lat) & np.isfinite(intensity))
            num = len(ind[0])

            if num > 0:
                # There is data in this bin, get the mean and standard dev.
                mean_intensity[ilat, ilt] = np.mean(intensity[ind])
                std_intensity[ilat, ilt] = np.std(intensity[ind])
                num_intensity[ilat, ilt] = num

    return mean_intensity, std_intensity, num_intensity, mlat_bins, mlt_bins
