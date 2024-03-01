#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
"""Satellite instrument support functions."""

import numpy as np


def get_auroral_slice(time_data, glat_data, glon_data, int_data,
                      clean_mask=None, start_time=None, hemisphere=1,
                      min_colat=45):
    """Retrieve an auroral image slice that spans the desired lat range.

    Parameters
    ----------
    time_data : array-like
        1D array of time indexes
    glat_data : array-like
        2D array of geographic latitudes
    glon_data : array-like
        2D array of geographic longitudes
    int_data : array-like
        2D array of intensity values
    clean_mask : array-like or NoneType
        None to create a mask of finite values or 2D mask array specifying
        good values (default=None)
    start_time : dt.datetime or NoneType
        Start time to search from or None to start from beginning of
        available data (default=None)
    hemisphere : int
        Hemisphere to consider, where 1 is North and -1 is South (default=1)
    min_colat : int or float
        Absolute value of the most equatorward latitude to include (default=45)

    Returns
    -------
    intensity : array-like
        Array with dimensions of time x sweep-locations containing the
        rectified intensity at the auroral daytime pierce point
    glat : array-like
        Array with dimensions of time x sweep-locations containing the
        geodetic latitude
    glon : array-like
        Array with dimensions of time x sweep-locations containing the
        geographic longitude
    sweep_times : array-like
        Array with the starting and ending times of the auroral sweep. Gives
        the start and end times searched if no sweep is found.

    Raises
    ------
    ValueError
        If imager data does not have the same shape

    """
    # Set the start time
    if start_time is None:
        start_time = time_data[0]
        itime = 0
    else:
        start_time = np.datetime64(start_time)
        time_data = np.asarray(time_data).astype(np.datetime64)
        itime = int(abs(start_time - time_data).argmin())

        if time_data[itime] < start_time:
            itime += 1

    # Create the data mask, if desired
    if clean_mask is None:
        clean_mask = np.isfinite(int_data)

    # Ensure the data arrays are shaped as expected
    if np.asarray(time_data).shape[0] != int_data.shape[0]:
        raise ValueError('first dimension of intensity data differs from time')

    if int_data.shape != glat_data.shape or int_data.shape != glon_data.shape:
        raise ValueError('intensity and location input shapes differ')

    if clean_mask.shape != int_data.shape:
        raise ValueError('clean mask shape differs from intensity data')

    # Find the start of an auroral sweep
    while itime < len(time_data):
        if (glat_data[itime] * hemisphere >= min_colat).any():
            if clean_mask[itime].any():
                # If there is no clean data at this time, keep looking
                break

        # Cycle to the next time
        itime += 1

    if itime < len(time_data):
        # Prepare the output
        time_inds = list()
        sweep_times = list()

        # Found an auroral sweep in the correct hemisphere
        while itime < len(time_data) and np.any(
                glat_data[itime] * hemisphere >= min_colat):
            if clean_mask[itime].any():
                time_inds.append(itime)

                if len(sweep_times) < 2:
                    sweep_times.append(time_data[itime])
                else:
                    sweep_times[-1] = time_data[itime]

            # Cycle the time index
            itime += 1

        # If the sweep ended without capturing sufficient data, set the stop
        # time
        if len(sweep_times) < 2:
            if itime >= len(time_data):
                itime = -1

            if len(sweep_times) == 0:
                # If no times were found, insert the start time
                sweep_times.append(start_time)

            sweep_times.append(time_data[itime])

        # The sweep has ended, select the desired time indices
        intensity = np.asarray(int_data[time_inds])
        glat = np.asarray(glat_data[time_inds])
        glon = np.asarray(glon_data[time_inds])
        sweep_times = np.asarray(sweep_times)
    else:
        intensity = np.asarray([])
        glat = np.asarray([])
        glon = np.asarray([])
        sweep_times = np.asarray([start_time, time_data[-1]])

    return intensity, glat, glon, sweep_times
