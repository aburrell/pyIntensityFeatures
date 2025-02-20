#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Functions for formatting and preparing output."""

import numpy as np
import xarray as xr

from pyIntensityFeatures.utils import grids


def init_boundary_dicts(opt_coords=None, lat_dim='lat'):
    """Initialize dict for output from multiple auroral image boundaries.

    Parameters
    ----------
    opt_coords : dict or NoneType
        Dict of coordinates to include in the output or None to only have
        the required coordinates (default=None)
    lat_dim : str
        Name of the latitude dimension

    Returns
    -------
    coord_dict : dict
        Dictionary with required and optional coordinates
    data_dict : dict
        Dictionary with initialized data lists and dimensions

    See Also
    --------
    proc.intensity.find_intensity_boundaries

    """

    # Initialize the coordinates
    coord_dict = {'sweep_start': list(), 'sweep_end': list(), 'mlt': None}

    # Update the coordinates
    if opt_coords is not None:
        for ckey in opt_coords.keys():
            coord_dict[ckey] = opt_coords[ckey]

    # Initialize the data
    data_dict = {'mlat': (('sweep_start', lat_dim), list()),
                 'eq_bounds': (('sweep_start', 'mlt'), list()),
                 'eq_uncert': (('sweep_start', 'mlt'), list()),
                 'po_bounds': (('sweep_start', 'mlt'), list()),
                 'po_uncert': (('sweep_start', 'mlt'), list()),
                 'eq_params': (('sweep_start', 'mlt', 'coeff'), list()),
                 'po_params': (('sweep_start', 'mlt', 'coeff'), list()),
                 'mean_intensity': (('sweep_start', lat_dim, 'mlt'), list()),
                 'std_intensity': (('sweep_start', lat_dim, 'mlt'), list()),
                 'num_intensity': (('sweep_start', lat_dim, 'mlt'), list())}

    return coord_dict, data_dict


def update_boundary_dicts(sweep_data, coord_dict, data_dict):
    """Update coordinate and data dicts with boundary output.

    Parameters
    ----------
    sweep_data : dict or NoneType
        Dict with desired boundary data, if data was found.  None otherwise.
    coord_dict : dict
        Dictionary with required and optional coordinates
    data_dict : dict
        Dictionary with data lists and dimensions

    Raises
    ------
    ValueError
       If there is a change in the magnetic local time bins.

    """
    # Save the output for this sweep
    if sweep_data is not None:
        # Update the coordinates
        coord_dict['sweep_start'].append(sweep_data['sweep_start'])
        coord_dict['sweep_end'].append(sweep_data['sweep_end'])

        if coord_dict['mlt'] is None:
            coord_dict['mlt'] = sweep_data['mlt']
        elif coord_dict['mlt'].shape != sweep_data['mlt'].shape:
            raise ValueError('change in magnetic local time bin increment')
        elif not (coord_dict['mlt'] == sweep_data['mlt']).all():
            raise ValueError('change in magnetic local time bin values')

        # Update the data
        for dkey in data_dict.keys():
            data_dict[dkey][1].append(sweep_data[dkey])

    return


def reshape_lat_coeff_data(coord_dict, data_dict, max_coeff, lat_dim='lat',
                           lat_var='mlat', coeff_dim='coeff'):
    """Reshape data that depends on latitude or coefficients to be uniform.

    Parameters
    ----------
    coord_dict : dict
        Dictionary for xr.Dataset coordinate input
    data_dict : dict
        Dictionary for xr.Dataset input
    max_coeff : int
        Maximum number of coefficients
    lat_dim : str
        Name of the latitude dimension
    lat_var : str
        Name of the latitude variable
    coeff_dim : str
        Name of the coefficient dimension

    Raises
    ------
    ValueError
        If the latitude bins have an unexpected shape or inconsistent
        increments.  Also if the coefficients have an unexpected shape.

    Returns
    -------
    mlat_bins : array-like
        Updated magnetic latitude bins.
    data_dict : array-like
        Updated data dictionary, with consistent dimensions and padded data.

    """
    # Determine the full latitude range
    min_lat = 90.0
    max_lat = -90.0
    lat_inc = list()

    for lats in data_dict[lat_var][1]:
        if np.nanmin(lats) < min_lat:
            min_lat = np.nanmin(lats)

        if np.nanmax(lats) > max_lat:
            max_lat = np.nanmax(lats)

        lincs = grids.unique(lats[1:] - lats[:-1])
        if len(lincs) > 1:
            raise ValueError('badly shaped latitude bins')

        lat_inc.append(abs(lincs[0]))

    lincs = grids.unique(lat_inc)
    if len(lincs) > 1:
        raise ValueError('inconsistent latitude increments')

    if len(lincs) == 0:
        raise ValueError('no latitude data')

    lat_inc = lincs[0]

    # Create the full-range latiutde bins
    mlat_bins = np.arange(min_lat, max_lat + lat_inc, lat_inc)

    if mlat_bins.max() < 0.0:
        mlat_bins = np.array(list(reversed(mlat_bins)))

    # Update the data variables that depend on latitude
    for dvar in data_dict.keys():
        update_data = False

        # Reshape the latitude or coefficeint dimension (can't do both)
        if dvar != lat_var and lat_dim in data_dict[dvar][0]:
            # Trigger to save the output
            update_data = True

            # Get a new empty data object with the desired shape
            new_shape = [len(coord_dict[dcoord])
                         if dcoord in coord_dict.keys()
                         else mlat_bins.shape[0]
                         for dcoord in data_dict[dvar][0]]
            ilat = list(data_dict[dvar][0]).index(lat_dim)
            new_data = np.full(shape=new_shape, fill_value=np.nan)

            # Get the indices that overlap the new and old data arrays for each
            # time sweep
            dat_slice = [slice(None) for i in range(len(new_shape) - 1)]
            for itime, lats in enumerate(data_dict[lat_var][1]):
                dat_slice[ilat - 1] = [i for i, lat in enumerate(mlat_bins)
                                       if lat in lats]
                new_data[itime][tuple(dat_slice)] = data_dict[dvar][1][itime]

        elif coeff_dim in data_dict[dvar][0]:
            # Trigger to save the output
            update_data = True

            # Get a new empty data object with the desired shape
            new_shape = [len(coord_dict[dcoord])
                         if dcoord in coord_dict.keys()
                         else max_coeff for dcoord in data_dict[dvar][0]]
            icoeff = list(data_dict[dvar][0]).index(coeff_dim)
            new_data = np.full(shape=new_shape, fill_value=np.nan)

            if len(new_shape) != 3 or icoeff != 2:
                raise ValueError('unexpected dimension order for coefficients')

            # Get the indices that overlap the new and old data arrays for each
            # UT and MLT sweep
            for itime, mlt_dat in enumerate(data_dict[dvar][1]):
                for imlt, coeff_dat in enumerate(mlt_dat):
                    icoeff = np.arange(0, len(coeff_dat), 1)
                    new_data[itime, imlt, icoeff] = coeff_dat

        if update_data:
            # Save the output
            data_dict[dvar] = list(data_dict[dvar])
            data_dict[dvar][1] = new_data
            data_dict[dvar] = tuple(data_dict[dvar])

    # Remove the latitude from the data dict output
    del data_dict[lat_var]

    # Return the new data dict and the new mlat bins
    return mlat_bins, data_dict


def convert_boundary_dict(coord_dict, data_dict, max_coeff, lat_dim='lat',
                          lat_var='mlat', coeff_dim='coeff', attr_dict=None):
    """Convert coordinate and data dictionaries to an xarray Dataset.

    Parameters
    ----------
    coord_dict : dict
        Dictionary for xr.Dataset coordinate input
    data_dict : dict
        Dictionary for xr.Dataset input
    max_coeff : int
        Maximum number of coefficients
    lat_dim : str
        Name of the latitude dimension
    lat_var : str
        Name of the latitude variable
    coeff_dim : str
        Name of the coefficient dimension
    attr_dict : dict or NoneType
        Dict containing global attributes or None to omit (default=None)

    Returns
    -------
    out_data : xr.Dataset
        Dataset with coordinates, values, and dimensions supplied by the
        input dictionaries (reshaped to have consistent dimensions).

    Raises
    ------
    ValueError
        If the latitude bins have an unexpected shape or inconsistent
        increments.  Also if the coefficients have an unexpected shape.

    """
    # Update the attribute dictionary, if necessary
    if attr_dict is None:
        attr_dict = {}

    # Determine if there is data to process
    if coord_dict['mlt'] is None:
        out_data = xr.Dataset()
    else:
        # Reshape the data so that all dimensions are consistent
        coord_dict[lat_dim], data_dict = reshape_lat_coeff_data(
            coord_dict, data_dict, max_coeff, lat_dim=lat_dim, lat_var=lat_var,
            coeff_dim=coeff_dim)

        # Recast the output as and xarray Dataset
        out_data = xr.Dataset(data_vars=data_dict, coords=coord_dict,
                              attrs=attr_dict)

    return out_data
