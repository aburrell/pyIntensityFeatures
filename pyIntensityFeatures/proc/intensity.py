#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
"""Functions for intensity processing."""

import numpy as np

from pyIntensityFeatures import utils
from pyIntensityFeatures.proc import boundaries
from pyIntensityFeatures.proc import fitting


def find_intensity_boundaries(intensity, glat, glon, sweep_times, alt,
                              min_mlat_base, max_coeff, method='ALLOWTRACE',
                              mlat_inc=1.0, mlt_inc=0.5, un_threshold=1.25,
                              dayglow_threshold=300.0, strict_fit=False):
    """Find the PALBs and EALBs for a slice of intensity data.

    Parameters
    ----------
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
    alt : float or array-like
        Altitude for the intensity data
    min_mlat_base : float
        Base minimum co-latitude for intensity profiles.
    max_coeff : int
        Maximum number of coefficients.
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
        Minimum allowable background intensity value in Rayleighs (default=300)
    strict_fit : bool
        Enforce positive values for the x-offsets in quadratic-Gaussian fits
        (default=False)

    Returns
    -------
    sweep_end : dt.datetime
        End time of the data
    out_data : dict or NoneType
        Dict with desired boundary data, if data was found.  None otherwise.
    max_coeff : int
        Maximum number of coefficients.

    See Also
    --------
    utils.coords.convert_geo_to_mag

    """
    # Initialize the output
    out_data = None

    # Set the evaluation method by number of peaks
    eval_method = {3: "best", 2: "mult", 1: "single"}
    ng_order = [1, 2, 3]

    # Ensure the input is appropriate
    if len(sweep_times) == 2 and len(glat) > 0:
        # Get the magnetic coordinates
        sweep_start = utils.coords.as_datetime(sweep_times[0])
        sweep_end = utils.coords.as_datetime(sweep_times[1])
        mlat, mlon, mlt = utils.coords.convert_geo_to_mag(sweep_start, glat,
                                                          glon, alt, method)

        # Get the mean intensity for this time
        (mean_intensity_orig, std_intensity, num_intensity, mlat_bins_orig,
         mlt_bins) = utils.grids.grid_intensity(intensity, mlat, mlt,
                                                mlat_inc=mlat_inc,
                                                mlt_inc=mlt_inc)

        # Determine the minimum magnetic latitude available at all MLT
        mlat_max, mlat_min = utils.coords.get_slice_mlat_max_min(
            num_intensity, mlat_bins_orig, mlt_bins)
        hemisphere = np.sign(mlat_max)

        # Ensure the minimum is not too high
        if abs(mlat_min) > min_mlat_base:
            mlat_min = hemisphere * min_mlat_base

        # Reshape the mean intensity data for the new ranges
        ibins = np.where((abs(mlat_bins_orig) > abs(mlat_min))
                         & (abs(mlat_bins_orig) <= abs(mlat_max)))[0]
        mlat_bins = mlat_bins_orig[ibins]
        mean_intensity = mean_intensity_orig[ibins, :]
        std_intensity = std_intensity[ibins, :]
        num_intensity = num_intensity[ibins, :]

        if len(ibins) > 0:
            # Initalize the parameter dicts
            params = dict()
            covar = dict()
            rvalue = dict()
            pvalue = dict()
            npeaks = dict()

            # Set the maximum number of Gaussian peaks to 1, 2, and 3
            for ng in [1, 2, 3]:
                # Get the single, double, or triple Gaussian fits at each MLT
                params[ng], covar[ng], rvalue[ng], pvalue[ng], npeaks[
                    ng] = fitting.get_gaussian_func_fit(
                        mlat_bins, mlt_bins, mean_intensity, std_intensity,
                        num_intensity, num_gauss=ng)

            # Initalize the boundary outputs at each MLT
            eq_bounds = np.full(shape=mlt_bins.shape, fill_value=np.nan)
            eq_uncert = np.full(shape=mlt_bins.shape, fill_value=np.nan)
            po_bounds = np.full(shape=mlt_bins.shape, fill_value=np.nan)
            po_uncert = np.full(shape=mlt_bins.shape, fill_value=np.nan)
            eq_params = list()
            po_params = list()

            # Locate the boundaries at each MLT
            for i in range(len(params[3])):
                bounds = dict()
                good_ng = dict()

                # Get the boundaries for each fit type
                for ng in params.keys():
                    if ng == 2 and np.shape(params[ng][i]) != (9,):
                        good_shape = False
                    else:
                        good_shape = True

                    if good_shape and covar[ng][i] is not None:
                        # Get the boundaries
                        (bounds[ng],
                         good_ng[ng]) = boundaries.get_eval_boundaries(
                             params[ng][i], covar[ng][i], rvalue[ng][i],
                             pvalue[ng][i], npeaks[ng][i], mlat_min, mlat_max,
                             eval_method[ng], un_threshold=un_threshold,
                             dayglow_threshold=dayglow_threshold,
                             strict_fit=strict_fit)

                        if np.isnan(bounds[ng]).all():
                            rvalue[ng][i] = None
                            pvalue[ng][i] = None
                    else:
                        bounds[ng] = [np.nan, np.nan, np.nan, np.nan]
                        good_ng[ng] = False
                        rvalue[ng][i] = None
                        pvalue[ng][i] = None

                # Select the best fit
                good_fits = [ng for ng in good_ng.keys() if good_ng[ng]]

                if len(good_fits) == 1:
                    # Use the only good fit that passed the extra test for
                    # robustness
                    ng = good_fits[0]
                    eq_bounds[i] = bounds[ng][0]
                    eq_uncert[i] = bounds[ng][2]
                    po_bounds[i] = bounds[ng][1]
                    po_uncert[i] = bounds[ng][3]

                    eq_params.append(params[ng][i])
                    po_params.append(params[ng][i])

                    if len(params[ng][i]) > max_coeff:
                        max_coeff = len(params[ng][i])
                else:
                    # Select the best fit between the boundaries
                    rvals = [rvalue[ng][i] for ng in ng_order]
                    pvals = [pvalue[ng][i] for ng in ng_order]
                    ebnds = [bounds[ng][0] for ng in ng_order]
                    eunc = [bounds[ng][2] for ng in ng_order]
                    pbnds = [bounds[ng][1] for ng in ng_order]
                    punc = [bounds[ng][3] for ng in ng_order]
                    prms = [params[ng][i] for ng in ng_order]
                    igood_eq, igood_po = utils.checks.compare_boundaries(
                        rvals, pvals, ebnds, eunc, pbnds, punc, mlat_min,
                        mlat_max)

                    # Only update the EALB if a good fit was found
                    if igood_eq in [0, 1, 2]:
                        eq_bounds[i] = ebnds[igood_eq]
                        eq_uncert[i] = eunc[igood_eq]
                        eq_params.append(prms[igood_eq])

                        if len(prms[igood_eq]) > max_coeff:
                            max_coeff = len(prms[igood_eq])
                    else:
                        eq_params.append([])

                    # Only update the PALB if a good fit was found
                    if igood_po in [0, 1, 2]:
                        po_bounds[i] = pbnds[igood_po]
                        po_uncert[i] = punc[igood_po]
                        po_params.append(prms[igood_po])

                        if len(prms[igood_po]) > max_coeff:
                            max_coeff = len(prms[igood_po])
                    else:
                        po_params.append([])

                # Ensure the best equatorward boundary is within the latitude
                # range for this slice
                if np.isfinite(eq_bounds[i]):
                    mlat_bound = hemisphere * abs(mlat_bins_orig[np.isfinite(
                        mean_intensity_orig[:, i])]).min() + 0.5 * (
                            hemisphere * mlat_inc)
                    if abs(eq_bounds[i]) <= abs(mlat_bound):
                        eq_bounds[i] = np.nan
                        eq_uncert[i] = np.nan

                # Ensure the best poleward boundary is within the latitude
                # range for this slice
                if np.isfinite(po_bounds[i]):
                    mlat_bound = hemisphere * abs(mlat_bins_orig[np.isfinite(
                        mean_intensity_orig[:, i])]).max() - 0.5 * (
                            hemisphere * mlat_inc)
                    if abs(po_bounds[i]) >= abs(mlat_bound):
                        po_bounds[i] = np.nan
                        po_uncert[i] = np.nan

            # Save the output for this sweep
            if np.isfinite(eq_bounds).any() or np.isfinite(po_bounds).any():
                out_data = {'sweep_start': sweep_start, 'sweep_end': sweep_end,
                            'mlt': mlt_bins, 'mlat': mlat_bins,
                            'eq_bounds': eq_bounds, 'eq_uncert': eq_uncert,
                            'eq_params': eq_params, 'po_bounds': po_bounds,
                            'po_uncert': po_uncert, 'po_params': po_params,
                            'mean_intensity': mean_intensity,
                            'std_intensity': std_intensity,
                            'num_intensity': num_intensity}
    else:
        # Get the time of the unsuccessful sweep for time cycling. There
        # may be one or two sweep elements, so use the last one.
        sweep_end = utils.coords.as_datetime(sweep_times[-1])

    return sweep_end, out_data, max_coeff
