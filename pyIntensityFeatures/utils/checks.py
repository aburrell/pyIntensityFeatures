#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Functions for testing the outcome of various functions."""

import numpy as np
import pandas as pds

from pyIntensityFeatures.utils import grids
from pyIntensityFeatures.utils.distributions import calc_quadratic
from pyIntensityFeatures import logger


fwhm_const = 2.0 * np.sqrt(2.0 * np.log(2.0))


def evaluate_gauss_quad(npeaks, fit_coeff, rvalue, pvalue, po_bound,
                        un_bound_po, eq_bound, un_bound_eq, min_mlat,
                        max_mlat, un_threshold=1.25):
    """Test the robustness of the quadratic Gaussian fitting results.

    Parameters
    ----------
    npeaks : int
        Number of Gaussian peaks in the fit
    fit_coeff : array-like
        List or array containing Guassian constant, quadratic multiplier for x,
        quadratic multiplier for x^2, and Gaussian amplitudes, x offsets, and
        exponential scalers in that order.
    rvalue : float
        Pearson correlation coefficient
    pvalue : float
        Pearson p-value
    po_bound : float
        Polar auroral boundary location in degrees latitude
    un_bound_po : float
        Uncertainty of the polar auroral boundary location in degrees latitude
    eq_bound : float
        Equatorward auroral boundary location in degrees latitude
    un_bound_eq : float
        Uncertainty of the equatorward auroral boundary location in degrees lat
    min_mlat : float
        Minimum latitude in intensity slice in degrees
    max_mlat : float
        Maximum latitude in intensity slice in degrees
    un_threshold : float
        Maximum acceptable uncertainty value in degrees (default=1.25)

    Returns
    -------
    bool
        True if all quality criteria are met, False if any fails

    References
    ----------
    Longden, N. S., et al. (2010) Estimating the location of the open-closed
    magnetic field line boundary from auroral images, 28 (9), p 1659-1678,
    doi:10.5194/angeo-28-1659-2010.

    Notes
    -----
    Differs from Longden, et al. (2010) by using Pearson correlation
    coefficients.

    """
    # All coefficients are finite
    if not np.all(np.isfinite(fit_coeff)):
        logger.info('Fit coefficients are not all finite.')
        return False

    # Cycle through each peak for individual tests
    num_narrow = 0
    for i in range(npeaks):
        icoeff = i * 3

        # The Gaussian coefficients have the expected sign
        if (np.array([fit_coeff[icoeff + 3],
                      fit_coeff[icoeff + 5]]) < 0.0).any():
            logger.info(''.join(['Gaussian amplitude and/or spread are ',
                                 'negative for peak {:d} of {:d}'.format(
                                     i, npeaks)]))
            return False

        if np.sign(fit_coeff[icoeff + 4]) != np.sign(max_mlat):
            logger.info(''.join(['Gaussian peak location in the wrong ',
                                 'hemisphere for peak {:d} of {:d}'.format(
                                     i, npeaks)]))
            return False

        # The center of the Gaussian peak is within the latitude range covered
        # by the intensity profile
        if abs(fit_coeff[icoeff + 4]) <= abs(min_mlat) or abs(fit_coeff[
                icoeff + 4]) >= abs(max_mlat):
            logger.info(''.join(['Gaussian peak is outside of the intensity ',
                                 'profile for peak {:d} of {:d}'.format(
                                     i, npeaks)]))
            return False

        # Calculate the background at the peak location using the quadradic fit
        bg = calc_quadratic(fit_coeff[icoeff + 4], fit_coeff[0], fit_coeff[1],
                            fit_coeff[2])

        # The mangitude of the Gaussian peak is at least 10% of the background
        if fit_coeff[icoeff + 3] < 0.1 * bg:
            logger.info(''.join(['Gaussian amplitude is less than 10% of the',
                                 ' background for peak {:d} of {:d}'.format(
                                     i, npeaks)]))
            return False

        # At least one Gaussian peak must exceed a minimum width
        if fit_coeff[icoeff + 5] < 1.0:
            num_narrow += 1

        # The Gaussian peak width should not be larger than the data range
        if fwhm_const * fit_coeff[icoeff + 5] > abs(max_mlat - min_mlat):
            logger.info(''.join(['Gaussian peak width is larger than the ',
                                 'data range for peak {:d} of {:d}'.format(
                                     i, npeaks)]))
            return False

    # Ensure at least one Gaussian peak exceeds the minimum width
    if num_narrow == npeaks:
        logger.info('All Gaussian peaks are too narrow.')
        return False

    # The uncertainty is below a given threshold for the polar boundary
    if not np.isfinite(un_bound_po) or un_bound_po >= un_threshold:
        logger.info(''.join(['The polar boundary uncertainty is unrealistic:',
                             ' {:.2f} >= {:.2f}.'.format(
                                 un_bound_po, un_threshold)]))
        return False

    # The uncertainty is below a given threshold for the equatorward boundary
    if not np.isfinite(un_bound_eq) or un_bound_eq >= un_threshold:
        logger.info(''.join(['The equatorward boundary uncertainty is ',
                             'unrealistic: {:.2f} >= {:.2f}.'.format(
                                 un_bound_eq, un_threshold)]))
        return False

    # The range of the PALB falls within the EALB and 90 degrees, while the
    # EALB falls withn the minimum latitude and the PALB
    if abs(po_bound) <= abs(eq_bound) or abs(po_bound) >= abs(max_mlat) or abs(
            eq_bound) <= abs(min_mlat):
        logger.info('The polar/equatorward boundary locations are mixed up.')
        return False

    # Ensure reasonable Pearson R and P values.
    return evaluate_pearson(rvalue, pvalue)


def evaluate_pearson(rvalue, pvalue, rthresh=0.9, pmax=1.0e-4):
    """Evaluate the Pearson correlation coefficient and p-value.

    Parameters
    ----------
    rvalue : float
        Pearson correlation coefficient
    pvalue : float
        Pearson p-value for testing non-correlation
    rthresh : float
        Minimum acceptable correlation coefficient (default=0.8)
    pmax : float
        Maximum acceptable p-value (default=1.0e-4)

    Returns
    -------
    bool
        True if thresholds pass, False if they fail

    """

    if rvalue is None or pvalue is None:
        logger.info('Did not supply all Pearson values for evaluation.')
        return False

    if np.isfinite(rvalue) and np.isfinite(pvalue):
        return rvalue >= rthresh and pvalue <= pmax
    else:
        logger.info('Not all Pearson values are finite.')
        return False


def evaluate_dayglow(fit_coeff, locations, thresh=300.0):
    """Evaluate the dayglow level across a range of locations.

    Parameters
    ----------
    fit_coeff : array-like
        Coefficients, the first three of which are the quadratic constant,
        x-term, and x-squared-term.
    locations : array-like
        Locations at which the dayglow level will be evaluated.
    thresh : float
        Minimum allowable background intensity value in Rayleighs (default=300)

    Returns
    -------
    good : bool
        True if the background level is low across all desired locations, False
        if any location is too high.

    """
    # Ensure the locations are array-like
    locations = np.asarray(locations)
    if locations.shape == ():
        locations = np.array([locations])

    # Calculate the background at all locations
    bg = calc_quadratic(locations, fit_coeff[0], fit_coeff[1], fit_coeff[2])

    # Evaluate the background
    good = True if np.all(bg <= thresh) else False

    return good


def compare_boundaries(rvalue, pvalue, eq_bounds, eq_uncert, po_bounds,
                       po_uncert, min_mlat, max_mlat, max_uncert=3.0):
    """Compare different boundaries and choose the best values.

    Parameters
    ----------
    rvalue : list-like
        Pearson correlation coefficient
    pvalue : list-like
        Pearson p-value for testing non-correlation
    eq_bounds : float
        Equatorward auroral boundary location in degrees latitude
    eq_uncert : float
        Equatorward auroral boundary uncertainty in degrees latitude
    po_bounds : float
        Polar auroral boundary location in degrees latitude
    po_uncert : float
        Polar auroral boundary uncertainty in degrees latitude
    min_mlat : float
        Minimum latitude in intensity slice in degrees
    max_mlat : float
        Maximum latitude in intensity slice in degrees
    max_uncert : float
        Maximum allowable boundary uncertainty in degrees (default=3.0)

    Returns
    -------
    igood_eq : int or NoneType
        Index corresponding to the best equatorial index, or None
    igood_po : int or NoneType
        Index corresponding to the best polar index, or None

    """
    # Initialize the output
    igood_eq = None
    igood_po = None

    # Determine which fit is the best
    ifit = compare_pearson(rvalue, pvalue)

    if ifit is not None:
        # Test the equatorward boundaries for the best value
        if not np.isfinite(eq_bounds[ifit]) or abs(eq_bounds[ifit]) <= abs(
                min_mlat) or abs(eq_bounds[ifit]) >= abs(max_mlat) or eq_uncert[
                    ifit] > max_uncert:
            ieqs = [i for i, eq in enumerate(eq_bounds) if i != ifit
                    and eq is not None and np.isfinite(eq)
                    and abs(eq) > abs(min_mlat) and abs(eq) < abs(max_mlat)
                    and np.isfinite(eq_uncert[i])
                    and eq_uncert[i] <= max_uncert]

            if len(ieqs) == 1:
                igood_eq = ieqs[0]
            elif len(ieqs) > 0:
                isel = compare_pearson(np.asarray(rvalue)[ieqs],
                                       np.asarray(pvalue)[ieqs])

                if isel is not None:
                    igood_eq = ieqs[isel]
        else:
            igood_eq = ifit

        # Test the poleward boundaries for the best value
        if not np.isfinite(po_bounds[ifit]) or abs(po_bounds[ifit]) >= abs(
                max_mlat) or abs(po_bounds[ifit]) <= abs(min_mlat) or po_uncert[
                    ifit] > max_uncert:
            ipos = [i for i, po in enumerate(po_bounds) if i != ifit
                    and po is not None and np.isfinite(po)
                    and abs(po) < abs(max_mlat) and np.isfinite(po_uncert[i])
                    and po_uncert[i] <= max_uncert and abs(po) > abs(min_mlat)]

            if len(ipos) == 1:
                igood_po = ipos[0]
            elif len(ipos) > 0:
                isel = compare_pearson(np.asarray(rvalue)[ipos],
                                       np.asarray(pvalue)[ipos])

                if isel is not None:
                    igood_po = ipos[isel]
        else:
            igood_po = ifit

    return igood_eq, igood_po


def compare_pearson(rvalue, pvalue):
    """Evaluate different Pearson correlation coefficients and choose the best.

    Parameters
    ----------
    rvalue : list-like
        Pearson correlation coefficient
    pvalue : list-like
        Pearson p-value for testing non-correlation

    Returns
    -------
    igood : int or NoneType
        Index correspeonding to the best fit, None if no fit is good

    """
    # Set the default output for no good fits
    igood = None

    # Replace NoneType with NaN
    p_value = np.asarray([np.nan if pval is None else pval for pval in pvalue])
    r_value = np.asarray([np.nan if rval is None else rval for rval in rvalue])

    # Identify which fits pass the threshold for correlation
    good_pval = np.where(np.isfinite(p_value) & (p_value <= 1.0e-4))[0]

    if len(good_pval) == 1:
        # There is only one value that passes the non-correlation test
        igood = good_pval[0]
    elif len(good_pval) > 0:
        # The rvalue ranges from -1 (anticorrelation) to 1 (correlation)
        igood = good_pval[r_value[good_pval].argmax()]

    return igood


def evaluate_boundary_in_mlt(bound_data, eq_key, po_key, lt_key, ut_key,
                             lt_bin=5.0, max_iqr=1.5):
    """Evaluate boundary consistency with local time.

    Parameters
    ----------
    bound_data : xr.Dataset
        Boundary data stored in an xarray Dataset
    eq_key : str
        Data key for the equatorward boundary data
    po_key : str
        Data key for the poleward boundary data
    lt_key : str
        Coordinate key for the MLT data
    ut_key : str
        Coordinate key for the UT data
    lt_bin : float
        Size of local time bin in hours over which the data will be evaluated
        (default=5.0)
    max_iqr : float
        Maximum multiplier for the interquartile range (IQR) used to identify
        outliers above or below the upper or lower quartile (default=1.5)

    Returns
    -------
    bdata : xr.Dataset
        Dataset with only good boundaries

    """
    # Test the input
    if np.any([key not in bound_data.data_vars for key in [eq_key, po_key]]):
        raise ValueError('bad boundary key(s)')

    if np.any([key not in bound_data.coords for key in [lt_key, ut_key]]):
        raise ValueError('unknown time coordinate(s)')

    # Initialize the output
    bdata = bound_data.copy()

    # Get the padded local time
    lt = list(bound_data[lt_key].values - 24.0)
    lt.extend(list(bound_data[lt_key].values))
    lt.extend(list(bound_data[lt_key].values + 24.0))

    # Determine the number of lt values in each window
    delta_lt = grids.unique(np.array(lt)[1:] - np.array(lt)[:-1])
    if len(delta_lt) != 1:
        raise ValueError('local time does not have a fixed frequency')

    nbin = int(lt_bin / delta_lt[0])

    for i, utime in enumerate(bound_data[ut_key]):
        # Evaluate the boundaries
        for bkey in [eq_key, po_key]:
            if np.isfinite(bound_data[bkey].values).any():
                # Pad the boundaries
                bound = list(bound_data[bkey][i].values)
                bound.extend(list(bound_data[bkey][i].values))
                bound.extend(list(bound_data[bkey][i].values))

                # Save the data as a Pandas series
                bounds = pds.Series(bound, index=lt)

                # Get the running mean and standard deviation
                broll = bounds.rolling(nbin, min_periods=1, center=True,
                                       closed='both')
                bplus = broll.quantile(0.75)
                bminus = broll.quantile(0.25)
                iqr = bplus - bminus

                # Determine which values should be removed
                bbad = (bounds > bplus + max_iqr * iqr) | (
                    bound < bminus - max_iqr * iqr)

                if bbad.any():
                    # If there is data to remove, update the output
                    bmask = bbad[bdata[lt_key].values].values
                    bdata[bkey][i][bmask] = np.nan

    return bdata
