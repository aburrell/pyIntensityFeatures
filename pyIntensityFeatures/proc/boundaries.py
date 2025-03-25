#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Functions for identifying auroral oval luminosity boundaries."""

import numpy as np

from pyIntensityFeatures.utils import checks


def locate_boundaries(fit_coeff, fit_covar, dominant_fit, min_mlat, max_mlat,
                      method='best', max_peak_diff=5.0, strict_fit=False):
    """Locate auroral luminosity boundaries using the Longden method.

    Parameters
    ----------
    fit_coeff : array-like
        Fit coefficients constant, quadratic multiplier for x, quadratic
        multiplier for x^2, and Gaussian amplitudes, x offsets, and exponential
        scalers for each Gaussian. The number of each Gaussian group must be
        the same; e.g., there must be two of each amplitude, x offset, and
        exponential scalers, but only one constant and quadratic multipliers.
    fit_covar : array-like
        Covarience matrix for the `fit_coeff` values
    dominant_fit : int
        Integer specifying whether the dominant fit is single (1), double (2),
        or multi (any integer) peaked.  If the 'single' or 'mult' method is
        specified, this must correspond to the desired method or no boundaries
        will be calculated.
    min_mlat : float
        Minimum latitude used to obtain fit in degrees
    max_mlat : float
        Maximum latitude used to obtain fit in degrees
    method : str
        Specify which method to use, single Gaussian ('single'), multi-peak
        Gaussian ('mult'), or use `dominant_fit` to identify the most
        appropriate method ('best'). (default='best')
    max_peak_diff : float
        For multi-peak fits, the maximum allowable difference between peak
        locations to be considered for boundary selection relative to the
        primary peak (default=5.0)
    strict_fit : bool
        Enforce positive values for the x-offsets in `fit_coeff` (default=False)

    Returns
    -------
    eq_bound : float or array-like
        Equatorial boundary of the auroral oval, NaN if not calculated
    po_bound : float or array-like
        Poleward boundary of the auroral oval, NaN if not calculated
    un_bound_eq : float or array-like
        Uncertainty of the auroral oval equatorward boundary, NaN if not
        calculated
    un_bound_po : float or array-like
        Uncertainty of the auroral oval poleward boundary, NaN if not
        calculated

    Raises
    ------
    ValueError
        If an unknown method is provided

    References
    ----------
    Longden, N. S., et al. (2010) Estimating the location of the open-closed
    magnetic field line boundary from auroral images, 28 (9), p 1659-1678,
    doi:10.5194/angeo-28-1659-2010.

    """
    if method.lower() not in ['single', 'mult', 'best']:
        raise ValueError("unexpected method, try: 'single', 'mult', or 'best'")

    if dominant_fit == 1 and method.lower() in ['single', 'best']:
        # Find the PALB and EALB using a single Gaussian peak
        (eq_bound, po_bound, un_bound_eq,
         un_bound_po) = locate_single_peak_boundaries(
             fit_coeff[4], fit_coeff[5], fit_covar, min_mlat, max_mlat,
             strict_fit)
    elif dominant_fit > 1 and method.lower() in ['mult', 'best']:
        # Find the PALB and EALB from multiple Gaussian peaks
        (eq_bound, po_bound, un_bound_eq,
         un_bound_po) = locate_mult_peak_boundaries(fit_coeff, fit_covar,
                                                    dominant_fit, min_mlat,
                                                    max_mlat, max_peak_diff,
                                                    strict_fit)
    else:
        # The provided fit and requested method do not allow evaluation
        eq_bound = np.nan
        po_bound = np.nan
        un_bound_eq = np.nan
        un_bound_po = np.nan

    return eq_bound, po_bound, un_bound_eq, un_bound_po


def calc_boundary_uncertainty(peak_number, fit_covar):
    """Calculate the uncertainty of a boundary location.

    Parameters
    ----------
    peak_number : int
        1-offset number of the primary peak (e.g., 1, 2, 3)
    fit_covar : array-like
        Covarience matrix for the `fit_coeff` values

    Returns
    -------
    un_bound : float
        Boundary uncertainty

    """
    # Calculate the uncertainty of the coefficients from the covariance
    sigma = np.sqrt(np.diagonal(fit_covar))

    # Peak numbers start from 1, not 0, as they are physical quantities
    peak_offset = (peak_number - 1) * 3

    # Set the constants and variables for the calculation
    u_mu = sigma[peak_offset + 4]  # Corresponding to the x offset
    del_mu = 1.0
    u_sigma = sigma[peak_offset + 5]  # Corresponding to the exp scalar
    covar_mu_sigma = fit_covar[peak_offset + 4, peak_offset + 5]

    # Calculate the components of the uncertainty
    weighted_uncer_mu = (del_mu**2) * (u_mu**2)
    weighted_uncer_sigma = (checks.fwhm_const**2) * (u_sigma**2)
    weighted_uncer_mu_sigma = 2.0 * del_mu * checks.fwhm_const * covar_mu_sigma

    # Calculate the uncertainty
    un_bound = np.sqrt(weighted_uncer_mu + weighted_uncer_sigma
                       + weighted_uncer_mu_sigma)

    return un_bound


def locate_single_peak_boundaries(fit_mu, fit_sigma, fit_covar, min_mlat,
                                  max_mlat, strict_fit=False):
    """Locate auroral luminosity boundaries assuming a single peak.

    Parameters
    ----------
    fit_mu : float
        Gaussian x offset from a single-peak fit, or dominant peak of a
        multi-peak fit.
    fit_sigma : float
        Gaussian exponential scalar from a single-peak fit, or dominant peak of
        a multi-peak fit.
    fit_covar : array-like
        Covarience matrix for the `fit_coeff` values
    min_mlat : float
        Minimum latitude used to obtain fit in degrees
    max_mlat : float
        Maximum latitude used to obtain fit in degrees
    strict_fit : bool
        Enforce positive values for `fit_sigma` (default=False)

    Returns
    -------
    eq_bound : float or array-like
        Equatorial boundary of the auroral oval, NaN if not calculated
    po_bound : float or array-like
        Poleward boundary of the auroral oval, NaN if not calculated
    un_bound_eq : float or array-like
        Uncertainty of the auroral oval equatorward boundary, NaN if not
        calculated
    un_bound_po : float or array-like
        Uncertainty of the auroral oval poleward boundary, NaN if not
        calculated

    References
    ----------
    Longden, N. S., et al. (2010) Estimating the location of the open-closed
    magnetic field line boundary from auroral images, 28 (9), p 1659-1678,
    doi:10.5194/angeo-28-1659-2010.

    """
    if fit_sigma > 0.0 or not strict_fit:
        # Calculate the FWHM for a single/first Gaussian peak and use it to
        # find the boundaries
        delta_lambda = checks.fwhm_const * abs(fit_sigma)
        hemi = np.sign(fit_mu)
        eq_bound = fit_mu - hemi * delta_lambda
        po_bound = fit_mu + hemi * delta_lambda

        # Calcualte the uncertainty contribution
        un_bound_eq = calc_boundary_uncertainty(1, fit_covar)
        un_bound_po = un_bound_eq

        # Ensure the boundaries encompass the fit region when considering
        # the uncertainty
        if abs(eq_bound) + un_bound_eq < abs(min_mlat):
            eq_bound = np.nan

        if abs(po_bound) - un_bound_po > abs(max_mlat):
            po_bound = np.nan
    else:
        # If the spread is a negative value, this is not a valid fit
        eq_bound = np.nan
        po_bound = np.nan
        un_bound_eq = np.nan
        un_bound_po = np.nan

    return eq_bound, po_bound, un_bound_eq, un_bound_po


def locate_mult_peak_boundaries(fit_coeff, fit_covar, dominant_fit, min_mlat,
                                max_mlat, max_peak_diff=5.0, strict_fit=False):
    """Locate auroral luminosity boundaries assuming a single peak.

    Parameters
    ----------
    fit_coeff : array-like
        Fit coefficients constant, quadratic multiplier for x, quadratic
        multiplier for x^2, and Gaussian amplitudes, x offsets, and exponential
        scalers for each Gaussian. The number of each Gaussian group must be
        the same; e.g., there must be two of each amplitude, x offset, and
        exponential scalers, but only one constant and quadratic multipliers.
    fit_covar : array-like
        Covarience matrix for the `fit_coeff` values
    dominant_fit : int
        Integer specifying the number of peaks used in the Gaussian fit.
    min_mlat : float
        Minimum latitude used to obtain fit in degrees
    max_mlat : float
        Maximum latitude used to obtain fit in degrees
    max_peak_diff : float
        For multi-peak fits, the maximum allowable difference between peak
        locations to be considered for boundary selection relative to the
        primary peak (default=5.0)
    strict_fit : bool
        Enforce positive values for the x-offsets in `fit_coeff` (default=False)

    Returns
    -------
    eq_bound : float or array-like
        Equatorial boundary of the auroral oval, NaN if not calculated
    po_bound : float or array-like
        Poleward boundary of the auroral oval, NaN if not calculated
    un_bound_eq : float or array-like
        Uncertainty of the auroral oval equatorward boundary, NaN if not
        calculated
    un_bound_po : float or array-like
        Uncertainty of the auroral oval poleward boundary, NaN if not
        calculated

    References
    ----------
    Longden, N. S., et al. (2010) Estimating the location of the open-closed
    magnetic field line boundary from auroral images, 28 (9), p 1659-1678,
    doi:10.5194/angeo-28-1659-2010.

    """
    # Initialize the list of valid boundaries, uncertainties, and amplities
    eq_bounds = list()
    po_bounds = list()
    un_bounds = list()
    amp_coeff = list()

    # Determine the hemisphere and peak widths, if appropriate fits are
    # provided
    hemi = np.sign(fit_coeff[4])
    delta_lambda = [checks.fwhm_const * abs(fit_coeff[i * 3 + 5])
                    if fit_coeff[i * 3 + 5] > 0.0 or not strict_fit else np.nan
                    for i in range(dominant_fit)]

    # Cycle through each peak
    for i in range(dominant_fit):
        # Evaluate the spacing between peaks and their location
        if i > 0:
            iloc = fit_coeff[i * 3 + 4]
            if abs(iloc) < abs(min_mlat) or abs(iloc) > abs(
                    max_mlat) or np.isnan(delta_lambda[i]):
                peak_diffs = [False]
            else:
                peak_diffs = list()
                for j in range(dominant_fit):
                    if j != i and np.isfinite(delta_lambda[j]):
                        jloc = fit_coeff[j * 3 + 4]
                        # If the peaks are both valid fits and fall within the
                        # FWHM, they're overlapping and pass
                        if iloc > jloc:
                            if (iloc - delta_lambda[i] <= jloc) or (
                                    iloc <= jloc + delta_lambda[j]):
                                peak_diffs.append(True)
                            else:
                                peak_diffs.append(
                                    (iloc - delta_lambda[i])
                                    - (jloc + delta_lambda[j])
                                    < max_peak_diff)
                        else:
                            if (jloc - delta_lambda[j] <= iloc) or (
                                    jloc <= iloc + delta_lambda[i]):
                                peak_diffs.append(True)
                            else:
                                peak_diffs.append(
                                    (jloc - delta_lambda[j])
                                    - (iloc + delta_lambda[i])
                                    < max_peak_diff)
        else:
            # Always evaluate the principle peak
            peak_diffs = [True]

        if np.any(peak_diffs):
            eq_bounds.append(fit_coeff[i * 3 + 4] - hemi * delta_lambda[i])
            po_bounds.append(fit_coeff[i * 3 + 4] + hemi * delta_lambda[i])
            amp_coeff.append(fit_coeff[i * 3 + 3] > 0)

            # Calcualte the uncertainty contribution
            un_bounds.append(calc_boundary_uncertainty(i + 1, fit_covar))

            # Ensure the boundaries encompass the fit region when
            # considering the uncertainty
            if abs(eq_bounds[-1]) + un_bounds[-1] < abs(min_mlat):
                eq_bounds[-1] = np.nan

            if abs(po_bounds[-1]) - un_bounds[-1] > abs(max_mlat):
                po_bounds[-1] = np.nan

    # Evaluate the PALB and EALB, selecting the most appropriate
    if sum(amp_coeff) == 1:
        iamp = amp_coeff.index(True)

        po_bound = po_bounds[iamp]
        un_bound_po = un_bounds[iamp]
        eq_bound = eq_bounds[iamp]
        un_bound_eq = un_bounds[iamp]

        for j, ebnd in enumerate(eq_bounds):
            if j != iamp and (eq_bounds[j] < eq_bound
                              or (not np.isfinite(eq_bound)
                                  and eq_bounds[j] < po_bound)):
                eq_bound = eq_bounds[j]
                un_bound_eq = un_bounds[j]
    else:
        # Determine the best poleward boundary
        try:
            iamp = np.nanargmax(abs(np.array(po_bounds)))
            po_bound = po_bounds[iamp]
            un_bound_po = un_bounds[iamp]
        except ValueError:
            po_bound = np.nan
            un_bound_po = np.nan

        # Determine the best equatorward boundary
        try:
            iamp = np.nanargmin(abs(np.array(eq_bounds)))
            eq_bound = eq_bounds[iamp]
            un_bound_eq = un_bounds[iamp]
        except ValueError:
            eq_bound = np.nan
            un_bound_eq = np.nan

    return eq_bound, po_bound, un_bound_eq, un_bound_po


def get_eval_boundaries(fit_coeff, fit_cov, rvalue, pvalue, num_peaks,
                        mlat_min, mlat_max, method, un_threshold=1.25,
                        dayglow_threshold=300.0, strict_fit=False):
    """Find and evaluate the PALB and EALB for a provided fit.

    Parameters
    ----------
    fit_coeff : array-like
        Fit coefficients constant, quadratic multiplier for x, quadratic
        multiplier for x^2, and Gaussian amplitudes, x offsets, and exponential
        scalers for each Gaussian. The number of each Gaussian group must be
        the same; e.g., there must be two of each amplitude, x offset, and
        exponential scalers, but only one constant and quadratic multipliers.
    fit_covar : array-like
        Covarience matrix for the `fit_coeff` values
    rvalue : float
        Pearson correlation coefficient
    pvalue : float
        Pearson p-value
    num_peaks : int
        Number of Gaussian peaks in the fit.
    min_mlat : float
        Minimum latitude used to obtain fit in degrees.
    max_mlat : float
        Maximum latitude used to obtain fit in degrees.
    method : str
        Specify which method to use, single Gaussian ('single'), multi-peak
        Gaussian ('mult'), or use `dominant_fit` to identify the most
        appropriate method ('best'). (default='best')
    un_threshold : float
        Maximum acceptable uncertainty value in degrees (default=1.25)
    dayglow_threshold : float
        Minimum allowable background intensity value in Rayleighs (default=300)
    strict_fit : bool
        Enforce positive values for the x-offsets in `fit_coeff` (default=False)

    Returns
    -------
    bounds : list
        List of floats containing the EALB, PALB, EALB uncertaintly, and PALB
        uncertainty in that order.  NaN if no realistic boundaries were found.
    good_bound : bool
        True if the boundaries pass all tests, False otherwise.

    """
    # Get the boundaries from the provided quadriatic Gaussian fit
    bounds = locate_boundaries(fit_coeff, fit_cov, num_peaks, mlat_min,
                               mlat_max, method, strict_fit=strict_fit)

    # Evaluate the background level
    good_bound = checks.evaluate_dayglow(fit_coeff, [bounds[0], bounds[1]],
                                         thresh=dayglow_threshold)

    if good_bound:
        # Evaluate the robustness of the boundaries
        good_bound = checks.evaluate_gauss_quad(
            num_peaks, fit_coeff, rvalue, pvalue, bounds[1], bounds[3],
            bounds[0], bounds[2], mlat_min, mlat_max, un_threshold=un_threshold)
    else:
        bounds = [np.nan, np.nan, np.nan, np.nan]

    return bounds, good_bound
