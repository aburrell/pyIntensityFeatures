#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
"""Functions used for fitting functions to intensity data."""

import numpy as np
from scipy.optimize import leastsq
from scipy import stats

from pyIntensityFeatures.utils import checks
from pyIntensityFeatures.utils import distributions


def estimate_peak_widths(data, data_loc, peak_inds, peak_mags):
    """Estimate the full width at half max of the peaks in a curve.

    Parameters
    ----------
    data : array-like
        Input data (y-axis)
    data_loc : array-like
        Input data location (x-axis)
    peak_inds : array-like
        Indexes of the peaks
    peak_mags : array-like
        Magnitude of the peaks

    Returns
    -------
    peak_sigmas : list-like
        FWHM values for each peak, or the closest estimate possible

    Notes
    -----
    Differs from Longden, et al. (2010) function by not assuming a fixed data
    location increment of 1.0 degrees.

    """
    # Ensure the input is array-like
    data = np.asarray(data)
    data_loc = np.asarray(data_loc)
    peak_inds = np.asarray(peak_inds)
    peak_mags = np.asarray(peak_mags)

    # Get the half-max for each peak
    half_max = 0.5 * peak_mags
    peak_sigmas = list()

    for i, ipeak in enumerate(peak_inds):
        # Step through all values of the curve data (`data`) with lower indices
        # than this peak until either the half-max threshold or another peak is
        # reached
        reached = False
        good_down = False
        j = ipeak
        while not reached and j > 0:
            if (data[j] >= half_max[i]) and (data[j - 1] <= half_max[i]):
                reached = True
                fwhm_loc = data_loc[j - 1] + (
                    data_loc[j] - data_loc[j - 1]) * (
                        half_max[i] - data[j - 1]) / (data[j] - data[j - 1])
                width_down = data_loc[ipeak] - fwhm_loc
                good_down = True
            else:
                if data[j] < data[j - 1]:
                    # Data is no longer decreasing. Since the correct width was
                    # not reached before finding the desired half-max value,
                    # set an approximate value
                    reached = True
                    width_down = data_loc[ipeak] - data_loc[j - 1]
            j -= 1

        if not good_down and not reached:
            # Reached edge of data before defining the width
            width_down = data_loc[ipeak] - data_loc[j]

        # Step through all values of the curve data (`data`) with higher
        # indices than this peak until either the half-max threshold or another
        # peak is reached
        reached = False
        good_up = False
        j = ipeak
        while not reached and j < len(data) - 1:
            if (data[j] >= half_max[i]) and (data[j + 1] <= half_max[i]):
                reached = True
                fwhm_loc = data_loc[j + 1] + (
                    data_loc[j] - data_loc[j + 1]) * (
                        half_max[i] - data[j + 1]) / (data[j] - data[j + 1])
                width_up = fwhm_loc - data_loc[ipeak]
                good_up = True
            else:
                if data[j] < data[j + 1]:
                    # Data is no longer decreasing. Since the correct width was
                    # not reached before finding the desired half-max value,
                    # set an approximate value
                    reached = True
                    width_up = data_loc[j + 1] - data_loc[ipeak]
            j += 1

        if not good_up and not reached:
            # Reached edge of data before defining the width
            width_up = data_loc[j] - data_loc[ipeak]

        # Calculate the peak sigmas using logic based on whether or not the
        # HWHM was found on each leg
        if good_up and good_down:
            peak_sigmas.append((width_up + width_down) / checks.fwhm_const)
        elif good_up:
            peak_sigmas.append((2.0 * width_up) / checks.fwhm_const)
        elif good_down:
            peak_sigmas.append((2.0 * width_down) / checks.fwhm_const)
        else:
            peak_sigmas.append(2.0 * min([width_up, width_down])
                               / checks.fwhm_const)

    return peak_sigmas


def gauss_quad_err(params, xvals, yvals, weights):
    """Calculate the error of a quadriatic Gaussian fit.

    Parameters
    ----------
    params : set
        Set of parameters used by `mult_gauss_quad`
    xvals : float or array-like
        Location values
    yvals : float or array-like
        Intensity values
    weights : float or array-like
        Weights for each data value

    Returns
    -------
    err : float or array-like
        Weighted error of the fit defined by `params`

    See Also
    --------
    utils.distributions.mult_gauss_quad

    """
    # Define the error function to fit
    err = (distributions.mult_gauss_quad(xvals, params) - yvals) * weights

    return err


def get_gaussian_func_fit(mlat_bins, mlt_bins, mean_intensity, std_intensity,
                          num_intensity, num_gauss=3, min_num=3,
                          min_intensity=0.0, min_lat_perc=70.0):
    """Fit intensity data using least-squares minimization in MLT bins.

    Parameters
    ----------
    mlat_bins : np.array
        Magnetic latitude of output bin centres
    mlt_bins : np.array
        Magnetic local time of output bin centres
    mean_intensity : np.array
       2D array of mean intensity values
    std_intensity : np.array
       2D array of intensity standard deviations
    num_intensity : np.array
       2D array of intensity counts per bin
    num_gauss : int
        Maximum number of Gaussians to fit (default=3)
    min_num : int
        Minimum number of samples contributing to the intensity mean
        (default=3)
    min_intensity : float
        Minimum intensity magnitude in Rayleighs (default=0.0)
    min_lat_perc : int
        Minimum percentage of latitude bins needed to define an intensity
        function (default=70.0)

    Returns
    -------
    func_params : list-like
        Set of parameters used by `mult_gauss_quad` for each MLT bin that
        defines the mean intensity as a function of magnetic latitude or set
        to a string output with a reason if no successful fit was performed.
    fit_cov : list-like
        Covariance matrix for each fit
    fit_pearsonr : list-like
        Pearson r-value for correlation between the observations and the fit
    fit_pearsonp : list-like
        Pearson p-value for correlation between the observations and the fit
    num_peaks : list-like
        Number of peaks used in the fit

    See Also
    --------
    utils.distributions.mult_gauss_quad

    """
    # Initialize the output
    func_params = list()
    fit_cov = list()
    fit_pearsonr = list()
    fit_pearsonp = list()
    num_peaks = list()

    # Get the percentage multiplier
    perc_mult = 100.0 / len(mlat_bins)

    # Loop through each MLT bin
    for ilt, mlt in enumerate(mlt_bins):
        ilats = np.where(np.isfinite(mean_intensity[:, ilt])
                         & (num_intensity[:, ilt] >= min_num)
                         & (mean_intensity[:, ilt] >= min_intensity))[0]

        if len(ilats) * perc_mult >= min_lat_perc:
            # Perform a first-order polynomial fit to the data to obtain
            # estimates for the background levels in the intensity profile. The
            # first term is the slope and the second term is the intercept.
            bg = np.polyfit(mlat_bins[ilats], mean_intensity[ilats, ilt], 1)

            # Normalize the intensity curve
            norm_intensity = (mean_intensity[ilats, ilt]
                              - mean_intensity[ilats, ilt].min()) / (
                                  mean_intensity[ilats, ilt].max()
                                  - mean_intensity[ilats, ilt].min())

            # Find the main peak and its characteristics
            ipeaks = [norm_intensity.argmax()]
            peak_sigmas = estimate_peak_widths(norm_intensity,
                                               mlat_bins[ilats], ipeaks,
                                               [norm_intensity[ipeaks[0]]])

            # Locate any additional peaks, up to a desired maximum
            mlt_gauss = num_gauss
            while len(ipeaks) < mlt_gauss:
                # Get a Gaussian for the prior peak
                prior_gauss = distributions.gauss(
                    mlat_bins[ilats], norm_intensity[ipeaks[-1]],
                    mlat_bins[ilats][ipeaks[-1]], peak_sigmas[-1], 0.0)

                # Remove the gaussian fit from the data
                norm_intensity -= prior_gauss

                # Get the next peak
                inew = norm_intensity.argmax()

                # Test to see if the new peak is significant
                if not np.any((mlat_bins[ilats][inew]
                               > mlat_bins[ilats][ipeaks]
                               - np.asarray(peak_sigmas) * 2.0)
                              & (mlat_bins[ilats][inew]
                                 < mlat_bins[ilats][ipeaks]
                                 + np.asarray(peak_sigmas) * 2.0)):
                    ipeaks.append(inew)
                    peak_sigmas.extend(estimate_peak_widths(
                        norm_intensity, mlat_bins[ilats], [ipeaks[-1]],
                        [norm_intensity[ipeaks[-1]]]))
                else:
                    mlt_gauss = len(ipeaks)

            # Use the peak information to set the Gaussian fitting parameters
            params = [bg[1], bg[0], 0.0]
            for i, ipeak in enumerate(ipeaks):
                params.extend([mean_intensity[ilats, ilt][ipeak],
                               mlat_bins[ilats][ipeak], peak_sigmas[i]])

            # Find the desired Gaussian + Quadratic fit using least-squares
            # if there are enough latitudes to provide a fit
            lsq_args = (mlat_bins[ilats], mean_intensity[ilats, ilt],
                        1.0 / std_intensity[ilats, ilt])
            num_peaks.append(len(ipeaks))
            if len(params) <= len(ilats):
                lsq_result = leastsq(gauss_quad_err, params, args=lsq_args,
                                     full_output=True)

                # Evaluate the least squares output and save the results
                if lsq_result[-1] in [1, 2, 3, 4]:
                    gauss_out = distributions.mult_gauss_quad(
                        mlat_bins[ilats], lsq_result[0])
                    fmask = np.isfinite(gauss_out)
                    if fmask.sum() > 3:
                        pres = stats.pearsonr(
                            mean_intensity[ilats, ilt][fmask],
                            gauss_out[fmask])
                        func_params.append(lsq_result[0])

                        # As of scipy 1.11.0 the output changed
                        if hassattr(pres, 'statistic'):
                            fit_pearsonr.append(pres.statistic)
                            fit_pearsonp.append(pres.pvalue)
                        else:
                            fit_pearsonr.append(pres[0])
                            fit_pearsonp.append(pres[1])

                        fit_cov.append(lsq_result[1])
                        found_fit = True
                    else:
                        found_fit = False
                else:
                    found_fit = False
            else:
                found_fit = False
                lsq_result = ["insufficient mlat coverage for expected peaks",
                              -1]

            if not found_fit:
                while not found_fit and len(ipeaks) > 1:
                    # Try reducing the number of peaks to obtain a valid fit
                    ipeaks.pop()
                    params = list(np.array(params)[:-3])

                    if len(params) <= len(ilats):
                        lsq_result = leastsq(gauss_quad_err, params,
                                             args=lsq_args, full_output=True)
                        num_peaks[-1] = len(ipeaks)
                        if lsq_result[-1] in [1, 2, 3, 4]:
                            gauss_out = distributions.mult_gauss_quad(
                                mlat_bins[ilats], lsq_result[0])
                            gmask = np.isfinite(gauss_out)

                            if sum(gmask) > 3:
                                pres = stats.pearsonr(
                                    mean_intensity[ilats, ilt][gmask],
                                    gauss_out[gmask])
                                found_fit = True
                                func_params.append(lsq_result[0])
                                fit_pearsonr.append(pres.statistic)
                                fit_pearsonp.append(pres.pvalue)
                                fit_cov.append(lsq_result[1])
                if not found_fit:
                    func_params.append(lsq_result[-2])
                    fit_cov.append(None)
                    fit_pearsonr.append(None)
                    fit_pearsonp.append(None)
        else:
            func_params.append("insufficient mlat coverage ({:} < {:})".format(
                len(ilats), min_lat_perc / perc_mult))
            fit_cov.append(None)
            fit_pearsonr.append(None)
            fit_pearsonp.append(None)
            num_peaks.append(0)

    return func_params, fit_cov, fit_pearsonr, fit_pearsonp, num_peaks
