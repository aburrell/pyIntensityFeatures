#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `proc.fitting`."""

import numpy as np
import unittest

from pyIntensityFeatures.utils.distributions import mult_gauss_quad
from pyIntensityFeatures.proc import fitting


class TestFittingFuncs(unittest.TestCase):
    """Tests for the fitting functions."""

    def setUp(self):
        """Set up the test runs."""
        self.coeffs = [1.0, 0.01, 0.001, 500.0, 75.0, 3.0, 400, 79.0, 1.0,
                       300, 65.0, 0.5]
        self.mlat_bins = np.arange(60, 90.0, 1.5)
        self.mlt_bins = np.arange(0.0, 24.0, 1.0)
        self.mean_intensity = np.full(
            shape=(self.mlt_bins.shape[0], self.mlat_bins.shape[0]),
            fill_value=mult_gauss_quad(
                self.mlat_bins, *self.coeffs)).transpose()
        self.std_intensity = np.random.normal(0.5, 0.1,
                                              size=self.mean_intensity.shape)
        self.num_intensity = np.random.normal(
            50, 30, size=self.mean_intensity.shape).astype(int)
        self.num_intensity[self.num_intensity < 0] = 0
        self.num_gauss = 3
        self.min_num = 3
        self.min_intensity = 10.0
        self.min_lat_perc = 70.0
        self.out = None
        self.num_bad = 0
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.coeffs, self.mlat_bins, self.mlt_bins, self.mean_intensity
        del self.std_intensity, self.num_intensity, self.num_gauss, self.min_num
        del self.min_lat_perc, self.out, self.num_bad
        return

    def get_peak_inds(self):
        """Get the MLat indices of the coefficient peaks."""

        self.min_num = []
        self.min_intensity = []
        for peak_loc in [self.coeffs[4], self.coeffs[7], self.coeffs[10]]:
            self.min_num.append(abs(self.mlat_bins - peak_loc).argmin())
            self.min_intensity.append(self.mean_intensity[self.min_num[-1], 0])
        return

    def eval_gaussian_fit_out(self, hemisphere):
        """Evaluate the output of `get_gaussian_func_fit`.

        Parameters
        ----------
        hemisphere : int
           -1 for Southern hemisphere, 1 for Northern hemisphere.

        """
        self.num_bad = 0
        messages = ["insufficient mlat coverage for expected peaks",
                    "insufficient mlat coverage"]

        # Evaluate the number of peaks, which should not exceed the maximum
        self.assertEqual(len(self.out[-1]), len(self.mlt_bins))
        self.assertTrue(
            np.all([npeak <= self.num_gauss for npeak in self.out[-1]]),
            msg="A fit with more peaks than allowed was found")

        # Evaluate the fit values and statistics
        for i, rvalue in enumerate(self.out[-3]):
            if rvalue is None:
                self.num_bad += 1

                # Evaluate the Pearson correlation coefficients
                self.assertIsNone(self.out[-2][i],
                                  msg="Mismatched Pearson values")

                # Evaluate the coefficients and covariance
                self.assertIsNone(self.out[1][i])

                has_msg = False
                for msg in messages:
                    if self.out[0][i].find(msg):
                        has_msg = True
                        break

                self.assertTrue(
                    has_msg,
                    msg="unexpected coefficient message: {:}".format(
                        self.out[0][i]))
            else:
                # Evaluate the Pearson correlation coefficients
                self.assertIsNotNone(self.out[-2][i],
                                     msg="Mismatched Pearson values")
                self.assertGreaterEqual(rvalue, -1.0)
                self.assertLessEqual(rvalue, 1.0)
                self.assertGreaterEqual(self.out[-2][i], 0.0)
                self.assertLessEqual(self.out[-2][i], 1.0)

                # Evaluate the coefficients and covariance shape
                self.assertTrue(np.all(np.isfinite(self.out[0][i])))
                if self.out[1][i] is None:
                    self.num_bad += 1
                else:
                    self.assertTupleEqual((self.out[0][i].shape[0],
                                           self.out[0][i].shape[0]),
                                          self.out[1][i].shape)
                    self.assertTrue(np.all(np.isfinite(self.out[1][i])))
        return

    def test_get_gaussian_func_fit(self):
        """Test intensity fitting success."""

        for hemi in [-1, 1]:
            with self.subTest(hemi=hemi):
                # Get the Gaussian fits
                self.out = fitting.get_gaussian_func_fit(
                    hemi * self.mlat_bins, self.mlt_bins, self.mean_intensity,
                    self.std_intensity, self.num_intensity,
                    num_gauss=self.num_gauss, min_num=self.min_num,
                    min_intensity=self.min_intensity,
                    min_lat_perc=self.min_lat_perc)

                # Evaluate the outputs
                self.eval_gaussian_fit_out(hemi)
                self.assertLess(self.num_bad, self.mlt_bins.shape[0])
        return

    def test_get_gaussian_func_fit_high_intensity_thresh(self):
        """Test intensity fitting failure with a high intensity threshold."""
        self.min_intensity = self.mean_intensity.max() + 1.0

        for hemi in [-1, 1]:
            with self.subTest(hemi=hemi):
                # Get the Gaussian fits
                self.out = fitting.get_gaussian_func_fit(
                    hemi * self.mlat_bins, self.mlt_bins, self.mean_intensity,
                    self.std_intensity, self.num_intensity,
                    num_gauss=self.num_gauss, min_num=self.min_num,
                    min_intensity=self.min_intensity,
                    min_lat_perc=self.min_lat_perc)

                # Evaluate the outputs
                self.eval_gaussian_fit_out(hemi)
                self.assertEqual(self.num_bad, self.mlt_bins.shape[0])
        return

    def test_get_gaussian_func_fit_high_count_thresh(self):
        """Test intensity fitting failure with a high count threshold."""
        self.min_num = self.num_intensity.max() + 1

        for hemi in [-1, 1]:
            with self.subTest(hemi=hemi):
                # Get the Gaussian fits
                self.out = fitting.get_gaussian_func_fit(
                    hemi * self.mlat_bins, self.mlt_bins, self.mean_intensity,
                    self.std_intensity, self.num_intensity,
                    num_gauss=self.num_gauss, min_num=self.min_num,
                    min_intensity=self.min_intensity,
                    min_lat_perc=self.min_lat_perc)

                # Evaluate the outputs
                self.eval_gaussian_fit_out(hemi)
                self.assertEqual(self.num_bad, self.mlt_bins.shape[0])
        return

    def test_get_gaussian_func_fit_high_lat_perc(self):
        """Test intensity fitting failure with a high lat threshold."""
        self.min_lat_perc = 100.0

        for hemi in [-1, 1]:
            with self.subTest(hemi=hemi):
                # Get the Gaussian fits
                self.out = fitting.get_gaussian_func_fit(
                    hemi * self.mlat_bins, self.mlt_bins, self.mean_intensity,
                    self.std_intensity, self.num_intensity,
                    num_gauss=self.num_gauss, min_num=self.min_num,
                    min_intensity=self.min_intensity,
                    min_lat_perc=self.min_lat_perc)

                # Evaluate the outputs
                self.eval_gaussian_fit_out(hemi)
                self.assertEqual(self.num_bad, self.mlt_bins.shape[0])
        return

    def test_gauss_quad_err(self):
        """Test the `gauss_quad_err` function returns correct values."""
        self.out = fitting.gauss_quad_err(self.coeffs, self.mlat_bins,
                                          self.mean_intensity[:, 0], 1.0)
        self.assertTupleEqual(self.out.shape, self.mlat_bins.shape)
        self.assertTrue(np.all(self.out == 0.0))
        return

    def test_estimate_peak_widths(self):
        """Test `estimate_peak_widths` function."""
        # Get the peak location indices in `min_num` and magnitudes in
        # `min_intensity`
        self.get_peak_inds()

        # Get the FWHM for these locations
        self.out = fitting.estimate_peak_widths(self.mean_intensity[:, 0],
                                                self.mlat_bins, self.min_num,
                                                self.min_intensity)

        # Compare the estimated FWHM to the coefficient sigmas
        for i, fwhm in enumerate(self.out):
            self.assertLessEqual(abs(fwhm - self.coeffs[3 * i + 5]), 0.5)
        return
