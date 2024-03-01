#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `proc.boundaries`."""

import numpy as np
import unittest

from pyIntensityFeatures.proc import boundaries


class TestBoundariesFuncs(unittest.TestCase):
    """Tests for the boundary ID functions."""

    def setUp(self):
        """Set up the test runs."""
        self.fit_coeff = [1.0, 0.01, 0.001, 500.0, 75.0, 3.0, 400, 79.0, 1.0,
                          300, 65.0, 0.5]
        self.fit_cov = np.full(shape=(
            len(self.fit_coeff), len(self.fit_coeff)), fill_value=0.1)
        self.rvalue = 0.95
        self.pvalue = 1.0e-5
        self.num_peaks = 3
        self.mlat_min = 60.0
        self.mlat_max = 90.0
        self.strict_fit = False
        self.bounds = None
        self.good_bounds = None
        self.check_bounds = {1: [67.93553986490716, 82.06446013509284],
                             2: [67.93553986490716, 82.06446013509284],
                             3: [63.822589977484526, 82.06446013509284]}
        self.check_uncert = 1.0608872482286447
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.fit_coeff, self.fit_cov, self.rvalue, self.pvalue
        del self.num_peaks, self.mlat_min, self.mlat_max, self.bounds
        del self.good_bounds, self.strict_fit
        return

    def get_coeffs_by_peak_and_hemisphere(self, npeaks, hemisphere):
        """Obtain temporary copies of the fit coefficients and covariance.

        Parameters
        ----------
        npeaks : int
            Number of peaks to be in the covariance matrix (max of 3)
        hemisphere : int
            -1 for Southern, 1 for Northern

        Returns
        -------
        coeff : list
            List of adjusted coefficients
        covar : array
            Covariance matrix adjusted to the correct size.

        """
        # Update the fit values
        coeffs = self.fit_coeff[:len(self.fit_coeff) - 3 * (self.num_peaks
                                                            - npeaks)]
        covar = self.fit_cov[:len(coeffs), :len(coeffs)]

        # Update the hemipshere-dependant coefficients
        for i in [4, 7, 10]:
            if i >= len(coeffs):
                break
            coeffs[i] *= hemisphere

        return coeffs, covar

    def eval_boundaries(self, npeaks, hemisphere):
        """Evaluate the output boundaries.

        Parameters
        ----------
        npeaks : int
            Number of peaks to be in the covariance matrix (max of 3)
        hemisphere : int
            -1 for Southern, 1 for Northern

        """
        self.assertAlmostEqual(self.bounds[-1], self.check_uncert)
        self.assertAlmostEqual(self.bounds[-2], self.check_uncert)
        self.assertAlmostEqual(hemisphere * self.bounds[0],
                               self.check_bounds[npeaks][0])
        self.assertAlmostEqual(hemisphere * self.bounds[1],
                               self.check_bounds[npeaks][1])
        return

    def test_get_eval_boundares_single(self):
        """Test outcome specifying a single fit."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find and eval function
                    (self.bounds,
                     self.good_bounds) = boundaries.get_eval_boundaries(
                         coeffs, covar, self.rvalue, self.pvalue, npeak,
                         hemi * self.mlat_min, hemi * self.mlat_max, "single",
                         strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    if npeak == 1:
                        self.assertTrue(self.good_bounds)
                        self.eval_boundaries(npeak, hemi)
                    else:
                        self.assertFalse(self.good_bounds)
                        self.assertTrue(np.isnan(self.bounds).all())
        return

    def test_get_eval_boundares_mult(self):
        """Test outcome specifying a multi-peaked fit."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find and eval function
                    (self.bounds,
                     self.good_bounds) = boundaries.get_eval_boundaries(
                         coeffs, covar, self.rvalue, self.pvalue, npeak,
                         hemi * self.mlat_min, hemi * self.mlat_max, "mult",
                         strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    if npeak > 1:
                        self.assertTrue(self.good_bounds)
                        self.eval_boundaries(npeak, hemi)
                    else:
                        self.assertFalse(self.good_bounds)
                        self.assertTrue(np.isnan(self.bounds).all())
        return

    def test_get_eval_boundares_best(self):
        """Test outcome specifying the best fitting method."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find and eval function
                    (self.bounds,
                     self.good_bounds) = boundaries.get_eval_boundaries(
                         coeffs, covar, self.rvalue, self.pvalue, npeak,
                         hemi * self.mlat_min, hemi * self.mlat_max, "best",
                         strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    self.assertTrue(self.good_bounds)
                    self.eval_boundaries(npeak, hemi)
        return

    def test_get_eval_boundares_bad_correlation(self):
        """Test outcome with a bad covariance."""
        self.pvalue = 1.0

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find and eval function
                    (self.bounds,
                     self.good_bounds) = boundaries.get_eval_boundaries(
                         coeffs, covar, self.rvalue, self.pvalue, npeak,
                         hemi * self.mlat_min, hemi * self.mlat_max, "best",
                         strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    self.assertFalse(self.good_bounds)
                    self.eval_boundaries(npeak, hemi)
        return

    def test_get_eval_boundares_bad_background(self):
        """Test outcome with a bad background."""
        self.fit_coeff[2] = 10.0

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find and eval function
                    (self.bounds,
                     self.good_bounds) = boundaries.get_eval_boundaries(
                         coeffs, covar, self.rvalue, self.pvalue, npeak,
                         hemi * self.mlat_min, hemi * self.mlat_max, "best",
                         strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    self.assertFalse(self.good_bounds)
                    self.assertTrue(np.isnan(self.bounds).all())
        return

    def test_get_eval_boundares_strict_fit(self):
        """Test outcome with a negative sigma."""
        for i in [5, 8, 11]:
            self.fit_coeff[i] *= -1.0

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                # Cycle through the strict/permissive fit flags
                for self.strict_fit in [True, False]:
                    with self.subTest(hemi=hemi, npeaks=npeak,
                                      strict_fit=self.strict_fit):
                        # Run the find and eval function
                        (self.bounds,
                         self.good_bounds) = boundaries.get_eval_boundaries(
                             coeffs, covar, self.rvalue, self.pvalue, npeak,
                             hemi * self.mlat_min, hemi * self.mlat_max, "best",
                             strict_fit=self.strict_fit)

                        # Evaluate the outcome
                        self.assertFalse(self.good_bounds)

                        if self.strict_fit:
                            self.assertTrue(np.isnan(self.bounds).all())
                        else:
                            self.eval_boundaries(npeak, hemi)
        return

    def test_locate_mult_peak_boundaries(self):
        """Test boundary ID for a multi-peaked fit."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(2, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find and eval function
                    self.bounds = boundaries.locate_mult_peak_boundaries(
                        coeffs, covar, npeak, hemi * self.mlat_min,
                        hemi * self.mlat_max, strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    self.eval_boundaries(npeak, hemi)
        return

    def test_locate_mult_peak_boundaries_all_bad(self):
        """Test boundary ID for a multi-peaked fit."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(2, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                # Set the first amplitude to be negative to ensure the first
                # peak isn't appropriate
                coeffs[3] *= -1.0

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find and eval function with bad max/min limits
                    # to ensure any secondary peaks are also rejected
                    self.bounds = boundaries.locate_mult_peak_boundaries(
                        coeffs, covar, npeak, hemi * self.mlat_max,
                        hemi * self.mlat_min, strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    self.assertTrue(np.isnan(self.bounds).all())
        return

    def test_locate_mult_peak_boundaries_max_spread(self):
        """Test boundary ID rejection for separated peaks."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Update the fit values
            coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(
                self.num_peaks, hemi)

            with self.subTest(hemi=hemi):
                # Run the find and eval function
                self.bounds = boundaries.locate_mult_peak_boundaries(
                    coeffs, covar, self.num_peaks, hemi * self.mlat_min,
                    hemi * self.mlat_max, max_peak_diff=0.1,
                    strict_fit=self.strict_fit)

                # Evaluate the outcome, which is the same as the double-peak, as
                # the third peak should be rejected
                self.eval_boundaries(2, hemi)
        return

    def test_locate_mult_peak_boundaries_strict_fit(self):
        """Test multiple boundary ID with a negative sigma."""
        for i in [5, 8, 11]:
            self.fit_coeff[i] *= -1.0

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(2, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                # Cycle through the strict/permissive fit flags
                for self.strict_fit in [True, False]:
                    with self.subTest(hemi=hemi, npeaks=npeak,
                                      strict_fit=self.strict_fit):
                        # Run the find and eval function
                        self.bounds = boundaries.locate_mult_peak_boundaries(
                            coeffs, covar, npeak, hemi * self.mlat_min,
                            hemi * self.mlat_max, strict_fit=self.strict_fit)

                        # Evaluate the outcome
                        if self.strict_fit:
                            self.assertTrue(
                                np.isnan(self.bounds[:2]).all(),
                                msg="Finite boundaries found: {:}".format(
                                    self.bounds[:2]))
                        else:
                            self.eval_boundaries(npeak, hemi)
        return

    def test_locate_single_peak_boundaries(self):
        """Test boundary ID for a single-peaked fit."""
        # Set the number of peaks
        self.num_peaks = 1

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Update the fit values
            coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(
                self.num_peaks, hemi)

            with self.subTest(hemi=hemi):
                # Run the find and eval function
                self.bounds = boundaries.locate_single_peak_boundaries(
                    coeffs[4], coeffs[5], covar, hemi * self.mlat_min,
                    hemi * self.mlat_max, strict_fit=self.strict_fit)

                # Evaluate the outcome
                self.eval_boundaries(self.num_peaks, hemi)
        return

    def test_locate_single_peak_boundaries_outside_fit_region(self):
        """Test boundary ID for a single-peaked fit that fail region test."""
        # Set the number of peaks
        self.num_peaks = 1

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Update the fit values
            coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(
                self.num_peaks, hemi)

            with self.subTest(hemi=hemi):
                # Run the find and eval function
                self.bounds = boundaries.locate_single_peak_boundaries(
                    coeffs[4], coeffs[5], covar, hemi * 70.0, hemi * 80.0,
                    strict_fit=self.strict_fit)

                # Evaluate the outcome
                self.assertTrue(np.isnan(self.bounds[0]))
                self.assertTrue(np.isnan(self.bounds[1]))
        return

    def test_locate_single_peak_boundaries_strict_fit(self):
        """Test single-peak boundary ID with a negative sigma."""
        # Update the sigma values for this test
        for i in [5, 8, 11]:
            self.fit_coeff[i] *= -1.0

        # Set the number of peaks
        self.num_peaks = 1

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Update the fit values
            coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(
                self.num_peaks, hemi)

            # Cycle through the strict/permissive fit flags
            for self.strict_fit in [True, False]:
                with self.subTest(hemi=hemi, strict_fit=self.strict_fit):
                    # Run the find and eval function
                    self.bounds = boundaries.locate_single_peak_boundaries(
                        coeffs[4], coeffs[5], covar, hemi * self.mlat_min,
                        hemi * self.mlat_max, strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    if self.strict_fit:
                        self.assertTrue(
                            np.isnan(self.bounds).all(),
                            msg="Finite boundaries found: {:}".format(
                                self.bounds))
                    else:
                        self.eval_boundaries(self.num_peaks, hemi)
        return

    def test_calc_boundary_uncertainty(self):
        """Test the outcome of the uncertainty calculation."""
        self.bounds = boundaries.calc_boundary_uncertainty(self.num_peaks,
                                                           self.fit_cov)

        self.assertAlmostEqual(self.bounds, self.check_uncert)
        return

    def test_locate_boundares_single(self):
        """Test general boundary ID specifying a single fit."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find function
                    self.bounds = boundaries.locate_boundaries(
                        coeffs, covar, npeak, hemi * self.mlat_min,
                        hemi * self.mlat_max, method="single",
                        strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    if npeak == 1:
                        self.eval_boundaries(npeak, hemi)
                    else:
                        self.assertTrue(np.isnan(self.bounds).all())
        return

    def test_locate_boundaries_using_mult_peak(self):
        """Test general boundary ID specifying a multi-peaked fit."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find function
                    self.bounds = boundaries.locate_boundaries(
                        coeffs, covar, npeak, hemi * self.mlat_min,
                        hemi * self.mlat_max, method="mult",
                        strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    if npeak > 1:
                        self.eval_boundaries(npeak, hemi)
                    else:
                        self.assertTrue(np.isnan(self.bounds).all())
        return

    def test_locate_boundaries_mult_peak_outside_fit_region(self):
        """Test general multi-peaked boundary ID with bounds outside region."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find function
                    self.bounds = boundaries.locate_boundaries(
                        coeffs, covar, npeak, hemi * 70.0, hemi * 75.0,
                        method="mult", strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    self.assertTrue(np.isnan(self.bounds[:2]).all(),
                                    msg="Some finite: {:}".format(
                                        self.bounds[:2]))
        return

    def test_locate_boundaries_with_best(self):
        """Test general boundary ID specifying the best fitting method."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                with self.subTest(hemi=hemi, npeaks=npeak):
                    # Run the find function
                    self.bounds = boundaries.locate_boundaries(
                        coeffs, covar, npeak, hemi * self.mlat_min,
                        hemi * self.mlat_max, method="best",
                        strict_fit=self.strict_fit)

                    # Evaluate the outcome
                    self.eval_boundaries(npeak, hemi)
        return

    def test_locate_boundaries_max_spread(self):
        """Test general boundary ID rejection for separated peaks."""
        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Update the fit values
            coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(
                self.num_peaks, hemi)

            with self.subTest(hemi=hemi):
                # Run the find and eval function
                self.bounds = boundaries.locate_boundaries(
                    coeffs, covar, self.num_peaks, hemi * self.mlat_min,
                    hemi * self.mlat_max, max_peak_diff=0.1,
                    strict_fit=self.strict_fit)

                # Evaluate the outcome, which is the same as the double-peak, as
                # the third peak should be rejected
                self.eval_boundaries(2, hemi)
        return

    def test_locate_boundares_strict_fit(self):
        """Test general boundary ID outcome with a negative sigma."""
        for i in [5, 8, 11]:
            self.fit_coeff[i] *= -1.0

        # Cycle through both hemisphere
        for hemi in [-1, 1]:
            # Set the number of peaks
            for npeak in np.arange(1, self.num_peaks + 1):
                # Update the fit values
                coeffs, covar = self.get_coeffs_by_peak_and_hemisphere(npeak,
                                                                       hemi)

                # Cycle through the strict/permissive fit flags
                for self.strict_fit in [True, False]:
                    with self.subTest(hemi=hemi, npeaks=npeak,
                                      strict_fit=self.strict_fit):
                        # Run the find function
                        self.bounds = boundaries.locate_boundaries(
                            coeffs, covar, npeak, hemi * self.mlat_min,
                            hemi * self.mlat_max, method="best",
                            strict_fit=self.strict_fit)

                        # Evaluate the outcome
                        if self.strict_fit:
                            self.assertTrue(
                                np.isnan(self.bounds[:2]).all(),
                                msg="Finite boundaries found: {:}".format(
                                    self.bounds[:2]))
                        else:
                            self.eval_boundaries(npeak, hemi)
        return

    def test_locate_boundares_bad_method(self):
        """Test general boundary ID raises ValueError with a bad method."""

        args = [self.fit_coeff, self.fit_cov, self.num_peaks, self.mlat_min,
                self.mlat_max, "method"]
        self.assertRaisesRegex(ValueError, "unexpected method, try: 'single'",
                               boundaries.locate_boundaries, *args)
        return
