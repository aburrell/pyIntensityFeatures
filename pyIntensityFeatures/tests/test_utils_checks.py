#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.checks`."""

import datetime as dt
from io import StringIO
import logging
import numpy as np
import unittest
import xarray as xr

from pyIntensityFeatures import logger
from pyIntensityFeatures.utils import checks


class TestPearsonFuncs(unittest.TestCase):
    """Tests for Pearson correlation value check functions."""

    def setUp(self):
        """Set up the test runs."""
        self.rthresh = 0.8
        self.pmax = 1.0e-5  # Should be a value lower than the default
        self.rvals = []
        self.pvals = []
        self.out = None

        self.lmsg = u''
        self.log_capture = StringIO()
        logger.addHandler(logging.StreamHandler(self.log_capture))
        logger.setLevel(logging.INFO)
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.rthresh, self.pmax, self.rvals, self.pvals, self.out, self.lmsg
        del self.log_capture
        return

    def test_evaluate_pearson_finite_success(self):
        """Test success for finite, numeric input values."""
        # Set the R values to fall at or above the R threshold and the p values
        # at or beow the p maximum
        self.rvals = np.linspace(self.rthresh, 1, num=3)
        self.pvals = np.linspace(0.0, self.pmax, num=3)

        for i, rval in enumerate(self.rvals):
            args = [rval, self.pvals[i]]
            with self.subTest(val=args):
                # Test the outcome is True
                self.out = checks.evaluate_pearson(*args, rthresh=self.rthresh,
                                                   pmax=self.pmax)
                self.assertTrue(self.out)
        return

    def test_evaluate_pearson_finite_failure(self):
        """Test failure for finite, numeric input values."""
        # Set the R values to fall at or above the R threshold and the p values
        # at or beow the p maximum
        self.rvals = np.linspace(0, self.rthresh - 1.0e-4, num=3)
        self.pvals = np.linspace(self.pmax + 1.0e-4, 1, num=3)

        for i, rval in enumerate(self.rvals):
            args = [rval, self.pvals[i]]
            with self.subTest(val=args):
                # Test the outcome is False
                self.out = checks.evaluate_pearson(*args, rthresh=self.rthresh,
                                                   pmax=self.pmax)
                self.assertFalse(self.out)
        return

    def test_evaluate_pearson_nonetype(self):
        """Test failure and logger message for NoneType inputs."""
        # Set the mixed and matched float/NoneType inputs
        self.rvals = [None, 1.0, None]
        self.pvals = [0.0, None, None]

        # Set the expected logging output
        self.lmsg = u'Did not supply all Pearson values for evaluation.'

        for i, rval in enumerate(self.rvals):
            args = [rval, self.pvals[i]]
            with self.subTest(val=args):
                # Test the outcome is False
                self.assertFalse(checks.evaluate_pearson(*args,
                                                         rthresh=self.rthresh,
                                                         pmax=self.pmax))

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging message
                self.assertTrue(self.out.find(self.lmsg) >= 0)
        return

    def test_evaluate_pearson_bad_float(self):
        """Test failure and logger message for infinite and NaN inputs."""
        # Set the mixed and matched float/NoneType inputs
        self.rvals = [np.nan, np.inf, -np.inf, 1.0, 1.0, 1.0, np.nan]
        self.pvals = [0.0, 0.0, 0.0, np.nan, np.inf, -np.inf, np.nan]

        # Set the expected logging output
        self.lmsg = u'Not all Pearson values are finite.'

        for i, rval in enumerate(self.rvals):
            args = [rval, self.pvals[i]]
            with self.subTest(val=args):
                # Test the outcome is False
                self.assertFalse(checks.evaluate_pearson(*args,
                                                         rthresh=self.rthresh,
                                                         pmax=self.pmax))

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging message
                self.assertTrue(self.out.find(self.lmsg) >= 0)
        return

    def test_compare_pearson_best_rval(self):
        """Test successful identification of the best Pearson values."""
        # Set the input lists
        self.rvals = [-1.0, 0.0, 0.5]
        self.pvals = [self.pmax, self.pmax, self.pmax]

        for igood in range(len(self.rvals)):
            # Set the input arguements, replacing the desired output index
            # with an R value of 1.0 (best possible value)
            args = [list(self.rvals), list(self.pvals)]
            args[0][igood] = 1.0

            with self.subTest(val=args):
                # Test the selected index has the desired value
                self.out = checks.compare_pearson(*args)
                self.assertEqual(igood, self.out)
        return

    def test_compare_pearson_valid_pmax(self):
        """Test successful identification of the best Pearson values."""
        # Set the input lists
        self.rvals = np.ones(shape=(3,))
        self.pvals = np.ones(shape=(3,))

        for igood in range(len(self.rvals)):
            # Set the input arguements, replacing the desired output index
            # with a p value below the valid threshold
            args = [list(self.rvals), list(self.pvals)]
            args[1][igood] = self.pmax

            with self.subTest(val=args):
                # Test the selected index has the desired value
                self.out = checks.compare_pearson(*args)
                self.assertEqual(igood, self.out)
        return

    def test_compare_pearson_best_finite(self):
        """Test successful ID of the best Pearson values with bad inputs."""
        # Set the input lists
        self.rvals = [np.nan, np.inf, None, 0.5, -1.0]
        self.pvals = [None, -np.inf, np.nan, self.pmax, np.inf]

        for igood in range(len(self.rvals)):
            # Set the input arguements, replacing the desired output index
            # with valid R and p values
            args = [list(self.rvals), list(self.pvals)]
            args[0][igood] = 1.0
            args[1][igood] = self.pmax

            with self.subTest(val=args):
                # Test the selected index has the desired value
                self.out = checks.compare_pearson(*args)
                self.assertEqual(igood, self.out)
        return

    def test_compare_pearson_invalid(self):
        """Test rejection of comparison for all bad Pearson values."""
        # Set the input lists
        self.rvals = [np.nan, np.inf, None]
        self.pvals = [None, -np.inf, np.nan]

        # Test the selected index has the desired value
        self.assertIsNone(checks.compare_pearson(self.rvals, self.pvals))
        return


class TestEvalGaussQuadFuncs(unittest.TestCase):
    """Tests for the `evaluate_gauss_quad` check function."""

    def setUp(self):
        """Set up the test runs."""
        self.npeaks = 1
        self.fit_coeff = [0.1, 0.01, 0.01, 500.0, 75.0, 5.0, 100.0, 77.0, 5.0,
                          90.0, 72.0, 8.0]
        self.rvalue = 1.0
        self.pvalue = 1.0e-5
        self.po_bound = 80.0
        self.un_bound_po = 0.5
        self.eq_bound = 70.0
        self.un_bound_eq = 0.25
        self.min_mlat = 60.0
        self.max_mlat = 90.0
        self.un_threshold = 1.5
        self.out = None

        self.lmsg = u''
        self.log_capture = StringIO()
        logger.addHandler(logging.StreamHandler(self.log_capture))
        logger.setLevel(logging.INFO)
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.npeaks, self.fit_coeff, self.rvalue, self.pvalue, self.po_bound
        del self.un_bound_po, self.eq_bound, self.un_bound_eq, self.min_mlat
        del self.max_mlat, self.un_threshold, self.out, self.lmsg
        del self.log_capture
        return

    def test_npeak_success(self):
        """Test success with different npeak values."""
        # Cycle through different number of Gaussian peak fits
        for self.npeaks in np.arange(1, 4, 1):
            # Update the fit coefficients to have the right shape
            coeffs = self.fit_coeff[:3 + 3 * self.npeaks]

            with self.subTest(npeaks=self.npeaks, coeffs=coeffs):
                # Run the test
                self.out = checks.evaluate_gauss_quad(
                    self.npeaks, coeffs, self.rvalue, self.pvalue,
                    self.po_bound, self.un_bound_po, self.eq_bound,
                    self.un_bound_eq, self.min_mlat, self.max_mlat,
                    un_threshold=self.un_threshold)

                # Get the logging message to provide an informative failure
                self.lmsg = self.log_capture.getvalue()

                # Test the output
                self.assertTrue(self.out, msg=self.lmsg)
        return

    def test_finite_coeff_failure(self):
        """Test failure with non-finite coefficient values."""
        # Set the expected logging output
        self.lmsg = u'Fit coefficients are not all finite.'

        for bad_val in [np.nan, -np.inf, np.inf]:
            coeffs = self.fit_coeff[:3 + 3 * self.npeaks]
            coeffs[0] = bad_val
            with self.subTest(coeffs=coeffs):
                # Run the test and ensure it fails
                self.assertFalse(
                    checks.evaluate_gauss_quad(
                        self.npeaks, coeffs, self.rvalue, self.pvalue,
                        self.po_bound, self.un_bound_po, self.eq_bound,
                        self.un_bound_eq, self.min_mlat, self.max_mlat,
                        un_threshold=self.un_threshold),
                    msg="Passed with bad coefficient value")

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging output message
                self.assertTrue(self.out.find(self.lmsg) >= 0,
                                msg="Unexpected logging message: {:}".format(
                                    self.out))
        return

    def test_finite_coeff_neg_amp_or_spread(self):
        """Test failure with negative amplitude or spread values."""
        # Set the expected logging output
        self.lmsg = u'Gaussian amplitude and/or spread are negative'

        for bad_index in [-3, -1]:
            coeffs = self.fit_coeff[:3 + 3 * self.npeaks]
            coeffs[bad_index] *= -1
            with self.subTest(coeffs=coeffs):
                # Run the test and ensure it fails
                self.assertFalse(
                    checks.evaluate_gauss_quad(
                        self.npeaks, coeffs, self.rvalue, self.pvalue,
                        self.po_bound, self.un_bound_po, self.eq_bound,
                        self.un_bound_eq, self.min_mlat, self.max_mlat,
                        un_threshold=self.un_threshold),
                    msg="Passed with negative coefficient value")

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging output message
                self.assertTrue(self.out.find(self.lmsg) >= 0,
                                msg="Unexpected logging message: {:}".format(
                                    self.out))
        return

    def test_finite_coeff_bad_hemisphere(self):
        """Test failure with x-offset in the wrong hemisphere."""
        # Set the expected logging output
        self.lmsg = u'Gaussian peak location in the wrong hemisphere for peak'

        # Update the coefficients to have the right number and be in the
        # opposite hemisphere
        coeffs = self.fit_coeff[:3 + 3 * self.npeaks]
        coeffs[-2] *= -1

        # Run the test and ensure it fails
        self.assertFalse(
            checks.evaluate_gauss_quad(
                self.npeaks, coeffs, self.rvalue, self.pvalue, self.po_bound,
                self.un_bound_po, self.eq_bound, self.un_bound_eq,
                self.min_mlat, self.max_mlat, un_threshold=self.un_threshold),
            msg="Passed with a bad peak hemisphere")

        # Get the logging message
        self.out = self.log_capture.getvalue()

        # Test the logging output message
        self.assertTrue(self.out.find(self.lmsg) >= 0,
                        msg="Unexpected logging message: {:}".format(self.out))
        return

    def test_finite_coeff_bad_location(self):
        """Test failure with x-offset outside the slice latitude range."""
        # Set the expected logging output
        self.lmsg = u'Gaussian peak is outside of the intensity profile for'

        for bad_lat in [self.min_mlat, self.max_mlat]:
            # Update the coefficients to have the right number and have a peak
            # outside the desired latitude range
            coeffs = self.fit_coeff[:3 + 3 * self.npeaks]
            coeffs[-2] = bad_lat

            with self.subTest(coeffs=coeffs):
                # Run the test and ensure it fails
                self.assertFalse(
                    checks.evaluate_gauss_quad(
                        self.npeaks, coeffs, self.rvalue, self.pvalue,
                        self.po_bound, self.un_bound_po, self.eq_bound,
                        self.un_bound_eq, self.min_mlat, self.max_mlat,
                        un_threshold=self.un_threshold),
                    msg="Passed with a bad peak location")

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging output message
                self.assertTrue(self.out.find(self.lmsg) >= 0,
                                msg="Unexpected logging message: {:}".format(
                                    self.out))
        return

    def test_finite_coeff_low_amplitude(self):
        """Test failure with an amplitude too close to the background."""
        # Set the expected logging output
        self.lmsg = u'Gaussian amplitude is less than 10% of the background'

        # Update the coefficients to have the right number and have a low
        # Gaussian peak amplitude
        coeffs = self.fit_coeff[:3 + 3 * self.npeaks]
        coeffs[-3] = 1.0

        # Run the test and ensure it fails
        self.assertFalse(
            checks.evaluate_gauss_quad(
                self.npeaks, coeffs, self.rvalue, self.pvalue, self.po_bound,
                self.un_bound_po, self.eq_bound, self.un_bound_eq,
                self.min_mlat, self.max_mlat, un_threshold=self.un_threshold),
            msg="Passed with a low amplitude value")

        # Get the logging message
        self.out = self.log_capture.getvalue()

        # Test the logging output message
        self.assertTrue(self.out.find(self.lmsg) >= 0,
                        msg="Unexpected logging message: {:}".format(self.out))
        return

    def test_finite_coeff_wide_spread(self):
        """Test failure with peak spread larger than the data range."""
        # Set the expected logging output
        self.lmsg = u'Gaussian peak width is larger than the data range for'

        # Update the coefficients to have the right number and have a wide
        # Gaussian peak width
        coeffs = self.fit_coeff[:3 + 3 * self.npeaks]
        coeffs[-1] = 50.0

        # Run the test and ensure it fails
        self.assertFalse(
            checks.evaluate_gauss_quad(
                self.npeaks, coeffs, self.rvalue, self.pvalue, self.po_bound,
                self.un_bound_po, self.eq_bound, self.un_bound_eq,
                self.min_mlat, self.max_mlat, un_threshold=self.un_threshold),
            msg="Passed with a wide sigma value")

        # Get the logging message
        self.out = self.log_capture.getvalue()

        # Test the logging output message
        self.assertTrue(self.out.find(self.lmsg) >= 0,
                        msg="Unexpected logging message: {:}".format(self.out))
        return

    def test_finite_coeff_narrow_width(self):
        """Test failure with a narrow peak width."""
        # Set the expected logging output
        self.lmsg = u'All Gaussian peaks are too narrow.'

        # Update the coefficients to have the right number and have a Gaussian
        # width too narrow to characterize the aurora
        coeffs = self.fit_coeff[:3 + 3 * self.npeaks]
        coeffs[-1] = 0.1

        # Run the test and ensure it fails
        self.assertFalse(
            checks.evaluate_gauss_quad(
                self.npeaks, coeffs, self.rvalue, self.pvalue, self.po_bound,
                self.un_bound_po, self.eq_bound, self.un_bound_eq,
                self.min_mlat, self.max_mlat, un_threshold=self.un_threshold),
            msg="Passed with a narrow sigma value")

        # Get the logging message
        self.out = self.log_capture.getvalue()

        # Test the logging output message
        self.assertTrue(self.out.find(self.lmsg) >= 0,
                        msg="Unexpected logging message: {:}".format(self.out))
        return

    def test_bad_polar_uncertainty(self):
        """Test failure with an unrealistic polar boundary uncertainty."""
        # Set the expected logging output
        self.lmsg = u'The polar boundary uncertainty is unrealistic'

        # Update the boundary uncertainty
        self.un_bound_po = self.un_threshold + 1.0

        # Run the test and ensure it fails
        self.assertFalse(
            checks.evaluate_gauss_quad(
                self.npeaks, self.fit_coeff, self.rvalue, self.pvalue,
                self.po_bound, self.un_bound_po, self.eq_bound,
                self.un_bound_eq, self.min_mlat, self.max_mlat,
                un_threshold=self.un_threshold),
            msg="Passed with a bad boundary uncertainty")

        # Get the logging message
        self.out = self.log_capture.getvalue()

        # Test the logging output message
        self.assertTrue(self.out.find(self.lmsg) >= 0,
                        msg="Unexpected logging message: {:}".format(self.out))
        return

    def test_bad_equator_uncertainty(self):
        """Test failure with an unrealistic equatorward boundary uncertainty."""
        # Set the expected logging output
        self.lmsg = u'The equatorward boundary uncertainty is unrealistic'

        # Update the boundary uncertainty
        self.un_bound_eq = self.un_threshold + 1.0

        # Run the test and ensure it fails
        self.assertFalse(
            checks.evaluate_gauss_quad(
                self.npeaks, self.fit_coeff, self.rvalue, self.pvalue,
                self.po_bound, self.un_bound_po, self.eq_bound,
                self.un_bound_eq, self.min_mlat, self.max_mlat,
                un_threshold=self.un_threshold),
            msg="Passed with a bad boundary uncertainty")

        # Get the logging message
        self.out = self.log_capture.getvalue()

        # Test the logging output message
        self.assertTrue(self.out.find(self.lmsg) >= 0,
                        msg="Unexpected logging message: {:}".format(self.out))
        return

    def test_bad_boundary_locations(self):
        """Test failure with an unrealistic boundary locations."""
        # Set the expected logging output
        self.lmsg = u'The polar/equatorward boundary locations are mixed up.'

        # Cycle through bad ALB pairs
        for ebnd, pbnd in [[self.po_bound, self.eq_bound],
                           [self.eq_bound, self.max_mlat],
                           [self.min_mlat, self.po_bound]]:
            with self.subTest(ebnd=ebnd, pbnd=pbnd):
                # Run the test and ensure it fails
                self.assertFalse(
                    checks.evaluate_gauss_quad(
                        self.npeaks, self.fit_coeff, self.rvalue, self.pvalue,
                        pbnd, self.un_bound_po, ebnd, self.un_bound_eq,
                        self.min_mlat, self.max_mlat,
                        un_threshold=self.un_threshold),
                    msg="Passed with a bad boundary locations")

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging output message
                self.assertTrue(self.out.find(self.lmsg) >= 0,
                                msg="Unexpected logging message: {:}".format(
                                    self.out))
        return

    def test_nonfinite_pearson_values(self):
        """Test failure with non-finite Pearson correlation coefficients."""
        # Set the expected logging output
        self.lmsg = u'Not all Pearson values are finite.'

        # Cycle through bad Pearson correllation coefficient pairs
        for rval, pval in [[self.rvalue, np.nan], [np.inf, self.pvalue],
                           [np.nan, -np.inf]]:
            with self.subTest(rvalue=rval, pvalue=pval):
                # Run the test and ensure it fails
                self.assertFalse(
                    checks.evaluate_gauss_quad(
                        self.npeaks, self.fit_coeff, rval, pval, self.po_bound,
                        self.un_bound_po, self.eq_bound, self.un_bound_eq,
                        self.min_mlat, self.max_mlat,
                        un_threshold=self.un_threshold),
                    msg="Passed with a non-finite Pearson value.")

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging output message
                self.assertTrue(self.out.find(self.lmsg) >= 0,
                                msg="Unexpected logging message: {:}".format(
                                    self.out))
        return

    def test_nonetype_pearson_values(self):
        """Test failure with NoneType Pearson correlation coefficients."""
        # Set the expected logging output
        self.lmsg = u'Did not supply all Pearson values for evaluation.'

        # Cycle through bad Pearson correllation coefficient pairs
        for rval, pval in [[self.rvalue, None], [None, self.pvalue],
                           [None, None]]:
            with self.subTest(rvalue=rval, pvalue=pval):
                # Run the test and ensure it fails
                self.assertFalse(
                    checks.evaluate_gauss_quad(
                        self.npeaks, self.fit_coeff, rval, pval, self.po_bound,
                        self.un_bound_po, self.eq_bound, self.un_bound_eq,
                        self.min_mlat, self.max_mlat,
                        un_threshold=self.un_threshold),
                    msg="Passed with a missing Pearson value.")

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging output message
                self.assertTrue(self.out.find(self.lmsg) >= 0,
                                msg="Unexpected logging message: {:}".format(
                                    self.out))
        return

    def test_bad_pearson_values(self):
        """Test failure with a bad Pearson correlation coefficient."""
        # Cycle through bad Pearson correllation coefficient pairs
        for rval, pval in [[self.rvalue, 1.0], [-1.0, self.pvalue]]:
            with self.subTest(rvalue=rval, pvalue=pval):
                # Run the test and ensure it fails
                self.assertFalse(
                    checks.evaluate_gauss_quad(
                        self.npeaks, self.fit_coeff, rval, pval, self.po_bound,
                        self.un_bound_po, self.eq_bound, self.un_bound_eq,
                        self.min_mlat, self.max_mlat,
                        un_threshold=self.un_threshold),
                    msg="Passed with a bad Pearson value.")

                # Get the logging message
                self.out = self.log_capture.getvalue()

                # Test the logging output message
                self.assertTrue(len(self.out) == 0,
                                msg="Unexpected logging message: {:}".format(
                                    self.out))
        return


class TestEvalDayglow(unittest.TestCase):
    """Tests for `evaluate_dayglow`."""

    def setUp(self):
        """Set up the test runs."""
        self.thresh = 400.0
        self.coeffs = [0.1, 0.01, 0.01, 500.0, 75.0, 5.0, 100.0, 77.0, 5.0,
                       90.0, 72.0, 8.0]
        self.locs = np.arange(50.0, 90.0, 3.0)
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.thresh, self.coeffs, self.locs
        return

    def test_low_dayglow(self):
        """Test success with a low quadratic background."""
        # Test with float and array location input
        for i in [0, slice(None, None)]:
            with self.subTest(locs=self.locs[i]):
                self.assertTrue(
                    checks.evaluate_dayglow(
                        self.coeffs, self.locs[i], thresh=self.thresh),
                    msg="Unexpectedly high quadratic background")
        return

    def test_high_dayglow(self):
        """Test success with a high quadratic background."""
        # Test with float and array location input
        for i in [0, slice(None, None)]:
            with self.subTest(locs=self.locs[i]):
                self.coeffs[0] = self.thresh
                self.assertFalse(
                    checks.evaluate_dayglow(self.coeffs, self.locs[i],
                                            thresh=self.thresh),
                    msg="Unexpectedly low quadratic background.")
        return

    def test_bad_location_for_dayglow(self):
        """Test success with a non-finite location value."""
        self.locs[0] = np.nan

        # Test with float and array location input
        for i in [0, slice(None, None)]:
            with self.subTest(locs=self.locs[i]):
                self.assertFalse(
                    checks.evaluate_dayglow(self.coeffs, self.locs[i],
                                            thresh=self.thresh),
                    msg="Passed with a NaN location.")
        return


class TestCompareBoundaries(unittest.TestCase):
    """Tests for `compare_boundaries`."""

    def setUp(self):
        """Set up the test runs."""
        self.rvalue = [0.8, 0.8, 0.8, 0.8]
        self.pvalue = [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5]
        self.po_bound = [80.0, 81.0, 82.0, 83.0]
        self.un_bound_po = [0.5, 0.5, 0.5, 0.5]
        self.eq_bound = [70.0, 69.0, 68.0, 67.0]
        self.un_bound_eq = [0.5, 0.5, 0.5, 0.5]
        self.min_mlat = 60.0
        self.max_mlat = 90.0
        self.max_uncert = 5.0
        self.ieq = None
        self.ipo = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.rvalue, self.pvalue, self.po_bound, self.un_bound_po
        del self.eq_bound, self.un_bound_eq, self.min_mlat, self.max_mlat
        del self.max_uncert, self.ieq, self.ipo
        return

    def test_compare_good_bounds(self):
        """Compare boundaries that are good and pick the best."""

        for igood in range(len(self.rvalue)):
            rvals = list(self.rvalue)
            rvals[igood] = 1.0
            with self.subTest(rvals=rvals):
                # Run the comparison
                self.ieq, self.ipo = checks.compare_boundaries(
                    rvals, self.pvalue, self.eq_bound, self.un_bound_eq,
                    self.po_bound, self.un_bound_po, self.min_mlat,
                    self.max_mlat, max_uncert=self.max_uncert)

                # Evaluate the outcome
                self.assertEqual(igood, self.ieq,
                                 msg="Identified the wrong EALB index.")
                self.assertEqual(igood, self.ipo,
                                 msg="Identified the wrong PALB index.")
        return

    def test_compare_equatorward_high_uncertainty(self):
        """Compare boundaries where the best R value has high EALB error."""

        igood = 1
        self.rvalue[igood] = 1.0
        self.un_bound_eq[igood] = self.max_uncert + 1.0

        # Run the comparison
        self.ieq, self.ipo = checks.compare_boundaries(
            self.rvalue, self.pvalue, self.eq_bound, self.un_bound_eq,
            self.po_bound, self.un_bound_po, self.min_mlat, self.max_mlat,
            max_uncert=self.max_uncert)

        # Evaluate the outcome
        self.assertNotEqual(igood, self.ieq,
                            msg="Identified the wrong EALB index.")
        self.assertIn(self.ieq, np.arange(0, len(self.rvalue)),
                      msg="Unable to identify a valid EALB index.")
        self.assertEqual(igood, self.ipo,
                         msg="Identified the wrong PALB index.")
        return

    def test_compare_equatorward_bad_values(self):
        """Compare boundaries where the best R value has a bad EALB location."""

        igood = 1
        self.rvalue[igood] = 1.0
        for bad_val in [np.nan, np.inf, -np.inf, self.min_mlat - 1.0, None]:
            if bad_val is None:
                # Instead of have one bad value, have all but one bad value
                self.eq_bound = [np.nan for i in range(len(self.eq_bound))]
                self.eq_bound[igood + 1] = self.max_mlat - 1.0
            else:
                self.eq_bound[igood] = bad_val

            with self.subTest(eq_bounds=self.eq_bound):
                # Run the comparison
                self.ieq, self.ipo = checks.compare_boundaries(
                    self.rvalue, self.pvalue, self.eq_bound, self.un_bound_eq,
                    self.po_bound, self.un_bound_po, self.min_mlat,
                    self.max_mlat, max_uncert=self.max_uncert)

                # Evaluate the outcome
                self.assertNotEqual(igood, self.ieq,
                                    msg="Identified the wrong EALB index.")
                self.assertIn(self.ieq, np.arange(0, len(self.rvalue)),
                              msg="Unable to identify a valid EALB index.")
                self.assertEqual(igood, self.ipo,
                                 msg="Identified the wrong PALB index.")
        return

    def test_compare_equatorward_all_bad_values(self):
        """Compare boundaries where the best R value has bad EALB locations."""

        igood = 1
        self.rvalue[igood] = 1.0
        self.eq_bound = [np.nan, np.inf, self.max_mlat + 1.0,
                         self.min_mlat - 1.0]

        # Run the comparison
        self.ieq, self.ipo = checks.compare_boundaries(
            self.rvalue, self.pvalue, self.eq_bound, self.un_bound_eq,
            self.po_bound, self.un_bound_po, self.min_mlat, self.max_mlat,
            max_uncert=self.max_uncert)

        # Evaluate the outcome
        self.assertIsNone(self.ieq, msg="Identified the wrong EALB index.")
        self.assertEqual(igood, self.ipo,
                         msg="Identified the wrong PALB index.")
        return

    def test_compare_polar_high_uncertainty(self):
        """Compare boundaries where the best R value has high PALB error."""

        igood = 1
        self.rvalue[igood] = 1.0
        self.un_bound_po[igood] = self.max_uncert + 1.0

        # Run the comparison
        self.ieq, self.ipo = checks.compare_boundaries(
            self.rvalue, self.pvalue, self.eq_bound, self.un_bound_eq,
            self.po_bound, self.un_bound_po, self.min_mlat, self.max_mlat,
            max_uncert=self.max_uncert)

        # Evaluate the outcome
        self.assertNotEqual(igood, self.ipo,
                            msg="Identified the wrong PALB index.")
        self.assertIn(self.ipo, np.arange(0, len(self.rvalue)),
                      msg="Unable to identify a valid PALB index.")
        self.assertEqual(igood, self.ieq,
                         msg="Identified the wrong EALB index.")
        return

    def test_compare_polar_bad_values(self):
        """Compare boundaries where the best R value has a bad PALB location."""

        igood = 1
        self.rvalue[igood] = 1.0
        for bad_val in [np.nan, np.inf, -np.inf, self.max_mlat + 1.0, None]:
            if bad_val is None:
                # Instead of have one bad value, have all but one bad value
                self.po_bound = [np.nan for i in range(len(self.po_bound))]
                self.po_bound[igood + 1] = self.max_mlat - 1.0
            else:
                self.po_bound[igood] = bad_val

            with self.subTest(po_bounds=self.po_bound, bad_val=bad_val):
                # Run the comparison
                self.ieq, self.ipo = checks.compare_boundaries(
                    self.rvalue, self.pvalue, self.eq_bound, self.un_bound_eq,
                    self.po_bound, self.un_bound_po, self.min_mlat,
                    self.max_mlat, max_uncert=self.max_uncert)

                # Evaluate the outcome
                self.assertNotEqual(igood, self.ipo,
                                    msg="Identified the wrong PALB index.")
                self.assertIn(self.ipo, np.arange(0, len(self.rvalue)),
                              msg="Unable to identify a valid PALB index.")
                self.assertEqual(igood, self.ieq,
                                 msg="Identified the wrong EALB index.")
        return

    def test_compare_polar_all_bad_values(self):
        """Compare boundaries where the best R value has bad PALB locations."""

        igood = 1
        self.rvalue[igood] = 1.0
        self.po_bound = [np.nan, np.inf, self.max_mlat + 1.0,
                         self.min_mlat - 1.0]

        # Run the comparison
        self.ieq, self.ipo = checks.compare_boundaries(
            self.rvalue, self.pvalue, self.eq_bound, self.un_bound_eq,
            self.po_bound, self.un_bound_po, self.min_mlat, self.max_mlat,
            max_uncert=self.max_uncert)

        # Evaluate the outcome
        self.assertIsNone(self.ipo, msg="Identified the wrong PALB index.")
        self.assertEqual(igood, self.ieq,
                         msg="Identified the wrong EALB index.")
        return


class TestEvalBoundariesMLT(unittest.TestCase):
    """Tests for `evaluate_boundaries_in_mlt`."""

    def setUp(self):
        """Set up the test runs."""
        self.eq_key = "eq_bound"
        self.po_key = "po_bound"
        self.lt_key = "mlt"
        self.ut_key = "sweep_start"
        self.mlt = np.arange(0, 24, 1.0)
        self.bound_data = xr.Dataset(
            {self.po_key: (("sweep_start", "mlt"), [
                np.linspace(70, 90, self.mlt.shape[0])]),
             self.eq_key: (("sweep_start", "mlt"), [
                 np.linspace(60, 70, self.mlt.shape[0])])},
            coords={self.ut_key: [dt.datetime(1999, 2, 11)],
                    self.lt_key: self.mlt})
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.eq_key, self.po_key, self.lt_key, self.ut_key, self.mlt
        del self.bound_data
        return

    def test_bad_boundary_keys(self):
        """Test raises ValueError with bad boundary keys."""
        for attr in ["eq_key", "po_key"]:
            # Update the key
            orig_val = getattr(self, attr)
            setattr(self, attr, "not_a_key")

            with self.subTest(bad_key=attr):
                # Run and check the error
                self.assertRaisesRegex(ValueError, "bad boundary key",
                                       checks.evaluate_boundary_in_mlt,
                                       *[self.bound_data, self.eq_key,
                                         self.po_key, self.lt_key, self.ut_key])

                # Reset the key
                setattr(self, attr, orig_val)
        return

    def test_bad_coord_keys(self):
        """Test raises ValueError with bad coordinate keys."""
        for attr in ["lt_key", "ut_key"]:
            # Update the key
            orig_val = getattr(self, attr)
            setattr(self, attr, "not_a_key")

            with self.subTest(bad_key=attr):
                # Run and check the error
                self.assertRaisesRegex(ValueError, "unknown time coordinate",
                                       checks.evaluate_boundary_in_mlt,
                                       *[self.bound_data, self.eq_key,
                                         self.po_key, self.lt_key, self.ut_key])

                # Reset the key
                setattr(self, attr, orig_val)
        return

    def test_bad_mlt_frequency(self):
        """Test raises ValueError with bad MLT coordinates."""
        # Update the MLT coordinates
        self.mlt[0] += 0.01
        self.bound_data.assign({self.lt_key: self.mlt})

        # Run and check the error
        self.assertRaisesRegex(ValueError, "not have a fixed frequency",
                               checks.evaluate_boundary_in_mlt,
                               *[self.bound_data, self.eq_key, self.po_key,
                                 self.lt_key, self.ut_key])
        return

    def test_eval_success_no_removal(self):
        """Test evaluation removes no outlier boundaries."""
        # Run the outlier check
        checks.evaluate_boundary_in_mlt(
            self.bound_data, self.eq_key, self.po_key, self.lt_key, self.ut_key)

        # Evalute the lack of removal
        self.assertTrue(np.isfinite(self.bound_data[self.po_key].values).all())
        self.assertTrue(np.isfinite(self.bound_data[self.eq_key].values).all())
        return

    def test_eval_success_with_removal(self):
        """Test evaluation removes no outlier boundaries."""
        # Insert an outlier
        iout = (0, 1)
        temp = self.bound_data[self.po_key].values
        temp[iout] = 40.0
        self.bound_data.assign({self.po_key: (self.bound_data[self.po_key].dims,
                                              temp)})

        # Run outlier check
        checks.evaluate_boundary_in_mlt(
            self.bound_data, self.eq_key, self.po_key, self.lt_key, self.ut_key)

        # Evaluate removal from only the one polar location
        self.assertTrue(np.isnan(self.bound_data[self.po_key][iout]))
        self.assertTrue(np.isfinite(self.bound_data[self.po_key].values).sum()
                        == self.mlt.shape[0] - 1)
        self.assertTrue(np.isfinite(self.bound_data[self.eq_key].values).all())
        return
