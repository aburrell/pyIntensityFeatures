#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `proc.intensity`."""

import datetime as dt
import numpy as np
import unittest

from pyIntensityFeatures.utils.distributions import mult_gauss_quad
from pyIntensityFeatures.proc import intensity

try:
    import apexpy
except ImportError:
    apexpy = None


class TestIntensityFuncs(unittest.TestCase):
    """Tests for the intensity functions."""

    def setUp(self):
        """Set up the test runs."""
        lats = np.arange(60, 90.1, 0.1)
        self.glat = np.full(shape=(10, lats.shape[0]), fill_value=lats)
        self.glon = np.full(shape=(self.glat.shape[1], self.glat.shape[0]),
                            fill_value=np.linspace(
                                -80.0, 80.0, self.glat.shape[0])).transpose()
        self.intensity = mult_gauss_quad(self.glat, *[
            1.0, 0.01, 0.001, 500.0, 75.0, 3.0, 400, 79.0, 1.0, 300, 65.0, 0.5])
        self.sweep_times = [dt.datetime(1999, 2, 11),
                            dt.datetime(1999, 2, 11, 0, 15, 1)]
        self.alt = 110.0
        self.min_mlat_base = 50.0
        self.max_coeff = 0
        self.mlat_inc = 2.0
        self.mlt_inc = 1.0
        self.strict_fit = False
        self.dayglow_thresh = 300.0
        self.aacgmv2_method = ['TRACE', 'ALLOWTRACE', 'BADIDEA', 'GEOCENTRIC']
        self.apexpy_method = ['apex', 'qd']
        self.sweep_end = None
        self.out_data = None
        self.out_coeff = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.glat, self.glon, self.intensity, self.sweep_times, self.alt
        del self.min_mlat_base, self.max_coeff, self.mlat_inc, self.mlt_inc
        del self.strict_fit, self.aacgmv2_method, self.apexpy_method
        del self.sweep_end, self.out_data, self.out_coeff, self.dayglow_thresh
        return

    def eval_intensity_output(self, hemisphere):
        """Evaluate the `out_data` attribute.

        Parameters
        ----------
        hemisphere : int
            -1 for Southern, 1 for Northern.

        """
        # Ensure data is set
        self.assertIsNotNone(self.out_data)

        # Evaluate the dictionary keys
        out_keys = ['sweep_start', 'sweep_end', 'mlt', 'mlat', 'eq_bounds',
                    'eq_uncert', 'eq_params', 'po_bounds', 'po_uncert',
                    'po_params', 'mean_intensity', 'std_intensity',
                    'num_intensity']
        self.assertListEqual(out_keys, list(self.out_data.keys()))

        # Evaluate the sweep times
        self.assertEqual(self.out_data['sweep_start'], self.sweep_times[0])
        self.assertEqual(self.out_data['sweep_end'], self.sweep_times[1])

        # Evaluate the grid bins
        res = np.unique(self.out_data['mlt'][1:] - self.out_data['mlt'][:-1])
        self.assertEqual(len(res), 1, msg="changing MLT resolution")
        self.assertEqual(res[0], self.mlt_inc, msg="wrong MLT resolution")

        res = np.unique(self.out_data['mlat'][1:] - self.out_data['mlat'][:-1])
        self.assertEqual(len(res), 1, msg="changing MLat resolution")
        self.assertEqual(hemisphere * res[0], self.mlat_inc,
                         msg="wrong MLat resolution")
        self.assertGreaterEqual(self.min_mlat_base + self.mlat_inc,
                                (hemisphere * self.out_data['mlat']).min(),
                                msg="Bad MLat bins: {:}".format(
                                    self.out_data['mlat']))

        # Evaluate the ALBs
        for i, ealb in enumerate(self.out_data['eq_bounds']):
            # Evaluate the EALB
            if np.isnan(ealb):
                self.assertTrue(np.isnan(self.out_data['eq_uncert'][i]))
                self.assertListEqual(self.out_data['eq_params'][i], [])
            else:
                self.assertGreater(hemisphere * ealb, self.min_mlat_base)
                self.assertLessEqual(self.out_data['eq_uncert'][i], 3.0)
                self.assertLessEqual(len(self.out_data['eq_params'][i]), 12)

            # Evaluate the PALB
            if np.isnan(self.out_data['po_bounds'][i]):
                self.assertTrue(np.isnan(self.out_data['po_uncert'][i]))
                self.assertListEqual(self.out_data['po_params'][i], [])
            else:
                self.assertLess(hemisphere * self.out_data['po_bounds'][i], 90)
                self.assertLessEqual(self.out_data['po_uncert'][i], 3.0)
                self.assertLessEqual(len(self.out_data['po_params'][i]), 12)

        # Evaluate the gridded intensity, standard deviation, and number
        self.assertLessEqual(np.nanmax(self.out_data['mean_intensity']),
                             self.intensity.max())
        self.assertGreaterEqual(np.nanmin(self.out_data['mean_intensity']),
                                self.intensity.min())
        self.assertTrue(np.isfinite(self.out_data['std_intensity']).any())
        self.assertEqual(self.out_data['num_intensity'].min(), 0)
        self.assertGreater(self.out_data['num_intensity'].max(), 1)
        return

    def test_find_intensity_boundaries_dayglow_threshold(self):
        """Test boundary ID rejection with a too-low dayglow threshold."""
        self.dayglow_thresh = -100.0

        for hemi in [-1, 1]:
            with self.subTest(hemi=hemi):
                # Get the boundary outputs
                (self.sweep_end, self.out_data,
                 self.out_coeff) = intensity.find_intensity_boundaries(
                     self.intensity, hemi * self.glat, self.glon,
                     self.sweep_times, self.alt, self.min_mlat_base,
                     self.max_coeff, mlat_inc=self.mlat_inc,
                     mlt_inc=self.mlt_inc, strict_fit=self.strict_fit,
                     dayglow_threshold=self.dayglow_thresh)

                # Evaluate the outputs
                self.assertEqual(self.sweep_end, self.sweep_times[-1])
                #self.assertIsNone(self.out_data)
                self.assertEqual(self.out_coeff, 0,
                                 msg='unexpected coefficients: {:} {:}'.format(
                                     self.out_coeff, self.out_data))
        return

    def test_find_intensity_boundaries_aaacgm(self):
        """Test boundary ID in an intensity slice using AACGMV2 coordinates."""

        for hemi in [-1, 1]:
            for method in self.aacgmv2_method:
                with self.subTest(hemi=hemi, method=method):
                    # Get the boundary outputs
                    (self.sweep_end, self.out_data,
                     self.out_coeff) = intensity.find_intensity_boundaries(
                         self.intensity, hemi * self.glat, self.glon,
                         self.sweep_times, self.alt, self.min_mlat_base,
                         self.max_coeff, method=method, mlat_inc=self.mlat_inc,
                         mlt_inc=self.mlt_inc, strict_fit=self.strict_fit)

                    # Evaluate the outputs
                    self.assertEqual(self.sweep_end, self.sweep_times[-1])
                    self.assertLessEqual(self.out_coeff, 12)
                    self.assertGreater(self.out_coeff, 0)
                    self.eval_intensity_output(hemi)
        return

    @unittest.skipIf(apexpy is None, "cannot test apexpy without module")
    def test_find_intensity_boundaries_apexpy(self):
        """Test boundary ID in an intensity slice using apexpy coordinates."""

        for hemi in [-1, 1]:
            for method in self.apexpy_method:
                with self.subTest(hemi=hemi, method=method):
                    # Get the boundary outputs
                    (self.sweep_end, self.out_data,
                     self.out_coeff) = intensity.find_intensity_boundaries(
                         self.intensity, hemi * self.glat, self.glon,
                         self.sweep_times, self.alt, self.min_mlat_base,
                         self.max_coeff, method=method, mlat_inc=self.mlat_inc,
                         mlt_inc=self.mlt_inc, strict_fit=self.strict_fit)

                    # Evaluate the outputs
                    self.assertEqual(self.sweep_end, self.sweep_times[-1])
                    self.assertLessEqual(self.out_coeff, 12)
                    self.assertGreater(self.out_coeff, 0)
                    self.eval_intensity_output(hemi)
        return

    @unittest.skipIf(apexpy is not None,
                     "cannot test apexpy failure with module")
    def test_find_intensity_boundaries_no_apexpy(self):
        """Test boundary ID raises ValueError with the apexpy methods."""
        # Cycle through all the apexpy methods
        for method in self.apexpy_method:
            with self.subTest(method=method):
                # Run the conversion and catch the error
                args = [self.intensity, self.glat, self.glon, self.sweep_times,
                        self.alt, self.min_mlat_base, self.max_coeff, method]
                self.assertRaisesRegex(ValueError, "apexpy is not available.",
                                       intensity.find_intensity_boundaries,
                                       *args)
        return
