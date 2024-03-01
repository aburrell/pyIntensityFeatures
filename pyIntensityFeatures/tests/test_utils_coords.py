#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.coords`."""

import datetime as dt
import numpy as np
import pandas as pds
import unittest

from pyIntensityFeatures.utils import coords

try:
    import apexpy
except ImportError:
    apexpy = None


class TestTimeFuncs(unittest.TestCase):
    """Tests for time-handling functions."""

    def setUp(self):
        """Set up the test runs."""
        self.dtime = dt.datetime(1999, 2, 11)
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.dtime, self.out
        return

    def test_as_datetime(self):
        """Test success for datetime casting."""
        # Cycle through potential time formats
        for in_time in [dt.date(self.dtime.year, self.dtime.month,
                                self.dtime.day),
                        np.datetime64(self.dtime.strftime('%Y-%m-%d')),
                        pds.datetime(self.dtime.year, self.dtime.month,
                                     self.dtime.day), self.dtime]:
            with self.subTest(in_time=in_time):
                # Convert the time
                self.out = coords.as_datetime(in_time)
                self.assertEqual(self.out, self.dtime)
        return


class TestSliceMinMax(unittest.TestCase):
    """Tests for `get_slice_mlat_max_min`."""

    def setUp(self):
        """Set up the test runs."""
        self.mlat_inc = 0.5
        self.mlat_bins = np.arange(50, 90, self.mlat_inc)
        self.mlt_bins = np.arange(0.0, 24.0, self.mlat_inc)
        self.num_samples = np.full(shape=(self.mlat_bins.shape[0],
                                          self.mlt_bins.shape[0]),
                                   fill_value=10)
        self.mlat_min = self.mlat_bins.min()
        self.mlat_max = self.mlat_bins.max()
        self.out_min = None
        self.out_max = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.mlat_inc, self.mlat_bins, self.mlt_bins, self.num_samples
        del self.mlat_min, self.mlat_max, self.out_min, self.out_max
        return

    def test_get_slice_mlat_max_min_all_good(self):
        """Test with all bins valid."""
        for hemi in [1.0, -1.0]:
            with self.subTest(hemisphere=hemi):
                self.out_max, self.out_min = coords.get_slice_mlat_max_min(
                    self.num_samples, hemi * self.mlat_bins, self.mlt_bins,
                    mlat_inc=self.mlat_inc)

                # Ensure difference in output max/min is less than the bin
                # increment away from the input min/max
                self.assertLess(abs(self.out_max - hemi * self.mlat_max),
                                self.mlat_inc)
                self.assertLess(abs(self.out_min - hemi * self.mlat_min),
                                self.mlat_inc)
        return

    def test_get_slice_mlat_max_min_with_cutout(self):
        """Test with a cutout in samples."""
        # Change the number of samples at the first time
        self.num_samples[:int(len(self.mlat_bins) / 2), 0] = 0

        for hemi in [1.0, -1.0]:
            with self.subTest(hemisphere=hemi):
                self.out_max, self.out_min = coords.get_slice_mlat_max_min(
                    self.num_samples, hemi * self.mlat_bins, self.mlt_bins,
                    mlat_inc=self.mlat_inc)

                # Ensure difference in output max/min is less than the bin
                # increment away from the input min/max
                self.assertLess(abs(self.out_max - hemi * self.mlat_max),
                                self.mlat_inc)

                # Ensure the returned minimum value is above the bin minimum
                self.assertGreater(hemi * self.out_min, self.mlat_min)
        return

    def test_get_slice_mlat_max_min_all_bad(self):
        """Test with no valid samples."""
        # Change the number of samples at the first time
        self.num_samples[:, :] = 0

        for hemi in [1.0, -1.0]:
            with self.subTest(hemisphere=hemi):
                self.out_max, self.out_min = coords.get_slice_mlat_max_min(
                    self.num_samples, hemi * self.mlat_bins, self.mlt_bins,
                    mlat_inc=self.mlat_inc)

                # Ensure the defaults are returned
                self.assertEqual(self.out_max, 0.0)
                self.assertEqual(self.out_min, 0.0)
        return


class TestConvertGeoToMag(unittest.TestCase):
    """Tests for `convert_geo_to_mag`."""

    def setUp(self):
        """Set up the test runs."""
        self.ctime = dt.datetime(1999, 2, 11)
        self.glat = np.linspace(45.0, 90.0, 25).reshape(5, 5)
        self.glon = np.full(shape=(5, 5), fill_value=180.0)
        self.alt = 110.0
        self.mlat = None
        self.mlon = None
        self.mlt = None
        self.aacgmv2_method = ['TRACE', 'ALLOWTRACE', 'BADIDEA', 'GEOCENTRIC']
        self.apexpy_method = ['apex', 'qd']
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.ctime, self.glat, self.glon, self.alt, self.mlat, self.mlon
        del self.mlt, self.aacgmv2_method, self.apexpy_method
        return

    def test_aacgmv2_conversion(self):
        """Test that the conversion with the AACGMV2 methods."""
        # Cycle through all the AACGMV2 methods
        for method in self.aacgmv2_method:
            with self.subTest(method=method):
                # Run the conversion
                self.mlat, self.mlon, self.mlt = coords.convert_geo_to_mag(
                    self.ctime, self.glat, self.glon, self.alt, method=method)

                # Test the output shapes
                self.assertEqual(self.mlat.shape, self.glat.shape)
                self.assertEqual(self.mlon.shape, self.glat.shape)
                self.assertEqual(self.mlt.shape, self.glat.shape)

                # Test the max and min latitude indices
                self.assertEqual(self.mlat.argmin(), self.glat.argmin())
                self.assertEqual(self.mlat.argmax(), self.glat.argmax())

    @unittest.skipIf(apexpy is None, "cannot test apexpy without module")
    def test_apexpy_conversion(self):
        """Test that the conversion with the apexpy methods."""
        # Cycle through all the apexpy methods
        for method in self.apexpy_method:
            with self.subTest(method=method):
                # Run the conversion
                self.mlat, self.mlon, self.mlt = coords.convert_geo_to_mag(
                    self.ctime, self.glat, self.glon, self.alt, method=method)

                # Test the output shapes
                self.assertEqual(self.mlat.shape, self.glat.shape)
                self.assertEqual(self.mlon.shape, self.glat.shape)
                self.assertEqual(self.mlt.shape, self.glat.shape)

                # Test the max and min latitude indices
                self.assertEqual(self.mlat.argmin(), self.glat.argmin())
                self.assertEqual(self.mlat.argmax(), self.glat.argmax())

    @unittest.skipIf(apexpy is not None,
                     "cannot test apexpy failure with module")
    def test_apexpy_conversion_no_apexpy(self):
        """Test the conversion raises ValueError with the apexpy methods."""
        # Cycle through all the apexpy methods
        for method in self.apexpy_method:
            with self.subTest(method=method):
                # Run the conversion and catch the error
                args = [self.ctime, self.glat, self.glon, self.alt, method]
                self.assertRaisesRegex(ValueError, "apexpy is not available.",
                                       coords.convert_geo_to_mag, *args)
        return
