#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `instruments.satellites`."""

import datetime as dt
import numpy as np
import unittest

from pyIntensityFeatures.instruments import satellites


class TestSatelliteFuncs(unittest.TestCase):
    """Tests for the satellite functions."""

    def setUp(self):
        """Set up the test runs."""
        # Intialize the inputs
        self.time_data = [dt.datetime(1999, 2, 11) + dt.timedelta(seconds=i)
                          for i in range(86400)]
        self.glat = np.ones(shape=(len(self.time_data), 30))
        self.glon = np.ones(shape=self.glat.shape)
        self.intensity = np.ones(shape=self.glat.shape)
        self.clean_mask = np.zeros(shape=self.glat.shape).astype(bool)
        self.min_colat = 40.0
        self.start_time = None

        self.glat[0] = np.linspace(-5.0, 5.0, num=self.glat.shape[1])
        self.glon[0] = np.linspace(230.0, 210.0, num=self.glon.shape[1])

        for i, lat in enumerate(self.glat[0]):
            self.glat[:, i] = lat + (90.0 - abs(lat)) * np.sin(
                np.linspace(0, 2.0 * np.pi, self.glat.shape[0]))
            self.glon[:, i] = self.glon[0, i] + (
                360.0 - self.glon[0, i]) * np.sin(
                    np.linspace(0, 2.0 * np.pi, self.glon.shape[0]))

        # Initalize the outputs
        self.out_intensity = None
        self.out_glat = None
        self.out_glon = None
        self.out_times = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.time_data, self.glat, self.glon, self.intensity
        del self.clean_mask, self.min_colat, self.out_intensity, self.out_glat
        del self.out_glon, self.out_times, self.start_time
        return

    def eval_auroral_slice(self):
        """Evaluate the output from `get_auroral_slice`."""
        # Ensure there is auroral data in the slice
        self.assertGreater(abs(self.out_glat).max(), self.min_colat)

        # Ensure the lat/lon/intensity output have the same shape
        self.assertTupleEqual(self.out_intensity.shape, self.out_glat.shape)
        self.assertTupleEqual(self.out_glon.shape, self.out_glat.shape)

        # Ensure the intensity value is appropriate
        self.assertTrue(np.all(self.out_intensity == self.intensity.min()),
                        msg="output intensities differ, check input")

        # Evaluate the times
        self.eval_times(is_last=False)
        return

    def eval_times(self, is_last=False):
        """Evaluate the output times.

        Parameters
        ----------
        is_last : bool
            Assert the last time is equal to the last time of the input times

        """
        # Ensure the start time is appropriate
        if self.start_time is None:
            self.assertTrue(self.out_times[0] >= self.time_data[0])
        else:
            self.assertTrue(self.out_times[0] >= self.start_time)

        # Ensure the end time is appropriate
        if is_last:
            self.assertTrue(self.out_times[1] == self.time_data[-1])
        else:
            self.assertTrue(self.out_times[1] <= self.time_data[-1])
        return

    def test_get_auroral_slice_bad_intensity_shape(self):
        """Test raises ValueError with the bad intensity data shape."""
        # Reshape the intensity data
        self.intensity = self.intensity.transpose()

        # Run the slice ID function and evaluate the error
        args = [self.time_data, self.glat, self.glon, self.intensity]
        self.assertRaisesRegex(ValueError, "first dimension of intensity data",
                               satellites.get_auroral_slice, *args)
        return

    def test_get_auroral_slice_bad_lat_shape(self):
        """Test raises ValueError with the bad latitude data shape."""
        # Reshape the intensity data
        self.glat = self.glat.transpose()

        # Run the slice ID function and evaluate the error
        args = [self.time_data, self.glat, self.glon, self.intensity]
        self.assertRaisesRegex(ValueError, "intensity and location input ",
                               satellites.get_auroral_slice, *args)
        return

    def test_get_auroral_slice_bad_lon_shape(self):
        """Test raises ValueError with the bad longitude data shape."""
        # Reshape the intensity data
        self.glon = self.glon.transpose()

        # Run the slice ID function and evaluate the error
        args = [self.time_data, self.glat, self.glon, self.intensity]
        self.assertRaisesRegex(ValueError, "intensity and location input ",
                               satellites.get_auroral_slice, *args)
        return

    def test_get_auroral_slice_bad_clean_mask_shape(self):
        """Test raises ValueError with the bad clean mask shape."""
        # Reshape the intensity data
        self.clean_mask = self.clean_mask[0, :]

        # Run the slice ID function and evaluate the error
        args = [self.time_data, self.glat, self.glon, self.intensity,
                self.clean_mask]
        self.assertRaisesRegex(ValueError, "clean mask shape differs from ",
                               satellites.get_auroral_slice, *args)
        return

    def test_get_auroral_slice_no_mask(self):
        """Test slice output without a mask."""
        # Cycle through both hemispheres
        for hemi in [-1, 1]:
            # Cycle through start time options
            for self.start_time in [None, self.time_data[20000],
                                    self.time_data[20000]
                                    + dt.timedelta(microseconds=1)]:
                with self.subTest(hemi=hemi, start_time=self.start_time):
                    # Run the slice ID function
                    (self.out_intensity, self.out_glat, self.out_glon,
                     self.out_times) = satellites.get_auroral_slice(
                         self.time_data, hemi * self.glat, self.glon,
                         self.intensity, start_time=self.start_time,
                         hemisphere=hemi, min_colat=self.min_colat)

                    # Evaluate the output
                    self.eval_auroral_slice()
        return

    def test_get_auroral_slice_mask(self):
        """Test slice output with a mask that removes all data."""
        # Cycle through both hemispheres
        for hemi in [-1, 1]:
            # Cycle through start time options
            for self.start_time in [None, self.time_data[20000],
                                    self.time_data[20000]
                                    + dt.timedelta(microseconds=1)]:
                with self.subTest(hemi=hemi, start_time=self.start_time):
                    # Run the slice ID function
                    (self.out_intensity, self.out_glat, self.out_glon,
                     self.out_times) = satellites.get_auroral_slice(
                         self.time_data, hemi * self.glat, self.glon,
                         self.intensity, clean_mask=self.clean_mask,
                         start_time=self.start_time, hemisphere=hemi,
                         min_colat=self.min_colat)

                    # Evaluate the output
                    self.assertTupleEqual(self.out_intensity.shape, (0, ))
                    self.assertTupleEqual(self.out_glat.shape, (0, ))
                    self.assertTupleEqual(self.out_glon.shape, (0, ))
                    self.eval_times(is_last=True)
        return

    def test_get_auroral_slice_bad_lat_range(self):
        """Test slice output with a lat range that doesn't contain a slice."""
        self.time_data = self.time_data[:5837]
        self.glat = self.glat[:5837, :]
        self.glon = self.glon[:5837, :]
        self.intensity = self.intensity[:5837, :]

        # Cycle through both hemispheres
        for hemi in [-1, 1]:
            # Cycle through start time options
            with self.subTest(hemi=hemi):
                # Run the slice ID function
                (self.out_intensity, self.out_glat, self.out_glon,
                 self.out_times) = satellites.get_auroral_slice(
                     self.time_data, hemi * self.glat, self.glon,
                     self.intensity, start_time=self.start_time,
                     hemisphere=hemi, min_colat=self.min_colat)

                # Evaluate the output
                self.assertTupleEqual(self.out_intensity.shape,
                                      (1, self.intensity.shape[1]))
                self.assertTupleEqual(self.out_glat.shape,
                                      (1, self.glat.shape[1]))
                self.assertTupleEqual(self.out_glon.shape,
                                      (1, self.glon.shape[1]))
                self.eval_times(is_last=True)
        return
