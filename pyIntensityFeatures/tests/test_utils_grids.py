#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.grids`."""

import numpy as np
import unittest

from pyIntensityFeatures.utils import grids


class TestGridUniq(unittest.TestCase):
    """Tests for local `unique` function."""

    def setUp(self):
        """Set up the test runs."""
        self.numbers = [0.0, 0.1, 0.100001, 10.0, 11.0]
        self.test_dec = [1, 0, -1, -2]
        self.test_nuniq = [4, 3, 2, 1]
        self.uniq = []
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.numbers, self.uniq, self.test_dec, self.test_nuniq

    def test_unique_decimal(self):
        """Test that `unique` yeilds the correct number of values."""
        for i, dec in enumerate(self.test_dec):
            with self.subTest(decimals=dec, num_uniq=self.test_nuniq[i]):
                self.uniq = grids.unique(self.numbers, decimals=dec)
                self.assertEqual(len(self.uniq), self.test_nuniq[i])
        return

    def test_unique_equal_nan(self):
        """Test that `unique` allows use of `equal_nan` kwarg."""
        self.numbers.extend([np.nan, np.nan])
        self.test_nuniq = np.array(self.test_nuniq) + 1

        for i, dec in enumerate(self.test_dec):
            with self.subTest(decimals=dec):
                # Test with NaN grouped together
                self.uniq = grids.unique(self.numbers, decimals=dec,
                                         equal_nan=True)
                self.assertEqual(len(self.uniq), self.test_nuniq[i])

                # Test with NaN not grouped together
                self.uniq = grids.unique(self.numbers, decimals=dec,
                                         equal_nan=False)
                self.assertEqual(len(self.uniq), self.test_nuniq[i] + 1)
        return

    def test_unique_axis(self):
        """Test that `unique` allows use of `axis` kwarg."""
        self.numbers = [self.numbers, self.numbers]
        self.test_nuniq = [(self.test_nuniq[-1],), (1, len(self.numbers[0])),
                           (len(self.numbers), self.test_nuniq[-1])]

        for i, axis in enumerate([None, 0, 1]):
            with self.subTest(axis=axis):
                self.uniq = grids.unique(self.numbers,
                                         decimals=self.test_dec[-1], axis=axis)
                self.assertTupleEqual(self.test_nuniq[i], self.uniq.shape)
        return

    def test_unique_return_counts(self):
        """Test that `unique` allows use of `return_counts` kwarg."""
        self.uniq = grids.unique(self.numbers, decimals=self.test_dec[-1],
                                 return_counts=True)

        self.assertEqual(len(self.uniq[0]), self.test_nuniq[-1])
        self.assertEqual(self.uniq[1][0], len(self.numbers))
        return

    def test_unique_return_index(self):
        """Test that `unique` allows use of `return_index` kwarg."""
        self.uniq = grids.unique(self.numbers, decimals=self.test_dec[-1],
                                 return_index=True)

        self.assertEqual(len(self.uniq[0]), self.test_nuniq[-1])
        self.assertEqual(self.uniq[1][0], 0)
        return

    def test_unique_return_inverse(self):
        """Test that `unique` allows use of `return_inverse` kwarg."""
        self.uniq = grids.unique(self.numbers, decimals=self.test_dec[-1],
                                 return_inverse=True)

        self.assertEqual(len(self.uniq[0]), self.test_nuniq[-1])
        self.assertEqual(len(self.uniq[1]), len(self.numbers),
                         msg="unexpected inverse return: {:}".format(
                             self.uniq[1]))
        self.assertListEqual(list(self.uniq[1]),
                             [0 for i in range(len(self.numbers))])
        return


class TestGridIntensity(unittest.TestCase):
    """Tests for `grid_intensity`."""

    def setUp(self):
        """Set up the test runs."""
        self.intensity = np.ones(shape=(100,), dtype=float)
        self.mlat = np.linspace(40.0, 90.0, 100)
        self.mlt = np.linspace(0.0, 24.0, 100)
        self.eq_mlat = 50.0
        self.mlat_inc = 5.0
        self.mlt_inc = 2.0
        self.mean_intensity = None
        self.std_intensity = None
        self.num_intensity = None
        self.mlat_bins = None
        self.mlt_bins = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.intensity, self.mlat, self.mlt, self.eq_mlat, self.mlat_inc
        del self.mlt_inc, self.mean_intensity, self.std_intensity
        del self.num_intensity
        return

    def eval_grid_output(self):
        """Evaluate the grid output values."""
        # Test the magnetic latitude bins
        self.assertLess(self.mlat_bins.shape[0], np.prod(self.mlat.shape))
        self.assertGreaterEqual(self.mlat_bins.min(), self.eq_mlat)
        res = np.unique(self.mlat_bins[1:] - self.mlat_bins[:-1])
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res[0], self.mlat_inc)

        # Test the magnetic local time bins
        self.assertLess(self.mlt_bins.shape[0], np.prod(self.mlt.shape))
        res = np.unique(self.mlt_bins[1:] - self.mlt_bins[:-1])
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res[0], self.mlt_inc)

        # Test the grid shape
        res = (self.mlat_bins.shape[0], self.mlt_bins.shape[0])
        self.assertEqual(self.mean_intensity.shape, res)
        self.assertEqual(self.std_intensity.shape, res)
        self.assertEqual(self.num_intensity.shape, res)

        # Test the grid fill values
        no_pnt_mask = self.num_intensity == 0
        self.assertTrue(np.all(np.isnan(self.mean_intensity[no_pnt_mask])))
        self.assertTrue(np.all(np.isnan(self.std_intensity[no_pnt_mask])))

        # Test the grid values, since all intensity values are the same the
        # mean will be the same as any intensity value and the standard
        # deviation will be zero
        self.assertTrue(np.all(self.mean_intensity[~no_pnt_mask]
                               == self.intensity.min()))
        self.assertTrue(np.all(self.std_intensity[~no_pnt_mask] == 0.0))
        return

    def test_grid_intensity_1d(self):
        """Test success of gridding 1D intensity data."""
        # Grid the data
        (self.mean_intensity, self.std_intensity, self.num_intensity,
         self.mlat_bins, self.mlt_bins) = grids.grid_intensity(
             self.intensity, self.mlat, self.mlt, eq_mlat=self.eq_mlat,
             mlat_inc=self.mlat_inc, mlt_inc=self.mlt_inc)

        # Test the output
        self.eval_grid_output()
        return

    def test_grid_intensity_multi_dim(self):
        """Test success of gridding multi-dimensional intensity data."""
        for new_shape in [(2, 50), (2, 2, 25), (2, 5, 2, 5)]:
            self.intensity = self.intensity.reshape(new_shape)
            self.mlat = self.mlat.reshape(new_shape)
            self.mlt = self.mlt.reshape(new_shape)

            with self.subTest(shape=new_shape):
                # Grid the data
                (self.mean_intensity, self.std_intensity, self.num_intensity,
                 self.mlat_bins, self.mlt_bins) = grids.grid_intensity(
                     self.intensity, self.mlat, self.mlt, eq_mlat=self.eq_mlat,
                     mlat_inc=self.mlat_inc, mlt_inc=self.mlt_inc)

                # Test the output
                self.eval_grid_output()
        return
