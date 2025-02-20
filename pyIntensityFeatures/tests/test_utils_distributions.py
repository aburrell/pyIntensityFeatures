#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.distributions`."""

import numpy as np
import unittest

from pyIntensityFeatures.utils import distributions as dist


class TestCalcQuadratic(unittest.TestCase):
    """Tests for `calc_quadratic`."""

    def setUp(self):
        """Set up the test runs."""
        # Retrieved quadratic values and results from:
        # https://www.mathsisfun.com/algebra/quadratic-equation.html
        self.a = 5.0
        self.b = 6.0
        self.c = 1.0
        self.x = [-0.2, -1]
        self.y = 0.0
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.a, self.b, self.c, self.x, self.y
        return

    def test_float_calc(self):
        """Test success for a float calculation."""
        # Cycle through the x-values
        for val in self.x:
            with self.subTest(x=val):
                # Solve the quadratic equation
                out = dist.calc_quadratic(val, self.c, self.b, self.a)
                self.assertAlmostEqual(out, self.y)
        return

    def test_array_calc(self):
        """Test success for an array calculation."""
        # Cast the x-values
        self.x = np.array(self.x)

        # Solve the quadratic equation
        out = dist.calc_quadratic(self.x, self.c, self.b, self.a)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(np.all(abs(self.y - out) < 1.0e-7),
                        msg="Unexpected y values: {:}".format(out))
        return


class TestSingleGauss(unittest.TestCase):
    """Tests for the single-peaked Gaussian functions."""

    def setUp(self):
        """Set up the test runs."""
        self.const = 1.0
        self.a = 0.001
        self.b = 0.01
        vals = np.random.normal(5.0, 0.3, 10000) + self.const
        self.num, bins = np.histogram(vals, 50)
        self.mu = vals.mean()
        self.sigma = vals.std()
        self.loc = (bins[:-1] + (bins[1:] - bins[:-1]) / 2.0).astype(float)
        self.num = self.num.astype(float)
        self.amp = self.num.max()
        self.max_mae = 20.0
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.const, self.num, self.mu, self.sigma, self.amp, self.loc
        del self.out, self.max_mae, self.a, self.b
        return

    def test_gauss(self):
        """Test the `gauss` function values."""
        # Get the Gaussian solution for the location
        self.out = dist.gauss(self.loc, self.amp, self.mu, self.sigma,
                              self.const)

        # Evaluate the mean absolute error
        self.assertLess(abs(np.mean(self.out - self.num)), self.max_mae)
        return

    def test_single_gauss_quad(self):
        """Test the `single_gauss_quad` function values."""
        # Add a quadratic background to the values and adjust the amplitude
        self.num += dist.calc_quadratic(self.loc, self.const, self.b, self.a)
        self.amp = self.num.max()

        # Get the Gaussian solution for the location
        self.out = dist.single_gauss_quad(self.loc, self.const, self.b, self.a,
                                          self.amp, self.mu, self.sigma)

        # Evaluate the mean absolute error
        self.assertLess(abs(np.mean(self.out - self.num)), self.max_mae)
        return


class TestMultGauss(unittest.TestCase):
    """Tests for the multi-peaked Gaussian functions."""

    def setUp(self):
        """Set up the test runs."""
        self.const = 1.0
        self.a = 0.001
        self.b = 0.01
        self.npeaks = 1
        self.mu = []
        self.sigma = []
        self.amp = []
        self.loc = None
        self.num = None
        self.max_mae = 20.0
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.const, self.num, self.mu, self.sigma, self.amp, self.loc
        del self.out, self.max_mae, self.a, self.b, self.npeaks
        return

    def set_gaussian(self, add_quad=False):
        """Set the Gaussian values to test against.

        Parameters
        ----------
        add_quad : bool
            Add a quadratic background if True (default=False)

        """
        # Reset the output arrays
        self.mu = []
        self.sigma = []
        self.amp = []

        # Get the points at each peak
        vals = []
        for peak_mu in range(self.npeaks):
            peak_vals = np.random.normal(peak_mu, 0.3, 1000)
            peak_num, _ = np.histogram(peak_vals, 50)

            # Set the peak-specific values that won't change upon addition
            self.mu.append(np.mean(peak_vals))
            self.sigma.append(np.std(peak_vals))

            # Add the peak-specific values to the output
            vals.extend(list(peak_vals))

        # Calculate the total locations and values
        self.num, bins = np.histogram(vals, 50)
        self.loc = (bins[:-1] + (bins[1:] - bins[:-1]) / 2.0).astype(float)
        self.num = self.num.astype(float)

        # If desired, add a quadratic background
        if add_quad:
            self.num += dist.calc_quadratic(self.loc, 0.0, self.b, self.a)

        # Get the amplitues at each peak location
        for x in self.mu:
            self.amp.append(self.num[abs(self.loc - x).argmin()])

        return

    def test_mult_gauss(self):
        """Test the `mult_gauss` function values for 1, 2, 3, and 4 peaks."""
        # Cycle through float and array-like input
        for iloc in [0, slice(None, None)]:
            # Cycle through the number of peaks
            for self.npeaks in range(4):
                with self.subTest(npeaks=self.npeaks, xind=iloc):
                    # Update the comparitive data
                    self.set_gaussian(add_quad=False)

                    # Get the Gaussian solution for the location
                    self.out = dist.mult_gauss(self.loc[iloc], self.amp,
                                               self.mu, self.sigma, self.const)

                    # Evaluate the mean absolute error
                    self.assertLess(abs(np.mean(self.out - self.num[iloc])),
                                    self.max_mae)
        return

    def test_mult_gauss_quad(self):
        """Test `mult_gauss_quad` for 1, 2, 3, and 4 Gaussian peaks."""
        # Cycle through the number of peaks
        for self.npeaks in np.arange(1, 5, 1):
            # Initalize the comparitive data
            self.set_gaussian(add_quad=True)

            # Initalize the parameter input
            params = [self.const, self.b, self.a]
            for i, amplitude in enumerate(self.amp):
                params.append(amplitude)
                params.append(self.mu[i])
                params.append(self.sigma[i])

            with self.subTest(npeaks=self.npeaks):
                # Get the Gaussian solution for the location
                self.out = dist.mult_gauss_quad(self.loc, *params)

                # Evaluate the mean absolute error
                self.assertLess(abs(np.mean(self.out - self.num)), self.max_mae)
        return

    def test_mult_gauss_quad_bad_amplitude(self):
        """Test `mult_gauss_quad` raises an error for an extra input."""

        args = [1.0, [self.const, self.b, self.a, 1.0, 2.0, 3.0, 1.0]]

        self.assertRaisesRegex(ValueError,
                               "Unexpected number of input parameters",
                               dist.mult_gauss_quad, *args)
        return

    def test_mult_gauss_bad_input_shape(self):
        """Test `mult_gauss` raises an error with badly shaped input."""

        self.set_npeaks = 5
        self.set_gaussian()
        self.mu.pop()
        args = [1.0, self.amp, self.mu, self.sigma, self.const]

        self.assertRaisesRegex(ValueError,
                               "Unexpected number of input parameters",
                               dist.mult_gauss, *args)
        return
