#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Tests for functions in `_main`."""

from io import StringIO
import logging
import datetime as dt
import numpy as np
import pandas as pds
import unittest
import xarray as xr

import pyIntensityFeatures


def clean_func(inst_data, clean_var="clean_flag", bad_val=1):
    """Clean data as a test for the `clean_func` attribute.

    Parameters
    ----------
    inst_data : dict, list, array, pds.DataFrame, or xr.Dataset
        Instrument data
    clean_var : str or int
        Data variable with the clean flag (default='clean_flag')
    bad_val : float or int
        Values to flag as bad (default=1)

    Returns
    -------
    clean_mask : 2D array or NoneType
        Cleaning mask output

    """
    if isinstance(inst_data, xr.Dataset):
        clean_mask = inst_data[clean_var].values != bad_val
    else:
        clean_mask = np.array(inst_data[clean_var]) != bad_val

    return clean_mask


class TestAuroralBounds(unittest.TestCase):
    """Tests for the AuroralBounds class."""

    def setUp(self):
        """Set up the test runs."""
        # Intialize the AuroralBounds attributes
        self.inst_data = {}
        self.time_var = 'time'
        self.glon_var = 'lon'
        self.glat_var = 'lat'
        self.intensity_var = 'intensity'
        self.alt = 110.0
        self.alb_kwargs = {
            'hemisphere': 1, 'transpose': False, 'opt_coords': None,
            'stime': None, 'etime': None, 'slice_kwargs': None,
            'clean_func': None, 'clean_kwargs': None,
            'slice_func':
            pyIntensityFeatures.instruments.satellites.get_auroral_slice}
        self.new_vals = {'inst_data': [], 'time_var': 0, 'glon_var': 1,
                         'glat_var': 2, 'intensity_var': 3, 'alt': 400.0,
                         'transpose': True, 'opt_coords': {'hi': 'test'},
                         'hemisphere': -1, 'stime': dt.datetime(1999, 2, 11),
                         'etime': dt.datetime(1999, 2, 11), 'slice_func': None,
                         'clean_func': clean_func}
        self.alb = None

        # Intialize the logging attributes
        self.msg = ""
        self.out = ""
        self.log_capture = StringIO()
        pyIntensityFeatures.logger.addHandler(logging.StreamHandler(
            self.log_capture))
        pyIntensityFeatures.logger.setLevel(logging.INFO)
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.inst_data, self.time_var, self.glon_var, self.glat_var
        del self.intensity_var, self.alt, self.alb_kwargs, self.msg, self.out
        del self.log_capture, self.new_vals
        return

    def set_inst_data(self, class_name='dict'):
        """Set the `inst_data` attribute and vars using the desired class.

        Parameters
        ----------
        class_name : str
            String specifying one of 'dict', 'list', 'array', 'pandas', or
            'xarray' (default='dict')

        """
        # Start by assuming a dict
        time_val = [dt.datetime(1999, 2, 11) + dt.timedelta(seconds=i)
                    for i in range(400)]
        shape2d = (len(time_val), 40)
        self.inst_data = {self.time_var: time_val,
                          self.glat_var: np.ones(shape=shape2d),
                          self.glon_var: np.ones(shape=shape2d),
                          self.intensity_var: np.full(
                              shape=shape2d, fill_value=500 * np.sin(
                                  np.linspace(0, np.pi, shape2d[1]))),
                          'clean_flag': np.ones(shape=shape2d)}

        self.inst_data[self.glat_var][0] = np.linspace(-6.0, 6.0, shape2d[1])
        self.inst_data[self.glon_var][0] = np.linspace(100.0, 300.0, shape2d[1])

        for i, lat in enumerate(self.inst_data[self.glat_var][0]):
            self.inst_data[self.glat_var][:, i] = lat + (
                90.0 - abs(lat)) * np.sin(np.linspace(0, 2.0 * np.pi,
                                                      shape2d[0]))
            self.inst_data[self.glon_var][:, i] = self.inst_data[
                self.glon_var][0, i] + (360.0 - self.inst_data[self.glon_var][
                    0, i]) * np.sin(np.linspace(0, np.pi, shape2d[0]))
            self.inst_data[self.intensity_var][:, i] *= np.sin(np.linspace(
                0, np.pi, shape2d[0]))

        if self.alb_kwargs['transpose']:
            for var in [self.glat_var, self.glon_var, self.intensity_var,
                        'clean_flag']:
                self.inst_data[var] = self.inst_data[var].transpose()

        # Update the non-dict class types
        if class_name.lower() == 'xarray':
            if self.alb_kwargs['transpose']:
                dat_dims = ["sweep_loc", self.time_var]
            else:
                dat_dims = [self.time_var, "sweep_loc"]

            # Cast an xarray Dataset from a reshaped dict
            self.inst_data = xr.Dataset.from_dict({
                key: {"dims": [self.time_var], "data": self.inst_data[key]}
                if key == self.time_var else
                {"dims": dat_dims,
                 "data": self.inst_data[key]} for key in self.inst_data})
        elif class_name.lower() in ['array', 'list']:
            # List and array require the same initial changes
            self.inst_data = [self.inst_data[self.time_var],
                              self.inst_data[self.glon_var],
                              self.inst_data[self.glat_var],
                              self.inst_data[self.intensity_var],
                              self.inst_data['clean_flag']]
            self.time_var = 0
            self.glon_var = 1
            self.glat_var = 2
            self.intensity_var = 3

            if class_name.lower() == 'array':
                # Data must be shaped the same
                self.inst_data[self.time_var] = np.full(
                    shape=(shape2d[1], shape2d[0]), fill_value=time_val)

                if not self.alb_kwargs['transpose']:
                    self.inst_data[self.time_var] = self.inst_data[
                        self.time_var].transpose()

                self.inst_data = np.array(self.inst_data)

        elif class_name.lower() == 'pandas':
            # Data must be shaped the same and be 1D
            self.inst_data[self.time_var] = np.full(
                shape=(shape2d[1], shape2d[0]), fill_value=time_val).transpose()

            for key in self.inst_data.keys():
                self.inst_data[key] = self.inst_data[key].flatten()

            # Cast a pandas DataFrame from the dict
            self.inst_data = pds.DataFrame(self.inst_data,
                                           index=self.inst_data[self.time_var])
        return

    def set_alb(self, set_attr=True):
        """Set the ALB test attribute for the tests.

        set_attr : bool
            If True, set the `alb` attribute, if False return the class object

        """
        if set_attr:
            self.alb = pyIntensityFeatures.AuroralBounds(
                self.inst_data, self.time_var, self.glon_var, self.glat_var,
                self.intensity_var, self.alt, **self.alb_kwargs)
            return
        else:
            return pyIntensityFeatures.AuroralBounds(
                self.inst_data, self.time_var, self.glon_var, self.glat_var,
                self.intensity_var, self.alt, **self.alb_kwargs)

    def eval_times(self, class_name="dict"):
        """Evaluate times based on the input.

        Parameters
        ----------
        class_name : str
            String specifying one of 'dict', 'list', 'array', 'pandas', or
            'xarray' (default='dict')

        """
        # Set the comparison time
        if len(self.inst_data) == 0:
            start = self.alb_kwargs['stime']
            end = self.alb_kwargs['etime']
        elif class_name == 'array':
            start = self.inst_data[self.time_var][0, 0]
            end = self.inst_data[self.time_var][-1, -1]
        elif class_name == 'xarray':
            start = pyIntensityFeatures.utils.coords.as_datetime(
                self.inst_data[self.time_var].values[0])
            end = pyIntensityFeatures.utils.coords.as_datetime(
                self.inst_data[self.time_var].values[-1])
        else:
            start = self.inst_data[self.time_var][0]
            end = self.inst_data[self.time_var][-1]

        # Evaluate the times
        self.assertTrue(self.alb.stime == start,
                        msg="{:} != {:}".format(self.alb.stime, start))
        self.assertTrue(self.alb.etime == end,
                        msg="{:} != {:}".format(self.alb.etime, end))
        return

    def eval_boundaries(self, min_mlat_base=59.0, mlat_inc=1.0,
                        mag_method="ALLOWTRACE", mlt_inc=0.5, un_threshold=1.25,
                        strict_fit=0, lt_out_bin=5.0, max_iqr=1.5):
        """Evaluate successful setting of boundaries.

        Parameters
        ----------
        min_mlat_base : float
            Base minimum co-latitude for intensity profiles. (default=59.0)
        method : str
            Method for converting between geographic and magnetic coordinates.
            (default='ALLOWTRACE')
        mlat_inc : float
            Magnetic latitude increment for gridding intensity. (default=1.0)
        mlt_inc : float
            Magnetic local time increment for gridding intensity. (default=0.5)
        un_threshold : float
            Maximum acceptable uncertainty value in degrees (default=1.25)
        strict_fit : int
            Enforce positive values for the x-offsets in quadratic-Gaussian fits
            using integer version of boolean (default=0)
        lt_out_bin : float
            Size of local time bin in hours over which outliers in the data
            will be identified (default=5.0)
        max_iqr : float
            Maximum multiplier for the interquartile range (IQR) used to
            identify outliers above or below the upper or lower quartile
            (default=1.5)

        """

        # Ensure un-run value is not present
        self.assertIsNotNone(self.alb.boundaries, msg="boundaries were not set")

        # Ensure no-boundary value is not present
        self.assertGreater(len(self.alb.boundaries), 0,
                           msg="no boundaries were found")

        # Evalute contents of the boundary object
        self.assertDictEqual({'min_mlat_base': min_mlat_base,
                              'mlat_inc': mlat_inc, 'mag_method': mag_method,
                              'mlt_inc': mlt_inc, 'un_threshold': un_threshold,
                              'strict_fit': strict_fit,
                              'lt_out_bin': lt_out_bin, 'max_iqr': max_iqr},
                             self.alb.boundaries.attrs,
                             msg="unexpected default attributes")
        self.assertDictEqual({'sweep_start': 1, 'mlt': 48, 'coeff': 6,
                              'lat': 31, 'sweep_end': 1},
                             dict(self.alb.boundaries.dims),
                             msg="unexpected default dimensions")
        self.assertListEqual(["sweep_start", "sweep_end", "mlt", "hemisphere",
                              "lat"],
                             [coord for coord
                              in self.alb.boundaries.coords.keys()],
                             msg="unexpected default coordinate")
        self.assertListEqual(["eq_bounds", "eq_uncert", "po_bounds",
                              "po_uncert", "eq_params", "po_params",
                              "mean_intensity", "std_intensity",
                              "num_intensity"],
                             [var for var
                              in self.alb.boundaries.data_vars.keys()],
                             msg="unexpected data variables")
        self.assertLessEqual(
            min_mlat_base, np.nanmax(self.alb.boundaries['eq_bounds'].values),
            msg="Bad equatorial boundary returned")
        self.assertLessEqual(
            min_mlat_base, np.nanmax(self.alb.boundaries['po_bounds'].values),
            msg="Bad polar boundary returned")
        self.assertLessEqual(
            0.0, np.nanmin(self.alb.boundaries['eq_uncert'].values),
            msg="Bad equatorial uncertainty returned")
        self.assertLessEqual(
            0.0, np.nanmin(self.alb.boundaries['po_uncert'].values),
            msg="Bad polar uncertainty returned")
        self.assertGreaterEqual(
            self.alb.inst_data[self.intensity_var].max(),
            np.nanmax(self.alb.boundaries['mean_intensity'].values),
            msg="Bad mean intensity returned")
        self.assertLessEqual(
            0, np.nanmin(self.alb.boundaries['num_intensity'].values),
            msg="Bad number of intensity points returned")
        return

    def test_init_empty_data(self):
        """Test initialization of the class with an empty data object."""
        self.msg = "".join(["unable to retrieve 'time' from `inst_data`, ",
                            "data may be empty"])
        for ctype, self.inst_data in [("dict", {}), ("array", np.array([])),
                                      ("list", []), ("pandas", pds.DataFrame()),
                                      ("xarray", xr.Dataset())]:
            with self.subTest(inst_data=self.inst_data):
                # Set the class object
                self.set_alb()

                # Evaluate the logging warnings
                self.out = self.log_capture.getvalue()
                self.assertRegex(self.out, self.msg)

                # Evaluate the times
                self.eval_times(ctype)

                # Test the boundaries
                self.assertIsNone(self.alb.boundaries)
        return

    def test_repr_string(self):
        """Test __repr__ method string."""
        # Set the class object without a slice function
        self.alb_kwargs['slice_func'] = None
        self.set_alb()

        # Get the representative output
        self.out = self.alb.__repr__()

        # Ensure the name and expected number of kwargs are present
        self.assertRegex(self.out, pyIntensityFeatures.AuroralBounds.__name__)
        self.assertEqual(
            len(self.out.split("=")), len(self.alb_kwargs.keys()) - 1,
            msg="unexpected number of kwargs AuroralBounds representation")

        # Test that a new AuroralBounds object can be created from repr
        # if there are not issues with the data or function reproduction
        self.out = eval(self.out)
        self.assertTrue(self.out == self.alb)
        return

    def test_print_string(self):
        """Test __str__ method string."""
        # Set the class object
        self.set_alb()

        # Get the representative output
        self.out = self.alb.__str__()

        # Ensure the expected headers are present
        self.msg = ["Auroral Boundary object", "Data Variables",
                    "Coordinate Attributes", "Instrument Functions"]
        for comp in self.msg:
            self.assertRegex(self.out, comp)

            # Remove this header from the output for future evaluations
            self.out = self.out.replace(comp, '')

        # After removing headers and new lines, ensure all args and kwargs
        # are displayed
        self.out = self.out.replace('=', '').replace('-', '').split('\n')

        while '' in self.out:
            self.out.pop(self.out.index(''))

        # There are 6 args and 2 of the input kwargs are not attributes
        self.assertEqual(len(self.out), len(self.alb_kwargs.keys()) + 4)
        return

    def test_equality(self):
        """Test class equality with empty data objects."""
        for self.inst_data in [{}, np.array([]), [], pds.DataFrame(),
                               xr.Dataset()]:
            with self.subTest(inst_data=self.inst_data):
                # Set the class object
                self.set_alb()

                # Set a comparison object
                self.out = self.set_alb(False)

                # Evaluate the equality
                self.assertEqual(self.out, self.alb)
        return

    def test_inequality_wrong_class(self):
        """Test class equality with different classes."""
        # Set the class object
        self.set_alb()

        # Evaluate the inequality
        self.assertFalse(self.alb == np.array([]))

        # Evalute the logging output
        self.msg = "wrong class"
        self.out = self.log_capture.getvalue()
        self.assertRegex(self.out, self.msg)
        return

    def test_inequality_extra_attributes(self):
        """Test class equality with extra attributes."""
        # Set the class object and comparison object
        self.set_alb()
        self.out = self.set_alb(False)
        setattr(self.out, "test_attr", "hi")

        # Evaluate the inequality
        self.assertFalse(self.alb == self.out)

        # Evalute the logging output
        self.msg = "object contains extra attribute: test_attr"
        self.out = self.log_capture.getvalue()
        self.assertRegex(self.out, self.msg)
        return

    def test_inequality_missing_attributes(self):
        """Test class equality with missing attributes."""
        # Set the class object and comparison object
        self.set_alb()
        self.out = self.set_alb(False)
        setattr(self.alb, "test_attr", "hi")

        # Evaluate the inequality
        self.assertFalse(self.alb == self.out)

        # Evalute the logging output
        self.msg = "object is missing attribute: test_attr"
        self.out = self.log_capture.getvalue()
        self.assertRegex(self.out, self.msg)
        return

    def test_inequality_attributes(self):
        """Test class equality with unequal standard attributes."""
        # Set the class object
        self.set_alb()

        # Cycle through the available attributes
        for attr in self.new_vals.keys():
            with self.subTest(attr=attr):
                # Set and update the comparison object
                if attr in self.alb_kwargs.keys():
                    orig_val = self.alb_kwargs[attr]
                    self.alb_kwargs[attr] = self.new_vals[attr]
                else:
                    orig_val = getattr(self, attr)
                    setattr(self, attr, self.new_vals[attr])

                self.out = self.set_alb(False)

                # Reset the original values
                if attr in self.alb_kwargs.keys():
                    self.alb_kwargs[attr] = orig_val
                else:
                    setattr(self, attr, orig_val)
                # Evaluate the inequality
                self.assertFalse(self.alb == self.out)

                # Evalute the logging output
                self.msg = "{:} differs".format(attr)
                self.out = self.log_capture.getvalue()
                self.assertRegex(self.out, self.msg)
        return

    def test_update_properties(self):
        """Test properties can be updated to different values."""
        self.set_alb()

        # Cycle through the properties
        for prop in ['alt', 'hemisphere', 'stime', 'etime']:
            with self.subTest(prop=prop):
                # Ensure the values do not equal the new values
                self.assertFalse(getattr(self.alb, prop) == self.new_vals[prop])

                # Update the property
                setattr(self.alb, prop, self.new_vals[prop])

                # Evaluate the new value
                self.assertTrue(getattr(self.alb, prop) == self.new_vals[prop])
        return

    def test_init_with_data(self):
        """Test times are set based on data when data is available."""
        # Class type order must end with 'list' and 'array'
        for ctype in ['dict', 'pandas', 'xarray', 'list', 'array']:
            with self.subTest(ctype=ctype):
                # Update the data
                self.set_inst_data(ctype)

                # Initialize the AuroralBounds object
                self.set_alb()

                # Test the times
                self.eval_times(ctype)

                # Test the boundaries
                self.assertIsNone(self.alb.boundaries)
        return

    def test_init_with_transposed_data(self):
        """Test times are set based on data when data is available."""
        self.alb_kwargs['transpose'] = True

        # Class type order must end with 'list' and 'array'
        for ctype in ['dict', 'pandas', 'xarray', 'list', 'array']:
            with self.subTest(ctype=ctype):
                # Update the data
                self.set_inst_data(ctype)

                # Initialize the AuroralBounds object
                self.set_alb()

                # Test the times
                self.eval_times(ctype)

                # Test that data is returned transposed internally
                for var in self.alb.transpose.keys():
                    self.out = self.alb._get_variable(var)
                    self.assertTupleEqual(self.out.transpose().shape,
                                          self.alb.inst_data[var].shape)

                # Test the boundaries
                self.assertIsNone(self.alb.boundaries)
        return

    def test_update_times_with_data(self):
        """Test times are set based on data when data is available."""
        # Class type order must end with 'list' and 'array'
        for ctype in ['dict', 'pandas', 'xarray', 'list', 'array']:
            with self.subTest(ctype=ctype):
                # Set the AuroralBounds object with data
                self.set_inst_data(ctype)
                self.set_alb()

                # Adjust the data
                if ctype in ['dict', 'list']:
                    self.inst_data[self.time_var] = self.inst_data[
                        self.time_var][:-10]
                elif ctype in 'array':
                    self.inst_data = self.inst_data[:, :-10]
                elif ctype == 'pandas':
                    self.inst_data = self.inst_data[:-10]
                else:
                    self.inst_data = xr.Dataset({
                        var: (self.inst_data[var].dims,
                              self.inst_data[var].values[:-10])
                        for var in [self.time_var, self.glat_var, self.glon_var,
                                    self.intensity_var, 'clean_flag']})
                self.alb.inst_data = self.inst_data

                # Update the times
                self.alb.update_times()

                # Test the times
                self.eval_times(ctype)

                # Test the boundaries
                self.assertIsNone(self.alb.boundaries)
        return

    def test_set_boundaries_empty_data(self):
        """Test initialization of the class with an empty data object."""
        self.msg = " data, cannot set boundaries"
        for ctype, self.inst_data in [("dict", {}), ("array", np.array([])),
                                      ("list", []), ("pandas", pds.DataFrame()),
                                      ("xarray", xr.Dataset())]:
            with self.subTest(inst_data=self.inst_data):
                # Set the class object
                self.set_alb()

                # Evaluate the times
                self.eval_times(ctype)

                # Set the boundaries
                self.alb.set_boundaries()

                # Test the logging message
                self.out = self.log_capture.getvalue()
                self.assertRegex(self.out, self.msg)

                # Test the boundaries
                self.assertIsNone(self.alb.boundaries)
        return

    def test_set_boundaries_with_data(self):
        """Test setting boundaries from data."""
        # Set the messages that should be raised for the test data
        self.msg = ["Gaussian peak is outside of the intensity profile",
                    "Auroral slice at ", "with data", "without data",
                    "Removing boundary outliers from",
                    "The polar/equatorward boundary locations are mixed up"]

        # Set the uncertainty threshold to be large for testing
        uncert = 12.0

        # Class type order must end with 'list'. Not testing array as an
        # appropriate slicing function is not available.
        for ctype in ['dict', 'pandas', 'xarray', 'list']:
            with self.subTest(ctype=ctype):
                # Update the data
                self.set_inst_data(ctype)

                # Initialize the AuroralBounds object
                self.set_alb()

                # Test the times
                self.eval_times(ctype)

                # Set the boundaries
                self.alb.set_boundaries(un_threshold=uncert)

                # Evaluate the logging messages
                self.out = self.log_capture.getvalue()
                for mes in self.msg:
                    self.assertRegex(self.out, mes)

                # Evaluate the boundaries
                self.eval_boundaries(un_threshold=uncert)
        return

    def test_set_boundaries_with_long_end_time(self):
        """Test setting boundaries with an end time beyond the data."""
        # Set the messages that should be raised for the test data
        self.msg = ["Gaussian peak is outside of the intensity profile",
                    "Auroral slice at ", "with data", "without data",
                    "Removing boundary outliers from",
                    "The polar/equatorward boundary locations are mixed up"]

        # Set the uncertainty threshold to be large for testing
        uncert = 12.0

        # Class type order must end with 'list'. Not testing array as an
        # appropriate slicing function is not available.
        for ctype in ['dict', 'pandas', 'xarray', 'list']:
            with self.subTest(ctype=ctype):
                # Update the data
                self.set_inst_data(ctype)

                # Initialize the AuroralBounds object
                self.set_alb()

                # Test the times
                self.eval_times(ctype)

                # Update the end time
                self.alb.etime += dt.timedelta(days=1)

                # Set the boundaries
                self.alb.set_boundaries(un_threshold=uncert)

                # Evaluate the logging messages
                self.out = self.log_capture.getvalue()
                for mes in self.msg:
                    self.assertRegex(self.out, mes)

                # Evaluate the boundaries
                self.eval_boundaries(un_threshold=uncert)
        return

    def test_set_boundaries_with_data_no_slice(self):
        """Test setting boundaries from data without slicing."""
        # Set the messages that should be raised for the test data
        self.msg = ["Gaussian peak is outside of the intensity profile",
                    "Auroral slice at ", "with data",
                    "Removing boundary outliers from",
                    "The polar/equatorward boundary locations are mixed up"]
        self.alb_kwargs['slice_func'] = self.new_vals['slice_func']
        uncert = 12.0

        # Class type order must end with 'list'. Not testing array as an
        # appropriate slicing function is not available.
        for ctype in ['dict', 'pandas', 'xarray', 'list']:
            with self.subTest(ctype=ctype):
                # Update the data
                self.set_inst_data(ctype)

                # Initialize the AuroralBounds object
                self.set_alb()

                # Test the times
                self.eval_times(ctype)

                # Set the boundaries
                self.alb.set_boundaries(un_threshold=uncert)

                # Evaluate the logging messages
                self.out = self.log_capture.getvalue()
                for mes in self.msg:
                    self.assertRegex(self.out, mes)

                # Evaluate the boundaries
                self.eval_boundaries(un_threshold=uncert)
        return

    def test_set_boundaries_with_mask(self):
        """Test setting boundaries from data while masking all data."""
        # Set the messages that should be raised for the test data
        self.msg = ["Gaussian peak is outside of the intensity profile",
                    "Auroral slice at ", "with data", "without data",
                    "Removing boundary outliers from",
                    "The polar/equatorward boundary locations are mixed up"]
        self.alb_kwargs['clean_func'] = self.new_vals['clean_func']

        # Not testing 'list' or 'array' as the clean function is not set up
        # to handle integer inputs
        for ctype in ['dict', 'pandas', 'xarray']:
            with self.subTest(ctype=ctype):
                # Update the data
                self.set_inst_data(ctype)

                # Initialize the AuroralBounds object
                self.set_alb()

                # Test the times
                self.eval_times(ctype)

                # Set the boundaries
                self.alb.set_boundaries()

                # Test the boundaries
                self.assertIsNotNone(self.alb.boundaries)
                self.assertEqual(0, len(self.alb.boundaries),
                                 msg="found boundaries without data")
        return
