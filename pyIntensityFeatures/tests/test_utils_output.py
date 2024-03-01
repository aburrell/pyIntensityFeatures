#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.output`."""

import datetime as dt
import numpy as np
import unittest

from pyIntensityFeatures.utils import output


class TestBoundaryDictFuncs(unittest.TestCase):
    """Tests for functions that create and alter the boundary dicts."""

    def setUp(self):
        """Set up the test runs."""
        self.lat_dim = 'mlat'
        self.attr_dict = None
        self.opt_coords = None
        self.coord_dict = None
        self.data_dict = None
        self.new_data = None
        self.dataset = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.opt_coords, self.coord_dict, self.data_dict, self.new_data
        del self.lat_dim, self.attr_dict, self.dataset
        return

    def eval_coord_dict(self):
        """Evaluate the coordinate dict."""
        coord_keys = ['sweep_start', 'sweep_end', 'mlt']

        if self.opt_coords is not None:
            coord_keys.extend(list(self.opt_coords.keys()))

        # Evaluate the presence of the desired keys and their data
        for key in coord_keys:
            self.assertIn(key, self.coord_dict.keys(),
                          msg="missing coordinate key: {:}".format(key))

            if self.opt_coords is not None:
                if key in self.opt_coords.keys():
                    self.assertEqual(self.coord_dict[key], self.opt_coords[key])
            else:
                if self.new_data is None:
                    if key == 'mlt':
                        self.assertIsNone(self.coord_dict[key])
                    else:
                        self.assertListEqual(self.coord_dict[key], [])
                else:
                    if key.find('sweep') == 0:
                        # New times should be added to the end
                        self.assertEqual(self.new_data[key],
                                         self.coord_dict[key][-1])
                    else:
                        # Other coordinates should not change
                        msg = "Unexpected coordinate data in {:}".format(key)
                        self.assertListEqual(list(self.new_data[key]),
                                             list(self.coord_dict[key]),
                                             msg=msg)
        return

    def eval_data_dict(self):
        """Evaluate the data dict."""
        data_keys = ['mlat', 'eq_bounds', 'eq_uncert', 'po_bounds',
                     'po_uncert', 'eq_params', 'po_params', 'mean_intensity',
                     'std_intensity', 'num_intensity']
        data_dims = ['sweep_start', 'lat', 'mlt', 'coeff']

        # Evaluate the presence of the desired keys and their data
        for key in data_keys:
            self.assertIn(key, self.data_dict.keys(),
                          msg="missing data key: {:}".format(key))

            if self.new_data is None:
                # Test the data dimensions
                self.assertTrue(isinstance(self.data_dict[key][0], tuple))
                for dim in self.data_dict[key][0]:
                    self.assertIn(dim, data_dims)

                # Test the data values
                self.assertListEqual(self.data_dict[key][1], [])
            else:
                # Test the data dimensions
                self.assertTrue(isinstance(self.data_dict[key][0], tuple))
                for dim in self.data_dict[key][0]:
                    self.assertIn(dim, data_dims)

                # Test the data values
                self.assertEqual(len(self.new_data[key]),
                                 len(self.data_dict[key][1][-1]),
                                 msg="Updated {:} is too short: {:}".format(
                                     key, self.data_dict[key]))
                self.assertTrue(np.all(self.data_dict[key][1][-1]
                                       == self.new_data[key]),
                                msg="{:} data arrays not equal".format(key))
        return

    def eval_dataset(self):
        """Evaluate a Dataset assigned to `new_data`."""
        # Evaluate global attributes
        if self.attr_dict is None:
            self.assertDictEqual(self.dataset.attrs, {})
        else:
            # Test access through attribute dict
            self.assertDictEqual(self.dataset.attrs, self.attr_dict)

            # Test access as attributes
            for attr in self.attr_dict:
                self.assertTrue(hasattr(self.dataset, attr))

        # Evalute the coordinates
        for coord in self.coord_dict.keys():
            self.assertIn(coord, self.dataset.coords)

            if coord in self.dataset.dims:
                if coord.find('sweep') < 0:
                    self.assertTrue(
                        np.all(self.dataset.coords[coord].values
                               == self.coord_dict[coord]),
                        msg="".join(["Bad coordinate values for ", coord,
                                     ": {:} != {:}".format(
                                         self.dataset.coords[coord].values,
                                         self.coord_dict[coord])]))
            else:
                self.assertEqual(self.dataset.coords[coord],
                                 self.coord_dict[coord],
                                 msg="unequal coordinate values for {:}".format(
                                     coord))

        # Evaluate the data
        for dvar in self.data_dict.keys():
            self.assertIn(dvar, self.dataset.data_vars)
            self.assertTrue(np.all(self.dataset[dvar].values
                                   == self.data_dict[dvar][1][0]))

        return

    def update_new_data(self, inc=0, hemi=1, mlt_inc=0.5):
        """Create data for the `new_data` test attribute.

        Parameters
        ----------
        inc : int
            Number by which values will be incremented.
        hemi : int
            1 for Northern and -1 for Southern hemisphere
        mlt_inc : float
            Increment for the MLT bins (default=0.5)

        """
        mlt = np.arange(0, 24, mlt_inc)
        mlat = hemi * np.arange(59.0, 90.0, 1.0)
        params = [1.0, 0.1, 0.01, 100.0, hemi * 70.0, 5.0]
        self.new_data = {'sweep_start': dt.datetime(1999, 2, 11, inc),
                         'sweep_end': dt.datetime(1999, 2, 11, inc, 50),
                         'mlt': mlt, 'mlat': mlat,
                         'eq_bounds': np.full(shape=mlt.shape,
                                              fill_value=60.0 + inc) * hemi,
                         'eq_uncert': np.ones(shape=mlt.shape),
                         'eq_params': np.full(shape=(mlt.shape[0], len(params)),
                                              fill_value=params),
                         'po_bounds': np.full(shape=mlt.shape,
                                              fill_value=80.0 + inc) * hemi,
                         'po_uncert': np.ones(shape=mlt.shape),
                         'po_params': np.full(shape=(mlt.shape[0], len(params)),
                                              fill_value=params),
                         'mean_intensity': np.ones(shape=(mlat.shape[0],
                                                          mlt.shape[0])),
                         'std_intensity': np.zeros(shape=(mlat.shape[0],
                                                          mlt.shape[0])),
                         'num_intensity': np.ones(shape=(mlat.shape[0],
                                                         mlt.shape[0]))}
        return

    def test_init_boundary_dicts(self):
        """Test success for initalizing the boundary dicts."""
        # Update with and without optional coordinates
        for self.opt_coords in [None, {'test': 'test_value'}]:
            with self.subTest(opt_coords=self.opt_coords):
                # Initalize the dicts
                self.coord_dict, self.data_dict = output.init_boundary_dicts(
                    opt_coords=self.opt_coords)

                # Test the output
                self.eval_coord_dict()
                self.eval_data_dict()
        return

    def test_update_boundary_dicts(self):
        """Test success for updating the boundary dicts."""
        # Update with and without any data
        for num_update in [0, 1, 2]:
            with self.subTest(num_update=num_update):
                # Initalize the dicts
                self.coord_dict, self.data_dict = output.init_boundary_dicts(
                    opt_coords=self.opt_coords)

                for inc in range(num_update):
                    self.update_new_data(inc=inc)

                    # Update the dicts
                    output.update_boundary_dicts(self.new_data, self.coord_dict,
                                                 self.data_dict)

                # Test the updated dicts
                self.eval_coord_dict()
                self.eval_data_dict()
        return

    def test_update_boundary_dicts_change_mlt_inc(self):
        """Test raises ValueError for updating with different MLT bin inc."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts(
            opt_coords=self.opt_coords)

        # Add data
        self.update_new_data()
        output.update_boundary_dicts(self.new_data, self.coord_dict,
                                     self.data_dict)

        # Change the MLT bins in the new data
        self.update_new_data(mlt_inc=5.0)

        # Update the dicts and evaluate the error
        self.assertRaisesRegex(ValueError,
                               'change in magnetic local time bin increment',
                               output.update_boundary_dicts,
                               *[self.new_data, self.coord_dict,
                                 self.data_dict])
        return

    def test_update_boundary_dicts_change_mlt_vals(self):
        """Test raises ValueError for updating with different MLT bin vals."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts(
            opt_coords=self.opt_coords)

        # Add data
        self.update_new_data()
        output.update_boundary_dicts(self.new_data, self.coord_dict,
                                     self.data_dict)

        # Change the MLT bins in the new data
        self.update_new_data()
        self.new_data['mlt'][0] += 0.01

        # Update the dicts and evaluate the error
        self.assertRaisesRegex(ValueError,
                               'change in magnetic local time bin values',
                               output.update_boundary_dicts,
                               *[self.new_data, self.coord_dict,
                                 self.data_dict])
        return

    def test_reshape_lat_coeff_data_no_lat_bins(self):
        """Test ValueError raised with no lat bins when reshaping data."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts()

        # Run with the empty dicts and evaluate the error raised
        self.assertRaisesRegex(ValueError, "no latitude data",
                               output.reshape_lat_coeff_data,
                               *[self.coord_dict, self.data_dict, 0])
        return

    def test_reshape_lat_coeff_data_bad_shaped_lat_bins(self):
        """Test ValueError with badly formed lat bins when reshaping data."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts()

        # Update the dicts and reset the magnetic latitude bins
        self.update_new_data()
        output.update_boundary_dicts(self.new_data, self.coord_dict,
                                     self.data_dict)
        self.data_dict['mlat'][1][0][0] -= 0.1

        # Run and evaluate the error raised
        self.assertRaisesRegex(ValueError, "badly shaped latitude bins",
                               output.reshape_lat_coeff_data,
                               *[self.coord_dict, self.data_dict, 6])
        return

    def test_reshape_lat_coeff_data_inconsistent_lat_bins(self):
        """Test ValueError with inconsistent lat bins when reshaping data."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts()

        # Update the dicts and reset the magnetic latitude bins
        self.update_new_data(inc=0)
        output.update_boundary_dicts(self.new_data, self.coord_dict,
                                     self.data_dict)
        self.update_new_data(inc=1)
        output.update_boundary_dicts(self.new_data, self.coord_dict,
                                     self.data_dict)
        self.data_dict['mlat'][1][-1] = np.arange(59.0, 90.0, 2.0)

        # Run and evaluate the error raised
        self.assertRaisesRegex(ValueError, "inconsistent latitude increments",
                               output.reshape_lat_coeff_data,
                               *[self.coord_dict, self.data_dict, 6])
        return

    def test_reshape_lat_coeff_data_bad_coeff_order(self):
        """Test ValueError with badly ordered coeff data when reshaping data."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts()

        # Update the dicts and reset the coefficient data
        self.update_new_data()
        output.update_boundary_dicts(self.new_data, self.coord_dict,
                                     self.data_dict)
        self.data_dict['po_params'] = list(self.data_dict['po_params'])
        self.data_dict['po_params'][1][0] = self.data_dict['po_params'][1][
            0].transpose()
        self.data_dict['po_params'][0] = (self.data_dict['po_params'][0][0],
                                          self.data_dict['po_params'][0][2],
                                          self.data_dict['po_params'][0][1])
        self.data_dict['po_params'] = tuple(self.data_dict['po_params'])

        # Run and evaluate the error raised
        self.assertRaisesRegex(ValueError,
                               "unexpected dimension order for coefficients",
                               output.reshape_lat_coeff_data,
                               *[self.coord_dict, self.data_dict, 6])
        return

    def test_reshape_lat_coeff_data_bad_coeff_dims(self):
        """Test ValueError with badly shaped coeff dims when reshaping data."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts()

        # Update the dicts and reset the coefficient data
        self.update_new_data()
        output.update_boundary_dicts(self.new_data, self.coord_dict,
                                     self.data_dict)
        self.data_dict['po_params'] = list(self.data_dict['po_params'])
        self.data_dict['po_params'][0] = (self.data_dict['po_params'][0][0],
                                          self.data_dict['po_params'][0][2])
        self.data_dict['po_params'] = tuple(self.data_dict['po_params'])

        # Run and evaluate the error raised
        self.assertRaisesRegex(ValueError,
                               "unexpected dimension order for coefficients",
                               output.reshape_lat_coeff_data,
                               *[self.coord_dict, self.data_dict, 6])
        return

    def test_reshape_lat_coeff_data(self):
        """Test success when reshaping data."""

        for hemi in [-1, 1]:
            # Initalize the dicts
            self.coord_dict, self.data_dict = output.init_boundary_dicts()

            with self.subTest(hemisphere=hemi):
                # Update the dicts
                self.update_new_data(inc=0, hemi=hemi)
                output.update_boundary_dicts(self.new_data, self.coord_dict,
                                             self.data_dict)
                self.update_new_data(inc=1, hemi=hemi)
                output.update_boundary_dicts(self.new_data, self.coord_dict,
                                             self.data_dict)

                # Run to reshape the output
                mlat_bins, self.data_dict = output.reshape_lat_coeff_data(
                    self.coord_dict, self.data_dict, 6)

                # Evaluate the latitude bins
                self.assertTrue(np.all(mlat_bins == self.new_data['mlat']),
                                msg="unexpected magnetic latitude bins")

                # Evaluate the data output
                for key in self.data_dict.keys():
                    self.assertTrue(isinstance(self.data_dict[key], tuple))

                    dims = list(self.data_dict[key][0])
                    self.assertEqual(
                        len(dims), len(np.array(
                            self.data_dict[key][1]).shape),
                        msg="Unexpected shape for {:}".format(key))
        return

    def test_convert_boundary_dict_empty(self):
        """Test creation of an empty Dataset without full coordinates."""
        # Initalize the dicts
        self.coord_dict, self.data_dict = output.init_boundary_dicts()

        # Create a dataset
        self.dataset = output.convert_boundary_dict(self.coord_dict,
                                                    self.data_dict, 0)

        # Evalute the empty dataset
        self.assertEqual(len(self.dataset.dims), 0)
        self.assertEqual(len(self.dataset.data_vars), 0)
        return

    def test_convert_boundary_dict(self):
        """Test creation of a Dataset from the boundary dicts."""
        # Cycle through the coordinate options
        for self.opt_coords in [None, {'test': 'test_value'}]:
            with self.subTest(opt_coords=self.opt_coords):

                # Cycle through the attribute options
                for self.attr_dict in [None, {'test': 'test_attr'}]:
                    with self.subTest(attrs=self.attr_dict):
                        # Initalize the dicts
                        (self.coord_dict,
                         self.data_dict) = output.init_boundary_dicts(
                             opt_coords=self.opt_coords, lat_dim=self.lat_dim)

                        # Update the dicts
                        self.update_new_data()
                        output.update_boundary_dicts(self.new_data,
                                                     self.coord_dict,
                                                     self.data_dict)

                        # Create a dataset
                        self.dataset = output.convert_boundary_dict(
                            self.coord_dict, self.data_dict, 6,
                            lat_dim=self.lat_dim, attr_dict=self.attr_dict)

                        # Evalute the dataset
                        self.eval_dataset()
        return
