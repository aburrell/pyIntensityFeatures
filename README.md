[![Documentation Status](https://readthedocs.org/projects/pyIntensityFeatures/badge/?version=latest)](http://pyIntensityFeatures.readthedocs.io/en/latest/?badge=latest)
[DOI]
[PyPI version]
[![Test Status](https://github.com/aburrell/pyIntensityFeatures/actions/workflows/main.yml/badge.svg)](https://github.com/aburrell/pyIntensityFeatures/actions/workflows/main.yml)
[![Coverage Status](https://coveralls.io/repos/github/aburrell/pyIntensityFeatures/badge.svg?branch=main)](https://coveralls.io/github/aburrell/pyIntensityFeatures?branch=main)

<h1> <img width="128" height="128" src="https://raw.githubusercontent.com/aburrell/pyIntensityFeatures/refs/heads/main/docs/figures/pyIntensityFeatures_logo.png" alt="Snakes marking the boundaries of the auroral oval" title="pyIntensityFeatures logo" style="float:left;">
Overview </h1>

pyIntensityFeatures is a Python module that identifies features in imager
intensity data.  It currently supports the identification of the Auroral
Luminosity Boundaries (ALBs).  This package is designed to be data source
agnostic and uses a fitting method based on Longden et al. (2009) to identify
the poleward and equatorward ALBs. Current support focuses on satellite-based
imagers.

# Dependencies

The required dependecies encompass:
  * aacgmv2
  * numpy
  * pandas
  * scipy
  * xarray

Optional dependencies include:
  * apexpy

Testing is performed using the python module, unittest.

# Installation

Installation is now available through pypi

```
    $ pip install pyIntensityFeatures
```

You may also checkout the repository and install it yourself:

```
    $ git clone git://github.com/aburrell/pyIntensityFeatures.git;
```

Change directories into the repository folder and build the wheel.

```
    $ cd pyIntensityFeatures/
    $ python -m build .
    $ pip install .
```

To run the unit tests,

```
    $ python -m unittest discover
```

# Example

This example uses pysatNASA to obtain GUVI sdr-imaging data. You can use this
as a template to create the necessary supporting functions to identify ALBs
in your own data set. In iPython, run:

```
import datetime as dt
import pyIntensityFeatures
import pysat
from pysatNASA.instruments import timed_guvi
```

Obtain the data from which you want to get boundaries.

```
stime = dt.datetime(2005, 1, 1)
guvi = pysat.Instrument(inst_module=timed_guvi, tag='sdr-imaging',
                        inst_id='high_res')
guvi.download(start=stime)
```

We need to create a cleaning function to select only good intensity data using
the supplied DQI flag. Because GUVI images the ionosphere at several
wavelenghts and the satellite slice-detection function expects a 2D array for
intensity, we also need to create a function that will add a single-channel
intensity variable to the pysat GUVI Instrument.

```
def create_channel_intensity(guvi, channel='LBHlong'):
    """Create a new variable for the single-channel auroral intensity.

    Parameters
    ----------
    guvi : pysat.Instrument
        pysat Instrument for the GUVI data.
    channel : str
        Desired channel name. (defualt='LBHlong')

    """
    int_var = 'DISK_RECTIFIED_INTENSITY_DAY_AURORAL'
    chan_var = 'nchan'

    # Get the channel index
    ichan = list(guvi[chan_var].values).index(channel)

    # Create the new intensity variable
    new_var = int_var.replace('DISK_RECTIFIED', channel)
    guvi[new_var] = guvi[int_var][:, :, ichan]
    return


def clean_guvi(guvi, int_var):
    """Get a mask for clean GUVI auroral intensity.

    Parameters
    ----------
    guvi : pysat.Instrument
        pysat Instrument for the GUVI data.
    int_var : str
        Intensity variable name

    Returns
    -------
    clean_mask : array-like
        2D array with a mask denoting the good data by the DQI auroral flag.

    """
    flag_var = 'DQI_DAY_AURORAL'

    # Get the clean mask from the flag
    clean_mask = np.full(shape=guvi[int_var].values.shape, fill_value=False)
    clean_mask = guvi[flag_var].values <= 0

    # The pyIntensityFeatures.instruments.satellites function requires time as
    # the first dimension
    return clean_mask.transpose()
```

pysat allows methods that update or alter the instrument to be run when loading
the data.  Attach the `create_channel_intensity` function and load the
desired day.

```
guvi.custom_attach(create_channel_intensity, kwargs={'channel': 'LBHlong'})
guvi.load(date=stime)
```

Then initialise an AuroralBounds class object. The `transpose` kwarg is set
to `True` to reshape the GUVI longitude, latitude, and intensity data such that
the time dimension is first.

```
ckwargs = {'int_var': 'LBHlong_INTENSITY_DAY_AURORAL'}
guvi_alb = pyIntensityFeatures.AuroralBounds(guvi, 'time_auroral',
                                            'PIERCEPOINT_DAY_LONGITUDE_AURORAL',
                                            'PIERCEPOINT_DAY_LATITUDE_AURORAL',
                                            'LBHlong_INTENSITY_DAY_AURORAL',
                                            110.0, hemisphere=1,
                                            opt_coords={'channel': 'LBHlong'},
                                            transpose=True,
                                            clean_func=clean_guvi,
                                            clean_kwargs=ckwargs)
print(guvi_alb)
```

The output should be as follows, with the start and end times set to reflect
the times at which data was provided:

```
Auroral Boundary object
=======================
Instrument Data: pysat Instrument object
-----------------------
Platform: 'timed'
Name: 'guvi'
Tag: 'sdr-imaging'
Instrument id: 'high_res'

Data Processing
---------------
Cleaning Level: 'clean'
Data Padding: None
Custom Functions: 1 applied
    0: <function create_channel_intensity>
     : Kwargs={'channel': 'LBHlong'}

Local File Statistics
---------------------
Number of files: 711
Date Range: 01 January 2005 --- 01 March 2005

Loaded Data Statistics
----------------------
Date: 01 January 2005
DOY: 001
Time range: 01 January 2005 00:12:38 --- 01 January 2005 23:59:59
Number of Times: 23989
Number of variables: 70

Variable Names:
ORBIT_DAY  LATITUDE_DAY  LONGITUDE_DAY
              ...
nCross  nCrossDayAur  LBHlong_INTENSITY_DAY_AURORAL 

pysat Meta object
-----------------
Tracking 8 metadata values
Metadata for 90 standard variables
Metadata for 40 global attributes


Data Variables
--------------
Time: time_auroral
Geo Lon: PIERCEPOINT_DAY_LONGITUDE_AURORAL
Geo Lat: PIERCEPOINT_DAY_LATITUDE_AURORAL
Intensity: LBHlong_INTENSITY_DAY_AURORAL
Transpose: {'PIERCEPOINT_DAY_LONGITUDE_AURORAL': True,
            'PIERCEPOINT_DAY_LATITUDE_AURORAL': True,
            'LBHlong_INTENSITY_DAY_AURORAL': True}

Coordinate Attributes
---------------------
Hemisphere: 1
Altitude: 110.00 km
Optional coords: {'channel': 'LBHlong', 'hemisphere': 1}
Start time: 2005-01-01T00:12:38.575362000
End time: 2005-01-01T23:59:59.722748000

Instrument Functions
--------------------
Slicing: functools.partial(<function get_auroral_slice>, transpose=True)
Cleaning: functools.partial(<function clean_guvi>,
                            int_var='LBHlong_INTENSITY_DAY_AURORAL')
```

Now, get the ALBs for the loaded data using the method defaults. This will
update the `boundaries` attribute from `None` to a filled or empty xarray
Dataset. We will update the end time to be shorter than the amount of loaded
data to limit the processing time.

```
guvi_alb.etime = dt.datetime(2005, 1, 1, 0, 25)
guvi_alb.set_boundaries()
print(guvi_alb.boundaries)
```

This will yield one slice of intensity data in the Northern hemisphere with
ALBs:

```
<xarray.Dataset>
Dimensions:         (sweep_start: 1, mlt: 48, coeff: 12, lat: 31, sweep_end: 1)
Coordinates:
  * sweep_start     (sweep_start) datetime64[ns] 2005-01-01T00:24:35.572131
  * sweep_end       (sweep_end) datetime64[ns] 2005-01-01T00:49:11.295639
  * mlt             (mlt) float64 0.25 0.75 1.25 1.75 ... 22.75 23.25 23.75
    channel         <U7 'LBHlong'
    hemisphere      int64 1
  * lat             (lat) float64 59.5 60.5 61.5 62.5 ... 86.5 87.5 88.5 89.5
Dimensions without coordinates: coeff
Data variables:
    eq_bounds       (sweep_start, mlt) float64 63.5 64.03 nan ... nan 62.98 nan
    eq_uncert       (sweep_start, mlt) float64 0.741 0.8169 nan ... 2.219 nan
    po_bounds       (sweep_start, mlt) float64 73.96 76.12 nan ... nan 71.24 nan
    po_uncert       (sweep_start, mlt) float64 0.97 2.213 nan ... nan 0.6226 nan
    eq_params       (sweep_start, mlt, coeff) float64 67.87 -2.033 ... nan nan
    po_params       (sweep_start, mlt, coeff) float64 67.87 -2.033 ... nan nan
    mean_intensity  (sweep_start, lat, mlt) float64 0.2494 0.8067 ... nan nan
    std_intensity   (sweep_start, lat, mlt) float64 10.72 9.089 ... nan nan
    num_intensity   (sweep_start, lat, mlt) float64 10.72 9.089 ... nan nan
Attributes:
    min_mlat_base:  59.0
    mag_method:     ALLOWTRACE
    mlat_inc:       1.0
    mlt_inc:        0.5
    un_threshold:   1.25
    lt_out_bin:     5.0
    max_iqr:        1.5
```

You can then save the data as a NetCDF using the built-in xarray Dataset method.
All the kwargs from the `set_boundaries` method have been added as attributes to
ensure reproducibility of the identified boundaries.
