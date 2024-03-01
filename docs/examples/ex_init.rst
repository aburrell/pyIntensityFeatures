.. _exinit:

Getting Started
===============

:py:mod:`pyIntensityFeatures` is centred around class objects that focus on
specific intensity phenomena and their important features.  Currently, there
is only one such class, which identifies Auroral Luminosity Boundaries (ALBs),
:py:class:`~pyIntensityFeatures._main.AuroralBounds`.


Initialise an AuroralBounds object
----------------------------------
Start a python or iPython session, and begin by importing :py:mod:`datetime`
and :py:mod:`pyIntensityFeatures`.

::


   import datetime as dt
   import pyIntensityFeatures
  
Next, initialise an :py:class:`~pyIntensityFeatures._main.AuroralBounds` object.
This sets up an object without any data.  Note that this class has several
arguements that must be provided to ensure unrealistic assumptions are not
made when performing the data processing.

::

   
   alb = pyIntensityFeatures.AuroralBounds({}, "time", "glat", "glon",
                                           "intensity", 110.0)
   print(alb)

   Auroral Boundary object
   =======================
   Instrument Data: {}

   Data Variables
   --------------
   Time: time
   Geo Lon: glat
   Geo Lat: glon
   Intensity: intensity
   Transpose: {'glat': False, 'glon': False, 'intensity': False}

   Coordinate Attributes
   ---------------------
   Hemisphere: 1
   Altitude: 110.00 km
   Optional coords: {'hemisphere': 1}
   Start time: None
   End time: None

   Instrument Functions
   --------------------
   Slicing: functools.partial(<function get_auroral_slice at 0x1391d1550>)
   Cleaning: None

The class does make assumptions about other things, though.  Note that it
assumes you are looking for ALBs in the northern hemisphere (1 stands for
northern, as it is a latitude multiplier that produces northern locations and
-1 stands for southern for the same reason). It also assumes that the data
(if it were present) has the time dimension as the first dimension (as
denoted by the :py:attr:`transpose` values).  The starting and ending times
could not be set, since there was no data.  It also supplies the satellite
slicing function,
:py:func:`~pyIntensityFeatures.instruments.satellites.get_auroral_slice`, which
will pick out auroral-region image data from the desired hemisphere.

Any of these attributes can be updated after object initialization. For example,
we can chose the southern hemisphere instead.

::

   alb.hemisphere = -1
   print(alb.hemisphere, alb.opt_coords)

   -1 {'hemisphere': -1}

Note that this updates the hemisphere value across all class attributes.

Time can be set to any value you like, or updated based on new data using
the :py:meth:`~pyIntensityFeatures._main.AuroralBounds.update_times` method.
The times are not tied together outside of this method, and updating the data
in :py:attr:`~pyIntensityFeatures._main.AuroralBounds.inst_data` will not change
the time attributes. This prevents the class from forcing you to calculate
boundaries at times you don't want to do so.

::

   alb.stime = dt.datetime(1999, 2, 11)
   print(alb.stime, "to", alb.etime)

   1999-02-11 00:00:00 to None
