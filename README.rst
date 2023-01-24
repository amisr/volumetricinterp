volumetricinterp
================

Volumetric interpolation of scalar values in an AMISR FoV

This code interpolates density measurements from a processed AMISR data file in two steps.  First, a command line program calculates interpolation parameters for and entire datafile and saves these parameters to an output hdf5 file.  Then this file can be loaded to an interpolator, which can be queried at any time or location that the original file covers.

Install
-------

1. Clone the `volumetricinterp` github repository:
  $ git clone https://github.com/amisr/volumetricinterp.git
2. Enter the repository:
  $ cd volumetricinterp
3. Use pip to install in a python 3 environment:
  $ pip install .

Create the Interpolation Parameter File
---------------------------------------
1. Create a config file based on `example_config.ini`.  At minimum, you should change `FILENAME` and `OUTPUTFILENAME` parameters to specify location/names of the AMISR file to fit and the output coefficient file.
2. Run volumetricinterp:
  $ volumetricinterp config.ini

Estimate interpolated values from coefficient file
--------------------------------------------------
1. Import the CalcInterp from the interp4model module:
  from volumetricinterp.interp4model import CalcInterp
2. Initialize an Estimate object with the coefficient file name:
  ci = CalcInterp('volumetric_interp_output.hdf5')
3. Use functions in the CalcInterp class to calculate density values at given times/locations:
  xgrid, ygrid, zgrid, dens_grid = ci.grid_enu(time, x, y, z)
