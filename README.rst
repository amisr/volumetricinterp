volumetricinterp
================

Volumetric interpolation of scalar values in an AMISR FoV

This code interpolates density measurements from an AMISR FoV by first performing a least-squares fit to the measured density at a particular time to find the coefficients for a 3D linear analytic model and then evaluating that function at any arbitrary point within the FoV to estimate the density at that point.  The coefficients for the fitted model are saved to a hdf5 file, which can be used to initialize an `Estimate` object which calculates the density at arbitrary locations.  A validation option is provided which may be useful for experimenting with different model parameters.

Install
-------

1. Clone the `volumetricinterp` github repository:
  $ git clone https://github.com/amisr/volumetricinterp.git
2. Enter the repository:
  $ cd volumetricinterp
3. Use pip to install in a python 3 environment:
  $ pip install .

Perform fitting on an AMISR datafile
------------------------------------
1. Create a config file based on `example_config.ini`.  At minimum, you should change `FILENAME` and `OUTPUTFILENAME` parameters to specify location/names of the AMISR file to fit and the output coefficient file.
2. Run volumetricinterp:
  $ volumetricinterp config.ini
3. To run the validation script, use the `--validate` flag:
  $ volumetricinterp --validate config.ini

volumetricinterp --spectrum config.ini

Estimate interpolated values from coefficient file
--------------------------------------------------
1. Import the Estimate module:
from volumetricinterp import Estimate
#lvg. Import other stuff:
import numpy as np
import datetime as dt
2. Initialize an Estimate object with the coefficient file name:
  est_params = Estimate('output_coefficient_file.h5')
3. Call the Estimate object to get electron density at provided geodetic coordinates:
  density = est_params(time, gdlat, gdlon, gdalt)
