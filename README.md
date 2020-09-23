# volumetricinterp
Volumetric interpolation of scalar values in an AMISR FoV

This code interpolates density measurements from an AMISR FoV by first fitting the measured density at a particular time to a 3D analytic function and then evaluating that function at any arbitrary point within the FoV.  The coefficients for the fitted model are saved to a hdf5 file, which can be used to initialize an `Evaluate` object which calculates the density at arbitrary locations.  A validation scrip (`run_validate.py`) is provided which may be useful for experimenting with different model parameters and demonstrating how code can be written with the `Evaluate` class.

## Perform fitting on an AMISR datafile
1. Create a config file based on `example_config.ini`.  At minimum, you should change `FILENAME` and `OUTPUTFILENAME` parameters to specify location/names of the AMISR file to fit and the output coefficient file.
2. Run with
```
$ python run_volumetric_interp.py config.ini
```

## Calculate interpolated values based on fitted coefficient file
1. Import the Evaluate module
```
from Evaluate import Evaluate
```
2. Initialize an Evaluate object with the coefficient file name
```
eval = Evaluate('output_coefficient_file.h5')
```
3. Call the `getparam()` function of `Evaluate` to get electron density at provided geodetic coordinates
```
density = eval.getparam(time, gdlat, gdlon, gdalt)
```
