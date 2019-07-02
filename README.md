# volumetricinterp
Volumetric interpolation of scalar values in an AMISR FoV

This code interpolates density measurements from an AMISR FoV by first fitting the measured density at a paricular time to a 3D analytic function and then evaluating that function at any arbitrary point within the FoV.  Example useage can be found in the `main()` function of `amisr_fit.py`.  Parameters can be changed at the top of `amisr_fit.py` (directly beneath import statements).  This code was written in python 2.7 and has not been tested in python 3.

To Run:
```
python amisr_fit.py
```
