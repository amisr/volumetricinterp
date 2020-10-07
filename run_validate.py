# run_validate.py
# This script uses the validate method of Fit to plot the results of a volumetric interpolation
#    at a particular altitude slice.  This is useful for testing how well a particular set of
#    configuration options does at recreating the density pattern.

from validate import Validate

description = "Calculate coefficients for volmetric interpolation of a scalar quantity in a fitted AMISR file."

config_file_help = """Calculate coefficients for volmetric interpolation
of a scalar quantity in a fitted AMISR file.

Requires a configuration file containing the following example format:

[DEFAULT]

# parameter to be interpolated
PARAM = dens

# input AMISR fitted filename
FILENAME = 20161127.002_lp_1min-fitcal.h5

# output filename to save the coefficients to
OUTPUTFILENAME = test_out.h5

# list of regularization methods to use (options are '0thorder' and 'curvature')
REGULARIZATION_LIST = curvature

# the method that should be used to determine the regularization parameter
REGULARIZATION_METHOD = chi2

# only consider points with errors between these limits
ERRLIM = 1e10,1e13

# only consider points with these fit codes
GOODFITCODE = 1,2,3,4

# only consider points with chi-squared values in this range
CHI2LIM = 0.1,10


[MODEL]
# Which model to use
MODEL = sphharmlag

# number of radial base functions used
MAXK = 4

# order of the spherical cap harmionics expansion
MAXL = 6

# limit of the cap (degrees)
CAP_LIM = 10

# the maximum z value to use in calculating the integrals for the 0th order regularization matricies
MAX_Z_INT = INF

LATCP = 78

LONCP = 262

EPS = 100000.0

LATRANGE = 74,80

LONRANGE = 260,285

ALTRANGE = 100,600

NUMGRIDPNT = 7

[VALIDATE]
# start time for Validation (ISO format)
STARTTIME = 2016-11-27T22:45:00

# endtime for Validation (ISO format)
ENDTIME = 2016-11-27T22:50:00

# list of altitudes to plot (km)
ALTITUDES = 250.0,300.0,350.0,400.0,450.0

# colorbar limits
COLORLIM = 0.0,5.0e11

# output png name for validate plots
OUTPNGNAME = test_fig.png

"""



def main():

    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    # Build the argument parser tree
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=description)
    arg = parser.add_argument('config_file',help=config_file_help)

    args = vars(parser.parse_args())

    validate = Validate(args['config_file'])
    validate.interpolate()
    validate.create_plots()


if __name__ == '__main__':
	main()
