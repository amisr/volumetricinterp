# run_volumetric_interp.py

from Fit import Fit

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
MODEL = Model
; MODEL = Model_RBF

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

"""

def main():

    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    # Build the argument parser tree
    parser = ArgumentParser(description=config_file_help,
                            formatter_class=RawDescriptionHelpFormatter)
    arg = parser.add_argument('config_file',help='A configuration file.')

    args = vars(parser.parse_args())

    fit = Fit(args['config_file'])
    fit.fit()
    fit.saveh5()



if __name__ == '__main__':
	main()
