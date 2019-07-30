# run_volumetric_interp.py

from Fit import Fit

config_file_help = """Calculate coefficients for volmetric interpolation
of a scalar quantity in a fitted AMISR file.

Requires a configuration file containing the following example format:

[DEFAULT]

# parameter to be interpolated
PARAM = 'dens'

# input AMISR fitted filename
FILENAME = '20171119.001_lp_1min-fitcal.h5'

# output filename to save the coefficients to
OUTPUTFILENAME = 'test_out.h5'

# number of radial base functions used
MAXK = 4

# order of the spherical cap harmionics expansion
MAXL = 6

# limit of the cap (degrees)
CAP_LIM = 6

# list of regularization methods to use (options are '0thorder' and 'curvature')
REGULARIZATION_LIST = ['0thorder']

# the method that should be used to determine the regularization parameter
REGULARIZATION_METHOD = 'chi2'

# the maximum z value to use in calculating the integrals for the 0th order regularization matricies
MAX_Z_INT = inf
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
	
