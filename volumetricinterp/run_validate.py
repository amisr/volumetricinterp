# run_validate.py
# This script uses the validate method of Fit to plot the results of a volumetric interpolation
#    at a particular altitude slice.  This is useful for testing how well a particular set of
#    configuration options does at recreating the density pattern.

from .validate import Validate

description = "Calculate coefficients for volmetric interpolation of a scalar quantity in a fitted AMISR file."

with open('example_config.ini', 'r') as f:
    config_file_help = f.readlines()

config_file_help = 'A configuration file that specifies the following parameters:\n'+''.join([line for line in config_file_help if not line.startswith('#') and len(line.strip())>0])


def main():

    from argparse import ArgumentParser, RawTextHelpFormatter

    # Build the argument parser tree
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, description=description)
    arg = parser.add_argument('config_file',help=config_file_help)

    args = vars(parser.parse_args())

    validate = Validate(args['config_file'])
    validate.interpolate()
    validate.create_plots()


if __name__ == '__main__':
	main()
