# run_volumetricinterp.py

from .interpolate import Interpolate
from .validate import Validate

description = "Calculate coefficients for volmetric interpolation of a scalar quantity in a fitted AMISR file."

with open('example_config.ini', 'r') as f:
    config_file_help = f.readlines()

config_file_help = 'A configuration file that specifies the following parameters:\n'+''.join([line for line in config_file_help if not line.startswith('#') and len(line.strip())>0])


def main():

    from argparse import ArgumentParser, RawTextHelpFormatter

    # Build the argument parser tree
    parser = ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    arg = parser.add_argument('config_file',help=config_file_help)
    arg = parser.add_argument('--validate', action='store_true')

    args = vars(parser.parse_args())

    # print(args)

    # if validate option set, run validation code, otherwise run normal intepolation on full file
    if args['validate']:
        validate = Validate(args['config_file'])
        validate.interpolate()
        validate.create_plots()
    else:
        interp = Interpolate(args['config_file'])
        interp.calc_coeffs()
        interp.saveh5()



if __name__ == '__main__':
	main()
