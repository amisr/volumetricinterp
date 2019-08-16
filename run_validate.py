# run_validate.py
# This script uses the validate method of Fit to plot the results of a volumetric interpolation
#    at a particular altitude slice.  This is useful for testing how well a particular set of
#    configuration options does at recreating the density pattern.

import datetime as dt
from Fit import Fit

def main():

    st = dt.datetime(2017,11,21,18,40)
    et = dt.datetime(2017,11,21,18,50)
    
    dayfit = Fit('config.ini')
    dayfit.validate(st, et, 350.)
	
	
if __name__ == '__main__':
	main()
