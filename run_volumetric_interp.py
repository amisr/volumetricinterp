# run_volumetric_interp.py

import datetime as dt

from Fit import Fit


def main():

    st = dt.datetime(2017,11,21,18,40)
    et = dt.datetime(2017,11,21,18,50)
    
    dayfit = Fit('config.ini')
#     dayfit.fit()
    dayfit.validate(st, et, 350.)
#     dayfit.saveh5()

	
	
if __name__ == '__main__':
	main()
	
