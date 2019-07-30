# run_volumetric_interp.py

# import datetime as dt
from Fit import Fit

def main():

#     st = dt.datetime(2017,11,21,18,40)
#     et = dt.datetime(2017,11,21,18,50)
    
    fit = Fit('config.ini')
#     fit.fit(st, et)
    fit.fit()
    fit.saveh5()

	
	
if __name__ == '__main__':
	main()
	
