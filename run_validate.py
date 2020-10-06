# run_validate.py
# This script uses the validate method of Fit to plot the results of a volumetric interpolation
#    at a particular altitude slice.  This is useful for testing how well a particular set of
#    configuration options does at recreating the density pattern.

import numpy as np
import datetime as dt
import h5py
import pymap3d as pm
import configparser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from Fit import Fit
from Evaluate import Evaluate

def read_config(config_file):

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    starttime = dt.datetime.strptime(config.get('VALIDATE','STARTTIME'),'%Y-%m-%dT%H:%M:%S')
    endtime = dt.datetime.strptime(config.get('VALIDATE','ENDTIME'),'%Y-%m-%dT%H:%M:%S')
    altitudes = [float(i) for i in config.get('VALIDATE', 'ALTITUDES').split(',')]
    colorlim = [float(i) for i in config.get('VALIDATE', 'COLORLIM').split(',')]
    outputpng = config.get('VALIDATE','OUTPNGNAME')
    outputfilename = config.get('DEFAULT','OUTPUTFILENAME')

    return starttime, endtime, altitudes, colorlim, outputpng, outputfilename

def interp(config_file, starttime, endtime):

    fit = Fit(config_file)
    fit.fit(starttime=starttime, endtime=endtime)
    fit.saveh5()

def validate(starttime, endtime, altitudes, outputfile, outputpng, colorlim):
    """
    Creates a basic map of the volumetric reconstruction with the original data at a particular altitude slice to confirm that the reconstruction is reasonable.
    This function is designed to fit and plot only a small subset of an experiment (between starttime and endtime) so that it can be used to fine-tune parameters
    without waiting for an entire experiment to be processed

    Parameters:
        starttime: [datetime]
            start of interval to validate
        endtime: [datetime]
            end of interval to validate
        altitude: [float]
            altitude of the slice
        altlim: [float]
            points that fall +/- altlim from altitude will be plotted on top of reconstructed contours as scatter
    """


    # initalize Evaluate object
    eval = Evaluate(outputfile)

    hull_lat, hull_lon, hull_alt = pm.ecef2geodetic(eval.hull_vert[:,0], eval.hull_vert[:,1], eval.hull_vert[:,2])

    # set input coordinates
    gdlat, gdlon, gdalt = np.meshgrid(np.linspace(np.nanmin(hull_lat), np.nanmax(hull_lat), 100), np.linspace(np.nanmin(hull_lon), np.nanmax(hull_lon), 100), altitudes)

    # get original raw data from file
    with h5py.File(outputfile, 'r') as f:
        raw_filename = f['/RawData/filename'][()]

    with h5py.File(raw_filename, 'r') as f:
        raw_alt = f['/Geomag/Altitude'][:]
        raw_lat = f['/Geomag/Latitude'][:]
        raw_lon = f['/Geomag/Longitude'][:]

        utime = f['Time/UnixTime'][:]
        idx = np.argwhere((utime[:,0]>=(starttime-dt.datetime.utcfromtimestamp(0)).total_seconds()) & (utime[:,1]<=(endtime-dt.datetime.utcfromtimestamp(0)).total_seconds())).flatten()
        raw_time = np.array([dt.datetime.utcfromtimestamp(t) for t in np.mean(utime, axis=1)[idx]])
        raw_dens = f['FittedParams/Ne'][idx,:,:]


    # setup figure
    fig = plt.figure(figsize=(len(altitudes)*2,len(raw_time)*2))
    gs = gridspec.GridSpec(len(raw_time), len(altitudes))
    gs.update(left=0.05,right=0.9,bottom=0.01,top=0.95)
    map_proj = ccrs.LambertConformal(central_latitude=np.nanmean(hull_lat), central_longitude=np.nanmean(hull_lon))


    for i, time in enumerate(raw_time):

        dens = eval.getparam(time,gdlat, gdlon, gdalt)

        for j, alt in enumerate(altitudes):

            # find index closeset to the projection altitude
            aidx = np.nanargmin(np.abs(raw_alt-alt),axis=1)
            rlat = raw_lat[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
            rlon = raw_lon[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
            rdens = raw_dens[i,tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]

            # create plot
            ax = fig.add_subplot(gs[i,j], projection=map_proj)
            ax.coastlines()
            ax.gridlines()

            # plot density contours from RISR
            c = ax.contourf(gdlon[:,:,j], gdlat[:,:,j], dens[:,:,j], np.linspace(colorlim[0],colorlim[1],31), extend='both', transform=ccrs.PlateCarree())
            ax.scatter(rlon, rlat, c='white', s=20, transform=ccrs.Geodetic())
            ax.scatter(rlon, rlat, c=rdens, s=10, vmin=colorlim[0], vmax=colorlim[1], transform=ccrs.Geodetic())
            ax.set_title('{} km'.format(alt/1000.))

        # plot time labels and colorbars
        pos = ax.get_position()
        plt.text(0.03,(pos.y0+pos.y1)/2.,time.time(),rotation='vertical',verticalalignment='center',horizontalalignment='center',transform=fig.transFigure)
        cax = fig.add_axes([0.91,pos.y0,0.03,pos.height])
        cbar = plt.colorbar(c, cax=cax)
        cbar.set_label(r'Ne (m$^{-3}$)')

    plt.savefig(outputpng)


def main():

    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    # Build the argument parser tree
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    arg = parser.add_argument('config_file',help='A configuration file.')

    args = vars(parser.parse_args())
    config_file = args['config_file']

    starttime, endtime, altitudes, colorlim, outputpng, outputfilename = read_config(config_file)

    interp(config_file, starttime, endtime)
    validate(starttime, endtime, np.array(altitudes)*1000., outputfilename, outputpng, colorlim)


if __name__ == '__main__':
	main()
