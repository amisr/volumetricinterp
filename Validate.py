# Validate.py

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


class Validate(object):
    def __init__(self, config_file):
        """
        Validate that the parameters chosen in a config file create a reaonable interpolation by first performing
          a standard interpolation on a small subsection of a file and then plotting the resulting interpolated
          density underneith the original density values.

        Parameters:
            config_file: [str]
                standard config file that specifies the interpolation options with a [VALIDATE] section

        """

        self.configfile = config_file
        self.read_config(self.configfile)

    def read_config(self, config_file):
        """
        Read fit parameters from input config file.

        Parameters:
            config_file: [str]
                config file name
        """

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.starttime = dt.datetime.strptime(config.get('VALIDATE','STARTTIME'),'%Y-%m-%dT%H:%M:%S')
        self.endtime = dt.datetime.strptime(config.get('VALIDATE','ENDTIME'),'%Y-%m-%dT%H:%M:%S')
        self.altitudes = [float(i) for i in config.get('VALIDATE', 'ALTITUDES').split(',')]
        self.colorlim = [float(i) for i in config.get('VALIDATE', 'COLORLIM').split(',')]
        self.outputpng = config.get('VALIDATE','OUTPNGNAME')
        # self.outputfilename = config.get('DEFAULT','OUTPUTFILENAME')

        # return starttime, endtime, altitudes, colorlim, outputpng, outputfilename

    def interp(self):
        """
        Perform interpolation with standard procedure using Fit class.
        """

        fit = Fit(self.configfile)
        fit.fit(starttime=self.starttime, endtime=self.endtime)
        fit.saveh5()
        self.outputfilename = fit.outputfilename

    # def validate(starttime, endtime, altitudes, outputfile, outputpng, colorlim):
    def validate(self):
        """
        Creates a basic map of the volumetric reconstruction with the original data at a particular altitude slice to confirm that the reconstruction is reasonable.
        This function is designed to fit and plot only a small subset of an experiment (between starttime and endtime) so that it can be used to fine-tune parameters
        without waiting for an entire experiment to be processed
        """


        # initalize Evaluate object
        eval = Evaluate(self.outputfilename)

        hull_lat, hull_lon, hull_alt = pm.ecef2geodetic(eval.hull_vert[:,0], eval.hull_vert[:,1], eval.hull_vert[:,2])

        # set input coordinates
        gdlat, gdlon, gdalt = np.meshgrid(np.linspace(np.nanmin(hull_lat), np.nanmax(hull_lat), 100), np.linspace(np.nanmin(hull_lon), np.nanmax(hull_lon), 100), np.array(self.altitudes)*1000.)

        # get original raw data from file
        with h5py.File(self.outputfilename, 'r') as f:
            raw_filename = f['/RawData/filename'][()]

        with h5py.File(raw_filename, 'r') as f:
            raw_alt = f['/Geomag/Altitude'][:]
            raw_lat = f['/Geomag/Latitude'][:]
            raw_lon = f['/Geomag/Longitude'][:]

            utime = f['Time/UnixTime'][:]
            idx = np.argwhere((utime[:,0]>=(self.starttime-dt.datetime.utcfromtimestamp(0)).total_seconds()) & (utime[:,1]<=(self.endtime-dt.datetime.utcfromtimestamp(0)).total_seconds())).flatten()
            raw_time = np.array([dt.datetime.utcfromtimestamp(t) for t in np.mean(utime, axis=1)[idx]])
            raw_dens = f['FittedParams/Ne'][idx,:,:]


        # setup figure
        fig = plt.figure(figsize=(len(self.altitudes)*2,len(raw_time)*2))
        gs = gridspec.GridSpec(len(raw_time), len(self.altitudes))
        gs.update(left=0.05,right=0.9,bottom=0.01,top=0.95)
        map_proj = ccrs.LambertConformal(central_latitude=np.nanmean(hull_lat), central_longitude=np.nanmean(hull_lon))


        for i, time in enumerate(raw_time):

            dens = eval.getparam(time,gdlat, gdlon, gdalt)

            for j, alt in enumerate(self.altitudes):

                # find index closeset to the projection altitude
                aidx = np.nanargmin(np.abs(raw_alt-alt*1000.),axis=1)
                rlat = raw_lat[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
                rlon = raw_lon[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
                rdens = raw_dens[i,tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]

                # create plot
                ax = fig.add_subplot(gs[i,j], projection=map_proj)
                ax.coastlines()
                ax.gridlines()

                # plot density contours from RISR
                c = ax.contourf(gdlon[:,:,j], gdlat[:,:,j], dens[:,:,j], np.linspace(self.colorlim[0],self.colorlim[1],31), extend='both', transform=ccrs.PlateCarree())
                ax.scatter(rlon, rlat, c='white', s=20, transform=ccrs.Geodetic())
                ax.scatter(rlon, rlat, c=rdens, s=10, vmin=self.colorlim[0], vmax=self.colorlim[1], transform=ccrs.Geodetic())
                ax.set_title('{} km'.format(alt))

            # plot time labels and colorbars
            pos = ax.get_position()
            plt.text(0.03,(pos.y0+pos.y1)/2.,time.time(),rotation='vertical',verticalalignment='center',horizontalalignment='center',transform=fig.transFigure)
            cax = fig.add_axes([0.91,pos.y0,0.03,pos.height])
            cbar = plt.colorbar(c, cax=cax)
            cbar.set_label(r'Ne (m$^{-3}$)')

        plt.savefig(self.outputpng)
