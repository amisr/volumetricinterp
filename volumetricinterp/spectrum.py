# Spectrum.py

import numpy as np
import datetime as dt
import h5py
import pymap3d as pm
import configparser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import sys
import math
from scipy import signal




from .interpolate import Interpolate
from .estimate import Estimate


class Spectrum(object):
    def __init__(self, config_file):
        """
        Validate that the parameters chosen in a config file create a reaonable interpolation by first performing
          a standard interpolation on a small subsection of a file and then plotting the resulting interpolated
          density underneith the original density values.

        Parameters:
            config_file: [str]
                standard config file that specifies the interpolation options with a [SPECTRUM] section

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

        self.starttime = dt.datetime.strptime(config.get('SPECTRUM','STARTTIME'),'%Y-%m-%dT%H:%M:%S')
        self.endtime = dt.datetime.strptime(config.get('SPECTRUM','ENDTIME'),'%Y-%m-%dT%H:%M:%S')
        self.altitudes = [float(i) for i in config.get('SPECTRUM', 'ALTITUDES').split(',')]
        self.colorlim = [float(i) for i in config.get('SPECTRUM', 'COLORLIM').split(',')]
        self.outputpng = config.get('SPECTRUM','OUTPNGNAME')
        self.latlonaltbegin = [float(i) for i in config.get('SPECTRUM', 'LATLONALTBEGIN').split(',')]
        self.latlonaltend = [float(i) for i in config.get('SPECTRUM', 'LATLONALTEND').split(',')]
        self.plasmares = [float(i) for i in config.get('SPECTRUM', 'PLASMARES').split(',')]
        # self.outputfilename = config.get('DEFAULT','OUTPUTFILENAME')

        # return starttime, endtime, altitudes, colorlim, outputpng, outputfilename

    def interpolate(self):
        """
        Perform interpolation with standard procedure using Fit class.
        """

        interp = Interpolate(self.configfile)
        interp.calc_coeffs(starttime=self.starttime, endtime=self.endtime)
        interp.saveh5()
        self.outputfilename = interp.outputfilename

    # def validate(starttime, endtime, altitudes, outputfile, outputpng, colorlim):
    def create_plots(self):
        """
        Creates a basic map of the volumetric reconstruction with the original data at a particular altitude slice to confirm that the reconstruction is reasonable.
        This function is designed to fit and plot only a small subset of an experiment (between starttime and endtime) so that it can be used to fine-tune parameters
        without waiting for an entire experiment to be processed
        """


        # initalize Evaluate object
        est_param = Estimate(self.outputfilename)

        hull_lat, hull_lon, hull_alt = pm.ecef2geodetic(est_param.hull_vert[:,0], est_param.hull_vert[:,1], est_param.hull_vert[:,2])


        # THESE ARE INPUT PARAMETERS THAT CAN BE CHANGED
        latbeg = np.nanmin(hull_lat)
        lonbeg = np.nanmin(hull_lon)
        latend = np.nanmax(hull_lat)
        lonend = np.nanmax(hull_lon)
        dres = 50
        
        
        alti = np.array(self.altitudes)*1000.

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
        for rep in range(3):
            fig = plt.figure(figsize=(len(self.altitudes)*2,len(raw_time)*2))
            gs = gridspec.GridSpec(len(raw_time), len(self.altitudes))
            gs.update(left=0.05,right=0.9,bottom=0.01,top=0.95)
            
            if rep == 0:
                map_proj = ccrs.LambertConformal(central_latitude=np.nanmean(hull_lat), central_longitude=np.nanmean(hull_lon))

            for j, alt in enumerate(self.altitudes):

                lathold = []
                lonhold = []
                dlenhold = []

                lon1 = lonbeg
                lat1 = latbeg
                alt1 = alti[j]
                lathold.extend([lat1])
                lonhold.extend([lon1])
                dlenhold.extend([0])

                while lon1 < lonend and lat1 < latend:
                    theta = math.atan2(math.sin(math.radians(lonend-lon1))*math.cos(math.radians(latend)), math.cos(math.radians(lat1))*math.sin(math.radians(latend)) - math.sin(math.radians(lat1))*math.cos(math.radians(latend))*math.cos(math.radians(lonend-lon1)))

                    lat2 = math.degrees(math.asin(math.sin(math.radians(lat1))*math.cos(dres/(6373.0+alt1/1000.)) + math.cos(math.radians(lat1))*math.sin(dres/(6373.0+alt1/1000.))*math.cos(theta)))

                    lon2 = lon1 + math.degrees(math.atan2(math.sin(theta)*math.sin(dres/(6373.0+alt1/1000.))*math.cos(math.radians(lat1)),math.cos(dres/(6373.0+alt1/1000.))-math.sin(math.radians(lat1))*math.sin(math.radians(lat2))))

                    lathold.extend([lat2])
                    lonhold.extend([lon2])
                    dlenhold.extend([dlenhold[-1]+dres])

                    lon1 = lon2
                    lat1 = lat2

                for i, time in enumerate(raw_time):

# THIS IS THE PREVIOUS DEFINITION FOR gdlat, gdlon, and gdalt
#        gdlat, gdlon, gdalt = np.meshgrid(np.linspace(latbeg,latend,100), np.linspace(lonbeg,lonend,100) , np.array(self.altitudes)*1000.)    

                    gdlat, gdlon, gdalt = np.meshgrid(lathold, lonhold, np.array(self.altitudes)*1000.)  
                    dens = est_param(time,gdlat, gdlon, gdalt)

                    # find index closeset to the projection altitude
                    aidx = np.nanargmin(np.abs(raw_alt-alt*1000.),axis=1)
                    rlat = raw_lat[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
                    rlon = raw_lon[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
                    rdens = raw_dens[i,tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]

                    # create plot
                    ax = fig.add_subplot(gs[i,j])

                    # the density is plotted on a map
                    if rep == 0:
                      # create plot
                      ax = fig.add_subplot(gs[i,j], projection=map_proj)
                      ax.coastlines()
                      ax.gridlines()

                      # plot density contours from RISR
                      c = ax.contourf(gdlon[:,:,j], gdlat[:,:,j], dens[:,:,j], np.linspace(self.colorlim[0],self.colorlim[1],31), extend='both', transform=ccrs.PlateCarree())
                      ax.scatter(rlon, rlat, c='white', s=20, transform=ccrs.Geodetic())
                      ax.scatter(rlon, rlat, c=rdens, s=10, vmin=self.colorlim[0], vmax=self.colorlim[1], transform=ccrs.Geodetic())
                      ax.plot([lonbeg,lonend], [latbeg,latend], c='black', transform=ccrs.Geodetic())
                      ax.set_title('{} km'.format(alt))

                    # the density and spectrum along a line are found
                    if rep != 0:
                        distrem = []
                        dendistrem = []

                        # haversine formula is used to find the density as a function of spatial distance from a beginning point
                        for h in range(len(gdlat[0])): 
                            dlon = gdlon[h,0,j] - gdlon[0,0,j]
                            dlat = gdlat[0,h,j] - gdlat[0,0,j]
                            adis = math.sin(math.radians(dlat)/2)**2+math.cos(math.radians(gdlat[0,0,j]))*math.cos(math.radians(gdlat[0,h,j]))*math.sin(math.radians(dlon)/2)**2            
                            cdis = 2 * math.atan2(math.sqrt(adis), math.sqrt(1 - adis))
                            if np.isnan(dens[h,h,j]) == False:
                              distrem.extend([(6373.0+gdalt[0,0,j]/1000.)*cdis])
                              dendistrem.extend([dens[h,h,j]])

# THIS IS STUFF I USED TO CHECK THAT MY SPECTRA WERE CORRECT. IT JUST REPLACES THE INTERPOLATED VALUES WITH A SINE WAVE.
#                        f = 0.00001
#                        omega = 2*math.pi*f
#                        dendistrem.extend([5*math.sin(omega*distrem[-1])])

                  # plot density as a function of distance
                        if rep == 1:
                            ax.plot(distrem, dendistrem)

                  # plot spctrum from density as a function of distance
                        if rep == 2:
                            f, Pxx_den = signal.periodogram(dendistrem, 1./(distrem[1]-distrem[0]))
                            ax.plot(f, Pxx_den)
                            ax.set_xscale("log")
                        ax.grid(True)
                        ax.set_title('{} km - {}'.format(alt,round((distrem[1]-distrem[0])*100.)/100))

              # plot time labels and colorbars
                pos = ax.get_position()
                plt.text(0.03,(pos.y0+pos.y1)/2.,time.time(),rotation='vertical',verticalalignment='center',horizontalalignment='center',transform=fig.transFigure)
                if rep == 0:
                    cax = fig.add_axes([0.91,pos.y0,0.03,pos.height])

            plt.savefig(self.outputpng+'-{}.png'.format(rep))
            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close() # Close a figure window
