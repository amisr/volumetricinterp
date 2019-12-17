# run_validate.py
# This script uses the validate method of Fit to plot the results of a volumetric interpolation
#    at a particular altitude slice.  This is useful for testing how well a particular set of
#    configuration options does at recreating the density pattern.

import numpy as np
import datetime as dt
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from Fit import Fit
from EvalParam import EvalParam

def interp(starttime, endtime):
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    # Build the argument parser tree
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    arg = parser.add_argument('config_file',help='A configuration file.')

    args = vars(parser.parse_args())

    fit = Fit(args['config_file'])
    fit.fit(starttime=starttime, endtime=endtime)
    fit.saveh5()

def validate(targtime, altitude, altlim=30.):
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

    # self.fit(starttime=starttime, endtime=endtime)
    #
    # lat0, lon0, alt0 = self.raw_coords

    # lat_coords = {'RISR-N':np.linspace(74., 80., 10), 'RISR-C':np.linspace(69., 75.,10)}
    # lon_coords = {'RISR-N':np.linspace(260., 285., 10), 'RISR-C':np.linspace(250., 270., 10)}

    eval = EvalParam('test_out.h5')
    # print(eval.model.latcp, eval.model.loncp)

    # set input coordinates
    latn, lonn = np.meshgrid(np.linspace(74., 80., 10), np.linspace(260., 285., 10))
    altn = np.full(latn.shape, altitude)
    # R0n = np.array([latn, lonn, altn])

    # Rshape = R0n.shape
    # R0 = R0n.reshape(Rshape[0], -1)

    map_proj = ccrs.LambertConformal(central_latitude=eval.model.latcp, central_longitude=eval.model.loncp)
    denslim = [0., 3.e11]

    dens = eval.getparam(targtime,latn.flatten(), lonn.flatten(), altn.flatten())
    dens = dens.reshape(latn.shape)
    # print(dens)

    # for i, (rd, C) in enumerate(zip(self.raw_data, self.Coeffs)):
    #     out = self.eval_model(R0,C)
    #     ne = out['param'].reshape(tuple(list(Rshape)[1:]))

    utargtime = (targtime-dt.datetime.utcfromtimestamp(0)).total_seconds()


    # get raw data from file
    with h5py.File('test_out.h5', 'r') as f:
        raw_filename = f['/RawData/filename'][()]
        # raw_lat = f['Geomag/Latitude'][:]
        # raw_lon = f['Geomag/Longitude'][:]

    # print(raw_filename, targtime)

    with h5py.File(raw_filename, 'r') as f:
        raw_alt = f['/Geomag/Altitude'][:]

        # raw_alt = raw_alt[tuple(np.arange(alt.shape[0])),tuple(aidx)]
        raw_lat = f['/Geomag/Latitude'][:]
        raw_lon = f['/Geomag/Longitude'][:]

        utime = f['Time/UnixTime'][:]
        tidx = np.argmin(np.abs(np.mean(utime,axis=1)-utargtime))
        raw_dens = f['FittedParams/Ne'][tidx,:,:]

    # find index closeset to the projection altitude
    aidx = np.nanargmin(np.abs(raw_alt-altitude*1000.),axis=1)
    # print(aidx)

    raw_lat = raw_lat[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
    raw_lon = raw_lon[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]
    raw_dens = raw_dens[tuple(np.arange(raw_alt.shape[0])),tuple(aidx)]

    # print(raw_lat.shape, raw_lon.shape, raw_dens.shape)
        # # select altitude at correct index for each beam (fancy indexing with tuples to avoid for loops)
        # latitude = lat[tuple(np.arange(alt.shape[0])),tuple(aidx)]
        # longitude = lon[tuple(np.arange(alt.shape[0])),tuple(aidx)]
        # altitude = alt[tuple(np.arange(alt.shape[0])),tuple(aidx)]
        # density = dens[:,tuple(np.arange(alt.shape[0])),tuple(aidx)]


    # print(raw_alt)
    # print(raw_alt.shape, raw_lat.shape, raw_lon.shape)

    # create plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.02, 0.1, 0.9, 0.8], projection=map_proj)
    ax.coastlines()
    ax.gridlines()
    # ax.set_extent([min(lon0),max(lon0),min(lat0),max(lat0)])

    # plot density contours from RISR
    c = ax.contourf(lonn, latn, dens, np.linspace(denslim[0],denslim[1],31), extend='both', transform=ccrs.PlateCarree())
    ax.scatter(raw_lon, raw_lat, c=raw_dens, vmin=denslim[0], vmax=denslim[1], transform=ccrs.Geodetic())
    # ax.scatter(lon0[np.abs(alt0-altitude)<altlim], lat0[np.abs(alt0-altitude)<altlim], c=rd[np.abs(alt0-altitude)<altlim], vmin=denslim[0], vmax=denslim[1], transform=ccrs.Geodetic())

    cax = fig.add_axes([0.91,0.1,0.03,0.8])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label(r'Electron Density (m$^{-3}$)')

    # plt.savefig('temp{:02d}.png'.format(i))
    # plt.close(fig)
    plt.show()


def main():

    targtime = dt.datetime(2017,11,21,18,46)
    st = dt.datetime(2017,11,21,18,40)
    et = dt.datetime(2017,11,21,18,50)
    #
    # dayfit = Fit('config.ini')
    interp(st,et)
    validate(targtime, 350.)


if __name__ == '__main__':
	main()
