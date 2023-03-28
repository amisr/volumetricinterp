# Class for interpolation technique developed in 2022
# This is a completely independent approach and prioritized features needed for
#   integrating AMISR measurements into numerical models/raytracers.  Specfically,
#   this algorithm forces boundary conditions to reasonable values and avoids
#   non-physical artifacts.  It fits a modified Chapman profile to clusters of
#   beams and performs a constrained 2D fit on the resulting profile parameters.

import configparser
import argparse
import h5py
import numpy as np
import pymap3d as pm
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay


# rearrange order/naming convention to match text
def chapman(z, N0, z0, HB, HT):

    # From Schunk and Nagy, 2009; eqn 11.57
    # Topside
    zp = (z-z0)/HT
    NeT = N0*np.exp(1-zp-np.exp(-zp))

    # Bottomside
    zp = (z-z0)/HB
    NeB = N0*np.exp(1-zp-np.exp(-zp))

    Ne = NeB.copy()
    Ne[z>=z0] = NeT[z>=z0]

    return Ne

def hav_new(az1, el1, b, d):
    el2 = np.arcsin(np.sin(el1)*np.cos(d) + np.cos(el1)*np.sin(d)*np.cos(b))
    az2 = az1 + np.arctan2(np.sin(b)*np.sin(d)*np.cos(el1), np.cos(d)-np.sin(el1)*np.sin(el2))
    return az2, el2

class BasisFunctions(object):
    def __init__(self, caz, cel):

        self.caz = caz
        self.cel = cel
        self.Nbasis = len(cel)

    # # Convert to solid angle???
    # def Phi(self, i, x, y):
    #     r = np.sqrt((x-self.cx[i])**2 + (y-self.cy[i])**2)
    #     # return r**2*np.log10(r)
    #     return r**3

    def Phi(self, i, az, el):
        a = np.sin((el-self.cel[i])/2)**2 + np.cos(el)*np.cos(self.cel[i])*np.sin((az-self.caz[i])/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        # r = np.sqrt((x-cx[i])**2 + (y-cy[i])**2)
        # return r**2*np.log10(r)
        return c**3





class Interp4Model(object):
    def __init__(self, config_file):
        filename = self.read_config(config_file)
        self.load_datafile(filename)
        self.cluster_beams()
        self.fit_profiles()
        self.fit_2d()
        self.save_output()

# def parse_list(s):
#     return [float(i) for i in s.split(',')]

    def read_config(self, config_file):
        # read config file
        # filename
        # data filter params
        # initial guess for chapman params
        # boundary location/values
        config = configparser.ConfigParser(converters={'list': lambda s: [float(i) for i in s.split(',')]})
        config.read(config_file)
        filename = config.get('DEFAULT', 'FILENAME')
        self.output_filename = config.get('DEFAULT', 'OUTPUTFILENAME')
        self.boundary_value = config.getlist('DEFAULT', 'BOUNDARY_VALUE')
        # self.boundary_circle = config.getfloat('DEFAULT', 'BOUNDARY_CIRCLE')
        self.cent_az = config.getfloat('DEFAULT', 'CENT_AZ')
        self.cent_el = config.getfloat('DEFAULT', 'CENT_EL')
        self.starttime = config.get('DEFAULT', 'STARTTIME')
        self.endtime = config.get('DEFAULT', 'ENDTIME')
        return filename

    def load_datafile(self, filename):
        # amisr_file = '/Users/e30737/Desktop/Data/AMISR/RISR-N/2017/20171119.001_lp_1min-fitcal.h5'
        # amisr_file = '/Users/e30737/Desktop/Data/AMISR/RISR-N/2019/20190510.001_lp_5min-fitcal.h5'
        # amisr_file = '/Users/e30737/Desktop/Data/AMISR/synthetic/imaging_chapman.h5'
        with h5py.File(filename, 'r') as h5:
            self.beamcode = h5['BeamCodes'][:]
            # lat = h5['Geomag/Latitude'][:]
            # lon = h5['Geomag/Longitude'][:]
            self.alt = h5['Geomag/Altitude'][:]
            self.dens = h5['FittedParams/Ne'][:]
            self.dens_err = h5['FittedParams/dNe'][:]
            chi2 = h5['FittedParams/FitInfo/chi2'][:]
            fitcode = h5['/FittedParams/FitInfo/fitcode'][:]
            site_lat = h5['Site/Latitude'][()]
            site_lon = h5['Site/Longitude'][()]
            site_alt = h5['Site/Altitude'][()]
            # self.time = h5['Time/UnixTime'][:10,0]
            self.time = h5['Time/UnixTime'][:,0]
            # print(dt.datetime.utcfromtimestamp(h5['Time/UnixTime'][0,0]))

        # print(np.diff(range0))
        # time = utime.astype(np.datetime64)
        # self.time = utime[:10].astype('datetime64[s]')
        # print(time[0].astype('datetime64[s]'))


        # lat = lat[np.isfinite(dens)]
        # lon = lon[np.isfinite(dens)]
        # alt = alt[np.isfinite(dens)]
        # eobs = dens_err[np.isfinite(dens)]
        # vobs = dens[np.isfinite(dens)]

        # data_check = np.array([dens_err>1.e10, dens_err<1.e12, chi2>0.1, chi2<10., np.isin(fitcode,[1,2,3,4])])
        data_check = np.array([chi2>0.1, chi2<10., np.isin(fitcode,[1,2,3,4])])
        # If ANY elements of data_check are FALSE, flag index as bad data
        bad_data = np.squeeze(np.any(data_check==False,axis=0,keepdims=True))
        self.dens[bad_data] = np.nan
        self.dens_err[bad_data] = np.nan

        # # Also remove high altitude points with extremely low error and low Ne values - these bias the topside fit
        # bad_data = ((alt>400.*1000) & (dens_err<1.e10) & (dens<5.e10))
        # dens[bad_data] = np.nan
        # dens_err[bad_data] = np.nan


# # find time index
# targtime = np.datetime64('2017-11-21T19:20')
# print(targtime)
# tidx = np.argmin(np.abs(time-targtime))
# print(tidx, time[tidx].astype('datetime64[s]'))

    def cluster_beams(self):

        r = np.cos(self.beamcode[:,2]*np.pi/180.)
        t = self.beamcode[:,1]*np.pi/180.
        points = np.array([r*np.sin(t), r*np.cos(t)]).T

        # Calculate Delaunay triangulation simplices
        tri = Delaunay(points).simplices

        # Eliminate simplices that are very close to colinear
        # If points truely colinear, resulting triangle has area of zero
        # Remove simplices where the determinate of the points is close to zero (<1e3)
        D = np.linalg.det(np.array([np.ones(tri.shape), points[tri,0], points[tri,1]]).transpose(1,0,2))
        self.tri2 = tri[D>1.e-3]

        # Find center of each cluster
        clust_points = np.mean(points[self.tri2, :], axis=1)
        self.clust_el = np.arccos(np.sqrt(clust_points[:,0]**2 + clust_points[:,1]**2))
        self.clust_az = np.arctan2(clust_points[:,0], clust_points[:,1])

# # Plot
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111, projection='polar')
# ax.set_theta_direction(-1)
# ax.set_theta_offset(np.pi/2.0)
# ax.set_rlabel_position(100.)
# elticks = np.arange(20., 90., 10.)
# ax.set_rticks(np.cos(elticks*np.pi/180.))
# ax.set_yticklabels([str(int(el))+u'\N{DEGREE SIGN}' for el in elticks])
#
# ax.scatter(t, r)
# ax.triplot(t, r, tri2)
# ax.scatter(clust_t, clust_r)
# # for i in range(len(clust_t)):
# #     ax.text(clust_t[i], clust_r[i], i)
# plt.savefig(figure_ouput_dir+'beam_clusters.png', bbox_inches='tight')


    def fit_profiles(self):

        # self.chapman_coefficients = np.empty((len(self.time),len(self.tri2),4))
        # plot_simplices = [50,29,18,25]
        # simplices_colors = ['lime','cyan','red','yellow']
        
        # print(self.time, np.datetime64(self.starttime).astype('int'))
        
        
        stidx = np.argmin(np.abs(self.time-np.datetime64(self.starttime).astype('int')))
        etidx = np.argmin(np.abs(self.time-np.datetime64(self.endtime).astype('int')))
        self.chapman_coefficients = np.empty((etidx-stidx,len(self.tri2),4))

        # for tidx in range(len(self.time)):
        for tidx in range(stidx, etidx):
            print('profile fit - ' + str(self.time[tidx].astype('datetime64[s]')))

            for i, clust_index in enumerate(self.tri2):

                # Select data points for each beam cluster
                a = self.alt[clust_index]
                d = self.dens[tidx,clust_index,:]
                dd = self.dens_err[tidx,clust_index,:]

                # Filter high altitude points with extremely low error and low Ne values - these bias the topside fit
                filt_data = ((a>400.*1000) & (dd<5.e10) & (d<5.e10))
                good_data = (np.isfinite(d) & np.isfinite(dd) & (dd!=0.) & ~filt_data)
                # print(a[good_data])
                # print(d[good_data], dd[good_data])
                # print(d[good_data], self.chapman(a[good_data], 4.e11,300.*1000.,50.*1000.,100.*1000.))

                try:
                    coeffs, _ = curve_fit(chapman, a[good_data], d[good_data], sigma=dd[good_data], p0=[4.e11,300.*1000.,50.*1000.,100.*1000.], bounds=[[0.,0.,0.,0.],[np.inf,np.inf,np.inf,np.inf]], absolute_sigma=True)
                except RuntimeError:
                    coeffs = [np.nan, np.nan, np.nan, np.nan]

                self.chapman_coefficients[tidx-stidx,i] = np.array(coeffs)
                # chapman_coefficients.append(coeffs)

                # if i in plot_simplices:
                #     fig = plt.figure(figsize=(7,5))
                #     ax = fig.add_subplot(111)
                #
                #     ax.scatter(d[filt_data], a[filt_data]/1000., color='red', s=150)
                #
                #     c = ax.scatter(d, a/1000., c=dd, vmin=0., vmax=3.e11, cmap='cividis')
                #     plt.colorbar(c, label=r'Electron Density Error (m$^{-3}$)')
                #
                #     ai = np.arange(100., 700., 1.)*1000.
                #     di = chapman(ai, *coeffs)
                #     ax.plot(di, ai/1000., color='orange')
                #     ax.set_xlim([0.,1.e12])
                #     ax.set_xlabel(r'Electron Density (m$^{-3}$)')
                #     ax.set_ylabel('Altitude (km)')
                #
                #     # print parameters
                #     nl = '\n'
                #     textstring = rf'$N_0$ = {coeffs[0]:.2e} m$^{{-3}}${nl}$z_0$ = {coeffs[1]/1000.:.2f} km{nl}$H_B$ = {coeffs[2]/1000.:.2f} km{nl}$H_T$ = {coeffs[3]/1000.:.2f} km'
                #     bbox = dict(edgecolor=simplices_colors[plot_simplices.index(i)], facecolor='white', linewidth=2)
                #     ax.text(0.49, 0.7, textstring, fontsize=15, bbox=bbox, transform=ax.transAxes)
                #
                #     plt.savefig(figure_ouput_dir+'profile_fit{:02d}.png'.format(i), bbox_inches='tight')

                # ax.text(100000., 9.0e11, 'HmT={:.2e}'.format(coeffs[1]))
                # ax.text(100000., 8.5e11, 'HmB={:.2e}'.format(coeffs[2]))
                # ax.text(100000., 8.0e11, 'hmF2={:.2e}'.format(coeffs[3]))

            # chapman_coefficients[tidx] = np.array(chapman_coefficients)
            # clust_az = np.array(clust_az)
            # clust_el = np.array(clust_el)
            # print(chapman_coefficients.shape, clust_az.shape, clust_el.shape)


    def fit_2d(self):

        # vobs = self.chapman_coefficients[:,1].copy()/1000.

        # print(np.max(clust_r), np.arccos(np.max(clust_r))*180./np.pi)

#         clust_x = self.clust_r*np.sin(self.clust_t)
#         clust_y = self.clust_r*np.cos(self.clust_t)

#         fov_xrng = [np.min(clust_x), np.max(clust_x)]
#         fov_yrng = [np.min(clust_y), np.max(clust_y)]
#         self.fov_cent = [np.mean(fov_xrng), np.mean(fov_yrng)]

#         xks, yks = np.meshgrid(np.linspace(fov_xrng[0],fov_xrng[1],8), np.linspace(fov_yrng[0],fov_yrng[1],8))
#         self.cx = xks.flatten()
#         self.cy = yks.flatten()

        caz = [self.cent_az*np.pi/180.]
        cel = [self.cent_el*np.pi/180.]
        circ_az, circ_el = hav_new(self.cent_az*np.pi/180., self.cent_el*np.pi/180., np.arange(0., 360., 60.)*np.pi/180., np.pi/24.)
        caz.extend(circ_az)
        cel.extend(circ_el)
        circ_az, circ_el = hav_new(self.cent_az*np.pi/180., self.cent_el*np.pi/180., np.arange(0., 360., 45.)*np.pi/180.+np.pi/8, np.pi/12.)
        caz.extend(circ_az)
        cel.extend(circ_el)
        circ_az, circ_el = hav_new(self.cent_az*np.pi/180., self.cent_el*np.pi/180., np.arange(0., 360., 30.)*np.pi/180., np.pi/8.)
        caz.extend(circ_az)
        cel.extend(circ_el)
        circ_az, circ_el = hav_new(self.cent_az*np.pi/180., self.cent_el*np.pi/180., np.arange(0., 360., 15.)*np.pi/180.+np.pi/24, np.pi/4.5)
        caz.extend(circ_az)
        cel.extend(circ_el)

        rbf = BasisFunctions(caz, cel)
        N = rbf.Nbasis
        
        self.caz = caz
        self.cel = cel
        #
        #
        # N = len(self.cx)

        bound_az, bound_el = hav_new(self.cent_az*np.pi/180., self.cent_el*np.pi/180., np.arange(0., 360., 10.)*np.pi/180., np.pi/5)

        # x_const = self.boundary_circle*np.cos(np.linspace(0.,2*np.pi, 40))+self.fov_cent[0]
        # y_const = self.boundary_circle*np.sin(np.linspace(0.,2*np.pi, 40))+self.fov_cent[1]
        C = len(bound_el)

        # X = list()
        
        stidx = np.argmin(np.abs(self.time-np.datetime64(self.starttime).astype('int')))
        etidx = np.argmin(np.abs(self.time-np.datetime64(self.endtime).astype('int')))

        self.X = np.empty((etidx-stidx, 4, N+C))
        print(self.chapman_coefficients.shape)

        for m, chap_coeff in enumerate(self.chapman_coefficients):
            print('2D fit - ' + str(self.time[stidx+m].astype('datetime64[s]')))

            for n, (vobs, const) in enumerate(zip(chap_coeff.T, self.boundary_value)):

                constraints = np.array([[az, el, const] for az, el in zip(bound_az, bound_el)])

                a = np.zeros((N+C,N+C))
                b = np.zeros(N+C)

                a[:N,:N] = np.array([[2*np.sum(rbf.Phi(i,self.clust_az,self.clust_el)*rbf.Phi(j,self.clust_az,self.clust_el)) for i in range(N)] for j in range(N)])
                a[:N,N:] = np.array([[-rbf.Phi(j,r[0],r[1]) for r in constraints] for j in range(N)])
                a[N:,:N] = np.array([[rbf.Phi(i,r[0],r[1]) for i in range(N)] for r in constraints])
                b[:N] = np.array([2*np.sum(vobs*rbf.Phi(j,self.clust_az,self.clust_el)) for j in range(N)])
                b[N:] = np.array([r[2] for r in constraints])
                # print(a.shape, b.shape)

                self.X[m,n,:] = np.linalg.solve(a,b)

#         for i, chap_coeff in enumerate(self.chapman_coefficients):
#             print('2D fit - ' + str(self.time[i].astype('datetime64[s]')))

#             for j, (vobs, const) in enumerate(zip(chap_coeff.T, self.boundary_value)):

#                 constraints = np.array([[x, y, const] for x, y in zip(x_const, y_const)])

#                 a = np.zeros((N+C,N+C))
#                 b = np.zeros(N+C)

#                 a[:N,:N] = np.array([[2*np.sum(rbf.Phi(i,clust_x,clust_y)*rbf.Phi(j,clust_x,clust_y)) for i in range(N)] for j in range(N)])
#                 a[:N,N:] = np.array([[-rbf.Phi(j,r[0],r[1]) for r in constraints] for j in range(N)])
#                 a[N:,:N] = np.array([[rbf.Phi(i,r[0],r[1]) for i in range(N)] for r in constraints])
#                 b[:N] = np.array([2*np.sum(vobs*rbf.Phi(j,clust_x,clust_y)) for j in range(N)])
#                 b[N:] = np.array([r[2] for r in constraints])
#                 # print(a.shape, b.shape)
#                 # print(np.linalg.solve(a,b))

#                 self.X[i,j,:] = np.linalg.solve(a,b)

        # self.X = np.array(X)

# Need to output and save X and fov_cent
    def save_output(self):

        stidx = np.argmin(np.abs(self.time-np.datetime64(self.starttime).astype('int')))
        etidx = np.argmin(np.abs(self.time-np.datetime64(self.endtime).astype('int')))

        with h5py.File(self.output_filename, 'w') as h5:
            h5.create_dataset('X', data=self.X)
            # h5.create_dataset('fov_cent', data=self.fov_cent)
            h5.create_dataset('cent_az', data=self.cent_az)
            h5.create_dataset('cent_el', data=self.cent_el)
            h5.create_dataset('caz', data=self.caz)
            h5.create_dataset('cel', data=self.cel)
            h5.create_dataset('time', data=self.time[stidx:etidx])
            h5.create_dataset('boundary_value', data=self.boundary_value)
            # h5.create_dataset('boundary_circle', data=self.boundary_circle)


class CalcInterp(object):
    def __init__(self, filename):

        self.load_file(filename)

        self.rbf = BasisFunctions(self.caz, self.cel)


    def load_file(self, filename):
        with h5py.File(filename, 'r') as h5:
            self.X = h5['X'][:]
            # self.fov_cent = h5['fov_cent'][:]
            self.cent_az = h5['cent_az'][()]
            self.cent_el = h5['cent_el'][()]
            self.caz = h5['caz'][:]
            self.cel = h5['cel'][:]
            utime = h5['time'][:]
            self.boundary_value = h5['boundary_value'][:]
            # self.boundary_circle = h5['boundary_circle'][()]

        self.time = utime.astype('datetime64[s]')


    def grid_enu(self, targtime, xrng, yrng, zrng):

        tidx = np.argmin(np.abs(self.time-targtime))
        print(self.time[tidx])

        Xgrid, Ygrid, Zgrid = np.meshgrid(xrng, yrng, zrng)

        azgrid, elgrid, _ = pm.enu2aer(Xgrid, Ygrid, Zgrid, deg=False)

        # xgrid = np.cos(elgrid)*np.sin(azgrid)
        # ygrid = np.cos(elgrid)*np.cos(azgrid)

        # out_of_bound = np.sqrt((xgrid-self.fov_cent[0])**2 + (ygrid-self.fov_cent[1])**2)>self.boundary_circle

        a = np.sin((elgrid-self.cent_el*np.pi/180.)/2)**2 + np.cos(elgrid)*np.cos(self.cent_el*np.pi/180.)*np.sin((azgrid-self.cent_az*np.pi/180.)/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        out_of_bound = c>np.pi/5

        pgrid = list()
        for x1, bc in zip(self.X[tidx], self.boundary_value):
            pg = np.sum(np.array([x1[i]*self.rbf.Phi(i,azgrid, elgrid) for i in range(self.rbf.Nbasis)]), axis=0)
            # print(x1)
            pg[out_of_bound] = bc
            pgrid.append(pg)
        pgrid = np.array(pgrid)
        # pgrid = np.array([np.sum(np.array([x1[i]*Phi(i,xgrid, ygrid) for i in range(N)]), axis=0) for x1 in X])

        vgrid = chapman(Zgrid, pgrid[0], pgrid[1], pgrid[2], pgrid[3])
        print(vgrid.shape)

        return Xgrid, Ygrid, Zgrid, vgrid

    def point_enu(self, targtime, epnt, npnt, upnt):

        tidx = np.argmin(np.abs(self.time-targtime))

        azpnt, elpnt, _ = pm.enu2aer(epnt, npnt, upnt, deg=False)

        #xpnt = np.cos(elpnt)*np.sin(azpnt)
        #ypng = np.cos(elpnt)*np.cos(azpnt)

        #out_of_bound = np.sqrt((xpnt-self.fov_cent[0])**2 + (ypnt-self.fov_cent[1])**2)>self.boundary_circle

        a = np.sin((elpnt-self.cent_el*np.pi/180.)/2)**2 + np.cos(elpnt)*np.cos(self.cent_el*np.pi/180.)*np.sin((azpnt-self.cent_az*np.pi/180.)/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        out_of_bound = c>np.pi/5

        pgrid = list()
        for x1, bc in zip(self.X[tidx], self.boundary_value):
            pg = np.sum(np.array([x1[i]*self.rbf.Phi(i, azpnt, elpnt) for i in range(self.rbf.Nbasis)]), axis=0)
            # print(x1)
            pg[out_of_bound] = bc
            pgrid.append(pg)
        pgrid = np.array(pgrid)
        # pgrid = np.array([np.sum(np.array([x1[i]*Phi(i,xgrid, ygrid) for i in range(N)]), axis=0) for x1 in X])

        vpnt = chapman(upnt, pgrid[0], pgrid[1], pgrid[2], pgrid[3])
        print(vpnt.shape)

        return vpnt

    def point_geodetic(self, targtime, lat, lon, alt):

        tidx = np.argmin(np.abs(self.time-targtime))

        azpnt, elpnt, _ = pm.geodetic2aer(lat, lon, h, lat0, lon0, h0)
        # azpnt, elpnt, _ = pm.enu2aer(epnt, npnt, upnt, deg=False)

        # fig = plt.figure(figsize=(30,6))
        # gs = gridspec.GridSpec(1,5)
        # for i in range(5):
        #     ax = fig.add_subplot(gs[i])
        #     aidx = 10*i+10
        #     c = ax.pcolormesh(Xgrid[:,:,aidx]/1000., Ygrid[:,:,aidx]/1000., vgrid[:,:,aidx], vmin=0., vmax=5.e11)
        #     # ax.pcolormesh(Xgrid[:,:,2*i], Ygrid[:,:,2*i], elgrid2[:,:,2*i]*180./np.pi, vmin=0., vmax=90.)
        #     ax.set_title('Alt = {} km'.format(int(Zgrid[0,0,aidx]/1000.)))
        #     ax.set_aspect('equal')
        # # cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        # # fig.colorbar(c, cax=cax, label=r'Electron Density (m$^{-3}$)')
        # # plt.savefig(figure_ouput_dir+'alt_slice.png', bbox_inches='tight')
        # plt.show()


def main():
    config_file_help = 'Some help string'

    # Build the argument parser tree
    parser = argparse.ArgumentParser(description=config_file_help,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    arg = parser.add_argument('config',help='Configuration file for volumetric interpolation.')
    args = vars(parser.parse_args())

    Interp4Model(args['config'])

if __name__=='__main__':
    main()
