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

    def Phi(self, i, az, el):
        a = np.sin((el-self.cel[i])/2)**2 + np.cos(el)*np.cos(self.cel[i])*np.sin((az-self.caz[i])/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        return c**3





class Interp4Model(object):
    def __init__(self, config_file):
        filename, starttime, endtime = self.read_config(config_file)
        self.load_datafile(filename, starttime, endtime)
        self.cluster_beams()
        self.fit_profiles()
        self.fit_2d()
        self.save_output()


    def read_config(self, config_file):

        config = configparser.ConfigParser(converters={'list': lambda s: [float(i) for i in s.split(',')]})
        config.read(config_file)
        filename = config.get('DEFAULT', 'FILENAME')
        self.output_filename = config.get('DEFAULT', 'OUTPUTFILENAME')
        self.boundary_value = config.getlist('DEFAULT', 'BOUNDARY_VALUE')
        self.cent_az = config.getfloat('DEFAULT', 'CENT_AZ')
        self.cent_el = config.getfloat('DEFAULT', 'CENT_EL')
        starttime = config.get('DEFAULT', 'STARTTIME')
        endtime = config.get('DEFAULT', 'ENDTIME')
        return filename, starttime, endtime

    def load_datafile(self, filename, starttime, endtime):

        with h5py.File(filename, 'r') as h5:
            self.beamcode = h5['BeamCodes'][:]
            self.alt = h5['Geomag/Altitude'][:]
            self.dens = h5['FittedParams/Ne'][:]
            self.dens_err = h5['FittedParams/dNe'][:]
            chi2 = h5['FittedParams/FitInfo/chi2'][:]
            fitcode = h5['/FittedParams/FitInfo/fitcode'][:]
            site_lat = h5['Site/Latitude'][()]
            site_lon = h5['Site/Longitude'][()]
            site_alt = h5['Site/Altitude'][()]
            self.time = h5['Time/UnixTime'][:,0]
 
        data_check = np.array([chi2>0.1, chi2<10., np.isin(fitcode,[1,2,3,4])])
        # If ANY elements of data_check are FALSE, flag index as bad data
        bad_data = np.squeeze(np.any(data_check==False,axis=0,keepdims=True))
        self.dens[bad_data] = np.nan
        self.dens_err[bad_data] = np.nan

        stidx = np.argmin(np.abs(self.time-np.datetime64(starttime).astype('int')))
        etidx = np.argmin(np.abs(self.time-np.datetime64(endtime).astype('int')))
        self.tidx_rng = [stidx, etidx]

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


    def fit_profiles(self):

        self.chapman_coefficients = np.empty((self.tidx_rng[1]-self.tidx_rng[0],len(self.tri2),4))

        for tidx in range(self.tidx_rng[0], self.tidx_rng[1]):
            print('profile fit - ' + str(self.time[tidx].astype('datetime64[s]')))

            for i, clust_index in enumerate(self.tri2):

                # Select data points for each beam cluster
                a = self.alt[clust_index]
                d = self.dens[tidx,clust_index,:]
                dd = self.dens_err[tidx,clust_index,:]

                # Filter high altitude points with extremely low error and low Ne values - these bias the topside fit
                filt_data = ((a>400.*1000) & (dd<5.e10) & (d<5.e10))
                good_data = (np.isfinite(d) & np.isfinite(dd) & (dd!=0.) & ~filt_data)

                try:
                    coeffs, _ = curve_fit(chapman, a[good_data], d[good_data], sigma=dd[good_data], p0=[4.e11,300.*1000.,50.*1000.,100.*1000.], bounds=[[0.,0.,0.,0.],[np.inf,np.inf,np.inf,np.inf]], absolute_sigma=True)
                except RuntimeError:
                    coeffs = [np.nan, np.nan, np.nan, np.nan]

                self.chapman_coefficients[tidx-self.tidx_rng[0],i] = np.array(coeffs)


    def fit_2d(self):

        # Create basis function center arrays
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

        # Define boundary
        bound_az, bound_el = hav_new(self.cent_az*np.pi/180., self.cent_el*np.pi/180., np.arange(0., 360., 10.)*np.pi/180., np.pi/5)

        C = len(bound_el)

        self.X = np.empty((self.tidx_rng[1]-self.tidx_rng[0], 4, N+C))

        for m, chap_coeff in enumerate(self.chapman_coefficients):
            print('2D fit - ' + str(self.time[self.tidx_rng[0]+m].astype('datetime64[s]')))

            for n, (vobs, const) in enumerate(zip(chap_coeff.T, self.boundary_value)):

                # create constraints array
                constraints = np.array([[az, el, const] for az, el in zip(bound_az, bound_el)])

                # form inversion arrays
                a = np.zeros((N+C,N+C))
                b = np.zeros(N+C)

                a[:N,:N] = np.array([[2*np.sum(rbf.Phi(i,self.clust_az,self.clust_el)*rbf.Phi(j,self.clust_az,self.clust_el)) for i in range(N)] for j in range(N)])
                a[:N,N:] = np.array([[-rbf.Phi(j,r[0],r[1]) for r in constraints] for j in range(N)])
                a[N:,:N] = np.array([[rbf.Phi(i,r[0],r[1]) for i in range(N)] for r in constraints])
                b[:N] = np.array([2*np.sum(vobs*rbf.Phi(j,self.clust_az,self.clust_el)) for j in range(N)])
                b[N:] = np.array([r[2] for r in constraints])

                self.X[m,n,:] = np.linalg.solve(a,b)


    def save_output(self):

        with h5py.File(self.output_filename, 'w') as h5:
            h5.create_dataset('X', data=self.X)
            # h5.create_dataset('fov_cent', data=self.fov_cent)
            h5.create_dataset('cent_az', data=self.cent_az)
            h5.create_dataset('cent_el', data=self.cent_el)
            h5.create_dataset('caz', data=self.caz)
            h5.create_dataset('cel', data=self.cel)
            h5.create_dataset('time', data=self.time[self.tidx_rng[0]:self.tidx_rng[1]])
            h5.create_dataset('boundary_value', data=self.boundary_value)


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

        self.time = utime.astype('datetime64[s]')


    def grid_enu(self, targtime, xrng, yrng, zrng):

        tidx = np.argmin(np.abs(self.time-targtime))
        print(self.time[tidx])

        Xgrid, Ygrid, Zgrid = np.meshgrid(xrng, yrng, zrng)

        azgrid, elgrid, _ = pm.enu2aer(Xgrid, Ygrid, Zgrid, deg=False)

        a = np.sin((elgrid-self.cent_el*np.pi/180.)/2)**2 + np.cos(elgrid)*np.cos(self.cent_el*np.pi/180.)*np.sin((azgrid-self.cent_az*np.pi/180.)/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        out_of_bound = c>np.pi/5

        pgrid = list()
        for x1, bc in zip(self.X[tidx], self.boundary_value):
            pg = np.sum(np.array([x1[i]*self.rbf.Phi(i,azgrid, elgrid) for i in range(self.rbf.Nbasis)]), axis=0)
            pg[out_of_bound] = bc
            pgrid.append(pg)
        pgrid = np.array(pgrid)

        vgrid = chapman(Zgrid, pgrid[0], pgrid[1], pgrid[2], pgrid[3])

        return Xgrid, Ygrid, Zgrid, vgrid

    def point_enu(self, targtime, epnt, npnt, upnt):

        tidx = np.argmin(np.abs(self.time-targtime))

        azpnt, elpnt, _ = pm.enu2aer(epnt, npnt, upnt, deg=False)

        vpnt = self.calc_point_arr(tidx, azpnt, elpnt, upnt)

        return vpnt

    def point_geodetic(self, targtime, lat, lon, alt):

        tidx = np.argmin(np.abs(self.time-targtime))

        azpnt, elpnt, _ = pm.geodetic2aer(lat, lon, h, lat0, lon0, h0, deg=False)

        vpnt = self.calc_point_arr(tidx, azpnt, elpnt, alt)

        return vpnt
 
    def calc_point_arr(self, tidx, azpnt, elpnt, upnt):

        a = np.sin((elpnt-self.cent_el*np.pi/180.)/2)**2 + np.cos(elpnt)*np.cos(self.cent_el*np.pi/180.)*np.sin((azpnt-self.cent_az*np.pi/180.)/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        out_of_bound = c>np.pi/5

        pgrid = list()
        for x1, bc in zip(self.X[tidx], self.boundary_value):

            pg = np.sum(np.array([x1[i]*self.rbf.Phi(i, azpnt, elpnt) for i in range(self.rbf.Nbasis)]), axis=0)

            pg[out_of_bound] = bc
            pgrid.append(pg)
        pgrid = np.array(pgrid)

        vpnt = chapman(upnt, pgrid[0], pgrid[1], pgrid[2], pgrid[3])

        return vpnt


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
