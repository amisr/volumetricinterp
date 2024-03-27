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


def epstein(z, N0, z0, HB, HT):

    # From Themens et al., 2019; Eqn 7
    # Topside
    zp = (z-z0)/HT
    NeT = N0/np.cosh(zp)**2

    # Bottomside
    zp = (z-z0)/HB
    NeB = N0/np.cosh(zp)**2

    Ne = NeB.copy()
    Ne[z>=z0] = NeT[z>=z0]

    return Ne


def chapman(z, N0, z0, HB, HT):

    # From Schunk & Nagy, 2009; Eqn 11.57
    # Topside
    zp = (z-z0)/HT
    NeT = N0*np.exp(0.5*(1 - zp - np.exp(zp)))

    # Bottomside
    zp = (z-z0)/HB
    NeB = N0*np.exp(0.5*(1 - zp - np.exp(zp)))

    # Combine two profiles
    Ne = NeB.copy()
    Ne[z>=z0] = NeT[z>=z0]

    return Ne

profile_functions = {'chapman':chapman, 'epstein':epstein}

def haversine_new_pnt(az0, el0, b, d):
    # Calculate position of new point from a base point with a given bearing and great-circle distance
    # Makes use of Haversine formulation
    el = np.arcsin(np.sin(el0)*np.cos(d) + np.cos(el0)*np.sin(d)*np.cos(b))
    az = az0 + np.arctan2(np.sin(b)*np.sin(d)*np.cos(el0), np.cos(d)-np.sin(el0)*np.sin(el))
    return az, el


def haversine_dist(az0, el0, az, el):
    # Calculate great circle distance between two points
    # Makes use of Haversine formulation
    a = np.sin((el-el0)/2)**2 + np.cos(el)*np.cos(el0)*np.sin((az-az0)/2)**2
    s = 2*np.arcsin(np.sqrt(a))
    return s


class BasisFunctions(object):
    def __init__(self, caz, cel, rbf_form):
        self.caz = caz
        self.cel = cel
        self.Nbasis = len(cel)
        self.rbf = getattr(self, rbf_form)

    def Phi(self, i, az, el):
        s = haversine_dist(self.caz[i], self.cel[i], az, el)
        p = self.rbf(s)
        return p

    def evaluate(self, x, az, el):
        p = np.sum(np.array([x[i]*self.Phi(i, az, el) for i in range(self.Nbasis)]), axis=0)
        return p

    def square(self, r):
        return r**2

    def cubic(self, r):
        return r**3

    def log(self, r):
        return r*np.log(r)



class Interpolate(object):
    def __init__(self, config_file):
        self.read_config(config_file)
        self.profile = profile_functions[self.profile_function]

    def run(self):

        self.load_datafile()

        if self.starttime:
            # Find indices for start and end times
            stidx = np.argmin(np.abs(self.time-np.datetime64(self.starttime).astype('int')))
            etidx = np.argmin(np.abs(self.time-np.datetime64(self.endtime).astype('int')))
        else:
            # Otherwise run for all indexes
            stidx = 0
            etidx = len(self.time)

        self.cluster_beams()
        self.setup_basis()
        self.setup_boundary()

        X = list()
        for tidx in range(stidx, etidx):
            coeffs, errs = self.fit_profiles(tidx)
            x = self.fit_2d(coeffs, errs)
            X.append(x)
        self.X = np.array(X)

        self.save_output(stidx, etidx)


    def read_config(self, config_file):

        config = configparser.ConfigParser(converters={'list': lambda s: [float(i) for i in s.split(',')]})
        config.read(config_file)
        self.filename = config.get('DEFAULT', 'FILENAME')
        self.output_filename = config.get('DEFAULT', 'OUTPUTFILENAME')
        self.starttime = config.get('DEFAULT', 'STARTTIME', fallback=None)
        self.endtime = config.get('DEFAULT', 'ENDTIME', fallback=None)
        self.profile_function = config.get('BASIS', 'PROFILE_FUNCTION')
        self.rbf_form = config.get('BASIS', 'RBF_FORM')
        self.basis_cent_az = np.deg2rad(config.getfloat('BASIS', 'BASIS_CENT_AZ'))
        self.basis_cent_el = np.deg2rad(config.getfloat('BASIS', 'BASIS_CENT_EL'))
        self.basis_ring_radius = np.deg2rad(config.getlist('BASIS', 'BASIS_RING_RADIUS'))
        self.basis_ring_number = config.getlist('BASIS', 'BASIS_RING_NUMBER')
        self.boundary_params = config.getlist('BOUNDARY', 'BOUNDARY_PARAMS')
        self.boundary_radius = np.deg2rad(config.getfloat('BOUNDARY', 'BOUNDARY_RADIUS'))
        self.boundary_number = config.getint('BOUNDARY', 'BOUNDARY_NUMBER')

    def load_datafile(self):

        with h5py.File(self.filename, 'r') as h5:
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
 
        # Filter high altitude points with extremely low error and low Ne values - these bias the topside fit
        filt_data = ((self.alt>200.*1000) & (self.dens_err<1.e10) & (self.dens<1.e10))
        data_check = np.array([chi2>0.1, chi2<10., np.isin(fitcode,[1,2,3,4]), ~filt_data])
        # If ANY elements of data_check are FALSE, flag index as bad data
        bad_data = np.squeeze(np.any(data_check==False,axis=0,keepdims=True))
        self.dens[bad_data] = np.nan
        self.dens_err[bad_data] = np.nan

        self.site_coords = np.array([site_lat, site_lon, site_alt])

    def cluster_beams(self):

        r = np.cos(np.deg2rad(self.beamcode[:,2]))
        t = np.deg2rad(self.beamcode[:,1])
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


    def fit_profiles(self, tidx):

        profile_params = np.empty((len(self.tri2),4))
        profile_errors = np.empty((len(self.tri2),4))

        for i, clust_index in enumerate(self.tri2):

            # Select data points for each beam cluster
            a = self.alt[clust_index]
            d = self.dens[tidx,clust_index,:]
            dd = self.dens_err[tidx,clust_index,:]

            # Flag NaN data
            good_data = (np.isfinite(d) & np.isfinite(dd) & (dd!=0.))

            try:
                # NOTE: Use pcov optional output to estimate errors on these parameters and use them in the 2D fit
                params, cov = curve_fit(self.profile, a[good_data], d[good_data], sigma=dd[good_data], p0=self.boundary_params, bounds=[[0.,0.,0.,0.],[np.inf,np.inf,np.inf,np.inf]], absolute_sigma=True)
                #params, cov = curve_fit(chapman, a[good_data], d[good_data], sigma=dd[good_data], p0=self.boundary_params, bounds=[[0.,0.,0.,0.],[np.inf,np.inf,np.inf,np.inf]], absolute_sigma=True)
                errors = np.sqrt(np.diag(cov))
            except (ValueError, RuntimeError):
                params = [np.nan, np.nan, np.nan, np.nan]
                errors = [np.nan, np.nan, np.nan, np.nan]

            profile_params[i] = np.array(params)
            profile_errors[i] = np.array(errors)

        return profile_params, profile_errors


    def setup_basis(self):

        caz = [self.basis_cent_az]
        cel = [self.basis_cent_el]
        
        for i, (r, n) in enumerate(zip(self.basis_ring_radius, self.basis_ring_number)):
            circ_pnt, s = np.linspace(0., 2.*np.pi, num=int(n), endpoint=False, retstep=True)
            p = s/2 if i%2 else 0.
            circ_az, circ_el = haversine_new_pnt(self.basis_cent_az, self.basis_cent_el, circ_pnt+p, r)
            caz.extend(circ_az)
            cel.extend(circ_el)

        self.rbf = BasisFunctions(caz, cel, self.rbf_form)
        
        self.caz = caz
        self.cel = cel


    def setup_boundary(self):

        # Define boundary
        bound_circ = np.linspace(0., 2*np.pi, num=self.boundary_number, endpoint=False)
        self.boundary_az, self.boundary_el = haversine_new_pnt(self.basis_cent_az, self.basis_cent_el, bound_circ, self.boundary_radius)

    def fit_2d(self, profile_params, profile_errors):

        N = self.rbf.Nbasis
        C = self.boundary_number

        X = np.empty((4, N+C))

        # Loop over profile parameters
        for n, (vobs, eobs, const) in enumerate(zip(profile_params.T, profile_errors.T, self.boundary_params)):

            # create constraints array
            constraints = np.array([[az, el, const] for az, el in zip(self.boundary_az, self.boundary_el)])

            # form inversion arrays
            a = np.zeros((N+C,N+C))
            b = np.zeros(N+C)

            a[:N,:N] = np.array([[2*np.sum(self.rbf.Phi(i,self.clust_az,self.clust_el)*self.rbf.Phi(j,self.clust_az,self.clust_el)/eobs**2) for i in range(N)] for j in range(N)])
            a[:N,N:] = np.array([[-self.rbf.Phi(j,r[0],r[1]) for r in constraints] for j in range(N)])
            a[N:,:N] = np.array([[self.rbf.Phi(i,r[0],r[1]) for i in range(N)] for r in constraints])
            b[:N] = np.array([2*np.sum(vobs*self.rbf.Phi(j,self.clust_az,self.clust_el)/eobs**2) for j in range(N)])
            b[N:] = np.array([r[2] for r in constraints])

            X[n,:] = np.linalg.solve(a,b)

        return X


    def save_output(self, stidx, etidx):

        with h5py.File(self.output_filename, 'w') as h5:
            h5.create_dataset('X', data=self.X)
            h5.create_dataset('site_coords', data=self.site_coords)
            h5.create_dataset('cent_az', data=self.basis_cent_az)
            h5.create_dataset('cent_el', data=self.basis_cent_el)
            h5.create_dataset('caz', data=self.caz)
            h5.create_dataset('cel', data=self.cel)
            h5.create_dataset('time', data=self.time[stidx:etidx])
            h5.create_dataset('boundary', data=self.boundary_radius)
            h5.create_dataset('boundary_value', data=self.boundary_params)
 

class Reconstruct(object):
    def __init__(self, filename):
        # intialize with filename OR config file?
        # instead of saving RBF centers, save config option that generates them?

        self.load_file(filename)

        self.rbf = BasisFunctions(self.caz, self.cel)


    def load_file(self, filename):
        with h5py.File(filename, 'r') as h5:
            self.X = h5['X'][:]
            self.site_coords = h5['site_coords'][:]
            cent_az = h5['cent_az'][()]
            cent_el = h5['cent_el'][()]
            self.caz = h5['caz'][:]
            self.cel = h5['cel'][:]
            utime = h5['time'][:]
            boundary = h5['boundary'][()]
            self.boundary_value = h5['boundary_value'][:]

        self.time = utime.astype('datetime64[s]')
        self.cent_az = np.deg2rad(cent_az)
        self.cent_el = np.deg2rad(cent_el)
        self.boundary = np.deg2rad(boundary)


#    def reconstruct_at_time(time, x, y, z):
#        # find X at correct time
#        # reconstruct_rbf()
#        # chapman()
#        pass
#
#    def grid_at_all_times(x, y, z):
#        # run for all times
#        for x1 in X:
#            coeff = reconstruct_rbf(x, ax, el)
#            dens = chapman(coeffs, z)
#        pass


    
    def check_bounds(self, az, el):
        # get distance from general function
        #a = np.sin((el-self.cent_el)/2)**2 + np.cos(el)*np.cos(self.cent_el)*np.sin((az-self.cent_az)/2)**2
        #s = 2*np.arcsin(np.sqrt(a))

        s = rad_dist(self.cent_az, self.cent_el, az, el)
        return s>self.boundary


    def evaluate_density(self, targtime, azgrid, elgrid, zgrid):
        # time selection here or somewhere else?
        tidx = np.argmin(np.abs(self.time-targtime))

        #d = vi.rad_dist(np.deg2rad(interp.cent_az), np.deg2rad(interp.cent_el), azgrid, elgrid)
        out_of_bounds = self.check_bounds(azgrid, elgrid)
        
        coeff = [self.rbf.evaluate(x1, azgrid, elgrid) for x1 in self.X[tidx]]
        for i,c in enumerate(coeff):
            c[out_of_bounds] = interp.boundary_value[i]
        #print(coeff[0].shape, zgrid.shape)
        dgrid = chapman(zgrid, *coeff)
        #print(dgrid.shape)
        return dgrid


#    def generate_coeffs(self, X, azpnt, elpnt):
#        
#        chap_coeffs = list()
#        for x1 in X:
#            pg = np.sum(np.array([x1[i]*self.rbf.Phi(i, azpnt, elpnt) for i in range(self.rbf.Nbasis)]), axis=0)
#            chap_coeffs.append(pg)
#
#        return chap_coeffs
#
#    def set_exterior(self, chap_coeff, azpnt, elpnt):
#
#        out_of_bound = self.check_bounds(azpnt, elpnt)
#        for coeff, bc in zip(chap_coeffs, self.boundary_value):
#            coeff[out_of_bound] = bc
#        return chap_coeff
#
#
#
#
#    def grid_enu(self, targtime, xrng, yrng, zrng):
#
#        tidx = np.argmin(np.abs(self.time-targtime))
#        print(self.time[tidx])
#
#        Xgrid, Ygrid, Zgrid = np.meshgrid(xrng, yrng, zrng)
#
#        azgrid, elgrid, _ = pm.enu2aer(Xgrid, Ygrid, Zgrid, deg=False)
#
#        #a = np.sin((el-self.cel[i])/2)**2 + np.cos(el)*np.cos(self.cel[i])*np.sin((az-self.caz[i])/2)**2
#        #c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
#        #s = 2*np.arcsin(np.sqrt(a))
#       # a = np.sin((elgrid-self.cent_el*np.pi/180.)/2)**2 + np.cos(elgrid)*np.cos(self.cent_el*np.pi/180.)*np.sin((azgrid-self.cent_az*np.pi/180.)/2)**2
#       # c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
#
#        out_of_bound = self.check_bounds(azgrid, elgrid)
##        out_of_bound = c>np.pi/5
#
#        pgrid = list()
#        for x1, bc in zip(self.X[tidx], self.boundary_value):
#            pg = np.sum(np.array([x1[i]*self.rbf.Phi(i,azgrid, elgrid) for i in range(self.rbf.Nbasis)]), axis=0)
#            pg[out_of_bound] = bc
#            pgrid.append(pg)
#        pgrid = np.array(pgrid)
#
#        vgrid = chapman(Zgrid, pgrid[0], pgrid[1], pgrid[2], pgrid[3])
#
#        return Xgrid, Ygrid, Zgrid, vgrid
#
#    def point_enu(self, targtime, epnt, npnt, upnt):
#
#        tidx = np.argmin(np.abs(self.time-targtime))
#
#        azpnt, elpnt, _ = pm.enu2aer(epnt, npnt, upnt, deg=False)
#
#        vpnt = self.calc_point_arr(tidx, azpnt, elpnt, upnt)
#
#        return vpnt
#
#    def point_geodetic(self, targtime, lat, lon, alt):
#
#        tidx = np.argmin(np.abs(self.time-targtime))
#
#        azpnt, elpnt, _ = pm.geodetic2aer(lat, lon, alt, self.site_coords[0], self.site_coords[1], self.site_coords[2])
#
#        vpnt = self.calc_point_arr(tidx, np.deg2rad(azpnt), np.deg2rad(elpnt), alt)
#
#        return vpnt
# 
#    def calc_point_arr(self, tidx, azpnt, elpnt, upnt):
#
#        #a = np.sin((elpnt-self.cent_el*np.pi/180.)/2)**2 + np.cos(elpnt)*np.cos(self.cent_el*np.pi/180.)*np.sin((azpnt-self.cent_az*np.pi/180.)/2)**2
#        #c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
#        out_of_bound = self.check_bounds(azpnt, elpnt)
#
#        pgrid = list()
#        for x1, bc in zip(self.X[tidx], self.boundary_value):
#
#            pg = np.sum(np.array([x1[i]*self.rbf.Phi(i, azpnt, elpnt) for i in range(self.rbf.Nbasis)]), axis=0)
#
#            pg[out_of_bound] = bc
#            pgrid.append(pg)
#        pgrid = np.array(pgrid)
#
#        vpnt = chapman(upnt, pgrid[0], pgrid[1], pgrid[2], pgrid[3])
#
#        return vpnt



def main():
    config_file_help = 'Some help string'

    # Build the argument parser tree
    parser = argparse.ArgumentParser(description=config_file_help,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    arg = parser.add_argument('config',help='Configuration file for volumetric interpolation.')
    args = vars(parser.parse_args())

    interp = Interp4Model(args['config'])
    interp.run()

if __name__=='__main__':
    main()
