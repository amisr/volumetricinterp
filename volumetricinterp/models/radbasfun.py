# Model.py

import numpy as np
import scipy.special as sp
import scipy.integrate
import configparser
import pymap3d as pm
from scipy.spatial import ConvexHull, Delaunay

RE = 6371.2*1000.           # Earth Radius (m)

class Model(object):
    # TODO: update docstring
    """
    This class defines the 3D analytic model that is used to describe density and temperature within an AMISR FoV.

    Parameters:
        maxk: [int]
            number of basis functions used in the vertical direction
        maxl: [int]
            number of basis functions used in the horizontal direction
        cap_lim: [double]
            colatitude limit of the polar cap in radians
        C: Optional [ndarray(nbasis)]
            array of fit coefficients - must be nbasis long
        dC: Optional [ndarray(nbasis,nbasis)]
            covariance matrix for fit coefficients - must be nbasis X nbasis

    Attributes:
        maxk: number of basis functions used in the vertical direction
        maxl: number of basis functions used in the horizontal direction
        nbasis: total number of 3D basis functions used
        cap_lim: colatitude (in radians) of the edge of the "cap" used for spherical cap harmonics (defaults to 6 degrees)
        C: array of fit coefficients
        dC: covariance matrix of fit coefficients

    Methods:
        basis_numbers: returns k, l, m given a single input 3D basis index
        nu: returns v, the non-integer degree for spherical cap harmonics
        eval_basis: returns a matrix of all basis functions calcuated at all input points
        eval_grad_basis: returns a maxtix of the gradient of all basis fuctions calculated at all input points
        eval_model: returns parameter and gradient arrays for all input points
        Az: azimuthal component
        dAz: derivative of azimuthal component
        Kvm: constant Kvm

    Notes:
        - All methods EXCEPT for eval_model() can be called without specifying C or dC.
    """

    def __init__(self, config_file):
        self.read_config(config_file)


        lat, lon, alt = np.meshgrid(np.linspace(self.latrange[0],self.latrange[1],self.numgridpnt),np.linspace(self.lonrange[0],self.lonrange[1],self.numgridpnt),np.linspace(self.altrange[0],self.altrange[1],self.numgridpnt)*1000.)

        X, Y, Z = pm.geodetic2ecef(lat.flatten(), lon.flatten(), alt.flatten())

        self.centers = np.array([X,Y,Z]).T
        self.nbasis = self.centers.shape[0]

        self.eval_reg_matricies = {}


    def read_config(self, config_file):
        # read config file
        config = configparser.ConfigParser()
        config.read_file(config_file)

        self.latcp = config.getfloat('MODEL','LATCP')
        self.loncp = config.getfloat('MODEL','LONCP')
        self.eps = config.getfloat('MODEL','EPS')

        self.latrange = [float(i) for i in config.get('MODEL', 'LATRANGE').split(',')]
        self.lonrange = [float(i) for i in config.get('MODEL', 'LONRANGE').split(',')]
        self.altrange = [float(i) for i in config.get('MODEL', 'ALTRANGE').split(',')]

        self.numgridpnt = config.getint('MODEL','NUMGRIDPNT')




    def basis(self,gdlat,gdlon,gdalt):
        """
        Calculates a matrix of the basis functions evaluated at all input points

        Parameters:
            R: [ndarray(3,npoints)]
                array of input coordinates
                R = [[z coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T
        Returns:
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
        Notes:
            - Something clever could probably be done to not recalculate the full expression when incrimenting n does not result in a change in k, or similar.
                All the evaluations of special functions here make it one of the slowest parts of the code.
        """

        # z, theta, phi = self.transform_coord(gdlat.flatten(), gdlon.flatten(), gdalt.flatten())

        R = self.transform_coords(gdlat.flatten(),gdlon.flatten(),gdalt.flatten())
        A = []
        for n in range(self.nbasis):
            c = self.centers[n]
            r = np.linalg.norm(R-c[:,None], axis=0)
            A.append(np.exp(-r**2/self.eps**2))
        nax = list(np.arange(gdlat.ndim)+1)
        nax.append(0)
        A0 = np.transpose(np.array(A).reshape((-1,)+gdlat.shape), axes=nax)
        # return np.array(A).T
        return A0


    # def grad_basis(self,R):
    #     # TODO: Needs to be updated
    #     """
    #     Calculates a matrix of the gradient of basis functions evaluated at all input points
    #
    #     Parameters:
    #         R: [ndarray(3,npoints)]
    #             array of input coordinates
    #             R = [[z coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
    #             if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T
    #     Returns:
    #         A: [ndarray(npoints,nbasis,3)]
    #             array of gradient of basis functions evaluated at all input points
    #     Notes:
    #         - Something clever could probably be done to not recalculate the full expression when incrimenting n does not result in a change in k, or similar.
    #             All the evaluations of special functions here make it one of the slowest parts of the code.
    #     """
    #     z = R[0]
    #     theta = R[1]
    #     phi = R[2]
    #     Ag = []
    #     x = np.cos(theta)
    #     y = np.sin(theta)
    #     e = np.exp(-0.5*z)
    #     for n in range(self.nbasis):
    #         k, l, m = self.basis_numbers(n)
    #         v = self.nu(n)
    #         L0 = sp.eval_laguerre(k,z)
    #         L1 = sp.eval_genlaguerre(k-1,1,z)
    #         Pmv = sp.lpmv(m,v,x)
    #         Pmv1 = sp.lpmv(m,v+1,x)
    #         A = self.Az(v,m,phi)
    #         zhat = -0.5*e*(L0+2*L1)*Pmv*A*100./RE
    #         that = e*L0*(-(v+1)*x*Pmv+(v-m+1)*Pmv1)*A/(y*(z/100.+1)*RE)
    #         phat = e*L0*Pmv*self.dAz(v,m,phi)/(y*(z/100.+1)*RE)
    #         Ag.append([zhat,that,phat])
    #     # print np.shape(np.array(Ag).T)
    #     return np.array(Ag).T
    #
    #
    # # TODO: Need to actually impliment this with the correct derivatives - currently meaningless
    # def eval_omega(self):
    #     omega = np.zeros((self.nbasis,self.nbasis))
    #     for ni in range(self.nbasis):
    #         for nj in range(ni, self.nbasis):
    #             omega[ni,nj]
    #             O = self.omega_ij(ni, nj)
    #             omega[ni,nj] = O
    #             omega[nj,ni] = O
    #     return omega
    #
    # def omega_ij(self,ni,nj):
    #     ki, li, mi = self.basis_numbers(ni)
    #     kj, lj, mj = self.basis_numbers(nj)
    #     vi = self.nu(ni)
    #     vj = self.nu(nj)
    #
    #     z_int = lambda z: np.exp(-1*z)*sp.eval_laguerre(ki,z)*sp.eval_laguerre(kj,z)/z**2
    #     t_int = lambda t: 1/np.sin(t)**3*(-1*vi*(vi*np.cos(t)**2+vi+1)*sp.lpmv(mi,vi,np.cos(t))+vi*(vi+mi)*np.cos(t)*sp.lpmv(mi,vi-1,np.cos(t))+vi*(vi-mi+1)*np.cos(t)*sp.lpmv(mi,vi+1,np.cos(t)))*(-1*vj*(vj*np.cos(t)**2+vj+1)*sp.lpmv(mj,vj,np.cos(t))+vj*(vj+mj)*np.cos(t)*sp.lpmv(mj,vj-1,np.cos(t))+vj*(vj-mj+1)*np.cos(t)*sp.lpmv(mj,vj+1,np.cos(t)))
    #     p_int = lambda p: self.Az(vi,mi,p)*self.Az(vj,mj,p)
    #
    #     Iz = scipy.integrate.quad(z_int, 0., self.max_z_int)
    #     It = scipy.integrate.quad(t_int, 0., self.cap_lim)
    #     Ip = scipy.integrate.quad(p_int, 0., 2*np.pi)
    #     O = Iz[0]*It[0]*Ip[0]
    #     return O
    #
    #
    # def eval_psi(self):
    #     psi = np.zeros((self.nbasis,self.nbasis))
    #     for ni in range(self.nbasis):
    #         for nj in range(ni, self.nbasis):
    #             P = self.psi_ij(ni, nj)
    #             psi[ni,nj] = P
    #             psi[nj,ni] = P
    #     return psi
    #
    # def psi_ij(self, ni, nj):
    #     ki, li, mi = self.basis_numbers(ni)
    #     kj, lj, mj = self.basis_numbers(nj)
    #     vi = self.nu(ni)
    #     vj = self.nu(nj)
    #
    #     z_int = lambda z: np.exp(-1*z)*sp.eval_laguerre(ki,z)*sp.eval_laguerre(kj,z)*z**2
    #     t_int = lambda t: sp.lpmv(mi,vi,np.cos(t))*sp.lpmv(mj,vj,np.cos(t))*np.sin(t)
    #     p_int = lambda p: self.Az(vi,mi,p)*self.Az(vj,mj,p)
    #
    #     Iz = scipy.integrate.quad(z_int, 0., self.max_z_int)
    #     It = scipy.integrate.quad(t_int, 0., self.cap_lim)
    #     Ip = scipy.integrate.quad(p_int, 0., 2*np.pi)
    #
    #     P = Iz[0]*It[0]*Ip[0]
    #     return P
    #
    # def eval_tau(self, reg_func):
    #     tau = np.zeros((self.nbasis,1))
    #     for ni in range(self.nbasis):
    #         tau[ni] = self.tau_i(ni, reg_func)
    #     return tau
    #
    # def tau_i(self, n, reg_func):
    #     k, l, m = self.basis_numbers(n)
    #     v = self.nu(n)
    #
    #     z_int = lambda z: np.exp(-0.5*z)*sp.eval_laguerre(k,z)*reg_func(z)*z**2
    #     t_int = lambda t: sp.lpmv(m,v,np.cos(t))*np.sin(t)
    #     p_int = lambda p: self.Az(v,m,p)
    #
    #     Iz = scipy.integrate.quad(z_int, 0., self.max_z_int)
    #     It = scipy.integrate.quad(t_int, 0., self.cap_lim)
    #     Ip = scipy.integrate.quad(p_int, 0., 2*np.pi)
    #     T = Iz[0]*It[0]*Ip[0]
    #     return T




    def transform_coords(self,lat,lon,alt):
        """
        Transform from spherical coordinates to something friendlier for calculating the basis fit.
        This involves a rotation so that the data is centered around the north pole and a trasformation
         of the radial component such that z = 100*(r/RE-1).

        Parameters:
            R0: [ndarray(3,npoints)]
                array of input points in geocentric coordinates
                R = [[r coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T
        Returns:
            R_trans: [ndarray(3,npoints)]
                array of input points transformed into model coordinates
                R_trans = [[z coordinates],[theta coordinates (rad)],[phi coordinates (rad)]]
            cp: [ndarray(2)]
                center point of the input coordinates R0
        Notes:

        """

        x, y, z = pm.geodetic2ecef(lat, lon, alt)
        R_trans = np.array([x, y, z])

        return R_trans



    # def inverse_transform(self,R0,vec):
    #     """
    #     Inverse transformation to recover the correct vector components at their original position after
    #      calling eval_model().  This is primarially nessisary to get the gradients correct.
    #
    #     Parameters:
    #         R0: [ndarray(3,npoints)]
    #             array of points in model coordinates corresponding to the location of each vector in vec
    #         vec: [ndarray(npoints,3)]
    #             array of vectors in model coordinates
    #     Returns:
    #         vec_rot: [ndarray(npoints,3)]
    #             array of vectors rotated back to original geocenteric coordinates
    #     """
    #
    #     phi0 = self.cp[1]
    #     theta0 = -1.*self.cp[0]
    #
    #     k = np.array([np.cos(phi0+np.pi/2.),np.sin(phi0+np.pi/2.),0.])
    #
    #     rx, ry, rz = cc.spherical_to_cartesian((R0[0]/100.+1.)*RE,R0[1],R0[2])
    #     Rc = np.array([rx,ry,rz])
    #     vx, vy, vz = cc.vector_spherical_to_cartesian(vec.T[0],vec.T[1],vec.T[2],(R0[0]/100.+1.)*RE,R0[1],R0[2])
    #     vc = np.array([vx,vy,vz])
    #
    #     rr = np.array([R*np.cos(theta0)+np.cross(k,R)*np.sin(theta0)+k*np.dot(k,R)*(1-np.cos(theta0)) for R in Rc.T]).T
    #     vr = np.array([v*np.cos(theta0)+np.cross(k,v)*np.sin(theta0)+k*np.dot(k,v)*(1-np.cos(theta0)) for v in vc.T]).T
    #     vr, vt, vp = cc.vector_cartesian_to_spherical(vr[0],vr[1],vr[2],rr[0],rr[1],rr[2])
    #
    #     vec_rot = np.array([vr,vt,vp]).T
    #
    #     return vec_rot
