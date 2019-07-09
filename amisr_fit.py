# amisr_fit.py


import os 
localpath = '/Volumes/AMISR_PROCESSED'
wdir = os.path.dirname(os.path.realpath(__file__))
dbname = 'RawData_Folders_Exps_Times_by_Radar.h5'

import numpy as np
import scipy
import scipy.integrate
import scipy.special as sp
from scipy.spatial import ConvexHull
import datetime as dt
import coord_convert as cc
import processed_file_list as pfl
import os
import tables

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D



RE = 6371.2*1000.           # Earth Radius (m)	

# if fitting, these parameters should be pulled from a config file
# if evaluating, these parameters should come from coefficient file
KMAX = 4
LMAX = 6
CAPLIMIT = 6.*np.pi/180.
REGULARIZATION_METHOD = ['0thorder']
REGULARIZATION_PARAMETER_METHOD = 'chi2'

PARAMETER_NAME = 'Electron Density'
MAX_Z_INT = np.inf
PARAMETER_RANGE = [0, 3e11]
PARAMETER_UNITS = 'm$^-3$'

# year = 2016
# month = 12
# day = 27
# hour = 22
# minute = 0

# date = dt.datetime(year,month,day)

FILENAME = '/home/jovyan/mount/data/RISR-N/20171119.001_lp_1min-fitcal.h5'
radar = 'RISR-N'
code = 'lp'






class Model(object):
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

    def __init__(self,maxk,maxl,cap_lim=6.*np.pi/180.,C=None,dC=None):
        self.maxk = maxk
        self.maxl = maxl
        self.nbasis = self.maxk*self.maxl**2
        self.cap_lim = cap_lim
        if C is not None:
            self.C = C
        if dC is not None:
            self.dC = dC

    def basis_numbers(self,n):
        """
        Converts a single 3D index number into 3 individual indexes for the radial, latitudinal, and azimulthal components

        Parameters:
            n: [int]
                single 3D index number

        Returns:
            k: [int]
                radial index number corresponding to n
            l: [int]
                latitudinal index number corresponding to n
            m: [int]
                azimuthal index number corresponding to n
        """
        k = n/(self.maxl**2)
        r = n%(self.maxl**2)
        l = np.floor(np.sqrt(r))
        m = r-l*(l+1)
        return k, l, m

    def nu(self,n):
        """
        Returns the non-integer order of the spherical cap harmonics given a 3D index number
        This is calculated using the approximation given in Thebault et al., 2006.

        Parameters:
            n: [int]
                single 3D index number
        Returns:
            v: [double]
                non-integer degree for the spherical cap harmonics
        """
        k, l, m = self.basis_numbers(n)
        v = (2*l+0.5)*np.pi/(2*self.cap_lim)-0.5
        return v


    def eval_basis(self,R):
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
        z = R[0]
        theta = R[1]
        phi = R[2]
        A = []
        for n in range(self.nbasis):
            k, l, m = self.basis_numbers(n)
            v = self.nu(n)
            A.append(np.exp(-0.5*z)*sp.eval_laguerre(k,z)*self.Az(v,m,phi)*sp.lpmv(m,v,np.cos(theta)))
        return np.array(A).T


    def eval_grad_basis(self,R):
        """
        Calculates a matrix of the gradient of basis functions evaluated at all input points

        Parameters:
            R: [ndarray(3,npoints)]
                array of input coordinates
                R = [[z coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T
        Returns:
            A: [ndarray(npoints,nbasis,3)]
                array of gradient of basis functions evaluated at all input points
        Notes:
            - Something clever could probably be done to not recalculate the full expression when incrimenting n does not result in a change in k, or similar.
                All the evaluations of special functions here make it one of the slowest parts of the code.
        """
        z = R[0]
        theta = R[1]
        phi = R[2]
        Ag = []
        x = np.cos(theta)
        y = np.sin(theta)
        e = np.exp(-0.5*z)
        for n in range(self.nbasis):
            k, l, m = self.basis_numbers(n)
            v = self.nu(n)
            L0 = sp.eval_laguerre(k,z)
            L1 = sp.eval_genlaguerre(k-1,1,z)
            Pmv = sp.lpmv(m,v,x)
            Pmv1 = sp.lpmv(m,v+1,x)
            A = self.Az(v,m,phi)
            zhat = -0.5*e*(L0+2*L1)*Pmv*A*100./RE
            that = e*L0*(-(v+1)*x*Pmv+(v-m+1)*Pmv1)*A/(y*(z/100.+1)*RE)
            phat = e*L0*Pmv*self.dAz(v,m,phi)/(y*(z/100.+1)*RE)
            Ag.append([zhat,that,phat])
        # print np.shape(np.array(Ag).T)
        return np.array(Ag).T

        


    def eval_model(self,R,calcgrad=True,calcerr=False,verbose=False):
        """
        Evaluate the density and gradients at the points in R given the coefficients C.
         If the covarience matrix, dC, is provided, the errors in the density and gradients will be calculated.  If not,
         just the density and gradient vectors will be returned by default.

        Parameters:
            R: [ndarray(3,npoints)]
                array of input coordinates
                R = [[z coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T
            calcgrad: [bool]
                indicates if gradients should be calculated
                True (default): gradients WILL be calculated
                False: gradients WILL NOT be calculated
                Setting calcgrad=False if gradients are not required may improve efficiency
            calcerr: [bool]
                indicates if errors on parameters and gradients should be calculated
                True: errors WILL be calculated
                False (default): errors WILL NOT be calculated
            verbose: [bool]
                indicates if function should be run in verbose mode
                This prints a warning if dC is not specified and the errors will not be calculated.
                True: verbose mode is ON
                False (default): verbose mode is OFF
        Returns:
            out: [dict]
                dictionary containing calculated parameter, gradient, and error arrays, as appropriate
                vaild keys:
                    'param': parameter
                    'grad': gradient (if calcgrad=True)
                    'err': error on parameter (if calcerr=True)
                    'gerr': error on gradient (if calcgrad=True AND calcerr=True)
        Notes:
            - A rough framework for error handling has been included in this code, but it has not been used often.
                The method needs to be validated still and there are probably errors in the code.
        """
        if self.C is None:
            print 'WARNING: C not specified in Model!'

        out = {}
        A = self.eval_basis(R)
        parameter = np.reshape(np.dot(A,self.C),np.shape(A)[0])
        out['param'] = parameter

        if calcgrad:
            Ag = self.eval_grad_basis(R)
            gradient = np.reshape(np.tensordot(Ag,self.C,axes=1),(np.shape(Ag)[0],np.shape(Ag)[1]))
            out['grad'] = gradient

        if calcerr:
            if self.dC is None:
                if verbose:
                    print 'Covariance matrix not provided. Errors will not be calculated.'
            error = np.diag(np.squeeze(np.dot(A,np.dot(self.dC,A.T))))
            out['err'] = error

            if calcgrad:
                gradmat = np.tensordot(Ag,np.tensordot(self.dC,Ag.T,axes=1),axes=1)
                graderr = []
                for i in range(np.shape(gradmat)[0]):
                    graderr.append(np.diag(gradmat[i,:,:,i]))
                graderr = np.array(graderr)
                out['gerr'] = graderr
        return out

        
    def Az(self,v,m,phi):
        """
        Evaluates the azimuthal function

        Parameters:
            v: [double]
                non-integer degree of spherical cap harmonics
            m: [int]
                order of spherical cap harmonics
            phi: [ndarray]
                array of phi values (radians)
        Returns:
            az: [ndarray]
                evaluated azimuthal function at all values of phi 
        """
        if m < 0:
            return self.Kvm(v,abs(m))*np.sin(abs(m)*phi)
        else:
            return self.Kvm(v,abs(m))*np.cos(abs(m)*phi)


    def dAz(self,v,m,phi):
        """
        Evaluates the derivative of the azimuthal function

        Parameters:
            v: [double]
                non-integer degree of spherical cap harmonics
            m: [int]
                order of spherical cap harmonics
            phi: [ndarray]
                array of phi values (radians)
        Returns:
            daz: [ndarray]
                evaluated derivative of the azimuthal function at all values of phi 
        """
        if m < 0:
            return abs(m)*self.Kvm(v,abs(m))*np.cos(abs(m)*phi)
        else:
            return -1*m*self.Kvm(v,abs(m))*np.sin(abs(m)*phi)


    def Kvm(self,v,m):
        """
        Evaluates the constant Kvm associated with spherical harmonics

        Parameters:
            v: [double]
                non-integer degree of spherical cap harmonics
            m: [int]
                order of spherical cap harmonics
        Returns:
            Kvm: [double]
                constant Kvm
        """
        Kvm = np.sqrt((2*v+1)/(4*np.pi)*sp.gamma(float(v-m+1))/sp.gamma(float(v+m+1)))
        if m != 0:
            Kvm = Kvm*np.sqrt(2)
        return Kvm



class EvalParam(Model):
    """
    This class evaluates the 3D analytic model that is used to describe density and temperature within an AMISR FoV.
    It handles a lot of the nitty-gritty coordinate transformations and loading coefficients from file that are nessisary before the model is evaluated.

    Parameters:
        datetime: [datetime object]
            date and time to evaluate the model at
        radar: [str]
            radar to evaluate model for (valid options are 'RISR-N', 'RISR-C', or 'PFISR')
        code: [str]
            Long-pulse ('lp') or Alternating Code ('ac')
        param: [AMISR_param object]
            parameter to evaluate - either density or temperature
        timetol: [double]
            tolerance for difference in time between the requested datetime and actual times where data's available (seconds)

    Atributes:
        datetime: time requested for this particular model instance
        radar: radar for this particular model instance
        code: radar code (LP vs AC)
        param: parameter for this model instance
        timetol: allowable variation from datetime in seconds (default is 60s)
        t: actual time from the data file of this model instance
        cp: center point ([lat, lon] in radians) of the data/location of the pole for the speherical cap
        hv: list of verticies for the convex hull

    Methods:
        loadh5: loads coefficents from a saved hdf5 file
        getparam: fully sets up and evaluates the model from input points
        transform_coords: transforms input cordinates so they can be handled by the model
        inverse_transform: transforms gradients from model coordinates back to input coordinates
        compute_hull: computes the convex hull that defines where model is valid
        check_hull: checks if the input coordinates are within the convex hull
    """
    def __init__(self,datetime=None,radar=None,code=None,param=None,timetol=60.,timeinterp=False):
        self.datetime = datetime
        self.radar = radar
        self.code = code
        self.param = param
        self.timetol = timetol
        self.timeinterp = timeinterp

        try:
            self.loadh5()
        except Exception as e:
            print e
            # print 'WARNING: {:04d}{:02d}{:02d}_{}_{}.h5 does not exist! A valid coefficient file must be loaded.'.format(self.datetime.year,self.datetime.month,self.datetime.day,self.radar,self.param.key)


    def loadh5(self,filename=None,raw=False):
        """
        Loads coefficients from a saved hdf5 file based on the date, radar, and param attributes

        Parameters:
            filename: Optional [str]
                file to load
                default file is ./Coefficients/YYYMMDD_RADAR_PARAM.h5
            raw: Optional [bool]
                flag to indicate if the raw data should be loaded or not
                default is False (raw data will NOT be loaded)
        Notes:
            - This saves coefficients and other parameters nessisary for the model as atributes of the class.
        """
        if filename is not None:
            cfilename = filename
        else:
            cfilename = wdir+'/Coefficients/{:04d}{:02d}{:02d}_{}_{}.h5'.format(self.datetime.year,self.datetime.month,self.datetime.day,self.radar,self.param.key)

        # print cfilename

        # hdir = '/UT'+str(self.datetime.hour).zfill(2)
        # if raw:
        #     data = io_utils.read_partial_h5file(cfilename,[hdir,hdir+'/Coeffs',hdir+'/FitParams',hdir+'/RawData'])
        # else:
        #     data = io_utils.read_partial_h5file(cfilename,[hdir,hdir+'/Coeffs',hdir+'/FitParams'])
        # utime = data[hdir]['UnixTime']
        # Coeffs = data[hdir+'/Coeffs']['C']
        # Covariance = data[hdir+'/Coeffs']['dC']
        # chi2 = data[hdir+'/FitParams']['chi2']
        # maxk = data[hdir+'/FitParams']['kmax']
        # maxl = data[hdir+'/FitParams']['lmax']
        # cap_lim = data[hdir+'/FitParams']['cap_lim']
        # cent_point = data[hdir+'/FitParams']['center_point']
        # hull_v = data[hdir+'/FitParams']['hull_vertices']
        # if raw:
        #     raw_coords = data[hdir+'/RawData']['coordinates']
        #     raw_data = data[hdir+'/RawData']['data']
        #     raw_error = data[hdir+'/RawData']['error']


        targtime = (self.datetime-dt.datetime(1970,1,1)).total_seconds()

        with tables.open_file(cfilename, 'r') as h5file:
            utime = h5file.get_node('/UnixTime')
            Coeffs = h5file.get_node('/Coeffs/C')
            Covariance = h5file.get_node('/Coeffs/dC')
            # chi2 = h5file.get_node(hdir+'/FitParams/chi2')
            maxk = h5file.get_node('/FitParams/kmax')
            maxl = h5file.get_node('/FitParams/lmax')
            cap_lim = h5file.get_node('/FitParams/cap_lim')
            cent_point = h5file.get_node('/FitParams/center_point')
            hull_v = h5file.get_node('/FitParams/hull_vertices')
            if raw:
                raw_coords = h5file.get_node('/RawData/coordinates')
                raw_data = h5file.get_node('/RawData/data')
                raw_error = h5file.get_node('/RawData/error')
                raw_filename = h5file.get_node('/RawData/filename')

            utime = utime.read()

            if self.timeinterp:
                midtime = np.array([(t[0]+t[1])/2. for t in utime])
                time = [dt.datetime.utcfromtimestamp(t) for t in midtime]
                rec = np.where((targtime>=midtime[:-1]) & (targtime<midtime[1:]))[0]
                if rec.size == 0:
                    raise ValueError('Requested time not included in {}'.format(cfilename))
                rec0 = rec[0]
                rec1 = rec[0] + 1
            else:
                utime = np.array(utime)
                time = [dt.datetime.utcfromtimestamp(t[0]) for t in utime]
                rec = np.where((targtime >= utime[:,0]) & (targtime < utime[:,1]))[0]
                if rec.size == 0:
                    raise ValueError('Requested time not included in {}'.format(cfilename))
                rec0 = rec[0]

            # print rec0, rec1

            if raw:
                self.rR = raw_coords[rec0]
                self.rd = raw_data[rec0]
                self.re = raw_error[rec0]
                self.rfn = raw_filename[rec0]


            self.maxk = maxk.read()
            self.maxl = maxl.read()
            self.nbasis = self.maxk*self.maxl**2
            self.cap_lim = cap_lim.read()

            self.t = time[rec0]
            self.C = np.array(Coeffs[rec0])
            self.dC = np.array(Covariance[rec0])
            self.cp = cent_point[rec0]
            self.hv = hull_v[rec0]

            if self.timeinterp:
                time0 = (time[rec0]-dt.datetime(1970,1,1)).total_seconds()
                time1 = (time[rec1]-dt.datetime(1970,1,1)).total_seconds()

                C0 = np.array(Coeffs[rec0])
                C1 = np.array(Coeffs[rec1])
                dC0 = np.array(Covariance[rec0])
                dC1 = np.array(Covariance[rec1])

                self.C = (targtime-time0)/(time1-time0)*(C1-C0) + C0
                self.dC = (targtime-time0)/(time1-time0)*(dC1-dC0) + dC0





    def getparam(self,R0,calcgrad=True,calcerr=False):
        """
        Fully calculates parameters and their gradients given input coordinates and a time.
        This is the main function that is used to retrieve reconstructed parameters.

        Parameters:
            R0: [ndarray(3,npoints)]
                array of input points in geocentric coordinates
                R = [[r coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T
             calcgrad: [bool]
                indicates if gradients should be calculated
                True (default): gradients WILL be calculated
                False: gradients WILL NOT be calculated
                Setting calcgrad=False if gradients are not required may improve efficiency
            calcerr: [bool]
                indicates if errors on parameters and gradients should be calculated
                True: errors WILL be calculated
                False (default): errors WILL NOT be calculated
       Returns:
            P: [ndarray(npoints)]
                array of the output parameter calculated at all input points
            dP: [ndarray(npoints,3)]
                array of the gradient of the output parameter calculated at all input points
                if calcgrad=False, dP is an array of NAN
        """


        check = self.check_hull(R0)
        R, __ = self.transform_coord(R0)

        out = self.eval_model(R,calcgrad=calcgrad,calcerr=calcerr)
        parameter = out['param']
        parameter[~check] = np.nan
        P = parameter
        dP = np.full((R0.shape[1],4),np.nan)
        if calcgrad:
            gradient = out['grad']
            gradient = self.inverse_transform(R,gradient)
            gradient[~check] = [np.nan,np.nan,np.nan]
            dP = np.array([gradient[:,0],gradient[:,1],gradient[:,2],np.zeros(len(parameter))]).T

        if calcerr:
            err = out['err']
            err[~check] = np.nan
            if calcgrad:
                graderr = out['gerr']
                graderr = self.inverse_transform(R,graderr,self.cp)
                graderr[~check] = [np.nan,np.nan,np.nan]

        return P, dP
    
    
    def transform_coord(self,R0):
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


        try:
            phi0 = self.cp[1]
            theta0 = self.cp[0]
        except:
            phi0 = np.average(R0[2])
            theta0 = -1*np.average(R0[1])
            self.cp = [theta0,phi0]


        k = np.array([np.cos(phi0+np.pi/2.),np.sin(phi0+np.pi/2.),0.])

        x, y, z = cc.spherical_to_cartesian(R0[0],R0[1],R0[2])
        Rp = np.array([x,y,z])
        Rr = np.array([R*np.cos(theta0)+np.cross(k,R)*np.sin(theta0)+k*np.dot(k,R)*(1-np.cos(theta0)) for R in Rp.T]).T
        r, t, p = cc.cartesian_to_spherical(Rr[0],Rr[1],Rr[2])
        R_trans = np.array([100*(r/RE-1),t,p])

        return R_trans, self.cp



    def inverse_transform(self,R0,vec):
        """
        Inverse transformation to recover the correct vector components at their original position after
         calling eval_model().  This is primarially nessisary to get the gradients correct.

        Parameters:
            R0: [ndarray(3,npoints)]
                array of points in model coordinates corresponding to the location of each vector in vec
            vec: [ndarray(npoints,3)]
                array of vectors in model coordinates
        Returns:
            vec_rot: [ndarray(npoints,3)]
                array of vectors rotated back to original geocenteric coordinates
        """

        phi0 = self.cp[1]
        theta0 = -1.*self.cp[0]

        k = np.array([np.cos(phi0+np.pi/2.),np.sin(phi0+np.pi/2.),0.])

        rx, ry, rz = cc.spherical_to_cartesian((R0[0]/100.+1.)*RE,R0[1],R0[2])
        Rc = np.array([rx,ry,rz])
        vx, vy, vz = cc.vector_spherical_to_cartesian(vec.T[0],vec.T[1],vec.T[2],(R0[0]/100.+1.)*RE,R0[1],R0[2])
        vc = np.array([vx,vy,vz])

        rr = np.array([R*np.cos(theta0)+np.cross(k,R)*np.sin(theta0)+k*np.dot(k,R)*(1-np.cos(theta0)) for R in Rc.T]).T
        vr = np.array([v*np.cos(theta0)+np.cross(k,v)*np.sin(theta0)+k*np.dot(k,v)*(1-np.cos(theta0)) for v in vc.T]).T
        vr, vt, vp = cc.vector_cartesian_to_spherical(vr[0],vr[1],vr[2],rr[0],rr[1],rr[2])

        vec_rot = np.array([vr,vt,vp]).T

        return vec_rot


    def compute_hull(self,R0):
        """
        Compute the convex hull that contains the original data.  This is nessisary to check if points requested from the
         model are within the vaild range of where the data constrains the model.

        Parameters:
            R0: [ndarray(3,npoints)]
                array of input points in geocentric coordinates
                R = [[r coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T
        Returns:
            hv: [ndarray(3,nverticies)]
                array of the coordinates of the verticies of the convex hull
        Notes:
            - hv is also saved as an attribute of the class
        """

        x, y, z = cc.spherical_to_cartesian(R0[0],R0[1],R0[2])
        R_cart = np.array([x,y,z]).T

        chull = ConvexHull(R_cart)
        vert = R_cart[chull.vertices]

        r, t, p = cc.cartesian_to_spherical(vert.T[0],vert.T[1],vert.T[2])
        vertices = np.array([r,t,p]).T

        self.hv = np.array(vertices).T
        return self.hv


    def check_hull(self,R0):
        """
        Check if the input points R0 are within the convex hull of the original data.

        Parameters:
            R0: [ndarray(3,npoints)]
                array of input points in geocentric coordinates
                R = [[r coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T

        """
        x, y, z = cc.spherical_to_cartesian(self.hv.T[0],self.hv.T[1],self.hv.T[2])
        vert_cart = np.array([x,y,z]).T

        hull = ConvexHull(vert_cart)
        check = []
        for R in R0.T:
            value = False

            x, y, z = cc.spherical_to_cartesian(R[0],R[1],R[2])

            pnt = np.array([[x,y,z]])
            pnts = np.append(vert_cart,pnt,axis=0)
            nh = ConvexHull(pnts)
            if np.array_equal(hull.vertices,nh.vertices):
                value = True
            check.append(value)
        return np.array(check)








class Fit(EvalParam):
    """
    This class performs the least-squares fit of the data to the 3D analytic model to find the coefficient vector for the model.
    It also handles calculating regularization matricies and parameters if nessisary.

    Parameters:
        date: [date object]
            date to evaluate the model at
        radar: [str]
            radar to evaluate model for (valid options are 'RISR-N', 'RISR-C', or 'PFISR')
        code: [str]
            Long-pulse ('lp') or Alternating Code ('ac')
        param: [AMISR_param object]
            parameter to evaluate - either density or temperature

    Atributes:
        date: date the model is fit for
        regularization_list: list of regularization methods to be used for the fit
        time: list of datetime objects corresponding to each event fit
        Coeffs: list of coefficient vectors corresponding to each event fit
        Covariance: list of covariance matrices corresponding to each event fit
        chi_sq: list of chi squared (goodness of fit test) corresponding to each event fit
        cent_point: list of the center points corresponding ot each event fit
        hull_v: list of hull verticies corresponding to each event fit
        raw_coords: list of the coordinates for the raw data corresponding to each event fit
        raw_data: list of the data value for the raw data corresponding to each event fit
        raw_error: list of the error vlue for the raw data corresponding to each event fit
        raw_filename: list of the raw data file names corresponding to each event fit
        raw_index: list of the index within each raw data file correspoinding ot each event fit

    Methods:
        get_ns: calculating the 2 n indicies for a NxN array from a 1D index q
        eval_omega: evaluate Omega, the curvature regularization matrix
        parallelize_omega: evaluates a single term in the Omega matrix, useful for parallelizing the loop
        omega_z_integrand: evaluates the z integrand for a term in the Omega matrix
        omega_t_integrand: evaluates the theta integrand for a term in the Omega matrix
        omega_p_integrand: evaluates the phi integrand for a term in the Omega matrix
        eval_Psi: evaluate Psi matrix, used for 0th order regularization
        parallelize_psi: evaluates a single term in the Psi maxtrix, useful for parallelizing the loop
        psi_z_integrand: evaluates the z integrand for a term in the Psi matrix
        psi_t_integrand: evaluates the theta integrand for a term in the Psi matrix
        psi_p_integrand: evaluates the phi integrand for a term in the Psi matrix
        eval_Tau: evaluates Tau vector, used for 0th order regularization
        parallelize_tau: evaluates a single term in the Tau vector, useful for parallelizing the loop
        tau_z_integrand: evaluates the z integrand for a term in the Tau vector
        tau_t_integrand: evaluates the theta integrand for a term in the Tau vector
        tau_p_integrand: evaluates the phi integrand for a term in the Tau vector
        find_reg_params: finds the regularization parameters
        chi2: finds the regularization parameter using the chi2-nu method
        chi2objfunct: objective function for the chi2-nu method
        gcv: finds the regularization parameter using generalized cross validation
        gcvobjfunct: the objective function for generalized cross validation
        manual: finds the regularization parameter via values manually hardcoded in function
        prompt: finds the regularization parameter via comand line prompts for user input
        eval_C: evaluates the coefficent vector and covariance matrix
        fit: performs fits to the 3D analytic model for data from a series of events
        saveh5: save the results of fit() to an output hdf5 file
        validate: plot the raw data and model fit to check for a good visual agreement

    """

    # def __init__(self,date=None,radar=None,code=None,param=None):
    def __init__(self,param=None):
#         self.date = date
        self.radar = radar
#         self.code = code
        self.param = param


    def generate_eventlist(self,starttime=None,endtime=None):
        """
        Generates an eventlist for a single day that includes all events from that day, regardless of the radar mode that
         was run.

        Returns:
            eventlist: [dict]
                list of dictionaries containing the timestamp, file name, radar mode, and index within the file for a 
                particular event
                vaild keys:
                    'time': timestamp (datetime object)
                    'filename': file name including full file path
                    'mode': radar mode
                    'index': index of this particular event within the file
        """
        if starttime is None:
            starttime = self.date
            endtime = self.date+dt.timedelta(hours=24)
        elif endtime is None:
            endtime = starttime+dt.timedelta(hours=1)
  

        filelist = pfl.file_list(self.date,radars=[self.radar],criteria=['lp','1min','fitcal'])

        eventlist = []
        for filename in filelist:
            experiment = os.path.basename(filename)
            with tables.open_file(dbname,'r') as h5file:
                tn = h5file.get_node('/Radars/{}'.format(self.radar.replace('-','')))
                en = h5file.get_node('/ExpNames/Names')
                ny = tn[:]['nyear']
                nm = tn[:]['nmonth']
                nd = tn[:]['nday']
                ns = tn[:]['nset']

                # for experiment in self.experiment_list:
                year = int(experiment[0:4])
                month = int(experiment[4:6])
                day = int(experiment[6:8])
                num = int(experiment[9:12])
                index = np.where((ny==year) & (nm==month) & (nd==day) & (ns==num))[0]
                if index.size == 0:
                    experiment['mode'] = '0000'
                else:
                    i = index[0]
                    eid = tn[i]['nExpId']
                    mode = en[eid][0]
                    print mode



            with tables.open_file(filename, 'r') as h5file:
                utime = h5file.get_node('/Time/UnixTime')
                utime = utime.read()
            for i,t in enumerate(utime):
                dh = (float(t[0])+float(t[1]))/2.
                tstmp = dt.datetime.utcfromtimestamp(dh)
                ststmp = dt.datetime.utcfromtimestamp(float(t[0]))
                etstmp = dt.datetime.utcfromtimestamp(float(t[1]))
                if tstmp >= starttime and tstmp < endtime:
                    eventlist.append({'time':tstmp,'starttime':ststmp,'endtime':etstmp,'filename':filename,'mode':mode,'index':i})

        # Sort eventlist by timestamp
        eventlist = sorted(eventlist, key=lambda event: event['time'])

        return eventlist


    def get_ns(self,q):
        """
        Calculates the n values for a given NxN 2D array that has been flattened into a 1D array indexed by q.  This mapping
         assumes a symetric matrix such that not all the elemements of the NxN array need to be calculated, which is much more
         efficient for computing Omega and Psi which involve integrating special functions.

        Parameters:
            q: [int]
                flattened 1D index
        Returns:
            ni: [int]
                index along on direction of the NxN 2D array
            nj: [int]
                index along the other direction of the NxN 2D array
        Notes:
            - hv is also saved as an attribute of the class
        """
        ni = 0
        while q >= self.nbasis:
            ni = ni + 1
            q = q - (self.nbasis-ni)
        nj = q
        return ni, nj






    def eval_omega(self):
        """
        Evaluates the curvature regularization matrix Omega

        Returns:
            omega: [ndarray(nbasis,nbasis)]
                curvature regularization matrix omega
        Notes:
            - Omega ONLY minimizes the curvature in the horizontal plane (the theta/phi directions).  This decision was made
                because evaluating the product of two "perpendicular" laplacians is seperable (unlike the full laplacian),
                which dramatically improves the speed at which the integrals are computed (e.g. computing three 1D integrals numerically
                is WAY faster than computing one 3D integral).  Because the 0th order regularization controls the vertical
                component of the model relatively well, this approimation is acceptable given the performance improvement.
            - Currently, the code is designed to call parallize_omega() in a single loop, which should make it easy to impliment
                multiprocessing if this is desired in the future.
        """

        # Define max q undex for flattened 1D array indexing
        qmax = sum(range(self.nbasis+1))

        # Calculate omega value for each q (this loop should be easy to parallelize if desired)
        # Because omega is symetric, only half the elements need to be computed, so q only cycles over essentally the upper right
        #  triangle of the full omega matrix
        output = []
        for q in range(qmax):
            output.append(self.parallize_omega(q))
        output.sort()
        omega1 = np.array([out[1] for out in output])
            
        # Reconstruct the full omega array from the list of elements computed (omega1) taking advantage of symmetry
        omega = np.zeros((self.nbasis,self.nbasis))
        for q in range(qmax):
            ni,nj = self.get_ns(q)
            omega[ni,nj] = omega1[q]
            omega[nj,ni] = omega1[q]

        return omega


    def parallize_omega(self,q):
        """
        Evaluates a single element in the curvature regularization matrix Omega based on the index q provided

        Parameters:
            q: [int]
                flattened 1D index
        Returns:
            q: [int]
                flattened 1D index
            O: [double]
                single element of the Omega array corresponding to index q        
        """
        ni, nj = self.get_ns(q)
        ki, li, mi = self.basis_numbers(ni)
        kj, lj, mj = self.basis_numbers(nj)
        vi = self.nu(ni)
        vj = self.nu(nj)
        Iz = scipy.integrate.quad(self.omega_z_integrand, 0., self.param.max_zint, args=(ki,kj))
        It = scipy.integrate.quad(self.omega_t_integrand, 0, self.cap_lim, args=(vi,vj,mi,mj))
        Ip = scipy.integrate.quad(self.omega_p_integrand, 0, 2*np.pi, args=(vi,vj,mi,mj))
        O = Iz[0]*It[0]*Ip[0]

        return q, O


    def omega_z_integrand(self,z,ki,kj):
        """
        Evaluates the z portion of the omega integrand

        Parameters:
            z: [double]
                z coordinate
            ki: [int]
                ki index
            kj: [int]
                kj index
        Returns:
            z portion of omega integrand evaluated at z
        """
        return np.exp(-1*z)*sp.eval_laguerre(ki,z)*sp.eval_laguerre(kj,z)/z**2


    def omega_t_integrand(self,t,vi,vj,mi,mj):
        """
        Evaluates the theta portion of the omega integrand

        Parameters:
            t: [double]
                theta coordinate
            vi: [double]
                vi degree of legendre polynomial
            vj: [double]
                vj degree of legendre polynomial
            mi: [int]
                mi order of legendre polynomial
            mj: [int]
                mj order of legendr polynomial
        Returns:
            theta portion of omega integrand evaluated at t
        """
        x = np.cos(t)
        Oi = -1*vi*(vi*x**2+vi+1)*sp.lpmv(mi,vi,x)+vi*(vi+mi)*x*sp.lpmv(mi,vi-1,x)+vi*(vi-mi+1)*x*sp.lpmv(mi,vi+1,x)
        Oj = -1*vj*(vj*x**2+vj+1)*sp.lpmv(mj,vj,x)+vj*(vj+mj)*x*sp.lpmv(mj,vj-1,x)+vj*(vj-mj+1)*x*sp.lpmv(mj,vj+1,x)
        return 1/np.sin(t)**3*Oi*Oj


    def omega_p_integrand(self,p,vi,vj,mi,mj):
        """
        Evaluates the phi portion of the omega integrand

        Parameters:
            p: [double]
                phi coordinate
            vi: [double]
                vi degree of legendre polynomial
            vj: [double]
                vj degree of legendre polynomial
            mi: [int]
                mi order of legendre polynomial
            mj: [int]
                mj order of legendr polynomial
        Returns:
            phi portion of omega integrand evaluated at p
        """
        return self.Az(vi,mi,p)*self.Az(vj,mj,p)








    def eval_psi(self):
        """
        Evaluates the 0th order regularization matrix Psi

        Returns:
            psi: [ndarray(nbasis,nbasis)]
                0th order regularization matrix psi
        Notes:
            - Currently, the code is designed to call parallize_psi() in a single loop, which should make it easy to impliment
                multiprocessing if this is desired in the future.
        """

        # Define max q index for flattened 1D array indexing
        qmax = sum(range(self.nbasis+1))

        # Calculate psi value for each q (this loop should be easy to parallelize if desired)
        # Because psi is symetric, only half the elements need to be computed, so q only cycles over essentally the upper right
        #  triangle of the full psi matrix
        output = []
        for q in range(qmax):
            output.append(self.parallize_psi(q))
        output.sort()
        psi1 = np.array([out[1] for out in output])
            
        # Reconstruct the full psi array from the list of elements computed (psi1) taking advantage of symmetry
        psi = np.zeros((self.nbasis,self.nbasis))
        for q in range(qmax):
            ni,nj = self.get_ns(q)
            psi[ni,nj] = psi1[q]
            psi[nj,ni] = psi1[q]
        # psi = psi.reshape((self.nbasis,self.nbasis))

        return psi


    def parallize_psi(self,q):
        """
        Evaluates a single element in the 0th order regularization matrix Psi based on the index q provided

        Parameters:
            q: [int]
                flattened 1D index
        Returns:
            q: [int]
                flattened 1D index
            P: [double]
                single element of the Psi array corresponding to index q        
        """
        ni, nj = self.get_ns(q)
        ki, li, mi = self.basis_numbers(ni)
        kj, lj, mj = self.basis_numbers(nj)
        vi = self.nu(ni)
        vj = self.nu(nj)
        Iz = scipy.integrate.quad(self.psi_z_integrand, 0., self.param.max_zint, args=(ki,kj))
        It = scipy.integrate.quad(self.psi_t_integrand, 0, self.cap_lim, args=(vi,vj,mi,mj))
        Ip = scipy.integrate.quad(self.psi_p_integrand, 0, 2*np.pi, args=(vi,vj,mi,mj))
        P = Iz[0]*It[0]*Ip[0]

        return q, P

    def psi_z_integrand(self,z,ki,kj):
        """
        Evaluates the z portion of the psi integrand

        Parameters:
            z: [double]
                z coordinate
            ki: [int]
                ki index
            kj: [int]
                kj index
        Returns:
            z portion of psi integrand evaluated at z
        """
        return self.w(z)*np.exp(-1*z)*sp.eval_laguerre(ki,z)*sp.eval_laguerre(kj,z)*z**2

    def psi_t_integrand(self,t,vi,vj,mi,mj):
        """
        Evaluates the theta portion of the psi integrand

        Parameters:
            t: [double]
                theta coordinate
            vi: [double]
                vi degree of legendre polynomial
            vj: [double]
                vj degree of legendre polynomial
            mi: [int]
                mi order of legendre polynomial
            mj: [int]
                mj order of legendr polynomial
        Returns:
            theta portion of psi integrand evaluated at t
        """
        return sp.lpmv(mi,vi,np.cos(t))*sp.lpmv(mj,vj,np.cos(t))*np.sin(t)

    def psi_p_integrand(self,p,vi,vj,mi,mj):
        """
        Evaluates the phi portion of the omega integrand

        Parameters:
            p: [double]
                phi coordinate
            vi: [double]
                vi degree of legendre polynomial
            vj: [double]
                vj degree of legendre polynomial
            mi: [int]
                mi order of legendre polynomial
            mj: [int]
                mj order of legendr polynomial
        Returns:
            phi portion of omega integrand evaluated at p
        """
        return self.Az(vi,mi,p)*self.Az(vj,mj,p)





    def eval_tau(self,R,data,error):
        """
        Evaluates the 0th order regularization vector Tau

        Returns:
            tau: [ndarray(nbasis)]
                0th order regularization vector tau
        Notes:
            - Currently, the code is designed to call parallize_tau() in a single loop, which should make it easy to impliment
                multiprocessing if this is desired in the future.
        """

        # Fit a zeroth-order function to the data
        try:
            self.param.eval_zeroth_order(R[0],data,error)
        except RuntimeError:
            tau = np.full((self.nbasis,),np.nan)
            return tau

        # # Short plotting procedure to check that the fit to the Chapman funciton is realistic.  In general, this
        # #   should be commented out, but it's useful for debugging so retain the code.
        # z = np.arange(0,11,0.1)
        # d = self.param.zeroth_order(z)
        # fig = plt.figure(figsize=(10,10))
        # f = fig.add_subplot(111)
        # f.plot(d,z)
        # f.scatter(data, R[0], c=error, vmin=0, vmax=1e11)
        # plt.show()

        # print max(R[0])

        # calculate tau for each basis function
        output = []
        for n in range(self.nbasis):
            output.append(self.parallize_tau(n))
        output.sort()
        tau = np.array([out[1] for out in output])

        tau = tau[:,None]
                
        return tau

    def parallize_tau(self,n):
        """
        Evaluates a single element in the 0th order regularization vector Tau based on the index n provided

        Parameters:
            n: [int]
                3D basis function index
        Returns:
            n: [int]
                3D basis function index
            T: [double]
                single element of the Tau array corresponding to index n       
        """
        k, l, m = self.basis_numbers(n)
        v = self.nu(n)
        Iz = scipy.integrate.quad(self.tau_z_integrand, 0., self.param.max_zint, args=(k))
        # Iz = scipy.integrate.quad(self.tau_z_integrand, 0., 15, args=(k))
        It = scipy.integrate.quad(self.tau_t_integrand, 0, self.cap_lim, args=(v,m))
        Ip = scipy.integrate.quad(self.tau_p_integrand, 0, 2*np.pi, args=(v,m))
        # print Iz, It, Ip
        T = Iz[0]*It[0]*Ip[0]

        return n, T

    def tau_z_integrand(self,z,k):
        """
        Evaluates the z portion of the tau integrand

        Parameters:
            z: [double]
                z coordinate
            ki: [int]
                ki index
            kj: [int]
                kj index
        Returns:
            z portion of tau integrand evaluated at z
        """
        return self.w(z)*np.exp(-0.5*z)*sp.eval_laguerre(k,z)*self.param.zeroth_order(z)*z**2

    def tau_t_integrand(self,t,v,m):
        """
        Evaluates the theta portion of the tau integrand

        Parameters:
            t: [double]
                theta coordinate
            vi: [double]
                vi degree of legendre polynomial
            vj: [double]
                vj degree of legendre polynomial
            mi: [int]
                mi order of legendre polynomial
            mj: [int]
                mj order of legendr polynomial
        Returns:
            theta portion of tau integrand evaluated at t
        """
        return sp.lpmv(m,v,np.cos(t))*np.sin(t)

    def tau_p_integrand(self,p,v,m):
        """
        Evaluates the phi portion of the tau integrand

        Parameters:
            p: [double]
                phi coordinate
            vi: [double]
                vi degree of legendre polynomial
            vj: [double]
                vj degree of legendre polynomial
            mi: [int]
                mi order of legendre polynomial
            mj: [int]
                mj order of legendr polynomial
        Returns:
            phi portion of tau integrand evaluated at p
        """
        return self.Az(v,m,p)

    def w(self,z):
        # return np.exp(-0.001*(z-10)**2)
        return 1.



    def find_reg_param(self,A,b,W,reg_matrices,method=None):
        """
        Find the regularization parameters.  A number of different methods are provided for this (se the reg_methods dictionary).
            - chi2: enforce the statistical condition chi2 = nu (e.g. Nicolls et al., 2014)
            - gcv: use Generalized Cross validation
            - manual: hardcode in regularization parameters
            - prompt: ask user for regularization parameters via a prompt
        
        Parameters:
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b: [ndarray(npoints)]
                array of raw input data
            W: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            method: [str]
                method that should be used to find regularization parameter (valid options include 'chi2', 'gcv', 'manual', 'prompt')
        Returns:
            reg_params: [dict]
                dictionary of regularization parameters needed based on regularization_list
        Notes:
            - Currently, the automated methods of calculating regularization parameters (chi2 and gcv) handle multiple parameters by
                solving for each parameter individually while assuming all other parameters are zero.  This is non-ideal and tends
                to lead to over smothing, but it's the best approach we have right now because most standard methods of selecting
                regularization parameters only consider one condition (one condition = one unkown).  There is some literature on how
                to find multiple regularization parameters simultaniously, but have not had luck implementing any of these techniques.
            - Regularization (particularly choosing an "appropriate" regularization parameter) is VERY finiky.  This function attempts 
                to have reasonable defaults and options, but it is HIGHLY unlikely these will work well in all cases.  The author has
                tried to make the code flexible so new methods can be added "easily", but appologizes in advance for the headache 
                this will undoubtably cause.
        """

        # Define reg_methods dictionary
        reg_methods = {'chi2':self.chi2,'gcv':self.gcv,'manual':self.manual,'prompt':self.prompt}

        # Default method is chi2
        if method is None:
            method = 'chi2'

        reg_params = {}
        for rl in self.regularization_list:
            try:
                reg_params[rl] = reg_methods[method](A,b,W,reg_matrices,rl)
            except ValueError as err:
                print(err)
                print 'Returning NANs for regularization parameters.'
                reg_params[rl] = np.nan

        return reg_params



    def chi2(self,A,b,W,reg_matrices,reg):
        """
        Find the regularization parameter using the chi2 method.

        Parameters:
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b: [ndarray(npoints)]
                array of raw input data
            W: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            reg: [str]
                regularization method (from regularization_list) that the regularization parameter is being solved for
        Returns:
            reg_param: [double]
                the value of the regularization parameter
        """

        # Set nu
        scale_factors = [0.6,0.7,0.8,0.9,1.0]
        N = len(b)
        bracket = False

        for sf in scale_factors:
            nu = N*sf
            # nu = N*1.

            # Determine the bracketing interval for the root (between 1e0 and 1e-100)
            alpha0 = 0.
            val0 = 1.
            alpha = 0.
            val = self.chi2objfunct(alpha,A,b,W,reg_matrices,nu,reg)
            if val<0:
                print 'Too smooth to find regularization parameter. Returning alpha=0.'
                return 0

            while val0*val > 0:
                # print val0, val
                bracket = True
                # nu = N*scale_factor
                val0 = val
                alpha0 = alpha
                alpha = alpha - 1.
                val = self.chi2objfunct(alpha,A,b,W,reg_matrices,nu,reg)
                if alpha < -100.:
                    bracket = False
                    break
            if bracket:
                break
            else:
                continue


        if not bracket:
            raise ValueError('Could not find any roots to the objective function chi^2-nu in the range (1e-100,1).')
        else:
            # Use the Brent (1973) method to find the root within the bracketing interval found above
            solution = scipy.optimize.brentq(self.chi2objfunct,alpha,alpha0,args=(A,b,W,reg_matrices,nu,reg),disp=True)

        reg_param = np.power(10.,solution)

        return reg_param



    def chi2objfunct(self,alpha,A,b,W,reg_matrices,nu,reg):
        """
        Objective function for the chi2 method of finding the regularization parameter.  Returns chi^2-nu for a given 
         regularization parameter.

        Parameters:
            alpha: [double]
                regularization parameter that is being found iteratively
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b: [ndarray(npoints)]
                array of raw input data
            W: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            nu: [double]
                value that chi2 should be equal to (usualy the number of data points)
            reg: [str]
                regularization method (from regularization_list) that the regularization parameter is being solved for
        Returns:
            chi2 - nu
        Notes:
            - The objective function is used in a root finder to find the regularization parameter which satisfies chi2-n=0.
        """

        # Make the reg_params dictionary from alpha and the defined reg
        reg_params = {}
        for rl in self.regularization_list:
            if rl == reg:
                reg_params[rl] = np.power(10.,alpha)
            else:
                reg_params[rl] = 0.

        # Evaluate coefficient vector
        C = self.eval_C(A,b,W,reg_matrices,reg_params)

        # compute chi^2
        val = np.squeeze(np.dot(A,C))
        chi2 = sum((val-np.squeeze(b))**2*np.squeeze(W))
 
        return chi2-nu




    def gcv(self,A,b,W,reg_matrices,reg):
        """
        Find the regularization parameter using the generalized cross validation method.

        Parameters:
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b: [ndarray(npoints)]
                array of raw input data
            W: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            reg: [str]
                regularization method (from regularization_list) that the regularization parameter is being solved for
        Returns:
            reg_param: [double]
                the value of the regularization parameter
        Notes:
            - Convergence of the minimizer can be very sensitive to the initial guess (alpha0) chosen.  This method has
                not been thoroughly tested for all AMISR parameters and modes, so it may be nessisary to alter the initial
                guess later, possibly having different initial guesses for density and temperature.
        """

        # Set initial guess
        alpha0 = -20.

        # Use the Nelder-Mead method to find the minimum of the GCV objective function
        solution = scipy.optimize.minimize(self.gcvobjfunct,alpha0,args=(A,b,W,reg_matrices,reg),method='Nelder-Mead')
        if not solution.success:
            raise ValueError('Minima of GCV function could not be found')

        reg_param = np.power(10.,solution.x[0])

        return reg_param


    def gcvobjfunct(self,alpha,A0,b0,W0,reg_matrices,reg):
        """
        Objective function for the GCV method of finding the regularization parameter.  Returns the value of the GCV function
         for a given regularization parameter.

        Parameters:
            alpha: [double]
                regularization parameter that is being found iteratively
            A0: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b0: [ndarray(npoints)]
                array of raw input data
            W0: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            reg: [str]
                regularization method (from regularization_list) that the regularization parameter is being solved for
        Returns:
            GCV function for regularization parameter alpha
        Notes:
            - The objective function is used in a minimizer to find the regularization parameter which minimizes the GCV
                function.
        """

        # Define reg_params dictionary        
        reg_params = {}
        for rl in self.regularization_list:
            if rl == reg:
                reg_params[rl] = np.power(10.,alpha)
            else:
                reg_params[rl] = 0.

        residuals = []
        for i in range(len(b0)):
            # Pull one data point out of arrays
            # data point in question:
            Ai = A0[i,:]
            bi = b0[i]
            Wi = W0[i]
            # arrays minus one data point:
            A = np.delete(A0,i,0)
            b = np.delete(b0,i,0)
            W = np.delete(W0,i,0)

            # Evaluate coefficient vector
            C = self.eval_C(A,b,W,reg_matrices,reg_params)

            # Calculate residual for the data point not included in the fit
            val = np.squeeze(np.dot(Ai,C))
            residuals.append((val-bi)**2*Wi)

        return sum(residuals)


    def manual(self,A,b,W,Omega,Psi,Tau,reg):
        """
        Manually hard-code the regularization parameter.

        Parameters:
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b: [ndarray(npoints)]
                array of raw input data
            W: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            reg: [str]
                regularization method (from regularization_list) that the regularization parameter is being solved for
        Returns:
            reg_param: [double]
                the value of the regularization parameter
        """

        lam = 1.e-28
        kappa = 1.e-23

        if reg == 'curvature':
            reg_param = lam
        if reg == '0thorder':
            reg_param = kappa

        return reg_param



    def prompt(self,A,b,W,Omega,Psi,Tau,reg):
        """
        Enter the regularization parameter via command line prompt.

        Parameters:
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b: [ndarray(npoints)]
                array of raw input data
            W: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            reg: [str]
                regularization method (from regularization_list) that the regularization parameter is being solved for
        Returns:
            reg_param: [double]
                the value of the regularization parameter
        """

        reg_param = raw_input('Enter {} regularization parameter: '.format(reg))

        reg_param = float(reg_param)

        return reg_param



    def eval_C(self,A,b,W,reg_matrices,reg_params,calccov=False):
        """
        Evaluate the coefficient array, C.

        Parameters:
            A: [ndarray(npoints,nbasis)]
                array of basis functions evaluated at all input points
            b: [ndarray(npoints)]
                array of raw input data
            W: [ndarray(npoint)]
                array of errors on raw input data
            reg_matrices: [dict]
                dictionary of all the regularization matrices needed based on regularization_list
            reg_params: [dict]
                dictionary of all the regularization parameter values needed based on regularization_list
            calccov: [bool]
                boolian value whether or not to calculate the covariance matrix
        Returns:
            C: [ndarray(nbasis)]
                array of basis function coefficients
            dC: Optional [ndarray(nbasis,nbasis)]
                covariance matrix for basis functions
        """

        AWA = np.dot(A.T,W*A)
        X = np.dot(A.T,W*A)
        y = np.dot(A.T,W*b)
        if 'curvature' in self.regularization_list:
            X = X + reg_params['curvature']*reg_matrices['Omega']
        if '0thorder' in self.regularization_list:
            X = X + reg_params['0thorder']*reg_matrices['Psi']
            y = y + reg_params['0thorder']*reg_matrices['Tau']
        C = np.squeeze(scipy.linalg.lstsq(X,y,overwrite_a=True,overwrite_b=True)[0])

        if calccov:
            H = scipy.linalg.pinv(X)
            dC = np.dot(H,np.dot(AWA,H.T))
            return C, dC
        else:
            return C


    def fit(self):
        """
        Perform fit on every event in eventlist.

        Parameters:
            eventlist: [dict]
                list of dictionaries containing the timestamp, file name, radar mode, and index within the file for a 
                particular event
                vaild keys:
                    'time': timestamp (datetime object)
                    'filename': file name including full file path
                    'mode': radar mode
                    'index': index of this particular event within the file
                eventlist can be created by the generate_eventlist() method of the Fit class
        """

#         if not eventlist:
#             raise ValueError('Event list is empty!')


        time = []
        Coeffs = []
        Covariance = []
        chi_sq = []
        cent_point = []
        hull_v = []
        raw_coords = []
        raw_data = []
        raw_error = []
        raw_filename = []
        raw_index = []

#         evaluated_modes = {}

#         mode_dict = {'WorldDay66m':{'maxk':4,'maxl':6,'cap_lim':6.*np.pi/180.,'reglist':['curvature','0thorder'],'regmethod':'chi2','regscalefac':1.0},
#                      'imaginglp':{'maxk':4,'maxl':6,'cap_lim':6.*np.pi/180.,'reglist':['0thorder'],'regmethod':'chi2','regscalefac':np.nan},
#                      'Convection67m':{'maxk':4,'maxl':6,'cap_lim':6.*np.pi/180.,'reglist':['0thorder'],'regmethod':'chi2','regscalefac':np.nan},
#                      'isinglass':{'maxk':4,'maxl':6,'cap_lim':6.*np.pi/180.,'reglist':['0thorder'],'regmethod':'chi2','regscalefac':np.nan}
#                     }

        
        self.maxl = LMAX
        self.maxk = KMAX
        self.cap_lim = CAPLIMIT
        self.nbasis = self.maxk*self.maxl**2
        self.regularization_list = REGULARIZATION_METHOD
        self.reg_method = REGULARIZATION_PARAMETER_METHOD
        self.reg_scale_factor = np.nan


#         if radar_mode in evaluated_modes:
#             reg_matrices = evaluated_modes[radar_mode]
#             if '0thorder' in self.regularization_list:
#                 reg_matrices['Tau'] = self.eval_tau(R,ne0,er0)
#         else:
#             print 'New mode {}.  Regularization matrices must be evaluated.  This may take a few minutes.'.format(radar_mode)

        print 'Evaluating Regularization matricies.  This may take a few minutes.'
        reg_matrices = {}
        if 'curvature' in self.regularization_list:
            reg_matrices['Omega'] = self.eval_omega()
        if '0thorder' in self.regularization_list:
            reg_matrices['Psi'] = self.eval_psi()
#             reg_matrices['Tau'] = self.eval_tau(R,ne0,er0)
#         evaluated_modes[radar_mode] = reg_matrices
 
        
        
#         # find total number of indicies in the file
#         # file looping/IO is hacky - need to rewrite param.get_data
#         with tables.open_file(FILENAME,'r') as h5file:
#             utime = h5file.get_node('/Time/UnixTime')[:]
            
            
        utime, R0, value, error = self.param.get_data(FILENAME)
 
        print utime.shape, R0.shape, value.shape, error.shape

        # Find convex hull of original data set
        verticies = self.compute_hull(R0)
#             try:
#                 vertices = self.compute_hull(R0)
#             except:
#                 continue

        # Transform coordinates
        R0, cp = self.transform_coord(R0)
#         ne0 = ne0
#         er0 = error
    
#         for index in range(len(utime)):
        for ne0, er0 in zip(value,error):
            print ne0
            print R0.shape, ne0.shape, er0.shape
            
            R = R0[:,np.isfinite(ne0)]
            er0 = er0[np.isfinite(ne0)]
            ne0 = ne0[np.isfinite(ne0)]

            print R.shape, ne0.shape, er0.shape

#         for item in eventlist:

#             print item['time']
#             print item['mode']

#             R0, ne0, error = self.param.get_data(FILENAME,index)
#             R0, ne0, error = self.param.get_data(item['filename'],item['index'])

#             # Find convex hull of original data set
#             try:
#                 vertices = self.compute_hull(R0)
#             except:
#                 continue

#             # Transform coordinates
#             R, cp = self.transform_coord(R0)
#             ne0 = ne0
#             er0 = error


#             radar_mode = item['mode'].split('.')[0]

#             self.maxl = LMAX
#             self.maxk = KMAX
#             self.cap_lim = CAPLIMIT
#             self.nbasis = self.maxk*self.maxl**2
#             self.regularization_list = REGULARIZATION_METHOD
#             self.reg_method = REGULARIZATION_PARAMETER_METHOD
#             self.reg_scale_factor = np.nan


#             if radar_mode in evaluated_modes:
#                 reg_matrices = evaluated_modes[radar_mode]
#                 if '0thorder' in self.regularization_list:
#                     reg_matrices['Tau'] = self.eval_tau(R,ne0,er0)
#             else:
#                 print 'New mode {}.  Regularization matrices must be evaluated.  This may take a few minutes.'.format(radar_mode)

#                 reg_matrices = {}
#                 if 'curvature' in self.regularization_list:
#                     reg_matrices['Omega'] = self.eval_omega()
#                 if '0thorder' in self.regularization_list:
#                     reg_matrices['Psi'] = self.eval_psi()
#                     reg_matrices['Tau'] = self.eval_tau(R,ne0,er0)
#                 evaluated_modes[radar_mode] = reg_matrices

            if '0thorder' in self.regularization_list:
                reg_matrices['Tau'] = self.eval_tau(R,ne0,er0)


            if np.any([np.any(np.isnan(v.flatten())) for k, v in reg_matrices.items()]):
                continue


            W = np.array(er0**(-2))[:,None]
            b = ne0[:,None]
            A = self.eval_basis(R)

            reg_params = self.find_reg_param(A,b,W,reg_matrices,method=self.reg_method)


            if np.any(np.isnan([v for k, v in reg_params.items()])):
                continue


            C, dC = self.eval_C(A,b,W,reg_matrices,reg_params,calccov=True)
            c2 = sum((np.squeeze(np.dot(A,C))-np.squeeze(b))**2*np.squeeze(W))

            # time.append(item['time'])
#             time.append([item['starttime'],item['endtime']])
#             time.append(utime[index])
            Coeffs.append(C)
            Covariance.append(dC)
            chi_sq.append(c2)
            cent_point.append(cp)
#             hull_v.append(vertices)
            raw_coords.append(R0)
            raw_data.append(ne0)
            raw_error.append(error)
#             raw_filename.append(item['filename'])
#             raw_index.append(item['index'])
            raw_filename.append(FILENAME)
#             raw_index.append(index)

#         self.time = time
        self.time = utime
        self.Coeffs = Coeffs
        self.Covariance = Covariance
        self.chi_sq = chi_sq
        self.cent_point = cent_point
#         self.hull_v = hull_v
        self.hull_v = verticies
        self.raw_coords = raw_coords
        self.raw_data = raw_data
        self.raw_error = raw_error
#         self.raw_filename = raw_filename
        self.raw_filename = FILENAME
#         self.raw_index = raw_index



    def saveh5(self,filename=None):
        """
        Saves coefficients to a hdf5 file

        Parameters:
            filename: Optional [str]
                name of file to save
                default file is ./Coefficients/YYYMMDD_RADAR_PARAM.h5
        """

        h5outname = wdir+'/Coefficients/{:04d}{:02d}{:02d}_{}_{}.h5'.format(self.date.year,self.date.month,self.date.day,self.radar,self.param.key)
        if filename:
            h5outname = filename

        h5out = tables.open_file(h5outname, 'w')

        utime = [[(t[0]-dt.datetime.utcfromtimestamp(0)).total_seconds(),(t[1]-dt.datetime.utcfromtimestamp(0)).total_seconds()] for t in self.time]

        cgroup = h5out.create_group('/','Coeffs','Dataset')
        fgroup = h5out.create_group('/','FitParams','Dataset')
        dgroup = h5out.create_group('/','RawData','Dataset')

        h5out.create_array('/', 'UnixTime', utime)

        h5out.create_array(cgroup, 'C', self.Coeffs)
        h5out.create_array(cgroup, 'dC', self.Covariance)

        h5out.create_array(fgroup, 'kmax', self.maxk)
        h5out.create_array(fgroup, 'lmax', self.maxl)
        h5out.create_array(fgroup, 'cap_lim', self.cap_lim)
        h5out.create_array(fgroup, 'reglist', self.regularization_list)
        h5out.create_array(fgroup, 'regmethod', self.reg_method)
        h5out.create_array(fgroup, 'regscalefac', self.reg_scale_factor)
        h5out.create_array(fgroup, 'chi2', self.chi_sq)
        h5out.create_array(fgroup, 'center_point', self.cent_point)
        vlarray = h5out.create_vlarray(fgroup, 'hull_vertices', tables.FloatAtom(shape=3))
        for v in self.hull_v:
            vlarray.append(v.T)

        h5out.create_array(dgroup, 'filename', self.raw_filename)
        h5out.create_array(dgroup, 'index', self.raw_index)
        rarray = h5out.create_vlarray(dgroup, 'coordinates', tables.FloatAtom(shape=3))
        darray = h5out.create_vlarray(dgroup, 'data', tables.FloatAtom(shape=1))
        earray = h5out.create_vlarray(dgroup, 'error', tables.FloatAtom(shape=1))
        for r,d,e in zip(self.raw_coords,self.raw_data,self.raw_error):
            rarray.append(r.T)
            darray.append(d[:,None])
            earray.append(e[:,None])

        h5out.close()





    def validate(self,time0,altitude,longitude):
        """
        Creates a basic plot of the volumetric reconstruction next to the original measurment points to confirm that the reconstruction is reasonable.

        Parameters:
            time0: [datetime]
                time at which the plots should be created
            altitude: [float]
                altitude of the latitude-longitude slice
            longitude: [float]
                longitude of the latitude-altitude slice
        """

        self.datetime = time0

        targtime = (self.datetime-dt.datetime(1970,1,1)).total_seconds()

        try:
            utime = np.array([[(t[0]-dt.datetime.utcfromtimestamp(0)).total_seconds(),(t[1]-dt.datetime.utcfromtimestamp(0)).total_seconds()] for t in self.time])
            rec = np.where((targtime >= utime[:,0]) & (targtime < utime[:,1]))[0]
            rec = rec[0]

            self.t = self.time[rec][0]
            self.C = self.Coeffs[rec]
            self.dC = self.Covariance[rec]
            self.hv = self.hull_v[rec].T
            self.cp = self.cent_point[rec]
            self.rR = self.raw_coords[rec].T
            self.rd = self.raw_data[rec]
            self.re = self.raw_error[rec]


        except AttributeError:
            self.loadh5(raw=True)


        lat, lon, alt = cc.spherical_to_geodetic(self.hv.T[0],self.hv.T[1],self.hv.T[2])
        latrange = np.linspace(min(lat),max(lat),50)
        lonrange = np.linspace(min(lon),max(lon),50)
        altrange = np.linspace(min(alt),max(alt),50)

        cent_lat = (min(lat)+max(lat))/2.
        cent_lon = (min(lon)+max(lon))/2.
        height = (max(lat)-min(lat))*np.pi/180.*(RE+altitude)
        width = (max(lon)-min(lon))*np.pi/180.*(RE+altitude)*np.cos(cent_lat*np.pi/180.)


        lat, lon = np.meshgrid(latrange,lonrange)
        alt = np.ones(np.shape(lat))*altitude
        r,t,p = cc.geodetic_to_spherical(lat.ravel(),lon.ravel(),alt.ravel())
        R0 = np.array([r,t,p])

        dens, grad = self.getparam(R0)
        dens = dens.reshape(lat.shape)
        grad = grad.reshape(lat.shape+(4,))

        dens[dens<0] = np.nan


        lat2, alt2 = np.meshgrid(latrange,altrange)
        lon2 = np.ones(np.shape(lat2))*longitude
        r,t,p = cc.geodetic_to_spherical(lat2.ravel(),lon2.ravel(),alt2.ravel())
        R02 = np.array([r,t,p])

        dens2, grad2 = self.getparam(R02)
        dens2 = dens2.reshape(lat2.shape)
        grad2 = grad2.reshape(lat2.shape+(4,))
        dens2[dens2<0] = np.nan

        lat3d, lon3d, alt3d = cc.spherical_to_geodetic(self.rR.T[0],self.rR.T[1],self.rR.T[2])
        dens3d = self.rd
        err3d = self.re

        print lat3d

        raw_lat = lat3d[np.where(abs(alt3d-altitude)<10.)]
        raw_lon = lon3d[np.where(abs(alt3d-altitude)<10.)]
        raw_dens = dens3d[np.where(abs(alt3d-altitude)<10.)]

        longitude = longitude+360.
        raw_lat2 = lat3d[np.where(abs(lon3d-longitude)<1.)]
        raw_alt2 = alt3d[np.where(abs(lon3d-longitude)<1.)]
        raw_dens2 = dens3d[np.where(abs(lon3d-longitude)<1.)]



        # maximum and minimum ne for color scale (in units of 10^11 m^-3)
        minv = self.param.vrange[0]
        maxv = self.param.vrange[1]


        fig = plt.figure(figsize=(15,5))
        cmap = plt.cm.get_cmap('viridis')

        # ax = fig.add_subplot(131,projection='3d')
        # ax.scatter(lat3d,lon3d,alt3d,c=dens3d, s=2.e11/err3d, vmin=minne, vmax=maxne, depthshade=False)
        ax = fig.add_subplot(131,projection='3d')
        # m = Basemap(projection='aeqd',lat_0=cent_lat,lon_0=cent_lon,llcrnrlon=250,llcrnrlat=65,urcrnrlon=330,urcrnrlat=80,resolution='l')
        m = Basemap(projection='aeqd',lat_0=cent_lat,lon_0=cent_lon,width=width,height=height,resolution='l')
        ax.add_collection3d(m.drawcoastlines())
        # ax.add_collection3d(m.scatter(lon3d,lat3d,alt3d,c=dens3d, s=sum(self.param.vrange)/err3d, latlon=True, vmin=minv, vmax=maxv,cmap=cmap))
        ax.add_collection3d(m.scatter(lon3d,lat3d,alt3d,c=dens3d, s=(dens3d/err3d)**2, latlon=True, vmin=minv, vmax=maxv,cmap=cmap))
        ax.set_title(self.t.strftime('%Y-%m-%d %H:%M:%S'))


        ax = fig.add_subplot(132)
        # m = Basemap(projection='aeqd',lat_0=cent_lat,lon_0=cent_lon,llcrnrlon=250,llcrnrlat=65,urcrnrlon=310,urcrnrlat=81,resolution='l')
        m = Basemap(projection='aeqd',lat_0=cent_lat,lon_0=cent_lon,width=width,height=height,resolution='l')
        f = m.contourf(lon,lat,dens,levels=np.linspace(minv,maxv,100),extend='both',latlon=True,zorder=2,cmap=cmap)
        m.scatter(raw_lon,raw_lat,c=raw_dens,vmin=minv,vmax=maxv,s=25,latlon=True,zorder=3,cmap=cmap)
        # m.scatter(raw_lon,raw_lat,c=raw_dens,vmin=minv,vmax=maxv,s=2*(dens3d/err3d)**2,latlon=True,zorder=3,cmap=cmap)
        u, v, x, y = m.rotate_vector(grad[:,:,2],-1*grad[:,:,1],lon,lat,returnxy=True)
        m.quiver(x,y,u,v,zorder=3)
        m.plot(lon2[0],lat2[0],latlon=True,zorder=4,color='white')
        m.drawcoastlines()
        m.drawmapboundary(fill_color='white')
        m.drawparallels([60.,70.,80.,90.])
        m.drawmeridians(np.arange(-180.,180.,30))

        ax = fig.add_subplot(133)
        f = ax.contourf(lat2,alt2,dens2,levels=np.linspace(minv,maxv,100),extend='both',zorder=2,cmap=cmap)
        ax.scatter(raw_lat2,raw_alt2,c=raw_dens2,vmin=minv,vmax=maxv,s=25,zorder=3,cmap=cmap)
        # ax.scatter(raw_lat2,raw_alt2,c=raw_dens2,vmin=minv,vmax=maxv,s=2*(dens3d/err3d)**2,zorder=3,cmap=cmap)
        ax.quiver(lat2,alt2,-1*grad2[:,:,1],grad2[:,:,0])
        ax.plot(lat[0],alt[0],zorder=4,color='white')


        fig.subplots_adjust(left=0.02,right=0.9)
        axc = fig.add_axes([0.91,0.15,0.02,0.7])
        cbar = fig.colorbar(f,cax=axc)
        cbar.set_label(self.param.name+' ('+self.param.units+')')

        plt.show()







class AMISR_param(object):
    """
    This class contains parameter-specific quantities that are nessisary for fitting and/or plotting.  It also 
    has methods to read the nessisary arrays from processed AMISR files.

    Parameters:
        key: [str]
            the parameter ('dens' or 'temp') to initalize the class with

    Atributes:
        name: parameter name
        max_zint: maximum z limit to integrate to
        vrange: range of expected values for the parameter (for plotting)
        units: units of the parameter
        key: parameter key
        p0: values of the zeroth-order fit

    Methods:
        get_data: read nessisary arrays from a data file
        eval_zeroth_order: evaluate the zeroth order function
        zeroth_order: zeroth order function
        chapman: chapman function (density zeroth order)
        sinh: hyperbolic sine function (temperature zeroth order)
        quickplot: produce a basic scatter plot of the data
    """

    def __init__(self,key):
        self.name = PARAMETER_NAME
        self.max_zint = MAX_Z_INT
        self.vrange = PARAMETER_RANGE
        self.units = PARAMETER_UNITS
        self.key = key



#     def get_data(self,filename,index):
# #     def get_data(self,filename):
#         """
#         Read a particular index of a processed AMISR hdf5 file and return the coordinates, values, and errors as arrays.

#         Parameters:
#             filename: [str]
#                 filename/path of processed AMISR hdf5 file
#             index: [int]
#                 record index

#         Returns:
#             R0: [ndarray (npointsx3)]
#                 coordinates of each data point in spherical coordinate system
#             value: [ndarray (npoints)]
#                 parameter value of each data point
#             error: [ndarray (npoints)]
#                 error in parameter values
#         """


#         with tables.open_file(filename,'r') as h5file:

#             alt = h5file.get_node('/Geomag/Altitude')
#             lat = h5file.get_node('/Geomag/Latitude')
#             lon = h5file.get_node('/Geomag/Longitude')
#             c2 = h5file.get_node('/FittedParams/FitInfo/chi2')
#             fc = h5file.get_node('/FittedParams/FitInfo/fitcode')
#             imass = h5file.get_node('/FittedParams/IonMass')
#             if self.key == 'dens':
#                 val = h5file.get_node('/FittedParams/Ne')
#                 err = h5file.get_node('/FittedParams/dNe')
#             else:
#                 fits = h5file.get_node('/FittedParams/Fits')
#                 err = h5file.get_node('/FittedParams/Errors')


#             altitude = alt.read().flatten()
#             latitude = lat.read().flatten()
#             longitude = lon.read().flatten()
#             chi2 = c2[index].flatten()
#             fitcode = fc[index].flatten()
#             imass = imass.read()


#             # This accounts for an error in some of the hdf5 files where chi2 is overestimated by 369.
#             if np.mean(chi2) > 100.:
#                 chi2 = chi2 - 369.

#             # choose index based on ending of key
#             if self.key.endswith('_O'):
#                 j = int(np.where(imass == 16)[0])
#             elif self.key.endswith('_O2'):
#                 j = int(np.where(imass == 32)[0])
#             elif self.key.endswith('_NO'):
#                 j = int(np.where(imass == 30)[0])
#             elif self.key.endswith('_N2'):
#                 j = int(np.where(imass == 28)[0])
#             elif self.key.endswith('_N'):
#                 j = int(np.where(imass == 14)[0])
#             else:
#                 j = -1


#             if self.key == 'dens':
#                 value = np.array(val[index].flatten())
#                 error = np.array(err[index].flatten())
#             if self.key.startswith('frac'):
#                 value = np.array(fits[index,:,:,j,0].flatten())
#                 error = np.array(err[index,:,:,j,0].flatten())
#             if self.key.startswith('temp'):
#                 value = np.array(fits[index,:,:,j,1].flatten())
#                 error = np.array(err[index,:,:,j,1].flatten())
#             if self.key.startswith('colfreq'):
#                 value = np.array(fits[index,:,:,j,2].flatten())
#                 error = np.array(err[index,:,:,j,2].flatten())



#         # data_check: 2D boolian array for removing "bad" data
#         # Each column correpsonds to a different "check" condition
#         # TRUE for "GOOD" point; FALSE for "BAD" point
#         # A "good" record that shouldn't be removed should be TRUE for EVERY check condition
#         if self.key == 'dens':
#             data_check = np.array([np.isfinite(error),error>1.e10,fitcode>0,fitcode<5,chi2<10,chi2>0.1])
#         elif 'temp' in self.key:
#             data_check = np.array([np.isfinite(error),fitcode>0,fitcode<5,chi2<10,chi2>0.1])
#         else:
#             data_check = np.array([np.isfinite(value)])

#         # ALL elements of data_check MUST be TRUE for a particular index to be kept
#         finite_indicies = np.where(np.all(data_check,axis=0))[0]

#         # reform the data arrays only with "good" data
#         altitude = np.array(altitude[finite_indicies])
#         latitude = np.array(latitude[finite_indicies])
#         longitude = np.array(longitude[finite_indicies])
#         error = np.array(error[finite_indicies])
#         value = np.array(value[finite_indicies])


#         # Convert input coordinates to geocentric-spherical
#         r, t, p = cc.geodetic_to_spherical(latitude,longitude,altitude/1000.)
#         R0 = np.array([r,t,p])

#         return R0, value, error


    def get_data(self,filename):
        """
        Read parameter from a processed AMISR hdf5 file and return the time, coordinates, values, and errors as arrays.

        Parameters:
            filename: [str]
                filename/path of processed AMISR hdf5 file

        Returns:
            utime: [ndarray (nrecordsx2)]
                start and end time of each record (Unix Time)
            R0: [ndarray (3xnpoints)]
                coordinates of each data point in spherical coordinate system
            value: [ndarray (nrecordsxnpoints)]
                parameter value of each data point
            error: [ndarray (nrecordsxnpoints)]
                error in parameter values
        """

        index_dict = {'frac':0, 'temp':1, 'colfreq':2}
        mass_dict = {'O':16, 'O2':32, 'NO':30, 'N2':28, 'N':14}

        with tables.open_file(filename,'r') as h5file:

            utime = h5file.get_node('/Time/UnixTime')[:]

            alt = h5file.get_node('/Geomag/Altitude')[:]
            lat = h5file.get_node('/Geomag/Latitude')[:]
            lon = h5file.get_node('/Geomag/Longitude')[:]
            c2 = h5file.get_node('/FittedParams/FitInfo/chi2')[:]
            fc = h5file.get_node('/FittedParams/FitInfo/fitcode')[:]
            imass = h5file.get_node('/FittedParams/IonMass')[:]

            if self.key == 'dens':
                val = h5file.get_node('/FittedParams/Ne')[:]
                err = h5file.get_node('/FittedParams/dNe')[:]
            else:
                param = self.key.split('_')
                # find i index based on what the key starts with
                i = index_dict[param[0]]
                # find m index based on what the key ends with
                try:
                    m = int(np.where(imass == mass_dict[param[1]])[0])
                except IndexError:
                    m = -1
                val = h5file.get_node('/FittedParams/Fits')[:,:,:,m,i]
                err = h5file.get_node('/FittedParams/Errors')[:,:,:,m,i]


        altitude = alt.flatten()
        latitude = lat.flatten()
        longitude = lon.flatten()
        chi2 = c2.reshape(c2.shape[0], -1)
        fitcode = fc.reshape(fc.shape[0], -1)
        
        value = val.reshape(val.shape[0], -1)
        error = err.reshape(err.shape[0], -1)

        # This accounts for an error in some of the hdf5 files where chi2 is overestimated by 369.
        if np.nanmedian(chi2) > 100.:
            chi2 = chi2 - 369.

        # data_check: 2D boolian array for removing "bad" data
        # Each column correpsonds to a different "check" condition
        # TRUE for "GOOD" point; FALSE for "BAD" point
        # A "good" record that shouldn't be removed should be TRUE for EVERY check condition
        if self.key == 'dens':
            data_check = np.array([np.isfinite(error),error>1.e10,fitcode>0,fitcode<5,chi2<10,chi2>0.1])
        elif 'temp' in self.key:
            data_check = np.array([np.isfinite(error),fitcode>0,fitcode<5,chi2<10,chi2>0.1])
        else:
            data_check = np.array([np.isfinite(value)])

        # If ANY elements of data_check are FALSE, flag index as bad data
        bad_data = np.squeeze(np.any(data_check==False,axis=0,keepdims=True))
        value[bad_data] = np.nan
        error[bad_data] = np.nan
        
        # remove the points where coordinate arrays are NaN
        # these points usually correspond to altitude bins that were specified by the fitter but a particular beam does not reach
        value = value[:,np.isfinite(altitude)]
        error = error[:,np.isfinite(altitude)]
        latitude = latitude[np.isfinite(altitude)]
        longitude = longitude[np.isfinite(altitude)]
        altitude = altitude[np.isfinite(altitude)]
        
        
        # Convert input coordinates to geocentric-spherical
        r, t, p = cc.geodetic_to_spherical(latitude,longitude,altitude/1000.)
        R0 = np.array([r,t,p])

        return utime, R0, value, error
    
    
    def eval_zeroth_order(self,x,data,error):
        """
        Find the coefficients for the zeroth order function

        Parameters:
            x: [ndarray]
                coordinates of the data
            data: [ndarray]
                data values
            error: [ndarray]
                data errors
        """
        # revise so this works in native data coordinates?
        if self.key == 'dens':
            pi = [5.e11,4.0,1.0,1.e10]
            # pi = [5.,4.0,1.0,1.]
            bounds = ([0.,0.,0.,0.],[1.e13,10.,10.,1.e11])
        if self.key == 'temp':
            pi = [400.,1.,1.,10.]
            bounds = ([0.,0.,0.1,0.],[10000,10.,10.,50.])
        p, __ = scipy.optimize.curve_fit(self.zeroth_order, x, data, p0=pi, sigma=error, bounds=bounds)
        self.p0 = p


    def zeroth_order(self,x,*args):
        """
        Evaluate the zeroth order function for any parameter

        Parameters:
            x: [ndarray]
                points at which to evaluate the function

        Returns:
            values of the function at x
        """
        if self.key == 'dens':
            return self.chapman(x,*args)
        if self.key == 'temp':
            return self.sinh(x,*args)

    def chapman(self,x,*args):
        """
        Chapman function (appropriate zeroth order for density profile)

        Parameters:
            x: [ndarray]
                points at which to evaluate the chapman function

        Returns:
            values of the chapman function at x
        """
        if len(args) == 0:
            A, B, H, C = self.p0
        else:
            A, B, H, C = args
        return A*np.exp(1.-(x-B)/H-np.exp(-1.*(x-B)/H))+C


    def sinh(self,x,*args):
        """
        Hyperbolic sine function (appropriate zeroth order for temperature profile)

        Parameters:
            x: [ndarray]
                points at which to evaluate the hyperbolic sine function

        Returns:
            values of the hyperbolic sine at x
        """
        if len(args) == 0:
            A, B, H, C = self.p0
        else:
            A, B, H, C = args
        return A*np.arcsinh((x-B)/H)+C

    def quickplot(self,filename,index):
        """
        Create a scatter plot of the raw data values from a particular record in a file

        Parameters:
            filename: [str]
                filename/path of processed AMISR hdf5 file
            index: [int]
                record index
        """
        R0, value, error = self.get_data(filename,index)
        altitude = (R0[0]-RE)/1000.      # altitude in km
        latitude = 90.-R0[1]*180./np.pi     # latitude in degrees
        longitude = R0[2]*180./np.pi        # longitude in degrees

        # plot collision frequency on a log color scale
        norm = matplotlib.colors.Normalize()
        if self.key.startswith('colfreq'):
            norm = matplotlib.colors.LogNorm()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        f = ax.scatter(latitude,longitude,altitude,c=value,vmin=self.vrange[0],vmax=self.vrange[1],norm=norm)
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Altitude')
        cbl = plt.colorbar(f)
        cbl.set_label('{} ({})'.format(self.name,self.units))
        plt.show()



	



def find_index(filename,time):
    """
    Find the index of a file that is closest to the given time

    Parameters:
        filename: [str]
            filename/path of processed AMISR hdf5 file
        time: [datetime object]
            target time

    Returns:
        rec: [int]
            index of the record that is closest to the target time
        time: [datetime object]
            actual time of the record
    """
    time0 = (time-dt.datetime(1970,1,1)).total_seconds()
    with tables.open_file(filename, 'r') as h5file:
        utime = h5file.get_node('/Time/UnixTime')
        utime = utime.read()

    time_array = np.array([(float(ut[0])+float(ut[1]))/2. for ut in utime])
    rec, time = min(enumerate(time_array), key=lambda t: abs(time0-t[1]))
    return rec, dt.datetime.utcfromtimestamp(time)



def generate_eventlist_standalone(date,radar):
    """
    Generates an eventlist for a single day that includes all events from that day, regardless of the radar mode that
     was run.  This standalone version exists so event lists can be generated without nessisarially initializing the
     Fit class.

    Returns:
        eventlist: [dict]
            list of dictionaries containing the timestamp, file name, radar mode, and index within the file for a 
            particular event
    """

    eventlist = []
    filedir = localpath+'/processed_data/'+radar+'/{:04d}/{:02d}'.format(date.year,date.month)

    num_sep = filedir.count(os.path.sep)
    for root, dirs, files in os.walk(filedir):
        num_sep_this = root.count(os.path.sep)
        if (num_sep + 2 == num_sep_this) and ('.bad' not in root) and ('.noproc' not in root) and ('.old' not in root):
            for file in files:
                if file.endswith('.h5') and ('lp' in file):
                    filename = os.path.join(root,file)
                    print filename
                    filepath = root.split('/')
                    experiment = filepath[num_sep+1]
                    mode = experiment.split('.')[0]

                    data = io_utils.read_partial_h5file(filename,['/Time'])
                    utime = data['/Time']['UnixTime']
                    for i,t in enumerate(utime):
                        dh = (float(t[0])+float(t[1]))/2.
                        tstmp = dt.datetime.utcfromtimestamp(dh)
                        if tstmp >= date and tstmp < date+dt.timedelta(hours=24):
                            eventlist.append({'time':tstmp,'filename':filename,'mode':mode,'index':i})

    # Sort eventlist by timestamp
    eventlist = sorted(eventlist, key=lambda event: event['time'])

    return eventlist


	
def main():

    param = AMISR_param('dens')
    dayfit = Fit(param=param)
#     eventlist = dayfit.generate_eventlist()
    dayfit.fit()
    # dayfit.saveh5(filename='test_out.h5')

    targtime = dt.datetime(year,month,day,hour,minute)
    altitude = 300.
    longitude = -90.
    dayfit.validate(targtime,altitude,longitude)

	
	
if __name__ == '__main__':
	main()
	
