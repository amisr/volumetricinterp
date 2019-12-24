# Evaluate.py

import numpy as np
import datetime as dt
import tables
from scipy.spatial import ConvexHull
import io
import configparser
import importlib
import pymap3d as pm


class Evaluate(object):
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

    def __init__(self,coeff_filename,timetol=60.,timeinterp=False):

        self.timetol = timetol
        self.timeinterp = timeinterp

        # load coefficient file
        self.loadh5(filename=coeff_filename)

        config_file = io.StringIO(self.config_file_text.decode('utf-8'))

        config = configparser.ConfigParser()
        config.read_file(config_file)
        self.model_name = config.get('MODEL', 'MODEL')

        config_file.seek(0)
        m = importlib.import_module(self.model_name)
        self.model = m.Model(config_file)


    def loadh5(self,filename=None,raw=False):
        """
        Loads coefficients from a saved hdf5 file based on the date, radar, and param attributes

        Parameters:
            filename: [str]
                file to load
            raw: Optional [bool]
                flag to indicate if the raw data should be loaded or not
                default is False (raw data will NOT be loaded)
        """

        with tables.open_file(filename, 'r') as h5file:
            self.Coeffs = h5file.get_node('/Coeffs/C')[:]
            self.Covariance = h5file.get_node('/Coeffs/dC')[:]

            self.time = h5file.get_node('/UnixTime')[:]

            self.hull_vert = h5file.get_node('/FitParams/hull_vert')[:]

            self.config_file_text = h5file.get_node('/ConfigFile/Contents').read()




    def getparam(self,time,gdlat,gdlon,gdalt,calcgrad=False,calcerr=False):
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


        # Rshape = R0.shape
        # R0 = R0.reshape(Rshape[0], -1)
        # check = self.check_hull(R0)

        C, dC = self.get_C(time)

        # use einsum to retain shape of input arrays correctly
        A = self.model.basis(gdlat, gdlon, gdalt)
        parameter = np.reshape(np.dot(A,C),np.shape(A)[0])

        return parameter

        # out = self.eval_model(R0,C)
        # parameter = out['param']
        # parameter[~check] = np.nan
        # P = parameter
        #
        # if calcgrad:
        #     gradient = out['grad']
        #     gradient = self.inverse_transform(R,gradient)
        #     gradient[~check] = [np.nan,np.nan,np.nan]
        #     dP = np.array([gradient[:,0],gradient[:,1],gradient[:,2],np.zeros(len(parameter))]).T
        #     return P, dP
        #
        # if calcerr:
        #     err = out['err']
        #     err[~check] = np.nan
        #     if calcgrad:
        #         graderr = out['gerr']
        #         graderr = self.inverse_transform(R,graderr,self.cp)
        #         graderr[~check] = [np.nan,np.nan,np.nan]
        #     return P, dP, err, graderr
        #
        # else:
        #     return P.reshape(tuple(list(Rshape)[1:]))





    def check_hull(self,lat0,lon0,alt0):
        """
        Check if the input points R0 are within the convex hull of the original data.

        Parameters:
            R0: [ndarray(3,npoints)]
                array of input points in geocentric coordinates
                R = [[r coordinates (m)],[theta coordinates (rad)],[phi coordinates (rad)]]
                if input points are expressed as a list of r,t,p points, eg. points = [[r1,t1,p1],[r2,t2,p2],...], R = np.array(points).T

        """
        # x, y, z = cc.spherical_to_cartesian(self.hull_v[:,0],self.hull_v[:,1],self.hull_v[:,2])
        # x, y, z = cc.geodetic_to_cartesian(self.hull_v[:,0],self.hull_v[:,1],self.hull_v[:,2])
        # x, y, z = pm.geodetic2ecef(self.hull_vert[:,0],self.hull_v[:,1],self.hull_v[:,2])
        # vert_cart = np.array([x,y,z]).T

        hull = ConvexHull(sel.hull_vert)
        check = []
        # for R in R0.T:
        for lat, lon, alt in zip(lat0, lon0, alt0):
            value = False

            # x, y, z = cc.spherical_to_cartesian(R[0],R[1],R[2])
            # x, y, z = cc.geodetic_to_cartesian(R[0],R[1],R[2])
            x, y, z = pm.geodetic2ecef(lat,lon,alt)
            pnt = np.array([[x,y,z]])

            pnts = np.append(self.hull_vert,pnt,axis=0)
            nh = ConvexHull(pnts)
            if np.array_equal(hull.vertices,nh.vertices):
                value = True
            check.append(value)
        return np.array(check)

    def get_C(self, t):
        """
        Return values for C and dC based on a given time and whether or not time intepolation has been selected

        Parameters:
            t: [datetime object]
                target time

        Returns:
            C: [ndarray(nbasis)]
                coefficient array
            dC: [ndarray(nbasisxnbasis)]
                covariance matrix
        """

        print(t)
        # find unix time of requested point
        t0 = (t-dt.datetime.utcfromtimestamp(0)).total_seconds()

        # find time of mid-points
        mt = np.array([(float(ut[0])+float(ut[1]))/2. for ut in self.time])

        if t0<np.min(mt) or t0>np.max(mt):
            print('Time out of range!')
            C = np.full(self.nbasis,np.nan)
            dC = np.full((self.nbasis,self.nbasis),np.nan)

        else:
            if self.timeinterp:
                # find index of neighboring points
                i = np.argwhere((t0>=mt[:-1]) & (t0<mt[1:])).flatten()[0]
                # calculate T
                T = (t0-mt[i])/(mt[i+1]-mt[i])
                # calculate interpolated values
                C = (1-T)*self.Coeffs[i,:] + T*self.Coeffs[i+1,:]
                dC = (1-T)*self.Covariance[i,:,:] + T*self.Covariance[i+1,:,:]

            else:
                i = np.argmin(np.abs(mt-t0))
                C = self.Coeffs[i]
                dC = self.Covariance[i]

        return C, dC

    def find_index(self, t):
        """
        Find the index of a file that is closest to the given time

        Parameters:
            t: [datetime object]
                target time

        Returns:
            rec: [int]
                index of the record that is closest to the target time
        """

        time0 = (t-dt.datetime.utcfromtimestamp(0)).total_seconds()
        time_array = np.array([(float(ut[0])+float(ut[1]))/2. for ut in self.time])
        rec = np.argmin(np.abs(time_array-time0))

        return rec
