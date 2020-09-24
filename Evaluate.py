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
    This class evaluates the 3D analytic model to return interpolated parameters within an AMISR FoV.  It is initalized with
    an interpolation coefficient filename, then parameters can be evaluated at a particular time and location.

    Parameters:
        coeff_filename: [str]
            Interpolation coefficient filename
        timetol: [double]
            tolerance for difference in time between the requested datetime and actual times where data's available (seconds)
        timeinterp: [bool]
            flag to detmine whether or not to interpolate in time

    Methods:
        loadh5: loads coefficents from a saved hdf5 file
        getparam: evaluates the model from input points
        check_hull: checks if the input coordinates are within the convex hull
        get_C: return correct C array given time
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


    def loadh5(self,filename=None):
        """
        Loads coefficients from a saved hdf5 file based on the date, radar, and param attributes

        Parameters:
            filename: [str]
                file to load
        """

        with tables.open_file(filename, 'r') as h5file:
            self.Coeffs = h5file.get_node('/Coeffs/C')[:]
            self.Covariance = h5file.get_node('/Coeffs/dC')[:]

            self.time = h5file.get_node('/UnixTime')[:]

            self.hull_vert = h5file.get_node('/FitParams/hull_vert')[:]

            self.config_file_text = h5file.get_node('/ConfigFile/Contents').read()




    def getparam(self,time,gdlat,gdlon,gdalt,calcgrad=False,calcerr=False,check_hull=True):
        """
        Fully calculates parameters and their gradients given input coordinates and a time.
        This is the main function that is used to retrieve reconstructed parameters.

        Parameters:
            time: [datetime]
                time parameters should be evaluated at
            gdlat: [ndarray]
                geodetic latitude (can be multidimensional array)
            gdlon: [ndarray]
                geodetic longitude (can be multidimensional array)
            gdalt: [ndarray]
                geodetic altitude (can be multidimensional array)
            calcgrad: [bool]
                indicates if gradients should be calculated
                True: gradients WILL be calculated
                False (default): gradients WILL NOT be calculated
            calcerr: [bool]
                indicates if errors on parameters and gradients should be calculated
                True: errors WILL be calculated
                False (default): errors WILL NOT be calculated
            check_hull: [bool]
                indicate if input points should be checked to confirm they are within the convex hull of the original data
                True (default): input points WILL be checked
                False: input points will not be checked
       Returns:
            P: [ndarray(npoints)]
                array of the output parameter calculated at all input points
            dP: [ndarray(npoints,3)]
                array of the gradient of the output parameter calculated at all input points
                if calcgrad=False, dP is an array of NAN
        """


        C, dC = self.get_C(time)

        # use einsum to retain shape of input arrays correctly
        A = self.model.basis(gdlat, gdlon, gdalt)
        # print(A.shape, C.shape)
        parameter = np.einsum('...i,i->...',A,C)
        # print(parameter.shape)
        # parameter = np.reshape(np.dot(A,C),np.shape(A)[0])

        if check_hull:
            check = self.check_hull(gdlat, gdlon, gdalt)
            parameter[~check]=np.nan

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
            lat0: [ndarray]
                geodetic latitude (can be multidimensional array)
            lon0: [ndarray]
                geodetic longitude (can be multidimensional array)
            alt0: [ndarray]
                geodetic altitude (can be multidimensional array)
        """
        # this is horribly inefficient, but mostly nessisary?

        hull = ConvexHull(self.hull_vert)
        check = []
        for lat, lon, alt in zip(lat0.ravel(), lon0.ravel(), alt0.ravel()):
            value = False

            x, y, z = pm.geodetic2ecef(lat,lon,alt)
            pnts = np.append(self.hull_vert, np.array([[x,y,z]]), axis=0)
            new_hull = ConvexHull(pnts)
            if np.array_equal(hull.vertices,new_hull.vertices):
                value = True
            check.append(value)
        return np.array(check).reshape(alt0.shape)

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

        # find unix time of requested point
        t0 = (t-dt.datetime.utcfromtimestamp(0)).total_seconds()

        # find time of mid-points
        mt = np.mean(self.time, axis=1)

        try:
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
                if np.abs(mt[i]-t0)>self.timetol:
                    raise IndexError
                C = self.Coeffs[i]
                dC = self.Covariance[i]

        except IndexError:
            raise ValueError('Requested time out of range of data file.')

        return C, dC
