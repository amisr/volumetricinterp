# Fit.py

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import configparser
import scipy
import scipy.integrate
from scipy.spatial import ConvexHull
import tables
import importlib
import os
import pymap3d as pm
import keras as k


class LossHistory(k.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class Interpolate(object):
    """
    This class performs the least-squares fit of the data to the 3D analytic model to find the coefficient vector for
    the model. It also handles calculating regularization matrices and parameters if necessary.

    Parameters:
        config_file: [str]
            config file that specifies the fit parameters/options

    Attributes:
        date: date the model is fit for
        regularization_list: list of regularization methods to be used for the fit
        time: list of datetime objects corresponding to each event fit
        Coeffs: list of coefficient vectors corresponding to each event fit
        Covariance: list of covariance matrices corresponding to each event fit
        chi_sq: list of chi squared (goodness of fit test) corresponding to each event fit
        cent_point: list of the center points corresponding ot each event fit
        hull_v: list of hull vertices corresponding to each event fit
        raw_coords: list of the coordinates for the raw data corresponding to each event fit
        raw_data: list of the data value for the raw data corresponding to each event fit
        raw_error: list of the error value for the raw data corresponding to each event fit
        raw_filename: list of the raw data file names corresponding to each event fit
        raw_index: list of the index within each raw data file corresponding ot each event fit

    Methods:
        read_config: read fit specifications from config file
        find_reg_params: finds the regularization parameters
        chi2: finds the regularization parameter using the chi2-nu method
        chi2objfunct: objective function for the chi2-nu method
        gcv: finds the regularization parameter using generalized cross validation
        gcvobjfunct: the objective function for generalized cross validation
        manual: finds the regularization parameter via values manually hardcoded in function
        prompt: finds the regularization parameter via command line prompts for user input
        eval_C: evaluates the coefficient vector and covariance matrix
        fit: performs fits to the 3D analytic model for data from a series of events
        get_data: read data from input file
        saveh5: save the results of fit() to an output hdf5 file

    """

    def __init__(self, config_file):

        # Read config file, and set parameters.
        self.configfile = config_file
        config = configparser.ConfigParser()
        config.read_file(open(self.configfile))
        self.regularization_list = list(filter(None, config.get('DEFAULT', 'REGULARIZATION_LIST').split(',')))
        self.reg_method = config.get('DEFAULT', 'REGULARIZATION_METHOD')
        self.filename = config.get('DEFAULT', 'FILENAME')
        self.outputfilename = config.get('DEFAULT', 'OUTPUTFILENAME')
        self.param = config.get('DEFAULT', 'PARAM')
        self.errlim = [float(i) for i in config.get('DEFAULT', 'ERRLIM').split(',')]
        self.chi2lim = [float(i) for i in config.get('DEFAULT', 'CHI2LIM').split(',')]
        self.goodfitcode = [int(i) for i in config.get('DEFAULT', 'GOODFITCODE').split(',')]
        self.model_name = config.get('MODEL', 'NAME')

        # Import and set model.
        m = importlib.import_module('.models.' + self.model_name, package='volumetricinterp')
        self.model = m.Model(open(self.configfile))

    # inform regularization parameter by beam seperation?
    # the more beams, the more features can be resolved?

    def find_reg_param(self, A, b, W, reg_matrices, method='chi2'):
        """
        Find the regularization parameters.  A number of different methods are provided for this (se the reg_methods
        dictionary).
            - chi2: enforce the statistical condition chi2 = nu (e.g. Nicolls et al., 2014)
            - gcv: use Generalized Cross validation
            - manual: hardcode in regularization parameters
            - prompt: ask user for regularization parameters via a prompt

        Parameters:
            A: [ndarray(npoints,nbasis)]
                Array of basis functions evaluated at all input points.
            b: [ndarray(npoints)]
                Array of raw input data.
            W: [ndarray(npoint)]
                Array of errors on raw input data.
            reg_matrices: [dict]
                Dictionary of all the regularization matrices needed based on regularization_list.
            method: [str]
                Method that should be used to find regularization parameter (valid options include 'chi2', 'gcv',
                'manual', 'prompt') Default: chi2.
        Returns:
            reg_params: [dict]
                Dictionary of regularization parameters needed based on regularization_list.
        Notes:
            - Currently, the automated methods of calculating regularization parameters (chi2 and gcv) handle multiple
            parameters by solving for each parameter individually while assuming all other parameters are zero. This is
            non-ideal and tends to lead to over smoothing, but it's the best approach we have right now because most
            standard methods of selecting regularization parameters only consider one condition (one condition = one
            unknown). There is some literature on how to find multiple regularization parameters simultaniously, but
            have not had luck implementing any of these techniques.
            - Regularization (particularly choosing an "appropriate" regularization parameter) is VERY finiky. This
            function attempts to have reasonable defaults and options, but it is HIGHLY unlikely these will work well
            in all cases. The author has tried to make the code flexible so new methods can be added "easily", but
            apologizes in advance for the headache this will undoubtably cause.
        """

        # Define reg_methods dictionary
        reg_methods = {'chi2': self.chi2, 'gcv': self.gcv, 'manual': self.manual, 'prompt': self.prompt}

        reg_params = {}
        for rl in self.regularization_list:
            try:
                reg_params[rl] = reg_methods[method](A, b, W, reg_matrices, rl)
            except ValueError as err:
                print(err)
                print('Returning NANs for regularization parameters.')
                reg_params[rl] = np.nan

        return reg_params

    # consider modularizing out regularization Methods
    # these are probably things that will be changing/evolving frequently
    def chi2(self, A, b, W, reg_matrices, reg):
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
        # nu = A.shape[0]-A.shape[1]
        nu = A.shape[0]
        # nu = 0.

        # alpha = np.arange(-100.,0.,0.1)
        # val = [self.chi2objfunct(a,A,b,W,reg_matrices,nu,reg) for a in alpha]
        # import matplotlib.pyplot as plt
        # plt.plot(alpha,val)
        # plt.show()

        # Use the Brent (1973) method to find the root within the bracketing interval found above
        solution = scipy.optimize.brentq(self.chi2objfunct, -10., 10., args=(A, b, W, reg_matrices, nu, reg), disp=True)

        reg_param = np.power(10., solution)

        return reg_param

    def chi2objfunct(self, alpha, A, b, W, reg_matrices, nu, reg):
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
                reg_params[rl] = np.power(10., alpha)
            else:
                reg_params[rl] = 0.

        # Evaluate coefficient vector
        C = self.eval_C(A, b, W, reg_matrices, reg_params)

        # Caluclate chi2
        val = np.einsum('ji,i->j', A, C)
        chi2 = sum((val - b) ** 2 * W)
        # print(chi2,nu)

        return chi2 - nu

    def gcv(self, A, b, W, reg_matrices, reg):
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

        alpha = np.arange(-40., -20., 0.1)
        val = [self.gcvobjfunct(a, A, b, W, reg_matrices, reg) for a in alpha]
        plt.plot(alpha, val)
        plt.show()

        # # Set initial guess
        # alpha0 = -20.
        #
        # # Use the Nelder-Mead method to find the minimum of the GCV objective function
        # solution = scipy.optimize.minimize(self.gcvobjfunct,alpha0,args=(A,b,W,reg_matrices,reg),method='Nelder-Mead')
        # if not solution.success:
        #     raise ValueError('Minima of GCV function could not be found')
        #
        # reg_param = np.power(10.,solution.x[0])
        reg_param = 0.

        return reg_param

    def gcvobjfunct(self, alpha, A0, b0, W0, reg_matrices, reg):
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
                reg_params[rl] = np.power(10., alpha)
            else:
                reg_params[rl] = 0.

        residuals = []
        for i in range(10):
            ri = np.random.randint(len(b0), size=int(0.1 * len(b0)))
            A = np.delete(A0, ri, 0)
            b = np.delete(b0, ri, 0)
            W = np.delete(W0, ri, 0)

            # Evaluate coefficient vector
            C = self.eval_C(A, b, W, reg_matrices, reg_params)

            # Caluclate chi2
            val = np.einsum('ji,i->j', A0, C)
            residuals.append(sum((val - b0) ** 2 * W0))

        # for i in range(len(b0)):
        #     print(alpha,i,'/',len(b0))
        #     # Pull one data point out of arrays
        #     # data point in question:
        #     Ai = A0[i,:]
        #     bi = b0[i]
        #     Wi = W0[i]
        #     # arrays minus one data point:
        #     A = np.delete(A0,i,0)
        #     b = np.delete(b0,i,0)
        #     W = np.delete(W0,i,0)
        #
        #     # Evaluate coefficient vector
        #     C = self.eval_C(A,b,W,reg_matrices,reg_params)
        #
        #     # Calculate residual for the data point not included in the fit
        #     val = np.squeeze(np.dot(Ai,C))
        #     residuals.append((val-bi)**2*Wi)

        return np.mean(residuals)

    def manual(self, A, b, W, Omega, Psi, Tau, reg):
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

    def prompt(self, A, b, W, Omega, Psi, Tau, reg):
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

    def compute_hull(self, lat, lon, alt):
        """
        Compute the convex hull that contains the original data to save to the output file.

        Parameters:
            lat: [ndarray]
                geodetic latitude
            lon: [ndarray]
                geodetic longitude
            alt: [ndarray]
                geodetic altitude
        """

        x, y, z = pm.geodetic2ecef(lat, lon, alt)
        R_cart = np.array([x, y, z]).T

        chull = ConvexHull(R_cart)
        self.hull_vert = R_cart[chull.vertices]

    def eval_C(self, A, b, W, reg_matrices, reg_params, calccov=False, epochs=10):
        """
        Evaluate the coefficient array, C.

        Parameters:
            A: [ndarray(npoints,nbasis)]
                Array of basis functions evaluated at all input points.
            b: [ndarray(npoints)]
                Array of raw input data.
            W: [ndarray(npoint)]
                Array of errors on raw input data.
            reg_matrices: [dict]
                Dictionary of all the regularization matrices needed based on regularization_list.
            reg_params: [dict]
                Dictionary of all the regularization parameter values needed based on regularization_list.
            calccov: [bool]
                Boolean value whether or not to calculate the covariance matrix.
        Returns:
            C: [ndarray(nbasis)]
                Array of basis function coefficients.
            dC: Optional [ndarray(nbasis,nbasis)]
                Covariance matrix for basis functions.
        """

        AWA = np.einsum('ji,j,jk->ik', A, W, A)
        X = AWA.copy()
        y = np.einsum('ji,j,j->i', A, W, b)

        # for reg in self.regularization_list:
        #    X = X + reg_params[reg] * reg_matrices[reg]
        # C = np.squeeze(scipy.linalg.lstsq(X, y, overwrite_a=True, overwrite_b=True)[0])
        C = np.zeros((1, 125))

        # Neural network.
        N = X.shape[0]
        X_nn = A  # *W[:, np.newaxis]
        y_nn = b  # *W
        # kernel_regularizer = k.regularizers.l2(reg_params['curvature'])
        kernel_regularizer = None
        network = k.Sequential([k.layers.Dense(units=1, input_shape=[N], kernel_regularizer=kernel_regularizer)])
        network.compile(optimizer=k.optimizers.Adam(learning_rate=0.01), loss='mse')
        checkpoint = k.callbacks.ModelCheckpoint('nn_weights.h5', verbose=0, monitor='loss', save_best_only=True,
                                                 mode='auto')
        history = LossHistory()
        network.fit(X_nn, y_nn, epochs=epochs, callbacks=[checkpoint, history], verbose=0)
        loss = min(history.losses)

        # Compute loss.
        # loss = sum((np.squeeze(np.dot(A, C)) - np.squeeze(b)) ** 2 * np.squeeze(W)) # Chi 2.

        if calccov:
            H = scipy.linalg.pinv(X)
            dC = np.einsum('ij,jk,kl->il', H, AWA, H)
            return C, dC, loss
        else:
            dC = np.zeros((N, N))
            return C, dC, loss

    def calc_coeffs(self, starttime=None, endtime=None, regpar=0, epochs=10):
        """
        Perform fit on every record in file.

        Parameters:
            starttime:
            endtime:
        """

        Coeffs = []
        Covariance = []
        chi_sq = []

        # print('Evaluating Regularization matrices. This may take a few minutes.')
        reg_matricies = {}
        for reg in self.regularization_list:
            try:
                reg_matricies[reg] = self.model.eval_reg_matricies[reg]()
            except KeyError as e:
                print('WARNING: The model {} does not support {} regularization!'.format(self.model_name, reg))
                print('If you would like to use {} regularization, please modify {}.py so that it includes functions '
                      'to calculate the appropriate regularization matrix.'.format(reg, self.model_name))
                raise e

        # Read data from AMISR fitted file.
        utime, lat, lon, alt, value, error = self.read_datafile(self.filename)

        # Compute hull that surrounds data.
        self.compute_hull(lat, lon, alt)

        # If a starttime and endtime are given, rewrite utime, value, and error arrays so they only contain records
        # between those two times.
        if starttime and endtime:
            idx = np.argwhere((utime[:, 0] >= (starttime - dt.datetime.utcfromtimestamp(0)).total_seconds()) & (
                    utime[:, 1] <= (endtime - dt.datetime.utcfromtimestamp(0)).total_seconds())).flatten()
            utime = utime[idx, :]
            value = value[idx]
            error = error[idx]

        # Loop over every record and calculate the coefficients if modeling time variation, this loop will change?
        for u, (ut, ne0, er0) in enumerate(zip(utime, value, error)):
            # print(dt.datetime.utcfromtimestamp(ut[0]))

            # Remove any points with NaN values. Any NaN in the input value array will result in all fit coefficients
            # being NaN.
            lat0 = lat[np.isfinite(ne0)]
            lon0 = lon[np.isfinite(ne0)]
            alt0 = alt[np.isfinite(ne0)]
            er0 = er0[np.isfinite(ne0)]
            ne0 = ne0[np.isfinite(ne0)]

            # Start by normalizing the values and errors.
            ne0 = ne0 / 1e11
            er0 = er0 / 1e11

            # Define matrices. W (Errors matrix), b (Values matrix), A (Basis function matrix).
            W = np.array(er0 ** (-2))
            b = ne0
            A = self.model.basis(lat0, lon0, alt0)

            # # Evaluate Tau regularization matrix - this is based on data and must be in loop
            # if '0thorder' in self.regularization_list:
            #
            #     # try:
            #     # self.param.eval_zeroth_order(R[0],data,error)
            #     self.param.eval_zeroth_order(R[0],ne0,er0)
            #     tau = self.eval_tau(self.param.zeroth_order)
            #     # except RuntimeError:
            #     #     tau = np.full((self.nbasis,),np.nan)
            #     reg_matrices['Tau'] = tau
            #     # reg_matrices['Tau'] = self.eval_tau(R,ne0,er0)

            # # if regularization matricies are NaN, skip this record - fit can not be computed
            # if np.any([np.any(np.isnan(v.flatten())) for v in reg_matricies.values()]):
            #     # NaNs in C, dC, c2
            #     Coeffs.append(np.full(self.model.nbasis, np.nan))
            #     Covariance.append(np.full((self.model.nbasis,self.model.nbasis), np.nan))
            #     chi_sq.append(np.nan)
            #     continue

            # # define matricies
            # W = np.array(er0**(-2))[:,None]
            # b = ne0[:,None]
            # A = self.model.eval_basis(R)

            # Calculate regularization parameters.
            # print('A', A)
            # print('b', b)
            # print('W', W)
            # reg_params = self.find_reg_param(A, b, W, reg_matricies, method=self.reg_method)
            reg_params = {'curvature': regpar}
            # print('Regularization parameters:', reg_params)

            # If regularization parameters are NaN, skip this record - fit can not be computed.
            if np.any(np.isnan([v for v in reg_params.values()])):
                # NaNs in C, dC, c2
                Coeffs.append(np.full(self.model.nbasis, np.nan))
                Covariance.append(np.full((self.model.nbasis, self.model.nbasis), np.nan))
                chi_sq.append(np.nan)
                continue

            # Calculate coefficients, covariance matrix, and loss.
            C, dC, loss = self.eval_C(A, b, W, reg_matricies, reg_params, calccov=False, epochs=epochs)

            # Append lists
            Coeffs.append(C)
            Covariance.append(dC)
            chi_sq.append(loss)

        # Average chi2.
        # print('Average chi 2:', sum(chi_sq) / len(chi_sq))

        # Set new variables.
        self.time = utime
        self.Coeffs = np.array(Coeffs)
        self.Covariance = np.array(Covariance)
        self.chi_sq = np.array(chi_sq)

        return sum(chi_sq) / len(chi_sq)

    def read_datafile(self, filename):
        """
        Read a processed AMISR hdf5 file and return the time, coordinates, values, and errors as arrays.

        Parameters:
            filename: [str]
                filename/path of processed AMISR hdf5 file

        Returns:
            utime: [ndarray (nrecordsx2)]
                start and end time of each record (Unix Time)
            latitude: [ndarray (npoints)]
                geodetic latitude of each point
            longitude: [ndarray (npoints)]
                geodetic longitude of each point
            altitude: [ndarray (npoints)]
                geodetic altitude of each point
            value: [ndarray (nrecordsxnpoints)]
                parameter value of each data point
            error: [ndarray (nrecordsxnpoints)]
                error in parameter values
        """

        index_dict = {'frac': 0, 'temp': 1, 'colfreq': 2}
        mass_dict = {'O': 16, 'O2': 32, 'NO': 30, 'N2': 28, 'N': 14}

        with tables.open_file(filename, 'r') as h5file:

            utime = h5file.get_node('/Time/UnixTime')[:]
            alt = h5file.get_node('/Geomag/Altitude')[:]
            lat = h5file.get_node('/Geomag/Latitude')[:]
            lon = h5file.get_node('/Geomag/Longitude')[:]
            c2 = h5file.get_node('/FittedParams/FitInfo/chi2')[:]
            fc = h5file.get_node('/FittedParams/FitInfo/fitcode')[:]
            imass = h5file.get_node('/FittedParams/IonMass')[:]

            if self.param == 'dens':
                val = h5file.get_node('/FittedParams/Ne')[:]
                err = h5file.get_node('/FittedParams/dNe')[:]
            else:
                param = self.param.split('_')
                # Find i index based on what the key starts with.
                i = index_dict[param[0]]
                # Find m index based on what the key ends with.
                try:
                    m = int(np.where(imass == mass_dict[param[1]])[0])
                except IndexError:
                    m = -1
                val = h5file.get_node('/FittedParams/Fits')[:, :, :, m, i]
                err = h5file.get_node('/FittedParams/Errors')[:, :, :, m, i]

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

        # data_check: 2D boolian array for removing "bad" data.
        # Each column correpsonds to a different "check" condition.
        # TRUE for "GOOD" point; FALSE for "BAD" point.
        # A "good" record that shouldn't be removed should be TRUE for EVERY check condition.
        data_check = np.array(
            [error > self.errlim[0], error < self.errlim[1], chi2 > self.chi2lim[0], chi2 < self.chi2lim[1],
             np.isin(fitcode, self.goodfitcode)])

        # If ANY elements of data_check are FALSE, flag index as bad data.
        bad_data = np.squeeze(np.any(data_check == False, axis=0, keepdims=True))
        value[bad_data] = np.nan
        error[bad_data] = np.nan

        # Remove the points where coordinate arrays are NaN.
        value = value[:, np.isfinite(altitude)]
        error = error[:, np.isfinite(altitude)]
        latitude = latitude[np.isfinite(altitude)]
        longitude = longitude[np.isfinite(altitude)]
        altitude = altitude[np.isfinite(altitude)]

        # Return.
        return utime, latitude, longitude, altitude, value, error

    def saveh5(self):
        # TODO: compress arrays to reduce coefficient file size
        """
        Saves coefficients to a hdf5 file

        Parameters:
            None
        """

        with tables.open_file(self.outputfilename, 'w') as h5out:
            cgroup = h5out.create_group('/', 'Coeffs', 'Dataset')
            fgroup = h5out.create_group('/', 'FitParams', 'Dataset')
            dgroup = h5out.create_group('/', 'RawData', 'Dataset')

            h5out.create_array('/', 'UnixTime', self.time)
            h5out.create_array(cgroup, 'C', self.Coeffs)
            h5out.create_array(cgroup, 'dC', self.Covariance)

            h5out.create_array(fgroup, 'reglist', self.regularization_list)
            h5out.create_array(fgroup, 'regmethod', self.reg_method.encode('utf-8'))
            h5out.create_array(fgroup, 'chi2', self.chi_sq)

            h5out.create_array(fgroup, 'hull_vert', self.hull_vert)

            h5out.create_array(dgroup, 'filename', self.filename.encode('utf-8'))

            # config file
            Path = os.path.dirname(os.path.abspath(self.configfile))
            Name = os.path.basename(self.configfile)
            with open(self.configfile, 'r') as f:
                Contents = ''.join(f.readlines())

            h5out.create_group('/', 'ConfigFile')
            h5out.create_array('/ConfigFile', 'Name', Name.encode('utf-8'))
            h5out.create_array('/ConfigFile', 'Path', Path.encode('utf-8'))
            h5out.create_array('/ConfigFile', 'Contents', Contents.encode('utf-8'))
