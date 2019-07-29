# Param.py

import numpy as np
import scipy
import tables
import coord_convert as cc

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
#         self.name = PARAMETER_NAME
#         self.max_zint = MAX_Z_INT
#         self.vrange = PARAMETER_RANGE
#         self.units = PARAMETER_UNITS
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


    # This function can be a dictionary
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

    # remove - never use this
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



	


# TODO: Remove this?
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



# def generate_eventlist_standalone(date,radar):
#     """
#     Generates an eventlist for a single day that includes all events from that day, regardless of the radar mode that
#      was run.  This standalone version exists so event lists can be generated without nessisarially initializing the
#      Fit class.

#     Returns:
#         eventlist: [dict]
#             list of dictionaries containing the timestamp, file name, radar mode, and index within the file for a 
#             particular event
#     """

#     eventlist = []
#     filedir = localpath+'/processed_data/'+radar+'/{:04d}/{:02d}'.format(date.year,date.month)

#     num_sep = filedir.count(os.path.sep)
#     for root, dirs, files in os.walk(filedir):
#         num_sep_this = root.count(os.path.sep)
#         if (num_sep + 2 == num_sep_this) and ('.bad' not in root) and ('.noproc' not in root) and ('.old' not in root):
#             for file in files:
#                 if file.endswith('.h5') and ('lp' in file):
#                     filename = os.path.join(root,file)
#                     print filename
#                     filepath = root.split('/')
#                     experiment = filepath[num_sep+1]
#                     mode = experiment.split('.')[0]

#                     data = io_utils.read_partial_h5file(filename,['/Time'])
#                     utime = data['/Time']['UnixTime']
#                     for i,t in enumerate(utime):
#                         dh = (float(t[0])+float(t[1]))/2.
#                         tstmp = dt.datetime.utcfromtimestamp(dh)
#                         if tstmp >= date and tstmp < date+dt.timedelta(hours=24):
#                             eventlist.append({'time':tstmp,'filename':filename,'mode':mode,'index':i})

#     # Sort eventlist by timestamp
#     eventlist = sorted(eventlist, key=lambda event: event['time'])

#     return eventlist
