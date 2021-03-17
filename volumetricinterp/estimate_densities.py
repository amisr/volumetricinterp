# Import the Estimate module.
from validate import Validate
import configparser
import numpy as np
import pandas as pd

# Call the Estimate object to get electron density at provided geodetic coordinates:
# density = est_params(time, gdlat, gdlon, gdalt)
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read_file(open(config_file))

# Inputs.
basis_function_shapes = [(5, 5)]
filename = '../../Data/Input/20170616.h5'
regularization_range = range(0, 1)
epochs_array = range(40, 41)
cap_lims = [i for i in range(1, 21)] + [25, 40, 60, 80, 100]

# Times.
# Winter day.
# times = [('2016-11-27T22:45:00', '2016-11-27T22:50:00'),
#         ('2016-11-27T23:05:00', '2016-11-27T23:10:00'),
#         ('2016-11-27T23:25:00', '2016-11-27T23:30:00')]
# times = [('2016-11-27T22:45:00', '2016-11-27T22:48:00')]

# Summer day.
# times = [('2017-06-16T08:00:00', '2017-06-16T08:05:00'),
#         ('2017-06-16T16:00:00', '2017-06-16T16:05:00'),
#         ('2017-06-16T23:00:00', '2017-06-16T23:05:00')]
times = [('2017-06-16T23:02:00', '2017-06-16T23:04:00')]

# Fake day.
# times = [('2016-11-27T23:25:00', '2016-11-27T23:28:00')]

for value in basis_function_shapes:
    maxk, maxl = value
    for e, time in enumerate(times):
        start, end = time
        for r in regularization_range:
            for epochs in epochs_array:
                for cap_lim in cap_lims:
                    # try:

                    # Update settings.
                    regpar = 10 ** r
                    config.set('DEFAULT', 'FILENAME', filename)
                    config.set('MODEL', 'MAXK', str(maxk))
                    config.set('MODEL', 'MAXL', str(maxl))
                    config.set('MODEL', 'CAP_LIM', str(cap_lim))
                    config.set('VALIDATE', 'STARTTIME', str(start))
                    config.set('VALIDATE', 'ENDTIME', str(end))
                    config.set('VALIDATE', 'outpngname', '../../Data/Output/{}_{}.pdf'.format('20170616', cap_lim))
                    with open('loop_config.ini', 'w') as configfile:
                        config.write(configfile)
                    # print('\nProcessing maxk {} maxl {} reg par {} start {} epochs {}'.format(maxk, maxl, regpar, start,
                    #                                                                           epochs))

                    # Interpolate ad create plots.
                    val = Validate('loop_config.ini')
                    loss = val.interpolate(regpar=regpar, epochs=epochs)
                    val.create_plots(cap_lim=cap_lim)

                    # Save coefficient values.
                    # coefficient_vals.append([start, end, r, maxk, maxl, loss])
                    # np_array = np.array(coefficient_vals)
                    # pd.DataFrame(np_array).to_csv("coefficient_values.csv", header=False, index=False)
                    # except:
                    #    continue

from PyPDF2 import PdfFileMerger

pdfs = ['../../Data/Output/{}_{}.pdf'.format('20170616', i) for i in cap_lims]
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(pdf)
merger.write('../../Data/Output/merged.pdf')
merger.close()