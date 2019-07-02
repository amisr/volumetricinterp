# processed_file_list.py

import os
import tables
import re
import datetime as dt
import numpy as np

# set variables for laptop
processed_data_directory = '/Volumes/AMISR_PROCESSED/processed_data/'
risrc_data_directory = os.path.expanduser("~") + '/Desktop/Data/AMISR/RISR-C/'
dbfile = 'RawData_Folders_Exps_Times_by_Radar.h5'


def file_list(date,endtime=None,radars=None,criteria=None):

    tps = (date-dt.datetime(1970,1,1)).total_seconds()
    if endtime:
        tpe = (endtime-dt.datetime(1970,1,1)).total_seconds()
    else:
        tpe = tps+24.*60.*60.

    if not radars:
        radars = ['RISR-N','RISR-C','PFISR']
    
    if not criteria:
        criteria = ['fitcal']
    filelist = []

    crit = criteria
    crit2 = [c for c in criteria if 'cal' not in c]

    for radar in radars:
        print radar

        with tables.open_file(dbfile,'r') as database:
            tn = database.get_node('/Radars/{}'.format(radar.replace('-','')))
            en = database.get_node('/ExpNames/Names')
            st = tn[:]['nStartTime']
            et = tn[:]['nEndTime']
            ya = tn[:]['nyear']
            ma = tn[:]['nmonth']
            da = tn[:]['nday']
            sa = tn[:]['nset']
            index = np.where(~((st<=tps) & (et<=tps)) & ~((st>tpe) & (et>tpe)))[0]
            exp_year = ya[index]
            exp_month = ma[index]
            exp_name = [en[tn[i]['nExpId']][0] for i in index]
            exp_num = ['{:04d}{:02d}{:02d}.{:03d}'.format(ya[i],ma[i],da[i],sa[i]) for i in index]


        for name, number, year, month in zip(exp_name,exp_num,exp_year,exp_month):
            # print number, name
            try:
                filedir = processed_data_directory+radar+'/{:04d}/{:02d}/'.format(year,month)+name+'/'+number
                if 'vvelsLat' in crit:
                    filedir = processed_data_directory+radar+'/{:04d}/{:02d}/'.format(year,month)+name+'/'+number+'/derivedParams/vvelsLat'
                fl = find_h5files_in_dir(filedir,criteria=crit)
                if radar == 'RISR-C' and not fl:
                    filedir = risrc_data_directory+'{:04d}/'.format(year)+number
                    fl = find_h5files_in_dir(filedir,criteria=crit2)
            except OSError:
                continue
            filelist.extend(fl)


    print filelist
    return filelist



def find_h5files_in_dir(directory,criteria=None,verbose=True):
    experiment_format1 = re.compile('^[0-9]{8}\.[0-9]{3}$')
    files = os.listdir(directory)
    if 'vvelsLat' in criteria:
        h5files = [f for f in files if (f.endswith('h5') and experiment_format1.match(f[0:12]))]
    else:
        h5files = [f for f in files if (f.endswith('h5') and experiment_format1.match(f[0:12]) and ('vvelsLat' not in f))]
    good_files = [os.path.join(directory,f) for f in h5files if np.all([c in f for c in criteria])]
    if len(good_files) > 1 and verbose:
        print 'WARNING! More than one data file in {} was found that matchs the given criteria!'.format(directory)
        for f in good_files:
            print os.path.basename(f)

    return good_files




def main():
    date = dt.datetime(2017,10,21)
    file_list(date,radars=['RISR-C'],criteria=['lp','fitcal'])

if __name__ == '__main__':
    main()