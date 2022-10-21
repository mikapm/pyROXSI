"""
Pre-process Nortek Vector ADV raw data. 
Remove bad measurements based on correlations and despiking. 
Save Level1 products as netcdf.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as DT

class ADV():
    """
    Main ADV class
    """
    def __init__(self, datadir, mooring_id):
        """
        Initialize ADV class.

        Parameters:
            datadir; str - Path to raw data directory
            mooring_id - str; ROXSI 2022 SSA mooring ID
        """
        self.datadir = datadir
        self.mid = mooring_id
        
    def loaddata(self):
        """
        Read raw data from chosen mooring ID into pandas
        dataframe. Header info can be found in .hdr files.
        """
        # Define column names (see .hdr files for info)
        cols_dat = ['burst', 'ensemble', 'u', 'v', 'w', 
                    'ampl1', 'ampl2', 'ampl3', 'SNR1', 'SNR2', 'SNR3',
                    'corr1', 'corr2', 'corr3', 'pressure', 'ai1', 'ai2',
                    'checksum',
                    ]
        # Find correct .dat file with data time series
        fn_dat = os.path.join(self.datadir, '{}.dat'.format(self.mid))
        # Read data into pandas dataframe
        data = pd.read_table(fn_dat, names=cols_dat, header=None,
            delimiter=' ', skipinitialspace=True).set_index('burst')

        # Read sensor info into another dataframe from .sen file
        cols_sen = ['month', 'day', 'year', 'hour', 'minute', 'second', 
                    'burst', 'no_vel_samples', 
                    'noise_amp1', 'noise_amp2', 'noise_amp3',
                    'noise_corr1', 'noise_corr2', 'noise_corr3', 
                    'dist_pr1_s', 'dist_pr2_s', 'dist_pr3_s', 'dist_svol_s',
                    'dist_pr1_e', 'dist_pr2_e', 'dist_pr3_e', 'dist_svol_e',
                    ]
        fn_sen = os.path.join(self.datadir, '{}.sen'.format(self.mid))
        sen = pd.read_table(fn_sen, names=cols_sen, header=None,
            delimiter=' ', skipinitialspace=True)
        # Parse dates from relevant columns
        sen['time'] = pd.to_datetime(sen[['year','month','day','hour','minute','second']])
        # Set timestamp as index
        sen.set_index('time', inplace=True)

        return data, sen


# Main script
if __name__ == '__main__':
    """
    Test script using synthetic example data.
    """
    import os
    import sys
    import glob
    import pandas as pd
    from scipy.io import loadmat
    from argparse import ArgumentParser

    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data root directory"),
                type=str,
                default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/Vectors',
                )
        parser.add_argument("-mid", 
                help=("Mooring ID"),
                type=str,
                choices=['C1v01', 'C5v02'],
                default='C1v01',
                )
        parser.add_argument("-M", 
                help=("Segment window length (number of samples)"),
                type=int,
                default=512,
                )
        parser.add_argument("-fmin", 
                help=("Min. frequency for attenuation correction"),
                type=float,
                default=0.05,
                )
        parser.add_argument("-fmax", 
                help=("Max. frequency for attenuation correction"),
                type=float,
                default=0.33,
                )

        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Initialize ADV class and read raw data
    adv = ADV(datadir=args.dr, mooring_id=args.mid)
    print('Reading raw data .dat file ...')
    data, sen = adv.loaddata()
