"""
Generate Level 1 netcdf files of raw RBR pressure (soloD)
and pressure-temperature (duetDT) sensors at Asilomar 2022
small-scale array.
"""

# Imports
import os
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime as DT
from scipy.io import loadmat
from scipy.signal import detrend
import matplotlib.pyplot as plt
import ipympl
# Interactive plots
%matplotlib widget 
# from roxsi_pyfuns import transfer_functions as tf
from roxsi_pyfuns import transfer_functions as tf
from roxsi_pyfuns import wave_spectra as ws

class RBR():
    """
    Main RBR data class.
    """
    def __init__(self, datadir, ser, zp=0.02, fs=16, burstlen=1200, 
                 outdir=None, mooring_info=None, patm=None, bathy=None, 
                 instr='RBRSoloD'):
        """
        Initialize RBR class.

        Parameters:
            datadir; str - Path to raw data (.mat files) directory
            ser - str; Signature serial number
            zp - scalar; height of sensor above seabed (m)
            fs - scalar; sampling frequency (Hz)
            magdec - scalar; magnetic declination (deg E) of location
            beam_ang - scalar; beam angle in degrees (25 for Sig1000)
            binsz - scalar; velocity bin size in meters
            outdir - str; if None, save output files in self.datadir,
                     else, specify outdir
            mooring_info - str; path to mooring info excel file (optional)
            patm - pd.DataFrame time series of atmospheric pressure (optional)
            bathy - xr.Dataset of SSA bathymetry (optional)
            instr - str; instrument name
        """
        self.datadir = datadir
        self.ser = str(ser)
        # Find all .mat files for specified serial number
        self._fns_from_ser()
        self.zp = zp
        self.fs = fs
        self.dt = 1 / self.fs # Sampling rate (sec)
        self.magdec = magdec
        self.beam_ang = beam_ang
        self.binsz = binsz
        self.instr = instr
        if outdir is None:
            # Use datadir as outdir
            self.outdir = datadir
        else:
            # Use specified outdir
            self.outdir = outdir
        # Read mooring info excel file if provided
        if mooring_info is not None:
            self.dfm = pd.read_excel(mooring_info, 
                                     parse_dates=['record_start', 'record_end'])
            # Get mooring ID
            key = 'mooring_ID'
            self.mid = self.dfm[self.dfm['serial_number'].astype(str)==self.ser][key].item()
            key = 'mooring_ID_long'
            self.midl = self.dfm[self.dfm['serial_number'].astype(str)==self.ser][key].item()
            # Get record start and end timestamps
            key = 'record_start'
            self.t0 = self.dfm[self.dfm['serial_number'].astype(str)==self.ser][key].item()
            key = 'record_end'
            self.t1 = self.dfm[self.dfm['serial_number'].astype(str)==self.ser][key].item()
        else:
            self.dfm = None # No mooring info dataframe
            self.mid = None # No mooring ID number
            self.t0 = None # No start time given
            self.t1 = None # No end time given
        # Atmospheric pressure time series, if provided
        self.patm = patm
        # Bathymetry dataset if provided
        self.bathy = bathy


if __name__ == '__main__':
    """
    Main script.
    """
    from argparse import ArgumentParser

    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data root directory"),
                type=str,
                # default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/Signatures',
                default=r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray',
                )
        parser.add_argument("-ser", 
                help=('Instrument serial number. To loop through all, select "ALL".'),
                type=str,
                choices=['103088', '103094', '103110', '103063', '103206', 'ALL'],
                default='103094',
                )
        parser.add_argument("-M", 
                help=("Pressure transform segment window length"),
                type=int,
                default=512*8,
                )
        parser.add_argument("-fmin", 
                help=("Min. frequency for pressure attenuation correction"),
                type=float,
                default=0.05,
                )
        parser.add_argument("-fmax", 
                help=("Max. frequency for pressure attenuation correction"),
                type=float,
                default=0.33,
                )
        parser.add_argument("--savefig", 
                help=("Make and save figures?"),
                action="store_true",
                )
        parser.add_argument("--overwrite_fig", 
                help=("Overwrite existing figures?"),
                action="store_true",
                )
        parser.add_argument("--overwrite_nc", 
                help=("Overwrite existing netcdf files?"),
                action="store_true",
                )

        return parser.parse_args(**kwargs)

    # Define paths and load data
    datestr = '20220630'
    # SSA mooring ID's for DuetDT: [L2vp, L4vp, C15p, C2vp, C3vp, C4vp]
    mooring_id = 'C15p'
    sensor_id = 'duetDT'
    rootdir = '/media/mikapm/ROXSI/Asilomar2022/SmallScaleArray/RBRDuetDT/Level1/'


    # Data directory
    if sensor_id == 'duetDT':
        datadir = '/home/mikapm/ROXSI/Asilomar2022/SmallScaleArray/RBRDuetDT/Level1/'
    elif sensor_id == 'soloD':
        rootdir = '/home/mikapm/ROXSI/Asilomar2022/SmallScaleArray/RBRSoloD/Level1/'
        rootdir = '/home/mikapm/ROXSI/Asilomar2022/SmallScaleArray/RBRSoloD/Level1/'
    datadir = os.path.join(rootdir, 'mat')
    # Output figure directory
    figdir = os.path.join(rootdir, 'p2eta_figs')

    # Load ROXSI pressure sensor time series
    fn_mat = glob.glob(os.path.join(datadir, 'roxsi_{}_L1_{}_*_{}.mat'.format(
        sensor_id, mooring_id, datestr)))[0]
    print('Loading pressure sensor mat file {}'.format(os.path.basename(fn_mat)))
    mat = loadmat(fn_mat)
    # Read pressure time series and timestamps
    pt = np.array(mat['DUETDT']['Pwater'].item()).squeeze()
    time_mat = np.array(mat['DUETDT']['time_dnum'].item()).squeeze()
    time_ind = pd.to_datetime(time_mat-719529,unit='d') # Convert timestamps
    # Read sampling frequency and sensor height above seabed
    fs = int(mat['DUETDT']['sample_freq'].item()[0].split(' ')[0])
    zp = mat['DUETDT']['Zbed'].item().squeeze().item()
    # Make pandas DataFrame
    dfp = pd.DataFrame(data={'eta_hyd':pt}, index=time_ind)