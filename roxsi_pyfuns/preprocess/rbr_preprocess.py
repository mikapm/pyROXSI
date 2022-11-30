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
# from roxsi_pyfuns import transfer_functions as tf
from roxsi_pyfuns import transfer_functions as rptf
from roxsi_pyfuns import wave_spectra as rpws

class RBR():
    """
    Main RBR data class.
    """
    def __init__(self, datadir, ser, zp=0.08, fs=16, burstlen=1200, 
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
        self.instr = instr
        # .mat dict instrument key
        if self.instr == 'RBRDuetDT':
            self.matkey = 'DUETDT'
        elif self.instr == 'RBRSoloD':
            self.matkey = 'SOLOD'
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

    def _fns_from_ser(self):
        """
        Returns a list of .mat filenames in self.datadir corresponding
        to serial number.
        """
        # List all .mat files with serial number in filename
        self.fns = sorted(glob.glob(os.path.join(self.datadir,
            'roxsi_*_{}_*.mat'.format(self.ser))))


    def p2eta_hyd(self, p, rho0=1025, grav=9.81):
        """
        Convert pressure measurements to hydrostatic depth with
        units of m.
        
        Parameters:
            p - pd.Series; time series of water pressure
            rho0 - scalar; water density (kg/m^3)
            grav - scalar; gravitational acceleration (m/s^2)
        """
        # Use hydrostatic assumption to get pressure head with unit [m]
        eta_hyd = rptf.eta_hydrostatic(p, self.patm, rho0=rho0, 
                                       grav=grav, interp=True)
        # Check if hydrostatic pressure is ever above 0
        if eta_hyd.max() == 0.0:
            print('Instrument most likely not in water')
            # Return NaN array for linear sea surface elevations
            eta_hyd = np.ones_like(p) * np.nan

        return eta_hyd


    def p2eta_lin(self, eta_hyd, M=512*8, fmin=0.05, fmax=0.33, 
                  att_corr=True):
        """
        Convert hydrostatic depth (m) to sea-surface elevation
        using linear wave theory.
        """
        # Initialize TRF class
        trf = rptf.TRF(fs=self.fs, zp=self.zp)
        # Apply linear transfer function
        eta = trf.p2eta_lin(eta_hyd, M=M, fmin=fmin, fmax=fmax,
                            att_corr=att_corr,)
        return eta


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
                choices=['210356', '210357', '210358', '210359', '210360', '210361',
                         '41428', '41429', '124107', '124108', '124109', '210362', 'all'],
                default='210356',
                )
        parser.add_argument("-instr", 
                help=('Instrument name.'),
                type=str,
                choices=['RBRDuetDT', 'RBRSoloD'],
                default='RBRDuetDT',
                )
        parser.add_argument("-M", 
                help=("Pressure transform segment window length"),
                type=int,
                default=512*8,
                )
        parser.add_argument("-fs", 
                help=("Sampling frequency [Hz]"),
                type=float,
                default=16.,
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

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Read atmospheric pressure time series and calculate
    # atmospheric pressure anomaly
    fn_patm = os.path.join(rootdir, 'noaa_atm_pressure.csv')
    if not os.path.isfile(fn_patm):
        # csv file does not exist -> read mat file and generate dataframe
        fn_matp = os.path.join(rootdir, 'noaa_atm_pres_simple.mat')
        mat = loadmat(fn_matp)
        mat_pres = mat['A']['atm_pres'].item().squeeze()
        mat_time = mat['A']['time_vec'].item().squeeze() # In Matlab char() format
        # Make into pandas dataframe
        dfa = pd.DataFrame(data={'hpa':mat_pres}, 
                           index=pd.to_datetime(mat_time))
        dfa.index.rename('time', inplace=True)
        # convert from mbar to hpa
        dfa['hpa'] /= 100
        dfa['hpa'] -= 0.032 # Empirical correction factor
        # Calculate anomaly from mean
        dfa['hpa_anom'] = dfa['hpa'] - dfa['hpa'].mean()
        # Save as csv
        dfa.to_csv(fn_patm)
    else:
        dfa = pd.read_csv(fn_patm, parse_dates=['time']).set_index('time')

    # Define paths and load data
    data_root = os.path.join(args.dr, args.instr)
    matdir = os.path.join(data_root, 'Level1', 'mat')
    outdir = os.path.join(data_root, 'Level1', 'netcdf')
    # Initialize RBR class object
    rbr = RBR(datadir=matdir, ser=args.ser, fs=args.fs, 
              instr=args.instr, patm=dfa)

    # Iterate over daily mat files and concatenate into pd.DataFrame
    for fi, fn_mat in enumerate(rbr.fns):
        print('Loading pressure sensor mat file {}'.format(os.path.basename(fn_mat)))
        mat = loadmat(fn_mat)
        # Read pressure time series and timestamps
        pt = np.array(mat[rbr.matkey]['P'].item()).squeeze()
        time_mat = np.array(mat[rbr.matkey]['time_dnum'].item()).squeeze()
        time_ind = pd.to_datetime(time_mat-719529,unit='d') # Convert timestamps
        # Get hydrostatic pressure (depth)
        sp = pd.Series(data=pt, index=time_ind)
        eta_hyd = rbr.p2eta_hyd(sp)
        # Make pandas DataFrame
        if rbr.instr == 'RBRSoloD':
            dfp = pd.DataFrame(data={'pressure':pt,
                                     'eta_hyd':eta_hyd,
                                     'eta_lin':np.ones_like(pt)*np.nan,
                                     }, 
                               index=time_ind)
        elif rbr.instr == 'RBRDuetDT':
            # Also save temperature from Duets
            temp = np.array(mat[rbr.matkey]['temperature'].item()).squeeze()
            dfp = pd.DataFrame(data={'pressure':pt,
                                     'temperature':temp,
                                     'eta_hyd':eta_hyd,
                                     'eta_lin':np.ones_like(pt)*np.nan,
                                     }, 
                               index=time_ind)

        # Crop timeseries at last full 20-min. segment
        if fi == (len(rbr.fns) - 1):
            t_end = dfp.index[-1].floor('20T')
            dfp = dfp.loc[:t_end]
        # Count number of full 20-minute (1200-sec) segments
        t0s = pd.Timestamp(dfp.index[0]) # Start timestamp
        t1s = pd.Timestamp(dfp.index[-1]) # End timestamp
        nseg = np.round((t1s - t0s).total_seconds() / 1200)
        # Iterate over 20-minute segments and convert pressure to
        # sea-surface elevation
        for sn, seg in enumerate(np.array_split(dfp['eta_hyd'], nseg)):
            # There's a gap of missing data on 2022-07-12 between 
            # 05:00 - 05:20. Skip that.
            if np.sum(np.isnan(seg.values)) > 1000:
                print('Too many gaps.')
                continue
            # Get segment start and end times
            t0ss = seg.index[0]
            t1ss = seg.index[-1]
            # Convert hydrostatic depth to eta w/ linear TRF
            dfp['eta_lin'].loc[t0ss:t1ss] = rbr.p2eta_lin(
                seg.interpolate(method='ffill').interpolate(method='bfill').values,
                M=args.M, fmin=args.fmin, fmax=args.fmax)