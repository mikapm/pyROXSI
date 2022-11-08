"""
* Pre-process Nortek Signature ADCP raw data. 
* Reads raw ADCP data from converted .mat files.
* Saves Level1 products as netcdf.
* Separate files for echogram data & rest (1D time series).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
import matplotlib.pyplot as plt

class ADCP():
    """
    Main ADCP data class.
    """
    def __init__(self, datadir, ser, zp=0.08, fs=4, burstlen=1200, 
                 magdec=12.86, outdir=None, mooring_info=None, 
                 instr='NortekSignature1000'):
        """
        Initialize ADCP class.

        Parameters:
            datadir; str - Path to raw data (.mat files) directory
            ser - str; Signature serial number
            zp - scalar; height of sensor above seabed (m)
            fs - scalar; sampling frequency (Hz)
            magdec - scalar; magnetic declination (deg E) of location
            burstlen - scalar; burst length (sec)
            outdir - str; if None, save output files in self.datadir,
                     else, specify outdir
            mooring_info - str; path to mooring info excel file (optional)
            instr - str; instrument name
        """
        self.datadir = datadir
        self.ser = ser
        # Find all .mat files for specified serial number
        self._fns_from_ser()
        self.zp = zp
        self.fs = fs
        self.dt = 1 / self.fs # Sampling rate (sec)
        self.magdec = magdec
        self.burstlen = burstlen
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
            self.mid = self.dfm[self.dfm['serial_number']==self.ser][key].item()
        else:
            self.dfm = None # No mooring info dataframe
            self.mid = None # No mooring ID number

    def _fns_from_ser(self):
        """
        Returns a list of .mat filenames in self.datadir corresponding
        to serial number.
        """
        # List all .mat files with serial number in filename
        self.fns = sorted(glob.glob(os.path.join(self.datadir,
            '*S{}A001*.mat'.format(self.ser))))

    def loaddata_1d(self, fn_mat):
        """
        Load Nortek Signature 1000 time series data into xr.Dataset.

        Parameters:
            fn_mat - str; path to .mat filename
        """
        # Read .mat structure
        mat = loadmat(fn_mat)

        # Read arrays to save in dataset
        time_mat = dsi['Data']['Burst_Time'].item().squeeze() # Time array
        # Convert Matlab times to datetime
        time_arr = pd.to_datetime(time_mat-719529, unit='D')
        # Velocities from beams 1-4
        vb1 = dsi['Data']['Burst_VelBeam1'].item().squeeze()
        vb2 = dsi['Data']['Burst_VelBeam2'].item().squeeze()
        vb3 = dsi['Data']['Burst_VelBeam3'].item().squeeze()
        vb4 = dsi['Data']['Burst_VelBeam4'].item().squeeze()
        # Save arrays to temporary dataframe
        dfi = pd.DataFrame(data={'vb1':vb1, 'vb2':vb2, 'vb3':vb3, 'vb4':vb4},
                           index=time_arr)
        # Acoustic surface tracking distance - AST
        ast = dsi['Data']['Burst_AltimeterDistanceAST'].item().squeeze()
        # Interpolate AST to "master" time array using AST time offsets
        ast_offs = dsi['Data']['Burst_AltimeterTimeOffsetAST'].item().squeeze()
        time_ast_mat = time_mat - ast_offs
        time_ast = pd.to_datetime(time_ast_mat-719529, unit='D')
        df_ast = pd.DataFrame(data={'ast':ast}, index=time_ast)

        return dfi


# Main script
if __name__ == '__main__':
    """
    Test script using synthetic example data.
    """
    from argparse import ArgumentParser

    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data root directory"),
                type=str,
                default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/Signatures',
                )
        parser.add_argument("-ser", 
                help=('Instrument serial number. To loop through all, select "ALL".'),
                type=str,
                choices=['103088', '103094', '103110', '103063', '103206'],
                default='103063',
                )
        parser.add_argument("-M", 
                help=("Segment window length (number of samples)"),
                type=int,
                default=512,
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
        parser.add_argument("-magdec", 
                help=("Magnetic declination to use (deg E)"),
                type=float,
                default=12.86,
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

    # Define Level1 output directory
    outdir = os.path.join(args.dr, 'Level1')
    figdir = os.path.join(outdir, 'img')

    # Mooring info excel file path (used when initializing ADV class)
    rootdir = os.path.split(args.dr)[0] # Root ROXSI SSA directory
    fn_minfo = os.path.join(rootdir, 'Asilomar_SSA_2022_mooring_info.xlsx')
    
    # Check if processing just one serial number or all
    if args.ser.lower() == 'all':
        # Loop through all serial numbers
        sers = ['103088', '103094', '103110', '103063', '103206']
    else:
        # Only process one serial number
        sers = [args.ser]

    # Iterate through serial numbers and read + preprocess data
    for ser in sers:
        print('Serial number: ', ser)
        # Initialize class
        adcp = ADCP(datadir=args.dr, ser=ser, )
        # Loop over raw .mat files and save data as netcdf
        ds_list = [] # Empty list for concatenating datasets
        for i,fn_mat in enumerate(adcp.fns):
            # Read mat structure
            dsi = adcp.loaddata_1d(fn_mat)
            # Append dataset to list for concatenating
            ds_list.append(dsi)


