"""
* Pre-process Nortek Signature ADCP raw data. 
* Save Level1 products as netcdf.
* Separate files for echogram data & rest.
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
    def __init__(self, datadir, mooring_id, zp=0.08, fs=4, burstlen=1200, 
                 magdec=12.86, outdir=None, mooring_info=None, patm=None, 
                 instr='Nortek Vector'):
        """
        Initialize ADCP class.

        Parameters:
            datadir; str - Path to raw data directory
            mooring_id - str; ROXSI 2022 SSA mooring ID
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
        self.mid = mooring_id
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
            # Get serial number
            key = 'serial_number'
            self.ser = self.dfm[self.dfm['mooring_ID']==self.mid][key].item()
        else:
            self.dfm = None
            self.ser = None

    def loaddata_1d(self):
        """
        Load Nortek Signature 1000 time series data into xr.Dataset.
        """
