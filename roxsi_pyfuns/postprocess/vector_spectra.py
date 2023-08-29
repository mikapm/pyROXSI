"""
Generate spectral netcdf files from Vector nc files.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from argparse import ArgumentParser
from roxsi_pyfuns import wave_spectra as rpws

# Main script
if __name__ == '__main__':
    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data root directory"),
                type=str,
                # default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/Vectors',
                default=r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray/Vectors',
                )
        parser.add_argument("-mid", 
                help=("Mooring ID (short)"),
                type=str,
                default='C2',
                choices=['C1','C2','C3','C4','C5','C6','L1','L2','L4','L5',]
                )
        parser.add_argument("-magdec", 
                help=("Magnetic declination to use (deg E)"),
                type=float,
                default=12.86,
                )
        parser.add_argument("--overwrite_nc", 
                help=("Overwrite existing netcdf files?"),
                action="store_true",
                )

        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Define Level1 output directory
    outdir_base = os.path.join(args.dr, 'Level1', args.mid)
    outdir = os.path.join(outdir_base, 'Spectra')
    # Make output dir. if it does not exist
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
