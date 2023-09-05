"""
Generate spectral netcdf files from Vector nc files.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from cftime import date2num
from scipy.spatial.transform import Rotation # For testing
from argparse import ArgumentParser
from roxsi_pyfuns import wave_spectra as rpws
from roxsi_pyfuns import coordinate_transforms as rpct

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

    # Reference date for netcdf timestamps
    ref_date=pd.Timestamp('2000-01-01')

    # Define expected headings in PCA coordinates for C2, C3 & C4
    heading_exp_1 = {'C2': -125, 'C3': -90, 'C4': -90, 'L2': None, 'L1': 160}
    # After 2022-07-13 04:00 (C4)
    heading_exp_2 = {'C2': -125, 'C3': -90, 'C4': -20, 'L2': None, 'L1': 160}

    # Define Level1 output directory
    outdir_base = os.path.join(args.dr, 'Level1', args.mid)
    outdir = os.path.join(outdir_base, 'Spectra')
    # Make output dir. if it does not exist
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # List all available Vector nectdf files
    ncf_vec = sorted(glob.glob(os.path.join(outdir_base, 'Asilomar_*.nc')))

    # List for storing daily spectral datasets for concatenating
    dsl_spec = []

    # Iterate over daily netcdf files and estimate spectra for 20-min. bursts
    for ncf in ncf_vec[17:18]:
        # Read netcdf file
        dsv = xr.open_dataset(ncf, decode_coords='all')
        # Make range of spectral segment start times (20 min of data / hour)
        t0 = pd.Timestamp(dsv.time[0].values).round('1H')
        t1 = pd.Timestamp(dsv.time[-1].values).round('1H')
        seg_range = pd.date_range(t0, t1, freq='1H')
        print(t0)
        # Iterate over spectral segments and estimate spectra
        for t0s, t1s in zip(seg_range[:-1], seg_range[1:]):
            # Take out 1-h segment (i.e. 20-min burst) 
            seg = dsv.sel(time=slice(t0s, t1s)).copy()
            # Estimate (directional) spectra
            z = seg.eta_lin_krms.values # Use linear K_rms surface elevation
            # Get current local water depth
            h = seg.z_hyd.mean().item()
            # Horizontal (along/cross-shore) velocities
            if args.mid in ['C2', 'C3']:
                # Can trust measured tilt angles (horizontal tilt sensor)
                u = seg.uE.values.squeeze() # Use East vel. for u velocity
                v = seg.uN.values.squeeze() # Use North vel. for v velocity
                w = seg.uU.values.squeeze() # Vertical velocity
                # Compute mean horizontal current magnitude and direction
                U = np.mean(np.sqrt(u**2 + v**2))
                # Estimate spectrum to get mean wave dir.
                dss_init = rpws.spec_uvz(z=z, u=u, v=v, fs=16, fmerge=5, depth=h)
                # Save mdir in geographical reference frame (met. convention)
                angle_met = dss_init.mdir.item()
                # Rotate E, N velocities to cross/longshore
                angle_math = 270 - angle_met # Math angle to rotate
                if angle_math < 0:
                    angle_math += 360
                angle_math = np.deg2rad(angle_math) # Radians
                # Rotate East and North velocities to cross-shore (cs) and 
                # long-shore (ls)
                ucs, uls = rpct.rotate_vel(u, v, angle_math)
            else:
                # Use PCA rotation for Vectors w/ vertical tilt sensor (eg C4)
                uxd = seg.uxd.to_dataframe() # Convert to pandas
                uxd = uxd.interpolate(method='bfill').interpolate('ffill')
                uxm = uxd.mean().item()
                uxd -= uxm
                # y vel, despiked
                uyd = seg.uyd.to_dataframe() # Convert to pandas
                uyd = uyd.interpolate(method='bfill').interpolate('ffill')
                uym = uyd.mean().item()
                uyd -= uym
                # z vel, despiked
                uzd = seg.uzd.to_dataframe() # Convert to pandas
                uzd = uzd.interpolate(method='bfill').interpolate('ffill')
                uzd -= uzd.mean()
                # No geographical wave angle available (do not know true heading)
                angle_met = np.nan
                # Get correct expected heading for C4
                if t0s <= pd.Timestamp('2022-07-13 04:00'):
                    he = heading_exp_1[args.mid]
                else:
                    he = heading_exp_2[args.mid]
                # Rotate velocities to cross/alongshore & vertical using PCA
                ucs, uls, w, eul = rpct.enu_to_loc_pca(ux=uxd.values.squeeze(), 
                                                       uy=uyd.values.squeeze(), 
                                                       uz=uzd.values.squeeze(),
                                                       heading_exp=he, 
                                                       return_eul=True,
                                                       # print_msg=True,
                                                       )               
                eul1 = np.rad2deg(eul['eul1'])
                eul2 = np.rad2deg(eul['eul2'])
                eul3 = np.rad2deg(eul['eul3'])
                angles = np.array([eul['eul1'], eul['eul2'], eul['eul3']])
                # Get rotation matrix (for debugging)
                R = Rotation.from_euler('xyz', angles).as_matrix()
                # R = Rotation.from_euler('zxz', angles).as_matrix()
                print(f'eul1: {eul1:.2f}, eul2: {eul2:.2f}, eul3: {eul3:.2f}, ')
                # print('R: ', R)
                # print('rotvec: {} \n'.format(Rotation.from_euler('xyz', angles).as_rotvec()))
                # Rotate point
                xyz = np.array([1, 0, 1])
                rot_arr = R.dot(xyz.T).T
                print('rot_arr: {}'.format(rot_arr))
            # Estimate spectrum from cross/longshore velocities
            dss = rpws.spec_uvz(z=z, u=ucs, v=uls, fs=16, fmerge=5)
            # If mdir = 90 -> flip CS velocity
            if abs(90 - dss.mdir.item()) < 25:
                ucs *= (-1)
                # Estimate new spectrum
                dss = rpws.spec_uvz(z=z, u=ucs, v=uls, fs=16, fmerge=5)
            # Assign time coordinate
            dss = dss.assign_coords(time=[t0s])
            print('mdir={} \n'.format(dss.mdir.item()))
            # Convert time array to numerical format
            time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(ref_date)
            time = pd.to_datetime(dss.time.values).to_pydatetime()
            time_vals = date2num(time, 
                                time_units, calendar='standard', 
                                has_year_zero=True)
            dss.coords['time'] = time_vals.astype(float)
            # print('mdir={} \n'.format(dss.mdir.item()))
            # Compute RMS orbital velocity
            uspec = rpws.spec_uvz(ucs, fs=16, fmerge=5)
            vspec = rpws.spec_uvz(uls, fs=16, fmerge=5)
            # Variance of cross- and alongshore orbital velocities
            m0u = rpws.spec_moment(uspec.Ezz.values, uspec.freq.values, 0)
            m0v = rpws.spec_moment(vspec.Ezz.values, vspec.freq.values, 0)
            Urms = np.sqrt(2 * (m0u + m0v))
#             # Compute mean vertical velocity
            W = np.nanmean(w)
#             U_dir = 
            # Append to list
            dsl_spec.append(dss)

# Concatenate hourly spectra to single spectral dataset
ds_out = xr.concat(dsl_spec, dim='time')