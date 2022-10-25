"""
Pre-process Nortek Vector ADV raw data. 
Remove bad measurements based on correlations and despiking. 
Save Level1 products as netcdf.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime as DT
from roxsi_pyfuns import despike

class ADV():
    """
    Main ADV class
    """
    def __init__(self, datadir, mooring_id, fs=16, burstlen=1200, magdec=12.86,
                 outdir=None):
        """
        Initialize ADV class.

        Parameters:
            datadir; str - Path to raw data directory
            mooring_id - str; ROXSI 2022 SSA mooring ID
            fs - scalar; sampling frequency (Hz)
            magdec - scalar; magnetic declination (deg E) of location
            burstlen - scalar; burst length (sec)
            outdir - str; if None, save output files in self.datadir,
                     else, specify outdir
        """
        self.datadir = datadir
        self.mid = mooring_id
        self.fs = fs
        self.dt = 1 / self.fs # Sampling rate (sec)
        self.magdec = magdec
        self.burstlen = burstlen
        if outdir is None:
            # Use datadir as outdir
            self.outdir = datadir
        else:
            # Use specified outdir
            self.outdir = outdir
        # Get raw data filenames
        self._fns_from_mid()


    def _fns_from_mid(self):
        """
        Get Vector data filenames from mooring ID and
        data directory.
        """        
        # Find correct .dat file with data time series
        self.fn_dat = os.path.join(self.datadir, '{}.dat'.format(self.mid))
        # Find correct .sen file with burst info
        self.fn_sen = os.path.join(self.datadir, '{}.sen'.format(self.mid))

        
    def loaddata(self, datestr=None, despike_corr=True, despike_GN02=True,
                 interp='linear'):
        """
        Read raw data from chosen mooring ID into pandas
        dataframe. Header info can be found in .hdr files.

        Raw data gets saved into netcdf files, with 1-Hz data from .sen
        files (e.g. heading, pitch & roll) interpolated to the sampling 
        rate of the velocity data.

        Parameters:
            datestr - str (yyyymmdd); if not None, read only requested
                      date of data. Else, read entire dataset from raw data.
            despike_corr - bool; if True, use correlations to get rid of bad
                           velocity data.
            despike_GN02 - bool; if True, use Goring & Nikora (2002) phase space
                           method to despike velocities (burst-wise).
            interp - str; interpolation method for despiking algorithms.
        """
        # Define column names (see .hdr files for info)
        cols_dat = ['burst', 'ensemble', 'u', 'v', 'w', 
                    'ampl1', 'ampl2', 'ampl3', 'SNR1', 'SNR2', 'SNR3',
                    'corr1', 'corr2', 'corr3', 'pressure', 'ai1', 'ai2',
                    'checksum',
                    ] # .dat file columns
        cols_sen = ['month', 'day', 'year', 'hour', 'minute', 'second', 
                    'error_code', 'status_code', 'battery_volt', 
                    'soundsp', 'heading', 'pitch', 'roll',
                    'tempC', 'analog_in', 'checksum', 
                    ] # .sen file columns

        # Are we requesting a specific date?
        if datestr is not None:
            # Define daily netcdf filename
            # Note that datestr format must be "yyyymmdd"
            fn_nc = os.path.join(self.outdir, 'Asilomar_SSA_{}.nc'.format(
                self.mid))
            # Read netcdf file if it exists
            if os.path.isfile(fn_nc):
                xr.read_csv(fn_nc)
            else:
                # Must generate netcdf first
                raise ValueError('Netcdf file for {} does not exist yet.'.format(
                    datestr))

        # Read data into pandas dataframe
        data = pd.read_table(self.fn_dat, names=cols_dat, header=None,
            delimiter=' ', skipinitialspace=True)

        # Read sensor info into another dataframe from .sen file
        sen = pd.read_table(self.fn_sen, names=cols_sen, header=None,
            delimiter=' ', skipinitialspace=True)
        # Parse dates from relevant columns
        date_cols = ['year','month','day','hour','minute','second']
        sen['time'] = pd.to_datetime(sen[date_cols])
        # Set timestamp as index
        sen.set_index('time', inplace=True)

        # Split raw data into daily segments and save as netcdf
        day0 = sen.index[0].day
        month0 = sen.index[0].month
        year0 = sen.index[0].year 
        day1 = sen.index[-1].day
        month1 = sen.index[-1].month
        year1 = sen.index[-1].year 
        # Start date
        t0 = pd.Timestamp('{:d}-{:02d}-{:02d}'.format(year0, month0, day0))
        # End date
        t1 = pd.Timestamp('{:d}-{:02d}-{:02d}'.format(year1, month1, day1))
        days = pd.date_range(t0, t1, freq='1D')
        # Iterate over days (i is date index starting at 0)
        dp_cnt = 0 # daily data point counter
        for i, date in enumerate(days[0:2]):
            # Copy daily (d) segment from sen dataframe
            sen_d = sen.loc[DT.strftime(date, '%Y-%m-%d')].copy()
            # Calculate number of bursts in the current day
            Nsen = self.burstlen+1 # Nortek adds +1s to burst
            Nb = int(len(sen_d) / Nsen) 
            print('Nb: ', Nb)
            # Take out the same number of bursts from data
            Nd = self.burstlen * self.fs # Number of data samples per burst
            print('Nd: ', Nd)
            dp = int(Nb * Nd) # Number of data points in the current day
            # Take out current date (not necessarily a full day)
            data_d = data.iloc[dp_cnt:(dp_cnt+dp)].copy()
            print('len(data_d): {} \n'.format(len(data_d)))
            dp_cnt += dp # Increase counter
            # Iterate over individual bursts of data
            uvw_dfs = [] # Empty list to store u,v,w dataframes for merging
            hpr_dfs = [] # Empty list to store sen dataframes for merging
            burst_cols = ['burst', 'u', 'v', 'w', 
                          'corr1', 'corr2', 'corr3', 'pressure']
            for bn, burst in enumerate(np.array_split(data_d[burst_cols], Nb)):
                # Take out corresponding sen burst
                senb_cols = ['heading', 'pitch', 'roll', 'tempC']
                burst_s = sen_d[senb_cols].iloc[Nsen*bn:Nsen*(bn+1)] 
                # Make burst time index at data sampling rate
                t0b = burst_s.index[0] # Burst index start time
                t1b = burst_s.index[-1] # Burst index end time
                # Velocity sampling starts at +2 sec from requested start time.
                # See:
                # https://support.nortekgroup.com/hc/en-us/articles/360029499432-
                # How-do-you-figure-out-the-time-to-assign-to-Vector-velocity-samples-
                t0b += pd.Timedelta(seconds=1) # Add one second
                t1b += pd.Timedelta(seconds=1) # Add one second
                # To get the most accurate velocity timestamps (from Nortek website):
                # "Since the Vector is pinging continuously during each sample, 
                #  the best time for the sample would be the midpoint of the 
                #  measurement."
                t0b += pd.Timedelta(seconds=self.dt/2) # Add a half timestep
                t1b += pd.Timedelta(seconds=self.dt/2) # Add a half timestep
                print('{} {}'.format(t0b, t1b))
                # Make burst timestamps using pre-defined sampling rate
                burst_index = pd.date_range(t0b, t1b, freq='{}S'.format(self.dt))
                burst_index = burst_index[:-1] # Correct length
                # Set index to burst segment dataframe
                burst = burst.set_index(burst_index)
                burst.index = burst.index.rename('time')
                # Add a half timestep to sen burst index for reindexing
                burst_s.index += pd.Timedelta(seconds=self.dt/2) 
                # Interpolate (lin) 1-Hz sensor data to data burst sampling rate
                burst_si = burst_s.reindex(burst.index, method=None).interpolate()
                # Append interpolated sen burst to list
                hpr_dfs.append(burst_si)

                # If requested, apply QC procedures on burst
                if despike_corr:
                    # Use correlations to get rid of bad measurements following
                    # Elgar et al. (2001, Jtech)
                    self.despike_correlations(burst, interp=interp)
                    corrd = True # Input arg to despike_GN02()
                else:
                    corrd = False # Velocities are not corrected for correlations
                if despike_GN02:
                    # Despike velocities following Goring and Nikora (2002)
                    self.despike_GN02(burst, corrd=corrd, interp=interp)

                # Save burst df to list for merging
                uvw_dfs.append(burst)

            # Merge burst dataframes to one daily dataframe
            uvw_d = pd.concat(uvw_dfs)
            # Same for interpolated sen (H,P,R,T) data
            hpr_d = pd.concat(hpr_dfs)
            # Combine both dataframes into one
            vec_d = pd.concat([uvw_d, hpr_d], axis=1)

            # Convert dataframe to xarray dataset and save to daily netcdf file
            if datestr is None:
                # At the end, return full dataset by concatenating daily datasets
                ds_list = [] # Append daily datasets for merging

        return vec_d
    

    def despike_correlations(self, df, interp='linear'):
        """
        Despike Nortek Vector velocities using low correlation values
        to discard unreliable measurements following Elgar et al. (2001, Jtech);
        https://doi.org/10.1175/1520-0426(2001)018<1735:CMPITS>2.0.CO;2

        Parameters:
            df - pd.DataFrame; dataframe with velocity (u,v,w) and
                 correlation (corr1, corr2, corr3) time series.
            interp - str; interpolation method for discarded data.
                     Options: see pandas.DataFrame.interpolate()
        
        Returns:
            df - Updated dataframe with corrected velocity columns added.
        """
        # Check that data format matches expectations
        vel_cols = ['u', 'v', 'w']
        cor_cols = ['corr1', 'corr2', 'corr3']
        if not np.all([c in df.columns for c in vel_cols]):
            raise ValueError('Input df must contain {}'.format(vel_cols))
        if not np.all([c in df.columns for c in cor_cols]):
            raise ValueError('Input df must contain {}'.format(cor_cols))
        # Copy velocity and correlation columns so we don't change the input
        vels = df[vel_cols].copy()
        cors = df[cor_cols].copy()
        
        # Iterate over velocity and correlation components
        for (kv, kc) in zip(vels, cors):
            vel = vels[kv]
            cor = cors[kc]
            # Set low-correlation cutoff following Elgar et al. (2001)
            corcutoff = 30 + 40*np.sqrt(self.dt/25)
            # Set velocity measurements corresponding to low correlation to NaN
            mask = (cor.values < corcutoff)
            vel[mask] = np.nan
            # Interpolate according to requested method
            vel.interpolate(method=interp, inplace=True)
            # Add corrected velocity column to input df
            df['{}_corr'.format(kv)] = vel


    def despike_GN02(self, df, interp='linear', sec_lim=1, corrd=True):
        """
        Despike Nortek Vector velocities using low correlation values
        to discard unreliable measurements following the Goring and
        Nikora (2002, J. Hydraul. Eng.) phase space method, including the
        modifications by Wahl (2003, J. Hydraul. Eng.) and Mori
        (2007, J. Eng. Mech.).

        Parameters:
            df - pd.DataFrame; dataframe with velocity (u,v,w) time series.
            interp - str; interpolation method for discarded data.
                          Options: see pandas.DataFrame.interpolate()
            sec_lim - scalar; maximum gap size to interpolate (sec)
            corrd - bool; if True, input velocities are correlation-
                    corrected by self.despike_correlations()
        
        Returns:
            df - Updated dataframe with despiked velocity columns added.
        """
        # Check that data format matches expectations
        if not corrd:
            vel_cols = ['u', 'v', 'w']
        else:
            vel_cols = ['u_corr', 'v_corr', 'w_corr']
        if not np.all([c in df.columns for c in vel_cols]):
            raise ValueError('Input df must contain {}'.format(vel_cols))
        # Copy velocity columns so we don't change the input
        vels = df[vel_cols].copy()
        
        # Iterate over velocity components
        for kv, k in zip(vels, ['u', 'v', 'w']):
            vel = vels[kv]
            mask = despike.phase_space_3d(vel.values)
            vel[mask] = np.nan
            # Interpolate according to requested method
            vel.interpolate(method=interp, limit=sec_lim*self.fs, inplace=True)
            # Add corrected velocity column to input df
            df['{}_desp'.format(k)] = vel
    

    def pd2nc(self, df):
        """
        Convert and save pd.DataFrame of Vector data to netcdf format.
        """



# Main script
if __name__ == '__main__':
    """
    Test script using synthetic example data.
    """
    import sys
    import glob
    from tqdm import tqdm
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
                help=('Mooring ID. To loop through all, select "ALL".'),
                type=str,
                choices=['C1v01', 'C2VP02', 'C3VP02', 'C4VP02', 'C5V02', 
                         'C6v01', 'L1v01', 'L2VP02', 'L4VP02', 'L5v01', 'ALL'],
                default='C3VP02',
                )
        parser.add_argument("-datestr", 
                help=("Specify date to read. Requires existing daily netcdf file."),
                type=str,
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

        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Define Level1 output directory
    outdir = os.path.join(args.dr, 'Level1')
    figdir = os.path.join(outdir, 'img')

    # Check if processing just one mooring or all
    if args.mid == 'ALL':
        # Loop through all mooring IDs
        mids = ['C1v01', 'C2VP02', 'C3VP02', 'C4VP02', 'C5V02', 
                'C6v01', 'L1v01', 'L2VP02', 'L4VP02', 'L5v01']
    else:
        # Only process one mooring
        mids = [args.mid]

    # Iterate over mooring ID(s)
    for mid in tqdm(mids):
        # Initialize ADV class and read raw data
        adv = ADV(datadir=args.dr, mooring_id=mid, magdec=args.magdec)
        print('Reading raw data .dat file "{}" ...'.format(
            os.path.basename(adv.fn_dat)))
        # Read data (specific date or all)
        if args.datestr is None:
            # Don't specify date
            vec_d = adv.loaddata()
        else:
            # Read specified date
            vec_d = adv.loaddata(datestr=args.datestr)

        # Plot heading, pitch & roll time series
        if args.savefig:
            # Define figure filename and check if it exists
            if args.datestr is None:
                fn_hpr = os.path.join(figdir, 'hpr_press_{}.pdf'.format(
                    mid))
            else:
                fn_hpr = os.path.join(figdir, 'hpr_press_{}_{}.pdf'.format(
                    mid, args.datestr))

            if not os.path.isfile(fn_hpr) or args.overwrite_fig:
                print('Plotting heading, pitch & roll timeseries ...')
                fig, axes = plt.subplots(figsize=(12,5), nrows=2, sharex=True,
                                         constrained_layout=True)
                vec_d[['heading', 'pitch', 'roll']].plot(ax=axes[0])
                vec_d['pressure'].plot(ax=axes[1])
                axes[0].set_ylabel('Degrees')
                axes[1].set_ylabel('dB')
                axes[1].legend()
                axes[0].set_title('Vector {} heading, pitch & roll + pressure'.format(
                    mid))
                # Save figure
                plt.savefig(fn_hpr, bbox_inches='tight', dpi=300)
                plt.close()

        # Plot raw vs. QC'd timeseries
        if args.savefig:
            # Define figure filename and check if it exists
            if args.datestr is None:
                fn_vel = os.path.join(figdir, 'vel_desp_{}.pdf'.format(
                    mid))
            else:
                fn_vel = os.path.join(figdir, 'vel_desp_{}_{}.pdf'.format(
                    mid, args.datestr))

            if not os.path.isfile(fn_vel) or args.overwrite_fig:
                fig, axes = plt.subplots(figsize=(12,7), nrows=3, 
                                        sharex=True, sharey=True, 
                                        constrained_layout=True)
                vec_d[['u', 'u_corr', 'u_desp']].plot(ax=axes[0])
                vec_d[['v', 'v_corr', 'v_desp']].plot(ax=axes[1])
                vec_d[['w', 'w_corr', 'w_desp']].plot(ax=axes[2])
                # Save figure
                plt.savefig(fn_vel, bbox_inches='tight', dpi=300)
                plt.close()


print('Done. \n')
