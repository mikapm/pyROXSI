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
from datetime import datetime as DT
from cftime import date2num, num2date
from astropy.stats import mad_std
from roxsi_pyfuns import despike as rpd


def max_runlen(arr, value):
     """
     Find runs of consecutive items in an array.
     Borrowed from alimanfoo at
     https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
     """ 
     # ensure array 
     x = np.asanyarray(arr) 
     if x.ndim != 1: 
         raise ValueError('only 1D array supported') 
     n = x.shape[0] 
  
     # handle empty array 
     if n == 0: 
         return 0 
     else: 
         # find run starts 
         loc_run_start = np.empty(n, dtype=bool) 
         loc_run_start[0] = True 
         np.not_equal(x[:-1], x[1:], out=loc_run_start[1:]) 
         run_starts = np.nonzero(loc_run_start)[0] 
  
         # find run values 
         run_values = x[loc_run_start] 
         # find run lengths 
         run_lengths = np.diff(np.append(run_starts, n)) 
         # Find max run length = to value 
         max_run = np.max(run_lengths[run_values==value]) 

         return max_run 

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

    def read_mat_times(self, mat=None, fn_mat=None):
        """
        Function to read only velocity timestamps from given
        Signature mat structure.

        Parameters:
            mat - dict; Signature mat structure with timestamps
            fn_mat - str; path to .mat file to read for mat structure
                     (not required if mat dict is given)

        Returns:
            time_mat - array of Matlab timestamps
            time_arr - array of timestamps in datetime.datetime
                       format.
        """
        # Check that given either mat dict or filename
        if mat is None and fn_mat is None:
            raise ValueError('Must input either mat or fn_mat.')

        if fn_mat is not None:
            # Read mat structure from file
            mat = loadmat(fn_mat)

        # Read and convert velocity time array
        time_mat = mat['Data']['Burst_Time'].item().squeeze()
        # Convert Matlab times to datetime
        time_arr = pd.Series(pd.to_datetime(time_mat-719529, 
                             unit='D'))
        time_arr = time_arr.dt.to_pydatetime()

        return time_mat, time_arr


    def loaddata_vel(self, fn_mat, ref_date=pd.Timestamp('2022-06-25'),
                     despike_vel=True, despike_ast=True):
        """
        Load Nortek Signature 1000 velocity and surface track (AST) 
        data into xr.Dataset.

        Parameters:
            fn_mat - str; path to .mat filename.
            ref_date - reference date to use for time axis.
            despike_vel - bool; if True, despikes velocities using
                          3D phase-space despiking method of Goring
                          and Nikora (2002), modified by Wahl (2003)
                          and Mori et al. (2007) in 20-minute
                          segments.
            despike_ast - bool; if True, despikes Acoustic Surface
                          Tracking (AST) data using Gaussian Process
                          method of Malila et al. (2022) in 20-minute
                          segments.
        """
        # Read .mat structure
        mat = loadmat(fn_mat)

        # Read and convert general time array
        time_mat, time_arr = self.read_mat_times(mat=mat)

        # Convert time array to numerical format
        time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(
            ref_date)
        time_vals = date2num(time_arr, time_units, 
                             calendar='standard', 
                             has_year_zero=True)

        # Acoustic surface tracking distance - AST
        ast = mat['Data']['Burst_AltimeterDistanceAST'].item().squeeze()
        # Interpolate AST to general time stamps using AST time offsets
        time_ast = pd.Series(pd.to_datetime(time_mat-719529, unit='D'))
        # Add AST time offsets to time array
        ast_offs = mat['Data']['Burst_AltimeterTimeOffsetAST'].item().squeeze()
        for i, offs in enumerate(ast_offs):
            time_ast[i] += pd.Timedelta(seconds=offs)        
        # Change time format to match dst.time
        time_ast = time_ast.dt.to_pydatetime()
        # Save AST array in pandas Series
        s = pd.Series(data=ast, index=time_ast)
        s = s.sort_index() # Sort indices (just in case)
        # Interpolate AST data to dst.time
        si = s.reindex(time_arr, method='nearest').interpolate()
        # Despike AST signal if requested
        if despike_ast:
            # Make dataframe for raw & despiked AST signals
            df_ast = si.to_frame(name='raw')
            # Add despiked column
            df_ast['des'] = np.ones_like(si.values) * np.nan
            # Add column for raw signal minus 20-min mean level
            df_ast['rdm'] = np.ones_like(si.values) * np.nan
            # Count number of full 20-minute (1200-sec) segments
            t0s = si.index[0] # Start timestamp
            t1s = si.index[-1] # End timestamp
            nseg = np.floor((t1s - t0s).total_seconds() / 1200)
            # Iterate over approx. 20-min long segments
            for sn, seg in enumerate(np.array_split(si, nseg)):
                # Get segment start and end times
                t0ss = seg.index[0]
                t1ss = seg.index[-1]
                print('seg: {} - {}'.format(t0ss, t1ss))
                # Remove mean from raw signal and save to dataframe
                df_ast['rdm'].loc[t0ss:t1ss] = seg - np.nanmean(seg)
                # Despike segment using GP method
                seg_d, mask_d = self.despike_GP(seg, 
                                                print_kernel=False,
                                               )
                # Save despiked segment to correct indices in df_ast
                df_ast['des'].loc[t0ss:t1ss] = seg_d

        # Velocities from beams 1-4
        vb1 = mat['Data']['Burst_VelBeam1'].item().squeeze()
        vb2 = mat['Data']['Burst_VelBeam2'].item().squeeze()
        vb3 = mat['Data']['Burst_VelBeam3'].item().squeeze()
        vb4 = mat['Data']['Burst_VelBeam4'].item().squeeze()
        # Read number of vertical cells for velocities
        ncells = mat['Config']['Burst_NCells'].item().squeeze()
        cell_arr = np.arange(ncells) # Velocity cell levels
        # Make output dataset and save to netcdf
        ds = xr.Dataset(
            data_vars={'vb1': (['time', 'cell'], vb1),
                       'vb2': (['time', 'cell'], vb2),
                       'vb3': (['time', 'cell'], vb3),
                       'vb4': (['time', 'cell'], vb4),
                       'astr': (['time'], df_ast['rdm'].values),
                       'astd': (['time'], df_ast['des'].values),
                      },
            coords={'time': (['time'], time_arr),
                    'cell': (['cell'], cell_arr)}
            )
        ds = ds.sortby('time') # Sort indices (just in case)

        return ds


    def despike_GP(self, arr, r2thresh=0.9, max_dropouts_perc=20, 
                   longest_dropout_sec=2, **kwargs):
        """
        Despike wave signal following the Gaussian Process-based
        methodology of Malila et al. (2022),

        Parameters:
            arr - 1D wave signal to despike
            r2thresh - scalar; lower threshold for R^2 of GP fit
            max_dropouts_perc - scalar; max fraction of allowed missing
                                values in input array (percent)
            longest_dropout_sec - scalar; longest consecutive missing
                                  value (seconds)
            **kwargs for rpd.GP_despike()
        
        Returns:
            out - despiked array with spikes and missing values
                  replaced with GP mean function.
            mask - boolean spike mask where 0 means bad sample.
        """
        # Copy input array
        z_train = arr.copy() 

        # Set GP length scale bounds based on sampling frequency
        bmin = self.fs # Lower bound (number of samples)
        bmax = self.fs * 6 # Upper bound
        
        # First remove obvious outliers using fixed MAD threshold
        median = np.nanmedian(z_train) # Median of the signal
        # robust STD estimate of the subset
        MAD = mad_std(z_train, ignore_nan=True) 
        mask_MAD = np.abs(z_train - median) > 10*MAD 
        # Set to NaN measurements that lie outside the 
        # +/- 10MAD range from the median (obvious outliers)
        z_train[mask_MAD] = np.nan

        # GP despiking step (iterative)
        z_train -= np.nanmean(z_train) 
        mask_dropouts = np.isnan(z_train) # Mask for dropouts
        if mask_dropouts.sum() > 0:
            # Compute maximum run length
            max_run = max_runlen(mask_dropouts, 1)
        else:
            # If no dropouts => set to zero
            max_run = 0
        # If too many dropouts => skip despiking (set arr to all NaN)
        dropouts_percent = (np.sum(mask_dropouts) / 
                            len(mask_dropouts)) * 100
        if dropouts_percent >= max_dropouts_perc:
            print('Too many dropouts in segment: {}%'.format(
                dropouts_percent))
            # Return NaN array and all zeros mask
            z_desp = np.ones_like(arr) * np.nan
            mask = np.zeros_like(arr)
            return z_desp, mask 
        # If longest dropout > longest allowed 
        # => also skip despiking (set chunk to all NaN)
        elif max_run > (self.dt * longest_dropout_sec):
            print('Longest dropout too long segment: {}sec'.format(
                max_run*self.dt))
             # Return NaN array and all zeros mask
            z_desp = np.ones_like(arr) * np.nan
            mask = np.zeros_like(arr)
            return z_desp, mask 
        else:
            # First iteration: Detect spikes, but don't replace
            z_desp, ms1, _, _, _, th1 = rpd.GP_despike(
                ts=z_train, dropout_mask=~mask_dropouts, 
                despike=True, length_scale_bounds=(bmin, bmax),
                score_thresh=r2thresh, **kwargs
                )
            if ms1.sum() == len(z_train):
                # No spikes were found => copy old mask
                r2_avg = r2thresh + 0.1 # Set arbitrary high r2_avg
                ms_old = mask_dropouts.copy()
            else:
                # Do replacement step if spikes were found
                # Combine dropout and spike masks for replacement step
                mask_dropouts_spikes = np.logical_or(
                    mask_dropouts, ~ms1)
                # Replace using GP mean function
                z_desp, _, _, _, _, th1 = rpd.GP_despike(
                    ts=z_train, dropout_mask=~mask_dropouts_spikes, 
                    despike=False, print_kernel=False, 
                    length_scale_bounds=(bmin, bmax), 
                    )

                # Check minimum R^2 score out of all blocks to decide 
                # whether to continue despiking
                r2_min = np.min(th1['score'])
                # Also compute block-avg R2
                r2_avg = np.mean(th1['score'])
                ms_old = mask_dropouts_spikes.copy()
                # Counter for additional despiking iterations
                cnt_ds = 1 
                # Max. number of despiking iterations to avoid 
                # infinite loop
                max_cnt = 2 
                # Copy R^2 scores for each chunk
                scores_in = th1['score'].copy()
                while r2_min < r2thresh:
                    # Perform despiking until all GP regression blocks 
                    # have R^2 >= r2thresh or until maximum number 
                    # of iterations is passed
                    cnt_ds += 1 # Increase counter
                    if cnt_ds > max_cnt:
                        print(('Max. No. of GP iterations exceeded ' + 
                              '-> moving on ... '))
                        # Force break while loop as soon as max 
                        # iterations are met
                        break
                    print(('r2_min = {:.4f} => Despiking iteration' + 
                          ' #{}').format(r2_min, cnt_ds))
                    # Despike using GP and sklearn
                    z_desp, msn, _, _, _, thn = rpd.GP_despike(
                        ts=z_train, dropout_mask=~ms_old, 
                        despike=True, print_kernel=False, 
                        length_scale_bounds=(bmin, bmax),
                        score_thresh=r2thresh, scores_in=scores_in,
                        )
                    if msn.sum() == len(z_train):
                        print('No new spikes found')
                        break
                    else:
                        # Update mask for replacement (sr) step
                        ms_old = np.logical_or(ms_old, ~msn)
                        # Replace
                        z_desp, _, _, _, _, thn = rpd.GP_despike(
                            ts=z_train, dropout_mask=~ms_old, 
                            despike=False, print_kernel=False, 
                            length_scale_bounds=(bmin, bmax), 
                            )
                        # Update r2_min
                        print('r2_min (updated): {:.4f}'.format(
                            np.min(thn['score'])))
                        r2_min = np.min(thn['score'])
                        # Also update block-avg R2
                        r2_avg = np.mean(thn['score'])
                        # Update scores_in list
                        scores_in = thn['score'].copy()
                    
            # If mean R^2 is high enough, return despiked signal
            # and spike mask
            if r2_avg >= r2thresh:
                # block-avg R2 high enough => return despiked signal
                print('Returning despiked array ... \n')
                return z_desp, ms_old
            else:
                # Block-avg. R2 too low => raise warning but still
                # return despiked array
                print('Block-avg too low: {:.4f} ... \n'.format(
                    r2_avg))
                # z_desp = np.ones_like(arr) * np.nan
                # mask = np.zeros_like(arr)
                return z_desp, ms_old


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
        # Read first mat structure and get start and end timestamps
        times_mat, times = adcp.read_mat_times(fn_mat=adcp.fns[0])
        date0 = times[0].date() # Date of first timestamp
        date1 = times[-1].date() # Date of last timestamp
        print('t0: {}, t1: {}'.format(date0, date1))
        # Save all datasets for the same date in list for concatenating
        dsv_daily = []
        for i,fn_mat in enumerate([adcp.fns[0]]):
            # Check if daily netcdf files already exist
            fn_nc0 = os.path.join(outdir, 
                'Asilomar_SSA_L1_Sig_Vel_{}_{}.nc'.format(
                    self.mid, date0))
            fn_nc1 = os.path.join(outdir, 
                'Asilomar_SSA_L1_Sig_Vel_{}_{}.nc'.format(
                    self.mid, date1))
            if not os.path.isfile(fn_nc0) or not os.path.isfile(fn_nc1):
                # Read mat structure
                dsv = adcp.loaddata_vel(fn_mat)
                # Check if start and end dates the same
                date0 = str(pd.Timestamp(dsv.time[0].values).date())
                date1 = str(pd.Timestamp(dsv.time[-1].values).date())
                if date0 == date1:
                    # Append entire dataset to list for concatenating
                    dsv_daily.append(dsv)
                else:
                    # Split dsv to date0 and date1
                    dsv0 = dsv.sel(time=date0).copy()
                    # Append only correct date
                    dsv_daily.append(dsv0)
                    # Concatenate daily datasets and save to netcdf
                    print('Concatenating daily datasets ...')
                    dsd = xr.concat(dsv_daily)
                    print('Saving daily dataset to netCDF ...')
                    dsd.to_netcdf(fn_nc0)
                    # Make new empty list and append the following day
                    dsv_daily = []
                    dsv1 = dsv.sel(time=date1).copy()
                    dsv_daily.append(dsv1)
                if i==(len(adsp.fns)-1):
                    # Last file, save last netcdf
                    print('Saving last daily dataset to netCDF ...')



