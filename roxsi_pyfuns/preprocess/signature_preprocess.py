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
from scipy.signal import detrend
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime as DT
from cftime import date2num, num2date
from astropy.stats import mad_std
from roxsi_pyfuns import despike as rpd
from roxsi_pyfuns import transfer_functions as rptf


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
                 patm=None, instr='NortekSignature1000'):
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
            patm - pd.DataFrame time series of atmospheric pressure (optional)
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
            self.mid = self.dfm[self.dfm['serial_number'].astype(str)==self.ser][key].item()
        else:
            self.dfm = None # No mooring info dataframe
            self.mid = None # No mooring ID number
        # Atmospheric pressure time series, if provided
        self.patm = patm

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
                     despike_vel=True, despike_ast=True, save_nc=True,
                    ):
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
                          segments. NB: Not implemented!
            despike_ast - bool; if True, despikes Acoustic Surface
                          Tracking (AST) data using Gaussian Process
                          method of Malila et al. (2022) in 20-minute
                          segments.
            save_nc - bool; if True, saves output dataset to netcdf
        
        Returns:
            ds - xarray.Dataset with data read from input .mat file

        """
        # Read .mat structure
        mat = loadmat(fn_mat)

        # As a sanity check, assert that sampling rate is consistent 
        # with user-provided value
        fs_conf = float(mat['Config']['Burst_SamplingRate'].item().squeeze())
        assert float(self.fs) == fs_conf, \
            "Given sampling rate fs={} is not consistent with value in .hdr file".format(
                self.fs)

        # Read and convert general (velocity) time array
        time_mat, time_arr = self.read_mat_times(mat=mat)
        # Convert time array to numerical format
        time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(
            ref_date)
        time_vals = date2num(time_arr, time_units, 
                             calendar='standard', 
                             has_year_zero=True)

        # Velocities from beams 1-4
        vb1 = mat['Data']['Burst_VelBeam1'].item().squeeze()
        vb2 = mat['Data']['Burst_VelBeam2'].item().squeeze()
        vb3 = mat['Data']['Burst_VelBeam3'].item().squeeze()
        vb4 = mat['Data']['Burst_VelBeam4'].item().squeeze()
        # 5th beam velocity and time
        vb5 = mat['Data']['IBurst_VelBeam5'].item().squeeze()
        tb5 = mat['Data']['IBurst_Time'].item().squeeze()
        # Convert to datetime
        tb5 = pd.Series(pd.to_datetime(tb5-719529, unit='D'))
        tb5 = tb5.dt.to_pydatetime()
        # Convert to DataArray for interpolation
        da5 = xr.DataArray(vb5,
                           coords=[tb5, np.arange(28)],
                           dims=['time', 'z'],
                          )
        # Interpolate 5th beam velocity to beam 1-4 time_arr
        dai5 = da5.interp(time=time_arr, method='cubic',
                          kwargs={"fill_value": "extrapolate"}
                         )
        # E,N,U velocities
        vE = mat['Data']['Burst_VelEast'].item().squeeze()
        vN = mat['Data']['Burst_VelNorth'].item().squeeze()
        vU1 = mat['Data']['Burst_VelUp1'].item().squeeze()
        vU2 = mat['Data']['Burst_VelUp2'].item().squeeze()
        # x,y,z velocities
        vx = mat['Data']['Burst_VelX'].item().squeeze()
        vy = mat['Data']['Burst_VelY'].item().squeeze()
        vz1 = mat['Data']['Burst_VelZ1'].item().squeeze()
        vz2 = mat['Data']['Burst_VelZ2'].item().squeeze()

        # Read number of vertical cells for velocities
        ncells = mat['Config']['Burst_NCells'].item().squeeze()
        # Transducer height above bottom (based on Olavo Badaro-Marques'
        # script Signature1000_proc_lvl_1.m)
        trans_hab = 31.88 / 100 # [m]
        # Cell size in meters
        binsz = mat['Config']['Burst_CellSize'].item().squeeze()
        # Height of the first cell center relative to transducer
        # (based on the Principles of Operation manual by Nortek, page 12)
        bl_dist = mat['Config']['Burst_BlankingDistance'].item().squeeze()
        hcc_b1 = bl_dist + binsz # height of cell center for bin #1
        # Get array of cell-center heights
        cell_centers = hcc_b1 + np.arange(ncells) * binsz
        # Account for transducer height above sea floor
        zhab = cell_centers + trans_hab

        # Acoustic surface tracking distance - AST
        ast = mat['Data']['Burst_AltimeterDistanceAST'].item().squeeze()
        # Interpolate AST to general time stamps using AST time offsets
        time_ast = pd.Series(pd.to_datetime(time_mat-719529, unit='D'))
        # Add AST time offsets (fractions of sec) to time array
        ast_offs = mat['Data']['Burst_AltimeterTimeOffsetAST'].item().squeeze()
        for i, offs in enumerate(ast_offs):
            time_ast[i] += pd.Timedelta(seconds=offs)        
        # Change time format to match time_arr
        time_ast = time_ast.dt.to_pydatetime()
        # Save AST array in xarray DataArray for interpolation
        da = xr.DataArray(ast,
                          coords=[time_ast],
                          dims=['time'],
                         )
        # Interpolate AST data to time_arr
        dai = da.interp(time=time_arr, method='cubic',
                        kwargs={"fill_value": "extrapolate"}
                       )
        # Make dataframe for raw & despiked AST signals
        df_ast = dai.to_dataframe(name='raw')
        # Despike AST signal if requested
        if despike_ast:
            # Add despiked column
            df_ast['des'] = np.ones_like(dai.values) * np.nan
            # Add column for raw signal minus 20-min mean level
            df_ast['rdm'] = np.ones_like(dai.values) * np.nan
            # Count number of full 20-minute (1200-sec) segments
            t0s = pd.Timestamp(dai.time.values[0]) # Start timestamp
            t1s = pd.Timestamp(dai.time.values[-1]) # End timestamp
            nseg = np.floor((t1s - t0s).total_seconds() / 1200)
            # Iterate over approx. 20-min long segments
            for sn, seg in enumerate(np.array_split(dai.to_series(), nseg)):
                # Get segment start and end times
                t0ss = seg.index[0]
                t1ss = seg.index[-1]
                print('Despike AST seg: {} - {}'.format(t0ss, t1ss))
                # Despike segment using GP method
                seg_d, mask_d = self.despike_GP(seg, 
                                                print_kernel=False,
                                               )
                # Save despiked segment to correct indices in df_ast
                df_ast['des'].loc[t0ss:t1ss] = seg_d

        # Despike velocities?
        if despike_vel:
            print('Despiking velocities ...')
            # Initialize arrays
            vEd = np.ones_like(vE) * np.nan
            vNd = np.ones_like(vN) * np.nan
            vU1d = np.ones_like(vU1) * np.nan
            vU2d = np.ones_like(vU2) * np.nan
            # Despike each vertical cell at a time
            for j, zr in tqdm(enumerate(zhab)):
                # Despike in 20-min bursts
                dfe = pd.DataFrame(data={'raw':vE[:,j].copy(),
                                         'des':np.ones_like(vE[:,j])*np.nan,
                                        }, 
                                    index=time_arr)
                nseg = (dfe.index[-1] - dfe.index[0]).total_seconds() / 1200
                for sn, seg in enumerate(np.array_split(dfe['raw'], np.floor(nseg))):
                    # Get segment start and end times
                    t0ss = seg.index[0]
                    t1ss = seg.index[-1]
                    # Only despike if range reading below min AST measurement
                    if despike_ast:
                        # Use despiked AST signal if available
                        ast_min = df_ast['des'].loc[t0ss:t1ss].min()
                    else:
                        ast_min = df_ast['raw'].loc[t0ss:t1ss].min()
                    if zr < ast_min:
                        dfe['des'].loc[t0ss:t1ss] = self.despike_GN02(
                            seg.values.squeeze())
                vEd[:,j] = dfe['des'].values
                # North velocity for current range
                dfn = pd.DataFrame(data={'raw':vN[:,j].copy(),
                                         'des':np.ones_like(vN[:,j])*np.nan,
                                        }, 
                                    index=time_arr)
                nseg = (dfn.index[-1] - dfn.index[0]).total_seconds() / 1200
                for sn, seg in enumerate(np.array_split(dfn['raw'], np.floor(nseg))):
                    t0ss = seg.index[0]
                    t1ss = seg.index[-1]
                    if zr < ast_min:
                        dfn['des'].loc[t0ss:t1ss] = self.despike_GN02(
                            seg.values.squeeze())
                vNd[:,j] = dfn['des'].values            
                # Up1 velocity
                dfu1 = pd.DataFrame(data={'raw':vU1[:,j].copy(),
                                          'des':np.ones_like(vU1[:,j])*np.nan,
                                         }, 
                                    index=time_arr)
                nseg = (dfu1.index[-1] - dfu1.index[0]).total_seconds() / 1200
                for sn, seg in enumerate(np.array_split(dfu1['raw'], np.floor(nseg))):
                    t0ss = seg.index[0]
                    t1ss = seg.index[-1]
                    if zr < ast_min:
                        dfu1['des'].loc[t0ss:t1ss] = self.despike_GN02(
                            seg.values.squeeze())
                vU1d[:,j] = dfu1['des'].values
                # Up2 velocity
                dfu2 = pd.DataFrame(data={'raw':vU2[:,j].copy(),
                                          'des':np.ones_like(vU2[:,j])*np.nan,
                                         }, 
                                    index=time_arr)
                nseg = (dfu2.index[-1] - dfu2.index[0]).total_seconds() / 1200
                for sn, seg in enumerate(np.array_split(dfu2['raw'], np.floor(nseg))):
                    t0ss = seg.index[0]
                    t1ss = seg.index[-1]
                    if zr < ast_min:
                        dfu2['des'].loc[t0ss:t1ss] = self.despike_GN02(
                            seg.values.squeeze())
                vU2d[:,j] = dfu2['des'].values

        # Also read pressure and reconstruct linear sea-surface elevation
        pres = mat['Data']['Burst_Pressure'].item().squeeze()
        # Make pd.Series and convert to eta
        pres = pd.Series(pres, index=time_arr)
        dfp = self.p2eta_lin(pres)

        # Also read temperature
        temp = mat['Data']['Burst_Temperature'].item().squeeze()

        # Read heading, pitch & roll timeseries
        heading = mat['Data']['Burst_Heading'].item().squeeze()
        pitch = mat['Data']['Burst_Pitch'].item().squeeze()
        roll = mat['Data']['Burst_Roll'].item().squeeze()

        # Define variable dictionary for output dataset
        data_vars={'vB1': (['time', 'range'], vb1), # Beam coord. vel.
                   'vB2': (['time', 'range'], vb2),
                   'vB3': (['time', 'range'], vb3),
                   'vB4': (['time', 'range'], vb4),
                   'vB5': (['time', 'range'], dai5.values),
                   # East, North, Up velocities
                   'vE': (['time', 'range'], vE),
                   'vN': (['time', 'range'], vN),
                   'vU1': (['time', 'range'], vU1),
                   'vU2': (['time', 'range'], vU2),
                   # x,y,z velocities
                   'vX': (['time', 'range'], vx),
                   'vY': (['time', 'range'], vy),
                   'vZ1': (['time', 'range'], vz1),
                   'vZ2': (['time', 'range'], vz2),
                   # Raw AST 
                   'ASTr': (['time'], df_ast['raw'].values),
                   # Pressure and reconstructed sea surface
                   'pressure':  (['time'], dfp['pressure']),
                   'eta_hyd':  (['time'], dfp['eta_hyd']),
                   'eta_lin':  (['time'], dfp['eta_lin']),
                   # Temperature (not calibrated?)
                   'temperature':  (['time'], temp),
                   # Heading, pitch, roll
                   'heading_ang':  (['time'], heading),
                   'pitch_ang':  (['time'], pitch),
                   'roll_ang':  (['time'], roll),
                   }
        # Add despiked arrays if available
        if despike_ast:
            # GP-despiked AST
            data_vars['ASTd'] = (['time'], df_ast['des'].values)
        if despike_vel:
            # Despiked East, North, Up velocities
            data_vars['vEd'] = (['time', 'range'], vEd)
            data_vars['vNd'] = (['time', 'range'], vNd)
            data_vars['vU1d'] = (['time', 'range'], vU1d)
            data_vars['vU2d'] = (['time', 'range'], vU2d)

        # Make output dataset and save to netcdf
        ds = xr.Dataset(data_vars=data_vars,
                        coords={'time': (['time'], time_arr),
                                'range': (['range'], zhab)}
            )
        ds = ds.sortby('time') # Sort indices (just in case)

        return ds


    def despike_GN02(self, u, interp='linear', sec_lim=2, corrd=True, 
                     min_new_spikes=10, max_iter=3):
        """
        Despike Nortek Vector velocities using low correlation values
        to discard unreliable measurements following the Goring and
        Nikora (2002, J. Hydraul. Eng.) phase space method, including the
        modifications by Wahl (2003, J. Hydraul. Eng.) and Mori et al.
        (2007, J. Eng. Mech.).

        Parameters:
            u - array; 1D array with velocity time series.
            interp - str; interpolation method for discarded data.
                          Options: see pandas.DataFrame.interpolate()
            sec_lim - scalar; maximum gap size to interpolate (sec)
            corrd - bool; if True, input velocities are correlation-
                    corrected by self.despike_correlations()
            min_new_spikes - int; iterate until number of new spikes detected 
                             is lower than this value. 
            max_iter - int; maximum number of despiking iterations
        
        Returns:
            ud - Despiked velocity array.
        """
 
        # Copy velocity array so we don't change the input
        ud = u.copy()
        # Make into pd.Series for easier interpolation
        ud = pd.Series(ud)
        
        # Initialize counter of new spikes detected
        n_spikes = min_new_spikes + 1
        # Initialize iteraction counter
        cnt = 0
        # Iterate until sufficiently low number of new spikes detected
        while n_spikes > min_new_spikes or cnt < max_iter:
            # Detect spikes using 3D phase space method
            mask = rpd.phase_space_3d(ud)
            # Convert detected spikes to NaN
            ud[mask] = np.nan
            # Interpolate according to requested method
            ud.interpolate(method=interp, limit=sec_lim*self.fs, 
                           inplace=True)
            # Count number of spikes detected
            n_spikes = mask.sum()
            # Add to iteration counter
            cnt += 1
            if n_spikes < min_new_spikes or cnt >= max_iter:
                break

        return ud.values


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
        z_mean = np.nanmean(z_train)
        z_train -= z_mean 
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
                    
            # Add back mean
            z_desp += z_mean

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


    def p2eta_lin(self, pt, rho0=1025, grav=9.81, M=512, fmin=0.05, 
                  fmax=0.33, att_corr=True,  detrend_out=True, 
                  return_hyd=True):
        """
        Use linear transfer function to reconstruct sea-surface
        elevation time series from sub-surface pressure measurements.

        If self.patm dataframe of atmospheric pressure is not available,
        this function assumes that the input time series is the hydrostatic 
        pressure.

        Parameters:
            pt - pd.Series; time series of water pressure
            rho0 - scalar; water density (kg/m^3)
            grav - scalar; gravitational acceleration (m/s^2)
            M - int; window segment length (512 by default)
            fmin - scalar; min. cutoff frequency
            fmax - scalar; max. cutoff frequency
            att_corr - bool; if True, applies attenuation correction
            detrend_out - bool; if True, returns detrended signal
            return_hyd - bool; if True, returns also hydrostatic pressure head
        """
        # Copy input
        pw = pt.copy()
        pw = pw.to_frame(name='pressure') # Convert to dataframe
        # Use hydrostatic assumption to get pressure head with unit [m]
        pw['eta_hyd'] = rptf.eta_hydrostatic(pw['pressure'], self.patm, 
            rho0=rho0, grav=grav, interp=True)
        # Check if hydrostatic pressure is ever above 0
        if pw['eta_hyd'].max() == 0.0:
            print('Instrument most likely not in water')
            # Return NaN array for linear sea surface elevations
            pw['eta_lin'] = np.ones_like(pw['eta_hyd'].values) * np.nan
        else:
            # Apply linear transfer function from p->eta
            trf = rptf.TRF(fs=self.fs, zp=self.zp, type=self.instr)
            pw['eta_lin'] = trf.p2eta_lin(pw['eta_hyd'], M=M, fmin=fmin, fmax=fmax,
                att_corr=att_corr)
            # Detrend if requested
            if detrend_out:
                pw['eta_lin'] = detrend(pw['eta_lin'].values)

        # Return dataframe
        return pw


# Main script
if __name__ == '__main__':
    """
    Main script for pre-processing Vector Signature1000 raw data.
    """
    from argparse import ArgumentParser

    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data root directory"),
                type=str,
                # default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/Signatures',
                default='/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray/Signatures',
                )
        parser.add_argument("-ser", 
                help=('Instrument serial number. To loop through all, select "ALL".'),
                type=str,
                choices=['103088', '103094', '103110', '103063', '103206'],
                default='103094',
                )
        parser.add_argument("-M", 
                help=("Pressure transform segment window length"),
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

    # QC plot functions
    def plot_hpr(ds, fn=None, ser=None):
        """
        Plot time series of pressure + heading, pitch and roll 
        from xr.Dataset.

        Parameters:
            ds - xr.Dataset with pressure + heading, pitch & roll data
            fn - str; figure filename (full path). If None, only shows
                 the figure
            ser - str; (optional) instrument serial number (or other string)
                  for plot title
        """
        # Initialize figure
        fig, axes = plt.subplots(figsize=(12,6), nrows=2, sharex=True,
                                 constrained_layout=True)
        # On first row plot H,P,R
        ds.heading_ang.plot(ax=axes[0], label='heading')
        ds.pitch_ang.plot(ax=axes[0], label='pitch')
        ds.roll_ang.plot(ax=axes[0], label='roll')
        # Plot horizontal dashed line at +/-25 deg
        axes[0].axhline(y=25, linestyle='--', color='k', alpha=0.6)
        axes[0].axhline(y=-25, linestyle='--', color='k', alpha=0.6)
        datestr = str(pd.Timestamp(dsv.time[0].values).date())
        if ser is not None:
            axes[0].set_title('Nortek Sig1000 {} {}'.format(ser, datestr))
        else:
            axes[0].set_title('Nortek Sig1000 {}'.format(datestr))
        axes[0].set_ylabel('Angle [deg]')
        axes[0].legend()
        # On second row plot pressure
        ds.pressure.plot(ax=axes[1], label='pressure')
        axes[1].set_ylabel('Pressure [hPa]')
        axes[1].legend()
        # Save if filename given
        plt.tight_layout()
        if fn is not None:
            plt.savefig(fn, bbox_inches='tight', dpi=300)
        else:
            # Else, show plot
            plt.show()
        plt.close()

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

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
        # Define input datadir
        datadir = os.path.join(args.dr, 'raw', ser)
        # Define Level1 output directories
        outdir = os.path.join(args.dr, 'Level1', ser)
        if not os.path.isdir(outdir):
            print('Making output dir. {}'.format(outdir))
            os.mkdir(outdir)
        figdir = os.path.join(outdir, 'img')
        if not os.path.isdir(figdir):
            print('Making output figure dir. {}'.format(figdir))
            os.mkdir(figdir)
        # Initialize class
        adcp = ADCP(datadir=datadir, ser=ser, mooring_info=fn_minfo)
        # Save all datasets for the same date in list for concatenating
        dsv_daily = [] # Velocities and 1D (eg AST) data
        dse_daily = [] # Echogram data
        # Loop over raw .mat files and save daily data as netcdf
        for i,fn_mat in enumerate(adcp.fns):
            # Check if daily netcdf files already exist
            times_mat, times = adcp.read_mat_times(fn_mat=fn_mat)
            date0 = str(times[0].date()) # Date of first timestamp
            date1 = str(times[-1].date()) # Date of last timestamp
            date0_str = ''.join(date0.split('-'))
            date1_str = ''.join(date1.split('-'))
            fn_nc0 = os.path.join(outdir, 
                'Asilomar_SSA_L1_Sig_Vel_{}_{}.nc'.format(
                    adcp.mid, date0_str))
            fn_nc1 = os.path.join(outdir, 
                'Asilomar_SSA_L1_Sig_Vel_{}_{}.nc'.format(
                    adcp.mid, date1_str))
            if not os.path.isfile(fn_nc0) or not os.path.isfile(fn_nc1):
                # Read mat structure for velocities and 1D timeseries
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
                    print('Concatenating daily datasets for {} ...'.format(
                        date0))
                    dsd = xr.concat(dsv_daily, dim='time')
                    if not os.path.isfile(fn_nc0):
                        print('Saving daily dataset for {} to netCDF ...'.format(
                            date0))
                        dsd.to_netcdf(fn_nc0)
                    # Make daily QC plots
                    fn_hpr = os.path.join(figdir, 'qc_hpr_{}_{}.pdf'.format(
                        ser, date0_str))
                    if not os.path.isfile(fn_hpr):
                        plot_hpr(dsd, fn=fn_hpr, ser=ser)
                    # Make new empty list and append the following day
                    dsv_daily = []
                    dsv1 = dsv.sel(time=date1).copy()
                    dsv_daily.append(dsv1)
                if i == (len(adcp.fns)-1):
                    # Last file, save last netcdf
                    dsd = xr.concat(dsv_daily, dim='time')
                    if not os.path.isfile(fn_nc0):
                        print('Saving last dataset for {} to netCDF ...'.format(
                            date0))
                        dsd.to_netcdf(fn_nc0)
                    # Make daily QC plots
                    fn_hpr = os.path.join(figdir, 'qc_hpr_{}_{}.pdf'.format(
                        ser, date0_str))
                    if not os.path.isfile(fn_hpr):
                        plot_hpr(dsd, fn=fn_hpr, ser=ser)



