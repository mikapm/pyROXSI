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
from PyPDF2 import PdfFileMerger, PdfFileReader
from roxsi_pyfuns import despike as rpd
from roxsi_pyfuns import transfer_functions as rptf
from roxsi_pyfuns import coordinate_transforms as rpct
from roxsi_pyfuns import wave_spectra as rpws


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
                 magdec=12.86, beam_ang=25, binsz=0.5, outdir=None, 
                 mooring_info=None, patm=None, 
                 instr='NortekSignature1000'):
        """
        Initialize ADCP class.

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

    def contamination_range(self, ha, binsz):
        """
        Calculate range of velocity contamination due to sidelobe 
        reflections following Lentz et al. (2021, Jtech): 
        DOI: 10.1175/JTECH-D-21-0075.1

        According to L21 (Eq. 2), the range cells contaminated by 
        sidelobe reflections are given by
            
            z_ic < h_a * (1 - cos(theta)) + 3*dz/2,
        
        where z_ic is depth below the surface of the contaminated region
        (range cell centers), h_a is the distance from the ADCP acoustic 
        head to the surface, theta is the ADCP beam angle from the 
        vertical (25 for Sig1000) and dz is the bin size in meters. 

        Parameters:
            ha - distance from ADCP acoustic head to sea surface
            binsz - binsize in meters
            (beam angle is saved in self.beam_ang)

        Returns:
            zic - depth below the surface of the contaminated region
        """
        zic = ha * (1 - np.cos(self.beam_ang)) + 3 * binsz / 2
        return zic


    def loaddata_vel(self, fn_mat, ref_date=pd.Timestamp('2022-06-25'),
                     despike_vel=True, despike_ast=True, only_hpr=False):
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
            only_hpr - bool; if True, output only pd.Dataframe of heading,
                       pitch and roll.
        
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

        # Read heading, pitch & roll timeseries
        heading = mat['Data']['Burst_Heading'].item().squeeze()
        pitch = mat['Data']['Burst_Pitch'].item().squeeze()
        roll = mat['Data']['Burst_Roll'].item().squeeze()

        # If requesting only H,D,R timeseries, return dataframe
        if only_hpr:
            pres = mat['Data']['Burst_Pressure'].item().squeeze()
            df_hpr = pd.DataFrame(data={'heading_ang':heading, 
                                        'pitch_ang':pitch, 
                                        'roll_ang':roll, 
                                        'pressure':pres,
                                        },
                                  index=time_arr,
                                  )
            return df_hpr

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
        vb5 = dai5.values
        # Calculate E,N,U velocities from beam velocities
        # Note order and sign of beams!!!
        beam_arr = np.array([-vb1, -vb3, -vb4, -vb2, -vb5])
        print('beam_arr.shape: ', beam_arr.shape)
#         enu_vel = rpct.beam2enu(beam_arr, heading=heading, 
#                                 pitch=pitch, roll=roll)

        # Nortek E,N,U velocities
        vE = mat['Data']['Burst_VelEast'].item().squeeze()
        vN = mat['Data']['Burst_VelNorth'].item().squeeze()
        vU1 = mat['Data']['Burst_VelUp1'].item().squeeze()
        vU2 = mat['Data']['Burst_VelUp2'].item().squeeze()
#         print('vE.shape: ', vE.shape)
#         print('vE: ', vE[:20,0])
#         print('enu_vel.shape: ', enu_vel.shape)
#         print('vEc: ', enu_vel[0,:20,0])

        # Read number of vertical cells for velocities
        ncells = mat['Config']['Burst_NCells'].item().squeeze()
        # Transducer height above bottom (based on Olavo Badaro-Marques'
        # script Signature1000_proc_lvl_1.m)
        trans_hab = 31.88 / 100 # [m]
        # Cell size in meters
        binsz = mat['Config']['Burst_CellSize'].item().squeeze()
        # Check that bin size is consistent with self.binsz
        assert float(self.binsz) == binsz, \
            "Given sampling rate fs={} is not consistent with value in .hdr file".format(
                self.binsz)
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
            nseg = np.round((t1s - t0s).total_seconds() / 1200)
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

        # Despike (beam) velocities?
        if despike_vel:
            print('Despiking beam velocities ...')
            # Initialize arrays
#             vb1d = np.ones_like(vb1) * np.nan
#             vb2d = np.ones_like(vb2) * np.nan
#             vb3d = np.ones_like(vb3) * np.nan
#             vb4d = np.ones_like(vb4) * np.nan
#             vb5d = np.ones_like(vb5) * np.nan
#             # Save despiked velocities in dfb DataFrame
#             dfb = pd.DataFrame(data={'vb1d':vb1d, 'vb2d':vb2d, 'vb3d':vb3d,
#                                      'vb4d':vb4d, 'vb5d':vb5d},
#                                index=time_arr,
#                               )
            # Initialize arrays
            vEd = np.ones_like(vb1) * np.nan
            vNd = np.ones_like(vb2) * np.nan
            vU1d = np.ones_like(vb3) * np.nan
            vU2d = np.ones_like(vb4) * np.nan
            # Save despiked velocities in dfb DataFrame
            dsb = xr.Dataset(data_vars={'vE': (['time', 'z'], vE), 
                                        'vN': (['time', 'z'], vN),
                                        'vU1': (['time', 'z'], vU1), 
                                        'vU2': (['time', 'z'], vU2),
                                        'vEd': (['time', 'z'], vEd), 
                                        'vNd': (['time', 'z'], vNd),
                                        'vU1d': (['time', 'z'], vU1d), 
                                        'vU2d': (['time', 'z'], vU2d),
                                        },
                            coords={'time': (['time'], time_arr),
                                    'z': (['z'], np.arange(28))
                                    },
                            )
            # Despike each beam at a time
            cols_r = ['vE', 'vN', 'vU1', 'vU2']
            cols_d = ['vEd', 'vNd', 'vU1d', 'vU2d']
            for colr, cold in zip(cols_r, cols_d):
                # Despike each vertical cell at a time
                for j, zr in tqdm(enumerate(zhab)):
                    # Despike in 20-min bursts
                    dfi = pd.DataFrame(data={'raw':dsb[colr][:,j].values.copy(),
                                             'des':np.ones_like(dsb[cold][:,j].values)*np.nan,
                                            }, 
                                       index=time_arr)
                    # Number of 20-min segments
                    nseg = (pd.Timestamp(dsb.time[-1].values) - 
                            pd.Timestamp(dsb.time[0].values)).total_seconds() / 1200
                    # Iterate over segments and despike
                    for sn, seg in enumerate(np.array_split(dfi['raw'], np.round(nseg))):
                        # Get segment start and end times
                        t0ss = seg.index[0]
                        t1ss = seg.index[-1]
                        # Only despike if range reading below min AST measurement
                        if despike_ast:
                            # Use despiked AST signal if available
                            ast_min = df_ast['des'].loc[t0ss:t1ss].min()
                        else:
                            ast_min = df_ast['raw'].loc[t0ss:t1ss].min()
                        # Subtract half of a binsize from ast_min (b/c
                        # range values are in middles of bins)
                        ast_min -= binsz / 2
                        if zr < ast_min:
                            dfi['des'].loc[t0ss:t1ss] = self.despike_GN02(
                                seg.values.squeeze())
                    # Save despiked segment in correct slice of dataframe
                    dsb[cold][:,j] = dfi['des'].values
        
        # Convert despiked beam velocities to E,N,U coordinates


        # Also read pressure and reconstruct linear sea-surface elevation
        pres = mat['Data']['Burst_Pressure'].item().squeeze()
        # Make pd.Series and convert to eta
        pres = pd.Series(pres, index=time_arr)
        dfp = self.p2eta_lin(pres)

        # Also read temperature
        temp = mat['Data']['Burst_Temperature'].item().squeeze()

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
            # Despiked beam velocities
#             data_vars['vB1d'] = (['time', 'range'], dfb['vb1d'].values)
#             data_vars['vB2d'] = (['time', 'range'], dfb['vb2d'].values)
#             data_vars['vB3d'] = (['time', 'range'], dfb['vb3d'].values)
#             data_vars['vB4d'] = (['time', 'range'], dfb['vb4d'].values)
#             data_vars['vB5d'] = (['time', 'range'], dfb['vb5d'].values)
            data_vars['vEd'] = (['time', 'range'], dsb['vEd'].values)
            data_vars['vNd'] = (['time', 'range'], dsb['vNd'].values)
            data_vars['vU1d'] = (['time', 'range'], dsb['vU1d'].values)
            data_vars['vU2d'] = (['time', 'range'], dsb['vU2d'].values)

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
                 ):
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

        
    def wavespec(self, ds, u='vEd', v='vNd', z='ASTd', seglen=1200, 
                 fmin=0.05, fmax=0.33):
        """
        Estimate wave spectra from ADCP data in the input dataset ds.
        Uses the despiked N&E velocities and AST surface elevation
        timeseries by default. Returns new dataset with spectra and
        some bulk parameters.

        Horizontal velocities are taken from the first range bin
        below the region of contamination due to sidelobe reflections
        following the definition of Lentz et al. (2021, Jtech). 

        Parameters:
            ds - input xr.Dataset. Must include variables for u,v,z
            u - str; key for desired u variable in ds
            v - str; key for desired v variable in ds
            z - str; key for desired z variable in ds
            seglen - scalar; spectral segment length in seconds 
            fmin - scalar; min. frequency for computing bulk params
            fmax - scalar; max. frequency for computing bulk params

        Returns:
            dss_concat - combined dataset of spectral segments
        """
        # Interpolate over possible dropouts in AST signal
        zA = ds[z].interpolate_na(dim='time', fill_value="extrapolate").values
        zA = pd.Series(zA, index=ds.time.values)


        # Count number of full 20-minute (1200-sec) segments
        t0s = pd.Timestamp(zA.index[0]) # Start timestamp
        t1s = pd.Timestamp(zA.index[-1]) # End timestamp
        nseg = np.round((t1s - t0s).total_seconds() / seglen)
        print('Estimating spectra ...')
        dss_list = [] # Empty list for concatenating
        for sn, seg in enumerate(np.array_split(zA, nseg)):
            # Get segment start and end times
            t0ss = seg.index[0]
#            print('seg start: {}'.format(t0ss))
#            print('seg start (round): {}'.format(
#                 pd.Timestamp(t0ss).round('20T')))
#             print(' ')
            t1ss = seg.index[-1]
            # Take time slice from dataset
            ds_seg = ds.sel(time=slice(t0ss, t1ss))
            # Estimate depth from surface of contamination region
            zic = self.contamination_range(binsz=self.binsz, ha=seg.min())
            # Get optimal velocity range bin number following Lentz et al. (2021)
            # Use 'bfill' to be conservative (round up)
            z_opt = seg.min() - ds_seg.sel(range=zic,
                                           method='bfill').range.item()
            # Save range cell value
            range_val = ds_seg.sel(range=z_opt,
                                   method='nearest').range.item() 
            # Interpolate E&N velocities
            vEd = ds_seg[u].interpolate_na(dim='time',
                fill_value="extrapolate").sel(range=z_opt, 
                                              method='nearest').values
            vEd = pd.Series(vEd, index=seg.index)
            vNd = ds_seg[v].interpolate_na(dim='time',
                fill_value="extrapolate").sel(range=z_opt, 
                                              method='nearest').values
            vNd = pd.Series(vNd, index=seg.index)
            # Estimate spectra from 20-min. segments
            dss = rpws.spec_uvz(z=seg, 
                                u=vEd, 
                                v=vNd, 
                                fs=adcp.fs,
                                timestamp=pd.Timestamp(t0ss).round('20T'),
                                fmin=fmin,
                                fmax=fmax,
                                )
            # Add range value to output dataset
            dss['vel_binh'] = (['time'], np.atleast_1d(range_val))
            # Append to list
            dss_list.append(dss)
        # Concatenate all spectrum datasets into one
        dss_concat = xr.concat(dss_list, dim='time')

        return dss_concat


    def save_vel_nc(self, ds, fn, overwrite=False, fillvalue=-9999.,
                     ref_date=pd.Timestamp('2022-06-25'), ):
        """
        Save velocity/AST/pressure dataset ds to netcdf format.

        Parameters:
            ds - input spectral xr.Dataset
            fn - str; path for netcdf filename
            overwrite - bool; if False, does not overwrite existing file
            fillvalue - scalar; fill value to denote missing values
            ref_date - reference date to use for time axis
        """
        # Check if file already exists
        if os.path.isfile(fn) and not overwrite:
            # Read and return existing dataset
            print('Requested netCDF file already exists. Set overwrite=True to overwrite.')
            return 
        # Set requested fill value
        ds = ds.fillna(fillvalue)
        # Convert time array to numerical format
        time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(ref_date)
        time = pd.to_datetime(ds.time.values).to_pydatetime()
        time_vals = date2num(time, 
                             time_units, calendar='standard', 
                             has_year_zero=True)
        ds.coords['time'] = time_vals.astype(float)
        # Make lon, lat coordinates
        lon = self.dfm[self.dfm['mooring_ID_long']==self.midl]['longitude'].item()
        ds = ds.assign_coords(lon=[lon])
        lat = self.dfm[self.dfm['mooring_ID_long']==self.midl]['latitude'].item()
        ds = ds.assign_coords(lat=[lat])
        # Set variable attributes for output netcdf file
        ds.lat.attrs['standard_name'] = 'latitude'
        ds.lat.attrs['long_name'] = 'Approximate latitude of small-scale array rock summit'
        ds.lat.attrs['units'] = 'degrees_north'
        ds.lat.attrs['valid_min'] = -90.0
        ds.lat.attrs['valid_max'] = 90.0
        ds.lon.attrs['standard_name'] = 'longitude'
        ds.lon.attrs['long_name'] = 'Approximate longitude of small-scale array rock summit'
        ds.lon.attrs['units'] = 'degrees_east'
        ds.lon.attrs['valid_min'] = -180.0
        ds.lon.attrs['valid_max'] = 180.0
        ds.time.encoding['units'] = time_units
        ds.time.attrs['units'] = time_units
        ds.time.attrs['standard_name'] = 'time'
        ds.time.attrs['long_name'] = 'Local time (PDT) of spectral segment start'
        # Sig. wave height and other integrated parameters
        ds.vel_binh.attrs['units'] = 'm'
        ds.vel_binh.attrs['standard_name'] = 'height'
        ds.vel_binh.attrs['long_name'] = 'horizontal velocity bin center height above seabed'
        ds.Hm0.attrs['units'] = 'm'
        ds.Hm0.attrs['standard_name'] = 'sea_surface_wave_significant_height'
        ds.Hm0.attrs['long_name'] = 'Hs estimate from 0th spectral moment'
        ds.Te.attrs['units'] = 's'
        ds.Te.attrs['standard_name'] = 'sea_surface_wave_energy_period'
        ds.Te.attrs['long_name'] = 'energy-weighted wave period'
        ds.Tp_ind.attrs['units'] = 's'
        ds.Tp_ind.attrs['standard_name'] = 'sea_surface_primary_swell_wave_period_at_variance_spectral_density_maximum'
        ds.Tp_ind.attrs['long_name'] = 'wave period at maximum spectral energy'
        ds.Tp_Y95.attrs['units'] = 's'
        ds.Tp_Y95.attrs['standard_name'] = 'sea_surface_primary_swell_wave_period_at_variance_spectral_density_maximum'
        ds.Tp_Y95.attrs['long_name'] = 'peak wave period following Young (1995, Ocean Eng.)'
        ds.Dp_ind.attrs['units'] = 'angular_degree' 
        ds.Dp_ind.attrs['standard_name'] = 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum'
        ds.Dp_ind.attrs['long_name'] = 'peak wave direction at maximum energy frequency'
        ds.Dp_Y95.attrs['units'] = 'angular_degree' 
        ds.Dp_Y95.attrs['standard_name'] = 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum'
        ds.Dp_Y95.attrs['long_name'] = 'peak wave direction at Tp_Y95 frequency'
        ds.nu_LH57.attrs['units'] = 'dimensionless' 
        ds.nu_LH57.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density_bandwidth'
        ds.nu_LH57.attrs['long_name'] = 'spectral bandwidth following Longuet-Higgins (1957)'
        # Fill values
        ds.vel_binh.attrs['missing_value'] = fillvalue
        ds.Hm0.attrs['missing_value'] = fillvalue
        ds.Te.attrs['missing_value'] = fillvalue
        ds.Tp_ind.attrs['missing_value'] = fillvalue
        ds.Tp_Y95.attrs['missing_value'] = fillvalue
        ds.Dp_Y95.attrs['missing_value'] = fillvalue
        ds.Dp_ind.attrs['missing_value'] = fillvalue
        ds.nu_LH57.attrs['missing_value'] = fillvalue

       # Global attributes
        ds.attrs['title'] = ('ROXSI 2022 Asilomar Small-Scale Array ' + 
                             'Signature1000 {} wave spectra'.format(self.mid))
        if z_var == 'ASTd':
            ds.attrs['summary'] =  ('Nearshore wave spectra from ADCP measurements. '+
                                    'Sea-surface elevation is the despiked ADCP ' + 
                                    'acoustic surface track (AST) signal, and the ' + 
                                    'horizontal velocities are despiked E&N velocities ' +
                                    'from the range bin specified by the variable ' +
                                    'vel_binh.')
        elif z_var == 'eta_lin':
            ds.attrs['summary'] =  ('Nearshore wave spectra from ADCP measurements. '+
                                    'Sea-surface elevation is the linear sea-surface ' + 
                                    'reconstruction from pressure, and the ' + 
                                    'horizontal velocities are despiked E&N velocities ' +
                                    'from the range bin specified by the variable ' +
                                    'vel_binh.')
        ds.attrs['instrument'] = 'Nortek Signature 1000'
        ds.attrs['mooring_ID'] = self.mid
        ds.attrs['serial_number'] = self.ser
        ds.attrs['segment_length'] = '1200 seconds'
        # Read more attributes from mooring info file if provided
        if self.dfm is not None:
            comments = self.dfm[self.dfm['mooring_ID_long']==self.midl]['notes'].item()
            ds.attrs['comment'] = comments
            config = self.dfm[self.dfm['mooring_ID_long']==self.midl]['config'].item()
            ds.attrs['configurations'] = config
        ds.attrs['Conventions'] = 'CF-1.8'
        ds.attrs['featureType'] = "timeSeries"
        ds.attrs['source'] =  "Sub-surface observation"
        ds.attrs['date_created'] = str(DT.utcnow()) + ' UTC'
        ds.attrs['references'] = 'https://github.com/mikapm/pyROXSI'
        ds.attrs['creator_name'] = "Mika P. Malila"
        ds.attrs['creator_email'] = "mikapm@unc.edu"
        ds.attrs['institution'] = "University of North Carolina at Chapel Hill"

        # Set encoding before saving
        encoding = {'time': {'zlib': False, '_FillValue': None},
                    'freq': {'zlib': False, '_FillValue': None},
                    'lat': {'zlib': False, '_FillValue': None},
                    'lon': {'zlib': False, '_FillValue': None},
                    'Euu': {'_FillValue': fillvalue},        
                    'Evv': {'_FillValue': fillvalue},        
                    'Ezz': {'_FillValue': fillvalue},        
                    'a1': {'_FillValue': fillvalue},        
                    'a2': {'_FillValue': fillvalue},        
                    'b1': {'_FillValue': fillvalue},        
                    'b2': {'_FillValue': fillvalue},        
                    'dspr': {'_FillValue': fillvalue},        
                    'vel_binh': {'_FillValue': fillvalue},        
                    'Hm0': {'_FillValue': fillvalue},        
                    'Te': {'_FillValue': fillvalue},
                    'Tp_ind': {'_FillValue': fillvalue},
                    'Tp_Y95': {'_FillValue': fillvalue},
                    'Dp_ind': {'_FillValue': fillvalue},
                    'Dp_Y95': {'_FillValue': fillvalue},
                    'nu_LH57': {'_FillValue': fillvalue},
                   }     

        # Save dataset in netcdf format
        print('Saving netcdf ...')
        ds.to_netcdf(fn, encoding=encoding)



    def save_spec_nc(self, ds, fn, overwrite=False, fillvalue=-9999.,
                     ref_date=pd.Timestamp('2022-06-25'), z_var='ASTd'):
        """
        Save wave spectrum dataset ds to netcdf format.

        Parameters:
            ds - input spectral xr.Dataset
            fn - str; path for netcdf filename
            overwrite - bool; if False, does not overwrite existing file
            fillvalue - scalar; fill value to denote missing values
            ref_date - reference date to use for time axis
            z_var - str; heave variable to use (ex. ASTd or eta_lin)
        """
        # Check if file already exists
        if os.path.isfile(fn) and not overwrite:
            # Read and return existing dataset
            print('Requested netCDF file already exists. Set overwrite=True to overwrite.')
            return 
        # Set requested fill value
        ds = ds.fillna(fillvalue)
        # Convert time array to numerical format
        time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(ref_date)
        time = pd.to_datetime(ds.time.values).to_pydatetime()
        time_vals = date2num(time, 
                             time_units, calendar='standard', 
                             has_year_zero=True)
        ds.coords['time'] = time_vals.astype(float)
        # Make lon, lat coordinates
        ds = ds.assign_coords(lon=[36.6250768146788])
        ds = ds.assign_coords(lat=[-121.94334673203694])
        # Set variable attributes for output netcdf file
        ds.Ezz.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density'
        if z_var == 'ASTd':
            z_str = 'AST'
        elif z_var == 'eta_lin':
            z_str = 'linear pressure reconstruction'
        ds.Ezz.attrs['long_name'] = 'scalar (frequency) wave variance density spectrum from {}'.format(
            z_str)
        ds.Ezz.attrs['units'] = 'm^2/Hz'
        ds.Evv.attrs['units'] = 'm^2/Hz'
        ds.Evv.attrs['standard_name'] = 'northward_sea_water_velocity_variance_spectral_density'
        ds.Evv.attrs['long_name'] = 'auto displacement spectrum from northward velocity component'
        ds.Euu.attrs['units'] = 'm^2/Hz'
        ds.Euu.attrs['standard_name'] = 'eastward_sea_water_velocity_variance_spectral_density'
        ds.Euu.attrs['long_name'] = 'auto displacement spectrum from eastward velocity component'
        ds.a1.attrs['units'] = 'dimensionless'
        ds.a1.attrs['standard_name'] = 'a1_directional_fourier_component'
        ds.a1.attrs['long_name'] = 'a1 following Kuik et al. (1988) and Herbers et al. (2012)'
        ds.a2.attrs['units'] = 'dimensionless'
        ds.a2.attrs['standard_name'] = 'a2_directional_fourier_component'
        ds.a2.attrs['long_name'] = 'a2 following Kuik et al. (1988) and Herbers et al. (2012)'
        ds.b1.attrs['units'] = 'dimensionless'
        ds.b1.attrs['standard_name'] = 'b1_directional_fourier_component'
        ds.b1.attrs['long_name'] = 'b1 following Kuik et al. (1988) and Herbers et al. (2012)'
        ds.b2.attrs['units'] = 'dimensionless'
        ds.b2.attrs['standard_name'] = 'b2_directional_fourier_component'
        ds.b2.attrs['long_name'] = 'b2 following Kuik et al. (1988) and Herbers et al. (2012)'
        ds.dspr.attrs['units'] = 'angular_degree'
        ds.dspr.attrs['standard_name'] = 'sea_surface_wind_wave_directional_spread'
        ds.dspr.attrs['long_name'] = 'directional spread as a function of frequency'
        ds.freq.attrs['standard_name'] = 'sea_surface_wave_frequency'
        ds.freq.attrs['long_name'] = 'spectral frequencies in Hz'
        ds.freq.attrs['units'] = 'Hz'
        ds.lat.attrs['standard_name'] = 'latitude'
        ds.lat.attrs['long_name'] = 'Approximate latitude of small-scale array rock summit'
        ds.lat.attrs['units'] = 'degrees_north'
        ds.lat.attrs['valid_min'] = -90.0
        ds.lat.attrs['valid_max'] = 90.0
        ds.lon.attrs['standard_name'] = 'longitude'
        ds.lon.attrs['long_name'] = 'Approximate longitude of small-scale array rock summit'
        ds.lon.attrs['units'] = 'degrees_east'
        ds.lon.attrs['valid_min'] = -180.0
        ds.lon.attrs['valid_max'] = 180.0
        ds.time.encoding['units'] = time_units
        ds.time.attrs['units'] = time_units
        ds.time.attrs['standard_name'] = 'time'
        ds.time.attrs['long_name'] = 'Local time (PDT) of spectral segment start'
        # Sig. wave height and other integrated parameters
        ds.vel_binh.attrs['units'] = 'm'
        ds.vel_binh.attrs['standard_name'] = 'height'
        ds.vel_binh.attrs['long_name'] = 'horizontal velocity bin center height above seabed'
        ds.Hm0.attrs['units'] = 'm'
        ds.Hm0.attrs['standard_name'] = 'sea_surface_wave_significant_height'
        ds.Hm0.attrs['long_name'] = 'Hs estimate from 0th spectral moment'
        ds.Te.attrs['units'] = 's'
        ds.Te.attrs['standard_name'] = 'sea_surface_wave_energy_period'
        ds.Te.attrs['long_name'] = 'energy-weighted wave period'
        ds.Tp_ind.attrs['units'] = 's'
        ds.Tp_ind.attrs['standard_name'] = 'sea_surface_primary_swell_wave_period_at_variance_spectral_density_maximum'
        ds.Tp_ind.attrs['long_name'] = 'wave period at maximum spectral energy'
        ds.Tp_Y95.attrs['units'] = 's'
        ds.Tp_Y95.attrs['standard_name'] = 'sea_surface_primary_swell_wave_period_at_variance_spectral_density_maximum'
        ds.Tp_Y95.attrs['long_name'] = 'peak wave period following Young (1995, Ocean Eng.)'
        ds.Dp_ind.attrs['units'] = 'angular_degree' 
        ds.Dp_ind.attrs['standard_name'] = 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum'
        ds.Dp_ind.attrs['long_name'] = 'peak wave direction at maximum energy frequency'
        ds.Dp_Y95.attrs['units'] = 'angular_degree' 
        ds.Dp_Y95.attrs['standard_name'] = 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum'
        ds.Dp_Y95.attrs['long_name'] = 'peak wave direction at Tp_Y95 frequency'
        ds.nu_LH57.attrs['units'] = 'dimensionless' 
        ds.nu_LH57.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density_bandwidth'
        ds.nu_LH57.attrs['long_name'] = 'spectral bandwidth following Longuet-Higgins (1957)'
        # Fill values
        ds.vel_binh.attrs['missing_value'] = fillvalue
        ds.Hm0.attrs['missing_value'] = fillvalue
        ds.Te.attrs['missing_value'] = fillvalue
        ds.Tp_ind.attrs['missing_value'] = fillvalue
        ds.Tp_Y95.attrs['missing_value'] = fillvalue
        ds.Dp_Y95.attrs['missing_value'] = fillvalue
        ds.Dp_ind.attrs['missing_value'] = fillvalue
        ds.nu_LH57.attrs['missing_value'] = fillvalue

       # Global attributes
        ds.attrs['title'] = ('ROXSI 2022 Asilomar Small-Scale Array ' + 
                             'Signature1000 {} wave spectra'.format(self.mid))
        if z_var == 'ASTd':
            ds.attrs['summary'] =  ('Nearshore wave spectra from ADCP measurements. '+
                                    'Sea-surface elevation is the despiked ADCP ' + 
                                    'acoustic surface track (AST) signal, and the ' + 
                                    'horizontal velocities are despiked E&N velocities ' +
                                    'from the range bin specified by the variable ' +
                                    'vel_binh.')
        elif z_var == 'eta_lin':
            ds.attrs['summary'] =  ('Nearshore wave spectra from ADCP measurements. '+
                                    'Sea-surface elevation is the linear sea-surface ' + 
                                    'reconstruction from pressure, and the ' + 
                                    'horizontal velocities are despiked E&N velocities ' +
                                    'from the range bin specified by the variable ' +
                                    'vel_binh.')
        ds.attrs['instrument'] = 'Nortek Signature 1000'
        ds.attrs['mooring_ID'] = self.mid
        ds.attrs['serial_number'] = self.ser
        ds.attrs['segment_length'] = '1200 seconds'
        # Read more attributes from mooring info file if provided
        if self.dfm is not None:
            comments = self.dfm[self.dfm['mooring_ID_long']==self.midl]['notes'].item()
            ds.attrs['comment'] = comments
            config = self.dfm[self.dfm['mooring_ID_long']==self.midl]['config'].item()
            ds.attrs['configurations'] = config
        ds.attrs['Conventions'] = 'CF-1.8'
        ds.attrs['featureType'] = "timeSeries"
        ds.attrs['source'] =  "Sub-surface observation"
        ds.attrs['date_created'] = str(DT.utcnow()) + ' UTC'
        ds.attrs['references'] = 'https://github.com/mikapm/pyROXSI'
        ds.attrs['creator_name'] = "Mika P. Malila"
        ds.attrs['creator_email'] = "mikapm@unc.edu"
        ds.attrs['institution'] = "University of North Carolina at Chapel Hill"

        # Set encoding before saving
        encoding = {'time': {'zlib': False, '_FillValue': None},
                    'freq': {'zlib': False, '_FillValue': None},
                    'lat': {'zlib': False, '_FillValue': None},
                    'lon': {'zlib': False, '_FillValue': None},
                    'Euu': {'_FillValue': fillvalue},        
                    'Evv': {'_FillValue': fillvalue},        
                    'Ezz': {'_FillValue': fillvalue},        
                    'a1': {'_FillValue': fillvalue},        
                    'a2': {'_FillValue': fillvalue},        
                    'b1': {'_FillValue': fillvalue},        
                    'b2': {'_FillValue': fillvalue},        
                    'dspr': {'_FillValue': fillvalue},        
                    'vel_binh': {'_FillValue': fillvalue},        
                    'Hm0': {'_FillValue': fillvalue},        
                    'Te': {'_FillValue': fillvalue},
                    'Tp_ind': {'_FillValue': fillvalue},
                    'Tp_Y95': {'_FillValue': fillvalue},
                    'Dp_ind': {'_FillValue': fillvalue},
                    'Dp_Y95': {'_FillValue': fillvalue},
                    'nu_LH57': {'_FillValue': fillvalue},
                   }     

        # Save dataset in netcdf format
        print('Saving netcdf ...')
        ds.to_netcdf(fn, encoding=encoding)




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
                default=r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray/Signatures',
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
            ds - xr.Dataset or pd.DataFrame with pressure + heading, 
                 pitch & roll data
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
        if isinstance(ds, pd.DataFrame):
            datestr = str(pd.Timestamp(ds.index[0]).date())
        elif isinstance(ds, xr.Dataset):
            datestr = str(pd.Timestamp(ds.time[0].values).date())
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

#     # UVZ spectra test
#     df = pd.read_csv('test_uvz.csv', parse_dates=['time']).set_index('time')
#     dss = rpws.spec_uvz(z=df['z'].values, u=df['u'].values, 
#         v=df['v'].values, fs=4, fmin=0.05, fmax=0.33, 
#         timestamp=pd.Timestamp(df.index[0].round('1H')))

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

    # Plot all HPR timeseries first
    for ser in sers:
        print('Serial number (HPR): ', ser)
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
        # Loop over raw .mat files and get HPR
        for i,fn_mat in tqdm(enumerate(adcp.fns)):
            # Define figure filename
            figdir_hpr = os.path.join(figdir, 'hpr')
            if not os.path.isdir(figdir_hpr):
                # Make figure dir.
                os.mkdir(figdir_hpr)
            fn_hpr = os.path.join(figdir_hpr, 'qc_hpr_{}_{:03d}.pdf'.format(
                ser, i))
            # Check if figure already exists
            if os.path.isfile(fn_hpr):
                continue
            # Read heading, pitch & roll time series 
            df_hpr = adcp.loaddata_vel(fn_mat, only_hpr=True)
            # Plot
            if not os.path.isfile(fn_hpr):
                plot_hpr(df_hpr, fn=fn_hpr, ser=ser)

    # Combine all individual HPR figures into one pdf
    fn_all_hpr = os.path.join(figdir_hpr, 'all_sig_hpr_{}.pdf'.format(ser))
    fns_pdf_hpr = sorted(glob.glob(os.path.join(figdir_hpr, 'qc_hpr*.pdf')))
    if len(fns_pdf_hpr):
        # Call the PdfFileMerger
        mergedObject = PdfFileMerger()
        # Loop through all pdf files and append their pages
        for fn in fns_pdf_hpr:
            mergedObject.append(PdfFileReader(fn, 'rb'))
        # Write all the files into a file which is named as shown below
        if not os.path.isfile(fn_all_hpr):
            print('Merging HPR pdfs into one ...')
            mergedObject.write(fn_all_hpr)

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
            # Check if date1 is before dataset starttime
            if pd.Timestamp(times[-1]) < pd.Timestamp(adcp.t0):
                print('.mat file endtime {} before dataset starttime {}'.format(
                    pd.Timestamp(times[-1]), pd.Timestamp(adcp.t0)))
                continue
            print('mat: ', os.path.basename(fn_mat))
            date0_str = ''.join(date0.split('-'))
            date1_str = ''.join(date1.split('-'))
            fn_nc0 = os.path.join(outdir, 
                'Asilomar_SSA_L1_Sig_Vel_{}_{}.nc'.format(
                    adcp.mid, date0_str))
            fn_nc1 = os.path.join(outdir, 
                'Asilomar_SSA_L1_Sig_Vel_{}_{}.nc'.format(
                    adcp.mid, date1_str))
            if not os.path.isfile(fn_nc0) or date0 != date1:
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

                    # Check if daily ds starttime is before dataset starttime
                    if pd.Timestamp(dsd.time[0].values) < pd.Timestamp(adcp.t0):
                        print('Cropping daily dataset to start from {}'.format(
                            pd.Timestamp(adcp.t0)))
                        dsd = dsd.sel(time=slice(adcp.t0, None))

                    if not os.path.isfile(fn_nc0):
                        print('Saving daily dataset for {} to netCDF ...'.format(
                            date0))
                        dsd.to_netcdf(fn_nc0)
            
                    # Estimate 20-min. wave spectra from AST
                    fn_spec_ast = os.path.join(outdir, 
                                           'Asilomar_SSA_L2_Sig_Spec_ASTd_{}_{}.nc'.format(
                                                adcp.mid, date0_str))
                    if not os.path.isfile(fn_spec_ast):
                        print('Estimating daily spectra from AST ...')
                        dss_daily = adcp.wavespec(dsd, seglen=1200, z='ASTd')
                        # Save spectra to netCDF
                        adcp.save_spec_nc(dss_daily, fn=fn_spec_ast, z_var='ASTd')
                    
                    # Also estimate 20-min. wave spectra from linear pressure reconstruction
                    fn_spec_etal = os.path.join(outdir, 
                                           'Asilomar_SSA_L2_Sig_Spec_ETAl_{}_{}.nc'.format(
                                                adcp.mid, date0_str))
                    if not os.path.isfile(fn_spec_etal):
                        print('Estimating daily spectra from eta_lin ...')
                        dss_daily = adcp.wavespec(dsd, seglen=1200, z='eta_lin')
                        # Save spectra to netCDF
                        adcp.save_spec_nc(dss_daily, fn=fn_spec_etal, z_var='eta_lin')

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

                    # Estimate 20-min. wave spectra from AST
                    fn_spec_ast = os.path.join(outdir, 
                                           'Asilomar_SSA_L2_Sig_Spec_ASTd_{}_{}.nc'.format(
                                                adcp.mid, date0_str))
                    if not os.path.isfile(fn_spec_ast):
                        print('Estimating daily spectra from AST ...')
                        dss_daily = adcp.wavespec(dsd, seglen=1200, z='ASTd')
                        # Save spectra to netCDF
                        adcp.save_spec_nc(dss_daily, fn=fn_spec_ast, z_var='ASTd')
                    
                    # Also estimate 20-min. wave spectra from linear pressure reconstruction
                    fn_spec_etal = os.path.join(outdir, 
                                           'Asilomar_SSA_L2_Sig_Spec_ETAl_{}_{}.nc'.format(
                                                adcp.mid, date0_str))
                    if not os.path.isfile(fn_spec_etal):
                        print('Estimating daily spectra from eta_lin ...')
                        dss_daily = adcp.wavespec(dsd, seglen=1200, z='eta_lin')
                        # Save spectra to netCDF
                        adcp.save_spec_nc(dss_daily, fn=fn_spec_etal, z_var='eta_lin')
            else:
                # Read existing file(s)
                dsd = xr.decode_cf(xr.open_dataset(fn_nc0, decode_coords='all'))
                # Estimate 20-min. wave spectra from AST
                fn_spec_ast = os.path.join(outdir, 
                                        'Asilomar_SSA_L2_Sig_Spec_ASTd_{}_{}.nc'.format(
                                            adcp.mid, date0_str))
                if not os.path.isfile(fn_spec_ast):
                    print('Estimating daily spectra from AST ...')
                    dss_daily = adcp.wavespec(dsd, seglen=1200, z='ASTd')
                    # Save spectra to netCDF
                    adcp.save_spec_nc(dss_daily, fn=fn_spec_ast, z_var='ASTd')
                
                # Also estimate 20-min. wave spectra from linear pressure reconstruction
                fn_spec_etal = os.path.join(outdir, 
                                        'Asilomar_SSA_L2_Sig_Spec_ETAl_{}_{}.nc'.format(
                                            adcp.mid, date0_str))
                if not os.path.isfile(fn_spec_etal):
                    print('Estimating daily spectra from eta_lin ...')
                    dss_daily = adcp.wavespec(dsd, seglen=1200, z='eta_lin')
                    # Save spectra to netCDF
                    adcp.save_spec_nc(dss_daily, fn=fn_spec_etal, z_var='eta_lin')





