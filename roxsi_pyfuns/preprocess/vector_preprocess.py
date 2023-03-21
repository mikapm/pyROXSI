"""
* Pre-process Nortek Vector ADV raw data. 
* Remove bad measurements based on correlations and despiking. 
* Convert pressure to sea-surface elevation using linear TRF.
* Save Level1 products as netcdf.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime as DT
from cftime import date2num, num2date
from roxsi_pyfuns import despike as rpd
from roxsi_pyfuns import coordinate_transforms as rpct
from roxsi_pyfuns import transfer_functions as rptf
from roxsi_pyfuns import wave_spectra as rpws

class ADV():
    """
    Main ADV class
    """
    def __init__(self, datadir, mooring_id, zp=0.08, fs=16, burstlen=1200, 
                 magdec=12.86, outdir=None, mooring_info=None, patm=None, 
                 bathy=None, instr='Nortek Vector'):
        """
        Initialize Vector ADV class.

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
            patm - pd.DataFrame time series of atmospheric pressure (optional)
            bathy - xr.Dataset of SSA bathymetry (optional)
            instr - str; instrument name
        """
        self.datadir = datadir
        self.midl = mooring_id
        self.zp = zp
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
        # Read mooring info excel file if provided
        if mooring_info is not None:
            self.dfm = pd.read_excel(mooring_info, 
                                     parse_dates=['record_start', 'record_end'])
            # Get mooring ID
            key = 'mooring_ID'
            self.mid = self.dfm[self.dfm['mooring_ID_long'].astype(str)==self.midl][key].item()
            key = 'serial_number'
            self.ser = self.dfm[self.dfm['mooring_ID_long'].astype(str)==self.midl][key].item()
            print('self.ser: ', self.ser)
        else:
            self.dfm = None
        # Get raw data filenames
        self._fns_from_mid()
        # Read config info from .hdr file
        self._read_hdr()
        # Atmospheric pressure time series, if provided
        self.patm = patm
        self.instr = instr
        # Bathymetry dataset if provided
        self.bathy = bathy
        # Get lat, lon coordinates of mooring
        if self.bathy is not None:
            self.lon = self.bathy['{}_llc'.format(self.mid[:2])].sel(llc='longitude').item()
            self.lat = self.bathy['{}_llc'.format(self.mid[:2])].sel(llc='latitude').item()


    def _fns_from_mid(self):
        """
        Get Vector data filenames from mooring ID and
        data directory.
        """        
        # Find correct .dat file with data time series
        self.fn_dat = os.path.join(self.datadir, '{}.dat'.format(self.midl))
        # Find correct .sen file with burst info
        self.fn_sen = os.path.join(self.datadir, '{}.sen'.format(self.midl))
        # Find correct .sen file with Vector configuration info
        self.fn_hdr = os.path.join(self.datadir, '{}.hdr'.format(self.midl))
        # Define standard netcdf output filename
        self.fn_nc = os.path.join(self.outdir, 'Asilomar_SSA_L1_Vec_{}.nc'.format(
            self.ser))


    def _read_hdr(self):
        """
        Read Nortek .hdr file for configuration info.
        Returns self.hdr dataframe.
        Example usage to get number of measurements in dataset:
            self.hdr['value'].loc[self.hdr['field'] == 'Number_of_measurements']
        """
        # Read fixed-width formatted .hdr file to pandas dataframe
        if os.path.isfile(self.fn_hdr):
            self.hdr =pd.read_fwf(self.fn_hdr, colspecs=[(0,38), (38,None)], 
                                skiprows=2,  header=None, 
                                names=['field', 'value'])
            # Convert field column names to lower case and combine words
            # with underscores if possible
            for i in range(len(self.hdr)):
                try:
                    self.hdr['field'].iloc[i] = self.hdr['field'].iloc[i].lower().replace(
                        ' ', '_')
                except:
                    pass
            # Check that sampling rate is consistent with user-provided value
            fs_hdr = int(self.hdr['value'][self.hdr['field']=='sampling_rate'].item().split(
                ' ')[0]) # Sampling rate from .hdr file
            assert self.fs == fs_hdr, \
                "Given sampling rate fs={} is not consistent with value in .hdr file".format(
                    self.fs)
            # Also check that burst length is consistent with .hdr file
            bl_hdr = int(self.hdr['value'][self.hdr['field']=='samples_per_burst'].item().split(
                ' ')[0]) # Sampling rate from .hdr file
            bl_in = int(self.fs * self.burstlen) # Input burst length (no. of samples)
            assert bl_in == bl_hdr, \
                "Given sampling rate fs={} is not consistent with value in .hdr file".format(
                    self.fs)
        else:
            self.hdr = None
        
        
    def loaddata(self, despike_corr=True, despike_GN02=True,
                 interp='linear', rec_start=None, rec_end=None,
                 p2eta=True, savenc=True, overwrite=False, date_spec=None):
        """
        Read raw data from chosen mooring ID into pandas
        dataframe. Header info can be found in .hdr files.

        Raw data gets saved into netcdf files if desired, with 1-Hz 
        data from .sen files (e.g. heading, pitch & roll) linearly 
        interpolated to the sampling rate of the velocity data.

        Parameters:
            datestr - str (yyyymmdd); if not None, read only requested
                      date of data. Else, read entire dataset from raw data.
            despike_corr - bool; if True, use correlations to get rid of bad
                           velocity data.
            despike_GN02 - bool; if True, use Goring & Nikora (2002) phase space
                           method to despike velocities (burst-wise).
            interp - str; interpolation method for despiking algorithms.
            rec_start - pd.Timestamp; optional record start time, crops record
                        prior to given timestamp.
            rec_end - pd.Timestamp; optional record end time, crops record
                      after given timestamp.
            p2eta - bool; if True, transforms pressure measurements to sea-
                    surface elevation using linear wave theory.
            savenc - bool; if True, saves output dataset as netcdf.
            overwrite - bool; if True, overwrites any existing netcdf file.
                        Else, reads and returns existing file if available.
            date_spec - str (yyyymmdd); Give specific date to read/produce
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
        # Make time array
        days = pd.date_range(t0, t1, freq='1D')
        # Empty list to store daily dataframes for merging\
        dfd_list = []
        # Iterate over days (i is date index starting at 0)
        dp_cnt = 0 # daily data point counter
        for i, date in tqdm(enumerate(days)):
            # Daily netcdf filename
            datestr = DT.strftime(date, '%Y%m%d')
            if date_spec is not None:
                if datestr != date_spec:
                    # Only read/process specified date
                    continue
            if rec_start is not None:
                if pd.Timestamp(datestr) < rec_start:
                    continue
            fn_nc = os.path.join(self.outdir, 
                'Asilomar_SSA_L1_Vec_{}_{}.nc'.format(self.ser, datestr))
            if not os.path.isfile(fn_nc):
                # Copy daily (d) segment from sen dataframe
                sen_d = sen.loc[DT.strftime(date, '%Y-%m-%d')].copy()
                # Calculate number of bursts in the current day
                Nsen = self.burstlen+1 # Nortek adds +1s to burst
                Nb = int(len(sen_d) / Nsen) 
                # Take out the same number of bursts from data
                Nd = self.burstlen * self.fs # Number of data samples per burst
                dp = int(Nb * Nd) # Number of data points in the current day
                # Take out current date (not necessarily a full day)
                data_d = data.iloc[dp_cnt:(dp_cnt+dp)].copy()
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
                    print('t0b: {}, t1b: {}'.format(t0b, t1b))
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

                    # Rotate despiked velocities to East, North, Up
                    vel_arr = np.array([burst['u_desp'].values, 
                                        burst['v_desp'].values, 
                                        burst['w_desp'].values]).T
                    enu = rpct.uvw2enu(vel=vel_arr, heading=burst_si['heading'].values, 
                        pitch=burst_si['pitch'].values, roll=burst_si['roll'].values, 
                        magdec=self.magdec)

                    # Save variables to dataset
                    burst['uE'] = enu[0,:].copy()
                    burst['uN'] = enu[1,:].copy()
                    burst['uU'] = enu[2,:].copy()

                    # Convert E,N velocities to local cross- & alongshore (x,y) components
                    angle_met = 300 # Cross-shore angle
                    angle_math = 270 - angle_met # Math angle to rotate
                    if angle_math < 0:
                        angle_math += 360
                    angle_math = np.deg2rad(angle_math) # Radians
                    # Rotate East and North velocities to cross-shore (cs) and 
                    # long-shore (ls)
                    ur, vr = rpct.rotate_vel(burst['uE'].values, burst['uN'].values, 
                        angle_math)
                    burst['ucs'] = ur
                    burst['uls'] = vr

                    # If requested, transform pressure to sea-surface elevation
                    if p2eta:
                        # Check if fp seems to make sense
                        spec = rpws.spec_uvz(z=burst['pressure'].values, fs=self.fs)
                        fp = (1 / spec.Tp_ind).item()
                        if fp < 0.5:
                            # Standard linear reconstruction (Tucker and Pitt, 2001)
                            _, burst['z_hyd'] = self.p2eta_lin(burst['pressure'], 
                                return_hyd=True)
                            # Detrend for eta
                            burst['eta_hyd'] = detrend(burst['z_hyd'].values)
                            # burst['eta_lin'] = detrend(burst['z_lin'].values)
                            # K_rms reconstructions (lin. + nonlin.) following 
                            # Martins et al (2021)
                            burst['eta_lin_krms'], _ = self.p2eta_krms(
                                burst['z_hyd'],)
                        else:
                            print('fp: ', fp)
                            # Something is wrong; set all to NaN
                            burst['z_hyd'] = np.ones_like(burst['pressure'].values * np.nan)
                            # burst['eta_hyd'] = np.ones_like(burst['pressure'].values * np.nan)
                            # burst['z_lin'] = np.ones_like(burst['pressure'].values * np.nan)
                            # burst['eta_lin'] = np.ones_like(burst['pressure'].values * np.nan)
                            burst['eta_lin_krms'] = np.ones_like(burst['pressure'].values * np.nan)
                            # burst['eta_nl_krms'] = np.ones_like(burst['pressure'].values * np.nan)

                    # Save burst df to list for merging
                    uvw_dfs.append(burst)

                # Merge burst dataframes to one daily dataframe
                uvw_d = pd.concat(uvw_dfs)
                # Same for interpolated sen (H,P,R,T) data
                hpr_d = pd.concat(hpr_dfs)
                # Combine both dataframes into one daily Vector dataframe
                uvw_d = pd.concat([uvw_d, hpr_d], axis=1)
                # Append daily dataframe to list for merging of whole dataset
                dfd_list.append(uvw_d)

                # Save to netcdf
                print('Converting to netcdf ...')
                datestr_ga = DT.strftime(date, '%Y-%m-%d') # For global attributes
                ds = self.df2nc(uvw_d, overwrite=overwrite, fn=fn_nc, 
                    datestr=datestr_ga)
            else:
                ds = xr.decode_cf(xr.open_dataset(fn_nc, decode_coords='all'))
        # Return xr.Dataset
        return ds


    def loaddata_old_nc(self, datadir, overwrite=False,):
        """
        Load data from old netcdf files and redo pressure reconstructions
        and velocity rotations.

        Parameters:
            datadir - str; path for old netcdf files.
            overwrite - bool; if True, overwrites any existing netcdf file.
                        Else, reads and returns existing file if available.
        """
        # Read old netcdf file
        fn_old = os.path.join(datadir, 'Asilomar_SSA_L1_Vec_{}.nc'.format(
            self.ser))
        dso = xr.decode_cf(xr.open_dataset(fn_old, decode_coords='all'))
        # Save data in daily dataframes
        t_start = pd.Timestamp(dso.time.values[0]).floor('1H')
        t_end = pd.Timestamp(dso.time.values[-1]).ceil('1H')
        # Iterate over dates
        date_range = pd.date_range(t_start.floor('1D'), t_end.ceil('1D'), 
            freq='1D')
        for date in date_range:
            daily_df = [] # Save daily dataframes for concatenating
            # Daily netcdf filename
            datestr = DT.strftime(date, '%Y%m%d')
            fn_nc = os.path.join(self.outdir, 
                'Asilomar_SSA_L1_Vec_{}_{}.nc'.format(self.ser, datestr))
            if os.path.isfile(fn_nc) and not overwrite:
                continue
            # Iterate over bursts
            burst_range = pd.date_range(date, date+pd.Timedelta(days=1),
                freq='1H')
            for t0 in burst_range[:-1]:
                if t0 < t_start:
                    continue
                if t0 >= t_end:
                    continue
                # Take out segment
                t1 = t0 + pd.Timedelta(hours=1)
                print('t0: {}, t1: {}'.format(t0, t1))
                spec_cols = ['ux', 'uy', 'uz', 'uxd', 'uyd', 'uzd', 
                            'pressure', 'heading_ang', 'pitch_ang', 
                            'roll_ang']
                cols_df = ['u', 'v', 'w', 'u_desp', 'v_desp', 'w_desp', 
                           'pressure', 'heading', 'pitch', 'roll', ]
                segs = {} # Store individual variable segments for concat
                # This only seems to work by iterating over each key individually
                for ki, k in enumerate(spec_cols):
                    # Take out current key
                    seg = dso[k].sel(time=slice(t0, t1)).copy()
                    if ki == 0:
                        # Save timestamps
                        segs['time'] = pd.to_datetime(seg.time.values)
                    # Convert to pandas
                    seg = seg.to_series()
                    # Add seg to dict
                    segs[cols_df[ki]] = seg.values
                    if k == 'pressure':
                        # Standard linear reconstruction (Tucker and Pitt, 2001)
                        _, z_hyd = self.p2eta_lin(seg, return_hyd=True)
                        # Detrend for eta
                        eta_hyd = detrend(z_hyd)
                        # K_rms reconstructions (lin. + nonlin.) following 
                        # Martins et al (2021)
                        eta_lin_krms, _ = self.p2eta_krms(z_hyd)
                        # Add reconstructions to dict
                        segs['z_hyd'] = z_hyd
                        segs['eta_hyd'] = eta_hyd
                        segs['eta_lin_krms'] = eta_lin_krms                    
                # Rotate despiked velocities to East, North, Up
                vel_arr = np.array([segs['u_desp'], segs['v_desp'], 
                                    segs['w_desp']]).T
                enu = rpct.uvw2enu(vel=vel_arr, heading=segs['heading'], 
                    pitch=segs['pitch'], roll=segs['roll'], 
                    magdec=self.magdec)

                # Save variables to dataset
                segs['uE'] = enu[0,:].copy()
                segs['uN'] = enu[1,:].copy()
                segs['uU'] = enu[2,:].copy()

                # Convert E,N velocities to local cross- & alongshore (x,y) components
                angle_met = 300 # Cross-shore angle
                angle_math = 270 - angle_met # Math angle to rotate
                if angle_math < 0:
                    angle_math += 360
                angle_math = np.deg2rad(angle_math) # Radians
                # Rotate East and North velocities to cross-shore (cs) and 
                # long-shore (ls)
                ur, vr = rpct.rotate_vel(segs['uE'], segs['uN'], angle_math)
                segs['ucs'] = ur
                segs['uls'] = vr

                # Combine all segments to dataframe
                df_seg = pd.DataFrame.from_dict(segs).set_index('time')
                # Append to daily list
                daily_df.append(df_seg)
            # Save daily netcdf
            uvw_d = pd.concat(daily_df)
            print('Converting to netcdf ...')
            datestr_ga = DT.strftime(date, '%Y-%m-%d') # For global attributes
            ds = self.df2nc(uvw_d, overwrite=overwrite, fn=fn_nc, 
                datestr=datestr_ga)
    

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


    def despike_GN02(self, df, interp='linear', sec_lim=2, corrd=True, 
                     min_new_spikes=10, max_iter=3):
        """
        Despike Nortek Vector velocities using low correlation values
        to discard unreliable measurements following the Goring and
        Nikora (2002, J. Hydraul. Eng.) phase space method, including the
        modifications by Wahl (2003, J. Hydraul. Eng.) and Mori et al.
        (2007, J. Eng. Mech.).

        Parameters:
            df - pd.DataFrame; dataframe with velocity (u,v,w) time series.
            interp - str; interpolation method for discarded data.
                          Options: see pandas.DataFrame.interpolate()
            sec_lim - scalar; maximum gap size to interpolate (sec)
            corrd - bool; if True, input velocities are correlation-
                    corrected by self.despike_correlations()
            min_new_spikes - int; iterate until number of new spikes detected 
                             is lower than this value. 
            max_iter - int; maximum number of despiking iterations
        
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
            vel = vels[kv] # Current velocity component
            # Initialize counter of new spikes detected
            n_spikes = min_new_spikes + 1
            # Initialize iteraction counter
            cnt = 0
            # Iterate until sufficiently low number of new spikes detected
            while n_spikes > min_new_spikes or cnt < max_iter:
                # Detect spikes using 3D phase space method
                mask = rpd.phase_space_3d(vel.values)
                # Convert detected spikes to NaN
                vel[mask] = np.nan
                # Interpolate according to requested method
                vel.interpolate(method=interp, limit=sec_lim*self.fs, 
                                inplace=True)
                if np.sum(np.isnan(vel.values)) > 0:
                    # Also interpolate potential leading/trailing NaNs
                    vel = vel.interpolate(method='bfill').interpolate(method='ffill')
                # Count number of spikes detected
                n_spikes = mask.sum()
                # Add to iteration counter
                cnt += 1
                if n_spikes < min_new_spikes or cnt >= max_iter:
                    break
            # Add corrected velocity column to input df
            df['{}_desp'.format(k)] = vel
    

    def p2eta_lin(self, pt, rho0=1025, grav=9.81, M=512, fmin=0.05, fmax=0.35, 
                  att_corr=True, return_hyd=True):
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
            return_hyd - bool; if True, returns also hydrostatic pressure head
        """
        # Copy input
        pw = pt.copy()
        pw = pw.to_frame(name='pressure') # Convert to dataframe
        # Use hydrostatic assumption to get pressure head with unit [m]
        pw['z_hyd'] = rptf.z_hydrostatic(pw['pressure'], self.patm, 
            rho0=rho0, grav=grav, interp=True)
        # Check if hydrostatic pressure is ever above 0
        if pw['z_hyd'].max() == 0.0:
            print('Instrument most likely not in water')
            # Return NaN array for linear sea surface elevations
            pw['z_lin'] = np.ones_like(pw['z_hyd'].values) * np.nan
        else:
            # Apply linear transfer function from p->eta
            trf = rptf.TRF(fs=self.fs, zp=self.zp,)
            pw['z_lin'] = trf.p2z_lin(pw['z_hyd'], M=M, fmin=fmin, fmax=fmax,
                att_corr=att_corr)

        if return_hyd:
            # Return also hydrostatic pressure head
            return pw['z_lin'], pw['z_hyd']
        else:
            return pw['z_lin']

    def p2eta_krms(self, pt, fmax=0.35, fp=None, krms=None, f_krms=None,
                   fix_ends=True, tail_method='constant'):
        """
        Use linear transfer function to reconstruct sea-surface
        elevation time series from sub-surface pressure measurements.

        If self.patm dataframe of atmospheric pressure is not available,
        this function assumes that the input time series is the hydrostatic 
        pressure.

        Parameters:
            pt - pd.Series; time series of hydrostatic pressure head [m]
            fmax - scalar; max. cutoff frequency
            fp - scalar; peak frequency
            krms - array; root-mean-square wavenumbers
            f_krms - array; frequencies corresponding to krms
            fix_ends - bool; if True, set first/last waves equal to 
                       hydrostatic surface
            tail_method - str; tail method for K_rms reconstruction.
                          Choices: ['constant', 'hydrostatic']
        """
        # Copy input
        pw = pt.copy()
        pw = pw.to_frame(name='z_hyd') # Convert to dataframe
        # Compute depth
        depth = pw['z_hyd'].mean().item()
        # Detrend
        eta_hyd = detrend(pw['z_hyd'].values)

        # Initialize TRF class
        trf = rptf.TRF(fs=self.fs, zp=self.zp, )
        # Is fp given?
        if fp is None:
            # Compute spectrum and get fp
            spec = rpws.spec_uvz(z=eta_hyd, fs=self.fs)
            fp = (1 / spec.Tp_ind).item()
        # Is K_rms given, or do we need to compute it?
        if krms is None:
            # Compute bispectrum and K_rms
            dsb = rpws.bispectrum(eta_hyd[::4], fs=self.fs, h0=depth, return_krms=True)
            krms = dsb.k_rms.values
            f_krms = dsb.freq1.values
        
        # Reconstruct surface
        eta_lin_krms, eta_nl_krms = trf.p2eta_krms(
            eta_hyd, h0=depth, fc=fmax, fcmax_allowed=fmax, f_krms=f_krms, 
            krms=krms, return_nl=True, fmax=fmax, fp=fp, fix_ends=fix_ends,
            tail_method=tail_method)
        # Save to dataframe
        pw['eta_lin_krms'] = eta_lin_krms
        pw['eta_nl_krms'] = eta_nl_krms

        return pw['eta_lin_krms'], pw['eta_nl_krms']
    

    def df2nc(self, df, ref_date=pd.Timestamp('2022-06-25'), 
              overwrite=False, crop=False, fillvalue=-9999., fn=None,
              datestr=None):
        """
        Convert and save pd.DataFrame of Vector data to netcdf 
        format following CF conventions.

        Parameters:
            df - input pd.DataFrame to convert. Should be produced by 
                 self.loaddata()
            ref_date - reference date to use for time axis
            overwrite - bool; if True, overwrites pre-existing netcdf file
            crop - bool; if True, crops time series using record_start and
                   record_end fields from mooring info excel file (self.dfm)
            fillvalue - scalar; fill value to denote missing values
            fn - str; Path to filename. If None, uses self.fn_nc
            datestr - str; Date+time string for global attributes
        """
        if fn is None:
            fn = self.fn_nc
        # Check if file already exists
        if os.path.isfile(fn) and not overwrite:
            # Read and return existing dataset
            print('Requested netCDF file already exists. ' + 
                  'Set overwrite=True to overwrite.')
            ds = xr.open_dataset(fn)
            return ds
        
        # Crop input dataframe if requested and sel.dfm exists
        if crop and self.dfm is not None:
            print('Cropping time series ...')
            t0n = pd.Timestamp(
                self.dfm[self.dfm['mooring_ID_long']==self.midl]['record_start'].item())
            t1n = pd.Timestamp(
                self.dfm[self.dfm['mooring_ID_long']==self.midl]['record_end'].item())
            # Crop dataframe
            df = df.loc[t0n:t1n]

        # Set requested fill value
        df = df.fillna(fillvalue)
        
        # Convert time array to numerical format
        time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(ref_date)
        time_vals = date2num(df.index.to_pydatetime(), 
                             time_units, calendar='standard', 
                             has_year_zero=True)
        # To convert back to datetime64:
        # pd.to_datetime(num2date(time_vals, time_units, 
        #                         only_use_python_datetimes=True, 
        #                         only_use_cftime_datetimes=False)
        #                         )
        # Convert arrays into Xarray Dataset
        ds = xr.Dataset(
            data_vars={'ux': (['time'], df['u']),
                       'uy': (['time'], df['v']),
                       'uz': (['time'], df['w']),
                       'uxd': (['time'], df['u_desp']),
                       'uyd': (['time'], df['v_desp']),
                       'uzd': (['time'], df['w_desp']),
                       'uE': (['time'], df['uE']),
                       'uN': (['time'], df['uN']),
                       'uU': (['time'], df['uU']),
                       'ucs': (['time'], df['ucs']),
                       'uls': (['time'], df['uls']),
                       'pressure':  (['time'], df['pressure']),
                       'z_hyd':  (['time'], df['z_hyd']),
                       # 'z_lin':  (['time'], df['z_lin']),
                       'eta_hyd':  (['time'], df['eta_hyd']),
                       # 'eta_lin':  (['time'], df['eta_lin']),
                       'eta_lin_krms':  (['time'], df['eta_lin_krms']),
                       # 'eta_nl_krms':  (['time'], df['eta_nl_krms']),
                       'heading_ang':  (['time'], df['heading']),
                       'pitch_ang':  (['time'], df['pitch']),
                       'roll_ang':  (['time'], df['roll']),
                       'lat': (['lat'], np.atleast_1d(self.lat)),
                       'lon': (['lon'], np.atleast_1d(self.lon)),
                       },
            coords={'time': (['time'], time_vals),}
            )
        # Set units
        ds.ux.attrs['units'] = 'm/s'
        ds.uy.attrs['units'] = 'm/s'
        ds.uz.attrs['units'] = 'm/s'
        ds.uxd.attrs['units'] = 'm/s'
        ds.uyd.attrs['units'] = 'm/s'
        ds.uzd.attrs['units'] = 'm/s'
        ds.uE.attrs['units'] = 'm/s'
        ds.uN.attrs['units'] = 'm/s'
        ds.uU.attrs['units'] = 'm/s'
        ds.ucs.attrs['units'] = 'm/s'
        ds.uls.attrs['units'] = 'm/s'
        ds.heading_ang.attrs['units'] = 'degrees'
        ds.pitch_ang.attrs['units'] = 'degrees'
        ds.roll_ang.attrs['units'] = 'degrees'
        ds.pressure.attrs['units'] = 'hPa'
        ds.z_hyd.attrs['units'] = 'm'
        # ds.z_lin.attrs['units'] = 'm'
        ds.eta_hyd.attrs['units'] = 'm'
        # ds.eta_lin.attrs['units'] = 'm'
        ds.eta_lin_krms.attrs['units'] = 'm'
        # ds.eta_nl_krms.attrs['units'] = 'm'
        ds.time.encoding['units'] = time_units
        ds.time.attrs['units'] = time_units
        ds.time.attrs['standard_name'] = 'time'
        ds.time.attrs['long_name'] = 'Local time (PDT), midpoints of sampling intervals'
        ds.lat.attrs['standard_name'] = 'latitude'
        ds.lat.attrs['long_name'] = 'Approximate latitude of instrument'
        ds.lat.attrs['units'] = 'degrees_north'
        ds.lat.attrs['valid_min'] = -90.0
        ds.lat.attrs['valid_max'] = 90.0
        ds.lon.attrs['standard_name'] = 'longitude'
        ds.lon.attrs['long_name'] = 'Approximate longitude of instrument'
        ds.lon.attrs['units'] = 'degrees_east'
        ds.lon.attrs['valid_min'] = -180.0
        ds.lon.attrs['valid_max'] = 180.0

        # Variable attributes
        ds.ux.attrs['standard_name'] = 'sea_water_x_velocity'
        ds.uy.attrs['standard_name'] = 'sea_water_y_velocity'
        ds.uz.attrs['standard_name'] = 'upward_sea_water_velocity'
        ds.uxd.attrs['standard_name'] = 'sea_water_x_velocity'
        ds.uyd.attrs['standard_name'] = 'sea_water_y_velocity'
        ds.uzd.attrs['standard_name'] = 'upward_sea_water_velocity'
        ds.uE.attrs['standard_name'] = 'eastward_sea_water_velocity'
        ds.uN.attrs['standard_name'] = 'northward_sea_water_velocity'
        ds.uU.attrs['standard_name'] = 'upward_sea_water_velocity'
        ds.ucs.attrs['standard_name'] = 'cross_shore_sea_water_velocity'
        ds.uls.attrs['standard_name'] = 'long_shore_sea_water_velocity'
        ds.heading_ang.attrs['standard_name'] = 'platform_orientation'
        ds.pitch_ang.attrs['standard_name'] = 'platform_pitch_angle'
        ds.roll_ang.attrs['standard_name'] = 'platform_roll_angle'
        ds.pressure.attrs['standard_name'] = 'sea_water_pressure_due_to_sea_water'
        ds.z_hyd.attrs['standard_name'] = 'depth'
        # ds.z_lin.attrs['standard_name'] = 'depth'
        ds.eta_hyd.attrs['standard_name'] = 'sea_surface_height_above_mean_sea_level'
        # ds.eta_lin.attrs['standard_name'] = 'sea_surface_height_above_mean_sea_level'
        ds.eta_lin_krms.attrs['standard_name'] = 'sea_surface_height_above_mean_sea_level'
        # ds.eta_nl_krms.attrs['standard_name'] = 'sea_surface_height_above_mean_sea_level'
        # Long names of velocity components
        ln_ux = 'x component of raw velocity in instrument reference frame'
        ln_uy = 'y component of raw velocity in instrument reference frame'
        ln_uz = 'z component of raw velocity in instrument reference frame'
        ds.ux.attrs['long_name'] = ln_ux
        ds.uy.attrs['long_name'] = ln_uy
        ds.uz.attrs['long_name'] = ln_uz
        ln_uxd = 'x component of despiked velocity in instrument reference frame'
        ln_uyd = 'y component of despiked velocity in instrument reference frame'
        ln_uzd = 'z component of despiked velocity in instrument reference frame'
        ds.uxd.attrs['long_name'] = ln_uxd
        ds.uyd.attrs['long_name'] = ln_uyd
        ds.uzd.attrs['long_name'] = ln_uzd
        ln_uE = 'Eastward despiked velocity in geographic reference frame'
        ln_uN = 'Northward despiked velocity in geographic reference frame'
        ln_uU = 'Upward despiked velocity in geographic reference frame'
        ds.uE.attrs['long_name'] = ln_uE
        ds.uN.attrs['long_name'] = ln_uN
        ds.uU.attrs['long_name'] = ln_uU
        ln_ucs = 'Cross-shore despiked velocity in local coordinate system'
        ln_uls = 'Long-shore despiked velocity in local coordinate system'
        ds.ucs.attrs['long_name'] = ln_uE
        ds.uls.attrs['long_name'] = ln_uN
        ds.ucs.attrs['zero_deg'] = '300 deg'
        ds.uls.attrs['zero_deg'] = '210 deg'
        ln_head = ('Linearly interpolated instrument heading time series in ' + 
                   'instrument reference frame')
        ds.heading_ang.attrs['long_name'] = ln_head
        ln_pitch = ('Linearly interpolated instrument pitch time series in ' + 
                    'instrument reference frame')
        ds.pitch_ang.attrs['long_name'] = ln_pitch
        ln_roll = ('Linearly interpolated instrument roll time series in ' + 
                   'instrument reference frame')
        ds.roll_ang.attrs['long_name'] = ln_roll
        ln_pres = ('Hydrostatic pressure recorded by instrument')
        ds.pressure.attrs['long_name'] = ln_pres
        ln_eh = ('Pressure head converted from hydrostatic pressure')
        ds.z_hyd.attrs['long_name'] = ln_eh
        ln_zl = ('Linear distance from bottom to surface')
        # ds.z_lin.attrs['long_name'] = ln_zl
        ln_el = ('Detrended sea-surface elevation reconstructed from ' + 
                 'hydrostatic pressure using linear transfer function')
        # ds.eta_lin.attrs['long_name'] = ln_el
        # ds.eta_lin.attrs['fmax'] = '{} Hz'.format(0.35)
        ln_elk = ('Detrended sea-surface elevation reconstructed from ' + 
                  'hydrostatic pressure using linear transfer function and' + 
                  'root-mean-square wavenumbers.')
        ds.eta_lin_krms.attrs['long_name'] = ln_elk
        ds.eta_lin_krms.attrs['fmax'] = '{} Hz'.format(0.35)
        ln_enk = ('Detrended sea-surface elevation reconstructed from ' + 
                  'hydrostatic pressure using nonlinear transfer function and' + 
                  'root-mean-square wavenumbers.')
        # ds.eta_nl_krms.attrs['long_name'] = ln_enk
        # ds.eta_nl_krms.attrs['fmax'] = '{} Hz'.format(0.35)
        # Fill values
#         ds.ux.attrs['missing_value'] = fillvalue
#         ds.uy.attrs['missing_value'] = fillvalue
#         ds.uz.attrs['missing_value'] = fillvalue
#         ds.uxd.attrs['missing_value'] = fillvalue
#         ds.uyd.attrs['missing_value'] = fillvalue
#         ds.uzd.attrs['missing_value'] = fillvalue
#         ds.uE.attrs['missing_value'] = fillvalue
#         ds.uN.attrs['missing_value'] = fillvalue
#         ds.uU.attrs['missing_value'] = fillvalue
#         ds.ucs.attrs['missing_value'] = fillvalue
#         ds.uls.attrs['missing_value'] = fillvalue
#         ds.heading_ang.attrs['missing_value'] = fillvalue
#         ds.pitch_ang.attrs['missing_value'] = fillvalue
#         ds.roll_ang.attrs['missing_value'] = fillvalue
#         ds.pressure.attrs['missing_value'] = fillvalue
#         ds.z_hyd.attrs['missing_value'] = fillvalue
#         # ds.z_lin.attrs['missing_value'] = fillvalue
#         ds.eta_hyd.attrs['missing_value'] = fillvalue
#         # ds.eta_lin.attrs['missing_value'] = fillvalue
#         ds.eta_lin_krms.attrs['missing_value'] = fillvalue
#         # ds.eta_nl_krms.attrs['missing_value'] = fillvalue
        
        # Global attributes
        if datestr is not None:
            ds.attrs['title'] = ('ROXSI 2022 Asilomar Small-Scale Array ' + 
                                'Vector {}, date: {}'.format(self.ser, datestr))
        else:
            ds.attrs['title'] = ('ROXSI 2022 Asilomar Small-Scale Array ' + 
                                'Vector {}'.format(self.ser))
        ds.attrs['summary'] =  "Nearshore acoustic doppler velocimeter measurements."
        ds.attrs['instrument'] = 'Nortek Vector ADV'
        ds.attrs['mooring_ID'] = self.mid[:2]
        ds.attrs['burst_length'] = '{} seconds'.format(self.burstlen)
        # Read attributes from .hdr file
        if self.hdr is not None:
            dpt_str = 'deployment_time'
            dpt = self.hdr['value'][self.hdr['field']==dpt_str].item() 
            ds.attrs[dpt_str] = dpt
            nm_str = 'number_of_measurements'
            n_meas = self.hdr['value'][self.hdr['field']==nm_str].item() 
            ds.attrs[nm_str] = n_meas
            cs_str = 'number_of_velocity_checksum_errors'
            n_cse = self.hdr['value'][self.hdr['field']==cs_str].item() 
            ds.attrs[cs_str] = n_cse
            css_str = 'number_of_sensor_checksum_errors'
            n_css = self.hdr['value'][self.hdr['field']==css_str].item() 
            ds.attrs[css_str] = n_css
            ndg_str = 'number_of_data_gaps'
            n_dg = self.hdr['value'][self.hdr['field']==ndg_str].item() 
            ds.attrs[ndg_str] = n_dg
            fs_str = 'sampling_rate'
            fs = self.hdr['value'][self.hdr['field']==fs_str].item() 
            ds.attrs[fs_str] = fs
            nvr_str = 'nominal_velocity_range'
            nvr = self.hdr['value'][self.hdr['field']==nvr_str].item() 
            ds.attrs[nvr_str] = nvr
            bi_str = 'burst_interval'
            bi = self.hdr['value'][self.hdr['field']==bi_str].item() 
            ds.attrs[bi_str] = bi
            sb_str = 'samples_per_burst'
            sb = self.hdr['value'][self.hdr['field']==sb_str].item() 
            ds.attrs[sb_str] = sb
            sv_str = 'sampling_volume'
            sv = self.hdr['value'][self.hdr['field']==sv_str].item() 
            ds.attrs[sv_str] = sv
            ml_str = 'measurement_load'
            ml = self.hdr['value'][self.hdr['field']==ml_str].item() 
            ds.attrs[ml_str] = ml
            tl_str = 'transmit_length'
            tl = self.hdr['value'][self.hdr['field']==tl_str].item() 
            ds.attrs[tl_str] = tl
            rl_str = 'receive_length'
            rl = self.hdr['value'][self.hdr['field']==rl_str].item() 
            ds.attrs[rl_str] = rl
            vs_str = 'velocity_scaling'
            vs = self.hdr['value'][self.hdr['field']==vs_str].item() 
            ds.attrs[vs_str] = vs
            im_str = 'imu_mode'
            im = self.hdr['value'][self.hdr['field']==im_str].item() 
            ds.attrs[im_str] = im
            cs_str = 'coordinate_system'
            cs = self.hdr['value'][self.hdr['field']==cs_str].item() 
            ds.attrs[cs_str] = cs
            ss_str = 'sound_speed'
            ss = self.hdr['value'][self.hdr['field']==ss_str].item() 
            ds.attrs[ss_str] = ss
            sal_str = 'salinity'
            sal = self.hdr['value'][self.hdr['field']==sal_str].item() 
            ds.attrs[sal_str] = sal
            nb_str = 'number_of_beams'
            nb = self.hdr['value'][self.hdr['field']==nb_str].values[0] 
            ds.attrs[nb_str] = nb
            sn_str = 'serial_number'
            sn = self.hdr['value'][self.hdr['field']==sn_str].values[0] 
            ds.attrs[sn_str] = sn
        # Read more attributes from mooring info file if provided
        if self.dfm is not None:
            comments = self.dfm[self.dfm['mooring_ID_long']==self.midl]['notes'].item()
            ds.attrs['comment'] = comments
            config = self.dfm[self.dfm['mooring_ID_long']==self.midl]['config'].item()
            ds.attrs['configurations'] = config
        ds.attrs['magnetic_declination'] = '{} degrees East'.format(self.magdec)
        ds.attrs['despiking'] = ('3D phase-space despiking of velocities ' +
                                 'following Goring and Nikora (2002), ' +
                                 'including modifications by Wahl (2003) ' +
                                 'and Mori et al. (2007).')
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
                    'lat': {'zlib': False, '_FillValue': None},
                    'lon': {'zlib': False, '_FillValue': None},
                    'ux': {'_FillValue': fillvalue},        
                    'uy': {'_FillValue': fillvalue},        
                    'uz': {'_FillValue': fillvalue},        
                    'uxd': {'_FillValue': fillvalue},        
                    'uyd': {'_FillValue': fillvalue},        
                    'uzd': {'_FillValue': fillvalue},        
                    'uE': {'_FillValue': fillvalue},        
                    'uN': {'_FillValue': fillvalue},        
                    'uU': {'_FillValue': fillvalue},  
                    'ucs': {'_FillValue': fillvalue},  
                    'uls': {'_FillValue': fillvalue},  
                    'heading_ang': {'_FillValue': fillvalue},        
                    'pitch_ang': {'_FillValue': fillvalue},        
                    'roll_ang': {'_FillValue': fillvalue},        
                    'pressure': {'_FillValue': fillvalue},
                    'z_hyd': {'_FillValue': fillvalue},
                    'eta_hyd': {'_FillValue': fillvalue},
                    # 'eta_lin': {'_FillValue': fillvalue},
                    'eta_lin_krms': {'_FillValue': fillvalue},
                    # 'eta_nl_krms': {'_FillValue': fillvalue},
                   }     

        # Save dataset in netcdf format
        print('Saving netcdf ...')
        ds.to_netcdf(fn, encoding=encoding)

        return ds

    
    def read_vecnc(self, fn=None, **kwargs):
        """
        Function wrapper to read Vector netcdf file generated by
        self.df2nc() into xarray dataset and convert time index to
        datetime format.
        """
        if fn is None:
            fn = self.fn_nc
        # Read netcdf file to xarray dataset
        ds = xr.decode_cf(xr.open_dataset(fn, decode_coords='all'))
        return ds


# Main script
if __name__ == '__main__':
    """
    Test script using synthetic example data.
    """
    import sys
    import glob
    from scipy.io import loadmat
    from argparse import ArgumentParser

    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data root directory"),
                type=str,
                # default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/Vectors',
                default=r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray/Vectors',
                )
        parser.add_argument("-midl", 
                help=('Mooring ID (long). To loop through all, select "ALL".'),
                type=str,
                choices=['C1v01', 'C2VP02', 'C3VP02', 'C4VP02', 'C5V02', 
                         'C6v01', 'L1v01', 'L2VP02', 'L4VP02', 'L5v01', 'ALL'],
                default='C3VP02',
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

    # Input data directory
    data_in = os.path.join(args.dr, 'converted')

    # Define Level1 output directory
    outdir_base = os.path.join(args.dr, 'Level1')

    # Mooring info excel file path (used when initializing ADV class)
    rootdir = os.path.split(args.dr)[0] # Root ROXSI SSA directory
    fn_minfo = os.path.join(rootdir, 'Asilomar_SSA_2022_mooring_info.xlsx')

    # Read bathymetry netcdf file
    bathydir = os.path.join(rootdir, 'Bathy')
    fn_bathy = os.path.join(bathydir, 'Asilomar_2022_SSA_bathy.nc')
    dsb = xr.decode_cf(xr.open_dataset(fn_bathy, decode_coords='all'))

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

    # Check if processing just one mooring or all
    if args.midl.lower() == 'all':
        # Loop through all mooring IDs
        mids = ['C1v01', 'C2VP02', 'C3VP02', 'C4VP02', 'C5V02', 
                'C6v01', 'L1v01', 'L2VP02', 'L4VP02', 'L5v01']
    else:
        # Only process one mooring
        mids = [args.midl]

    # Iterate over mooring ID(s)
    for midl in tqdm(mids):
        # Skip mooring ID C6v01 for now, suspicious data
        if midl == 'C6v01':
            print('Not processing Mooring ID {} due to suspicious raw data.'.format(
                midl))
            continue
        mid_short = midl[:2] # Short mooring ID
        # Initialize ADV class and read raw data
        rec_start = pd.Timestamp('2022-06-25 00:00:00')
        if mid_short == 'C4':
            rec_start = pd.Timestamp('2022-07-05 00:00:00')
        rec_end = pd.Timestamp('2022-07-22 00:00:00')
        outdir = os.path.join(outdir_base, midl)
        if not os.path.isdir(outdir):
            print('Making outdir {} ...'.format(outdir))
            os.mkdir(outdir)
        figdir = os.path.join(outdir, 'img')
        adv = ADV(datadir=data_in, mooring_id=midl, magdec=args.magdec,
                  mooring_info=fn_minfo, outdir=outdir, patm=dfa, bathy=dsb,
                  )
        print('Reading raw data .dat file "{}" ...'.format(
            os.path.basename(adv.fn_dat)))
        # Read data 
        # vec = adv.loaddata(overwrite=args.overwrite_nc, rec_start=rec_start, 
        #     rec_end=rec_end)
        data_old = os.path.join(args.dr, 'Level1', 'netcdf_old')
        vec = adv.loaddata_old_nc(datadir=data_old, overwrite=args.overwrite_nc,)
        
#        # Rotate despiked velocities to (E,N,U) reference frame
#        vel_cols = ['u_desp', 'v_desp', 'w_desp']
#        # Take out velocity array
#        vel = vec[vel_cols].values
#        # Check if tilt sensor was vertical
#        config = dfm[dfm['mooring_ID']==mid]['config'].item().split(' ')[0]
#        if config.lower() == 'vert':
#            # Vertical tilt sensor - use compass-measured headings
#            print('Vertical tilt sensor')
#            heading_deg = float(
#                dfm[dfm['mooring_ID']=='L1v01']['notes'].item().split(' ')[-1]
#                )
#            print('heading (deg): ', heading_deg)
#            # Make array of constant heading
#            heading = np.ones_like(vec['heading'].values) * heading_deg
#            print('heading: ', heading)
#        else:
#            # Use sensor-read headings
#            heading = vec['heading']
#        # Rotate velocities
#        vel_enu = rpct.uvw2enu(vel, heading=heading,
#                               pitch=vec['pitch'], roll=vec['roll'],
#                               magdec=adv.magdec)
#        # Add rotated (despiked) velocities to dataframe
#        print('vel_enu shape: ', vel_enu.shape)
#        vec['uE'] = vel_enu[0,:]
#        vec['uN'] = vel_enu[1,:]
#        vec['uU'] = vel_enu[2,:]
#
#        # Plot heading, pitch & roll time series
#        if args.savefig:
#            # Define figure filename and check if it exists
#            if args.datestr is None:
#                fn_hpr = os.path.join(figdir, 'hpr_press_{}.pdf'.format(
#                    mid))
#            else:
#                fn_hpr = os.path.join(figdir, 'hpr_press_{}_{}.pdf'.format(
#                    mid, args.datestr))
#
#            if not os.path.isfile(fn_hpr) or args.overwrite_fig:
#                print('Plotting heading, pitch & roll timeseries ...')
#                fig, axes = plt.subplots(figsize=(12,5), nrows=2, sharex=True,
#                                         constrained_layout=True)
#                vec[['heading', 'pitch', 'roll']].plot(ax=axes[0])
#                vec['pressure'].plot(ax=axes[1])
#                axes[0].set_ylabel('Degrees')
#                axes[1].set_ylabel('dB')
#                axes[1].legend()
#                axes[0].set_title('Vector {} heading, pitch & roll + pressure'.format(
#                    mid))
#                # Save figure
#                plt.savefig(fn_hpr, bbox_inches='tight', dpi=300)
#                plt.close()
#
#        # Plot raw vs. QC'd timeseries
#        if args.savefig:
#            # Define figure filename and check if it exists
#            if args.datestr is None:
#                fn_vel = os.path.join(figdir, 'vel_desp_{}.pdf'.format(
#                    mid))
#            else:
#                fn_vel = os.path.join(figdir, 'vel_desp_{}_{}.pdf'.format(
#                    mid, args.datestr))
#
#            if not os.path.isfile(fn_vel) or args.overwrite_fig:
#                fig, axes = plt.subplots(figsize=(12,7), nrows=3, 
#                                        sharex=True, sharey=True, 
#                                        constrained_layout=True)
#                vec[['u', 'u_corr', 'u_desp']].plot(ax=axes[0])
#                vec[['v', 'v_corr', 'v_desp']].plot(ax=axes[1])
#                vec[['w', 'w_corr', 'w_desp']].plot(ax=axes[2])
#                # Save figure
#                plt.savefig(fn_vel, bbox_inches='tight', dpi=300)
#                plt.close()
#
## Test plot
#fig, axes = plt.subplots(figsize=(12,7), nrows=3, 
#                         sharex=True, sharey=True, 
#                         constrained_layout=True)
## vec_d[['u', 'u_corr', 'u_desp', 'uE']].plot(ax=axes[0])
#vec[['u', 'u_corr', 'u_desp']].plot(ax=axes[0])
## vec_d[['v', 'v_corr', 'v_desp', 'uN']].plot(ax=axes[1])
#vec[['v', 'v_corr', 'v_desp']].plot(ax=axes[1])
## vec_d[['w', 'w_corr', 'w_desp', 'uU']].plot(ax=axes[2])
#vec[['w', 'w_corr', 'w_desp']].plot(ax=axes[2])
#
#plt.tight_layout()
#plt.show()
#
#print('Done. \n')
