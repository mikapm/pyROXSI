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
import xarray as xr
from datetime import datetime as DT
from cftime import date2num
from scipy.io import loadmat
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
        # Get lat, lon coordinates of mooring
        self.lon = self.bathy['{}_llc'.format(self.mid)].sel(llc='longitude').item()
        self.lat = self.bathy['{}_llc'.format(self.mid)].sel(llc='latitude').item()

    def _fns_from_ser(self):
        """
        Returns a list of .mat filenames in self.datadir corresponding
        to serial number.
        """
        # List all .mat files with serial number in filename
        self.fns = sorted(glob.glob(os.path.join(self.datadir,
            'roxsi_*_{:06d}_*.mat'.format(int(self.ser)))))


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


    def dfp2dsp(self, dfp, time_vals, time_units, fillvalue):
        """
        Convert pressure dataframe dfp to dataset dsp.
        """
        # Define dataset
        dsp = xr.Dataset(data_vars={'pressure': (['time'], dfp['pressure'].values), 
                                    'z_hyd': (['time'], dfp['z_hyd'].values),
                                    'z_lin': (['time'], dfp['z_lin'].values),
                                    'patm': (['time'], dfp['patm'].values),
                                   },
                         coords={'time': (['time'], time_vals.astype('f8')),},
                        )
        if self.instr == 'RBRDuetDT':
            # Add temperature timeseries
            dsp['temperature'] = (['time'], dfp['temperature'].values)
        # Set requested fill value
        dsp = dsp.fillna(fillvalue)
        # Set coordinate, variable and global attributes
        dsp.time.encoding['units'] = time_units
        dsp.time.attrs['units'] = time_units
        dsp.time.attrs['standard_name'] = 'time'
        dsp.time.attrs['long_name'] = 'Local time (PDT) of sample'
        # Get lat, lon coordinates of instrument
        # lon = dsb['{}_llc'.format(rbr.mid)].sel(llc='longitude').item()
        dsp = dsp.assign_coords(lon=[self.lon])
        # lat = dsb['{}_llc'.format(rbr.mid)].sel(llc='latitude').item()
        dsp = dsp.assign_coords(lat=[self.lat])
        dsp.lat.attrs['standard_name'] = 'latitude'
        dsp.lat.attrs['long_name'] = 'Mooring latitude estimated from orthophoto'
        dsp.lat.attrs['units'] = 'degrees_north'
        dsp.lat.attrs['valid_min'] = -90.0
        dsp.lat.attrs['valid_max'] = 90.0
        dsp.lon.attrs['standard_name'] = 'longitude'
        dsp.lon.attrs['long_name'] = 'Mooring longitude estimated from orthophoto'
        dsp.lon.attrs['units'] = 'degrees_east'
        dsp.lon.attrs['valid_min'] = -180.0
        dsp.lon.attrs['valid_max'] = 180.0
        dsp.pressure.attrs['standard_name'] = 'sea_water_pressure'
        dsp.pressure.attrs['long_name'] = 'Pressure recorded at sea bottom'
        dsp.pressure.attrs['units'] = 'dbar'
        dsp.pressure.attrs['missing_value'] = fillvalue
        dsp.z_hyd.attrs['standard_name'] = 'depth'
        dsp.z_hyd.attrs['long_name'] = 'Hydrostatic pressure head'
        dsp.z_hyd.attrs['units'] = 'm'
        dsp.z_hyd.attrs['missing_value'] = fillvalue
        dsp.z_lin.attrs['standard_name'] = 'depth'
        dsp.z_lin.attrs['long_name'] = 'Linear reconstruction of instantaneous distance to sea surface'
        dsp.z_lin.attrs['units'] = 'm'
        dsp.z_lin.attrs['missing_value'] = fillvalue
        dsp.patm.attrs['standard_name'] = 'air_pressure'
        dsp.patm.attrs['long_name'] = 'Atmospheric pressure from NOAA MRY tidal station'
        dsp.patm.attrs['station_ID'] = '9413450'
        dsp.patm.attrs['units'] = 'dbar'
        dsp.patm.attrs['missing_value'] = fillvalue
        if args.instr == 'RBRDuetDT':
            dsp.temperature.attrs['standard_name'] = 'sea_water_temperature_at_sea_floor'
            dsp.temperature.attrs['long_name'] = 'Temperature recorded by instrument'
            dsp.temperature.attrs['units'] = 'degree_Celsius'
            dsp.temperature.attrs['missing_value'] = fillvalue

       # Global attributes
        dsp.attrs['title'] = ('ROXSI 2022 Asilomar Small-Scale Array ' + 
                              '{} pressure measurements from serial number ' + 
                              '{}.'.format(self.instr, self.ser))
        dsp.attrs['summary'] =  ('Nearshore bottom-mounted pressure sensor measurements.')
        dsp.attrs['instrument'] = self.instr
        dsp.attrs['height_above_seafloor'] = '0.08 m'
        dsp.attrs['mooring_ID'] = self.mid
        dsp.attrs['serial_number'] = self.ser
        dsp.attrs['Conventions'] = 'CF-1.8'
        dsp.attrs['featureType'] = "timeSeries"
        dsp.attrs['source'] =  "Sub-surface observation"
        dsp.attrs['date_created'] = str(DT.utcnow()) + ' UTC'
        dsp.attrs['references'] = 'https://github.com/mikapm/pyROXSI'
        dsp.attrs['creator_name'] = "Mika P. Malila"
        dsp.attrs['creator_email'] = "mikapm@unc.edu"
        dsp.attrs['institution'] = "University of North Carolina at Chapel Hill"

        # Encoding
        encoding = {'lat': {'zlib': False, '_FillValue': None},
                    'lon': {'zlib': False, '_FillValue': None},
                    'time': {'zlib': False, '_FillValue': None},
                    'pressure': {'_FillValue': fillvalue},        
                    'z_lin': {'_FillValue': fillvalue},        
                    'z_hyd': {'_FillValue': fillvalue},        
                    'patm': {'_FillValue': fillvalue},        
                }     
        if args.instr == 'RBRDuetDT':
            encoding['temperature'] = {'_FillValue': fillvalue}
        
        return dsp, encoding

    
    def dss2cf(self, ds, fillvalue, time_units, eta='lin'):
        """
        Convert spectral dataset ds to CF conventions. Specify type
        of surface reconstruction by eta='lin' (linear) or eta='hyd'
        (hydrostatic).
        """
        # Set requested fill value
        ds = ds.fillna(fillvalue)
        # Set attributes
        # Set variable attributes for output netcdf file
        ds.Ezz.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density'
        if eta == 'lin':
            spec_str = 'linear'
        elif eta == 'hyd':
            spec_str = 'hydrostatic'
        ds.Ezz.attrs['long_name'] = ('Scalar (frequency) wave variance density ' + 
                                    'spectrum from {} surface reconstruction'.format(
                                        spec_str))
        ds.Ezz.attrs['units'] = 'm^2/Hz'
        ds.freq.attrs['standard_name'] = 'sea_surface_wave_frequency'
        ds.freq.attrs['long_name'] = 'spectral frequencies in Hz'
        ds.freq.attrs['units'] = 'Hz'
        ds = ds.assign_coords(lat=self.lat)
        ds = ds.assign_coords(lon=self.lon)
        ds.lat.attrs['standard_name'] = 'latitude'
        ds.lat.attrs['long_name'] = 'Mooring latitude estimated from orthophoto'
        ds.lat.attrs['units'] = 'degrees_north'
        ds.lat.attrs['valid_min'] = -90.0
        ds.lat.attrs['valid_max'] = 90.0
        ds.lon.attrs['standard_name'] = 'longitude'
        ds.lon.attrs['long_name'] = 'Mooring longitude estimated from orthophoto'
        ds.lon.attrs['units'] = 'degrees_east'
        ds.lon.attrs['valid_min'] = -180.0
        ds.lon.attrs['valid_max'] = 180.0
        ds.time.encoding['units'] = time_units
        ds.time.attrs['units'] = time_units
        ds.time.attrs['standard_name'] = 'time'
        ds.time.attrs['long_name'] = 'Local time (PDT) of spectral segment start'
        # Sig. wave height and other integrated parameters
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
        ds.nu_LH57.attrs['units'] = 'dimensionless' 
        ds.nu_LH57.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density_bandwidth'
        ds.nu_LH57.attrs['long_name'] = 'spectral bandwidth following Longuet-Higgins (1957)'
        # Fill values
        ds.Hm0.attrs['missing_value'] = fillvalue
        ds.Te.attrs['missing_value'] = fillvalue
        ds.Tp_ind.attrs['missing_value'] = fillvalue
        ds.Tp_Y95.attrs['missing_value'] = fillvalue
        ds.nu_LH57.attrs['missing_value'] = fillvalue

       # Global attributes
        ds.attrs['title'] = ('ROXSI 2022 Asilomar Small-Scale Array ' + 
                             '{} wave spectra for serial number {}'.format(self.instr, self.mid))
        ds.attrs['summary'] =  ('Nearshore wave spectra from subsurface pressure measurements. '+
                                'Sea-surface elevation is the {} surface reconstruction.'.format(
                                    spec_str))
        ds.attrs['instrument'] = self.instr
        ds.attrs['mooring_ID'] = self.mid
        ds.attrs['serial_number'] = self.ser
        ds.attrs['segment_length'] = '1200 seconds'
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
                    'Ezz': {'_FillValue': fillvalue},        
                    'Hm0': {'_FillValue': fillvalue},        
                    'Te': {'_FillValue': fillvalue},
                    'Tp_ind': {'_FillValue': fillvalue},
                    'Tp_Y95': {'_FillValue': fillvalue},
                    'nu_LH57': {'_FillValue': fillvalue},
                   }     

        return ds, encoding


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
                help=('Instrument serial number.'),
                type=str,
                choices=['210356', '210357', '210358', '210359', '210360', '210361',
                         '41428', '41429', '124107', '124108', '124109', '210362'],
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
        parser.add_argument("-fillvalue", 
                help=("Fill value for NaN"),
                type=float,
                default=-9999.,
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

    # Define paths and load data
    data_root = os.path.join(args.dr, args.instr)
    matdir = os.path.join(data_root, 'Level1', 'mat')
    outdir = os.path.join(data_root, 'Level1', 'netcdf')
    
    # Read atmospheric pressure time series and calculate
    # atmospheric pressure anomaly
    fn_patm = os.path.join(args.dr, 'noaa_atm_pressure.csv')
    if not os.path.isfile(fn_patm):
        # csv file does not exist -> read mat file and generate dataframe
        fn_matp = os.path.join(args.dr, 'noaa_atm_pres_simple.mat')
        mat = loadmat(fn_matp)
        mat_pres = mat['A']['atm_pres'].item().squeeze()
        mat_time = mat['A']['time_vec'].item().squeeze() # In Matlab char() format
        # Make into pandas dataframe
        dfa = pd.DataFrame(data={'dbar':mat_pres}, 
                           index=pd.to_datetime(mat_time))
        dfa.index.rename('time', inplace=True)
        # convert from mbar to dbar
        dfa['dbar'] /= 100
        dfa['dbar'] -= 0.032 # Empirical correction factor
        # Calculate anomaly from mean
        dfa['dbar_anom'] = dfa['dbar'] - dfa['dbar'].mean()
        # Save as csv
        dfa.to_csv(fn_patm)
    else:
        dfa = pd.read_csv(fn_patm, parse_dates=['time']).set_index('time')

    # Read bathymetry netcdf file
    bathydir = os.path.join(args.dr, 'Bathy')
    fn_bathy = os.path.join(bathydir, 'Asilomar_2022_SSA_bathy.nc')
    dsb = xr.decode_cf(xr.open_dataset(fn_bathy, decode_coords='all'))

   # Mooring info excel file path (used when initializing ADV class)
    fn_minfo = os.path.join(args.dr, 'Asilomar_SSA_2022_mooring_info.xlsx')
    
    # Initialize RBR class object
    rbr = RBR(datadir=matdir, ser=args.ser, fs=args.fs, 
              instr=args.instr, patm=dfa, mooring_info=fn_minfo,
              bathy=dsb)

    # Define reference date and units for timestamps
    ref_date=pd.Timestamp('2022-06-25')
    time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(
        ref_date)

    # Filename of output netcdf files
    fn_ncout = os.path.join(outdir, 'Asilomar_2022_SSA_L1_{}_{}_pressure.nc'.format(
        args.instr, args.ser)) # Pressure timeseries
    # Spectra
    fn_ncspec_l = os.path.join(outdir, 'Asilomar_2022_SSA_L1_{}_{}_spec_lin.nc'.format(
        args.instr, args.ser)) # Pressure timeseries
    fn_ncspec_h = os.path.join(outdir, 'Asilomar_2022_SSA_L1_{}_{}_spec_hyd.nc'.format(
        args.instr, args.ser)) # Pressure timeseries
    # Check if output netcdf files already exist
    isnc_p = os.path.isfile(fn_ncout)
    isnc_sl = os.path.isfile(fn_ncspec_l)
    isnc_sh = os.path.isfile(fn_ncspec_h)
    if not isnc_p or not isnc_sl or not isnc_sh or args.overwrite_nc:
        # Save datasets in list for concatenating
        dsp_list = [] # Daily pressure
        dsld_list = [] # Daily spectra (linear)
        dshd_list = [] # Daily spectra (hydrostatic)
        # Iterate over daily mat files and concatenate into pd.DataFrame
        for fi, fn_mat in enumerate(rbr.fns):
            print('Loading pressure sensor mat file {}'.format(os.path.basename(fn_mat)))
            mat = loadmat(fn_mat)
            # Read pressure time series pt
            pt = np.array(mat[rbr.matkey]['P'].item()).squeeze()
            # Hydrostatic pressure ph
            ph = np.array(mat[rbr.matkey]['Pwater'].item()).squeeze()
            # Timestamps
            time_mat = np.array(mat[rbr.matkey]['time_dnum'].item()).squeeze()
            time_ind = pd.to_datetime(time_mat-719529,unit='d') # Convert timestamps

            # Get atmospheric pressure for current time
            dfpa = rbr.patm.loc[time_ind[0]:time_ind[-1]].copy()
            # Reindex to water pressure time index
            dfpa = dfpa.reindex(time_ind, method='bfill').interpolate()
            # Make pandas DataFrame
            if rbr.instr == 'RBRSoloD':
                dfp = pd.DataFrame(data={'pressure':pt,
                                        'z_hyd':ph,
                                        'z_lin':np.ones_like(pt)*np.nan,
                                        'patm':dfpa['dbar'].values,
                                        }, 
                                index=time_ind)

            elif rbr.instr == 'RBRDuetDT':
                # Also save temperature from Duets
                temp = np.array(mat[rbr.matkey]['Twater'].item()).squeeze()
                dfp = pd.DataFrame(data={'pressure':pt,
                                        'temperature':temp,
                                        'z_hyd':ph,
                                        'z_lin':np.ones_like(pt)*np.nan,
                                        'patm':dfpa['dbar'].values,
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
            dsl_list = [] # List for concatenating linear spectra
            dsh_list = [] # List for concatenating hydrostatic spectra
            for sn, seg in enumerate(np.array_split(dfp['z_hyd'], nseg)):
                # There's a gap of missing data on 2022-07-12 between 
                # 05:00 - 05:20. Skip that.
                if np.sum(np.isnan(seg.values)) > 1000:
                    print('Too many gaps.')
                    continue
                # Get segment start and end times
                t0ss = seg.index[0]
                t1ss = seg.index[-1]
                # Convert hydrostatic depth to eta w/ linear TRF
                dfp['z_lin'].loc[t0ss:t1ss] = rbr.p2eta_lin(
                    seg.interpolate(method='ffill').interpolate(method='bfill').values,
                    M=args.M, fmin=args.fmin, fmax=args.fmax)
                # Compute spectra
                tdt = pd.to_datetime(pd.Timestamp(seg.index[0]).round('20T')).to_pydatetime()
                time_val = date2num(tdt, time_units, calendar='standard', 
                                    has_year_zero=True).astype('f8')
                spec_seg = dfp['z_lin'].loc[t0ss:t1ss].interpolate(method='bfill').interpolate(method='ffill').values
                dsl = rpws.spec_uvz(spec_seg, timestamp=time_val, fs=args.fs,
                                    fmin=args.fmin, fmax=args.fmax)
                dsl_list.append(dsl)
                spec_seg = dfp['z_hyd'].loc[t0ss:t1ss].interpolate(method='bfill').interpolate(method='ffill').values
                dsh = rpws.spec_uvz(spec_seg, timestamp=time_val, fs=args.fs,
                                    fmin=args.fmin, fmax=args.fmax)
                dsh_list.append(dsh)
            # Convert times to python datetime and then to floats for netcdf
            time_dt = pd.to_datetime(dfp.index.values).to_pydatetime()
            time_vals = date2num(time_dt, time_units, calendar='standard', 
                                 has_year_zero=True)
            # Convert daily dataframe to xr.Dataset
            dsp, encoding = rbr.dfp2dsp(dfp, time_vals=time_vals,
                                        time_units=time_units,
                                        fillvalue=args.fillvalue)
            # Append daily dataset to list for concatenating
            dsp_list.append(dsp)
            # Concatenate spectral datasets to daily datasets
            dsld = xr.concat(dsl_list, dim='time')
            dshd = xr.concat(dsh_list, dim='time')
            # Append daily spectral datasets to lists
            dsld_list.append(dsld)
            dshd_list.append(dshd)

        # Concatenate daily datasets into one
        ds = xr.concat(dsp_list, dim='time')
        # Spectra
        dss_lin = xr.concat(dsld_list, dim='time')
        dss_hyd = xr.concat(dshd_list, dim='time')

        # Save to netcdf
        if not os.path.isfile(fn_ncout) or args.overwrite_nc:
            ds.to_netcdf(fn_ncout, encoding=encoding)
        # Save linear spectra
        dss_lin, encoding_l = rbr.dss2cf(dss_lin, fillvalue=args.fillvalue, 
                                         time_units=time_units, eta='lin')
        if not os.path.isfile(fn_ncspec_l) or args.overwrite_nc:
            dss_lin.to_netcdf(fn_ncspec_l, encoding=encoding_l)
        # Save hydrostatic spectra
        dss_hyd, encoding_h = rbr.dss2cf(dss_hyd, fillvalue=args.fillvalue, 
                                         time_units=time_units, eta='hyd')
        if not os.path.isfile(fn_ncspec_h) or args.overwrite_nc:
            dss_hyd.to_netcdf(fn_ncspec_h, encoding=encoding_h)
    else:
        ds = xr.decode_cf(xr.open_dataset(fn_ncout, decode_coords='all'))
        dss_lin = xr.decode_cf(xr.open_dataset(fn_ncspec_l, decode_coords='all'))
        dss_hyd = xr.decode_cf(xr.open_dataset(fn_ncspec_h, decode_coords='all'))

