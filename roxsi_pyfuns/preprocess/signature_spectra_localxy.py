"""
Estimate cross-alongshore wave spectra from Nortek
Signature1000 ADCP data using local ROXSI x,y coord.
system.
"""

# Imports
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats, signal
from datetime import datetime as DT
from cftime import date2num
from argparse import ArgumentParser
from tqdm import tqdm
from signature_preprocess import ADCP
from roxsi_pyfuns import coordinate_transforms as rpct
from roxsi_pyfuns import wave_spectra as rpws

# Input arguments
def parse_args(**kwargs):
    parser = ArgumentParser()
    parser.add_argument("-dr", 
            help=("Path to data root directory"),
            type=str,
            default=r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray',
            )
    parser.add_argument("-fillvalue", 
            help=("Fill value for NaN to save in netcdf file"),
            type=float,
            default=-9999.,
            )
    parser.add_argument("-ser", 
            help=("Signature serial number"),
            type=str,
            default='103094',
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

# Paths, files
sigdir = os.path.join(args.dr, 'Signatures', 'Level1', args.ser)
outdir = os.path.join(sigdir, 'Spectra')
if not os.path.isdir(outdir):
    # Make output dir.
    os.mkdir(outdir)
# Output netcdf filename
fn_out = os.path.join(outdir, 'Asilomar_2022_SSA_Signature_{}_spec_AST.nc'.format(
    args.ser))

# Dict for connecting serial no. w/ mooring ID
mids = {'103063': 'L1', '103088': 'C1', '103094': 'C3', 
        '103110': 'C6', '103206': 'L5'}
# Get correct mooring ID for chosen serial number
mid = mids[args.ser]

# Use bathymetry to get location + MSL
bathydir = os.path.join(args.dr, 'Bathy') # Bathymetry dir.
fnb = os.path.join(bathydir, 'Asilomar_2022_SSA_bathy_updated_1m.nc')
dsb = xr.open_dataset(fnb, decode_coords='all') # Bathymetry dataset
lat = dsb[mid].attrs['latitude'].item()
lon = dsb[mid].attrs['longitude'].item()
xl = dsb[mid].attrs['x_loc'].item() # Local ROXSI x-coord.
yl = dsb[mid].attrs['y_loc'].item() # Local ROSXI y-coord.
z_msl = dsb[mid].item() # Mean depth of mooring (negative)
print('Mooring {}, depth={:.2f}m, lat={}, lon={}'.format(
    mid, z_msl, lat, lon))
# Use L5 as depth of surrounding region
z_msl_surr = dsb['C5'].item() # Mean depth of mooring (negative)

# Initialize ADCP class object
adcp = ADCP(sigdir, args.ser,)
# Save hourly spectra in list for concatenation
dss_list = []

# Check if output netcdf file already exists
if not os.path.isfile(fn_out):
    # Make date range array
    t0 = pd.Timestamp('2022-06-23') # Start (earliest) date
    t1 = pd.Timestamp('2022-07-22') # End (latest) date
    date_range = pd.date_range(t0, t1, freq='1D')
    # Iterate over dates
    for date in date_range:
        print('date: ', date)
        # Read daily Sig netcdf file
        fn = os.path.join(sigdir, 'Asilomar_SSA_L1_Sig_Vel_{}_{}.nc'.format(
            mid, pd.Timestamp.strftime(date, '%Y%m%d')))
        if os.path.isfile(fn):
            ds = xr.open_dataset(fn, decode_coords='all')
        else:
            # No dataset for current date -> skip
            continue
        # Make 1-h segment range array for current date
        seg_range = pd.date_range(date, date + pd.Timedelta(days=1),
                                  freq='1H')
        # Iterate over segments and estimate spectra
        for i, t0s in tqdm(enumerate(seg_range[:-1])):
            # Segment end time
            t1s = seg_range[i+1]
            # print('{} {}'.format(t0s, t1s))
            # Take out time segment
            seg = ds.sel(time=slice(t0s, t1s))

            # Check if current segment has data
            if np.all(np.isnan(seg.vE)):
                # All NaN velocities -> skip
                continue

            # Despiked AST signal for sea-surface elevation
            ast = seg.ASTd.copy()
            if np.sum(np.isnan(ast)) > 1000:
                noast = True # Flag
                # Too many missing AST values -> use eta_lin_krms
                print('No AST at {}'.format(t0s))
                ast = seg.eta_lin_krms.copy()
                # Depth of mooring (use linear surface reconstruction)
                depth_adcp = seg.z_lin.mean().item()
            else:
                noast = False # Flag
                # Depth of mooring
                depth_adcp = ast.mean().item() # hourly mean depth of ADCP
            depth_loc = depth_adcp + adcp.zp # Local water depth to bottom
            msl_dev = depth_loc + z_msl # Deviation from MSL
            # Surrounding depth based on L5
            depth_surr = -z_msl_surr + msl_dev

            # Rotate East, North velocities to cross-/alongshore vel.
            ui = seg.vEhpr.values
            vi = seg.vNhpr.values
            # Rotation angle (use ROXSI LSA reference angle from bathy)
            ref_ang = int(dsb.attrs['reference_angle'][:3])
            angle_math = 270 - ref_ang
            if angle_math < 0:
                angle_math += 360
            angle_math = np.deg2rad(angle_math) # radians
            # Initialize cross-/longshore velocity arrays
            ul = np.ones_like(ui) * np.nan # Cross-shore vel.
            vl = np.ones_like(vi) * np.nan # Along-shore vel.
            # Rotate at each range level
            for i,r in enumerate(seg.range.values):
                # Copy E, N velocity components
                ur = seg.vEhpr.sel(range=r).values.copy()
                vr = seg.vNhpr.sel(range=r).values.copy()
                # Rotate
                ul[:,i], vl[:,i] = rpct.rotate_vel(ur, vr, angle_math)
            # Save rotated velocities in dataset segment 
            seg['vCS'] = (['time', 'range'], ul)
            seg['vLS'] = (['time', 'range'], vl)

            # Estimate spectrum using cross-/alongshore velocities
            seglen = 60 * 60 # Segment length (seconds)
            if noast:
                # No AST signal -> use eta_lin_krms
                dss = adcp.wavespec(seg, u='vCS', v='vLS', z='eta_lin_krms',
                                    fmin=0.05, fmax=1.0, seglen=seglen)
            else:
                dss = adcp.wavespec(seg, u='vCS', v='vLS', z='ASTd',
                                    fmin=0.05, fmax=1.0, seglen=seglen)
            # Convert time array to numerical format
            time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(ref_date)
            time = pd.to_datetime(dss.time.values).to_pydatetime()
            time_vals = date2num(time, 
                                time_units, calendar='standard', 
                                has_year_zero=True)
            dss.coords['time'] = time_vals.astype(float)
            # Assign lat, lon coords
            dss = dss.assign_coords(lon=[lon])
            dss = dss.assign_coords(lat=[lat])

            # Save additional variables to spectral dataset
            dss['depth_adcp'] = (['time'], [depth_adcp])
            dss.depth_adcp.attrs['standard_name'] = 'depth'
            dss.depth_adcp.attrs['long_name'] = 'Water depth from instrument'
            dss.depth_adcp.attrs['units'] = 'm'
            dss['depth_loc'] = (['time'], [depth_loc])
            dss.depth_loc.attrs['standard_name'] = 'depth'
            dss.depth_loc.attrs['long_name'] = 'Water depth from seabed'
            dss.depth_loc.attrs['units'] = 'm'
            dss['depth_surr'] = (['time'], [depth_surr])
            dss.depth_surr.attrs['standard_name'] = 'depth'
            dss.depth_surr.attrs['long_name'] = 'Surrounding water depth from seabed at L5 mooring'
            dss.depth_surr.attrs['units'] = 'm'
            dss['msl_dev'] = (['time'], [msl_dev])
            dss.msl_dev.attrs['standard_name'] = 'sea_surface_height_above_sea_level'
            dss.msl_dev.attrs['long_name'] = 'Mean water level above NAVD 88 datum'
            dss.msl_dev.attrs['units'] = 'm'
            dss['z_msl'] = (['time'], [z_msl])
            dss.z_msl.attrs['standard_name'] = 'sea_floor_depth_below_mean_sea_level'
            dss.z_msl.attrs['long_name'] = 'Mooring depth relative to NAVD 88 datum'
            dss.z_msl.attrs['units'] = 'm'
            # Mean velocities
            z_range = dss.vel_binh.item() # Range bin used for velocities
            uCS_mean = seg.vCS.sel(range=z_range).mean(dim='time').item()
            dss['uCS_mean'] = (['time'], [uCS_mean])
            dss.uCS_mean.attrs['standard_name'] = 'mean_cross_shore_water_velocity'
            dss.uCS_mean.attrs['long_name'] = 'Mean cross-shore velocity'
            dss.uCS_mean.attrs['units'] = 'm/s'
            uLS_mean = seg.vLS.sel(range=z_range).mean(dim='time').item()
            dss['uLS_mean'] = (['time'], [uLS_mean])
            dss.uLS_mean.attrs['standard_name'] = 'mean_along_shore_water_velocity'
            dss.uLS_mean.attrs['long_name'] = 'Mean along-shore velocity'
            dss.uLS_mean.attrs['units'] = 'm/s'
            # Surface elevation statistics
            ast_eta = ast - ast.mean() # AST surface elevation
            skew = stats.skew(ast_eta) # Surf. elev. skewness
            dss['skew'] = (['time'], [skew])
            dss.skew.attrs['standard_name'] = 'sea_surface_skewness'
            dss.skew.attrs['long_name'] = 'Sea surface elevation skewness'
            dss.skew.attrs['units'] = 'dimensionless'
            kurt = stats.kurtosis(ast_eta) # Kurtosis
            dss['kurt'] = (['time'], [kurt])
            dss.kurt.attrs['standard_name'] = 'sea_surface_kurtosis'
            dss.kurt.attrs['long_name'] = 'Sea surface elevation kurtosis'
            dss.kurt.attrs['units'] = 'dimensionless'
            # Asymmetry
            asym = stats.skew(np.imag(signal.hilbert(ast_eta.values)))
            dss['asym'] = (['time'], [asym])
            dss.asym.attrs['standard_name'] = 'sea_surface_asymmetry'
            dss.asym.attrs['long_name'] = 'Sea surface elevation asymmetry'
            dss.asym.attrs['units'] = 'dimensionless'
            # Ursell number
            Tp = dss.Tp_ind.item() # Peak freq
            fp = 1 / dss.Tp_ind.item() # Peak freq
            dss['fp'] = (['time'], [fp])
            dss.fp.attrs['standard_name'] = 'sea_surface_wave_frequency_at_variance_spectral_density_maximum'
            dss.fp.attrs['long_name'] = 'Spectral peak frequency'
            dss.fp.attrs['units'] = 'Hz'
            # Peak wavenumber (linear, surrounding water depth)
            kp = rpws.waveno_full(2*np.pi * fp, d=depth_surr).item()
            dss['kp'] = (['time'], [kp])
            dss.kp.attrs['standard_name'] = 'sea_surface_wavenumber_at_variance_spectral_density_maximum'
            dss.kp.attrs['long_name'] = 'Linear peak wavenumber in surrounding water depth at L5 mooring'
            dss.kp.attrs['units'] = 'rad/m'
            # Shallowness parameter (local water depth)
            mu = (kp * depth_loc)**2 
            dss['mu'] = (['time'], [mu])
            dss.mu.attrs['standard_name'] = 'shallowness_parameter'
            dss.mu.attrs['long_name'] = 'Spectral shallowness parameter in local water depth'
            dss.mu.attrs['units'] = 'dimensionless'
            # Steepness (local depth)
            eps = 2 * np.nanstd(ast_eta.values) / depth_loc 
            dss['eps'] = (['time'], [eps])
            dss.eps.attrs['standard_name'] = 'steepness_parameter'
            dss.eps.attrs['long_name'] = 'Spectral steepness parameter in local water depth'
            dss.eps.attrs['units'] = 'dimensionless'
            Ur = eps / mu # (use local water depth)
            # Ur = (3/4) * (9.81/(8*np.pi**2)) * (dss.Hm0.item()*Tp**2 / depth_loc**2)
            dss['Ur'] = (['time'], [Ur])
            dss.Ur.attrs['standard_name'] = 'ursell_number'
            dss.Ur.attrs['long_name'] = 'Ursell parameter'
            dss.Ur.attrs['units'] = 'dimensionless'
            # 0-th moments (ie variance) of elevation and velocity spectra
            m0z = rpws.spec_moment(dss.Ezz.values, dss.freq.values, 0).item()
            dss['m0z'] = (['time'], [m0z])
            dss.m0z.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density'
            dss.m0z.attrs['long_name'] = 'Zeroth-order elevation variance spectral moment m0'
            dss.m0z.attrs['units'] = 'dimensionless'
            m0u = rpws.spec_moment(dss.Euu.values, dss.freq.values, 0).item()
            dss['m0u'] = (['time'], [m0u])
            dss.m0u.attrs['standard_name'] = 'cross_shore_velocity_variance_spectral_density'
            dss.m0u.attrs['long_name'] = 'Zeroth-order cross-shore velocity variance spectral moment m0'
            dss.m0u.attrs['units'] = 'dimensionless'
            m0v = rpws.spec_moment(dss.Evv.values, dss.freq.values, 0).item()
            dss['m0v'] = (['time'], [m0v])
            dss.m0v.attrs['standard_name'] = 'along_shore_velocity_variance_spectral_density'
            dss.m0v.attrs['long_name'] = 'Zeroth-order along-shore velocity variance spectral moment m0'
            dss.m0v.attrs['units'] = 'dimensionless'

            # Set variable attributes for output netcdf file
            dss.Ezz.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density'
            dss.Efth.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density'
            z_str = 'AST'
            dss.Ezz.attrs['long_name'] = 'scalar (frequency) wave variance density spectrum from {}'.format(
                z_str)
            dss.Efth.attrs['long_name'] = 'directional wave variance density spectrum from {}'.format(
                z_str)
            dss.Ezz.attrs['units'] = 'm^2/Hz'
            dss.Efth.attrs['units'] = 'm^2/Hz/deg'
            dss.Evv.attrs['units'] = 'm^2/Hz'
            dss.Evv.attrs['standard_name'] = 'northward_sea_water_velocity_variance_spectral_density'
            dss.Evv.attrs['long_name'] = 'auto displacement spectrum from northward velocity component'
            dss.Euu.attrs['units'] = 'm^2/Hz'
            dss.Euu.attrs['standard_name'] = 'eastward_sea_water_velocity_variance_spectral_density'
            dss.Euu.attrs['long_name'] = 'auto displacement spectrum from eastward velocity component'
            dss.a1.attrs['units'] = 'dimensionless'
            dss.a1.attrs['standard_name'] = 'a1_directional_fourier_moment'
            dss.a1.attrs['long_name'] = 'a1 following Kuik et al. (1988) and Herbers et al. (2012)'
            dss.a2.attrs['units'] = 'dimensionless'
            dss.a2.attrs['standard_name'] = 'a2_directional_fourier_moment'
            dss.a2.attrs['long_name'] = 'a2 following Kuik et al. (1988) and Herbers et al. (2012)'
            dss.b1.attrs['units'] = 'dimensionless'
            dss.b1.attrs['standard_name'] = 'b1_directional_fourier_moment'
            dss.b1.attrs['long_name'] = 'b1 following Kuik et al. (1988) and Herbers et al. (2012)'
            dss.b2.attrs['units'] = 'dimensionless'
            dss.b2.attrs['standard_name'] = 'b2_directional_fourier_moment'
            dss.b2.attrs['long_name'] = 'b2 following Kuik et al. (1988) and Herbers et al. (2012)'
            dss.dspr_freq.attrs['units'] = 'angular_degree'
            dss.dspr_freq.attrs['standard_name'] = 'sea_surface_wind_wave_directional_spread'
            dss.dspr_freq.attrs['long_name'] = 'directional spread as a function of frequency'
            dss.dspr.attrs['units'] = 'angular_degree'
            dss.dspr.attrs['standard_name'] = 'sea_surface_wind_wave_directional_spread'
            dss.dspr.attrs['long_name'] = 'mean directional spread following Kuik et al. (1988)'
            dss.mdir.attrs['units'] = 'angular_degree'
            dss.mdir.attrs['standard_name'] = 'sea_surface_wind_wave_direction'
            dss.mdir.attrs['long_name'] = 'mean wave direction following Kuik et al. (1988)'
            dss.dirs_freq.attrs['units'] = 'angular_degree'
            dss.dirs_freq.attrs['standard_name'] = 'sea_surface_wind_wave_direction'
            dss.dirs_freq.attrs['long_name'] = 'wave energy directions per frequency'
            dss.freq.attrs['standard_name'] = 'sea_surface_wave_frequency'
            dss.freq.attrs['long_name'] = 'spectral frequencies in Hz'
            dss.freq.attrs['units'] = 'Hz'
            dss.direction.attrs['standard_name'] = 'sea_surface_wave_direction'
            dss.direction.attrs['long_name'] = 'directional distribution in degrees (dir. from, nautical convention)'
            dss.direction.attrs['units'] = 'deg'
            dss.lat.attrs['standard_name'] = 'latitude'
            dss.lat.attrs['long_name'] = 'Approximate latitude of mooring'
            dss.lat.attrs['units'] = 'degrees_north'
            dss.lat.attrs['valid_min'] = -90.0
            dss.lat.attrs['valid_max'] = 90.0
            dss.lon.attrs['standard_name'] = 'longitude'
            dss.lon.attrs['long_name'] = 'Approximate longitude of mooring'
            dss.lon.attrs['units'] = 'degrees_east'
            dss.lon.attrs['valid_min'] = -180.0
            dss.lon.attrs['valid_max'] = 180.0
            dss.time.encoding['units'] = time_units
            dss.time.attrs['units'] = time_units
            dss.time.attrs['standard_name'] = 'time'
            dss.time.attrs['long_name'] = 'Local time (PDT) of spectral segment start'
            # Sig. wave height and other integrated parameters
            dss.vel_binh.attrs['units'] = 'm'
            dss.vel_binh.attrs['standard_name'] = 'height'
            dss.vel_binh.attrs['long_name'] = 'horizontal velocity bin center height above seabed'
            dss.coh_uz.attrs['units'] = 'dimensionless'
            dss.coh_uz.attrs['standard_name'] = 'coherence'
            dss.coh_uz.attrs['long_name'] = 'coherence of horizontal velocity and surface elevation'
            dss.coh_vz.attrs['units'] = 'dimensionless'
            dss.coh_vz.attrs['standard_name'] = 'coherence'
            dss.coh_vz.attrs['long_name'] = 'coherence of horizontal velocity and surface elevation'
            dss.coh_uv.attrs['units'] = 'dimensionless'
            dss.coh_uv.attrs['standard_name'] = 'coherence'
            dss.coh_uv.attrs['long_name'] = 'coherence of horizontal velocities'
            dss.ph_uv.attrs['units'] = 'radians'
            dss.ph_uv.attrs['standard_name'] = 'phase_angle'
            dss.ph_uv.attrs['long_name'] = 'phase angle of horizontal velocities'
            dss.ph_vz.attrs['units'] = 'radians'
            dss.ph_vz.attrs['standard_name'] = 'phase_angle'
            dss.ph_vz.attrs['long_name'] = 'phase angle of horizontal velocity and surface elevation'
            dss.ph_uz.attrs['units'] = 'radians'
            dss.ph_uz.attrs['standard_name'] = 'phase_angle'
            dss.ph_uz.attrs['long_name'] = 'phase angle of horizontal velocity and surface elevation'
            dss.Hm0.attrs['units'] = 'm'
            dss.Hm0.attrs['standard_name'] = 'sea_surface_wave_significant_height'
            dss.Hm0.attrs['long_name'] = 'Hs estimate from 0th spectral moment'
            dss.Te.attrs['units'] = 's'
            dss.Te.attrs['standard_name'] = 'sea_surface_wave_energy_period'
            dss.Te.attrs['long_name'] = 'energy-weighted wave period'
            dss.Tp_ind.attrs['units'] = 's'
            dss.Tp_ind.attrs['standard_name'] = 'sea_surface_primary_swell_wave_period_at_variance_spectral_density_maximum'
            dss.Tp_ind.attrs['long_name'] = 'wave period at maximum spectral energy'
            dss.Tp_Y95.attrs['units'] = 's'
            dss.Tp_Y95.attrs['standard_name'] = 'sea_surface_primary_swell_wave_period_at_variance_spectral_density_maximum'
            dss.Tp_Y95.attrs['long_name'] = 'peak wave period following Young (1995, Ocean Eng.)'
            dss.Dp_ind.attrs['units'] = 'angular_degree' 
            dss.Dp_ind.attrs['standard_name'] = 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum'
            dss.Dp_ind.attrs['long_name'] = 'peak wave direction at maximum energy frequency'
            dss.Dp_Y95.attrs['units'] = 'angular_degree' 
            dss.Dp_Y95.attrs['standard_name'] = 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum'
            dss.Dp_Y95.attrs['long_name'] = 'peak wave direction at Tp_Y95 frequency'
            dss.nu_LH57.attrs['units'] = 'dimensionless' 
            dss.nu_LH57.attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density_bandwidth'
            dss.nu_LH57.attrs['long_name'] = 'spectral bandwidth following Longuet-Higgins (1957)'

        # Global attributes
            dss.attrs['title'] = ('ROXSI 2022 Asilomar Small-Scale Array ' + 
                                'Signature1000 {} wave spectra'.format(mid))
            dss.attrs['summary'] =  ('Nearshore wave spectra from ADCP measurements. '+
                                    'Sea-surface elevation is the despiked ADCP ' + 
                                    'acoustic surface track (AST) signal, and the ' + 
                                    'horizontal velocities are despiked cross-/alongshore ' + 
                                    'velocities from the range bin specified by the variable ' +
                                    'vel_binh.')
            dss.attrs['instrument'] = 'Nortek Signature 1000'
            dss.attrs['x_loc'] = xl
            dss.attrs['y_loc'] = yl
            dss.attrs['reference_angle'] = '{} deg'.format(ref_ang)
            dss.attrs['mooring_ID'] = mid
            dss.attrs['serial_number'] = args.ser
            dss.attrs['transducer_height'] = '{} m'.format(adcp.zp)
            dss.attrs['segment_length'] = '3600 seconds'
            dss.attrs['Conventions'] = 'CF-1.8'
            dss.attrs['featureType'] = "timeSeries"
            dss.attrs['source'] =  "Sub-surface observation"
            dss.attrs['date_created'] = str(DT.utcnow()) + ' UTC'
            dss.attrs['references'] = 'https://github.com/mikapm/pyROXSI'
            dss.attrs['creator_name'] = "Mika P. Malila"
            dss.attrs['creator_email'] = "mikapm@unc.edu"
            dss.attrs['institution'] = "University of North Carolina at Chapel Hill"

            # Set encoding before saving
            encoding = {'time': {'zlib': False, '_FillValue': None},
                        'freq': {'zlib': False, '_FillValue': None},
                        'direction': {'zlib': False, '_FillValue': None},
                        'lat': {'zlib': False, '_FillValue': None},
                        'lon': {'zlib': False, '_FillValue': None},
                    }
            # Set variable fill values
            for k in list(dss.keys()):
                encoding[k] = {'_FillValue': args.fillvalue}

            # Append dss to list for concatenating
            dss_list.append(dss)

    # Concatenate all spectra into one
    dsc = xr.concat(dss_list, dim='time')

    # Save concatenated dataset as netcdf
    print('Saving netcdf ...')
    dsc.to_netcdf(fn_out, encoding=encoding)

