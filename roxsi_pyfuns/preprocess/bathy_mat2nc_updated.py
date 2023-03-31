"""
Convert updated Asilomar SSA bathymetry file from .mat to netcdf.
Based on zmsl_Asilomar_gridded.mat by Olavo Marques;
mooring coordinates from Asilomar_bathy_SSarray.m script by 
Johanna Rosman.
"""

# Imports
import os
import sys
import utm
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as DT
import salem
from pyproj import Proj
from mat73 import loadmat
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cmocean
from argparse import ArgumentParser

# Input arguments
def parse_args(**kwargs):
    parser = ArgumentParser()
    parser.add_argument("-dr", 
            help=("Path to data root directory"),
            type=str,
            default=r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/mfiles',
            )
    parser.add_argument("-out", 
            help=("Path to output directory"),
            type=str,
            default=r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray/Bathy',
            )
    parser.add_argument("-fillvalue", 
            help=("Fill value for NaN to save in netcdf file"),
            type=float,
            default=9999.,
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

# Read API key from environment variable
api_key = os.environ["OPENAI_API_KEY"]

# Read bathymetry .mat file
fn_mat = os.path.join(args.dr, 'zmsl_Asilomar_gridded_1m.mat')
mat = loadmat(fn_mat)

# Read large-scale array mooring locations table
fn_lsa = os.path.join(args.dr, 'ROXSI2022_LSA_mooring_locations.csv')
df_lsa = pd.read_csv(fn_lsa)#.set_index('Mooring')
# Fill missing values w/ NaN
df_lsa = df_lsa.replace(9999999999.0, np.nan)

# Read lat, lon and z_msl from Olavo's .mat file
lat = mat['bathymetry']['latitude'].squeeze()
lon = mat['bathymetry']['longitude'].squeeze()
# Use griddata-interpolated z_msl
z_msl = mat['bathymetry']['z_msl_gd'].squeeze() 
# Read local x,y coordinates too
x = mat['bathymetry']['x'].squeeze()
y = mat['bathymetry']['y'].squeeze()

# Reference lon, lat for local coord. system (following LSA)
utm_zone = 10 # Standard UTM zone number (northern hemisphere)
lon_ref = -121.9403306888889 # see ROXSI_xygrids.mat
lat_ref = 36.624032972222224
originEasting, originNorthing, _, _ = utm.from_latlon(lat_ref, lon_ref,)
ang_ref = 293 # Rotation angle for cross-/longshore dir.

# Estimated mooring locations in lat, lon coordinates
# (UTM coords calculated by C1_loc[0]+eastingMin, 
# C1_loc[1]+northingMin, and then converted
# from UTM to lat, lon (zone 10N) using converter at
# http://rcn.montana.edu/resources/Converter.aspx)
ssa_moorings = {
    'C1':np.array([36.62509622046095, -121.94341728936496]),
    'C2':np.array([36.6250784479023, -121.94337648871583]),
    'C3':np.array([36.62506909478911, -121.94332154537203]),
    'C4':np.array([36.62506246035691, -121.94328763063506]),
    'C5':np.array([36.62505607173514, -121.94325612377533]),
    'C6':np.array([36.62506316794421, -121.9432242068824]),
    'L1':np.array([36.62497362592774, -121.9433549582822]),
    'L2':np.array([36.62501601212474, -121.94333846632198]),
    'L4':np.array([36.62513112216428, -121.94335193932969]),
    'L5':np.array([36.62515099871671, -121.94328510346281]),
    'C2L2':np.array([36.62511160385235, -121.94336027372623]),
    'C2L4':np.array([36.62505305530001, -121.94338426919181]),
    'C4L2':np.array([36.62502930641834, -121.9432964486891]),
    'C4L4':np.array([36.625094856799286, -121.94327898509006]),
    'C1_5':np.array([36.62508267285272, -121.94339883098975]),
    'M1':np.array([36.62516568059234, -121.94340238052408]),
}

# Save bathymetry to netcdf
fn_nc = os.path.join(args.out, 'Asilomar_2022_SSA_bathy_updated_1m.nc')
if not os.path.isfile(fn_nc) or args.overwrite_nc:
    # Define variables dict
    data_vars = {} # Empty for now
    # Generate xr.Dataset
    dsb = xr.Dataset(data_vars={'z_msl': (['y', 'x'], z_msl)},
                     coords={'x': (['x'], x),
                             'y': (['y'], y),
                             'lat': (['y', 'x'], lat),
                             'lon': (['y', 'x'], lon),
                            }
                    )
    # Get SSA mooring location depths
    ssa_depths = {} # Dict to store depths at SSA mooring locations
    for mid in ssa_moorings.keys():
        # Lon, lat coordinates of current point
        mlat = ssa_moorings[mid][0]
        mlon = ssa_moorings[mid][1]
        # Convert to UTM eastings, northings
        eu, nu, _, _ = utm.from_latlon(mlat, mlon)
        # Get local x,y coords. from UTM
        angle_rot = np.deg2rad(270 - ang_ref)
        # Shift data relative to origin
        x_aux = eu - originEasting
        y_aux = nu - originNorthing
        # Compute local x,y coordinates
        xl = x_aux*np.cos(angle_rot) + y_aux*np.sin(angle_rot)
        yl = y_aux*np.cos(angle_rot) - x_aux*np.sin(angle_rot)
        # Get nearest depth from gridded bathymetry
        depth = dsb.z_msl.sel(x=xl, y=yl, method='nearest').item()
        # Save to dsb
        dsb[mid] = ([], depth)
        # Attributes
        dsb[mid].attrs['latitude'] = mlat
        dsb[mid].attrs['longitude'] = mlon
        dsb[mid].attrs['x_loc'] = xl
        dsb[mid].attrs['y_loc'] = yl
        dsb[mid].attrs['standard_name'] = 'depth'
        dsb[mid].attrs['long_name'] = 'SSA {} mooring depth'.format(mid)

    # Add large-scale array mooring locations and depths
    for row in df_lsa.iterrows():
        if row[1]['Array'] == 'Asilomar':
            # Get mooring location and ID
            lon_lsa = row[1]['Deployed longitude']
            lat_lsa = row[1]['Deployed latitude']
            mid_lsa = row[1]['Mooring'][:3] # Mooring ID
            ins_lsa = row[1]['Instrument'] # Instrument
            # Get depth at mooring location from gridded bathymetry
            if not np.isnan(lon_lsa):
                eu, nu, _, _ = utm.from_latlon(lat_lsa, lon_lsa,)
                # Get local x,y coords. from UTM
                angle_rot = np.deg2rad(270 - ang_ref)
                # Shift data relative to origin
                x_aux = eu - originEasting
                y_aux = nu - originNorthing
                # Compute local x,y coordinates
                xl = x_aux*np.cos(angle_rot) + y_aux*np.sin(angle_rot)
                yl = y_aux*np.cos(angle_rot) - x_aux*np.sin(angle_rot)
                # Get nearest depth from gridded bathymetry
                depth = dsb.z_msl.sel(x=xl, y=yl, method='nearest').item()
                # Add to dsb
                dsb['{}'.format(mid_lsa)] = ([], depth)
                dsb['{}'.format(mid_lsa)].attrs['standard_name'] = 'depth'
                dsb['{}'.format(mid_lsa)].attrs['long_name'] = 'LSA mooring {} depth'.format(
                    mid_lsa)
                dsb['{}'.format(mid_lsa)].attrs['instrument'] = ins_lsa
                dsb['{}'.format(mid_lsa)].attrs['latitude'] = lat_lsa
                dsb['{}'.format(mid_lsa)].attrs['longitude'] = lon_lsa
                dsb['{}'.format(mid_lsa)].attrs['x_loc'] = xl
                dsb['{}'.format(mid_lsa)].attrs['y_loc'] = yl

    # TODO: Units and other attributes
    dsb.lat.attrs['standard_name'] = 'latitude'
    dsb.lat.attrs['long_name'] = 'Grid latitude values'
    dsb.lat.attrs['units'] = 'degrees_north'
    dsb.lat.attrs['valid_min'] = -90.0
    dsb.lat.attrs['valid_max'] = 90.0
    dsb.lon.attrs['standard_name'] = 'longitude'
    dsb.lon.attrs['long_name'] = 'Grid longitude values'
    dsb.lon.attrs['units'] = 'degrees_east'
    dsb.lon.attrs['valid_min'] = -180.0
    dsb.lon.attrs['valid_max'] = 180.0
    dsb.x.attrs['standard_name'] = 'projection_x_coordinate'
    dsb.x.attrs['long_name'] = 'Local x coordinates'
    dsb.x.attrs['units'] = 'm'
    dsb.x.attrs['missing_value'] = args.fillvalue
    dsb.y.attrs['standard_name'] = 'projection_y_coordinate'
    dsb.y.attrs['long_name'] = 'Local y coordinates'
    dsb.y.attrs['units'] = 'm'
    dsb.y.attrs['missing_value'] = args.fillvalue
    dsb.z_msl.attrs['standard_name'] = 'depth'
    dsb.z_msl.attrs['long_name'] = 'Water depth relative to NAVD88-0.905m'
    dsb.z_msl.attrs['units'] = 'm'
    dsb.z_msl.attrs['missing_value'] = args.fillvalue

    # Global attributes
    dsb.attrs['title'] = ('ROXSI 2022 Asilomar small-scale array ' + 
                         'bathymetry data. Data curation and gridding ' +
                         'performed by Olavo Marques (SIO & NPS).')
    dsb.attrs['summary'] = ('Combined bathymetry for small-scale array ' + 
                           'site.')
    dsb.attrs['source'] = ('CSUMB, eTracMultibeam, eTracLidar, DiveJet, NOAALidar')
    dsb.attrs['magnetic_declination'] = '12.86 degE'
    dsb.attrs['reference_angle'] = '{} deg'.format(ang_ref)
    dsb.attrs['reference_latitude'] = '{}'.format(lat_ref)
    dsb.attrs['reference_longitude'] = '{}'.format(lon_ref)
    dsb.attrs['Conventions'] = 'CF-1.8'
    dsb.attrs['feature_type'] =  "bathymetry"
    dsb.attrs['date_created'] = str(DT.utcnow()) + ' UTC'
    dsb.attrs['references'] = 'https://github.com/mikapm/pyROXSI'
    dsb.attrs['creator_name'] = "Mika P. Malila"
    dsb.attrs['creator_email'] = "mikapm@unc.edu"
    dsb.attrs['institution'] = "University of North Carolina at Chapel Hill"

    # Set encoding before saving
    encoding = {'x': {'zlib': False, '_FillValue': None},
                'y': {'zlib': False, '_FillValue': None},
                'lat': {'zlib': False, '_FillValue': None},
                'lon': {'zlib': False, '_FillValue': None},
               }     
    # Set variable fill values
    for k in list(dsb.keys()):
        encoding[k] = {'_FillValue': args.fillvalue}

    # Save to netcdf
    dsb.to_netcdf(fn_nc, encoding=encoding)
else:
    dsb = xr.decode_cf(xr.open_dataset(fn_nc, decode_coords='all'))


# Plot bathymetry and mooring locations
fig, ax = plt.subplots(figsize=(6,6))
dsb.z_msl.plot.pcolormesh(x='x', y='y', ax=ax, vmin=-10, vmax=0)
# dsb.z_msl.plot.pcolormesh(x='lon', y='lat', ax=ax, vmin=-8, vmax=0)
# Iterate over mooring IDs and mark their location
for k in dsb.keys():
    if k == 'z_msl':
        continue
    # Plot location of current mooring
    if k[0] == 'X':
        # LAS moorings, use black color
        ax.scatter(x=dsb[k].attrs['x_loc'], y=dsb[k].attrs['y_loc'], 
                   marker='^', color='k')
        pass
    else:
        pass
        # SSA moorings in red
        ax.scatter(x=dsb[k].attrs['x_loc'], y=dsb[k].attrs['y_loc'], 
                   marker='+', color='r')
        # Mooring IDs
        ax.text(x=dsb[k].attrs['x_loc'], y=dsb[k].attrs['y_loc'], 
                s=k, color='k')

plt.tight_layout()
plt.show()



