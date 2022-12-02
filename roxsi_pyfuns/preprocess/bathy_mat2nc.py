"""
Convert Asilomar SSA bathymetry file from .mat to netcdf.
Based on Asilomar_bathy_SSarray.m script by Johanna Rosman.

From the original docstring:

    GPS coordinates of corner points, and the highest point 
    (taken from boat using a safety sausage sent to the surface)

    Corner points (WGS84)
    62: 36.62511 deg N, 121.94321 deg W; time 10:27
    63: 36.62520 deg N, 121.94334 deg W; time 10:33
    64: 36.62514 deg N, 121.94343 deg W; time 10:38
    65: 36.62505 deg N, 121.94335 deg W; time 10:46

    Highest point
    66: 36.62512 deg N, 121.94337 deg W; time 10:51

    Corner points converted to UTM - Var names use labels 
    combining I-Inshore, O-Offshore, N-North, S-South

    Since the points don't match up well with the bathymetry, try
    sliding the points around a bit to get a better match. Error
    in GPS measurements could by up to 3 m due to GPS and about 
    another 4 m due to currents acting on safety sausage.
    FFX = 0;   % Fudge Factor in x
    FFY = 0;   % Fudge Factor in y
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
from scipy.io import loadmat
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
fn_mat = os.path.join(args.dr, 'AsilomarAsubset_v2.mat')
mat = loadmat(fn_mat)

# Read large-scale array mooring locations table
fn_lsa = os.path.join(args.dr, 'ROXSI2022_LSA_mooring_locations.csv')
df_lsa = pd.read_csv(fn_lsa).set_index('Mooring')
# Fill missing values w/ NaN
df_lsa = df_lsa.replace(9999999999.0, np.nan)

utm_zone = 10 # Standard UTM zone number (northern hemisphere)
FFX = 0 # Fudge Factor in x (East)
FFY = -5 # Fudge Factor in y (North) -5 seems to work well

# UTM coordinates of box corners and middle of rock from GPS 
# measurements of safety sausage from boat. Errors in these are 
# up to several meters due to GPS error and currents
IN = [594490.19 + FFX, 4053805.82 + FFY] # Inshore North
ON = [594478.45 + FFX, 4053815.67 + FFY] # Offshore North
OS = [594470.48 + FFX, 4053808.93 + FFY] # Offshore South
IS = [594477.74 + FFX, 4053799.02 + FFY] # Inshore South
PEAK = [594475.87 + FFX, 4053806.77 + FFY] # Middle of rock

# UTM coordinates of box corners and middle of rock, 
# approximated by eye based on estimated positions relative 
# to bathymetric features 
IS_est = np.array([0.594476227663551, 4.053794712172897]) * 1e6
OS_est = np.array([0.594470871355140, 4.053806078995327]) * 1e6
IN_est = np.array([0.594488330467290, 4.053799782266355]) * 1e6
ON_est = np.array([0.594479294252336, 4.053812580163552]) * 1e6

# Calculate line segment lengths
ON_IN = np.sqrt((ON[0]-IN[0])**2 + (ON[1]-IN[1])**2)
IN_IS = np.sqrt((IN[0]-IS[0])**2 + (IN[1]-IS[1])**2)
IS_OS = np.sqrt((IS[0]-OS[0])**2 + (IS[1]-OS[1])**2)
OS_ON = np.sqrt((OS[0]-ON[0])**2 + (OS[1]-ON[1])**2)

ON_IN_est = np.sqrt((ON_est[0]-IN_est[0])**2 + 
                    (ON_est[1]-IN_est[1])**2)
IN_IS_est = np.sqrt((IN_est[0]-IS_est[0])**2 + 
                    (IN_est[1]-IS_est[1])**2)
IS_OS_est = np.sqrt((IS_est[0]-OS_est[0])**2 + 
                    (IS_est[1]-OS_est[1])**2)
OS_ON_est = np.sqrt((OS_est[0]-ON_est[0])**2 + 
                    (OS_est[1]-ON_est[1])**2)
   
print('\nLine segment lengths from GPS measurements\n')
print('ON_IN: {:.2f} m measured with transect tape to be: 15 m\n'.format(ON_IN))
print('IN_IS: {:.2f} m measured with transect tape to be: 17 m, but passes high over rock \n'.format(IN_IS))
print('IS_OS: {:.2f} m measured with transect tape to be: 15 m, but passes over rock\n'.format(IS_OS))
print('OS_ON: {:.2f} m measured with transect tape to be: 10 m\n'.format(OS_ON))

print('\nLine segment lengths from positions estimated from bathymetry\n')
print('ON_IN_est: {:.2f} m measured with transect tape to be: 15 m\n'.format(ON_IN_est))
print('IN_IS_est: {:.2f} m measured with transect tape to be: 17 m, but passes high over rock \n'.format(IN_IS_est))
print('IS_OS_est: {:.2f} m measured with transect tape to be: 15 m, but passes over rock\n'.format(IS_OS_est))
print('OS_ON_est: {:.2f} m measured with transect tape to be: 10 m\n'.format(OS_ON_est))

# Select subset of bathymetry data in region around small-scale 
# array. Make a box that lines up with UTM of subsurface buoys 
# with a buffer region around it. 
BUFFERREGION = 10  # size of buffer region beyond coords of subsurface buys
eastingMin = min(np.array([IN[0],ON[0],OS[0],IS[0]]) - BUFFERREGION)
eastingMax = max(np.array([IN[0],ON[0],OS[0],IS[0]]) + BUFFERREGION)
northingMin = min(np.array([IN[1],ON[1],OS[1],IS[1]]) - BUFFERREGION)
northingMax = max(np.array([IN[1],ON[1],OS[1],IS[1]]) + BUFFERREGION)
# Min, Max lon & lat
latMin, lonMin = utm.to_latlon(eastingMin, northingMin, 
                               zone_number=utm_zone, northern=True)
latMax, lonMax = utm.to_latlon(eastingMax, northingMax, 
                               zone_number=utm_zone, northern=True)

# New coordinate system that is 0 in SW corner of subset bathymetry
asilomarAsubset_x = mat['asilomarAsubset_x'].squeeze() 
asilomarAsubset_y = mat['asilomarAsubset_y'].squeeze() 
asilomarAsubset_z = mat['asilomarAsubset_z'].squeeze() 
x = asilomarAsubset_x - eastingMin
y = asilomarAsubset_y - northingMin

# Create grid for gridded bathymetry data in local coordinate system
nx = 81
xsl = np.linspace(0, 40, nx)
ny = 71
ysl = np.linspace(0, 35, ny)
Xsl, Ysl = np.meshgrid(xsl, ysl)
# Interpolate topography to Xs, Ys grid
Zsl = griddata((x,y), asilomarAsubset_z, (Xsl,Ysl))

# Create grid for gridded bathymetry data in UTM coordinates
xsu = np.linspace(asilomarAsubset_x.min(), asilomarAsubset_x.max(), nx)
dx = xsu[1] - xsu[0]
ysu = np.linspace(asilomarAsubset_y.min(), asilomarAsubset_y.max(), ny)
dy = ysu[1] - ysu[0]
Xsu, Ysu = np.meshgrid(xsu, ysu)
# Interpolate topography to Xs, Ys grid
Zsu = griddata((asilomarAsubset_x, asilomarAsubset_y), asilomarAsubset_z, 
               (Xsu,Ysu))

# Also convert grids to lat, lon
latr, lonr = utm.to_latlon(asilomarAsubset_x, asilomarAsubset_y,
                           zone_number=utm_zone, northern=True)
lat = np.linspace(latMin, latMax, nx)
lon = np.linspace(lonMin, lonMax, ny)
lats, lons = utm.to_latlon(Xsu, Ysu, zone_number=utm_zone, 
                           northern=True)
Zll = griddata((latr, lonr), asilomarAsubset_z, (lats, lons))

# compute subsurface buoy locations in local coordinate system
INl = IN - np.array([eastingMin, northingMin])
ONl = ON - np.array([eastingMin,northingMin])
ISl = IS - np.array([eastingMin,northingMin])
OSl = OS - np.array([eastingMin,northingMin])
PEAKl = PEAK - np.array([eastingMin,northingMin])

IN_est_l = IN_est - np.array([eastingMin,northingMin])
ON_est_l = ON_est - np.array([eastingMin,northingMin])
IS_est_l = IS_est - np.array([eastingMin,northingMin])
OS_est_l = OS_est - np.array([eastingMin,northingMin])

# Estimated mooring positions based on bathymetry and orthophoto
# from GoPro (in local coordinate system)
C1_loc = np.array([11.1874, 20.0633])
C2_loc = np.array([14.8573, 18.1319])
C3_loc = np.array([19.7815, 17.1484])
C4_loc = np.array([22.8221, 16.4458])
C5_loc = np.array([25.6471, 15.7681])
C6_loc = np.array([28.4923, 16.5867])
L1_loc = np.array([16.9104, 6.5250])
L2_loc = np.array([18.3333, 11.2432])
# L4_loc = np.array([17.8111, 23.7644]) # Old est.
L4_loc = np.array([16.9881, 23.9993])
L5_loc = np.array([22.94, 26.27])
C2L2_loc = np.array([16.2667, 21.8259])
C2L4_loc = np.array([14.1926, 15.3074])
C4L2_loc = np.array([22.0741, 12.7593])
C4L4_loc = np.array([23.5556, 20.0481])
C1_5_loc = np.array([12.8544, 18.5786])
M1_loc = np.array([12.4357, 27.7833])

# Estimated mooring locations in UTM coordinates
C1_utm = C1_loc + [eastingMin, northingMin]
C2_utm = C2_loc + [eastingMin, northingMin]
C3_utm = C3_loc + [eastingMin, northingMin]
C4_utm = C4_loc + [eastingMin, northingMin]
C5_utm = C5_loc + [eastingMin, northingMin]
C6_utm = C6_loc + [eastingMin, northingMin]
L1_utm = L1_loc + [eastingMin, northingMin]
L2_utm = L2_loc + [eastingMin, northingMin]
L4_utm = L4_loc + [eastingMin, northingMin]
L5_utm = L5_loc + [eastingMin, northingMin]
C2L2_utm = C2L2_loc + [eastingMin, northingMin]
C2L4_utm = C2L4_loc + [eastingMin, northingMin]
C4L2_utm = C4L2_loc + [eastingMin, northingMin]
C4L4_utm = C4L4_loc + [eastingMin, northingMin]
C1_5_utm = C1_5_loc + [eastingMin, northingMin]
M1_utm = M1_loc + [eastingMin, northingMin]

# Estimated mooring locations in lat, lon coordinates
# (UTM coords calculated by C1_loc[0]+eastingMin, 
# C1_loc[1]+northingMin, and then converted
# from UTM to lat, lon (zone 10N) using converter at
# http://rcn.montana.edu/resources/Converter.aspx)
C1_llc = np.array([36.62509622046095, -121.94341728936496])
C2_llc = np.array([36.6250784479023, -121.94337648871583])
C3_llc = np.array([36.62506909478911, -121.94332154537203])
C4_llc = np.array([36.62506246035691, -121.94328763063506])
C5_llc = np.array([36.62505607173514, -121.94325612377533])
C6_llc = np.array([36.62506316794421, -121.9432242068824])
L1_llc = np.array([36.62497362592774, -121.9433549582822])
L2_llc = np.array([36.62501601212474, -121.94333846632198])
# L4_llc = np.array([36.62512892, -121.9433428]) # Old est.
L4_llc = np.array([36.62513112216428, -121.94335193932969])
L5_llc = np.array([36.62515099871671, -121.94328510346281])
C2L2_llc = np.array([36.62511160385235, -121.94336027372623])
C2L4_llc = np.array([36.62505305530001, -121.94338426919181])
C4L2_llc = np.array([36.62502930641834, -121.9432964486891])
C4L4_llc = np.array([36.625094856799286, -121.94327898509006])
C1_5_llc = np.array([36.62508267285272, -121.94339883098975])
M1_llc = np.array([36.62516568059234, -121.94340238052408])


# Save bathymetry to netcdf
fn_nc = os.path.join(args.out, 'Asilomar_2022_SSA_bathy.nc')
if not os.path.isfile(fn_nc) or args.overwrite_nc:
    N = len(asilomarAsubset_z)
    # Define variables dict
    data_vars = {'x_pts': (['index'], asilomarAsubset_x),
                 'y_pts': (['index'], asilomarAsubset_y),
                 'z_pts': (['index'], asilomarAsubset_z),
                 'z_utm': (['northings', 'eastings'], Zsu),
                 'z_llc': (['lon', 'lat'], Zll),
                 # SSA mooring locations in UTM
                 'C1_utm': (['utm'], C1_utm),
                 'C2_utm': (['utm'], C2_utm),
                 'C3_utm': (['utm'], C3_utm),
                 'C4_utm': (['utm'], C4_utm),
                 'C5_utm': (['utm'], C5_utm),
                 'C6_utm': (['utm'], C6_utm),
                 'L1_utm': (['utm'], L1_utm),
                 'L2_utm': (['utm'], L2_utm),
                 'L4_utm': (['utm'], L4_utm),
                 'L5_utm': (['utm'], L5_utm),
                 'C2L2_utm': (['utm'], C2L2_utm),
                 'C2L4_utm': (['utm'], C2L4_utm),
                 'C4L2_utm': (['utm'], C4L2_utm),
                 'C4L4_utm': (['utm'], C4L4_utm),
                 'C1_5_utm': (['utm'], C1_5_utm),
                 'M1_utm': (['utm'], M1_utm),
                 # SSA mooring locations in lat, lon
                 'C1_llc': (['llc'], C1_llc),
                 'C2_llc': (['llc'], C2_llc),
                 'C3_llc': (['llc'], C3_llc),
                 'C4_llc': (['llc'], C4_llc),
                 'C5_llc': (['llc'], C5_llc),
                 'C6_llc': (['llc'], C6_llc),
                 'L1_llc': (['llc'], L1_llc),
                 'L2_llc': (['llc'], L2_llc),
                 'L4_llc': (['llc'], L4_llc),
                 'L5_llc': (['llc'], L5_llc),
                 'C2L2_llc': (['llc'], C2L2_llc),
                 'C2L4_llc': (['llc'], C2L4_llc),
                 'C4L2_llc': (['llc'], C4L2_llc),
                 'C4L4_llc': (['llc'], C4L4_llc),
                 'C1_5_llc': (['llc'], C1_5_llc),
                 'M1_llc': (['llc'], M1_llc),
                }
    # Generate xr.Dataset
    dsb = xr.Dataset(data_vars=data_vars,
                    coords={'index': (['index'], np.arange(N).astype('i4')),
                            'eastings': (['eastings'], xsu),
                            'northings': (['northings'], ysu),
                            'lat': (['lat'], lat),
                            'lon': (['lon'], lon),
                            'utm': (['utm'], ['easting', 'northing']),
                            'llc': (['llc'], ['latitude', 'longitude']),
                           }
                   )
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
    dsb.x_pts.attrs['standard_name'] = 'projection_x_coordinate'
    dsb.x_pts.attrs['long_name'] = 'UTM eastings coordinates of eTrac bathymetry points'
    dsb.x_pts.attrs['units'] = 'm'
    dsb.x_pts.attrs['missing_value'] = args.fillvalue
    dsb.y_pts.attrs['standard_name'] = 'projection_y_coordinate'
    dsb.y_pts.attrs['long_name'] = 'UTM northings coordinates of eTrac bathymetry points'
    dsb.y_pts.attrs['units'] = 'm'
    dsb.y_pts.attrs['missing_value'] = args.fillvalue
    dsb.z_pts.attrs['standard_name'] = 'depth'
    dsb.z_pts.attrs['long_name'] = 'eTracs water depth points.'
    dsb.z_pts.attrs['units'] = 'm'
    dsb.z_pts.attrs['missing_value'] = args.fillvalue
    dsb.z_utm.attrs['standard_name'] = 'depth'
    dsb.z_utm.attrs['long_name'] = 'Gridded eTracs water depth points interpolated to UTM coordinates.'
    dsb.z_utm.attrs['units'] = 'm'
    dsb.z_utm.attrs['missing_value'] = args.fillvalue
    dsb.z_llc.attrs['standard_name'] = 'depth'
    dsb.z_llc.attrs['long_name'] = 'Gridded eTracs water depth points interpolated to lat,lon coordinates.'
    dsb.z_llc.attrs['units'] = 'm'
    dsb.z_llc.attrs['missing_value'] = args.fillvalue
    dsb.C1_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C1_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C2_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C2_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C3_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C3_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C4_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C4_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C5_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C5_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C6_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C6_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.L1_utm.attrs['standard_name'] = 'mooring_location'
    dsb.L1_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.L2_utm.attrs['standard_name'] = 'mooring_location'
    dsb.L2_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.L4_utm.attrs['standard_name'] = 'mooring_location'
    dsb.L4_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.L5_utm.attrs['standard_name'] = 'mooring_location'
    dsb.L5_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C2L2_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C2L2_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C2L4_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C2L4_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C4L2_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C4L2_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C4L4_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C4L4_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C1_5_utm.attrs['standard_name'] = 'mooring_location'
    dsb.C1_5_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.M1_utm.attrs['standard_name'] = 'mooring_location'
    dsb.M1_utm.attrs['long_name'] = 'SSA mooring location (easting, northing)'
    dsb.C1_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C1_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C2_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C2_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C3_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C3_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C4_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C4_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C5_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C5_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C6_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C6_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.L1_llc.attrs['standard_name'] = 'mooring_location'
    dsb.L1_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.L2_llc.attrs['standard_name'] = 'mooring_location'
    dsb.L2_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.L4_llc.attrs['standard_name'] = 'mooring_location'
    dsb.L4_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.L5_llc.attrs['standard_name'] = 'mooring_location'
    dsb.L5_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C2L2_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C2L2_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C2L4_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C2L4_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C4L2_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C4L2_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C4L4_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C4L4_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.C1_5_llc.attrs['standard_name'] = 'mooring_location'
    dsb.C1_5_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'
    dsb.M1_llc.attrs['standard_name'] = 'mooring_location'
    dsb.M1_llc.attrs['long_name'] = 'SSA mooring location (latitude, longitude)'

    # Global attributes
    dsb.attrs['title'] = ('ROXSI 2022 Asilomar small-scale array ' + 
                         'bathymetry data.')
    dsb.attrs['summary'] = ('eTrac bathymetry for small-scale array ' + 
                           'site. File contains both individual ' +
                           'data points (x_pts, y_pts, z_pts) ' + 
                           'and interpolated grid. Default coordinates ' +
                           'are standard UTM (zone 10). Mooring ' +
                           'locations are given in both UTM and lat, lon.')
    dsb.attrs['gridEastingMin'] = eastingMin
    dsb.attrs['gridEastingMax'] = eastingMax
    dsb.attrs['gridNorthingMin'] = northingMin
    dsb.attrs['gridNorthingMax'] = northingMax
    dsb.attrs['gridLonMin'] = lonMin
    dsb.attrs['gridLonMax'] = lonMax
    dsb.attrs['gridLatMin'] = lonMin
    dsb.attrs['gridLatMax'] = lonMax
    dsb.attrs['magnetic_declination'] = '12.86 degE'
    dsb.attrs['utm_zone'] = '{} N'.format(utm_zone)
    dsb.attrs['Conventions'] = 'CF-1.8'
    dsb.attrs['feature_type'] =  "bathymetry"
    dsb.attrs['source'] =  "Bathymetric survey"
    dsb.attrs['date_created'] = str(DT.utcnow()) + ' UTC'
    dsb.attrs['references'] = 'https://github.com/mikapm/pyROXSI'
    dsb.attrs['creator_name'] = "Mika P. Malila"
    dsb.attrs['creator_email'] = "mikapm@unc.edu"
    dsb.attrs['institution'] = "University of North Carolina at Chapel Hill"

    # Set encoding before saving
    encoding = {'lat': {'zlib': False, '_FillValue': None},
                'lon': {'zlib': False, '_FillValue': None},
                'index': {'zlib': False, '_FillValue': None},
                'eastings': {'zlib': False, '_FillValue': None},
                'northings': {'zlib': False, '_FillValue': None},
                'x_pts': {'_FillValue': args.fillvalue},        
                'y_pts': {'_FillValue': args.fillvalue},        
                'z_pts': {'_FillValue': args.fillvalue},        
                'z_utm': {'_FillValue': args.fillvalue},        
                'z_llc': {'_FillValue': args.fillvalue},        
                'C1_utm': {'_FillValue': args.fillvalue},        
                'C2_utm': {'_FillValue': args.fillvalue},        
                'C3_utm': {'_FillValue': args.fillvalue},        
                'C4_utm': {'_FillValue': args.fillvalue},        
                'C5_utm': {'_FillValue': args.fillvalue},        
                'C6_utm': {'_FillValue': args.fillvalue},        
                'L1_utm': {'_FillValue': args.fillvalue},        
                'L2_utm': {'_FillValue': args.fillvalue},        
                'L4_utm': {'_FillValue': args.fillvalue},        
                'L5_utm': {'_FillValue': args.fillvalue},        
                'C2L2_utm': {'_FillValue': args.fillvalue},        
                'C2L4_utm': {'_FillValue': args.fillvalue},        
                'C4L2_utm': {'_FillValue': args.fillvalue},        
                'C4L4_utm': {'_FillValue': args.fillvalue},        
                'C1_5_utm': {'_FillValue': args.fillvalue},        
                'M1_utm': {'_FillValue': args.fillvalue},        
                'C1_llc': {'_FillValue': args.fillvalue},        
                'C2_llc': {'_FillValue': args.fillvalue},        
                'C3_llc': {'_FillValue': args.fillvalue},        
                'C4_llc': {'_FillValue': args.fillvalue},        
                'C5_llc': {'_FillValue': args.fillvalue},        
                'C6_llc': {'_FillValue': args.fillvalue},        
                'L1_llc': {'_FillValue': args.fillvalue},        
                'L2_llc': {'_FillValue': args.fillvalue},        
                'L4_llc': {'_FillValue': args.fillvalue},        
                'L5_llc': {'_FillValue': args.fillvalue},        
                'C2L2_llc': {'_FillValue': args.fillvalue},        
                'C2L4_llc': {'_FillValue': args.fillvalue},        
                'C4L2_llc': {'_FillValue': args.fillvalue},        
                'C4L4_llc': {'_FillValue': args.fillvalue},        
                'C1_5_llc': {'_FillValue': args.fillvalue},        
                'M1_llc': {'_FillValue': args.fillvalue},
               }     

    # Save to netcdf
    dsb.to_netcdf(fn_nc, encoding=encoding)
else:
    dsb = xr.decode_cf(xr.open_dataset(fn_nc, decode_coords='all'))


# Plot raw eTrac data vs. interpolated grid
fn_fig = os.path.join(args.out, 'Asilomar_2022_SSA_bathy.pdf')
# Plot bathymetry
if not os.path.isfile(fn_fig) or args.overwrite_fig:
    fig, axes = plt.subplots(figsize=(15,6.25), ncols=2, sharex=True,
                                    sharey=True, constrained_layout=True)
    axes[0].scatter(asilomarAsubset_x, asilomarAsubset_y, 
                    c=asilomarAsubset_z)
    axes[0].plot(IN[0],IN[1], marker='*', color='r') # GPS measurements
    axes[0].plot(ON[0],ON[1], marker='*', color='r')
    axes[0].plot(OS[0],OS[1], marker='*', color='r')
    axes[0].plot(IS[0],IS[1], marker='*', color='r')
    axes[0].plot(PEAK[0],PEAK[1], marker='*', color='r')
    axes[0].plot(IN_est[0],IN_est[1], marker='*', color='k') # estimated from bathymetry
    axes[0].plot(ON_est[0],ON_est[1], marker='*', color='k')
    axes[0].plot(OS_est[0],OS_est[1], marker='*', color='k')
    axes[0].plot(IS_est[0],IS_est[1], marker='*', color='k')
    axes[0].set_xlabel('distance East (m)')
    axes[0].set_ylabel('distance North (m)')
    
    # Plot interpolated bathymetry
    cf = axes[1].contourf(Xsu, Ysu, Zsu, vmin=-7.5, vmax=-4.0)
    axes[1].plot(IN[0],IN[1]) # GPS measurements
    axes[1].plot(ON[0],ON[1])
    axes[1].plot(OS[0],OS[1])
    axes[1].plot(IS[0],IS[1])
    axes[1].plot(PEAK[0],PEAK[1])
    axes[1].plot(IN_est[0],IN_est[1]) # estimated from bathymetry
    axes[1].plot(ON_est[0],ON_est[1])
    axes[1].plot(OS_est[0],OS_est[1])
    axes[1].plot(IS_est[0],IS_est[1])
    axes[1].set_xlabel('distance East (m)')
    
    axes[1].plot(C1_utm[0], C1_utm[1], marker='*', color='r')
    axes[1].plot(C6_utm[0], C6_utm[1], marker='*', color='r')
    axes[1].plot(C2_utm[0], C2_utm[1], marker='*', color='r')
    axes[1].plot(C3_utm[0], C3_utm[1], marker='*', color='r')
    axes[1].plot(C4_utm[0], C4_utm[1], marker='*', color='r')
    axes[1].plot(C5_utm[0], C5_utm[1], marker='*', color='r')
    axes[1].plot(C6_utm[0], C6_utm[1], marker='*', color='r')
    axes[1].plot(L1_utm[0], L1_utm[1], marker='*', color='r')
    axes[1].plot(L2_utm[0], L2_utm[1], marker='*', color='r')
    axes[1].plot(L4_utm[0], L4_utm[1], marker='*', color='r')
    axes[1].plot(L5_utm[0], L5_utm[1], marker='*', color='r')
    axes[1].plot(C2L2_utm[0], C2L2_utm[1], marker='*', color='r')
    axes[1].plot(C2L4_utm[0], C2L4_utm[1], marker='*', color='r')
    axes[1].plot(C4L2_utm[0], C4L2_utm[1], marker='*', color='r')
    axes[1].plot(C4L4_utm[0], C4L4_utm[1], marker='*', color='r')
    axes[1].plot(C1_5_utm[0], C1_5_utm[1], marker='*', color='r')
    axes[1].plot(M1_utm[0], M1_utm[1], marker='*', color='r')
    
    for ax in axes:
            ax.set_xlim([asilomarAsubset_x.min(), asilomarAsubset_x.max()])
            ax.set_ylim([asilomarAsubset_y.min(), asilomarAsubset_y.max()])
    axes[0].set_title('Raw bathymetry')
    axes[1].set_title('Interpolated bathymetry')
    
    plt.savefig(fn_fig, bbox_inches='tight', dpi=300)
    plt.close()


# Plot w/ google satellite image following example from
# https://salem.readthedocs.io/en/stable/auto_examples/plot_googlestatic.html
fn_sat = os.path.join(args.out, 'Asilomar_2022_SSA_bathy_satellite.pdf')
if not os.path.isfile(fn_sat) or args.overwrite_fig:
    vmin = -8.0
    vmax = -3.5
    fig, axes = plt.subplots(figsize=(15,6.25), ncols=2, 
                             constrained_layout=True)

    # Define background boundaries for plot (P)
    buffer = 140 # Buffer distance (m)
    latMinP, lonMinP = utm.to_latlon(eastingMin-buffer, northingMin-buffer, 
                            zone_number=utm_zone, northern=True)
    latMaxP, lonMaxP = utm.to_latlon(eastingMax+buffer, northingMax+buffer, 
                            zone_number=utm_zone, northern=True)

    # If you need to do a lot of maps you might want
    # to use an API key and set it here with key='YOUR_API_KEY'
    g = salem.GoogleVisibleMap(x=[lonMinP, lonMaxP], y=[latMinP, latMaxP],
                            scale=2,  # scale is for more details
                            maptype='satellite',
                            key=api_key)  # try out also: 'terrain'
    # The google static image is a standard rgb image
    ggl_img = g.get_vardata()
    # ax1.imshow(ggl_img)
    # ax1.set_title('Google static map')

    # Make a map of the same size as the image (no country borders)
    sm = salem.Map(g.grid, factor=2, countries=False)
    sm.set_shapefile()  # add the glacier outlines
    sm.set_rgb(ggl_img)  # add the background rgb image
    sm.set_scale_bar(location=(0.88, 0.94))  # add scale
    sm.visualize(ax=axes[0])  # plot it

    # Define UTM WGS84 projection for SSA grid
    proj_utm = Proj(proj='utm', zone=utm_zone, ellps='WGS84', 
                    preserve_units=False)
    # Make salem grid object with UTM projection (not really needed)
    grid = salem.Grid(nxny=(nx, ny),
                    dxdy=(dx, dy), 
                    x0y0=(eastingMin, northingMin), 
                    proj=proj_utm)
    # Map grid on top of satellite image
    smap = salem.Map(grid)
    # Get UTM coordinates from grid (almost the same as Xsu, Ysu)
    xG, yG = grid.xy_coordinates
    # Transform SSA grid coordinates to google map projection
    # xx, yy = sm.grid.transform(xG, yG, crs=p)
    xx, yy = sm.grid.transform(Xsu, Ysu, crs=proj_utm)
    axes[0].contourf(xx, yy, Zsu, cmap=cmocean.cm.deep_r, 
                     vmin=vmin, vmax=vmax)
    # Mark large-scale array mooring locations
    for row in df_lsa.iterrows():
        if row[1]['Array'] == 'Asilomar':
            # Plot mooring location
            lon = row[1]['Deployed longitude']
            lat = row[1]['Deployed latitude']
            # Convert to UTM
            xp, yp = proj_utm(lon, lat)
            # Convert UTM coordinates to google image projection
            xp, yp = sm.grid.transform(xp, yp, crs=proj_utm)
            axes[0].scatter(xp, yp, marker='+', color='r', s=50)

    # On second axis, plot closeup of SSA grid with mooring locations marked
    # axes[1].contourf(dsb.eastings, dsb.northings, dsb.z_utm, 
    #                  vmin=-7, vmax=-4.5, cmap=cmocean.cm.deep)
    dl = salem.DataLevels(dsb.z_llc, extend='both', cmap=cmocean.cm.deep_r,                      
                    levels=np.linspace(vmin, vmax, 10), )
    # axes[1].contourf(lons, lats, dsb.z_llc, 
    #                  vmin=vmin, vmax=vmax, cmap=cmocean.cm.deep_r)
    axes[1].contourf(dsb.eastings, dsb.northings, dsb.z_utm, 
                    vmin=vmin, vmax=vmax, cmap=cmocean.cm.deep_r)
    dl.append_colorbar(axes[1], label='Depth [m]')
    # SSA Mooring locations
    axes[1].scatter(dsb.C1_utm[0].item(), dsb.C1_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.C2_utm[0].item(), dsb.C2_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.C3_utm[0].item(), dsb.C3_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.C4_utm[0].item(), dsb.C4_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.C5_utm[0].item(), dsb.C5_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.C6_utm[0].item(), dsb.C6_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.L1_utm[0].item(), dsb.L1_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.L2_utm[0].item(), dsb.L2_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.L4_utm[0].item(), dsb.L4_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.L5_utm[0].item(), dsb.L5_utm[1].item(), marker='+', color='r', s=60)
    axes[1].scatter(dsb.C2L2_utm[0].item(), dsb.C2L2_utm[1].item(), marker='^', color='k', s=50)
    axes[1].scatter(dsb.C2L4_utm[0].item(), dsb.C2L4_utm[1].item(), marker='^', color='k', s=50)
    axes[1].scatter(dsb.C4L2_utm[0].item(), dsb.C4L2_utm[1].item(), marker='^', color='k', s=50)
    axes[1].scatter(dsb.C4L4_utm[0].item(), dsb.C4L4_utm[1].item(), marker='^', color='k', s=50)

    axes[1].grid(alpha=0.5)
    axes[1].set_ylabel('Northings [m]')
    axes[1].set_xlabel('Eastings [m]')

    # Show/save plot
    # plt.tight_layout()
    plt.savefig(fn_sat, bbox_inches='tight', dpi=300)
    plt.close()