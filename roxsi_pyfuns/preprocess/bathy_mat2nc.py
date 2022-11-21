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
import numpy as np
import xarray as xr
from scipy.io import loadmat
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
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
    parser.add_argument("--savefig", 
            help=("Make and save figure?"),
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

# Read bathymetry .mat file
fn_mat = os.path.join(args.dr, 'AsilomarAsubset_v2.mat')
mat = loadmat(fn_mat)

FFX = 0 # Fudge Factor in x (East)
FFY = -5 # Fudge Factor in y (North) -5 seems to work well

# UTM coordinates of box corners and middle of rock from GPS 
# measurements of safety sausage from boat. Errors in these are 
# up to several meters due to GPS error and currents
IN = [594490.19 + FFX, 4053805.82 + FFY]
ON = [594478.45 + FFX, 4053815.67 + FFY]
OS = [594470.48 + FFX, 4053808.93 + FFY]
IS = [594477.74 + FFX, 4053799.02 + FFY]
PEAK = [594475.87 + FFX, 4053806.77 + FFY]

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

# New coordinate system that is 0 in SW corner of subset bathymetry
asilomarAsubset_x = mat['asilomarAsubset_x'].squeeze() 
asilomarAsubset_y = mat['asilomarAsubset_y'].squeeze() 
asilomarAsubset_z = mat['asilomarAsubset_z'].squeeze() 
x = asilomarAsubset_x - eastingMin
y = asilomarAsubset_y - northingMin

# Create grid for gridded bathymetry data
xs = np.linspace(0, 40, 81)
ys = np.linspace(0, 40, 71)
Xs, Ys = np.meshgrid(xs, ys)
Zs = griddata((x,y), asilomarAsubset_z, (Xs,Ys))

# compute subsurface buoy locations in new coordinate system
IN = IN - np.array([eastingMin, northingMin])
ON = ON - np.array([eastingMin,northingMin])
IS = IS - np.array([eastingMin,northingMin])
OS = OS - np.array([eastingMin,northingMin])
PEAK = PEAK - np.array([eastingMin,northingMin])

IN_est = IN_est - np.array([eastingMin,northingMin])
ON_est = ON_est - np.array([eastingMin,northingMin])
IS_est = IS_est - np.array([eastingMin,northingMin])
OS_est = OS_est - np.array([eastingMin,northingMin])

# Estimated mooring positions based on bathymetry and orthophoto
# from GoPro
C1_est = np.array([11.1874, 20.0633])
C2_est = np.array([14.8573, 18.1319])
C3_est = np.array([19.7815, 17.1484])
C4_est = np.array([22.8221, 16.4458])
C5_est = np.array([25.6471, 15.7681])
C6_est = np.array([28.4923, 16.5867])
L1_est = np.array([16.9104, 6.5250])
L2_est = np.array([18.3333, 11.2432])
L4_est = np.array([17.8111, 23.7644])
L5_est = np.array([22.94, 26.27])


# Plot bathymetry
if args.savefig:
    # Figure filename
    fn_fig = os.path.join(args.out, 'bathy.pdf')
    if not os.path.isfile(fn_fig) or args.overwrite_fig:
        fig, axes = plt.subplots(figsize=(16,8), ncols=2, sharey=True,
                                constrained_layout=True)
        axes[0].scatter(x, y, c=asilomarAsubset_z)
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
        
        cf = axes[1].contourf(Xs, Ys, Zs, vmin=-7.5, vmax=-4.0)
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

        axes[1].plot(C1_est[0], C1_est[1], marker='*', color='r')
        axes[1].plot(C6_est[0], C6_est[1], marker='*', color='r')
        axes[1].plot(C2_est[0], C2_est[1], marker='*', color='r')
        axes[1].plot(C3_est[0], C3_est[1], marker='*', color='r')
        axes[1].plot(C4_est[0], C4_est[1], marker='*', color='r')
        axes[1].plot(C5_est[0], C5_est[1], marker='*', color='r')
        axes[1].plot(C6_est[0], C6_est[1], marker='*', color='r')
        axes[1].plot(L1_est[0], L1_est[1], marker='*', color='r')
        axes[1].plot(L2_est[0], L2_est[1], marker='*', color='r')
        axes[1].plot(L4_est[0], L4_est[1], marker='*', color='r')
        axes[1].plot(L5_est[0], L5_est[1], marker='*', color='r')

        for ax in axes:
            ax.set_xlim([0, 40])
            ax.set_ylim([0, 35])

        plt.show()
        plt.savefig(fn_fig, bbox_inches='tight', dpi=300)
        plt.close()

# Save bathymetry to netcdf
fn_nc = os.path.join(args.out, 'Asilomar2022_SSA_bathy.nc')
if not os.path.isfile(fn_nc) or args.overwrite_nc:
    # Generate xr.Dataset
    N = len(asilomarAsubset_z)
    data_vars = {'x_pts': (['index'], asilomarAsubset_x),
                 'y_pts': (['index'], asilomarAsubset_y),
                 'z_pts': (['index'], asilomarAsubset_z),
                 'z_grd': (['Y', 'X'], Zs),
                 'C1_est_x': ([], C1_est[0]),
                 'C1_est_y': ([], C1_est[1]),
                 'C2_est_x': ([], C2_est[0]),
                 'C2_est_y': ([], C2_est[1]),
                 'C3_est_x': ([], C3_est[0]),
                 'C3_est_y': ([], C3_est[1]),
                 'C4_est_x': ([], C4_est[0]),
                 'C4_est_y': ([], C4_est[1]),
                 'C5_est_x': ([], C5_est[0]),
                 'C5_est_y': ([], C5_est[1]),
                 'C6_est_x': ([], C6_est[0]),
                 'C6_est_y': ([], C6_est[1]),
                 'L1_est_x': ([], L1_est[0]),
                 'L1_est_y': ([], L1_est[1]),
                 'L2_est_x': ([], L2_est[0]),
                 'L2_est_y': ([], L2_est[1]),
                 'L4_est_x': ([], L4_est[0]),
                 'L4_est_y': ([], L4_est[1]),
                 'L5_est_x': ([], L5_est[0]),
                 'L5_est_y': ([], L5_est[1]),
                }
    ds = xr.Dataset(data_vars=data_vars,
                    coords={'index': (['index'], np.arange(N)),
                            'X': (['X'], xs),
                            'Y': (['Y'], ys),
                           }
                   )
    # TODO: Units and other attributes

    # Save to netcdf
    ds.to_netcdf(fn_nc)
else:
    ds = xr.decode_cf(xr.open_dataset(fn_nc, decode_coords='all'))