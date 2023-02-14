"""
Plot comparisons of wave spectra from Signatures.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean
from tqdm import tqdm
from argparse import ArgumentParser
from PyPDF2 import PdfFileMerger, PdfFileReader
from roxsi_pyfuns import wave_spectra as rpws
from roxsi_pyfuns import adcp_funs as rpaf
from roxsi_pyfuns import zero_crossings as rpzc
from roxsi_pyfuns import transfer_functions as rptf
from roxsi_pyfuns import plotting as rppl

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
    parser.add_argument("-d0", 
            help=('Analysis period start date in format "yyyymmdd".'),
            type=str,
            default='20220705',
            )
    parser.add_argument("-d1", 
            help=('Analysis period end date in format "yyyymmdd".'),
            type=str,
            default='20220714',
            )
    parser.add_argument("-t0", 
            help=('Analysis period start time in format "HHMMSS".'),
            type=str,
            default='000000',
            )
    parser.add_argument("-t1", 
            help=('Analysis period end time in format "HHMMSS".'),
            type=str,
            default='235959',
            )

    return parser.parse_args(**kwargs)

# Call args parser to create variables out of input arguments
args = parse_args(args=sys.argv[1:])

# Level 1 data directory
data_root = os.path.join(args.dr, 'Level1')

# Sig100 serial number to visualize
veldir = os.path.join(data_root, '{}'.format(args.ser)) # Velocity netcdf directory
specdir = os.path.join(data_root, '{}'.format(args.ser), 'Spectra') # Spectra netcdf directory

# Output dirs
spec_outdir = os.path.join(specdir, 'img')
if not os.path.isdir(spec_outdir):
    print('Making output figure directory {}'.format(spec_outdir))
    os.mkdir(spec_outdir)
img_outdir = os.path.join(veldir, 'img')
if not os.path.isdir(img_outdir):
    print('Making output figure directory {}'.format(img_outdir))
    os.mkdir(img_outdir)

# List all Level 1 velocity netcdf files
fns_v = sorted(glob.glob(os.path.join(veldir, 'Asilomar_*.nc')))
fns_s = sorted(glob.glob(os.path.join(specdir, 'Asilomar_*.nc')))

# Define start timestamp for requested time period
y0 = args.d0[:4]
mon0 = args.d0[4:6]
d0 = args.d0[6:]
h0 = args.t0[:2]
min0 = args.t0[2:4]
s0 = args.t0[4:]
# Start timestamp
t0 = pd.Timestamp('{}-{}-{} {}:{}:{}'.format(y0, mon0, d0, h0, min0, s0))
# Define end timestamp
y1 = args.d1[:4]
mon1 = args.d1[4:6]
d1 = args.d1[6:]
h1 = args.t1[:2]
min1 = args.t1[2:4]
s1 = args.t1[4:]
# End timestamp
t1 = pd.Timestamp('{}-{}-{} {}:{}:{}'.format(y1, mon1, d1, h1, min1, s1))

# Compare zero-crossing wave and crest heights - save to dicts for plotting
Hws = {'hyd':[], 'lin':[], 'linkrms':[], 'nlkrms':[], 'ast':[]} # Wave heights
Hcs = {'hyd':[], 'lin':[], 'linkrms':[], 'nlkrms':[], 'ast':[]} # Crest heights

# Compare spectral moments of spectra, save in dicts for plotting
eta_keys = ['ETAh', 'ETAl', 'ETAlkrms', 'ETAnlkrms', 'ASTd']
m0d = {'{}'.format(k):[] for k in eta_keys}
m1d = {'{}'.format(k):[] for k in eta_keys}
m2d = {'{}'.format(k):[] for k in eta_keys}
m3d = {'{}'.format(k):[] for k in eta_keys}
m4d = {'{}'.format(k):[] for k in eta_keys}
# Also save water depths
wdl = []

# Iterate over dates in requested period and compute spectra
date_range = pd.date_range(t0, t1, freq='1D')
for date in tqdm(date_range, desc='Date: '):
    # Read Signature velocity dataset for current date
    datestr = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    fn_sig = [f for f in fns_v if datestr in f]
    ds = xr.decode_cf(xr.open_dataset(fn_sig[0], decode_coords='all'))
    
    # Iterate over 20-min periods for spectra
    time_range = pd.date_range(date, date + pd.Timedelta(days=1), freq='20T')
    for si, (t0s, t1s) in enumerate(zip(time_range[:-1], time_range[1:])):
        # Take 20-min slice from dataset
        seg = ds.sel(time=slice(t0s, t1s)).copy() # Segment slice
        # Test directional wave spectrum function
        z = seg.ASTd_eta.values
        u = seg.vEhpr.isel(range=4).values
        v = seg.vNhpr.isel(range=4).values
        spec = rpws.spec_uvz(z, u=u, v=v, fs=4)
        dm = {'a1':spec.a1.values, 'a2':spec.a2.values, 'b1':spec.b1.values,
              'b2':spec.b2.values, 'E':spec.Ezz.values}
        # Save to .mat
        # from scipy.io import savemat
        # savemat('dir_moments.mat', dm)
        # Plot the kx-ky and the directional spectra of 1st record
#         fig = plt.figure(figsize=(6,6))
#         ax1 = plt.subplot(111, projection='polar')
#         # cs1 = spec.Efth.sel(freq=slice(0.01, 0.5)).plot.contourf(ax=ax1)
#         import matplotlib.colors as colors
#         cs1 = ax1.contourf(np.deg2rad(spec.Efth.direction.values),
#                            spec.Efth.sel(freq=slice(0.01, 0.5)).freq.values,
#                            spec.Efth.sel(freq=slice(0.01, 0.5)).values,
#                            norm=colors.LogNorm(),
#                           )
#         ax1.set_theta_zero_location("N")
#         ax1.set_theta_direction(-1)
#         r = 5
#         x = 0.2
#         y = spec.mdir.item()
#         ax1.arrow(spec.mdir.item()/180.*np.pi, 0.3, 0.0, 0.02, width = 0.0015,
#                   color='r', head_width=0.07)
#         plt.colorbar(cs1, ax=ax1)
#         plt.tight_layout()
#         plt.show()
#         plt.close()

        # raise ValueError

        
        # ********************************************************************
        # Get zero-crossing wave and crest heights from different surface
        # elevation products and save for later plotting.
        # ********************************************************************
        # Use zero-crossings from hydrostatic surface elevation (smoothest?)
        zc, Hw, Hc, Ht = rpzc.get_waveheights(seg.eta_hyd.values)
        Hws['hyd'].extend(Hw)
        Hcs['hyd'].extend(Hc)
        # eta-lin wave/crest heights
        _, Hw, Hc, Ht = rpzc.get_waveheights(seg.eta_lin.values, zero_crossings=zc)
        Hws['lin'].extend(Hw)
        Hcs['lin'].extend(Hc)
        # eta-lin-krms wave/crest heights
        _, Hw, Hc, Ht = rpzc.get_waveheights(seg.eta_lin_krms.values, 
                                             zero_crossings=zc)
        Hws['linkrms'].extend(Hw)
        Hcs['linkrms'].extend(Hc)
        # eta-nl-krms wave/crest heights
        _, Hw, Hc, Ht = rpzc.get_waveheights(seg.eta_nl_krms.values, 
                                             zero_crossings=zc)
        Hws['nlkrms'].extend(Hw)
        Hcs['nlkrms'].extend(Hc)
        # AST wave/crest heights
        _, Hw, Hc, Ht = rpzc.get_waveheights(seg.ASTd_eta.values, zero_crossings=zc)
        Hws['ast'].extend(Hw)
        Hcs['ast'].extend(Hc)
        
        # ********************************************************************
        # Compare surface elevation spectra against velocity spectra from different
        # range bins.
        # In deep water, Ezz = (Euu + Evv) / (2*pi*freq)**2
        # (see eg. Thomson et al. 2018 DOI: 10.1175/JTECH-D-17-0091.1)
        # ********************************************************************
        # Get first range bin below contamination region below sea surface
        ast = seg.ASTd.copy() # AST signal (despiked)
        if np.all(np.isnan(ast.values)):
            # All NaN AST signal -> use z_lin instead
            ast = seg.z_lin.copy()
        zic = rpaf.contamination_range(ast.min().item(), binsz=0.5, beam_angle=25)
        # Highest range bin is zic m below AST minimum
        z_opt = ast.min().item() - seg.sel(range=zic, method='bfill').range.item()
        bin_max = seg.sel(range=z_opt, method='nearest').range.item()
        # Get range index of bin_max value
        bmi = np.argwhere(seg.range.values == bin_max).squeeze().item()
        # Check if figure already exists
        fn_fig = os.path.join(spec_outdir, 'spec_zvsuv_{}_{:03d}.pdf'.format(
            datestr, si))
        if not os.path.isfile(fn_fig):
            # Initialize f^-2 spectrum figure
            fig, ax = plt.subplots(figsize=(6,6))
            # Compute spectra using hor. velocities from all bins up to bin_max
            for bi in range(bmi):
                # Get horizontal (E,N) velocities from current range bin
                u = seg.vE.isel(range=bi).values # East vel.
                v = seg.vN.isel(range=bi).values # North vel.
                z = seg.ASTd_eta.values
                if np.all(np.isnan(z)):
                    # All NaN AST signal -> use eta_lin instead
                    z = seg.eta_lin.values
                dss = rpws.spec_uvz(z=z, u=u, v=v, fs=4)
                # Plot wave spectrum vs horizontal velocity spectra
                d = seg.z_lin.mean().item() # Water depth
                bind = seg.isel(range=bi).range.item() - d # Bin depth under z=0
                omega = 2*np.pi * dss.freq.values # Radian frequencies
                k = rptf.waveno_full(omega, d=d) # Wavenumbers
                # Hyperbolic term
                depth_corr = (np.cosh(k*(d + 0)) / np.sinh(k*d)) 
                Euv = (dss.Euu + dss.Evv) / (omega**2 * depth_corr**2)
                Euv.plot(ax=ax, label='Euv bin={}'.format(bi))
            # Finally, plot surface elevation variance spectrum on top
            dss.Ezz.plot(ax=ax, color='k', label='E_AST')
            # Set axes scale to log-log
            ax.set_xscale('log')
            ax.set_yscale('log')
            # ax.set_ylim([1e-3, 1e2])
            mwd = ast.mean().item()
            ax.set_title('{} - {} \n mean depth: {:.3f} m'.format(t0s, t1s, mwd))
            ax.legend()
            # Save figure
            plt.tight_layout()
            plt.savefig(fn_fig, bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close()

            # raise ValueError('Stop')
        
    # ********************************************************************
    # Based on the Euv vs Ezz spectral comparison, using the max. bin below
    # the contamination region (as used in original spectrum netcdf files) 
    # is fine. Next compare spectral moments and shapes from spectra estimated
    # with all different surface elevation products.
    # ********************************************************************
    cs = ['C0', 'C1', 'C2', 'C3', 'k',] # Line colors
    # Read spectral datasets into dict
    specd = {}
    for key in eta_keys:
        # Get spectrum netcdf filename that matches with current key
        fns_spec = [f for f in fns_s if key in f] # All dates for key
        fn_spec = [f for f in fns_spec if datestr in f] # Current date
        dss = xr.decode_cf(xr.open_dataset(fn_spec[0], decode_coords='all'))
        specd[key] = dss

    # Iterate over 20-min segments and plot spectral comparisons
    for ti, tseg in enumerate(specd['ASTd'].time.values):
        # Initialize figure for comparing spectra
        fn_spec_comp = os.path.join(spec_outdir, 'spec_comp_{}_{:03d}.pdf'.format(
            datestr, ti))
        fmin = 0.05 # Min. freq for moments
        fmax = 0.33 # Max. freq for moments
        if not os.path.isfile(fn_spec_comp):
            fig, ax = plt.subplots(figsize=(6,6))
            # Plot various spectra on the same plot
            for ki, key in enumerate(eta_keys):
                specd[key].Ezz.sel(time=tseg).plot(ax=ax, label=key, color=cs[ki])
                # Compute spectral moments and save to dicts
                # S = specd[key].Ezz.sel(time=tseg, freq=slice(fmin, fmax)).values
                # F = specd[key].sel(freq=slice(fmin, fmax)).freq.values
                S = specd[key].Ezz.sel(time=tseg).values
                F = specd[key].freq.values
                m0 = rpws.spec_moment(S=S, F=F, order=0)
                m1 = rpws.spec_moment(S=S, F=F, order=1)
                m2 = rpws.spec_moment(S=S, F=F, order=2)
                m3 = rpws.spec_moment(S=S, F=F, order=3)
                m4 = rpws.spec_moment(S=S, F=F, order=4)
                m0d[key].append(m0)
                m1d[key].append(m1)
                m2d[key].append(m2)
                m3d[key].append(m2)
                m4d[key].append(m2)
            # Also append water depth
            wdl.append(specd[key].sel(time=tseg).water_depth.item())
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(r'$S_{zz}$ [$m^2/\mathrm{Hz}$]')
            ax.set_title('{} \n mean depth {:.2f} m: '.format(
                tseg, specd[key].sel(time=tseg).water_depth.item()))
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            # Save figure
            plt.tight_layout()
            plt.savefig(fn_spec_comp, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            for ki, key in enumerate(eta_keys):
                # Only compute spectral moments and save to dicts, don't plot
                # S = specd[key].Ezz.sel(time=tseg, freq=slice(fmin, fmax)).values
                # F = specd[key].sel(freq=slice(fmin, fmax)).freq.values
                S = specd[key].Ezz.sel(time=tseg).values
                F = specd[key].freq.values
                m0 = rpws.spec_moment(S=S, F=F, order=0)
                m1 = rpws.spec_moment(S=S, F=F, order=1)
                m2 = rpws.spec_moment(S=S, F=F, order=2)
                m3 = rpws.spec_moment(S=S, F=F, order=3)
                m4 = rpws.spec_moment(S=S, F=F, order=4)
                m0d[key].append(m0)
                m1d[key].append(m1)
                m2d[key].append(m2)
                m3d[key].append(m2)
                m4d[key].append(m2)
            # Also append water depth
            wdl.append(specd[key].sel(time=tseg).water_depth.item())

    # Combine all individual Euv vs Ezz comparison figures into one pdf
    fn_all_zvsuv = os.path.join(img_outdir, 'all_spec_zvsuv_{}_{}.pdf'.format(
        args.ser, datestr))
    fns_pdf_zvsuv = sorted(glob.glob(os.path.join(spec_outdir, 
        'spec_zvsuv_{}*.pdf'.format(datestr))))
    if len(fns_pdf_zvsuv):
        # Call the PdfFileMerger
        mergedObject = PdfFileMerger()
        # Loop through all pdf files and append their pages
        for fn in fns_pdf_zvsuv:
            mergedObject.append(PdfFileReader(fn, 'rb'))
        # Write all the files into a file which is named as shown below
        if not os.path.isfile(fn_all_zvsuv):
            mergedObject.write(fn_all_zvsuv)

    # Combine all individual Ezz comparison figures into one pdf
    fn_all_comp = os.path.join(img_outdir, 'all_spec_comp_{}_{}.pdf'.format(
        args.ser, datestr))
    fns_pdf_comp = sorted(glob.glob(os.path.join(spec_outdir, 
        'spec_comp_{}*.pdf'.format(datestr))))
    if len(fns_pdf_comp):
        # Call the PdfFileMerger
        mergedObject = PdfFileMerger()
        # Loop through all pdf files and append their pages
        for fn in fns_pdf_comp:
            mergedObject.append(PdfFileReader(fn, 'rb'))
        # Write all the files into a file which is named as shown below
        if not os.path.isfile(fn_all_comp):
            mergedObject.write(fn_all_comp)

print('Making plots ...')
# Make scatter and QQ-plots of wave heights vs AST
fn_qq_hw = os.path.join(img_outdir, 'qq_hw_{}_{}.pdf'.format(args.d0, args.d1))
if not os.path.isfile(fn_qq_hw):
    fig, axes = plt.subplots(figsize=(8,8), nrows=2, ncols=2)
    keys = ['hyd', 'lin', 'linkrms', 'nlkrms']
    ylab = [r'$H_\mathrm{w}$ (hyd) [m]',
            r'$H_\mathrm{w}$ (lin) [m]', 
            r'$H_\mathrm{w}$ (lin-$K_\mathrm{rms}$) [m]', 
            r'$H_\mathrm{w}$ (nl-$K_\mathrm{rms}$) [m]',
           ]
    x = np.array(Hws['ast']) # AST wave heights (always on x axis)
    for i,k in enumerate(keys):
        ax = axes.flatten()[i]
        # Take out wave heights (AST always on x axis)
        y = np.array(Hws[k])
        _, _, _, im = ax.hist2d(x, y, norm=mpl.colors.LogNorm(), cmap=mpl.cm.gray, 
                                bins=[50,50])
        rppl.qqplot(x, y, ax=ax, scatter=False, color='r')
        # Linear regression
        res = linregress(x, y)
        rmse = mean_squared_error(x, y, squared=False)
        xp = np.linspace(0, x.max())
        # Linear regression line
        ax.plot(xp, res.intercept + res.slope*xp, color='k', linewidth=1)
        # One-to-one line
        ax.plot(xp, xp, color='k', alpha=0.6, linestyle='--')
        # Annotate regression statistics
        rsq = r'$R^2$={:.2f}'.format(res.rvalue**2)
        ax.annotate(rsq, xy=(0.1, 0.9), xycoords="axes fraction")
        rmsestr = 'RMSE={:.2f}'.format(rmse)
        ax.annotate(rmsestr, xy=(0.1, 0.82), xycoords="axes fraction")
        bias = 'Bias={:.2f}'.format(res.slope - 1)
        ax.annotate(bias, xy=(0.1, 0.74), xycoords="axes fraction")
        ax.set_ylabel(ylab[i])
        ax.set_xlabel(r'$H_\mathrm{w}$ (AST) [m]')
        ax.set_xlim([0, x.max()+0.5])
        ax.set_ylim([0, y.max()+0.5])
    plt.tight_layout()
    plt.savefig(fn_qq_hw, bbox_inches='tight', dpi=300)
    plt.close()

# Make scatter and QQ-plots of crest heights vs AST
fn_qq_hc = os.path.join(img_outdir, 'qq_hc_{}_{}.pdf'.format(args.d0, args.d1))
if not os.path.isfile(fn_qq_hc):
    fig, axes = plt.subplots(figsize=(8,8), nrows=2, ncols=2)
    keys = ['hyd', 'lin', 'linkrms', 'nlkrms']
    ylab = [r'$H_\mathrm{c}$ (hyd) [m]',
            r'$H_\mathrm{c}$ (lin) [m]', 
            r'$H_\mathrm{c}$ (lin-$K_\mathrm{rms}$) [m]', 
            r'$H_\mathrm{c}$ (nl-$K_\mathrm{rms}$) [m]',
           ]
    x = np.array(Hcs['ast']) # AST wave heights (always on x axis)
    for i,k in enumerate(keys):
        ax = axes.flatten()[i]
        # Take out wave heights (AST always on x axis)
        y = np.array(Hcs[k])
        _, _, _, im = ax.hist2d(x, y, norm=mpl.colors.LogNorm(), cmap=mpl.cm.gray, 
                                bins=[50,50])
        rppl.qqplot(x, y, ax=ax, scatter=False, color='r')
        # Linear regression
        res = linregress(x, y)
        rmse = mean_squared_error(x, y, squared=False)
        xp = np.linspace(0, x.max())
        # Linear regression line
        ax.plot(xp, res.intercept + res.slope*xp, color='k', linewidth=1)
        # One-to-one line
        ax.plot(xp, xp, color='k', alpha=0.6, linestyle='--')
        # Annotate regression statistics
        rsq = r'$R^2$={:.2f}'.format(res.rvalue**2)
        ax.annotate(rsq, xy=(0.1, 0.9), xycoords="axes fraction")
        rmsestr = 'RMSE={:.2f}'.format(rmse)
        ax.annotate(rmsestr, xy=(0.1, 0.82), xycoords="axes fraction")
        bias = 'Bias={:.2f}'.format(res.slope - 1)
        ax.annotate(bias, xy=(0.1, 0.74), xycoords="axes fraction")
        ax.set_ylabel(ylab[i])
        ax.set_xlabel(r'$H_\mathrm{c}$ (AST) [m]')
        ax.set_xlim([0, x.max()+0.5])
        ax.set_ylim([0, y.max()+0.5])
    plt.tight_layout()
    plt.savefig(fn_qq_hc, bbox_inches='tight', dpi=300)
    plt.close()

# Plot example time series of failed Krms nonlinear reconstruction
t0 = pd.Timestamp('2022-07-14T20:26:00')
t1 = pd.Timestamp('2022-07-14T20:26:40')
# Read correct dataset
datestr = '20220714'
fn_bad_krms = os.path.join(img_outdir, 'bad_krms_{}.pdf'.format(datestr))
if not os.path.isfile(fn_bad_krms):
    fnex = [f for f in fns_v if datestr in f]
    ds = xr.decode_cf(xr.open_dataset(fnex[0], decode_coords='all'))
    fig, ax = plt.subplots(figsize=(7,3),)
    ds.eta_nl_krms.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{nl}$ ($K_\mathrm{rms}$)')
    ds.eta_lin_krms.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{lin}$ ($K_\mathrm{rms}$)')
    ds.ASTd_eta.sel(time=slice(t0, t1)).plot(ax=ax, label=r'$\eta_\mathrm{AST}$',
                    color='k')
    ax.set_ylabel(r'$\eta$ [m]')
    ax.set_xlabel(None)
    ax.set_title(t0.date())
    ax.legend()
    plt.tight_layout()
    plt.savefig(fn_bad_krms, bbox_inches='tight', dpi=300)
    plt.close()

# Plot example time series of bad AST signal
t0 = pd.Timestamp('2022-07-14T20:53:10')
t1 = pd.Timestamp('2022-07-14T20:54:30')
fn_bad_ast = os.path.join(img_outdir, 'bad_ast_{}.pdf'.format(datestr))
if not os.path.isfile(fn_bad_ast):
    fig, ax = plt.subplots(figsize=(7,3),)
    ds.eta_nl_krms.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{nl}$ ($K_\mathrm{rms}$)')
    ds.eta_lin_krms.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{lin}$ ($K_\mathrm{rms}$)')
    ds.ASTd_eta.sel(time=slice(t0, t1)).plot(ax=ax, label=r'$\eta_\mathrm{AST}$',
                    color='k')
    ax.set_ylabel(r'$\eta$ [m]')
    ax.set_xlabel(None)
    ax.set_title(t0.date())
    ax.legend()
    plt.tight_layout()
    plt.savefig(fn_bad_ast, bbox_inches='tight', dpi=300)
    plt.close()

# Plot example time series with all reconstructions
t0 = pd.Timestamp('2022-07-06T20:28:00')
t1 = pd.Timestamp('2022-07-06T20:28:30')
# Read correct dataset
datestr = '20220706'
fn_ex = os.path.join(img_outdir, 'etas_example_{}.pdf'.format(datestr))
if not os.path.isfile(fn_ex):
    fnex = [f for f in fns_v if datestr in f]
    ds = xr.decode_cf(xr.open_dataset(fnex[0], decode_coords='all'))
    fig, ax = plt.subplots(figsize=(7,3),)
    ds.eta_hyd.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{hyd}$')
    ds.eta_lin.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{lin}$')
    ds.eta_lin_krms.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{lin}$ ($K_\mathrm{rms}$)')
    ds.eta_nl_krms.sel(time=slice(t0, t1)).plot(ax=ax, 
        label=r'$\eta_\mathrm{nl}$ ($K_\mathrm{rms}$)')
    ds.ASTd_eta.sel(time=slice(t0, t1)).plot(ax=ax, label=r'$\eta_\mathrm{AST}$',
                    color='k')
    ax.set_ylabel(r'$\eta$ [m]')
    ax.set_xlabel(None)
    ax.set_title(t0.date())
    ax.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(fn_ex, bbox_inches='tight', dpi=300)
    plt.close()

# Make scatter plots of spectral moments from AST vs surface reconstructions
md = {'m0d':m0d, 'm1d':m1d, 'm2d':m2d, 'm3d':m3d, 'm4d':m4d, }
for mom in range(5):
    fn_mom = os.path.join(img_outdir, 'm{}_comp_{}_{}.pdf'.format(mom, args.d0, args.d1))
    if not os.path.isfile(fn_mom):
        fig, axes = plt.subplots(figsize=(4,12), sharex=True, sharey=True, nrows=4)
        alpha = 0.8
        ylab = [r'(hyd)', r'(lin)', r'(lin-$K_\mathrm{rms}$)', r'(nl-$K_\mathrm{rms}$)',]
        ress = [] # List to store lin. reg. results for later plotting
        rmses = [] # List to store RMSE values for later plotting
        xymins = [] # List to store min. xy limits
        xymaxs = [] # List to store max. xy limits
        for ax in axes:
            ax.set_facecolor('#E6E6E6')
            ax.grid(color='w', alpha=0.4)
        for i,k in enumerate(eta_keys[:-1]):
            x = np.array(md['m{}d'.format(mom)]['ASTd'])
            y = np.array(md['m{}d'.format(mom)][k])
            sc = axes[i].scatter(x, y, c=wdl, cmap=cmocean.cm.balance, alpha=alpha)
                # edgecolors='k')
            # Linear regression
            res = linregress(x, y)
            ress.append(res)
            rmse = mean_squared_error(x, y, squared=False)
            rmses.append(rmse)
            # One-to-one line
            xymin = np.min((ax.get_xlim()[0], ax.get_ylim()[0]))
            xymins.append(xymin)
            xymax = np.max((ax.get_xlim()[1], ax.get_ylim()[1]))
            xymaxs.append(xymax)
            axes[i].set_ylabel('m{} '.format(mom) + ylab[i])
        # Plot linear regression lines
        for i, ax in enumerate(axes):
            res = ress[i]
            xymin = np.min(xymins)
            xymax = np.max(xymaxs)
            xp = np.linspace(xymin, xymax)
            # Linear regression line
            ax.plot(xp, res.intercept + res.slope*xp, color='k', linewidth=1)
            # One-to-one lines
            ax.plot(xp, xp, color='k', alpha=0.6, linestyle='--')
            # Annotate regression statistics
            rsq = r'$R^2$={:.2f}'.format(res.rvalue**2)
            ax.annotate(rsq, xy=(0.1, 0.9), xycoords="axes fraction")
            rmse = 'RMSE={:.2f}'.format(rmses[i])
            ax.annotate(rmse, xy=(0.1, 0.82), xycoords="axes fraction")
            bias = 'Bias={:.2f}'.format(res.slope - 1)
            ax.annotate(bias, xy=(0.1, 0.74), xycoords="axes fraction")
        axes[3].set_xlabel(r'AST')
        # Colorbar
        cax = axes[0].inset_axes([0.0, 1.04, 1, 0.05], transform=axes[0].transAxes)
        cbar = fig.colorbar(sc, ax=axes[0], cax=cax, orientation='horizontal',)
        cbar.set_label('water depth [m]', y=1.05, rotation=0)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

        plt.tight_layout()
        plt.savefig(fn_mom, bbox_inches='tight', dpi=300)
        plt.close()


print('Done.')