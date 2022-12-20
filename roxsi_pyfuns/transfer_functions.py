"""
Transfer functions for wave properties, e.g. pressure -> surface
elevation.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.signal import hann
from roxsi_pyfuns import wave_spectra as rpws
from roxsi_pyfuns import zero_crossings as rpzc


def waveno_deep(omega):
    """
    Returns wavenumber array assuming linear deep water 
    dispersion relation omega**2 = g*k.
    Parameters:
        f - np.array; radian frequency array
    Returns:
        k - wavenumber
    """
    # Make sure omega is a numpy array
    omega = np.atleast_1d(omega)
    return omega**2 / 9.81

def waveno_shallow(omega, d):
    """
    Returns peak wavelength assuming shallow water 
    linear dispersion relation omega = k*sqrt(gd).
    Parameters:
        omega - np.array; radian frequency array
        d - scalar; water depth (m)
    Returns:
        k - np.array; wavenumber array
    """
    # Make sure omega is a numpy array
    omega = np.atleast_1d(omega)
    return omega / np.sqrt(9.81 * d)

def waveno_full(omega, d, k0=None, **kwargs):
    """
    Returns wavelength array assuming full linear 
    dispersion relation omega**2 = g*k * tanh(k*d).
    Uses shallow-water wavenumber array as initial guess unless 
    another guess is given.
    Parameters:
        omega - np.array; radian frequency array
        d - scalar; water depth
        k0 - scalar; initial guess wavelength. If None, uses S-W k
             as initial guess.
        **kwargs for scipy.optimize.newton()
    Returns:
        k - np.array; wavenumber array
    """
    # Make sure omega is a numpy array
    omega = np.atleast_1d(omega)

    # Function to solve
    def f(x, omega, d): 
        return 9.81*x * np.tanh(x*d) - omega**2

    # Estimate shallow-water wavenumber if not given
    if k0 is None:
        k0 = waveno_shallow(omega, d)
    # Solve f(x) for intermediate water k using secant method
    k = optimize.newton(f, k0, args=(omega, d))

    return k

def eta_hydrostatic(pt, patm=None, rho0=1025, grav=9.81, interp=True):
    """
    Returns hydrostatic pressure head in units of meters from water
    pressure time series and atmospheric pressure time series.

    Parameters:
        pt - pd.Series; time series (w/ time index) of water pressure (dbar)
        patm - pd.Series; time series of atmospheric pressure anomaly (dbar)
        rho0 - scalar; water density (kg/m^3)
        grav - scalar; gravitational acceleration (m/s^2)
        interp - bool; if True, interpolate atmospheric pressure to 
                 time index of water pressure.
    Returns:
        eta - pd.Series; time series of hydrostatic pressure head (m)
    """
    # Copy input series
    pw = pt.copy()
    pw = pw.to_frame(name='pressure') # Convert to dataframe
    if patm is not None:
        # Correct for atmospheric pressure
        dfa = patm.copy()
        if interp:
            # Interpolate atmospheric pressure to water pressure time index
            dfa = dfa.reindex(pt.index, method='bfill').interpolate()
        # Concatenate input arrays
        df = pd.concat([pw, dfa], axis=1)
        # Correct for atmospheric pressure anomaly
        df['eta_hyd'] = df['pressure'] - df['hpa_anom']
    else:
        # Do not correct for atmospheric pressure
        df = pw.copy()
        df['eta_hyd'] = df['pressure'].copy()

    # Convert from hPa to meters (pressure head)
    factor = rho0*grav/10000.0
    # Remove negative values
    ii = np.where(df['eta_hyd']<0)[0]
    df['eta_hyd'][ii] = 0
    df['eta_hyd'] /= factor # Pressure head, unit [m]

    return df['eta_hyd']


def k_rms(h0, f, P, B):
    """
    Compute root-mean-square wavenumber following Herbers et al. (2002)
    definition (Eq. 12). Function implementation based on fun_compute_rms.m
    function by Kevin Martins.
    
    Parameters:
        h0 - scalar; mean water depth
        f - frequency array
        P - 1D array; power spectrum [m^2]. Note: not a density
        B - 2D array; bispectrum [m^2]. Note: not a density

    Returns:
        krms - array or RMS wavenumbers, same shape as f

    Note: f, P and B arrays are two-sided in regards to f; f should 
    be centered around 0 Hz.
    """
    # Initialisation
    krms = np.ones_like(f) * np.nan
    nmid = int(len(f) / 2) # Middle frequency (f=0)
    g = 9.81 # Gravity

    # Transforming P and B to densities
    df = abs(f[1]-f[0])
    P  /= df
    B  /= df**2

    # Iterate over frequencies and compute wavenumbers
    for fi in range(len(f)):
        # Initialisation of the cumulative sum and wavenumbers
        sumtmp = 0 
        krms[fi] = 2 * np.pi * f[fi] / np.sqrt(g*h0) # Linear part
        
        # Loop over frequencies
        for ff in range(len(f)):
            ifr3 = nmid + (fi-nmid) - (ff-nmid)
            if (ifr3 >= 0) and (ifr3 < len(f)):
                sumtmp += df * np.real(B[ff, ifr3])
        
        # Non-linear terms (Eqs. 19 and 20 of Martins et al., 2021)
        Beta_fr  = h0 * (2 * np.pi * f[fi])**2 / (3*g)
        Beta_am  = 3 * sumtmp / (2 * h0 * P[fi])
        
        # Non-linear wavenumber
        krms[fi] *= np.sqrt(1 + Beta_fr - Beta_am)

    return krms


class TRF():
    """
    Pressure-to-sea surface linear transfer function (TRF) class.
    """
    def __init__(self, fs=16, zp=0, type='RBR SoloD'):
        """
        Initialize TRF class with pressure sensor parameters.
        Parameters:
            fs - scalar; sampling frequency (Hz)
            zp - scalar; height of pt sensor from seabed (m)
            type - str; Sensor type
        """
        self.fs = fs
        self.dt = 1 / self.fs # Sampling rate (sec)
        self.zp = zp
        self.type = type


    def p2eta_lin(self, pp, M=512, fmin=0.05, fmax=0.33, 
                  att_corr=True, max_att_corr=5, return_kp=False):
        """
        Linear transfer function from water pressure to sea-surface 
        elevation eta.

        Based on the pr_corr.m Matlab function written by Urs Neumeier
        (http://neumeier.perso.ch/matlab/waves.html).

        Parameters:
            pp - np.array; water pressure fluctuation time series (m)
            M - int; window segment length (512 by default)
            fmin - scalar; min. cutoff frequency
            fmax - scalar; max. cutoff frequency
            att_corr - bool; if True, applies attenuation correction
            max_att_corr - scalar; maximum attenuation correction.
                           Should not be higher than 5.
            return_kp - bool; set to True to also return the transfer
                        function Kp and wavenumbers ks
        Returns:
            eta - np.array; linear sea surface elevation time series
        """
        
        pt = pp.copy() # Copy p array so we don't change the input
        if not len(pt):
            raise ValueError('Empty pressure sensor array.')
        # Also make sure pt has no NaN values
        if np.any(np.isnan(pt)):
            raise ValueError('NaNs found in pt array, remove them.')

        # Make sure M is even by checking the remainder
        m = len(pt)
        M = int(min(M, m))
        if (M % 2) != 0:
            raise ValueError('M must be even.')

        # Define length of overlapping segments and zero-padding
        # array length N
        N_ol = M//2 # length of overlap
        # length of array zero-padded to nearest multiple of M
        N = (np.ceil(m/M) * M ).astype(int)

        # Make frequency array (only need the first half)
        freqs = np.arange(M/2+1) * self.fs / M
        oms = freqs * np.pi*2 # Radian frequencies

        # Allocate output array and define overlapping window (Hann)
        eta = np.zeros(N)
        win = hann(M)
        win[M//2:] = 1 - win[:M//2]

        # Detrend data in overlapping segments 
        for ss in np.arange(0, N-N_ol, N_ol).astype(int):
            ## ss - segment start index; se - segment end index
            se = min(ss + M, m);
            # Take out segment to detrend
            seg = pt[ss:se].copy()
            seglen = len(seg)
            # Calculate trend in segment
            x = np.arange(seglen) # Inputs
            trend = np.polyfit(x, seg, 1)
            # Calculate mean water depth
            dseg = np.polyval(trend, seglen/2)
            # Remove trend
            seg_dt = seg - np.polyval(trend, x)

            # Calculate wavenumbers using full dispersion relation
            ks = waveno_full(oms, d=dseg)

            # Apply cutoff limits to ks to avoid overflow warnings
            acidx = np.logical_or(freqs<fmin, freqs>fmax)
            ks_m = ks.copy()
            ks_m[acidx] = 0 # No correction outside of range
            # Calculate pressure response factor Kp
            Kp = np.cosh(ks_m*self.zp) / np.cosh(ks_m*dseg)
            # Kp[acidx] = 1 # No correction outside of range
            if att_corr:
                # Apply attenuation correction factor to desired range
                Kp[Kp < 1/max_att_corr] = 1 / max_att_corr
                # Linear decrease of correction for freqs above fmax
                Kphf = Kp[freqs>fmax].copy() # high-freq part of Kp
                # Get Kp index closest to fmax
                ifmax = np.where(freqs<=fmax)[0][-1]
                # Get indices for linear decrease
                idxLin = np.arange(ifmax, min(ifmax+np.fix(len(ks)/10)+2, len(ks)))
                idxLin = idxLin.astype(int)
                dfac = (Kp[ifmax]-1) / len(idxLin) # Decrease factor
                # Apply linear decrease to defined range
                Kp[idxLin] = np.arange(len(idxLin))[::-1] * dfac + 1
                Kp[0] = 1
            # Duplicate symmetric and mirrored second half
            Kp = np.concatenate([Kp[:-1], np.flip(Kp[1:])])
            # If segment length < M, zero pad end of segment
            if len(seg_dt) < M:
                seg_dt = np.pad(seg_dt, (0, M-len(seg_dt)), 'constant')
            
            # Take FFT of detrended segment
            pHat_seg = np.fft.fft(seg_dt)
            # Apply correction factor
            etaHat_seg = pHat_seg / Kp
            # Inverse FFT to get sea surface elevations
            eta_seg = np.real(np.fft.ifft(etaHat_seg))
            # Remove potential zero padding
            eta_seg = eta_seg[:seglen]
            # Add segment trend back to data
            eta_seg += np.polyval(trend, x)

            # Apply overlapping window
            eta[ss:se] += eta_seg * win[:seglen]
            # Make sure first and last segments are correct
            if ss == 0:
                se_n = int(min(N_ol, seglen)) # Updated segment end index
                eta[:se_n] = eta_seg[:se_n]
            if ss + M >= N and seglen > N_ol:
                eta[ss+N_ol:se] = eta_seg[N_ol:]
        
        # Reshape output array
        eta = eta[:m]

        if return_kp:
            return eta, Kp, ks
        else:
            return eta


    def p2eta_krms(self, eta_hyd, h0, tail_method='constant', fc=None, fc_fact=2.5,
                   fmax=2.0, krms=None, f_krms=None, return_nl=True,
                   fix_ends=True):
        """
        Fully dispersive sea-surface reconstruction from sub-surface
        pressure measurements. The reconstruction uses linear wave theory 
        but with measured or predicted dominant wavenumbers kappa 
        (e.g. kappa_rms from Herbers et al., 2002) that are provided as input.

        Based on fun_pReWave_krms_complete.m function by Kevin Martins.
        Reference: Martins et al. (2021): https://doi.org/10.1175/JPO-D-21-0061.1

        Parameters:
            eta_hyd - 1D array; hydrostatic surface elevation [m]
            h0 - scalar; mean water depth [m]
            tail_method - str; one of ['hydrostatic', 'constant'], uses either
                          hydrostatic pressure field or constant transfer 
                          function above cutoff frequency
            fc - scalar; cutoff frequency [Hz]
            fc_fact - scalar; factor to multiply fp to get fc if fc=None
            fmax - scalar; max. frequency for FFT
            krms - array; root-mean-square wave number array following 
                   Herbers et al. (2002). If None, krms if computed from input
                   array eta_hyd (note: bispectrum estimation is slow)
            f_krms - array like krms; frequency array corresponding to krms.
                     Must be given if krms is given.
            return_nl - bool; if True, return both linear and nonlinear (nl)
                        surface reconstructions.
            fix_ends - bool; if True, sets first and last waves of (non)linear
                       reconstructions equal to the first and last waves of the
                       hydrostatic surface to avoid wiggles at endpoints of
                       the (non)linear reconstructions.
        
        Returns:
            eta_lin - array; linear surface reconstruction using K_rms [m]
            if return_nl is True:
                eta_nl - array; nonlinear surface reconstruction using K_rms [m]
        """
        # Sample rate, data length, Gravity
        N = len(eta_hyd)
        Gravity = 9.81
        if fc is None:
            # Estimate power spectrum
            dss = rpws.spec_uvz(eta_hyd, fs=self.fs)
            # Compute cutoff frequency from peak frequency
            fp = 1 / dss.Tp_Y95.item() # peak freq. (Young, 1995)
            fc = fc_fact * fp
        
        # Frequency array
        freq = np.arange(0, N/2 + 1) / (N/2) * self.fs/2
        ic = np.round(fc / self.fs * N) # index for cutoff frequency

        # Compute K_rms if not given
        if krms is None:
            # First compute bispectrum (slow)
            print('Calculating bispectrum ...')
            dsb = rpws.bispectrum(eta_hyd, fs=self.fs, h0=h0, fp=fp,
                                  timestamp=dfe.index[0].round('20T'), 
                                  return_krms=True)
            krms = dsb.k_rms.values
            f_krms = dsb.freq.values
        else:
            if f_krms is None:
                raise ValueError('Must also input f_krms')

        # Reinterpolate k_rms to the local frequency structure
        k_loc = np.interp(freq, f_krms, krms)

        # Computing FFT on input signal
        fft_e_HY = np.fft.fft(eta_hyd)
        # Set to zero FFT components at frequencies > fmax
        fidx = np.argwhere(freq > fmax).squeeze()
        fft_e_HY[fidx] = 0

        # Dealing with cutoff frequency
        ifp = np.argmax((np.abs(fft_e_HY**2)))
        fp = freq[ifp]
        # Index corresponding to start of smoothing around fc
        ihw_b = round((fc - fp/4) / self.fs*N) 
        # Index corresponding to end of smoothing around fc
        ihw_e = round((fc + fp/4) / self.fs*N) 

        # Initialisations
        fft_d1_e = np.zeros(N).astype(complex) # 1st derivative
        fft_d2_e = np.zeros(N).astype(complex) # 2nd derivative
        fft_e_L  = fft_e_HY.copy()
        fft_f_term_eqB6 = np.zeros(N).astype(complex)
  
        # Correction of the pressure signal
        for i in np.arange(1, len(freq)):
            # Different correction depending on the frequency
            # 1 - Measured or predicted kappa (before cutoff frequency)
            if i <= ihw_b:
                # Getting dominant wavenumber from interpolated entry
                kn0 = k_loc[i]
                # Computing fft of linear reconstruction
                fft_e_L[i] = fft_e_HY[i] * np.cosh(kn0*h0) / np.cosh(kn0*self.zp)
            
            # 2 - Around the cutoff frequency, we smoothly change from measured k to:
            #     1 - k --> 0 (i.e., Kp = 1) for hydrostatic pressure field
            #     2 - k ~ k(fc)
            if tail_method == 'constant':
                ktail = k_loc[ihw_e]
            if (i > ihw_b) and (i < ihw_e):
                if tail_method == 'hydrostatic':
                    # Last transfer function computed
                    Kp_b = np.cosh(kn0*h0) / np.cosh(kn0*self.zp) 
                    Kp_e = 1 # Hydrostatic treatment
                else:
                    # Last transfer function computed
                    Kp_b = np.cosh(kn0*h0) / np.cosh(kn0*self.zp) 
                    # Value around fc
                    Kp_e = np.cosh(ktail*h0) / np.cosh(ktail*self.zp) 
                # Interpolation of Kp
                Kp_loc = np.interp(freq[i], [freq[ihw_b], freq[ihw_e]], [Kp_b, Kp_e])
                # Computing fft of linear reconstruction
                fft_e_L[i] = fft_e_HY[i] * Kp_loc
            
            # 3 - After cutoff frequency, we assume a hydrostatic pressure field
            if i > ihw_e:
                # Correction
                fft_e_L[i] = fft_e_HY[i] * Kp_e # whichever last Kp_e, we use it
            
            # Coefficient for additional term of complete formula (Eq. B6 of JFM)
            coef_EQB6 = np.sinh(kn0*self.zp) / np.sinh(kn0*h0)
            
            # Derivatives
            fft_d1_e[i] = 1j * 2*np.pi * freq[i] * fft_e_L[i]
            fft_d2_e[i] = -(2*np.pi * freq[i])**2 * fft_e_L[i]
            fft_f_term_eqB6[i] = coef_EQB6 * fft_d1_e[i]
            
            # Conjugates
            fft_e_L[N-i]  = np.conjugate(fft_e_L[i])
            fft_d1_e[N-i] = np.conjugate(fft_d1_e[i])
            fft_d2_e[N-i] = np.conjugate(fft_d2_e[i])
            fft_f_term_eqB6[N-i] = np.conjugate(fft_f_term_eqB6[i])

        # Inverse Fourier transform
        e_L = np.real(np.fft.ifft(fft_e_L)) # Linear reconstruction

        # Also return nonlinear reconstruction?
        if return_nl:
            d1_e = np.real(np.fft.ifft(fft_d1_e))
            d2_e = np.real(np.fft.ifft(fft_d2_e))
            f_term_eqB6 = np.real(np.fft.ifft(fft_f_term_eqB6))

            # Now that we have the f term, we can compute the G term
            fft_G_term_eqB6 = np.fft.fft(f_term_eqB6**2)
            for i in np.arange(1,len(freq)):
                # Different correction depending on the frequency
                # 1 - Measured or predicted kappa (before cutoff frequency)
                if i <= ihw_b:
                    # Getting dominant wavenumber from interpolated entry
                    kn0 = k_loc[i]
                # Coefficient for additional term of complete formula (Eq. B6 of JFM)
                coef_EQB6 = np.cosh(kn0*h0) / np.cosh(kn0*self.zp)
                # Computing G term
                fft_G_term_eqB6[i] = coef_EQB6 * fft_G_term_eqB6[i]
                fft_G_term_eqB6[N-i] = np.conjugate(fft_G_term_eqB6[i])

            G_term_eqB6 = np.real(np.fft.ifft(fft_G_term_eqB6))
            
            # Computation of e_NL   
            e_NL = e_L - (1/Gravity) * (e_L*d2_e+d1_e**2) + (1/Gravity) * G_term_eqB6

            # Fix start and end points?
            if fix_ends:
                # Get indices of first and last zero-crossings of hydrostatic eta
                zc, _, _, _ = rpzc.get_waveheights(eta_hyd, method='down')
                # Set first and last zero crossings of e_NL equal to eta_hyd
                e_NL[:zc[0]] = eta_hyd[:zc[0]]
                e_NL[zc[-1]:] = eta_hyd[zc[-1]:]
                # Same for e_L
                e_L[:zc[0]] = eta_hyd[:zc[0]]
                e_L[zc[-1]:] = eta_hyd[zc[-1]:]

            # Return both linear and nonlinear reconstructions
            return e_L, e_NL

        else:
            # Only return linear reconstruction
            return e_L


# Test script
if __name__ == '__main__':
    """
    Test script using synthetic example data.
    """
    import os
    import sys
    import glob
    import pandas as pd
    from scipy.io import loadmat
    from argparse import ArgumentParser

    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data root directory"),
                type=str,
                #default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/RBRDuetDT/Level1/mat',
                default = r'/media/mikapm/T7 Shield/ROXSI/Asilomar2022/SmallScaleArray'
                )
        parser.add_argument("-date", 
                help=("File date (yyyymmdd)"),
                type=str,
                default='20220627',
                )
        parser.add_argument("-mid", 
                help=("Mooring ID"),
                type=str,
                default='C2vp',
                )
        parser.add_argument("-sid", 
                help=("Sensor ID (duetDT or soloD)"),
                type=str,
                default='duetDT',
                )
        parser.add_argument("-M", 
                help=("Segment window length (number of samples)"),
                type=int,
                default=512,
                )
        parser.add_argument("-fmin", 
                help=("Min. frequency for attenuation correction"),
                type=float,
                default=0.05,
                )
        parser.add_argument("-fmax", 
                help=("Max. frequency for attenuation correction"),
                type=float,
                default=0.33,
                )
        parser.add_argument("-fact", 
                help=("Factor to multiply fp by to get fc"),
                type=float,
                default=2.5,
                )
        parser.add_argument("-tail", 
                help=("Tail method"),
                type=str,
                choices=['hydrostatic', 'constant'],
                default='hydrostatic',
                )

        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Load ROXSI pressure sensor test time series
    outdir = '/home/mikapm/Github/Martins_pressure_reconstruction/data'
    fn_test = os.path.join(outdir, 'test_data.csv')
    dfe = pd.read_csv(fn_test, parse_dates=['time']).set_index('time')
    h0 = dfe['z_hyd'].mean().item()
    eta_hyd = dfe['eta_hyd'].values.squeeze()

    # Read bispectrum of eta_hyd
    fn_bisp = os.path.join(outdir, 'bispec_test_etalin.nc')
    dsb = xr.open_dataset(fn_bisp, engine='h5netcdf')
    krms = dsb.k_rms.values
    f_krms = dsb.freq.values

    fs = 16
    zp = 0.08

    # Initialize class
    trf = TRF(fs=fs, zp=zp)
    # Transform pressure -> eta (linear) for 20-min chunk
    z_lin = trf.p2eta_lin(dfe['z_hyd'].values, M=args.M,
                          fmin=args.fmin, fmax=args.fmax)
    eta_lin = z_lin - np.mean(z_lin)

    # Test K_rms reconstruction
    eL, eNL = trf.p2eta_krms(eta_hyd, h0=h0, krms=krms, f_krms=f_krms, 
                             tail_method=args.tail, fc_fact=args.fact,)

    # Plot time series
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(eta_hyd, label=r'$\eta_\mathrm{hyd}$')
    ax.plot(eta_lin, label=r'$\eta_\mathrm{lin}$')
    ax.plot(eL, label=r'$\eta_\mathrm{lin}$ $(K_\mathrm{rms})$')
    ax.plot(eNL, label=r'$\eta_\mathrm{nl}$ $(K_\mathrm{rms})$')
    ax.set_title('Tail method: {}'.format(args.tail))
    ax.legend()

    plt.show()
    plt.clf()

    # Plot spectra
    fig, ax = plt.subplots(figsize=(8,8))
    dss = rpws.spec_uvz(eta_hyd, fs=16)
    ax.loglog(dss.freq, dss.Ezz, label=r'$\eta_\mathrm{hyd}$')
    dss = rpws.spec_uvz(eta_lin, fs=16)
    ax.loglog(dss.freq, dss.Ezz, label=r'$\eta_\mathrm{lin}$')
    dss = rpws.spec_uvz(eL, fs=16)
    ax.loglog(dss.freq, dss.Ezz, label=r'$\eta_\mathrm{lin}$ $(K_\mathrm{rms})$')
    dss = rpws.spec_uvz(eNL, fs=16)
    ax.loglog(dss.freq, dss.Ezz, label=r'$\eta_\mathrm{nl}$ $(K_\mathrm{rms})$')
    plt.axvline(x=args.fmax, color='gray', linestyle='--')
    ax.set_title('Tail method: {}'.format(args.tail))
    ax.legend()

    plt.show()
    plt.clf()


    
    # fn_mat = glob.glob(os.path.join(args.dr, 'roxsi_{}_L1_{}_*_{}.mat'.format(
    #     args.sid, args.mid, args.date)))[0]
    # print('Loading pressure sensor mat file {}'.format(fn_mat))
    # mat = loadmat(fn_mat)
    # # Read pressure time series and timestamps
    # pt = np.array(mat['DUETDT']['Pwater'].item()).squeeze()
    # time_mat = np.array(mat['DUETDT']['time_dnum'].item()).squeeze()
    # # Convert timestamps
    # time_ind = pd.to_datetime(time_mat-719529,unit='d') 
    # # Read sampling frequency and sensor height above seabed
    # fs = int(mat['DUETDT']['sample_freq'].item()[0].split(' ')[0])
    # zp = mat['DUETDT']['Zbed'].item().squeeze().item()
    # # Make pandas DataFrame
    # dfp = pd.DataFrame(data={'pt':pt}, index=time_ind)
