"""
Functions to compute turbulence parameters.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import detrend
from roxsi_pyfuns import stats as rps

def k_spec_wavephase(w, U, fs=16, k_int=None):
    """
    Estimate wavenumber spectrum from short segment
    of vertical velocity w sampled at rate fs (Hz)
    using Taylor's hypothesis. 
    
    Follows approach of George et al. (1994; JGR) to
    estimate dissipation from short velocity segments 
    during a wave phase.
    
    Note that this function does not do any window-averaging, 
    but the input time series is tapered once using a standard 
    sinusoidal  window.

    If k_int is given, the output k spectrum is interpolated
    to the wavenumbers specified by k_int.

    Parameters:
        w - array; time series for spectrum
        U - scalar; mean current during w segment
        fs - scalar; sampling frequency [Hz]
        k_int - array; wavenumber array to interpolate
                spectrum to (optional)

    Returns:
        dss -  xr.Dataset with spectrum as data and (interpolated) 
               wavenumbers as coordinates.
    """
    # Copy input timeseries
    wseg = w.copy()
    seglen = len(wseg)
    # Frequencies
    n_freqs = int(np.floor(seglen / 2 )) # No. of frequency bands
    # Calculate Nyquist frequency and bandwidth (freq. resolution)
    nyquist = 0.5 * fs # Max spectral frequency
    bandwidth = nyquist / n_freqs
    # Find middle of each frequency band, only works when
    # merging odd number of bands
    f = 1/seglen + bandwidth/2 + bandwidth*np.arange(n_freqs)
    # Detrend w segment
    wseg = detrend(wseg)
    # Taper and rescale (to preserve variance)
    taper = np.sin(np.arange(seglen) * np.pi/seglen)
    wseg_taper = wseg * taper
    # Find the correction factor (comparing old/new variance)
    corr = np.sqrt(np.var(wseg) / np.var(wseg_taper))
    wseg_corr = wseg_taper * corr
    # FFT of vertical velocity segment
    fft_win = np.fft.fft(wseg_corr)
    # Power spectrum
    ps_win = 2 * np.real(fft_win[:seglen//2] * np.conj(fft_win[:seglen//2]))
    ps_win /= (seglen*fs) # Normalize
    # Convert from f-space to k-space using Eq. (3) of George et al. (1994)
    # (Taylor's hypothesis)
    k = (2*np.pi * f) / U # Wavenumber array
    ps_k = ps_win / (2*np.pi / U) # Wavenumber spectrum
    # Check that the variance is conserved
    df = f[2] - f[1]
    dk = k[2] - k[1]
    var_f = np.sum(ps_win) * df # f-spec variance
    var_k = np.sum(ps_k) * dk # k-spec variance
    assert np.isclose(var_f, var_k)
    # Interpolate to given wavenumber range?
    if k_int is not None:
        # Interpolate spectrum to specified frequency range for averaging
        # but don't extrapolate beyond k_int range (left, right)
        ps_i = np.interp(k_int, k, ps_k, left=np.nan, right=np.nan)
        # Define output dataset variables, save spectrum and U
        data_vars = {'k_spec':(['k'], ps_i,),
                     'U': ([], U),
                     }
        # Define coordinates
        coords = {'k': (['k'], k_int)}
    else:
        # Don't interpolate spectrum
        data_vars = {'k_spec':(['k'], ps_k,),
                     'U': ([], U),
                     }
        # Define coordinates
        coords = {'k': (['k'], k)}
    # Create output dataset
    dss = xr.Dataset(data_vars=data_vars, coords=coords,)

    return dss


def dissipation_rate(k, spec, fit='curve'):
    """
    Estimate turbulence dissipation rate from wavenumber spectrum.
    Fits k^{-5/3} curve to given spectrum and computes dissipation
    rate epsilon by

        epsilon = (Cf / C)**(3/2), (1)
    
    where Cf is the intertial subrange fit coefficient and C=0.5,
    following George et al. (1994, JGR)

    Parameters:
        k - array; wavenumbers for spectrum
        spec - array; wavenumber spectrum
        fit - str; either 'curve' or 'linear'. If 'linear' fits
              linear function to log transform of data.
    
    Returns:
        epsilon - dissipation rate from (1)
        r_squared - R^2 of k^{-5-3} fit to spectrum
        coeff - fit coefficient
    """
    # Define fit functions
    def fun(x, c):
        """
        Standard curve fit to inertial subrange k^{-5/3}.
        """
        return c * x ** (-5/3)
    def funl(x, c):
        """
        Linear fit to inertial subrange with log transform.
        """
        return np.log(c) + (-5/3) * np.log(x)
    # Standard curve fit
    if fit == 'curve':
        # Fit k^-5/3 curve function
        popt, pcov = curve_fit(fun, k, spec, p0=1e-4)
        coeff = popt[0] # Fit coeff.
        # Compute R^2 of fit
        r_squared = rps.r_squared(spec, fun(k, coeff))
    # Linear fit to log transform of data
    elif fit == 'linear':
        # Fit k^-5/3 linear function in log space
        popt, pcov = curve_fit(funl, k, np.log(spec), p0=1e-4)
        coeff = popt[0] # Fit coeff.
        # Compute R^2 of fit (to log-transforms)
        r_squared = rps.r_squared(np.log(spec), np.log(fun(k, coeff)))
    # Compute dissipation rate epsilon following (1)
    # C = 1.5
    # epsilon = (coeff / (12/55*C))**(3/2)
    C = 0.5
    epsilon = (coeff / C)**(3/2)

    return epsilon, r_squared, coeff

def dissipation_rate_LT83(f, spec, U, sigma, theta=0, fit='linear',
                          fmin=None, fmax=None, min_fitlen=10, 
                          skip_f=1):
    """
    Estimate turbulence dissipation rate from frequency
    spectrum following Lumley and Terray (1983, JPO).
    This implementation follows the approach outlined by
    Trowbridge and Elgar (2001, JPO), and is based on the Matlab
    code calc_stats_iso3.m by Johanna Rosman.

    The model of the spectrum is (T&E01, Eq. (7)):

    P_ww = 12/55 * C * eps**(2/3) * U**(2/3) * omega**(-5/3) * I,

    where C=1.5 and I accounts for the effects of the surface waves 
    (Eq. (A13)).

    Parameters:
        f - array; frequency array for spectrum
        spec - array; frequency spectrum
        fit - str; either 'curve' or 'linear'. If 'linear', fits
              linear function to log transform of data.
        U - scalar; the magnitude of the current
        sigma - scalar; std of the wave-induced horizontal velocity
        theta - scalar; angle between waves and current (deg)
        fmin - scalar; optional, min. freq. for fit
        fmax - scalar; optional, max. freq. for fit
        min_fitlen - int; min. allowable length of fit (points)
        skip_f - int; step size to skip over freq. ranges when
                 finding best fit (>1 to speed up)
    
    Returns:
        epsilon - dissipation rate
        r_squared - R^2 of k^{-5-3} of best fit to spectrum
        coeff - fit coefficient
    """
    # Define fit functions
    def fun(x, c):
        """
        Standard curve fit to inertial subrange k^{-5/3}.
        """
        return c * x ** (-5/3)
    def funl(x, c):
        """
        Linear fit to inertial subrange with log transform.
        """
        return np.log(c) + (-5/3) * np.log(x)
    # Define y variable U / sigma
    y = U / sigma
    # Compute I(y, theta) following Eq. (A13) of Trowbridge
    # and Elgar (2001, JPO)
    dx = 0.001 # x spacing
    xmin, xmax = -1000, 1000 # integration x-variable limits
    nx = int(np.abs(xmin - xmax) / dx) + 1
    x = np.linspace(-1000, 1000, nx) # x variable of (A13)
    integral = (x**2 - (2/y)*x + 1/y**2)**(1/3) * np.exp(-0.5*x**2)
    I_arr = 1 / (np.sqrt(2*np.pi)) * y**(2/3) * np.sum(integral*dx)
    # Set f limits of fits if not specified
    if fmin is None:
        # Set fmin as 3*fp
        fp = f[np.argmax(spec).item()] # Peak freq. of spectrum
        fmin = 3 * fp
    if fmax is None:
        # Use last available frequency as max.
        fmax = f[-1]
    # Get nearest indices for fmin, fmax
    fmin_idx = (np.abs(f - fmin)).argmin()
    fmax_idx = (np.abs(f - fmax)).argmin()
    # Convert spectrum and freqs to radian frequencies
    spec_rad = spec / (2*np.pi)
    omega = f * 2*np.pi
    # Fit -5/3 power law to spectrum, find best fit by iterating
    # over frequency ranges
    start_inds = np.arange(fmin_idx, (fmax_idx-min_fitlen))[::skip_f]
    end_inds = np.arange((fmin_idx+min_fitlen), fmax_idx)[::skip_f]
    # Make lists of R^2, epsilon and fit coeff. of fits
    r2_list = []
    eps_list = []
    coeff_list = []
    for si in start_inds:
        for ei in end_inds:
            # Get spectral and frequency segments
            sseg = spec_rad[si:ei].copy()
            oseg = omega[si:ei].copy()
            if len(sseg) < min_fitlen:
                # Segment shorter than min. fit length -> skip
                continue
            # Fit power law to spectral segment using desired method
            if fit == 'curve':
                # Fit f^-5/3 curve function
                popt, pcov = curve_fit(fun, oseg, sseg, p0=1e-4)
                c_i = popt[0] # Fit coeff.
                # Compute R^2 of fit
                r2_i = rps.r_squared(sseg, fun(oseg, c_i))
            # Linear fit to log transform of data
            elif fit == 'linear':
                # Fit f^-5/3 linear function in log space
                popt, pcov = curve_fit(funl, oseg, np.log(sseg), p0=1e-4)
                c_i = popt[0] # Fit coeff.
                # Compute R^2 of fit (to log-transforms)
                r2_i = rps.r_squared(np.log(sseg), np.log(fun(oseg, c_i)))
            # Estimate dissipation rate as in Trowbridge & Elgar (2001)
            C = 1.5
            denom = (12/55 * C * U**(2/3) * I_arr) 
            eps_i = (c_i / denom)**(3/2)
            # Append to lists
            r2_list.append(r2_i)
            eps_list.append(eps_i)
            coeff_list.append(c_i)
    # Return parameters corresponding to highest R^2 score
    max_rsq_ind = np.argmax(r2_list) # Ind. of Max. R^2
    epsilon = eps_list[max_rsq_ind]
    r_squared = r2_list[max_rsq_ind]
    coeff = coeff_list[max_rsq_ind]

    return epsilon, r_squared, coeff


    

