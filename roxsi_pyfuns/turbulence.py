"""
Functions to compute turbulence parameters.
"""

import numpy as np
import pandas as pd
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
        dfs -  pd.DataFrame with spectrum as data and (interpolated) 
               wavenumbers as index.
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
    if k_int is not None:
        # Interpolate spectrum to specified frequency range for averaging
        # but don't extrapolate beyond k_int range (left, right)
        ps_i = np.interp(k_int, k, ps_k, left=np.nan, right=np.nan)
        # Save k-spectrum to dataframe
        dfs = pd.DataFrame(data=ps_i, index=k_int)
    else:
        # Don't interpolate spectrum
        dfs = pd.DataFrame(data=ps_k, index=k)

    return dfs


def dissipation_rate(k, spec, fit='curve'):
    """
    Estimate turbulence dissipation rate from wavenumber spectrum.
    Fits k^{-5/3} curve to given spectrum and computes dissipation
    rate epsilon by

        epsilon = (Cf / (24/55*C))**(3/2), (1)
    
    where Cf is the intertial subrange fit coefficient and C=1.5.

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
    # Define fit function
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
    C = 1.5
    epsilon = (coeff / (24/55*C))**(3/2)

    return epsilon, r_squared, coeff
