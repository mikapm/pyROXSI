"""
Transfer functions for wave properties, e.g. pressure -> surface
elevation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


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


class TRF():
    """
    Pressure-to-sea surface transfer function (TRF) class.
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


    def p2eta_lin(self, pt, d, M=512, fmin=0.05, fmax=0.33, 
                  max_att_corr=5, detrend=True):
        """
        Linear transfer function from water pressure to sea-surface 
        elevation eta.

        Based on the pr_corr.m Matlab function written by Urs Neumeier
        (http://neumeier.perso.ch/matlab/waves.html).

        Parameters:
            pt - np.array; water pressure fluctuation time series (m)
            d - scalar; mean water depth (m)
            M - int; window segment length (512 by default)
            fmin - scalar; min. frequency for attenuation correction
            fmax - scalar; max. frequency for attenuation correction
            max_att_corr - scalar; maximum attenuation correction.
                           Should not be higher than 5.
            detrend - bool; set to False if data already detrended
        Returns:
            eta - np.array; linear sea surface elevation time series
        """
        
        # Make sure pt is np.array and check that it is not empty
        pt = np.atleast_1d(pt)
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
        N_ol = M/2 # length of overlap
        # length of array zero-padded to nearest multiple of M
        N = (np.ceil(m/M) * M ).astype(int)

        # Make frequency array (only need the first half)
        freqs = np.arange(M/2+1) * self.fs / M
        oms = freqs * np.pi*2 # Radian frequencies

        # Detrend data in overlapping segments if requested
        if detrend:
            for ss in np.arange(0, N-N_ol, N_ol).astype(int):
                ## ss - segment start index; se - segment end index
                se = min(ss + M, m);
                print('ss={}, se={}'.format(ss,se))
                # Take out segment to detrend
                seg = pt[ss:se]
                seglen = len(seg)
                # Calculate trend in segment
                x = np.arange(seglen) # Inputs
                trend = np.polyfit(x, seg, 1)
                # Calculate mean water depth
                dseg = np.polyval(trend, seglen/2)
                # Remove trend
                seg -= np.polyval(trend, x)
                # Calculate wavenumbers using full dispersion relation
                ks = waveno_full(oms, d=dseg)
                # Calculate pressure response factor Kp
                Kp = np.cosh(ks*self.zp) / np.cosh(ks*dseg)
                # Apply limits to attenuation correction (ac)
                acidx = np.logical_or(freqs<fmin, freqs>fmax)
                Kp[acidx] = 1 # No correction outside of range
                # Apply attenuation correction to desired range
                Kp[Kp < 1/max_att_corr] = 1 / max_att_corr
                # Linear decrease of correction for freqs above fmax
                Kphf = Kp[freqs>fmax].copy() # high-freq part of Kp
                # Get Kp value closest to fmax
                Kpfmax = Kp[np.where(freqs<=fmax)[0][-1]] 
                dfac = Kpfmax / len(Kphf) # Decrease factor
                Kp[freqs>fmax] = np.arange(len(Kphf))[::-1] * dfac

        return Kp, freqs



# Test script
if __name__ == '__main__':
    pt = (np.sin(2*np.linspace(0, 24*2*np.pi, 2*12*60)) + 
                1.2*np.sin(0.5*np.linspace(0, 24*2*np.pi, 2*12*60)) + 
                5)
    trf = TRF(fs=2, zp=0.08)
    Kp, freqs = trf.p2eta_lin(pt, np.mean(pt))
    
    
