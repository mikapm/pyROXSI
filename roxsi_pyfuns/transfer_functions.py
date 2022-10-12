"""
Transfer functions for wave properties, e.g. pressure -> surface
elevation.
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import optimize


def waveno_deep(freq):
    """
    Returns peak wavelength assuming linear deep water 
    dispersion relation.
    Parameters:
        f - float; period
    Returns:
        k - wavenumber
    """
    w = 2 * pi * freq
    return Tp**2 * 9.81 / (2*pi)

def waveno_shallow(freq, d):
    """
    Returns peak wavelength assuming shallow water 
    linear dispersion relation.
    Parameters:
        freq - np.array; frequency array
        d - float; water depth
    Returns:
        k - np.array; wavenumber array
    """
    return Tp * np.sqrt(9.81 * d)

def waveno_full(freq, d, k0=None, **kwargs):
    """
    Returns peak wavelength assuming full linear 
    dispersion relation.
    Uses shallow-water wavenumber as initial guess unless 
    another value is given.
    Parameters:
        freq - np.array; frequency array
        d - float; water depth
        k0 - float; initial guess wavelength. If None, uses S-W k
             as initial guess.
        **kwargs for scipy.optimize.newton()
    Returns:
        k - np.array; wavenumber array
    """
    # Function to solve
    def f(x, T, d): 
        return ((2*pi*9.81)/x * np.tanh(2*pi*d/x) - (2*pi/T)**2)

    # Estimate shallow-water wavenumber if not given
    if k0 is None:
        k0 = waveno_shallow(freq)
    # Solve f(x) for intermediate water L using secant method
    L = optimize.newton(f, Ld, args=(Tp, d))

    return L

class TRF(self):
    """
    Pressure-to-sea surface transfer function (TRF) class.
    """
    def __init__(fs, zp, type='RBR SoloD'):
        """
        Initialize class with pressure sensor parameters.
        Parameters:
            fs - float; sampling frequency (Hz)
            zp - float; height of pt sensor from seabed
        """
        self.fs = fs
        self.zp = zp


    def p2eta_lin(self, pt, z0, fmin=0.05, fmax=0.33, M=512):
        """
        Linear transfer function from water pressure to sea-surface 
        elevation eta.

        Based on the pr_corr.m Matlab function written by Urs Neumeier
        (http://neumeier.perso.ch/matlab/waves.html).

        Parameters:
            pt - np.array; water pressure time series
            z0 - float; mean water depth (m)
            fmin - float; min. frequency for attenuation correction
            fmax - float; max. frequency for attenuation correction
            M - int; segment length (512 by default)
        Returns:
            eta - np.array; linear sea surface elevation time series
        """
        
        # Make sure pt is np.array and check that it is not empty
        pt = np.atleast_1d(pt)
        if not len(pt):
            raise ValueError('Empty pressure sensor array.')
        
        # Make sure M is even by checking the remainder
        m = len(pt)
        M = min(M, m)
        if (M % 2) != 0:
            raise ValueError('M must be even.')

        # Define length of overlapping segments and zero-padding
        # array length N
        N_ol = M/2 # length of overlap
        # length of array zero-paded to nearest multiple of M
        N = np.ceil(m/M) * M 


    
