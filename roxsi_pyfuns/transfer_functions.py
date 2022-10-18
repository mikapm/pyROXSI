"""
Transfer functions for wave properties, e.g. pressure -> surface
elevation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.signal import hann


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


    def p2eta_lin(self, pp, M=512, fmin=0.05, fmax=0.33, 
                  att_corr=True, max_att_corr=5):
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
        print('N = {}'.format(N))

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
            print('ss={}, se={}'.format(ss,se))
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

            # Calculate pressure response factor Kp
            Kp = np.cosh(ks*self.zp) / np.cosh(ks*dseg)
            # Apply limits to attenuation correction (ac)
            acidx = np.logical_or(freqs<fmin, freqs>fmax)
            Kp[acidx] = 1 # No correction outside of range
            if att_corr:
                # Apply attenuation correction to desired range
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

        return eta



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
                default='/home/malila/ROXSI/Asilomar2022/SmallScaleArray/RBRDuetDT/Level1/mat',
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

        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Load ROXSI pressure sensor time series
    fn_mat = glob.glob(os.path.join(args.dr, 'roxsi_{}_L1_{}_*_{}.mat'.format(
        args.sid, args.mid, args.date)))[0]
    print('Loading pressure sensor mat file {}'.format(fn_mat))
    mat = loadmat(fn_mat)
    # Read pressure time series and timestamps
    pt = np.array(mat['DUETDT']['Pwater'].item()).squeeze()
    time_mat = np.array(mat['DUETDT']['time_dnum'].item()).squeeze()
    # Convert timestamps
    time_ind = pd.to_datetime(time_mat-719529,unit='d') 
    # Read sampling frequency and sensor height above seabed
    fs = int(mat['DUETDT']['sample_freq'].item()[0].split(' ')[0])
    zp = mat['DUETDT']['Zbed'].item().squeeze().item()
    # Make pandas DataFrame
    dfp = pd.DataFrame(data={'pt':pt}, index=time_ind)

    # Initialize class
    trf = TRF(fs=fs, zp=zp)
    # Transform pressure -> eta for 20-min chunk
    eta = trf.p2eta_lin(dfp['pt'].iloc[0:19200], M=args.M, 
                        fmin=args.fmin, fmax=args.fmax)

    
    
