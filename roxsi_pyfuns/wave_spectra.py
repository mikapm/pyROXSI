"""
Functions to estimate surface wave variance spectra.
"""

import numpy as np
from scipy.signal import detrend


def spec_uvz(z, u=None, v=None, wsec=256, fs=5.0, fmerge=3,
        fmin=0.001, fmax=None, hpfilt=False, return_freq=True,
        return_aux=False, fillvalue=None):
    """
    Returns 1D wave spectrum (y) from time series of sea
    surface elevation/heave displacements z and (not yet functional) 
    wave velocities u and v. 
    TODO: Compute co-spectra including u and v.
    
    Based on UVZwaves.m by Jim Thomson, available in
    SWIFT-codes git repository at
    https://github.com/jthomson-apluw/SWIFT-codes

    Parameters:
        z - ndarray; 1D up displacements [m]
        u - ndarray; 1D east displacements [m/s]
        v - ndarray; 1D north displacements [m/s]
        wsec - window length in seconds
        fs - sampling frequency in Hz
        fmerge - int; freq bands to merge, must be odd
        hpfilt - bool; Set to True to high-pass filter data
        return_aux - bool; Set to True to return aux dict with
                     spectral bulk parameters
    Returns:
        E - wave energy spectrum
        f - frequency arrays to go with E
        aux - dict with bulk parameters (if return_aux=True)
    """

    def rc_hpfilt(data, fs, RC=3.5):
        """
        High-pass RC filter function for sampling rate fs and time
        constant RC. Returns filtered signal.
        TODO: Optimal value for RC?
        """
        data_filt = np.zeros_like(data) # Initialize filtered array
        # Smoothing factor
        alpha = RC / (RC + 1. / fs)
        # Filter
        for ui in range(1, data.shape[1]-1):
            data_filt[:,ui] = alpha * data_filt[:,ui-1] + alpha * \
                    (data[:,ui] - data[:,ui-1])
        return data_filt

    if fillvalue is not None:
        # Remove fillvalues
        z = z[z!=fillvalue]

    npts = len(z) # Number of data points
    # Combine z, u, v into common array such that
    # z=arr[0], u=arr[1], v=arr[2]
    if u is not None and v is not None:
        zuv = np.array((z, u, v))
    else:
        zuv = np.atleast_2d(np.array(z)) # Needs to work with only z

    # Split into windows with 75% overlap
    win_len = int(round(fs * wsec)) # window length in data points
    if (win_len % 2) != 0: win_len -= 1 # Make win_len even
    # Define number of windows, the 4 comes from a 75% overlap
    n_windows = int(np.floor(4 * (npts/win_len - 1) + 1))
    dof = 2 * n_windows * fmerge # degrees of freedom
    #print('DoF: ', dof)
    n_freqs = int(np.floor(win_len / (2 * fmerge))) # No. of frequency bands

    # Minimum length and quality for further processing
    if npts >= 2*wsec and fs >= 0.5:
        # Detrend data
        zuv = detrend(zuv)
        if hpfilt:
            # High-pass RC filter data
            zuv_filt = rc_hpfilt(zuv, fs)
            zuv = zuv_filt.copy()
    else:
    #    raise ValueError('Data not long enough of fs too low.')
        return np.zeros(n_freqs)
    # Loop over z, u and v and split into windows
    for i in range(int(zuv.shape[0])):
        arr = zuv[i,:] # Select current array
        arr_win = np.zeros((n_windows, win_len)) # Initialize window array
        arr_win = np.atleast_2d(arr_win) # Needs to also work with only z
        # Loop over windows
        for q in range(1, int(n_windows)+1):
            si = int((q-1) * 0.25 * win_len) # Start index of window
            ei = int((q-1) * 0.25*win_len + win_len) # End index of window
            arr_win[q-1, :] = arr[si:ei].copy()
        # Detrend individual windows (full series already detrended
        arr_win = detrend(arr_win)
        # Taper and rescale (to preserve variance)
        # TODO: Options for different taper functions
        taper = np.sin(np.arange(win_len) * np.pi/win_len)
        arr_win_taper = arr_win * taper
        # Find the correction factor (comparing old/new variance)
        corr = np.sqrt(np.var(arr_win, axis=1) / np.var(arr_win_taper, axis=1))
        # Correct for the change in variance
        arr_win_corr = arr_win_taper * corr[:, np.newaxis]
        # Calculate Fourier coefficients
        fft_win = np.fft.fft(arr_win_corr)
        # Second half of fft is redundant, so throw it out
        fft_win = fft_win[:, :win_len//2]
        # Throw out the mean (first coef) and add a zero 
        # (to make it the right length).
        # Move 1st element last:
        fft_win = np.array([np.roll(row, -1) for row in fft_win]) 
        fft_win[:,-1] = 0 # Set last element to zero
        # Power spectra (auto-spectra)
        ps_win = np.real(fft_win * np.conj(fft_win))
        # Cross spectra
        uv_win = fft_win[1] * np.conj(fft_win[2]) # UV
        uz_win = fft_win[1] * np.conj(fft_win[0]) # UZ
        vz_win = fft_win[2] * np.conj(fft_win[0]) # VZ
        # Merge neighboring frequency bands
        #n_freqs = int(np.floor(win_len / (2 * fmerge))) # No. of frequency bands
        ps_win_merged = np.zeros((n_freqs, n_windows))
        # TODO: also compute co-spectra (can't do it inside "i" loop)
#        uv_win_merged = np.zeros((int(np.floor(win_len / (2 * fmerge))))) * 1j
#        uz_win_merged = np.zeros((int(np.floor(win_len / (2 * fmerge))))) * 1j
#        vz_win_merged = np.zeros((int(np.floor(win_len / (2 * fmerge))))) * 1j
        for mi in range(fmerge, win_len//2, fmerge):
            ps_win_merged[int(mi/fmerge-1),:] = np.mean(ps_win[:, mi-fmerge:mi])

        nyquist = 0.5 * fs # Max spectral frequency
        bandwidth = nyquist / n_freqs
        # Find middle of each frequency band, only works when
        # merging odd number of bands
        f = 1/wsec + bandwidth/2 + bandwidth*np.arange(n_freqs)

        # Ensemble average windows together:
        # Take the average of all windows at each freq band
        # and divide by N*(sample rate) to get power spectral density
        # The two is b/c we threw the redundant half of the FFT away,
        # so need to multiply the PSD by 2.
        psd = np.mean(ps_win_merged, axis=1) / (win_len/2 * fs)

    if fmax is None:
        fmax = nyquist
    f_mask = np.logical_and(f>=fmin, f<=fmax) # Mask for low/high freqs
    E = psd.copy() # Output energy spectrum
    E = E[f_mask] # Truncate too high/low frequencies
    f = f[f_mask]

    # Compute bulk parameters
    aux = {} # Initialize output aux dict
    # Significant wave height
    aux['Hs'] = 4 * np.sqrt(np.sum(E) * bandwidth)
    #print('Hs: {:.2f} m'.format(aux['Hs']))
    # Energy period
    fe = np.sum(f * E) / np.sum(E)
    aux['Ta'] = 1 / fe
    #print('Ta: {:.2f} s'.format(aux['Ta']))
    # Peak period
    aux['Tp'] = 1 / f[E==E.max()][0]
    #print('Tp: {:.2f} s'.format(aux['Tp']))

    if return_aux and return_freq:
        return E, f, aux
    elif return_freq and not return_aux:
        return E, f
    else:
        return E
