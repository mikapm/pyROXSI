"""
Functions to estimate surface wave variance spectra.
"""

import numpy as np
import xarray as xr
from scipy.signal import detrend

def spec_moment(S, F, order):
    """
    Calculate nth-order moment of spectrum S. Order is specified
    by order, F is the frequency array corresponding to S. Assuming
    regular frequency spacing.
    """
    # Get frequency resolution (assuming regularly spaced F)
    df = F[1] - F[0]
    # Compute nth order moment
    return np.nansum((F**order) * S) * df

def significant_waveheight(S, F):
    """
    Returns the spectral significant wave height.
    """
    m0 = spec_moment(S, F, 0) # variance
    return 4 * np.sqrt(m0)

def peak_freq(S, F):
    """
    Calculate peak frequency of spectrum S following
    Young I. R., Ocean Eng., 22, 7 (1995).
    """
    # Get frequency resolution (assuming regularly spaced F)
    df = F[1] - F[0]
    fp = (np.nansum(F * S**4) * df) / (np.nansum(S**4) * df)

    # Compute nth order moment
    return fp

def spec_bandwidth(S, F, method='longuet'):
    """
    Compute spectral bandwidth nu following various methods described in 
    Saulnier et al. (2011, Oc. Eng.): 
    'Wave groupiness and spectral bandwidth as relevant parameters for the 
    performance assessment of wave energy converters', 

    and 

    Holthuijsen (2007): Waves in Oceanic and Coastal Waters, p. 67 for the
    Battjes and van Vledder (1984) method ('battjes').

    """
    # Compute moments
    if method == 'longuet':
        m0 = spec_moment(S, F, 0)
        m1 = spec_moment(S, F, 1)
        m2 = spec_moment(S, F, 2)
        # Longuet-Higgins (1957, 1984) method, sensitive to high frequency content
        # in spectrum due to use of m2.
        nu = np.sqrt((m0*m2) / (m1**2) - 1)
    elif method == 'mollison':
        # Mollison (1985) method, less sensitive to hig frequencies than L-H method
        m0 = spec_moment(S, F, 0)
        mminus2 = spec_moment(S, F, -2)
        mminus1 = spec_moment(S, F, -1)
        nu = np.sqrt((m0*mminus2) / (mminus1**2) - 1)
    elif method == 'smith':
        # Smith et al. (2006) method, also less sensitive to high frequencies.|
        m0 = spec_moment(S, F, 0)
        m1 = spec_moment(S, F, 1)
        mminus1 = spec_moment(S, F, -1)
        nu = np.sqrt((m1*mminus1) / (m0**2) - 1)
    elif method == 'battjes':
        # Battjes and van Vledder (1984) method, which according to Holthuijsen
        # (2007, p. 67) "is superior in several respects to epsilon and nu (see van
        # Vledder, 1992)" in characterising wave groupiness.
        df = F[1] - F[0]
        m0 = spec_moment(S, F, 0)
        m2 = spec_moment(S, F, 2)
        fm = np.sqrt(m2/m0) # Mean frequency
        nu = np.sqrt((1/m0**2) * ((np.nansum(S * np.cos(2*np.pi*F/fm)) * df)**2 +\
                (np.nansum(S * np.sin(2*np.pi*F/fm)) * df)**2))
    elif method=='goda':
        # Essentially the same envelope correlation parameter as the 'battjes' 
        # method, but defining the mean frequency in terms of m1 instead of m2, 
        # following e.g. Goda and Kudaka (2007).
        df = F[1] - F[0]
        m0 = spec_moment(S, F, 0)
        m1 = spec_moment(S, F, 1)
        fm = m1/m0 # Mean frequency
        nu = np.sqrt((1/m0**2) * ((np.nansum(S * np.cos(2*np.pi*F/fm)) * df)**2 +\
                (np.nansum(S * np.sin(2*np.pi*F/fm)) * df)**2))
    elif method=='vanvledder':
        # Another iteration of the 'battjes' parameter, this time with the mean
        # frequency modified to take into account the effect of finite spectral
        # bandwidth following Tayfun (1990). This method follows
        # GP van Vledder - Coastal Engineering 1992: Statistics of wave group
        # parameters.
        df = F[1] - F[0]
        m0 = spec_moment(S, F, 0)
        m1 = spec_moment(S, F, 1)
        m2 = spec_moment(S, F, 2)
        Tm = np.sqrt(m0/m2) # Mean frequency
        # Take finite bandwidth into account following Tayfun (1990)
        # Eq. (9.7) in van Vledder (1992)
        nu_LH = np.sqrt((m0*m2) / (m1**2) - 1)
        T_hat = Tm * (1 - 0.5 * nu_LH**2)
        nu = np.sqrt((1/m0**2) * ((np.nansum(S * np.cos(2*np.pi*F*T_hat)) * df)**2 +\
                (np.nansum(S * np.sin(2*np.pi*F*T_hat)) * df)**2))

    return nu


def spec_uvz(z, u=None, v=None, wsec=256, fs=5.0, fmerge=3,
        fmin=0.001, fmax=None, hpfilt=False, return_freq=True, 
        fillvalue=None, timestamp=None):
    """
    Returns wave spectrum from time series of sea
    surface elevation/heave displacements z and (optional) 
    wave velocities u and v. If input array is 1D (z only),
    estimates frequency spectrum. If u and v input arrays
    are included, estimates also directional moments.
    
    Based on UVZwaves.m by Jim Thomson, available in
    SWIFT-codes git repository at
    https://github.com/jthomson-apluw/SWIFT-codes

    Parameters:
        z - ndarray; 1D up displacements [m]
        u - ndarray; (optional) 1D east displacements [m/s]
        v - ndarray; (optional) 1D north displacements [m/s]
        wsec - window length in seconds
        fs - sampling frequency in Hz
        fmerge - int; freq bands to merge, must be odd
        fmin - scalar; minimum frequency for calculating bulk params
        fmax - scalar; maximum frequency for calculating bulk params
        hpfilt - bool; Set to True to high-pass filter data
        fillvalue - scalar; (optional) fill value for NaN elements
        timestamp - if not None, assigns a time coordinate using
                    given value to output dataset.
    Returns:
        ds - xarray.Dataset with spectrum and bulk parameters
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
    ndim = int(zuv.shape[0]) # Number of dimensions

    # Split into windows with 75% overlap
    win_len = int(round(fs * wsec)) # window length in data points
    if (win_len % 2) != 0: win_len -= 1 # Make win_len even
    # Define number of windows, the 4 comes from a 75% overlap
    n_windows = int(np.floor(4 * (npts/win_len - 1) + 1))
    dof = 2 * n_windows * fmerge # degrees of freedom
    #print('DoF: ', dof)
    n_freqs = int(np.floor(win_len / (2 * fmerge))) # No. of frequency bands
    # Calculate Nyquist frequency and bandwidth (freq. resolution)
    nyquist = 0.5 * fs # Max spectral frequency
    bandwidth = nyquist / n_freqs
    # Find middle of each frequency band, only works when
    # merging odd number of bands
    f = 1/wsec + bandwidth/2 + bandwidth*np.arange(n_freqs)

    # Minimum length and quality for further processing
    if npts >= 2*wsec and fs >= 0.5:
        # Detrend data component-wise
        for zi, component in enumerate(zuv):
            zuv[zi,:] = detrend(component)
        if hpfilt:
            # High-pass RC filter data
            zuv_filt = rc_hpfilt(zuv, fs)
            zuv = zuv_filt.copy()
    else:
        print('Input data not long enough or fs too low.')
        return np.zeros(n_freqs)
    # Define dictionary to store U,V,Z fft windows, if applicable
    if ndim > 1:
        order = ['z', 'u', 'v'] # Order of components
        # Output ds spectral variables
        data_vars={'Ezz': (['freq'], np.zeros(n_freqs)),
                   'Euu': (['freq'], np.zeros(n_freqs)),
                   'Evv': (['freq'], np.zeros(n_freqs)),
                  }
    else:
        order = ['z']
        data_vars={'Ezz': (['freq'], np.zeros(n_freqs)),
                  }
    fft_dict = {'{}'.format(k):np.ones(n_freqs)*np.nan for k in order}
    # Also store PSD estimate(s)
    psd_dict = {'{}{}'.format(k, k):np.ones(n_freqs)*np.nan for k in order}
    # Is a timestamp given?
    if timestamp is not None:
        time = [timestamp] # time coordinate
        # Initialize output dataset
        ds = xr.Dataset(data_vars=data_vars, 
                        coords={'freq': (['freq'], f),
                                'time': (['time'], time)},
                       )
    else:
        time = [] # No time coord.
        # Initialize output dataset
        ds = xr.Dataset(data_vars=data_vars, 
                        coords={'freq': (['freq'], f),},
                       )


    # Loop over u, v and z and split into windows
    for i, key in zip(range(int(ndim)), order):
        arr = zuv[i,:].copy() # Copy current array
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
        # Save FFT windo to dict
        fft_dict[key] = fft_win
        # Power spectra (auto-spectra)
        ps_win = np.real(fft_win * np.conj(fft_win))
        # Merge neighboring frequency bands
        ps_win_merged = np.zeros((n_freqs, n_windows))
        for mi in range(fmerge, win_len//2, fmerge):
            ps_win_merged[int(mi/fmerge-1),:] = np.mean(ps_win[:, mi-fmerge:mi])

        # Ensemble average windows together:
        # Take the average of all windows at each freq band
        # and divide by N*(sample rate) to get power spectral density
        # The two is b/c we threw the redundant half of the FFT away,
        # so need to multiply the PSD by 2.
        psd = np.mean(ps_win_merged, axis=1) / (win_len/2 * fs)
        # Save variance density spectrum to output dataset
        ds['E{}{}'.format(key, key)].values = psd

    # Get index of spectral peak
    fpind = np.where(ds['Ezz']==ds['Ezz'].max())

    # If input array is 3D, calculate cross spectra
    if ndim == 3:
        uv_win = fft_dict['u'] * np.conj(fft_dict['v']) # UV window
        uz_win = fft_dict['u'] * np.conj(fft_dict['z']) # UZ window
        vz_win = fft_dict['v'] * np.conj(fft_dict['z']) # VZ window
        # Merge frequency bands in cross spectra
        uv_win_merged = np.zeros((int(np.floor(win_len / (2 * fmerge))), n_windows)) * 1j
        uz_win_merged = np.zeros((int(np.floor(win_len / (2 * fmerge))), n_windows)) * 1j
        vz_win_merged = np.zeros((int(np.floor(win_len / (2 * fmerge))), n_windows)) * 1j
        for mi in range(fmerge, win_len//2, fmerge):
            uv_win_merged[int(mi/fmerge-1),:] = np.mean(uv_win[:, mi-fmerge:mi])
            uz_win_merged[int(mi/fmerge-1),:] = np.mean(uz_win[:, mi-fmerge:mi])
            vz_win_merged[int(mi/fmerge-1),:] = np.mean(vz_win[:, mi-fmerge:mi])
        # Ensemble average windows together
        uv = np.mean(uv_win_merged, axis=1) / (win_len/2 * fs)
        uz = np.mean(uz_win_merged, axis=1) / (win_len/2 * fs)
        vz = np.mean(vz_win_merged, axis=1) / (win_len/2 * fs)
        # Auto and cross displacement spectra, assuming deep water
        Euu = ds['Euu'] / (2 * 3.14 * f)**2  # [m^2/Hz]
        Evv = ds['Evv'] / (2 * 3.14 * f)**2  # [m^2/Hz]
        # Check factor for circular orbits
        # In deep water (only), check = unity
        check = (Euu + Evv) / ds['Ezz']
        # Quad- and co-spectra
        Quz = np.imag(uz) # [m^2/Hz], quadspec. of vert. and hor. displacements
        Cuz = np.real(uz) # [m^2/Hz], cospec. of vert. and hor. displacements
        Qvz = np.imag(vz) # [m^2/Hz], quadspec. of vert. and hor. displacements
        Cvz = np.real(vz) # [m^2/Hz], cospec. of vert. and hor. displacements
        Cuv = np.real(uv) # [m^2/Hz], cospec. of hor. displacements
        # Wave spectral moments 
        # Would use Qxz instead of Cuz etc. for actual displacements
        uu = ds['Euu'].values
        vv = ds['Evv'].values
        zz = ds['Ezz'].values
        a1 = Cuz / np.sqrt((uu + vv) * zz) # []
        ds['a1'] = (['freq'], a1)
        b1 = Cvz / np.sqrt((uu + vv) * zz) # []
        ds['b1'] = (['freq'], b1)
        a2 = (uu - vv) / (uu + vv)
        ds['a2'] = (['freq'], a2)
        b2 = 2 * Cuv / (uu + vv)
        ds['b2'] = (['freq'], b2)
        # Wave directions from Kuik et al, JPO, 1988 and Herbers et al, JTech, 2012
        # note that 0 deg is for waves headed towards positive x 
        # (EAST, right hand system)
        dir1 = np.arctan2(b1, a1) # [rad], 4 quadrant
        dir2 = np.arctan2(b2, a2) / 2 # [rad], only 2 quadrant
        spread1 = np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2)))
        spread2 = np.sqrt(abs(0.5 - 0.5*(a2*np.cos(2*dir2) + b2*np.cos(2*dir2))))
        # Spectral directions
        dirs = -180 / 3.14 * dir1 # switch from rad to deg, and CCW to CW (negate)
        dirs += 90  # rotate from eastward = 0 to northward  = 0
        # Take NW quadrant from negative to 270-360 range
        dirs[dirs < 0] += 360 
        westdirs = np.where(dirs > 180)[0]
        eastdirs = np.where(dirs < 180)[0]
        # Take reciprocals such that wave direction is FROM, not TOWARDS
        dirs[westdirs] -= 180 
        dirs[eastdirs] += 180 
        # Directional spread
        spread = 180 / 3.14 * spread1
        ds['dspr'] = (['freq'], spread)
        # Dominant direction
        Dp = dirs[fpind] # Dominant (peak) direction, use peak f
        # Screen for bad direction estimate,     
        inds = fpind + np.array([-1, 0, 1]) # pick neighboring bands
        if np.all(inds>0) and np.max(inds) <= len(dirs): 
            dirnoise = np.std(dirs[inds])
            if dirnoise > 45:
                # Directions too noisy -> save NaN
                Dp = np.nan
        else:
            # Peak freq. too low or too high
            Dp = np.nan

    # If not specified, use Nyquist frequency as max. frequency
    if fmax is None:
        fmax = nyquist
    # Compute bulk parameters
    f_mask = np.logical_and(f>=fmin, f<=fmax) # Mask for low/high freqs
    E = ds['Ezz'].values.copy() # Output energy (variance) spectrum
    E = E[f_mask] # Truncate too high/low frequencies
    f = f[f_mask]
    # Save scalar variables with time coordinate if specified
    if timestamp is not None:
        # Peak direction
        ds['Dp_ind'] = (['time'], np.atleast_1d(Dp))
        # Significant wave height
        ds['Hm0'] = (['time'], np.atleast_1d(4*np.sqrt(np.sum(E)*bandwidth)))
        # Energy period
        fe = np.sum(f * E) / np.sum(E)
        ds['Te'] = (['time'], np.atleast_1d(1 / fe))
        # Peak period (by index of max. energy)
        ds['Tp_ind'] = (['time'], np.atleast_1d(1/f[E==E.max()][0]))
        # Peak period following Young (1995)
        fpy = peak_freq(E, f)
        ds['Tp_Y95'] = (['time'], np.atleast_1d(1 / fpy))
        # Peak direction at Y95 peak freq.
        indpy = (np.abs(f - fpy)).argmin()
        Dpy = dirs[indpy]
        ds['Dp_Y95'] = (['time'], np.atleast_1d(Dpy))
        # Spectral bandwidth following Longuet-Higgins (1957)
        ds['nu_LH57'] = (['time'], np.atleast_1d(spec_bandwidth(E, f)))
    else:
        # Peak direction
        ds['Dp_ind'] = ([], Dp)
        # Significant wave height
        ds['Hm0'] = ([], 4 * np.sqrt(np.sum(E)*bandwidth))
        # Energy period
        fe = np.sum(f * E) / np.sum(E)
        ds['Te'] = ([], 1 / fe)
        # Peak period
        ds['Tp_ind'] = ([], 1/f[E==E.max()][0])
        # Peak period following Young (1995)
        fpy = peak_freq(E, f)
        ds['Tp_Y95'] = ([], (1 / fpy))
        # Peak direction at Y95 peak freq.
        indpy = (np.abs(f - fpy)).argmin()
        Dpy = dirs[indpy]
        ds['Dp_Y95'] = ([], Dpy)
        # Spectral bandwidth following Longuet-Higgins (1957)
        ds['nu_LH57'] = ([], spec_bandwidth(E, f))

    # Save fmin, fmax in scalar variable attributes
    ds['Hm0'].attrs['fmin'] = fmin
    ds['Hm0'].attrs['fmax'] = fmax
    ds['Te'].attrs['fmin'] = fmin
    ds['Te'].attrs['fmax'] = fmax
    ds['Tp_ind'].attrs['fmin'] = fmin
    ds['Tp_ind'].attrs['fmax'] = fmax
    ds['Tp_Y95'].attrs['fmin'] = fmin
    ds['Tp_Y95'].attrs['fmax'] = fmax
    ds['Dp_Y95'].attrs['fmin'] = fmin
    ds['Dp_Y95'].attrs['fmax'] = fmax
    ds['nu_LH75'].attrs['fmin'] = fmin
    ds['nu_LH75'].attrs['fmax'] = fmax

    return ds
