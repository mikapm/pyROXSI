"""
Functions to estimate surface wave variance spectra.
"""

import numpy as np
import xarray as xr
from scipy.signal import detrend, windows
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
    k = optimize.newton(f, k0, args=(omega, d), **kwargs)

    return k

def dirs_nautical(dtheta=2, recip=False):
    """
    Make directional array in Nautical convention (compass dir from).
    """
    # Convert directions to nautical convention (compass dir FROM)
    # Start with cartesian (a1 is positive east velocities, b1 is positive north)
    theta = -np.arange(-180, 179, dtheta)  
    # Rotate, flip and sort
    theta += 90
    theta[theta < 0] += 360
    if recip:
        westdirs = (theta > 180)
        eastdirs = (theta < 180)
        # Take reciprocals such that wave direction is FROM, not TOWARDS
        theta[westdirs] -= 180
        theta[eastdirs] += 180

    return theta


def k_rms(h0, f, P, B, two_sided=True):
    """
    Compute root-mean-square wavenumber following Herbers et al. (2002)
    definition (Eq. 12). Function implementation based on fun_compute_rms.m
    function by Kevin Martins.
    
    Parameters:
        h0 - scalar; mean water depth
        f - frequency array [Hz]
        P - 1D array; power spectrum [m^2]. 
        B - 2D array; bispectrum [m^2]. 
        two_sided - bool; if True, input frequencies are 2-sided, 
                    else 1-sided.

    Returns:
        krms - array or RMS wavenumbers, same shape as f

    **Note: if two_sided=True: f, P and B arrays are two-sided in regards to f; a
            f should be centered around 0 Hz.
    """
    # Initialisation
    krms = np.ones_like(f) * np.nan
    if two_sided:
        # 2-sided frequencies
        nmid = int(len(f) / 2) # Middle frequency (f=0)
    else:
        # 1-sided frequencies
        nmid= 0
    g = 9.81 # Gravity

    df = abs(f[1]-f[0])
    # Transforming P and B to densities
    if two_sided:
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


def spec_uvz(z, u=None, v=None, wsec=256, fs=5.0, dth=2, fmerge=3,
             fmin=0.001, fmax=None, depth=None, hpfilt=False,  
             fillvalue=None):
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
        dth - directional resolution in degrees
        fmerge - int; freq bands to merge, must be odd
        fmin - scalar; minimum frequency for calculating bulk params
        fmax - scalar; maximum frequency for calculating bulk params
        depth - scalar; mean water depth of segment. Required for computing
                wave energy fluxes.
        hpfilt - bool; Set to True to high-pass filter data
        fillvalue - scalar; (optional) fill value for NaN elements

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

    g = 9.81 # gravity (m s^-2)
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
    if ndim == 3:
        order = ['z', 'u', 'v'] # Order of components
        direction = dirs_nautical(dtheta=dth)
        n_dirs = len(direction)
        # Output ds spectral variables
        data_vars={'Ezz': (['freq'], np.zeros(n_freqs)),
                   'Euu': (['freq'], np.zeros(n_freqs)),
                   'Evv': (['freq'], np.zeros(n_freqs)),
                   'Ein': (['freq'], np.zeros(n_freqs)),
                   'Eout': (['freq'], np.zeros(n_freqs)),
                   'Efth': (['freq', 'direction'], np.zeros((n_freqs, n_dirs))),
                   }
        if depth is not None:
            data_vars['Ein'] = (['freq'], np.zeros(n_freqs))
            data_vars['Eout'] = (['freq'], np.zeros(n_freqs))
    else:
        order = ['z']
        data_vars={'Ezz': (['freq'], np.zeros(n_freqs)),
                }
    fft_dict = {'{}'.format(k):np.ones(n_freqs)*np.nan for k in order}
    # Initialize output dataset
    if ndim == 3:
        ds = xr.Dataset(data_vars=data_vars, 
                        coords={'freq': (['freq'], f),
                                'direction': (['direction'], np.sort(direction)),
                                },
                        )
    else:
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
        # Save FFT window to dict
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

    if depth is not None:
        # Compute linear wavenumbers and group speed
        k = waveno_full(omega=2*np.pi*f, d=depth) # linear wavenumbers 
        # Compute group velocity per frequency/wavenumber
        Cg = 0.5 * (2*np.pi*f) * k * (1 + (2*k*depth) / np.sinh(2*k*depth))

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
        check = (Euu + Evv) / ds['Ezz'].values
        # Quad- and co-spectra
        Quz = np.imag(uz) # [m^2/Hz], quadrature-spec. of vert. and hor. displacements
        Cuz = np.real(uz) # [m^2/Hz], cospec. of vert. and hor. displacements
        Qvz = np.imag(vz) # [m^2/Hz], quadspec. of vert. and hor. displacements
        Cvz = np.real(vz) # [m^2/Hz], cospec. of vert. and hor. displacements
        Quv = np.imag(uv) # [m^2/Hz], quadspec. of vert. and hor. displacements
        Cuv = np.real(uv) # [m^2/Hz], cospec. of hor. displacements
        # Calculate coherence and phase
        ds['coh_uz'] = (['freq'], np.abs(uz)**2 / (ds['Euu'].values * ds['Ezz'].values))
        ds['coh_vz'] = (['freq'], np.abs(vz)**2 / (ds['Evv'].values * ds['Ezz'].values))
        ds['coh_uv'] = (['freq'], np.abs(uv)**2 / (ds['Euu'].values * ds['Evv'].values))
        ds['ph_uz'] = (['freq'], np.angle(uz))
        ds['ph_vz'] = (['freq'], np.angle(vz))
        ds['ph_uv'] = (['freq'], np.angle(uv))
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
        # Directional spread (function of frequency)
        spread = 180 / 3.14 * spread1
        # Dominant direction
        Dp = dirs[fpind] # Dominant (peak) direction, use peak f
        # Screen for bad direction estimate,     
        inds = fpind + np.array([-1, 0, 1]) # pick neighboring bands
        if np.all(inds>0) and np.max(inds) < len(dirs): 
            dirnoise = np.std(dirs[inds])
            if dirnoise > 45:
                # Directions too noisy -> save NaN
                Dp = np.nan
        else:
            # Peak freq. too low or too high
            Dp = np.nan

        # Estimate directional spectrum with Maximum Entropy Method of
        # Lygre and Krogstad (1986)
        ds_efth = MEM_directionalestimator(E=ds.Ezz.values, F=f, a1=a1, a2=a2, 
            b1=b1, b2=b2, dtheta=dth, fmin=fmin, fmax=fmax, convert=False)
        # Save to output ds
        ds['Efth'].values = ds_efth.Efth.values
        # Sort directions of Efth
        ds = ds.sortby('direction')

        # Estimate energy flux (m^3 s^-1 Hz^-1 ) 
        # Spectral estimator from T.H.C. Herbers, using LFDT group velocity 
        # **OR** shallow water:
        # robust to partial standing waves, but assumes shore normal propagation
        if depth is not None:
            # Cg = np.sqrt(g * depth) # shallow water version, see Sheremet et al, 2002
            # energy flux (by freq) in cartesian coordinates (assumes X propagation 
            # dominates... valid near beach)
            Ein  = (1/4 * Cg * (np.abs(zz) + ((2*np.pi * f/(g*k))**2 ) * np.abs(uu) + 
                    2 * (2*np.pi * f/(g*k)) * np.real(uz)))
            Eout = (1/4 * Cg * (np.abs(zz) + ((2*np.pi* f /(g*k))**2 ) * np.abs(uu) - 
                    2 * (2*np.pi * f/(g*k)) * np.real(uz)))

    # If not specified, use Nyquist frequency as max. frequency
    if fmax is None:
        fmax = nyquist
    # Compute bulk parameters
    f_mask = np.logical_and(f>=fmin, f<=fmax) # Mask for low/high freqs
    E = ds['Ezz'].values.squeeze().copy() # Output energy (variance) spectrum
    E = E[f_mask] # Truncate too high/low frequencies
    f = f[f_mask]
    
    # Save scalar variables to output dataset
    ds['Hm0'] = ([], 4 * np.sqrt(np.sum(E)*bandwidth)) # Sig. waveheight
    # Energy period
    fe = np.sum(f * E) / np.sum(E)
    ds['Te'] = ([], 1 / fe)
    # Peak period
    ds['Tp_ind'] = ([], 1/f[E==E.max()][0])
    # Peak period following Young (1995)
    fpy = peak_freq(E, f)
    ds['Tp_Y95'] = ([], (1 / fpy))
    # Spectral bandwidth following Longuet-Higgins (1957)
    ds['nu_LH57'] = ([], spec_bandwidth(E, f))
    # Group speed and linear wavenumber
    if depth is not None:
        ds['Cg'] = (['freq'], Cg)
        ds['k_lin'] = (['freq'], k)
    # Directional params
    if ndim == 3:
        # Energy directions per frequency
        ds['dirs_freq'] = (['freq'], dirs)
        # Dir. spread per frequency
        ds['dspr_freq'] = (['freq'], spread)
        # Peak direction
        ds['Dp_ind'] = ([], np.atleast_1d(Dp).squeeze().item())
        # DSPR at peak freq. (average about neighboring freqs)
        ifp = np.argwhere(ds.Ezz.values == ds.Ezz.max().item()).item()
        ds['dspr_fp'] = ([], ds.dspr_freq.isel(freq=slice(ifp-1, ifp+2)).mean().item())
        # Peak direction at Y95 peak freq.
        indpy = (np.abs(f - fpy)).argmin()
        Dpy = dirs[indpy]
        ds['Dp_Y95'] = ([], Dpy)
        # Save directional spread and mean direction
        ds['dspr'] = ([], ds_efth.dspr.item())
        ds['mdir'] = ([], ds_efth.mdir.item())
        if depth is not None:
            # Shoreward/seaward energy fluxes
            ds['Ein'] = (['freq'], Ein)
            ds['Eout'] = (['freq'], Eout)

    # Save fmin, fmax in scalar variable attributes
    ds['Hm0'].attrs['fmin'] = fmin
    ds['Hm0'].attrs['fmax'] = fmax
    ds['Te'].attrs['fmin'] = fmin
    ds['Te'].attrs['fmax'] = fmax
    ds['Tp_ind'].attrs['fmin'] = fmin
    ds['Tp_ind'].attrs['fmax'] = fmax
    ds['Tp_Y95'].attrs['fmin'] = fmin
    ds['Tp_Y95'].attrs['fmax'] = fmax
    ds['nu_LH57'].attrs['fmin'] = fmin
    ds['nu_LH57'].attrs['fmax'] = fmax
    if ndim == 3:
        ds['Dp_ind'].attrs['fmin'] = fmin
        ds['Dp_ind'].attrs['fmax'] = fmax
        ds['Dp_Y95'].attrs['fmin'] = fmin
        ds['Dp_Y95'].attrs['fmax'] = fmax
        ds['dspr'].attrs['fmax'] = fmax
        ds['dspr'].attrs['fmin'] = fmin
        ds['mdir'].attrs['fmax'] = fmax
        ds['mdir'].attrs['fmin'] = fmin

    return ds


def MEM_directionalestimator(E, F, a1, a2, b1, b2, convert=False, 
                             dtheta=2, fmin=None, fmax=None):
    """
    This function calculates the Maximum Entropy Method estimate of
    the Directional Distribution of a wave field.
    
    NOTE: The normalized directional distribution array (NS) and the Energy
    array (NE) have been converted to a geographic coordinate frame in which
    direction is direction from.
    
     Version: 1.1 - 5/2003,      Paul Jessen, NPS
     *** altered, 2/2005 ****    Jim Thomson, WHOI
     Python version, 1/2023      Mika Malila, UNC
    
     Written by: Paul F. Jessen
                 Department of Oceanography
                 Naval Postgraduate School
                 Monterey, CA
    
    DESCRIPTION:
    Function calculates directional distribution of a wave field using the 
    Maximum Entropy Method of Lygre & Krogstad (JPO V16 1986). User passes the
    directional moments (a1,b1,a2,b2) and energy density (en) to the function.
    The directional moments are expected to be in a right hand coordinate 
    system (i.e. north, west) with direction being the direction towards.
    The returned energy and directional distributions have been converted to
    nautical convention with direction being the direction from.
    
    Parameters:
        E - np.array; Wave frequency spectrum
        F - np.array; Frequency array
        a1 - np.array; First directional Fourier moment (function of frequency)
        b1 - np.array; Second directional Fourier moment (function of frequency)
        a2 - np.array; Third directional Fourier moment (function of frequency)
        b2 - np.array; Fourth directional Fourier moment (function of frequency)
        convert - bool; If True, convert directions to geographical coordinates
        dtheta - scalar; directional resolution (degrees)
        fmin - scalar; Min. frequency for directional spread calculation
        fmax - scalar; Max. frequency for directional spread calculation
    
    Returns:
        ds - xr.Dataset with directional spectrum
    """
    # Switch to Krogstad notation
    d1 = a1.copy()
    d2 = b1.copy()
    d3 = a2.copy()
    d4 = b2.copy()
    en = E.copy()
    freq = F.copy()
    c1 = d1 + d2 * 1j
    c2 = d3 + d4 * 1j
    p1 = (c1 - c2 * np.conj(c1)) / (1 - np.abs(c1)**2)
    p2 = c2 - c1 * p1
    x1 = 1 - p1 * np.conj(c1) - p2 * np.conj(c2)
    
    # Define directional domain, this is still in Datawell convention
    direc = np.arange(0, 359.9, dtheta)
    ndir = len(direc)
    nfreq = len(en)
    # get distribution with "dtheta" degree resolution (still in right hand system)
    dr = np.pi / 180
    tot = 0
    # Initialise directional distribution D
    D = np.zeros((nfreq, ndir)).astype(complex)
    for n in range(ndir):
        alpha = direc[n] * dr
        e1 = np.cos(alpha) - np.sin(alpha) * 1j
        e2 = np.cos(2 * alpha) - np.sin(2 * alpha) * 1j
        y1 = np.abs(1 - p1 * e1 - p2 * e2)**2
        # S(:, n) is the directional distribution across all frequencies (:)
        # and directions (n).
        D[:, n] = x1 / y1
    D = np.real(D)

    # Normalize each frequency band by the total across all directions
    # so that the integral of D(theta:f) is 1. Dn is the normalized directional
    # distribution
    Dn = np.zeros_like(D)
    tot = np.sum(D,1) * dtheta
    for ii in range(nfreq):
        Dn[ii, :] = D[ii, :] / tot[ii]

    # Calculate energy density by multiplying the energies at each frequency
    # by the normalized directional distribution at that frequency
    E = np.zeros_like(Dn)
    for ii in range(nfreq):
        E[ii, :] = Dn[ii, :] * en[ii]

    if convert:
        # Convert to a geographic coordinate frame
        ndirec = np.abs(direc - 360)
        # Convert from direction towards to direction from
        ndirec += 180
        ia = np.where(ndirec >= 360)[0]
        ndirec[ia] -= 360

        # the Energy and distribution (s) arrays now don't go from 0-360.
        # They now go from 180-5 and then from 360-185. Create new Energy and
        # distribution matrices that go from 0-360.
        NE = np.zeros_like(E)
        ND = np.zeros_like(Dn)
        for ii in range(ndir):
            ia = np.where(ndirec==direc[ii])[0]
            if len(ia) > 0:
                NE[:, ii] = E[:, ia.item()]
                ND[:, ii] = Dn[:, ia.item()]
    else:
        NE = E.copy()
        ND = Dn.copy()

    # Convert directions to nautical convention (compass dir FROM)
    theta = dirs_nautical(dtheta=dtheta, recip=False)  

    # Sort spectrum according to new directions
    dsort = np.argsort(theta)
    NE = NE[:, dsort]
    theta = np.sort(theta)


    # Save output to xr.Dataset
    data_vars={'Efth': (['freq', 'direction'], NE),}
    ds = xr.Dataset(data_vars=data_vars, 
                    coords={'freq': (['freq'], freq),
                            'direction': (['direction'], theta)},
                   )
    # Compute mean direction and directional spread and save to dataset
    dspr, mdir = mspr(ds, key='Efth', fmin=fmin, fmax=fmax)
    ds['mdir'] = ([], mdir)
    ds['dspr'] = ([], dspr)

    return ds


def mspr(spec_xr, key='Efth', norm=False, fmin=None, fmax=None):
    """
    Mean directional spread following Kuik (1988), 
    coded by Jan Victor Björkqvist

    Use norm=True with model data (eg WW3) (?)
    """
    theta = np.deg2rad(spec_xr.direction.values)
    dD = 360 / len(theta)
    if norm:
        # Normalizing here so that integration over direction becomes summing
        efth = spec_xr[key] * dD * np.pi/180
    else:
        efth = spec_xr[key]
    ef = efth.sum(dim='direction')  # Omnidirection spectra
    eth = efth.integrate(coord='freq')  # Directional distribution
    m0 = ef.sel(freq=slice(fmin, fmax)).integrate(coord='freq').item()
    # Function of frequency:
    c1 = ((np.cos(theta) * efth).sel(freq=slice(fmin, fmax)).sum(dim='direction'))  
    s1 = ((np.sin(theta) * efth).sel(freq=slice(fmin, fmax)).sum(dim='direction'))
    a1m = c1.integrate(coord='freq').values / m0  # Mean parameters
    b1m = s1.integrate(coord='freq').values / m0
    thetam = np.arctan2(b1m, a1m)
    m1 = np.sqrt(b1m**2 + a1m**2)
    # Mean directional spread
    sprm = (np.sqrt(2 - 2*(m1)) * 180/np.pi).item()
    # Mean wave direction
    dirm = (np.mod(thetam * 180/np.pi, 360)).item()
    return sprm, dirm 


def bispectrum_martins(x, fs, h0, fp=None, nfft=None, overlap=75, wind='rectangular', 
                       mg=5, return_krms=True):
    """
    Compute the bispectrum B of signal x using FFT-based approach.
    Based on fun_compute_bispectrum.m by Kevin Martins.

    Also computes bicoherence Bc and biphase Bp following Kim and Powers (1979): 
    "Digital bispectral analysis and its application to nonlinear wave interactions"; 
    see also e.g. Guedes et al. (2013, JGR): "Observations of wave energy fluxes 
    and swash motions on a slow-sloping, dissipative beach".

    NB: Bicoherence normalization does not work properly, so use the function
    bispectrum() instead (below). The bispectrum() function is also a lot faster.

    Parameters:
        x - 1D array; input signal (hydrostatic sea surface)
        fs - scalar; sampling frequency
        h0 - scalar; mean water depth [m]
        fp - scalar; peak frequency. If None, computes it from spectrum of x.
        nfft - scalar; fft length (default 512*fs)
        overlap - scalar; percentage overlap
        wind - str; Type of window for tappering (only 'rectangular'
               implemented)
        mg - scalar; length of spectral bandwith over which to merge
        return_krms - bool; if True, compute rms wavenumbers following 
                      Herbers et al. (2002).

    Returns:
        dsb - xr.Dataset with bispectral information. See code for details.
    """
    lx = len(x) # Length of input signal
    # Compute shallowness and nonlinearity parameters
    if fp is None:
        dss = spec_uvz(x, fs=fs)
        fp = 1 / dss.Tp_Y95.item() # Peak frequency following Young (1995)
    kp = waveno_full(2*np.pi*fp, d=h0).item()
    mu = (kp * h0)**2
    mu_info = 'Shallowness parameter'
    eps = 2 * np.nanstd(x) / h0
    eps_info = 'Nonlinear "amplitude" parameter'
    Ur = eps / mu
    Ur_info  = 'Ursell parameter'

    # Nonlinear moderately dispersive reconstruction 
    if nfft is None:
        nfft = 512 * fs
    overlap = min(99, max(overlap, 0))
    nfft -= np.remainder(nfft, 2)
    eadvance = int(np.fix(nfft * overlap / 100))
    nadvance = int(nfft - eadvance)
    nblock   = int(np.fix((lx-eadvance)/nadvance)+1) # +1 for not throwing away data
    freqs = np.arange(-nfft/2, nfft/2) / nfft * fs
    df = freqs[1] - freqs[0]

    # Initialize arrays
    P = np.zeros(nfft+1) # Power spectrum [m^2]
    B = np.zeros((nfft+1, nfft+1)).astype(complex) # Bispectrum [m^3]
    Bcd1 = np.zeros((nfft+1, nfft+1)).astype(float) # Denominator 1 for bicoherence
    Bcd2 = np.zeros((nfft+1, nfft+1)).astype(float) # Denominator 2 for bicoherence
    # print('freqs={}, len(freqs)={}, df={}, P={}, B={}'.format(
    #     freqs, len(freqs), df, P.shape, B.shape))

    # Initialization
    A = np.zeros((nfft+1, nblock)).astype(complex) # Fourier coeffs for each block
    nmid = int(nfft / 2) # Middle frequency index (f = 0)
    locseg = np.arange(nfft) # Indices for first block

    # Computing FFT (loop over blocks)
    for kk in range(nblock-2):
        # print('kk: ', kk)
        # Preparing block kk timeseries
        # For the rectangular window, we force a certain continuity between blocks
        xseg = x[locseg]
        xseg = detrend(xseg) # Detrend
        xseg -= np.mean(xseg) # De-mean
        if wind == 'rectangular':
            # Trying to make it periodic
            count = 0
            while abs(xseg[-1]-xseg[0]) > 0.2*np.std(xseg):
                # Updating locseg
                if kk == 0:
                    locseg += 1
                else:
                    locseg -= 1
                # Updating xseg
                xseg = x[locseg]
                count += 1
            if count > 1:
                xseg = detrend(xseg) # Detrend
                xseg -= np.mean(xseg) # De-mean
            # Smoothing both the timeseries' head and tail
            beg_ts = xseg[:2*fs] 
            end_ts = xseg[-2*fs:]
            merged_ts0 = np.concatenate([end_ts, beg_ts])
            merged_ts = merged_ts0.copy()
            dti = int(np.round(fs/8))
            for tt in range(dti, len(merged_ts)-dti-1):
                merged_ts[tt] = np.mean(merged_ts0[tt-dti:tt+dti+1])
                # print('m: ', merged_ts0[tt-dti:tt+dti+1])
            xseg[:2*fs] = merged_ts[-2*fs:]
            xseg[-2*fs:] = merged_ts[:2*fs]
            
            # Final windowing
            ww = windows.boxcar(nfft) 
            normFactor = np.mean(ww**2)
            xseg *= (ww / np.sqrt(normFactor))

            # FFT of segment
            A_loc = np.fft.fft(xseg , nfft) / nfft
            A[:,kk] = np.concatenate([A_loc[nmid:nfft], A_loc[:nmid+1]]) # FFTshift
            A[nmid,kk] = 0

            # Indices for next block
            locseg += nadvance
        
    # Last block, to not throw out data
    kk = nblock - 1
    locseg = np.arange(len(x)-nfft,len(x))
    xseg = x[locseg]
    xseg = detrend(xseg)
    xseg -= np.mean(xseg)
    if wind == 'rect':
        # Trying to make it periodic
        count = 0
        while abs(xseg[-1]-xseg[0]) > 0.2*np.std(xseg):
            # Updating locseg
            locseg -= 1
            # Updating xseg
            xseg = x[locseg]
            count += 1
        if count > 1:
            xseg = detrend(xseg)
            xseg -= np.mean(xseg)
        # Smoothing both the timeseries' head and tail
        beg_ts = xseg[:2*fs] 
        end_ts = xseg[-2*fs:]
        merged_ts0 = np.concatenate([end_ts, beg_ts])
        merged_ts = merged_ts0.copy()
        dti = int(np.round(fs/8))
        for tt in range(dti, len(merged_ts)-dti-1):
            merged_ts[tt] = np.mean(merged_ts0[tt-dti:tt+dti+1])
        xseg[:2*fs] = merged_ts[-2*fs:]
        xseg[-2*fs:] = merged_ts[:2*fs]
        
        # Final windowing
        ww = windows.boxcar(nfft) 
        normFactor = np.mean(ww**2)
        xseg *= (ww / np.sqrt(normFactor))

        # FFT of segment
        A_loc = np.fft.fft(xseg , nfft) / nfft
        A[:,kk] = np.concatenate([A_loc[nmid:nfft], A_loc[:nmid+1]]) # FFTshift
        A[nmid,kk] = 0

    #  ------------------- Bispectrum computation ---------------------
    # Deal with f1 + f2 = f3 indices
    ifr1, ifr2 = np.meshgrid(np.arange(nfft+1), np.arange(nfft+1))
    ifr3 = nmid + (ifr1-nmid) + (ifr2-nmid) 
    ifm3val = np.logical_and((ifr3 >= 0), (ifr3 < nfft+1))
    ifr3[~ifm3val] = 0

    # Accumulating triple products (loop over blocks)
    for kk in range(nblock):
        # Block kk FFT
        A_loc  = A[:,kk]
        CA_loc = np.conj(A[:,kk])
        # Compute bispectrum, bicoherence and PSD
        B += A_loc[ifr1] * A_loc[ifr2] * CA_loc[ifr3]
        P += np.abs(A_loc**2)
        # Bicoherence denominator
        Bcd1 += np.abs(A_loc[ifr1]*A_loc[ifr2])**2 
        Bcd2 += np.abs(A_loc[ifr3])**2

    # Expected values (i.e. averages)
    B /= nblock
    B[~ifm3val] = 0
    Bcd1 /= nblock
    Bcd1[~ifm3val] = 0
    Bcd2 /= nblock
    Bcd2[~ifm3val] = 0
    P /= nblock
    if np.any(Bcd1==0):
        print('Zeros in Bcd1: ', np.sum(Bcd1==0))
    if np.any(Bcd2==0):
        print('Zeros in Bcd2: ', np.sum(Bcd2==0))

    # Normalize bicoherence
    Bc = np.abs(B) / np.sqrt((Bcd1 * Bcd2))

    #  ------------------- Skewness and asymmetry ---------------------
    #  Notes: 
    #         Computation following Elgar and Guza (1985)
    #         Observations of bispectra of shoaling surface gravity wave
    #         Journal of Fluid Mechanics, 161, 425-448

    # We work only in one frequency quadrant
    ifreq  = np.arange(nmid, nmid+int((nfft)/2))
    sumtmp = 6 * B[nmid, nmid] # Initialisation with first diagonal term

    # Loop over frequencies 
    for indf in ifreq[1:]:
        # Diagonal
        sumtmp += 6 * B[indf, indf]
        # Non-diagonal
        sumtmp += 12 * np.sum(B[indf, np.arange(nmid, indf)])

    # Skewness & asymmetry parameters
    Sk = np.real(sumtmp) / np.mean((x-np.mean(x))**2)**(3/2)
    As = np.imag(sumtmp) / np.mean((x-np.mean(x))**2)**(3/2)
    # print('Sk={}, As={}'.format(Sk, As))

    # ------------------- Merging over frequencies -------------------
    # Initialization
    mg = int(mg - np.remainder(mg+1, 2))
    mm = int((mg - 1) / 2) # Half-window for averaging
    nmid = int(nfft/2) # Middle frequency (f = 0)
    ifrm = np.concatenate([np.arange(nmid,1+mm-mg,-mg)[::-1], 
                           np.arange(nmid+mg, nfft+1-mm, mg)]) # Frequency indices
    ifrm = ifrm[1:] # Lose first negative frequency (to comply with Matlab code)
    Bm = np.zeros((len(ifrm), len(ifrm))).astype(complex) # Merged bispec (unit m^3)
    Bcm = np.zeros((len(ifrm), len(ifrm))).astype(float) # Merged bicoh (real)
    Pm = np.zeros(len(ifrm)) # Merged PSD (unit m^2)

    # Remove half of diagonals
    for ff in range(len(ifreq)):
        B[ff,ff] = 0.5 * B[ff,ff]
        Bc[ff,ff] = 0.5 * Bc[ff,ff]

    # Loop over frequencies
    for jfr1 in range(len(ifrm)):
        # Rows
        ifb1 = ifrm[jfr1] # mid of jfr1-merge-block
        # PSD
        Pm[jfr1] = np.sum(P[np.arange(ifb1-mm, ifb1+mm)])
        # Columns for bispectrum and bicoherence
        for jfr2 in range(len(ifrm)):
            ifb2 = ifrm[jfr2] # mid of jfr2-merge-block
            Bm[jfr1,jfr2] = np.sum(np.sum(B[np.arange(ifb1-mm, ifb1+mm), 
                                            np.arange(ifb2-mm, ifb2+mm)]))
            Bcm[jfr1,jfr2] = np.sum(np.sum(Bc[np.arange(ifb1-mm, ifb1+mm), 
                                              np.arange(ifb2-mm, ifb2+mm)]))

    # Updating arrays
    freqs = freqs[ifrm]
    df = np.abs(freqs[1]-freqs[0])

    # Compute DoF as 2 * n_windows * n_freqs_merged
    dof = 2 * nblock * (nfft/len(ifrm))
    # The 95% significance level on zero bicoherence following Haubrich (1965)
    b95 = np.sqrt(6 / dof)

    # Generate output dataset
    data_vars={'B': (['freqx', 'freqy'], Bm),
               'Bc': (['freqx', 'freqy'], Bcm),
               'PST': (['freqx'], Pm),
               'fp': ([], fp),
               'kp': ([], kp),
               'h0': ([], h0),
               'mu': ([], mu),
               'eps': ([], eps),
               'Ur': ([], Ur),
               'Sk': ([], Sk),
               'As': ([], As),
               'DoF': ([], dof),
               'b95': ([], b95),
               }
    # Initialize output dataset
    dsb = xr.Dataset(data_vars=data_vars, 
                        coords={'freqx': (['freqx'], freqs),
                                'freqy': (['freqy'], freqs),
                            },
                    )
    if return_krms:
        # Also compute rms wavenumbers K_rms following Herbers et al. (2002)
        krms = k_rms(h0=h0, f=freqs, P=Pm, B=Bm)
        dsb['k_rms'] = (['freq'], krms)
        dsb['k_rms'].attrs['standard_name'] = 'sea_surface_wave_rms_wavenumber'
        dsb['k_rms'].attrs['long_name'] = 'Root-mean-square wavenumbers following H02'
        dsb['k_rms'].attrs['units'] = '1/m'
    # Save some attributes
    dsb.freq.attrs['standard_name'] = 'sea_surface_wave_frequency'
    dsb.freq.attrs['long_name'] = 'spectral frequencies in Hz'
    dsb.freq.attrs['units'] = 'Hz'
    dsb['B'].attrs['standard_name'] = 'bispectrum'
    dsb['B'].attrs['long_name'] = 'Bispectrum following Matlab code by Kevin Martins'
    dsb['B'].attrs['units'] = 'm^3'
    dsb['PST'].attrs['standard_name'] = 'power_spectrum'
    dsb['PST'].attrs['long_name'] = 'Power spectrum following Matlab code by Kevin Martins'
    dsb['PST'].attrs['units'] = 'm^2'
    dsb['fp'].attrs['standard_name'] = 'peak_frequency'
    dsb['fp'].attrs['long_name'] = 'Peak wave frequency'
    dsb['fp'].attrs['units'] = 'Hz'
    dsb['kp'].attrs['standard_name'] = 'peak_wavenumber'
    dsb['kp'].attrs['long_name'] = 'Peak wavenumber (linear wave theory)'
    dsb['kp'].attrs['units'] = 'rad/m'
    dsb['h0'].attrs['standard_name'] = 'depth'
    dsb['h0'].attrs['long_name'] = 'Mean water depth of signal segment'
    dsb['h0'].attrs['units'] = 'm'
    dsb['mu'].attrs['standard_name'] = 'shallowness_parameter'
    dsb['mu'].attrs['long_name'] = mu_info
    dsb['mu'].attrs['units'] = 'dimensionless'
    dsb['eps'].attrs['standard_name'] = 'nonlinearity_parameter'
    dsb['eps'].attrs['long_name'] = eps_info
    dsb['eps'].attrs['units'] = 'm'
    dsb['Ur'].attrs['standard_name'] = 'ursell_parameter'
    dsb['Ur'].attrs['long_name'] = Ur_info
    dsb['Sk'].attrs['standard_name'] = 'wave_skewness'
    dsb['Sk'].attrs['long_name'] = 'Skewness parameter following Elgar and Guza (1985)'
    dsb['As'].attrs['standard_name'] = 'wave_asymmetry'
    dsb['As'].attrs['long_name'] = 'Asymmetry parameter following Elgar and Guza (1985)'
    dsb['DoF'].attrs['standard_name'] = 'degrees_of_freedom'
    dsb['DoF'].attrs['long_name'] = 'Degrees of freedom of bispectral estimate'
    dsb['b95'].attrs['standard_name'] = 'significance_level'
    dsb['b95'].attrs['long_name'] = '95-perc significance level on zero bicoherence following Haubrich (1965)'

    # Some global attributes
    dsb.attrs['nfft'] = '{} samples'.format(nfft)
    dsb.attrs['overlap'] = '{}%'.format(overlap)
    dsb.attrs['nblocks'] = nblock
    dsb.attrs['fft_window'] = wind
    dsb.attrs['merged_frequencies'] = '{} frequencies'.format(mg)
    dsb.attrs['frequency_resolution'] = '{} Hz'.format(df)

    return dsb


def bispectrum(z, wsec=256, fs=5.0, fmerge=3, h0=0, return_krms=True):
    """
    Bispectral function based on spec_uvz().
    
    Parameters:
        z - ndarray; 1D up displacements [m]
        wsec - window length in seconds
        fs - sampling frequency in Hz
        fmerge - int; freq bands to merge, must be odd
        h0 - scalar; water depth [m]. Required if return_krms=True
        return_krms - bool; if True, also returns RMS wavenumbers following
                      Herbers et al. (2002)
    Returns:
        ds - xarray.Dataset with bispectrum, bicoherence and biphase
    """
    # Copy input array so we don't change it
    eta = z.copy()
    npts = len(eta) # Number of data points

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
        # Detrend full eta array
        eta = detrend(eta)
    else:
        print('Input data not long enough or fs too low.')
        return np.zeros(n_freqs)
    # Define dictionary to store U,V,Z fft windows, if applicable
    data_vars={'B': (['freq1', 'freq2'], np.zeros((n_freqs, n_freqs))),
               'Bc': (['freq1', 'freq2'], np.zeros((n_freqs, n_freqs))),
               'Bp': (['freq1', 'freq2'], np.zeros((n_freqs, n_freqs))),
               'PSD': (['freq1'], np.zeros(n_freqs)),
               }
    ds = xr.Dataset(data_vars=data_vars, 
                    coords={'freq1': (['freq1'], f),
                            'freq2': (['freq2'], f)},
                   )

    # Initialize window array
    arr_win = np.zeros((n_windows, win_len)) 
    arr_win = np.atleast_2d(arr_win) # Needs to also work with only z
    # Loop over windows
    for q in range(1, int(n_windows)+1):
        si = int((q-1) * 0.25 * win_len) # Start index of window
        ei = int((q-1) * 0.25*win_len + win_len) # End index of window
        arr_win[q-1, :] = eta[si:ei].copy()
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

    # Compute power- and bispectrum, loop over windows
    ps_win = np.zeros((n_windows, win_len//2))
    B_win = np.zeros((win_len//2, win_len//2)).astype(complex)
    Bd1_win = np.zeros((win_len//2, win_len//2)).astype(float) # bicoh denominator 1
    Bd2_win = np.zeros((win_len//2, win_len//2)).astype(float) # bicoh denominator 2
    # Deal with f1 + f2 = f3 indices
    ifr1, ifr2 = np.meshgrid(np.arange(win_len//2), np.arange(win_len//2))
    ifr3 = ifr1 + ifr2
    # Mask for indices not corresponding to f1+f2
    ifm3val = np.logical_or((ifr3 < 0), (ifr3 >= win_len//2))
    ifr3[ifm3val] = 0
    # Iterate over windows and accumulate triple products
    for w in range(n_windows):
        win = fft_win[w, :]
        winc = np.conj(fft_win[w, :])
        # Compute power spectrum
        ps_win += np.real(win * winc)
        # Compute bispectrum
        B_win += win[ifr1] * win[ifr2] * winc[ifr3]
        # Denominator terms of bicoherence
        Bd1_win += np.abs(win[ifr1] * win[ifr2])**2
        Bd2_win += np.abs(win[ifr3])**2

    # Ensemble average windows (take expected value)
    ps_win /= n_windows
    B_win[ifm3val] = 0
    B_win /= n_windows
    Bd1_win[ifm3val] = 0
    Bd1_win /= n_windows
    Bd2_win[ifm3val] = 0
    Bd2_win /= n_windows

    # ------------------- Merging over frequencies -------------------
    ps_win_merged = np.zeros((n_freqs, n_windows))
    # Initialization
    mg = int(fmerge - np.remainder(fmerge + 1, 2))
    mm = int((mg - 1) / 2) # Half-window for averaging
    # ifrm = np.concatenate([np.arange(0, 1+mm-mg, -mg)[::-1], 
    #                        np.arange(mg, win_len+1-mm, mg)]) # Frequency indices
    ifrm = np.arange(mm, win_len//2+1-mm, mg)#[::-1]

    Bmw = np.zeros((n_freqs, n_freqs, n_windows)).astype(complex)
    Bd1w = np.zeros((n_freqs, n_freqs, n_windows)).astype(float)
    Bd2w = np.zeros((n_freqs, n_freqs, n_windows)).astype(float)
    # Loop over frequencies
    for i,jfr1 in enumerate(range(n_freqs)):
        # Rows
        ifb1 = ifrm[jfr1] # mid of jfr1-merge-block
        # PSD
        ps_win_merged[jfr1,:] = np.mean(ps_win[:, ifb1-mm:ifb1+mm+1])
        # Columns for bispectrum and bicoherence
        for jfr2 in range(n_freqs):
            ifb2 = ifrm[jfr2] # mid of jfr2-merge-block
            Bmw[jfr1,jfr2,:] = np.mean(np.mean(B_win[np.arange(ifb1-mm, ifb1+mm+1), 
                                                     np.arange(ifb2-mm, ifb2+mm+1)]))
            Bd1w[jfr1,jfr2,:] = np.mean(np.mean(Bd1_win[np.arange(ifb1-mm, ifb1+mm+1), 
                                                        np.arange(ifb2-mm, ifb2+mm+1)]))
            Bd2w[jfr1,jfr2,:] = np.mean(np.mean(Bd2_win[np.arange(ifb1-mm, ifb1+mm+1), 
                                                        np.arange(ifb2-mm, ifb2+mm+1)]))

    # Ensemble average windows together:
    # Take the average of all windows at each freq band
    # and divide by N*(sample rate) to get power spectral density
    # The two is b/c we threw the redundant half of the FFT away,
    # so need to multiply the PSD by 2.
    psd = np.mean(ps_win_merged, axis=1) / (win_len/2 * fs)
    Bm = np.mean(Bmw, axis=2) / (win_len/2 * fs**2)
    Bd1m = np.mean(Bd1w, axis=2) / (win_len/2 * fs**2)
    Bd2m = np.mean(Bd2w, axis=2) / (win_len/2 * fs**2)
    # Bicoherence
    np.seterr(all="ignore") # Ignore RuntimeWarnings
    Bc = np.abs(Bm) / np.sqrt(Bd1m*Bd2m)
    # Bi-phase
    
    # Save arrays to output dataset
    ds['PSD'].values = psd
    ds['B'].values = Bm
    ds['Bc'].values = Bc
    # Save DoF and 80%, 90%, 95% and 99% thresholds for significance (Elgar & Guza, 1985)
    b99 = np.sqrt(9.2 / (dof-1))
    b95 = np.sqrt(6 / (dof-1))
    b90 = np.sqrt(4.6 / (dof-1))
    b80 = np.sqrt(3.2 / (dof-1))
    ds['dof'] = ([], dof)
    ds['b99'] = ([], b99)
    ds['b95'] = ([], b95)
    ds['b90'] = ([], b90)
    ds['b80'] = ([], b80)
    # Depth
    ds['h0'] = ([], h0)

    if return_krms:
        # Also compute rms wavenumbers K_rms following Herbers et al. (2002)
        krms = k_rms(h0=h0, f=f, P=psd, B=Bm, two_sided=False)
        ds['k_rms'] = (['freq'], krms)
        ds['k_rms'].attrs['standard_name'] = 'sea_surface_wave_rms_wavenumber'
        ds['k_rms'].attrs['long_name'] = 'Root-mean-square wavenumbers following H02'
        ds['k_rms'].attrs['units'] = '1/m'

    return ds


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Test script, generate synthetic signal
    fs = 4
    a = 2 * np.sqrt(2)
    fp = 0.1
    t = np.arange(3*4800) / fs
    noise_power = 1e-3 * fs / 2
    eta = a * np.cos( 2*np.pi*fp * t) + a/4 * np.cos( 2 * 2*np.pi*fp * t) 
    rng = np.random.default_rng()
    noise = rng.normal(scale=np.sqrt(noise_power), size=t.shape)
    eta += noise

    # Test bispectrum
    dsb = bispectrum(z=eta, fs=fs)

    fig, axes = plt.subplots(figsize=(7,4), ncols=2)
    dsb.PSD.plot(ax=axes[0], )
    axes[0].axvline(x=fp, color='k', linestyle='--', alpha=0.5)
    axes[0].axvline(x=2*fp, color='k', linestyle='--', alpha=0.5)
    bs = dsb.Bc**2
    # (bs.where(bs > dsb.b95)**2).plot.pcolormesh(ax=axes[1])
    bs.plot.pcolormesh(ax=axes[1])
    axes[1].set_xlim([0,0.5])
    axes[1].set_ylim([0,0.5])

    plt.tight_layout()
    plt.show()
