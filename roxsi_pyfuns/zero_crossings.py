"""
Functions for wave-by-wave analysis by zero crossings.
"""
import numpy as np
import pandas as pd

def crossings_nonzero_pos2neg(data):
    """
    Returns indices for downwdard zero crossings in a time
    series, i.e. the zero crossings from positive to negative.

    Note that each index is for the 'last' value of the previous
    'wave', that is the last value of opposite sign before the
    zero crossing. Therefore must add one to the indices returned if
    don't want to include these values.
    
    Parameters:
        data: 1D array of sea-surface elevations.

    Borrowed from 
    https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    """
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0]


def crossings_nonzero_neg2pos(data):
    """
    Upward-zero-crossings function.
    """
    pos = data > 0
    npos = ~pos
    return ((npos[:-1] & pos[1:])).nonzero()[0]


# Function for zero-crossing wave-, crest-, and through heights 
def get_waveheights(ts, method='down', zero_crossings=None, func=None, minlen=0):
    """
    Computes individual wave heights from signal ts.

    Returns the downward zero-crossing indices as well as an array 
    of individual wave heights (Hw) in the same units as the input 
    array. Also returns an array with crest heights (Hc) and trough 
    depths (Ht).

    If input mean_signal is given, compute mean values of this during 
    each wave. Should be the same size as ts.
    
    Parameters:
        ts - time series; np or xr array of measurements
        method - str; 'up' for upward-zero crossings, 'down' for downward z.c.
        zero_crossings - if None, use predefined zero-crossings
        func - str; Choices: ['mean', 'std']. If not None, computes either
               mean or std of values of ts between zero-crossings.
        minlen - scalar; minimum sample distance between zero crossings to include

    Returns:
        zero_crossings - array of zero-crossing indices
        Hw - array of zero-crossing wave heights
        Hc - array of zero-crossing crest heights
        Ht - array of zero-crossing trough depths
    """
    # Copy and make time series an np array just in case
    eta = np.array(ts).copy()

    # Get zero-crossings if not predefined
    if zero_crossings is None:
        # Find indices of downward zero crossings in eta
        if method == 'down':
            # Downward-zero crossings
            zero_crossings = crossings_nonzero_pos2neg(eta)
        elif method == 'up':
            # Downward-zero crossings
            zero_crossings = crossings_nonzero_neg2pos(eta)
        # Add one to each index to get the correct intervals
        if zero_crossings[-1] < (len(eta)-1):
            zero_crossings += int(1)
        else:
            zero_crossings[:-1] += int(1)

    # Loop over the individual waves using zero-crossing indices
    if func is not None:
        # Remove waves with too-short period
        zero_crossings_final = [] # List for final zero-crossings
        Hf = [] # List for mean or std between zero-crossings
        for i in range(len(zero_crossings)-1):
            start = zero_crossings[i]
            end = zero_crossings[i+1]
            # Check if period is long enough
            if (end - start) < minlen:
                continue
            else:
                zero_crossings_final.append(start)
            # individual wave
            wave = eta[start:end]
            # Mean/std value of eta during the wave
            if func == 'mean':
                h_f = np.nanmean(wave)
            elif func == 'std':
                h_f = np.nanstd(wave)
            # Append to list
            Hf.append(h_f)

        # Make lists into np.arrays
        Hf = np.array(Hf)

        return zero_crossings, Hf
    else:
        # Remove waves with too-short period
        zero_crossings_final = [] # List for final zero-crossings
        Hc = [] # List for crest heights
        Ht = [] # List for trough heights
        Hw = [] # List for wave heights
        for i in range(len(zero_crossings)-1):
            start = zero_crossings[i]
            end = zero_crossings[i+1]
            # Check if period is long enough
            if (end - start) < minlen:
                continue
            else:
                zero_crossings_final.append(start)
            # individual wave
            wave = eta[start:end]
            # crest of the wave
            h_crest = np.nanmax(wave)
            Hc.append(h_crest)
            # trough of the wave
            h_trough = np.nanmin(wave)
            Ht.append(h_trough)
            # wave height
            h_wave = h_crest - h_trough
            Hw.append(h_wave)

        # Make lists into np.arrays
        zero_crossings_final.append(end) # Append last zero crossing
        zero_crossings_final = np.array(zero_crossings_final)
        Hw = np.array(Hw)
        Hc = np.array(Hc)
        Ht = np.array(Ht)

        return zero_crossings_final, Hw, Hc, Ht


def exceedance_prob(x):
    """
    Returns exceedance probabilities y for input array x. Also 
    returns theoretical stability bound (standard deviation) around
    mean exceedance probability following Tayfun and Fedele (2007).
    """
    # Make sure x is np.array
    x = np.array(x)
    # Compute exceedance probabilities
    n = len(x)
    y = 1/((n + 1)) * (np.arange(n, 0, -1))
    # Empirical stability bound (standard dev.) following Tayfun 
    # and Fedele (2007, Oc. Eng.) Eq. (16)
    j = y[::-1] # inverted y array
    std = 1 / (n+1) * np.sqrt((j * (n - j + 1)) / (n + 2))

    return y, std


def interpolate_phase(z, N, label=None):
    """
    Interpolate array z to 0-2*pi phase axis of length N.

    Parameters:
        z - data array to interpolate
        N - scalar; length of interpolation target phase axis
        label - str; if not None, labels variable of output df

    Returns pd.DataFrame with intepolated signal and target
    phase axis as index.
    """
    # Define target phase (x) axis to interpolate to
    x_phase = np.linspace(0, 2*np.pi, N)
    # Define true phase x axis of wave
    x_wave = np.linspace(0, 1, len(z)) * 2*np.pi
    # Interpolate input array to regular phase
    z_interp = np.interp(x_phase, x_wave, z)
    # Save interpolated signal to output dataframe
    if label is not None:
        dfi = pd.DataFrame(data={label: z_interp}, index=x_phase)
    else:
        dfi = pd.DataFrame(data=z_interp, index=x_phase)

    return dfi
