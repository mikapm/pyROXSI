"""
Functions for wave-by-wave analysis by zero crossings.
"""
import numpy as np

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
def get_waveheights(ts, method='down', zero_crossings=None):
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

    Returns:
        zero_crossings - array of zero-crossing indices
        Hw - array of zero-crossing wave heights
        Hc - array of zero-crossing crest heights
        Ht - array of zero-crossing trough depths
    """
    # Make time series an np array just in case
    eta = np.array(ts)

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
    Hc = [] # List for crest heights
    Ht = [] # List for trough heights
    Hw = [] # List for wave heights
    for i in range(len(zero_crossings)-1):
        start = zero_crossings[i]
        end = zero_crossings[i+1]
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
    Hw = np.array(Hw)
    Hc = np.array(Hc)
    Ht = np.array(Ht)

    return zero_crossings, Hw, Hc, Ht
