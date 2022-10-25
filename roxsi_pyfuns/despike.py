#!/usr/bin/env python3
from __future__ import division
import os
import re
import numpy as np
from scipy.special import erfinv
from statsmodels import robust
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
import linecache
from itertools import islice
from functools import reduce
from datetime import datetime as DT
from collections import OrderedDict
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from itertools import chain
from copy import deepcopy
import sw_pyfuns as swp
from sw_pyfuns import data_extract as swd
from sw_pyfuns import tools as swt
from sw_pyfuns import interpolate as swi
from sw_pyfuns import filters as swf
from sw_pyfuns import plotting
from sw_pyfuns import nigp as swn
from sw_pyfuns.nigp import nigp

"""
Functions to despike laser altimeter data.
"""

def otsu_1979_threshold(diff_signal):
    """
    Algorithm developed by Otsu (1979): A Threshold
    Selection Method from Gray-Level Histograms, for
    optimal threshold selection.

    This code is based on the adaptation of Otsu's 
    method by Feuerstein et al. (2009): Practical
    Methods for Noise Removal ... in a Matlab code
    distributed as supporting information to that
    paper here:
    https://pubs.acs.org/doi/suppl/10.1021/ac900161x/
    suppl_file/ac900161x_si_001.pdf

    Otsu's method is based on a histogram of a 
    difference signal, and computes the threshold
    that best separates two data classes: spikes
    and the remaining signal. The optimal threshold is 
    that which minimises the within-class variance or,
    equivalently, maximises the between-class variance
    (Feuerstein et al., 2009).

    ------------------------------------------------------
    ------------------------------------------------------

    From Otsu (1979):
    The fundamental principle is to divide a signal
    (or, originally, image pixels) into two classes:
    background (C0) and objects (C1), or e.g. spikes and
    remaining signal. The classes are separated by a threshold
    level k such that C0 includes elements with pixel 
    levels (or signal amplitudes) [1, ..., k] and C1 has
    the elements [k+1, ..., L], where L is the total
    number of pixels. This gives the probabilities of
    class occurrence

    pr_C0 = \sum_{i=1}^{k} p_i
    pr_C1 = \sum_{i=k+1}^{L} p_i = 1 - pr_C0

    where p_i = n_i/N is the normalised histogram, or
    pdf, of the pixel levels.

    Following Feuerstein et al. (2009), the initial
    threshold is set to the level at k=2.

    After computing the class probabilities, the class
    means are computed as

    m_C0 = \sum_{i=1}^{k} i*p_i / pr_C0
    m_C1 = \sum_{i=k+1}^{L} i*p_i / pr_C1
    
    Now, the between-class variance can be computed as

    var_C0C1 = pr_C0*pr_C1 * (m_C0 - m_C1)**2

    The above three steps are repeated for every threshold
    index k. The optimal threshold opt_thresh is the one that
    maximises var_C0C1.

    ------------------------------------------------------
    ------------------------------------------------------

    Parameters:
        diff_signal: difference b/w raw and filtered
                     signal
        
    Returns:
        opt_thresh - optimal threshold for diff_signal
    """

    L = len(diff_signal)

    # Make a histogram of the difference signal
    hist, bin_edges = np.histogram(diff_signal)

    # Normalise hist by length of difference signal
    hist = hist/L

    # Use bin centres instead of bin edges (as in Matlab)
    bin_cents = bin_edges[:-1] + np.diff(bin_edges)/2
    N = len(bin_cents)

    # Initialise arrays
    thresholds = np.zeros(N)
    pr_C0 = np.zeros(N)
    pr_C1 = np.zeros(N)
    m_C0 = np.zeros(N)
    m_C1 = np.zeros(N)
    var_C0C1 = np.zeros(N)

    # Initial threshold: k=2
    thresholds[0] = bin_cents[1]
    # Initial probabilities
    pr_C0[0] = np.sum(hist[0])
    pr_C1[0] = 1 - pr_C0[0]
    # Initial means
    m_C0[0] = np.sum(bin_cents[0] * hist[0] / pr_C0[0])
    m_C1[0] = np.sum(bin_cents[1:] * hist[1:] / pr_C1[0])

    # Compute remaining probabilities of class occurrence
    # using remaining thresholds (bin centres), and test
    # the effect of different thresholds on between-class
    # variance.
    for i in range(1,N-1):
        thresholds[i] = bin_cents[i+1]
        pr_C0[i] = np.sum(hist[:i+1])
        pr_C1[i] = 1 - pr_C0[i]
        m_C0[i] = np.sum(bin_cents[:i+1]*hist[:i+1]/pr_C0[i])
        m_C1[i] = np.sum(bin_cents[i+1:]*hist[i+1:]/pr_C1[i])
        var_C0C1[i] = pr_C0[i]*pr_C1[i]*(m_C0[i]-m_C1[i])**2

    # Find the maximum between-class variance and its index k
    var_C0C1_max = np.max(var_C0C1)
    k = np.where(var_C0C1 == var_C0C1_max)

    # The optimal threshold maximises the between-class variance
    # If multiple equal max. variances, use the first one (as by
    # default in Matlab(?)).
    opt_thresh = thresholds[k][0]

    return opt_thresh


def feuerstein(ts, filt='sg', sg_winlen=13, sg_order=4,
        replace_single='linear', replace_multi='sg_filter',
        min_thresh=None, figname=None, return_thresh=False):
    """
    Simple spike detection algorithm described in Feuerstein
    et al (2009): Practical Methods for Noise Removal ...
    
    Spikes are located by subtracting a filtered signal
    from the original signal and using a threshold determined
    by Otsu's (1979) method to separate the spikes from the
    signal. 

    Parameters:
        ts - the time series array to despike
        filt - str; filtering method to use. Either
               'sg' for Savitzky-Golay (SG) smoothing (preferred), or
               'diff' for differentiation of the signal
        sg_winlen - Window length for SG filter; must be an odd
                    integer
        sg_order - The order of the polynomial used in SG filtering
        replace_single - str; interpolation method for single-point 
                         spikes in swp.interpolate_replace_1d()
        replace_multi - str; interpolation method for multi-point 
                        spikes in swp.interpolate_replace_1d()
        min_thresh - Minimum allowed spike threshold, to avoid high-
                     quality data (signals close to the median) getting
                     an excessively low threshold, which can lead to
                     false spike detections.
        figname - str; save figure
        return_thresh - return the computed threshold value 

    Returns:
        u - despiked time series
        spike_mask - mask for detected spikes
        (if return_thresh is True):
        opt_thresh - threshold computed by Otsu's method
    """
    # Copy the input time series
    u = ts.copy() 
    u = np.array(u)

    # Filter the raw data and subtract
    if filt=='sg':
        # Use Savitzky-Golay smoothing
        u_filt = swf.savitzky_golay(u, window_size=sg_winlen,
                order=sg_order)
        u_diff = np.abs(u - u_filt)
    elif filt=='diff':
        # Calculate abs of the first derivative of the signal
        u_diff = np.abs(np.gradient(u))

    # Use Otsu's (1979) method to determine the optimal
    # threshold for spike detection
    opt_thresh = otsu_1979_threshold(u_diff)
    if not min_thresh:
        # Min. thresh = 2 * robust std (1.438*MAD, see Wahl, 2003)
        min_thresh = 2*1.483*robust.mad(ts, c=1)
    if opt_thresh < min_thresh:
        # Don't allow thresholds lower than min_thresh
        # TODO: define optimal min_thresh
        opt_thresh = min_thresh

    # Find indices where the difference is larger than opt_thresh
    spike_mask = np.ones(len(ts), dtype=bool)
    bad_ind = np.where(u_diff > opt_thresh)
    spike_mask[bad_ind] = 0

    # Replace spikes by chosen interpolation methods
    if (spike_mask!=1).any():
        # Check if there are consequtive bad indices
        #bad_ind_consec = swt.consecutive(bad_ind)
        # Replace spikes
        u = swi.replace_1d(u, spike_mask, 
                replace_single=replace_single,
                replace_multi=replace_multi)

    if figname is not None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        # Plot the difference signal
        ax.plot(u_diff, label='diff signal')
        # Plot the threshold
        ax.plot(np.ones(len(u_diff))*opt_thresh, linestyle='--', label='threshold')
        ax.legend()
        plt.show()
        plt.close()



    if return_thresh is True:
        return u, spike_mask, opt_thresh
    else:
        return u, spike_mask


def rc_filter(ts, k=3):
    """
    Elementary despiking algorithm described in
    Goring and Nikora (2002): Despiking Acoustic Doppler 
    Velocimeter Data. Based on a description in Otnes 
    and Enochson (1978): Applied time series analysis.

    Before applying the method, the mean and any long-
    period fluctuations should be removed from the time
    series (Goring and Nikora, 2002). In the paper the
    authors low-pass filter the data, but that is not
    done here as the laser data already is of relatively
    low frequency (2 or 5 Hz).

    Parameters:
        ts - the time series array to despike
        k - a parameter, usually set between 3 and 9
            (Goring and Nikora, 2002)

    Returns:
        u - despiked time series
        bad_ind - indices of detected spikes in ts
    """

    # Copy the input time series
    u = ts.copy() # ds - despiked
    u = np.array(u)
    # Calculate sample variance
    sigma2 = np.var(u)
    # Standard deviation
    sigma = np.sqrt(sigma2)
    # Store indices for spikes in list bad_ind
    bad_ind = []
    # The point i+1 is treated as a spike if it deviates
    # more than k*sigma from the previous point i.
    i = 0 # index counter
    for sample in u:
        while i < (len(u)-1): # To avoid exceeding array length
            if u[i+1] < (sample-k*sigma):
                bad_ind.append(i+1)
            elif u[i+1] > (sample+k*sigma):
                bad_ind.append(i+1)
            i += 1

    # Also check end points using mirrored boundary conditions
    if u[0] < (u[1]-k*sigma):
        # Append 0 to start of bad_ind list
        bad_ind.insert(0,0)    
    elif u[0] > (u[1]+k*sigma):
        bad_ind.insert(0,0)    
    if u[-1] < (u[-2]-k*sigma): 
        bad_ind.append(int(len(u)-1))
    elif u[-1] > (u[-2]+k*sigma):
        bad_ind.append(int(len(u)-1))

    # Replace spikes by linear interpolation
    # TODO include other interpolation methods
    if len(bad_ind):
        # Check if there are consecutive bad indices
        bad_ind_consec = swt.consecutive(bad_ind)
        # Replace spikes
        u = swi.replace_1d(u, bad_ind_consec)

    return u, bad_ind


def tukey_53h(ts, k=1.0, replace_single='linear',
        replace_multi='last_nonsp', otsu_k=False):
    """
    Algorithm also described in Goring and Nikora (2002) and
    based on Otnes and Enochson (1978). Uses the principle of
    the robustness of the median as an estimator of the mean.
    Name in G&N02: Tukey 53H.

    This function is modified to check for multiple-point
    constant spikes ("plateaus") if two detected spike indices
    are consecutive. This modification is due to the ubiquity 
    of such spikes in the LASAR array data, and the inability
    of the original Tukey 53H algorithm to detect such spikes
    (see e.g. the figure tukey_53_illustration.pdf in 
    $PHDDIR/latex/texts/lasar_preprocessing/despiking_examples).

    Parameters: 
        ts - the time series (1D) array to despike
        k - scaling parameter for sigma, usually set to around 1.5
            (https://cotede.readthedocs.io/en/latest/qctests.html)
        replace_single - str; interpolation method for single-point 
                         spikes in swp.interpolate_replace_1d()
        replace_multi - str; interpolation method for multi-point 
                        spikes in swp.interpolate_replace_1d()
        otsu_k - bool; use Otsu's method to find optimal threshold k
                 NB: Doesn't give good results for .wa4 LASAR data

    Returns:
        u - despiked time series       
        spike_mask - mask for detected spikes
    """
    # Copy the input time series
    u = ts.copy() 
    u = np.array(u)
    # Calculate sample variance
    sigma2 = np.var(u)
    # Standard deviation
    sigma = np.sqrt(sigma2)

    # 1. Compute the median u1 of the five points from i-2 to i+2
    u1 = np.zeros_like(u)
    for i, sample in enumerate(u[2:-2]):
        u1[i+2] = np.median(u[i:i+5])
    # End points with mirroring boundary conditions
    u1[0] = np.median(np.concatenate((u[:3],u[1:3]), axis=None))
    u1[1] = np.median(np.concatenate((u[:4],u[1]), axis=None))
    u1[-1] = np.median(np.concatenate((u[-3:],u[-3:-1]), axis=None))
    u1[-2] = np.median(np.concatenate((u[-4:],u[-2]), axis=None))

    # 2. Compute u2: the 3-point median of i-1 to i+1 of u1
    u2 = np.zeros_like(u)
    for i, sample in enumerate(u1[1:-1]):
        u2[i+1] = np.median(u1[i:i+3])
    # End points with mirroring boundary conditions
    u2[0] = np.median([u1[0], u[1], u[1]])
    u2[-1] = np.median([u[-1], u[-2], u[-2]])

    # 3. Compute a Hanning smoothing filter u3
    u3 = np.zeros_like(u)
    for i, sample in enumerate(u2[1:-1]):
        u3[i+1] = 0.25*(u2[i] + 2*sample + u2[i+2])
    # End points with mirroring boundary conditions
    u3[0] = 0.25*(u2[1] + 2*u2[0] + u2[1])
    u3[-1] = 0.25*(u2[-2] + 2*u2[-1] + u2[-2])

    # If specified, use Otsu's method to find the optimal threshold
    # NB! Doesn't give good results for LASAR data!
    if otsu_k:
        k = otsu_1979_threshold(np.abs(u-u3))
        # Don't use std if using Otsu-threshold
        sigma = 1

    # 4. Find spikes by taking the abs. difference between the
    #    original and the filtered data and comparing
    #    that to k*sigma
    # Make 1D array for masking detected spikes
    spike_mask = np.ones(len(ts), dtype=bool)
    for i, sample in enumerate(u):
        if np.abs(sample-u3[i]) > k*sigma:
            spike_mask[i] = 0
            # *** The following is a modification to the original
            #     algorithm described in G&N02 ***
            # Check if the spike has multiple points
#            try:
#                # See if previous measurement was flagged
#                if spike_mask[i-1] == 0:
#                    # Differentiate original data 15 points
#                    # forward to check if spike is a plateau
#                    interval_15 = np.arange(i, i+15)
#                    # Use np.diff() as an approx. differentiation
#                    u_diff = np.diff(u[interval_15])
#                    # Loop through u_diff and check if there are
#                    # consecutive zero-derivatives (diffs)
#                    for n, d in (u_diff,1):
#                        if d==0:
#                            spike_mask[i+n] = 0
#                        else:
#                            break
#            except IndexError:
#                pass

    # Replace spikes by linear interpolation
    # TODO include other interpolation methods
    if (spike_mask != 1).any():
        # Replace spikes
        u = swi.replace_1d(u, spike_mask, 
                replace_single=replace_single,
                replace_multi=replace_multi)

    return u, spike_mask


def accel_thresh(ts, dt=0.2, k=9, l=1):
    """
    Acceleration Thresholding Method following Goring and 
    Nikora (2002).

    Parameters:
        ts - the time series to despike
        dt = sampling interval of ts in seconds
        k - magnitude threshold, usually k=1.5 (G&N02)
        l - lambda_a in G&N02, usually between 1-1.5;
            scaling for the gravity g

    Returns:
        The despiked time series u

    NB! Can't handle multi-point spikes properly!
    """
    # Gravity
    g = 9.81
    # Copy the input time series
    u = ts.copy() # ds - despiked
    u = np.array(u)
    # Calculate sample variance
    sigma2 = np.var(u)
    # Standard deviation
    sigma = np.sqrt(sigma2)
    # Calculate the acceleration (actually just the
    # derivative, since our time series are not velocities)
    bad_ind = [] # List to store bad indices
    test_ind = [0] # List to check if spikes remain
    acc = np.zeros_like(u)
    # Iterate until no bad indices are found
    iterations = 0
    # Set upper limit for number of iterations
    iter_lim = 5
    while len(test_ind):
        # acc_i = ( u_i - u_(i-1) ) / dt 
        for i, s in enumerate(u[1:]):
            acc[i] = (s - u[i])/dt
            if np.abs(acc[i]) > l*g:
                bad_ind.append(i)
            elif np.abs(s) > k*sigma:
                bad_ind.append(i)
        if len(bad_ind):
            # Check for multi-point spikes
            bad_ind_consec = swt.consecutive(bad_ind)
            # Replace bad points
            u = swi.replace_1d(u, bad_ind_consec)
        # Check if spikes still remain
        test_ind = [i for i, s in enumerate(acc) if np.abs(s)>l*g
                or np.abs(u[i])>k*sigma]
        iterations += 1
        if iterations>5:
            # Multi-point spikes are a problem, just stop after
            # a few iterations, else will get infinite loop.
            print('Stopping iterating after %s iterations'
                    % iter_lim)
            break
        # Clear bad_ind list for next iteration
        bad_ind=[]

    return u, bad_ind


def wavelet_thresh(ts, l):
    """
    Wavelet thresholding method following Goring and Nikora (2002).
    Based on noise-removal algorithm developed by Donoho and
    Johnstone (1994): Ideal spatial adaptation by wavelet shrinkage.

    Parameters: 
        ts - the time series to despike
        l - lambda_u, scaling for sigma threshold
    """
    # Copy the input time series
    u = ts.copy() # ds - despiked
    u = np.array(u)


def array_median(ts_array, ts_axis=0, replace_single='linear',
        replace_multi='linear', min_thresh = 2.0):
    """
    Despiking method designed specifically for LASAR array
    data. Inspired by Feuerstein et al. (2009) and their
    use of Otsu's (1979) method.
    
    Takes as input an array of time series (e.g. from
    the four lasers), and compares each individual signal
    to the median of all signals. A spike threhold is 
    calculated using Otsu's (1979) method.

    Parameters:
        ts_array - array of time series, e.g. the four LASAR
                   arrays
        ts_axis - the axis of the time series to be despiked,
               e.g. for a ts_array with shape (4,6000), 
               ts_axis=1 means that the time series to be despiked
               is ts_array[1,:]. 
               Note that the median of ts_array is always taken 
               for axis=0!
        replace_single - str; interpolation method for single-point 
                         spikes in swp.interpolate_replace_1d()
        replace_multi - str; interpolation method for multi-point 
                        spikes in swp.interpolate_replace_1d()
        min_thresh - Minimum allowed spike threshold, to avoid high-
                     quality data (signals close to the median) getting
                     an excessively low threshold, which can lead to
                     false spike detections.

    Returns:
        u - despiked (1D) time series
        spike_mask - mask for detected spikes
               
    """

    # Initialise despiked output time series
    u = ts_array[ts_axis, :]

    # Take the median of all time series along axis=0
    ts_median = np.median(ts_array, axis=0)

    # Take the absolute difference between u and ts_median
    diff_signal = np.abs(u - ts_median)
    # Use information from all lasers to calculate threshold
    diff_signal_ts = np.abs(ts_array - ts_median)

    # Find the optimal threshold for spike detection using
    # Otsu's (1979) method
    #opt_thresh = otsu_1979_threshold(diff_signal)
    opt_thresh = otsu_1979_threshold(diff_signal_ts)
    if opt_thresh < min_thresh:
        # Don't allow thresholds lower than min_thresh
        # TODO: define optimal min_thresh
        opt_thresh = min_thresh

    # Find indices where the difference is larger than opt_thresh
    bad_ind = np.where(diff_signal > opt_thresh)[0]
    # Initialise spike mask
    spike_mask = np.ones(len(u), dtype=bool)
    spike_mask[bad_ind] = 0

    # Replace spikes by linear interpolation
    # TODO include other interpolation methods
    if (spike_mask != 1).any():
        u = swi.replace_1d(u, spike_mask, 
                replace_single=replace_single,
                replace_multi=replace_multi)


    return u, spike_mask


def GP_fit(ts, dropout_mask=None, chunksize=200, overlap=None, return_aux=True,
        length_scale_bounds=(5.0, 30.0), predict_all=True, use_nigp=True,
        use_sklearn=False, despike=True, score_thresh=0.99, print_kernel=True):
    """
    Non-parametric despiking and spike replacement based on fitting
    a Gaussian process to the data, following Bohlinger et al. (2019):
    A probabilistic approach to wave model validation applied to 
    satellite observations. Chunks up data according to input arg
    chunksize to speed up processing, which can be slow for long datasets 
    (computational expense O(N^3)).

    Parameters:
        ts - float array; 1D time series to despike
        dropout_mask - bool array; mask for missing values (and obvious
                       spikes), where 0 means a bad value (dropout or spike)
        chunksize - int; no. of datapoints in which to chunk up the signal;
                    if the signal is longer than ~1000 points the 
                    function is quite slow without chunking.
        overlap - int; number of points to overlap the chunks, 
                  adds 2*overlap to the true chunksize
        return_aux - bool; if true, also return y_pred, sigma, y_samp
        length_scale_bounds - tuple; bounds for allowable length scales
                              for the GP kernel.
        predict_all - bool; if set to False, will remove the data points
                      specified by dropout_mask in the prediction stage.
                      This may help with identifying outliers in cases
                      of many dropouts.
        despike - bool; if False, only replace samples given in dropout_mask, 
                  i.e no spike detection is done.
        score_thresh - float; R^2 upper threshold for despiking/no despiking

    Returns (if return_aux=True):
        y_desp - despiked time series (only spikes and dropouts replaced)
        spike_mask - bool array; mask for detected spikes
        (if return_aux=True, then also):
        y_pred - predicted (smooth) signal produced by GP fit
        sigma - std for each data point given by the GP fit
        y_samp - 100 samples drawn from the pdf of each dropout 
                 (+ obvious spike) point. Only works for use_sklearn=True.
    """

    # Check if ts is all NaN
    ts = np.array(ts)
    if np.sum(np.isnan(ts)) == len(ts):
        print('Input signal all NaN, nothing to despike ... \n')
        # Return array of NaNs.
        return np.ones(len(ts)) * np.nan
    elif len(ts[ts==-999]) == len(ts):
        print('Input signal all NaN, nothing to despike ... \n')
        # Return array of NaNs.
        return np.ones(len(ts)) * np.nan

    n = chunksize

    if dropout_mask is not None:
        mask_do = dropout_mask
    else:
        # No dropout mask given -> don't mask anything
        mask_do = np.ones(len(ts), dtype=bool)

    # Initialize output arrays
    y_desp = deepcopy(ts) # Output time series with spikes & dropouts replaced
    spike_mask = np.ones(len(ts), dtype=bool)
    y_pred = np.zeros(len(ts)) # Full predicted mean signal
    sigma = np.zeros_like(y_pred) # Uncertainty (std) for each data point
    y_samp = np.ones(len(ts), dtype=(float,100)) * np.nan # Samples from posterior

    # ********************************
    # Chunk up the time series for more efficient computing
    # ********************************

    chunk_range = np.arange(n, len(ts)+n, step=n)
    # Initialize hyperparameters dict theta
    theta = {
            'l': np.zeros(len(chunk_range)),
            'sigma_f': np.zeros(len(chunk_range)),
            'sigma_y': np.zeros(len(chunk_range)),
            'sigma_x': np.zeros(len(chunk_range)),
            'score': np.zeros(len(chunk_range)),
            }
    for chunk_no, chunk in enumerate(np.arange(n, len(ts)+n, step=n)):
        if print_kernel:
            print('Chunk number: ', chunk_no)
        # Define chunk interval (overlap vs. no overlap)
        if overlap is not None:
            if (chunk-n-overlap) < 0:
                # First chunk with overlap only at the end
                interval = np.arange(chunk-n, chunk+overlap, dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n, chunk+overlap)
            elif (chunk+overlap) > len(ts):
                # Last chunk with overlap only at the start
                interval = np.arange(chunk-n-overlap, chunk, dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n-overlap, chunk)
            else:
                # All other chunks with overlap at start & end
                interval = np.arange(chunk-n-overlap, chunk+overlap, dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n-overlap, chunk+overlap)
        else:
            # No overlap
            interval = np.arange(chunk-n, chunk, dtype=int)
            # Full range of x's for prediction
            x_pred = np.arange(chunk-n, chunk)

        # Chunk up data and mask based on interval
        y_chunk = ts[interval] # take out test chunk
        mask_do_chunk = mask_do[interval] # same for the dropout mask
        # Cut out dropouts from y_chunk
        y_chunk_nodo = y_chunk[mask_do_chunk]

        # Make x (t) axis for GP fit to training data
        x_train = deepcopy(interval)
        # Cut out dropouts (how the GP fit works)
        x_train = x_train[mask_do_chunk] 

        # If requested, remove points specified in dropout_mask from x_pred
        # This means that the prediction skips the data points determined by
        # dropout_mask. In the output, those skipped data points will be
        # assigned NaN.
        if not predict_all:
            x_pred = x_pred[mask_do_chunk]
            y_chunk = y_chunk[mask_do_chunk]
            # Set skipped (i.e. not predicted) data points to NaN in output
            skipped = interval[~mask_do_chunk]
            y_pred[skipped] = np.nan
            y_desp[skipped] = np.nan
            # Also remove masked points from interval, as it is used later
            interval = interval[mask_do_chunk]

        # ********************************
        # Train the data and fit GP model
        # ********************************

        yn_mean = np.nanmean(y_chunk_nodo)
        yn_median = np.nanmedian(y_chunk_nodo)
        # Copy zn for GP fitting
        y_train = deepcopy(y_chunk_nodo)
        # Remove mean
        y_train = y_train - yn_mean

        if use_sklearn is False:
            if not use_nigp:
                sigma_x_init = 0
            else:
                sigma_x_init = 1e-5
            # Numpy implementation of NIGP/GP
            gp = nigp.NIGPRegressor(
                    use_nigp = use_nigp,
                    l_bounds = length_scale_bounds,
                    sigma_x_init = sigma_x_init,
                    sf_bounds = (1e-5, 50.0),
                    sy_bounds = (1e-5, 50.0),
                    sx_bounds = (1e-5, 50.0),
                    )
            # Train GP model on data and optimise hyperparameters 
            # (including input noise if use_nigp=True)
            #mu_s, cov_s, theta_s = gp1.train(X, X_train1, Y_train1)
            #mu_s_chunk, cov_s_chunk, theta_s = gp1.train(
            y_pred_n, sigma_n, theta_s, score = gp.train(x_pred.reshape(-1,1), 
                    x_train.reshape(-1,1), y_train.reshape(-1,1),
                    return_score=True)
            # Add score to theta_s
            theta_s['score'] = score
        else:
            # kernel = parameterization of covariance matrix + Gaussian noise,
            # eq. 5 in Bohlinger et al.
            #kernel = (RBF(length_scale=1, length_scale_bounds=length_scale_bounds) +
            #kernel = (1*RBF(length_scale=1) +
            kernel = (ConstantKernel(1.0) * \
                    RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds) + \
                    WhiteKernel(noise_level=1.0)#, noise_level_bounds=(0.0, 10.0))
                    )
            # Define GP
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
            # Train GP process on data
            gp.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))
            gp.kernel_
            if print_kernel:
                print('GP kernel (sklearn): ', gp.kernel_)
            # Compute coefficient of determination R^2
            score = gp.score(x_train.reshape(-1,1), y_train.reshape(-1,1))
            # Make prediction based on GP (fills in missing points).
            # Remember, x_pred includes the entire range of indices in the chunk,
            # unless predict_all is set to False.
            # TODO: This might not be optimal for spike detection, as found in
            # testing with the time series from 3 Jan 2019, 00:40. A better way
            # might be to run an initial spike detection run in which dropout and
            # obvious spike locations are removed from x_pred, as this may have a
            # greater success rate at detecting spikes. Afterward, another GP fit
            # could be run with the full x_pred vector, in order to fill in all the
            # gaps.
            y_pred_n, sigma_n = gp.predict(np.array(x_pred).reshape(-1,1), 
                    return_std=True)
            # Save trained hyperparams and R^2 score
            l = gp.kernel_.k1.get_params()['k2__length_scale']
            sigma_f = np.sqrt(gp.kernel_.k1.get_params()['k1__constant_value'])
            sigma_y = np.sqrt(gp.kernel_.k2.get_params()['noise_level'])
            theta_s = {'l':l, 'sigma_f':sigma_f, 'sigma_y':sigma_y,
                    'sigma_x':0.0, 'score':score}

        if print_kernel:
            print('score: ', score)
        # Add hyperparameters to theta dict
        for key in theta.keys():
            theta[key][chunk_no] = theta_s[key]
        # Add the mean back to the prediction
        y_pred_n += yn_mean
        y_pred_n = np.squeeze(y_pred_n)
        sigma_n = np.squeeze(sigma_n)

        # ********************************
        # Update output arrays
        # ********************************

        y_pred[interval] = y_pred_n
        sigma[interval] = sigma_n
        #if despike and score < 0.99:
        if despike and score < score_thresh:
            # Detect outliers and replace both newly found spikes and dropouts
            mask = np.logical_and(y_chunk>y_pred_n-2*sigma_n, y_chunk<y_pred_n+2*sigma_n)
            if print_kernel:
                print('Spikes found: {} \n'.format(np.sum(~mask)))
            mask = np.logical_or(~mask, ~mask_do_chunk)
            mask = ~mask
        else:
            # Only replace points defined in dropout_mask
            mask = mask_do_chunk.copy()
            #mask = ~mask
        spike_mask[interval] = mask
        # Replace spikes and dropouts by corresponding y_pred values
        y_chunk[~mask] = y_pred_n[~mask]
        y_desp[interval] = y_chunk


        # ********************************
        # Sample y values for dropouts from posterior
        # (if prediction was made for those points)
        # ********************************
        if predict_all and use_sklearn:
            x_dropouts = interval[~mask_do_chunk].reshape(-1,1)
            # Sample from posterior (only implemented in sklearn for now)
            if len(x_dropouts):
                # Make matrix of samples for each dropout location
                S = gp.sample_y(x_dropouts, 100) 
                S += yn_mean # Add the mean back
                # Make samples into tuples
                y_samp_do = [list(map(tuple,i)) for i in S]
                y_samp_do = list(chain(*y_samp_do))
                x_dropouts = x_dropouts.squeeze()
                if len(np.atleast_1d(x_dropouts))==1:
                    x_dropouts = int(x_dropouts)
                # Store samples into output array
                #y_samp[x_dropouts] = y_samp_do
                y_samp[x_dropouts] = S.squeeze()

    # Don't want the initial dropouts to be included in the returned spike mask
    # -> add those back
    spike_mask[dropout_mask==0] = 1

    if return_aux is True:
        return y_desp, spike_mask, y_pred, sigma, y_samp, theta, score
    else:
        # Only return the despiked signal
        return y_desp


def phase_space_3d(ts, replace_single='linear', replace_multi='linear',
        threshold='universal', figname=None):
    """
    3D phase space method for despiking originally developed by
    Goring and Nikora (2002), and later modified by Wahl (2003) and
    Mori (2005). 
    
    This function is based on the Matlab function func_despike_phasespace3d
    by Mori (2005).

    Parameters:
        ts - time series to despike
        threshold - either 'universal' or 'chauvenet'
    """
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    # Universal threshold lambda_u: the expected maximum of N independent
    # samples drawn from a standard normal distribution (Goring & Nikora, 2002)
    N = len(ts)
    if threshold == 'universal':
        lambda_u = np.sqrt(2*np.log(N))
    # Chauvenet's criterion is independent of the size of the sample (Wahl, 2003)
    elif threshold == 'chauvenet':
        p = 1 / (2*N) # rejection probability
        Z = np.sqrt(2) * erfinv(1-p)

    # Subtract the median (more robust than the mean, see Wahl (2003))
    ts_med = np.median(ts)
    u = deepcopy(ts)
    u = u - ts_med

    # Take the 1st and 2nd derivatives of the time series
    du1 = np.gradient(u)
    du2 = np.gradient(du1)

    # Estimate the rotation angle theta of the principal axis 
    # of u versus du2 using the cross correlation. 
    # (for u vs. du1 and du1 vs. du2, theta = 0 due to symmetry)
    theta = np.arctan2(np.dot(u, du2), np.sum(u**2, dtype=float))

    # Look for outliers in 3D phase space
    # Following the func_excludeoutlier_ellipsoid3d.m Matlab script by Mori
    if theta == 0:
        x = u
        y = du1
        z = du2
    else:
        # Rotation matrix about y-axis (du1 axis)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0], 
            [-np.sin(theta), 0 , np.cos(theta)]
            ])
        x = u * R[0,0] + du1 * R[0,1] + du2 * R[0,2]
        y = u * R[1,0] + du1 * R[1,1] + du2 * R[1,2]
        z = u * R[2,0] + du1 * R[2,1] + du2 * R[2,2]
    
    # G&N02: For each pair of variables, calculate the ellipse that has max. and min.
    # from the previous computation. Use the MAD (median of absolute deviation)
    # instead of std for more robust scale estimators (Wahl, 2003).
    # Semi axes of the ellipsoid
    a = lambda_u * 1.483*robust.mad(x, c=1)
    b = lambda_u * 1.483*robust.mad(y, c=1)
    c = lambda_u * 1.483*robust.mad(z, c=1)

    # Mask for detected spikes (1=spike)
    spike_mask = np.zeros(N, dtype=bool)

    # Check for outliers (points not within the ellipsoid)
    for i in range(N):
        # Data point u, du1, du2 coordinates
        x1 = x[i]
        y1 = y[i]
        z1 = z[i]
        # Point on the ellipsoid given by a, b, c
        x2 = (a*b*c) * x1 / np.sqrt((a*c*y1)**2 + b**2*(c**2*x1**2+a**2*z1**2))
        y2 = (a*b*c) * y1 / np.sqrt((a*c*y1)**2 + b**2*(c**2*x1**2+a**2*z1**2))
        zt = c**2 * (1-(x2/a)**2 - (y2/b)**2)
        if z1 < 0:
            z2 = -np.sqrt(zt)
        elif z1 > 0:
            z2 = np.sqrt(zt)
        else:
            z2 = 0
        # Check for outliers from the ellipsoid by subtracting the ellipsoid
        # corresponding to the data (x1, y1, z1) from the ellipsoid given by
        # a,b and c. If the difference is less than 0 the point lies outside
        # the ellipsoid.
        dis = (x2**2 + y2**2 + z2**2) - (x1**2 + y1**2 + z1**2)
        if dis < 0:
            spike_mask[i] = 1

#     # Replace spikes if any were detected
#     if spike_mask.sum() != N:
#         # Replace spikes by interpolation
#         u = swi.replace_1d(u, spike_mask, 
#                 replace_single=replace_single,
#                 replace_multi=replace_multi)

    if figname is not None:
        fig = plt.figure(figsize=(12,12))
        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        pu = np.ma.masked_array(u, mask=spike_mask)
        pdu1 = np.ma.masked_array(du1, mask=spike_mask)
        pdu2 = np.ma.masked_array(du2, mask=spike_mask)
        ax.scatter(pu, pdu1, pdu2, marker='*', color='r', alpha=0.8, s=40)
        ax.scatter(u[spike_mask], du1[spike_mask], du2[spike_mask], marker='.',
                color='k', alpha=0.2)
        #ax.scatter(x, y, z, 'r*')
        swp.plotting.plotEllipsoid(center=[0,0,0], radii=[a,b,c], rotation=R,
                ax=ax, plotAxes=True)
        #plt.savefig(figname)
        plt.show()
        plt.close()


    # Add back median
    u = u + ts_med

#     return u, spike_mask
    return spike_mask


def phase_space_2d(ts, replace_single='linear', replace_multi='linear',
        threshold='universal', scale_mad=1.483, figname=None):
    """
    """

    def inside_ellipse(x, y, a, b, origin=(0,0), theta=0):
        """
        Check if point given by coordinates x,y lies inside
        the ellipse centred at origin with semi axis in x-dir.
        a and semi axis in y-dir. b, and (possibly) rotated by
        angle (in radians) theta.

        Returns True if point is inside, False otherwise.

        Borrowed from: 
        https://www.geeksforgeeks.org/check-if-a-point-is-inside-
        outside-or-on-the-ellipse/

        Equation for rotated ellipse from:
        https://math.stackexchange.com/questions/426150/what-is-
        the-general-equation-of-the-ellipse-that-is-not-in-the-
        origin-and-rotate
        """

        h = origin[0]
        k = origin[1]
        p = ( ((x-h)*np.cos(theta) + (y-k)*np.sin(theta))**2 / a**2 +
                ((x-h)*np.sin(theta) - (y-k)*np.cos(theta))**2 / b**2 )

        if p < 1:
            return True
        else:
            return False


    # Universal threshold lambda_u: the expected maximum of N independent
    # samples drawn from a standard normal distribution (Goring & Nikora, 2002)
    N = len(ts)
    if threshold == 'universal':
        lambda_u = np.sqrt(2*np.log(N))

    # Subtract the median (more robust than the mean, see Wahl (2003))
    u = ts.copy()
    ts_med = np.median(ts)
    ts = ts- ts_med # for plotting
    u = u - ts_med

    # Take the 1st and 2nd derivatives of the time series
    du1 = np.gradient(u)
    du2 = np.gradient(du1)

    # Estimate the rotation angle theta of the principal axis 
    # of u versus du2 using the cross correlation. 
    # (for u vs. du1 and du1 vs. du2, theta = 0 due to symmetry)
    theta = np.arctan2(np.dot(u, du2), np.sum(u**2, dtype=float))
    # Rotation matrix
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
        ])

    # Compute robust estimates of scale for u, du1 & du2 using MAD
    # NB: to avoid automatic scaling by 1/1.483, set c=1
    mad_u = robust.mad(u, c=1)
    mad_du1 = robust.mad(du1, c=1)
    mad_du2 = robust.mad(du2, c=1)
    # Scale estimates (instead of std)
    Su = scale_mad * mad_u
    Sdu1 = scale_mad * mad_du1
    Sdu2 = scale_mad * mad_du2

    # Compute the semi axes and origins of the ellipses
    # TODO define correct origins
    a_udu1 = lambda_u * Su
    b_udu1 = lambda_u * Sdu1
    #o_udu1 = (mad_u, mad_du1)
    o_udu1 = (0,0)
    a_du1du2 = lambda_u * Sdu1
    b_du1du2 = lambda_u * Sdu2
    #o_du1du2 = (mad_du1, mad_du2)
    o_du1du2 = (0,0)
    # a and b for u vs. du2 are given by the solutions to eqn system (10)&(11) in G&N02
    a_udu2 = lambda_u * Su
    #a_udu2 = np.sqrt((a_udu1**2*np.cos(theta)**2-b_du1du2**2*np.sin(theta)**2)/np.cos(2*theta))
    b_udu2 = lambda_u * Sdu2
    #b_udu2 = np.sqrt((b_du1du2**2*np.cos(theta)**2-a_udu1**2*np.sin(theta)**2)/np.cos(2*theta))
    #o_udu2 = (mad_u, mad_du2)
    o_udu2 = (0,0)

    # Check if points lie inside the ellipses. If outside -> spike
    spike_mask = np.ones(N, dtype=bool)
    # For plotting, make three different masks
    sm0 = np.ones(N, dtype=bool)
    sm1 = np.ones(N, dtype=bool)
    sm2 = np.ones(N, dtype=bool)
    for i in range(N):
        # Check u-du1 phase space
        if not inside_ellipse(x=u[i], y=du1[i], a=a_udu1, b=b_udu1, origin=o_udu1):
            spike_mask[i] = 0
            sm0[i] = 0
        # Check du1-du2 phase space
        if not inside_ellipse(x=du1[i], y=du2[i], a=a_du1du2, b=b_du1du2, origin=o_du1du2):
            spike_mask[i] = 0
            sm1[i] = 0
        # Check (rotated due to correlation) u-du2 phase space
        if not inside_ellipse(x=u[i], y=du2[i], a=a_udu2, b=b_udu2, origin=o_udu2, theta=theta):
            spike_mask[i] = 0
            sm2[i] = 0

    if spike_mask.sum() != N:
        # Replace spikes by interpolation
        u = swi.replace_1d(u, spike_mask, 
                replace_single=replace_single,
                replace_multi=replace_multi)

    # Plot if requested
    if figname is not None:
        fig, ax = plt.subplots(ncols=3, figsize=(30,10))
        # all pts
        ax[0].scatter(ts[sm0], du1[sm0], marker='.', color='k', alpha=0.2)
        # pts outside the ellipse in red
        ax[0].scatter(np.ma.masked_array(ts, mask=sm0),
                np.ma.masked_array(du1, mask=sm0),
                marker='*', s=40, color='r', alpha=0.8)
        # ellipse axes
        ax[0].plot(np.linspace(-a_udu1, a_udu1, 100), np.zeros(100), color='k', alpha=0.6,
                label=('a = %0.2f'%a_udu1))
        ax[0].plot(np.zeros(100), np.linspace(-b_udu1, b_udu1, 100), color='k', alpha=0.6,
                label=('b = %0.2f'%b_udu1))
        # ellipse
        ell = Ellipse(xy=o_udu1, width=2*a_udu1, height=2*b_udu1, edgecolor='k')
        ax[0].add_artist(ell)
        ell.set_clip_box(ax[0].bbox)
        ell.set_alpha(0.2)
        # text box with semi axis values (legend without a line)
        ax[0].legend(handletextpad=0, handlelength=0, fontsize=14)
        ax[0].set_xlabel(r'$\eta$ [m]', fontsize=16)
        ax[0].set_ylabel(r'$\Delta \eta$ [m]', fontsize=16)
        ax[0].set_xlim(-1.25*a_udu1, 1.25*a_udu1)
        ax[0].set_ylim(-1.25*b_udu1, 1.25*b_udu1)

        # all pts
        ax[1].scatter(du1[sm1], du2[sm1], marker='.', color='k', alpha=0.4)
        # pts outside the ellipse in red
        ax[1].scatter(np.ma.masked_array(du1, mask=sm1),
                np.ma.masked_array(du2, mask=sm1),
                marker='*', s=40, color='r', alpha=0.8)
        # ellipse axes
        ax[1].plot(np.linspace(-a_du1du2, a_du1du2, 100), np.zeros(100), color='k', alpha=0.4,
                label=('a = %0.2f'%a_du1du2))
        ax[1].plot(np.zeros(100), np.linspace(-b_du1du2, b_du1du2, 100), color='k', alpha=0.4,
                label=('b = %0.2f'%b_du1du2))
        # ellipse
        ell1 = Ellipse(xy=o_du1du2, width=2*a_du1du2, height=2*b_du1du2, edgecolor='k')
        ax[1].add_artist(ell1)
        ell1.set_clip_box(ax[1].bbox)
        ell1.set_alpha(0.2)
        ax[1].legend(handletextpad=0, handlelength=0, fontsize=14)
        ax[1].set_xlabel(r'$\Delta \eta$ [m]', fontsize=16)
        ax[1].set_ylabel(r'$\Delta^2 \eta$ [m]', fontsize=16)
        ax[1].set_xlim(-1.25*a_du1du2, 1.25*a_du1du2)
        ax[1].set_ylim(-1.25*b_du1du2, 1.25*b_du1du2)
        
        # all pts
        ax[2].scatter(ts[sm2], du2[sm2], marker='.', color='k', alpha=0.4)
        # pts outside the ellipse in red
        ax[2].scatter(np.ma.masked_array(ts, mask=sm2),
                np.ma.masked_array(du2, mask=sm2),
                marker='*', s=40, color='r', alpha=0.8)
        # parameterised ellipse
        t = np.linspace(0, 2*np.pi, 1000)
        xp = a_udu2 * np.cos(t) * np.cos(theta) - b_udu2 * np.sin(t) * np.sin(theta) # + x0
        yp = a_udu2 * np.cos(t) * np.sin(theta) + b_udu2 * np.sin(t) * np.cos(theta) # + y0
#        ax[2].plot(x,y)
        # patches ellipse (should be equal to parameterised ellipse)
        ell2 = Ellipse(xy=o_udu2, width=2*a_udu2, height=2*b_udu2,
                angle=np.degrees(theta), edgecolor='k')
        ax[2].add_artist(ell2)
        ell2.set_clip_box(ax[2].bbox)
        ell2.set_alpha(0.2)
        ax[2].legend(handletextpad=0, handlelength=0, fontsize=14)
        ax[2].set_xlabel(r'$\eta$ [m]', fontsize=16)
        ax[2].set_ylabel(r'$\Delta^2 \eta$ [m]', fontsize=16)
        ax[2].set_xlim(1.25*np.min(xp), 1.25*np.max(xp))
        ax[2].set_ylim(1.25*np.min(yp), 1.25*np.max(yp))

        plt.tight_layout()
        #plt.savefig(figname)
        plt.show()
        plt.close()

    # Add back median
    u = u + ts_med

    return u, spike_mask











    







