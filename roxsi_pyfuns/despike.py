#!/usr/bin/env python3
from __future__ import division
import numpy as np
from scipy.special import erfinv
from statsmodels import robust
import matplotlib.pyplot as plt
from itertools import chain
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from copy import deepcopy

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

@ignore_warnings(category=ConvergenceWarning)
def GP_despike(ts, dropout_mask=None, chunksize=200, overlap=None, 
        return_aux=True, length_scale_bounds=(5.0, 30.0), 
        predict_all=True, kernel=None, despike=True, 
        score_thresh=0.995, print_kernel=False,
        scores_in=None):
    """
    Non-parametric despiking and spike replacement based on 
    fitting a Gaussian process to the data, following 
    
    Malila et al. (2021, Jtech):
    "A nonparametric, data-driven approach to despiking ocean
    surface wave time series."  
    https://doi.org/10.1175/JTECH-D-21-0067.1
    
    Chunks up data according to input arg
    chunksize to speed up processing, which can be slow for long 
    datasets (computational expense O(N^3)).

    Parameters:
        ts - float array; 1D time series to despike
        dropout_mask - bool array; mask for missing values 
                       (and obvious spikes), where 0 means a 
                       bad value (dropout or spike)
        chunksize - int; no. of datapoints in which to chunk up the
                    signal; if the signal is longer than ~1000 
                    points the function is quite slow without chunking.
        overlap - int; number of points to overlap the chunks, 
                  adds 2*overlap to the true chunksize
        return_aux - bool; if True, also return y_pred, sigma, y_samp
        length_scale_bounds - tuple; bounds for allowable length 
                              scales for the GP kernel.
        kernel - If not None, input a specified covariance kernel of 
                 own choice
        predict_all - bool; if set to False, will remove the data 
                      points specified by dropout_mask in the 
                      prediction stage. This may help with identifying
                      outliers in cases of many dropouts.
        despike - bool; if False, only replace samples given in 
                  dropout_mask, i.e no spike detection is done.
        score_thresh - float; R^2 upper threshold for 
                       despiking/no despiking
        print_kernel - bool; set to False to not print kernel 
                       parameters
        scores_in - array_like; list of pre-calculated scores
                    to use for skipping chunks that have already been
                    found to have high R^2 score.

    Returns:
        y_desp - despiked time series (only spikes and dropouts 
                 replaced)
        (if return_aux=True, then also returns):
        spike_mask - bool array; mask for detected spikes
        y_pred - predicted (smooth) signal produced by GP fit
        sigma - std for each data point given by the GP fit
        y_samp - 100 samples drawn from the pdf of each dropout 
                 (+ obvious spike) point.
        theta - dict; GP fit hyperparameters for each chunk
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
    y_desp = ts.copy() # Output array w/ spikes & dropouts replaced
    spike_mask = np.ones(len(ts), dtype=bool)
    y_pred = np.zeros(len(ts)) # Full predicted mean signal
    # Uncertainty (std) for each data point
    sigma = np.zeros_like(y_pred) 
    # Samples from posterior
    y_samp = np.ones(len(ts), dtype=(float,100)) * np.nan 

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
                interval = np.arange(chunk-n, chunk+overlap, 
                dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n, chunk+overlap)
            elif (chunk+overlap) > len(ts):
                # Last chunk with overlap only at the start
                interval = np.arange(chunk-n-overlap, chunk, 
                dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n-overlap, chunk)
            else:
                # All other chunks with overlap at start & end
                interval = np.arange(chunk-n-overlap, chunk+overlap,
                                     dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n-overlap, chunk+overlap)
        else:
            # No overlap
            interval = np.arange(chunk-n, 
                                 min(chunk, len(ts)-1),
                                 dtype=int)
            # Full range of x's for prediction
            x_pred = np.arange(chunk-n, min(chunk, len(ts)-1))

        # Chunk up data and mask based on interval
        y_chunk = ts[interval].copy() # take out test chunk
        mask_do_chunk = mask_do[interval] # same for the dropout mask
        # Cut out dropouts from y_chunk
        y_chunk_nodo = y_chunk[mask_do_chunk]

        # Check if input scores list given
        if scores_in is not None:
            # Check if current chunk has high score and can be skipped
            si = scores_in[chunk_no]
            if si >= score_thresh:
                # Chunk score already high enough 
                # => don't despike again
                y_desp[interval] = y_chunk
                spike_mask[interval] = np.ones_like(y_chunk).astype(
                    bool)
                continue

        # Make x (t) axis for GP fit to training data
        x_train = interval.copy()
        # Cut out dropouts (how the GP fit works)
        x_train = x_train[mask_do_chunk] 

        # If requested, remove points specified in dropout_mask 
        # from x_pred. This means that the prediction skips the 
        # data points determined by dropout_mask. In the output, 
        # those skipped data points will be assigned NaN.
        if not predict_all:
            x_pred = x_pred[mask_do_chunk]
            y_chunk = y_chunk[mask_do_chunk]
            # Set skipped (i.e. not predicted) data points to NaN 
            # in output
            skipped = interval[~mask_do_chunk]
            y_pred[skipped] = np.nan
            y_desp[skipped] = np.nan
            # Also remove masked points from interval, as it is
            # used later
            interval = interval[mask_do_chunk]

        # ********************************
        # Train the data and fit GP model
        # ********************************

        yn_mean = np.nanmean(y_chunk_nodo)
        # Copy chunk for GP fitting
        y_train = y_chunk_nodo.copy()
        # Remove mean
        y_train -= yn_mean

        if kernel is None:
            # kernel = parameterization of covariance matrix + 
            # Gaussian noise, Eq. (5) in Bohlinger et al.
            kernel = (ConstantKernel(1.0) * \
                    RBF(length_scale=length_scale_bounds[0], 
                        length_scale_bounds=length_scale_bounds) + \
                    WhiteKernel(noise_level=1.0))
        # Define GP
        gp = gaussian_process.GaussianProcessRegressor(
            kernel=kernel)
        # Train GP process on data
        gp.fit(np.array(x_train).reshape(-1,1), 
                np.array(y_train).reshape(-1,1))
        gp.kernel_
        if print_kernel:
            print('GP kernel (sklearn): ', gp.kernel_)
        # Compute coefficient of determination R^2
        score = gp.score(x_train.reshape(-1,1), 
                            y_train.reshape(-1,1))
        # Make prediction based on GP (fills in missing points).
        # Remember, x_pred includes the entire range of indices 
        # in the chunk, unless predict_all is set to False.
        # TODO: This might not be optimal for spike detection, 
        # as found when testing with the time series from 
        # 3 Jan 2019, 00:40. A better way might be to run an 
        # initial spike detection run in which dropout and 
        # obvious spike locations are removed from x_pred, 
        # as this may have a greater success rate at detecting 
        # spikes. Afterward, another GP fit could be run with 
        # the full x_pred vector, in order to fill in all gaps.
        y_pred_n, sigma_n = gp.predict(np.array(x_pred).reshape(-1,1), 
                return_std=True)
        # Save trained hyperparams and R^2 score
        l = gp.kernel_.k1.get_params()['k2__length_scale']
        sigma_f = np.sqrt(
            gp.kernel_.k1.get_params()['k1__constant_value'])
        sigma_y = np.sqrt(
            gp.kernel_.k2.get_params()['noise_level'])
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
        if despike and score < score_thresh:
            # Detect outliers and replace both newly found spikes 
            # and dropouts
            mask = np.logical_and(y_chunk>y_pred_n-2*sigma_n, 
                                  y_chunk<y_pred_n+2*sigma_n)
            if print_kernel:
                print('Spikes found: {} \n'.format(np.sum(~mask)))
            mask = np.logical_or(~mask, ~mask_do_chunk)
            mask = ~mask
        else:
            # Only replace points defined in dropout_mask
            mask = mask_do_chunk.copy()
        spike_mask[interval] = mask
        # Replace spikes and dropouts by corresponding y_pred values
        y_chunk[~mask] = y_pred_n[~mask]
        y_desp[interval] = y_chunk


    # Don't want the initial dropouts to be included in the returned 
    # spike mask -> add those back
    spike_mask[dropout_mask==0] = 1

    if return_aux is True:
        return y_desp, spike_mask, y_pred, sigma, y_samp, theta
    else:
        # Only return the despiked signal
        return y_desp



def phase_space_3d(ts, replace_single='linear', replace_multi='linear',
        threshold='universal', figname=None):
    """
    3D phase space method for despiking originally developed by
    Goring and Nikora (2002), and later modified by Wahl (2003) and
    Mori (2005). 
    
    This function is based on the Matlab function 
    func_despike_phasespace3d.m by Mori (2005).

    Parameters:
        ts - time series to despike
        threshold - either 'universal' or 'chauvenet'
    """
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    # Universal threshold lambda_u: the expected maximum of N 
    # independent samples drawn from a standard normal distribution 
    # (Goring & Nikora, 2002)
    N = len(ts)
    if threshold == 'universal':
        lambda_u = np.sqrt(2*np.log(N))
    # Chauvenet's criterion is independent of the size of the 
    # sample (Wahl, 2003)
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
    theta = np.arctan2(np.dot(u, du2), np.sum(u**2).astype(float))

    # Look for outliers in 3D phase space
    # Following the func_excludeoutlier_ellipsoid3d.m Matlab script 
    # by Mori
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
    
    # G&N02: For each pair of variables, calculate the ellipse that 
    # has max. and min. from the previous computation. Use the MAD 
    # (median of absolute deviation) instead of std for more robust 
    # scale estimators (Wahl, 2003). 
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
        # Check for outliers from the ellipsoid by subtracting the 
        # ellipsoid corresponding to the data (x1, y1, z1) from the 
        # ellipsoid given by a,b and c. If the difference is less
        # than 0 the point lies outside the ellipsoid.
        dis = (x2**2 + y2**2 + z2**2) - (x1**2 + y1**2 + z1**2)
        if dis < 0:
            spike_mask[i] = 1

#     # Replace spikes if any were detected
#     if spike_mask.sum() != N:
#         # Replace spikes by interpolation
#         u = swi.replace_1d(u, spike_mask, 
#                 replace_single=replace_single,
#                 replace_multi=replace_multi)


    # Add back median
    u = u + ts_med

#     return u, spike_mask
    return spike_mask

