"""
Plotting functions.
"""

import numbers
import numpy as np
import matplotlib.pyplot as plt


def fit_ellipse_svd(x, y, ax=None, plot_ellipse=False, plot_semiaxes=True,
    **kwargs):
    """
    Fit an ellipse to arrays x and y using singular value decomposition (SVD).

    Borrowed from
    http://notmatthancock.github.io/2016/02/03/ellipse-princpal-axes.html

    Parameters:
        x - 1D array; x-axis values (same length as y)
        y - 1D array; y-axis values (same length as x)
        ax - matplotlib axes object; if not None, plots ellipse and
             semi axes on ax
        plot_ellipse - bool; if True, plots ellipse on ax
        plot_semiaxes - bool; if True, plots semi-axes on ax
        ** kwargs for ax.plot()

    Returns:
        ellipse - array for fitted ellipse
        angle - rotation angle of ellipse (rad)
        a - scalar; semi-major axis of ellipse
        b - scalar; semi-minor axis of ellipse
    """
    # Means
    xmean = np.nanmean(x)
    ymean = np.nanmean(y)
    # Number of points
    N = len(x)
    X = np.c_[x, y] # Concatenate x and y
    t = np.linspace(0, 1, N, endpoint=False)
    # Rotation matrix
    rotation_matrix = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    # Fit the ellipse.
    u, s, vt = np.linalg.svd((X-X.mean(axis=0))/np.sqrt(N), full_matrices=False)
    # Semi axes
    a = np.sqrt(2) * s[0]
    b = np.sqrt(2) * s[1]
    # Ellipse function
    ellipse = np.sqrt(2) * np.c_[s[0]*np.cos(2*np.pi*t), s[1]*np.sin(2*np.pi*t)]
    # Rotation angle of ellipse 
    angle = np.arctan2(vt[0,1], vt[0,0])
    # Check if angle is negative
#     if angle < 0:
#         # Convert to positive angle
#         angle += 2 * np.pi
    # Want angle in 1st or 4th quadrant
    if abs(angle) >= np.pi/2:
        if angle > 0:
            angle -= np.pi
        else:
            angle += np.pi
    # Rotate ellipse
    ellipse = np.dot(rotation_matrix(angle), ellipse.T).T
    # Add mean x and mean y (origin) to ellipse
    ellipse[:,0] += xmean
    ellipse[:,1] += ymean

    # Plot if requested
    if ax is not None:
        ta = np.linspace(0, a) 
        tb = np.linspace(0, b)
        # Ellipse
        if plot_ellipse:
            ax.plot(ellipse[:,0], ellipse[:,1], **kwargs)
        # Semi axes
        if plot_semiaxes:
            ax.plot(xmean + ta*np.cos(angle), ymean + ta*np.sin(angle), 
                **kwargs)
            ax.plot(xmean + tb*np.cos(angle+np.pi/2), 
                    ymean + tb*np.sin(angle+np.pi/2), 
                    **kwargs)
    
    return ellipse, angle, a, b


def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, scatter=True, **kwargs):
    """
    Draw a quantile-quantile plot for `x` versus `y`.

    Borrowed from user 'Artem Mavrin' at 
    https://stats.stackexchange.com/questions/403652/two-sample-quantile-quantile-plot-in-python

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    scatter : bool; if False, plots line plot instead of scatter plot

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles)
    y_quantiles = np.quantile(y, quantiles)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    if scatter:
        ax.scatter(x_quantiles, y_quantiles, **kwargs)
    else:
        # Return line plot instead
        ax.plot(x_quantiles, y_quantiles, **kwargs)



def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    """
    Axis labels as mul;tiples of pi. Borrowed from Scott Centoni:
    https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
    """
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, 
        self.number, self.latex))
