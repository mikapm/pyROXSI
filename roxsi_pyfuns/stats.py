import numpy as np

def r_squared(y_obs, y_pred):
    """
    Compute R^2 for fit of model to observations following example in
    https://stackoverflow.com/questions/19189362/
    getting-the-r-squared-value-using-curve-fit

    Parameters:
        y_obs - np.array; observed data
        y_pred - np.array; modeled data

    Returns:
        r_squared - R^2 value
    """
    residuals = y_obs - y_pred
    # Sum of squared residuals
    ss_res = np.sum(residuals**2)
    # Sum of squared differences
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared