"""
Functions to perform various coordinate transforms.
"""

import numpy as np

def uvw2enu(vel, heading, pitch, roll, magdec, deg_in=True):
    """
    Transform velocities (u,v,w) from instrument coordinates 
    (x,y,z) to Earth coordinates (East, North, Up) using 
    heading, pitch and roll information. Based on 
    vector_coord_trans.m function by Johanna Rosman.

    It is assumed that x is approximately horizontal and 
    heading is angle from north clockwise to the x direction.

    The vertical coordinate z should be approximately upward.

    Parameters:
        vel - [3,N] array; time series of velocity components in 
              xyz coordinates
        heading - array; time series of heading
        pitch - array; time series of pitch
        roll - array; time series of roll
        magdec - scalar; magnetic declinations (deg E)
        deg_in - bool; if True, the input angles are in degrees.
                 Set to False if input angles are in radians.

    Returns
        vel_out - rotated (E,N,U) velocities
    """
    # Copy xyz velocity components so we don't change the input
    ux = vel[:,0].copy()
    vy = vel[:,1].copy()
    wz = vel[:,2].copy()

    # Convert angles to radians if needed
    if deg_in:
        heading *= np.pi/180
        pitch *= np.pi/180
        roll *= np.pi/180
        magdec *= np.pi/180
    # Correct heading for magnetic declination
    heading += magdec

    # Define trigonometric terms
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    cos_r = np.cos(roll)
    sin_r = np.sin(roll)

    # Rotate to level coordinate system
    ul = ux*cos_p + vy*sin_p*sin_r + wz*cos_r*sin_p
    vl = vy*cos_r - wz*sin_r
    wl = -ux*sin_r + vy*cos_p*sin_r + wz*cos_r*cos_p

    # Rotate to East and North using heading
    uN = ul*cos_h + vl*sin_h
    uE = ul*sin_h - vl*cos_h

    # Merge output arrays
    vel_out = np.array([uE, uN, wl])

    return vel_out
    
