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
        magdec - scalar; magnetic declination (deg E)
        deg_in - bool; if True, the input angles are in degrees.
                 Set to False if input angles are in radians.

    Returns
        vel_out - rotated (E,N,U) velocities
    """
    # Copy xyz velocity components so we don't change the input
    ux = vel[:,0].copy()
    vy = vel[:,1].copy()
    wz = vel[:,2].copy()
    hd = heading.copy()
    pt = pitch.copy()
    rl = roll.copy()

    # Convert angles to radians if needed
    if deg_in:
        hd *= np.pi/180
        pt *= np.pi/180
        rl *= np.pi/180
        magdec *= np.pi/180
    # Correct heading for magnetic declination
    hd += magdec

    # Define trigonometric terms
    cos_h = np.cos(hd)
    sin_h = np.sin(hd)
    cos_p = np.cos(pt)
    sin_p = np.sin(pt)
    cos_r = np.cos(rl)
    sin_r = np.sin(rl)

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
    

def beam2enu(beam_vel, heading, pitch, roll, magdec=0, theta=25,
             nortek=True, deg_in=True):
    """
    Transform 5-beam ADCP velocities from beam coordinates
    to Earth (east, north, up) coordinates using heading,
    pitch and roll timeseries. Based on the janus2earth.m 
    function from https://github.com/apaloczy/ADCPtools.

    Parameters:
        beam_vel - array; beam velocities
        heading - array; time series of heading
        pitch - array; time series of pitch
        roll - array; time series of roll
        magdec - scalar; magnetic declination (deg E)
        theta - scalar; beam angle from vertical
        nortek - bool; if True, assumes headings are from Nortek
                 instrument, which means that they need to be
                 adjusted by 90 degrees.
        deg_in - bool; if True, assumes input angles are in deg
    
    Returns:
        enu_vel - array; velocity components in ENU coordinates

    From the janus2earth.m docstring:
    Nortek convention:
    * Velocity toward transducers' faces: NEGATIVE
    * Counter-clockwise PITCH (tilt about y-AXIS, equivalent to 
      -ROLL in the TRDI convention): POSITIVE (beam 1 higher than 
      beam 3)
    * Clockwise ROLL (tilt about x-AXIS, equivalent to PITCH in 
      the TRDI convention):  POSITIVE (beam 4 higher than beam 2)

    References:
        Appendix A of Dewey & Stringer (2007), Equations A3-A11
    """
    # Convert angles to radians if needed
    if deg_in:
        heading *= np.pi/180
        pitch *= np.pi/180
        roll *= np.pi/180
    
    # Correct Nortek heading angles if needed
    if nortek:
        ang_corr = 90 * np.pi/180
        heading -= ang_corr

    # Time-dependent angles (heading, pitch and roll).
    Sph1 = np.sin(heading)
    Sph2 = np.sin(pitch)
    Sph3 = np.sin(roll)
    Cph1 = np.cos(heading)
    Cph2 = np.cos(pitch)
    Cph3 = np.cos(roll)

    # Correct headings (D&S 2007, eq. A2)
    Sph2Sph3 = Sph2 * Sph3
    heading += np.arcsin(Sph2Sph3 / np.sqrt(Cph2**2 + Sph2Sph3**2))
    Sph1 = np.sin(heading)
    Cph1 = np.cos(heading)

    # Convert instrument-referenced velocities
    # to Earth-referenced velocities.
    # Option 1: Classic four-beam solution.
    # Option 2: five-beam solution for [u, v, w].
    cx1 = Cph1 * Cph3 + Sph1 * Sph2 * Sph3
    cx2 = Sph1 * Cph3 - Cph1 * Sph2 * Sph3
    cx3 = Cph2 * Sph3
    cy1 = Sph1 * Cph2
    cy2 = Cph1 * Cph2
    cy3 = Sph2
    cz1 = Cph1 * Sph3 - Sph1 * Sph2 * Cph3
    cz2 = Sph1 * Sph3 + Cph1 * Sph2 * Cph3
    cz3 = Cph2 * Cph3
