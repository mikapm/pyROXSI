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
    

def beam2enu(beam_vel, heading, pitch, roll, theta=25, nortek=True, 
             deg_in=True, beam5=True):
    """
    Transform 5-beam ADCP velocities from beam coordinates
    to Earth (east, north, up) coordinates using heading,
    pitch and roll timeseries. Based on the janus5beam2earth.m 
    function from https://github.com/apaloczy/ADCPtools.

    From the janus5beam2earth.m docstring (note order of velocities):

        For Nortek instruments, call function like this:
        enu_vel = beam2enu(np.array([-b1, -b3, -b4, -b2, -b5]),
                           heading, roll, ptch, theta, )

    Note: in this implementation, the heading and pitch 
    angles and sign (for Nortek data) are corrected within 
    the function.

    Parameters:
        beam_vel - array; beam velocities
        heading - array; time series of heading
        pitch - array; time series of pitch
        roll - array; time series of roll
        theta - scalar; beam angle from vertical
        nortek - bool; if True, assumes headings are from Nortek
                 instrument, which means that they need to be
                 adjusted by 90 degrees.
        deg_in - bool; if True, assumes input angles are in deg
        beam5 - bool; if True, calculates velocities using 5th beam.
                If False, uses only beams 1-4.
    
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
    # Copy input velocities
    vb1 = beam_vel[0,:,:].copy()
    vb2 = beam_vel[1,:,:].copy()
    vb3 = beam_vel[2,:,:].copy()
    vb4 = beam_vel[3,:,:].copy()
    vb5 = beam_vel[4,:,:].copy()

    # Convert angles to radians if needed
    if deg_in:
        heading *= np.pi/180
        pitch *= np.pi/180
        roll *= np.pi/180
    
    # Correct Nortek heading angles and pitch sign if needed
    if nortek:
        ang_corr = 90 * np.pi/180
        heading -= ang_corr
        pitch *= -1

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
    # Option 1: five-beam solution for [u, v, w].
    # Option 2: Classic four-beam solution.
    cx1 = Cph1 * Cph3 + Sph1 * Sph2 * Sph3
    cx2 = Sph1 * Cph3 - Cph1 * Sph2 * Sph3
    cx3 = Cph2 * Sph3
    cy1 = Sph1 * Cph2
    cy2 = Cph1 * Cph2
    cy3 = Sph2
    cz1 = Cph1 * Sph3 - Sph1 * Sph2 * Cph3
    cz2 = Sph1 * Sph3 + Cph1 * Sph2 * Cph3
    cz3 = Cph2 * Cph3

    # Convert beam-referenced velocities to instrument-referenced 
    # velocities.
    # NOTE: The convention used here (positive x axis = 
    # horizontally away from beam 1) and positive y axis = 
    # horizontally away from beam 3) is not the same as the 
    # one used by the instrument's firmware if the coordinate 
    # transformation mode is set to "instrument coordinates" 
    # before deployment.
    xyz_vel = beam2xyz(np.array([vb1, vb2, vb3, vb4, vb5]), theta)
    print('xyz_vel.shape: ', xyz_vel.shape)

    w5 = np.multiply(xyz_vel[3,:,:],  cz3) # w from beam 5 only.

    if beam5:
        # Use vertical 5th beam to calculate vE, vN & vU
        vE = (np.multiply(xyz_vel[0,:,:], cx1) + 
              np.multiply(xyz_vel[1,:,:], cy1) +
              np.multiply(xyz_vel[3,:,:], cz1))
        print('shape: ', vE.shape)
        vN = (-np.multiply(xyz_vel[0,:,:], cx2) + 
              np.multiply(xyz_vel[1,:,:], cy2) - 
              np.multiply(xyz_vel[3,:,:], cz2))
        vU = (-np.multiply(xyz_vel[0,:,:], cx3) + 
              np.multiply(xyz_vel[1,:,:], cy3) + w5)
    else:
        # Use only beams 1-4
        vE = + xyz_vel[0,:,:]*cx1 + xyz_vel[1,:,:]*cy1 + xyz_vel[2,:,:]*cz1
        vN = -xyz_vel[0,:,:]*cx2 + xyz_vel[1,:,:]*cy2 - xyz_vel[2,:,:]*cz2
        vU = -xyz_vel[0,:,:]*cx3 + xyz_vel[1,:,:]*cy3 + xyz_vel[2,:,:]*cz3

    return np.array([vE.T, vN.T, vU.T, w5.T])
    

def beam2xyz(beam_vel, theta):
    """
    Convert 5-beam ADCP beam velocities to instrument-referenced
    x,y,z coordinates. Based on the janus5beam2xyz.m function
    from https://github.com/apaloczy/ADCPtools.

    Parameters:
        beam_vel - array; beam velocities
        theta - scalar; beam angle from vertical

    Returns:
        xyz_vel - (4,nt,nz) array; velocities in instrument coordinates.
                  Note: fourth component is the z velocity component 
                  from beam 5 only. The third component is the z 
                  velocity component considering the conventional form
                  of w (average of the estimates of w from planes 1-2 
                  and 3-4).
    """
    # Check input array shape
    if beam_vel.shape[1] < beam_vel.shape[2]:
        nb, nz, nt = beam_vel.shape
    else:
        nb, nt, nz = beam_vel.shape
        # Swap axes so the matrix has the right shape (nb, nz, nt)
        beam_vel = np.swapaxes(beam_vel, 1, 2)
    print('beam_vel.shape: ', beam_vel.shape)

    # Multiplication factors
    uvfac = 1 / (2 * np.sin(theta))
    wfac = 1 / (4 * np.cos(theta)) # For w derived from beams 1-4.
    # 3rd row: w from the average of the 4 beams.
    # 4rd row: w from the 5th beam only.
    A = np.array([[-1, 1, 0, 0, 0],
                  [0, 0, -1, 1, 0],
                  [-1, -1, -1, -1, 0],
                  [0, 0, 0, 0, -1]])

    # Transform velocity components bin-wise
    xyz_vel = np.zeros((4, nz, nt)) # Output array
    for z in range(nz):
        for t in range(nt):
            xyz_vel[:,z,t] = A @ beam_vel[:,z,t].squeeze()

    # Multiply velocity components with respective factors
    xyz_vel[0,:,:] *= uvfac # East velocity
    xyz_vel[1,:,:] *= uvfac # North velocity
    xyz_vel[2,:,:] *= wfac # Up velocity

    return xyz_vel

