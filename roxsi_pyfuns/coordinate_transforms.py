"""
Functions to perform various coordinate transforms.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation   
from roxsi_pyfuns import wave_spectra as rpws


def rotate_pca(ux, uy, uz=None, return_r=False, return_eul=False, 
               flipx=False, flipy=False, flipz=False, heading_exp=None,
               eul1_o=None, eul2_o=None, eul3_o=None):
    """
    Rotate x,y or x,y,z components according to their principal axes
    using principal component analysis (PCA).

    Parameters:
        ux - shape N array; x-component of e.g. velocity
        uy - shape N array; y-component of e.g. velocity
        uz - shape N array; z-component of e.g. velocity (optional)
        return_r - bool; if True, returns rotation matrix R
        return_eul - bool; if True, returns Euler angles dict 'eul'
        flipx - bool; if True, flips (*-1) first column in R
        flipy - bool; if True, flips (*-1) second column in R
        flipz - bool; if True, flips (*-1) third column in R
        heading_exp - scalar; expected heading angle
        eul1_o - scalar; angles (deg) to add to eul1 for testing
        eul2_o - scalar; angles (deg) to add to eul2 for testing
        eul3_o - scalar; angles (deg) to add to eul3 for testing

    Returns:
        rot_arr - shape (N,2) or (N,3) array; rotated vectors such that:
            u_pc1 = rot_arr[:,0]
            u_pc2 = rot_arr[:,1]
            u_pc3 = rot_arr[:,2]
        if return_r is True:
            R - shape 3,3 array; rotation matrix (eigenvectors from PCA)
        if return_eul is True:
            eul - dict; Euler angles: eul1 = rotation about y axis (pitch)
                                      eul2 = rotation about x axis (roll)
                                      eul3 = rotation about z axis (heading)
    """
    # Check if 2D or 3D
    if uz is not None:
        # 3D array
        ndim = 3
        # Combine vectors into one array
        vel_arr = np.vstack([ux.squeeze(), uy.squeeze(), uz.squeeze()]).T
    else:
        # Set return_eul to False just in case
        return_eul = False
        # 2D array
        ndim = 2
        # Combine vectors into one array
        vel_arr = np.vstack([ux.squeeze(), uy.squeeze()]).T
    # Get principal components
    pca = PCA(n_components=ndim)
    pca.fit(vel_arr)
    # Get eigenvectors (ie rotation matrix R)
    R = pca.components_
    # Flip axes in R?
    if flipx:
        # R[:,0] *= (-1)
        R[0,:] *= (-1)
    if flipy:
        # R[:,1] *= (-1)
        R[1,:] *= (-1)
    if flipz:
        # R[:,2] *= (-1)
        R[2,:] *= (-1)

    # Euler angles
    eul = {} # Output dict
    # Get Euler angles from rotation matrix R if ndim=3
    if ndim == 3:
#         eul['eul1'] = -np.arcsin(R[2,0]) # Pitch
#         eul['eul2'] = np.arctan2(R[2,1] / np.cos(eul['eul1']), 
#                                  R[2,2] / np.cos(eul['eul1'])) # Roll
#         eul['eul3'] = np.arctan2(R[1,0] / np.cos(eul['eul1']), 
#                                  R[0,0] / np.cos(eul['eul1'])) # Heading
        r = Rotation.from_matrix(R)
        # Sequence is 'x,y,z' b/c heading is eul3 (?)
        angles = r.as_euler("xyz", degrees=False)
        # Change angles (for testing)?
        if eul1_o is not None:
            angles[0] += eul1_o
        if eul2_o is not None:
            angles[1] += eul2_o
        if eul3_o is not None:
            angles[2] += eul3_o
        # Flag for angle offset testing
        testing = eul1_o is not None or eul2_o is not None or eul3_o is not None
        if testing:
            # Get new rotation matrix based on offset angle(s)
            R = Rotation.from_euler(angles)
        # Get angle components for output
        eul['eul1'], eul['eul2'], eul['eul3'] = angles

    # Rotate velocity components with R
    rot_arr = R.dot(vel_arr.T).T

    # Only return rotated components?
    if not return_eul and not return_r:
        return rot_arr
    # Return Euler angles but not R?
    elif return_eul and not return_r:
        return rot_arr, eul
    # Return R but not Euler angles?
    elif return_r and not return_eul:
        return rot_arr, R
    # Return all 3?
    elif return_r and return_eul:
        return rot_arr, R, eul

def enu_to_loc_pca(ux, uy, uz, heading_exp=None, print_msg=False,
                   return_eul=False):
    """
    Use PCA rotation to convert from instrument coordinates to
    local cross-shore (PC1), along-shore (PC2) and vertical (PC3)
    velocities.

    Performs various checks based on PCA rotation angles; designed
    to work with ROXSI 2022 Asilomar Vector ADV velocities.

    Parameters:
        ux - shape N array; x-component of e.g. velocity
        uy - shape N array; y-component of e.g. velocity
        uz - shape N array; z-component of e.g. velocity
        heading_exp - scalar; expected heading angle in deg
        print_msg - bool; prints messages if True
    """
    # First rotate velocities based on principal axes
    vel_pca, R, eul = rotate_pca(ux=ux, uy=uy, uz=uz, return_r=True, 
                                 return_eul=True, )
    # Check if R is left-handed (det(R)=-1) 
    if np.linalg.det(R) < 0 and heading_exp is not None: 
        # Check if heading is off relative to expected heading
        if np.abs(np.abs(heading_exp) - np.abs(np.rad2deg(eul['eul3']))) > 45:
            if print_msg:
                print('Flipping y axis ...')
            # Redo rotation, but flip y axis
            vel_pca, R, eul = rotate_pca(ux=ux, uy=uy, uz=uz, return_r=True, 
                                         return_eul=True, flipy=True, ) 
    # Assume u_pc1=ucs, u_pc2=uls, u_pc3=uw
    ucs = vel_pca[:,0].copy()
    uls = vel_pca[:,1].copy()
    uw = vel_pca[:,2].copy()
    # Check if some component(s) need to be flipped
    if np.abs(np.rad2deg(eul['eul1'])) > 90:
        # z-axis points downward -> flip vertical velocity
        if print_msg:
            print('Flipping vertical velocity ... ')
        uw *= (-1)
    if np.abs(np.rad2deg(eul['eul2'])) > 90:
        # z-axis points downward -> flip vertical velocity
        if print_msg:
            print('Flipping vertical velocity ... ')
        uw *= (-1)
    # Check if heading is off if det(R) = 1
    if np.linalg.det(R) > 0 and heading_exp is not None:
        if np.abs(np.abs(heading_exp) - np.abs(np.rad2deg(eul['eul3']))) > 90:
            if print_msg:
                print('Flipping horizontal velocity ... ')
            ucs *= (-1)
    # Compute coherence and phase to check components
    spec_r = rpws.spec_uvz(z=uw, u=ucs, v=uls, fs=16)
    # Get index of max. coherence^2 b/w ucs and uw
    ind_mcu = np.argmax((spec_r.coh_uz**2).sel(freq=slice(0.05, 0.3)).values).item()
    ind_mcv = np.argmax((spec_r.coh_vz**2).sel(freq=slice(0.05, 0.3)).values).item()
    # Compute ucs-uw phase at max coherence
    pmc = np.rad2deg(spec_r.ph_uz.sel(freq=slice(0.05, 0.3)).isel(freq=ind_mcu).item())
    # If phase is +90 -> flip ux velocity
    if pmc > 0:
        ucs *= (-1)
        # Compute coherence and phase again
        spec_r = rpws.spec_uvz(z=uw, u=ucs, v=uls, fs=16)
        pmc = np.rad2deg(spec_r.ph_uz.sel(freq=slice(0.05, 0.3)).isel(freq=ind_mcu).item())
    # Print ucs-uw phase at max coherence
    if print_msg:
        print('ucs-uw phase at max coh: {:.2f}'.format(pmc))

    if not return_eul:
        return ucs, uls, uw
    else:
        return ucs, uls, uw, eul


def rotate_euler(ux, uy, uz, eul1, eul2, eul3):
    """
    Rotate vectors ux, uy, uz by Euler angles eul1 (pitch),
    eul2 (roll) and eul3 (heading).

    Parameters:
        ux - shape N array; x-component of e.g. velocity
        uy - shape N array; y-component of e.g. velocity
        uz - shape N array; z-component of e.g. velocity
        eul1 - Euler angle for rotation about y axis (pitch)
        eul2 - Euler angle for rotation about x axis (roll)
        eul3 - Euler angle for rotation about z axis (heading)

    Returns:
        rot_arr - shape (N,3) array; rotated vectors such that:
            ux_rot = rot_arr[:,0]
            uy_rot = rot_arr[:,1]
            uz_rot = rot_arr[:,2]
    """
    # Combine vectors into one array
    vel_arr = np.vstack([ux.squeeze(), uy.squeeze(), uz.squeeze()]).T

    # Generate rotation matrix about z axis (heading/yaw)
    Rz = np.array([[np.cos(eul3), -np.sin(eul3), 0],
                   [np.sin(eul3),  np.cos(eul3), 0],
                   [0,             0,            1]
                  ])
    # Generate rotation matrix about y axis (pitch)
    Ry = np.array([[np.cos(eul1),  0, np.sin(eul1)],
                   [0,             1, 0           ],
                   [-np.sin(eul1), 0, np.cos(eul1)]
                  ])
    # Generate rotation matrix about x axis (roll)
    Rx = np.array([[1, 0,             0           ],
                   [0, np.cos(eul2), -np.sin(eul2)],
                   [0, np.sin(eul2),  np.cos(eul2)]
                  ])

    # Combine individual rotations to one rotation matrix R
    # R = Rz @ Ry @ Rx
    r = Rotation.from_euler('xyz', [eul1, eul2, eul3])
    R = r.as_matrix()
    # Rotate vectors
    rot_arr = R.dot(vel_arr.T).T

    return rot_arr, R



def rotate_zgrid(xg, yg, zg, angle_rad):
    """
    Rotate 2D grid with z coordinates by angle given by angle_rad (in radians). 

    Borrowed from user 'hpaulj' at 
    https://stackoverflow.com/questions/31816754/numpy-einsum-for-rotation-of-meshgrid

    Parameters:
        xg - 2D array of x coordinates
        yg - 2D array of y coordinates
        zg - 2D array of z coordinates
        angle_rad - angle to rotate grid (radians) in math convention
    
    Returns:
        xr - 2D array of rotated x coordinates
        yr - 2D array of rotated y coordinates
        zr - 2D array of rotated z coordinates
    """
    # Check if NaNs in zg
    if np.sum(np.isnan(zg)):
        nans = True
        zg = np.nan_to_num(zg, nan=-999.)
    else:
        nans = False
    # Stack grids
    xyz = np.vstack([xg.ravel(), yg.ravel(), zg.ravel()]).T
    # Make rotation matrix
    sin = np.sin(angle_rad)
    cos = np.cos(angle_rad)
    rot = [[cos, sin, 0],
           [-sin,  cos, 0],
           [0, 0, 1]]
    # Rotate 3D grid
    xyz_r = np.einsum('ij,kj->ki', rot, xyz)
    xrot = xyz_r[:,0].reshape(xg.shape)
    yrot = xyz_r[:,1].reshape(yg.shape)
    zrot = xyz_r[:,2].reshape(zg.shape)
    # Put back potential NaNs in zr
    if nans == True:
        zrot[zrot==-999.] = np.nan

    return xrot, yrot, zrot


def rotate_xygrid(xspan, yspan, RotRad=0):
    """
    Generate a meshgrid and rotate it by RotRad radians.

    Borrowed from user 'wilywampa' at
    https://stackoverflow.com/questions/29708840/rotate-meshgrid-with-numpy
    """

    # Clockwise, 2D rotation matrix
    RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                          [-np.sin(RotRad), np.cos(RotRad)]])

    x, y = np.meshgrid(xspan, yspan)
    return np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x, y]))


def rotate_vel(u, v, rot=0):
    """
    Rotate velocity components u and v by angle rot [rad].
    """
    # Clockwise, 2D rotation matrix
    RotMatrix = np.array([[np.cos(rot),  np.sin(rot)],
                          [-np.sin(rot), np.cos(rot)]])
    return RotMatrix @ np.vstack([u, v])


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


def uvw2enu(vel, heading, pitch, roll, magdec=0, deg_in=True, 
            inverted=False):
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
        inverted - bool; set to True if Vector coord. system is inverted
                   (due to instrument being upside-down when recording
                   was started).

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

    # Was Vector inverted when recording was started?
    if inverted:
        # Fix heading (opposite dir. of rotation)
        hd = 360 - hd
        # Fix pitch and roll
        pt *= -1
        rl *= -1

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
             deg_in=True, beam5=False):
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
    # Copy input velocities and angles
    vb1 = beam_vel[0,:,:].copy()
    vb2 = beam_vel[1,:,:].copy()
    vb3 = beam_vel[2,:,:].copy()
    vb4 = beam_vel[3,:,:].copy()
    vb5 = beam_vel[4,:,:].copy()
    hd = heading.copy()
    pt = pitch.copy()
    rl = roll.copy()

    # Convert angles to radians if needed
    if deg_in:
        hd = np.deg2rad(hd)
        pt = np.deg2rad(pt)
        rl = np.deg2rad(rl)
        theta = np.deg2rad(theta)

    # Correct Nortek heading angles and pitch sign if needed
    if nortek:
        ang_corr = np.deg2rad(90)
        hd -= ang_corr
        rl *= -1

    # Time-dependent angles (heading, pitch and roll).
    Sph1 = np.sin(hd)
    Sph2 = np.sin(rl)
    Sph3 = np.sin(pt)
    Cph1 = np.cos(hd)
    Cph2 = np.cos(rl)
    Cph3 = np.cos(pt)

    # Correct pitch (D&S 2007, eq. A1)
    pitch = np.arcsin((Sph2 * Cph3) / np.sqrt(1 - (Sph2 * Sph3)**2))
    Sph2 = np.sin(rl)
    Cph2 = np.cos(rl)

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

    w5 = np.multiply(xyz_vel[3,:,:],  cz3) # w from beam 5 only.

    if beam5:
        # Use vertical 5th beam to calculate vE, vN & vU
        vE = (np.multiply(xyz_vel[0,:,:], cx1) + 
              np.multiply(xyz_vel[1,:,:], cy1) +
              np.multiply(xyz_vel[3,:,:], cz1))
        vN = (-np.multiply(xyz_vel[0,:,:], cx2) + 
              np.multiply(xyz_vel[1,:,:], cy2) - 
              np.multiply(xyz_vel[3,:,:], cz2))
        vU = (-np.multiply(xyz_vel[0,:,:], cx3) + 
              np.multiply(xyz_vel[1,:,:], cy3) + 
              w5)
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

    # Multiplication factors
    uvfac = 1 / (2 * np.sin(theta))
    wfac = 1 / (4 * np.cos(theta)) # For w derived from beams 1-4.
    # 3rd row: w from the average of the 4 beams.
    # 4rd row: w from the 5th beam only.
    A = np.array([[-1,  1,  0,  0,  0],
                  [ 0,  0, -1,  1,  0],
                  [-1, -1, -1, -1,  0],
                  [ 0,  0,  0,  0, -1]])

    # Transform velocity components bin-wise
    xyz_vel = np.zeros((4, nz, nt)) # Output array
    for z in range(nz):
        for t in range(nt):
            xyz_vel[:,z,t] = A @ beam_vel[:,z,t].squeeze()

    # Multiply velocity components with respective factors
    xyz_vel[0,:,:] *= uvfac # East velocity
    xyz_vel[1,:,:] *= uvfac # North velocity
    xyz_vel[2,:,:] *= wfac # Up velocity 1

    return xyz_vel

