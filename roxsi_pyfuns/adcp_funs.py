"""
General ADCP analysis/processing functions.
"""

import numpy as np

def contamination_range(ha, binsz=0.5, beam_angle=25):
    """
    Calculate range of velocity contamination due to sidelobe 
    reflections following Lentz et al. (2021, Jtech): 
    DOI: 10.1175/JTECH-D-21-0075.1

    According to L21 (Eq. 2), the range cells contaminated by 
    sidelobe reflections are given by
        
        z_ic < h_a * (1 - cos(theta)) + 3*dz/2,
    
    where z_ic is depth below the surface of the contaminated region
    (range cell centers), h_a is the distance from the ADCP acoustic 
    head to the surface, theta is the ADCP beam angle from the 
    vertical (25 for Sig1000) and dz is the bin size in meters. 

    Parameters:
        ha - distance from ADCP acoustic head to sea surface
        binsz - binsize in meters
        beam_angle - beam angle in degrees

    Returns:
        zic - depth below the surface of the contaminated region
    """
    zic = ha * (1 - np.cos(beam_angle)) + 3 * binsz / 2
    return zic