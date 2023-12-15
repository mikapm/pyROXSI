# Raw data processing routines

## Overview of files and their functionality

### [ADCP processing: signature_preprocess.py](signature_preprocess.py)

Process raw Nortek Signature1000 ADCP data. Note that this file takes .mat files as input. The script outputs daily netCDF files that contain surface elevation (both AST and pressure-derived) and currents.

Main/example script (with input arguments for running via command line) at the end of the file, after `if __name__ == '__main__':`.

The main class is called `ADCP()`; see docstring for `__init__()` for further initialization instructions. The main functions under the ADCP class are summarized below (see also the docstrings of each individual function for more detailed instructions):

`_fns_from_ser()`: Get a sorted list of .mat files for given serial number. Used to combine raw data from several smaller (1Gb) .mat files into more easily readable netcdf files.

`contamination_range()`: Estimate near-surface contamination range due to side lobe reflections.

`echo2ds()`: Read raw echosounder data (from .mat) into xarray.Dataset. Only used for analysis.

`ampcorr2ds()`: As above, but for beam velocity amplitude and correlation arrays.

`loaddata_vel()`: Main data reader function for both AST and velocities. Input flags for e.g. QC; see docstring for detailed instructions.
