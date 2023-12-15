# Raw data processing routines

## Overview of files and their functionality

### [ADCP processing: signature_preprocess.py](signature_preprocess.py)

Process raw Nortek Signature1000 ADCP data. Note that this file takes .mat files as input. The script outputs daily netCDF files that contain surface elevation (both AST and pressure-derived) and currents.

Main/example script (with input arguments for running via command line) at the end of the file, after `if __name__ == '__main__':`.

The main class is called `ADCP()`; see docstring for `__init__()` for further initialization instructions. The main functions under the ADCP class are summarized below (see also the docstrings of each individual function for more detailed instructions):

* `_fns_from_ser()`: Get a sorted list of .mat files for given serial number. Used to combine raw data from several smaller (1Gb) .mat files into more easily readable netcdf files.

* `contamination_range()`: Estimate near-surface contamination range due to side lobe reflections.

* `echo2ds()`: Read raw echosounder data (from .mat) into xarray.Dataset. Only used for analysis.

* `ampcorr2ds()`: As above, but for beam velocity amplitude and correlation arrays.

* `loaddata_vel()`: Main data reader function for both AST, pressure and velocities. Input flags for e.g. QC; see docstring for detailed instructions. Also converts velocities from beam coordinates to East, North, Up components using the heading, pitch and roll time series (see the `beam2enu()` function in [../coordinate_transforms.py](../coordinate_transforms.py)).

* `despike_GN02()`: Velocity despiking scheme following Goring and Nikora (2002). By default not used for ADCP data, only ADV data (see [vector_preprocess.py](vector_preprocess.py)). To use, set `despike_vel=True` in the `loaddata_vel()` function. Despikes velocity timeseries for each depth bin separately.

* `despike_GP()`: AST despiking scheme using Gaussian Process (GP) based method of Malila et al. (2023). Used by default in `loaddata_vel()`; set `despike_ast=False` to disable. Quite slow, but seems to work well for the specific noise characteristics of the AST signal. The main script of `signature_preprocess.py` also saves separate .csv files containing only the despiked AST signals, which can be read later if needed.

* `p2z_lin()`: Linear transfer function for pressure--sea surface transformation.

* `p2eta_krms()`: Linear and weakly nonlinear pressure--sea surface transfer functions following Martins et al. (2021). Uses root-mean-square wavenumbers of Herbers et al. (2002) instead of linear wavenumbers. This can help delay the blow up of high-frequency components in intermediate water depth.

* `wavespec()`: Estimate (directional) wave spectra using surface elevation (AST or pressure-derived) and East/North velocity components. Calls the `spec_uvz()` function in [../wave_spectra.py](../wave_spectra.py).

* `save_vel_nc()`: Saves velocity/AST/pressure data read and processed from .mat files into netCDF. Also sets CF-compliant units and attributes to variables and coordinates.

* `save_spec_nc()`: Saves wave spectra to separate netCDF files. Also tries to set CF-compliant units and attributes.

### [ADV processing: vector_preprocess.py](vector_preprocess.py) 

Process raw Nortek Vector ADV data. Similar structure to the [signature_preprocess.py](signature_preprocess.py) script, with main class called `ADV()`. Instead of .mat files, the ADV readers use Nortek's .dat, .sen, and .hdr file formats. The central functions under the `ADV` class are described below. Again, see individual functions' docstrings for detailed usage instructions.

* `loaddata()`: Main data reader for velocities and pressure. Uses the .dat files for reading raw data, and .sen files for heading, pitch and roll (H,P,R) time series. The H,P,R data is only saved at 1-Hz resolution, so it is linearly interpolated to the 16-Hz sampling rate of the ADV velocities.
