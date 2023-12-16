# Overview of functions

The python files in this folder contain general functions used both in processing and analysis scripts. The contents of the .py files are summarized below; see docstrings for the individual functions for detailed usage instructions.

In order to be able to import these functions, make sure to add the base repository to your Python path: `export PYTHONPATH="$PYTHONPATH:PATH TO/pyROXSI"`.

* [adcp_funs.py](adcp_funs.py): Only contains the contamination_range() function, used to estimate sidelobe contamination range for ADCP data.

* [coordinate_transforms.py](coordinate_transforms.py): Various functions used to transform variables from one coordinate system to another; e.g., instrument (x,y,z) to Earth (u,v,w) coordinate transforms for Vector ADV velocities (`uvw2enu()`) and beam to Earth coordinates for Signature1000 ADCP velocities (`beam2enu()`).

* [despike.py](despike.py): Despiking functions for both velocity and surface elevation data.

* [plotting.py](plotting.py): Useful wrapper functions for plotting things with matplotlib.

* [stats.py](stats.py): Statistics functions. Currently only contains the function `r_squared()`.

* [transfer_functions.py](transfer_functions.py): Linear and (weakly) nonlinear transfer functions. Also contains functions to transform frequencies to wavenumbers using linear dispersion relation. Test/example script at the end.

* [turbulence.py](turbulence.py): Functions to estimate turbulence spectra and dissipation rates from ADV velocity data.

* [wave_spectra.py](wave_spectra.py): Functions to estimate wave spectra (both 1D and frequency-directional spectra from velocity data) and standard integrated parameters. Also includes a function (`bispectrum()`) for estimating bispectra.

* [zero_crossings.py](zero_crossings.py): Functions for zero-crossing analysis.
