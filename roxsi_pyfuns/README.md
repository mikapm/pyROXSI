# Overview of functions

The python files in this folder contain general functions used both in processing and analysis scripts. The contents of the .py files are summarized below; see docstrings for the individual functions for detailed usage instructions.

* [adcp_funs.py](adcp_funs.py): Only contains the contamination_range() function, used to estimate sidelobe contamination range for ADCP data.

* [coordinate_transforms.py](coordinate_transforms.py): Various functions used to transform variables from one coordinate system to another; e.g., instrument (x,y,z) to Earth (u,v,w) coordinate transforms for Vector ADV velocities (`uvw2enu()`) and beam to EArth coordinates for Signature1000 ADCP velocities (`beam2enu()`).

* [despike.py](despike.py): Despiking functions for both velocity and surface elevation data.

* [plotting.py](plotting.py): Useful wrapper functions for plotting things with matplotlib.

* [stats.py](stats.py): Statistics functions. Currently only contains the function `r_squared()`.

* [transfer_functions.py](transfer_functions.py):  
