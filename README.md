# pyROXSI
Functions and scripts for processing and analyzing data from the ROXSI experiments. The main focus of these functions is the small-scale instrument array deployed at Asilomar beach, Monterey, CA in June-July 2022.

## Installation
```
$ mamba env create -f environment.yml
$ conda activate roxsi
$ # Install pip libraries
$ /home/mikapm/miniconda3/envs/roxsi/bin/pip install ssqueezepy
$ /home/mikapm/miniconda3/envs/roxsi/bin/pip install mat73
```

## Update environment
```
mamba env update --file environment.yml --prune
```

## Usage
This repository consists mainly of two types of files: Python scripts for processing raw ADCP and ADV data, and Jupyter Notebooks for analyzing the processed data.

Most processing scripts and a readme with instructions are located in the **roxsi-pyfuns/preprocess** folder. 

Analysis notebooks are located in the **roxsi-pyfuns/notebooks** folder.

General functions used in both processing scripts and analysis notebooks are located in the main **roxsi-pyfuns** folder.

