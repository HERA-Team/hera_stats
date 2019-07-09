# ``hera_stats``: HERA statistics and null tests

[![Build Status](https://travis-ci.org/HERA-Team/hera_stats.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_stats)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_stats/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_stats?branch=master)
[![Documentation](https://readthedocs.org/projects/hera-stats/badge/?version=latest)](https://readthedocs.org/projects/hera-stats/badge/?version=latest)

The ``hera_stats`` module provides a collection of functions and container 
objects to help calculate various statistics on sets of delay spectra, and 
manage splits and other aspects of null tests.

For usage examples and documentation, see http://hera-stats.readthedocs.io/en/latest/.

## Installation
Preferred method of installation for users is simply `pip install .`
(or `pip install git+https://github.com/HERA-Team/hera_stats`). This will install 
required dependencies. See below for manual dependency management.
 
Optionally, if you want to make use of the Jupyter notebook automation features 
in the `hera_pspec.automate` module::

    $ conda/pip install jupyter
    
### Dependencies
If you are using `conda`, you may wish to install the following dependencies manually
to avoid them being installed automatically by `pip`::

    $ conda install -c conda-forge "numpy>=1.15" "astropy>=2.0" "aipy>=3.0rc2" h5py pyuvdata scipy matplotlib pyyaml h5py scikit-learn

### Developing
If you are developing `hera_stats`, it is preferred that you do so in a fresh `conda`
environment. The following commands will install all relevant development packages::

    $ git clone https://github.com/HERA-Team/hera_stats.git
    $ cd hera_stats
    $ conda create -n hera_stats python=3
    $ conda activate hera_stats
    $ conda env update -n hera_stats -f environment.yml
    $ pip install -e . 

This will install extra dependencies required for testing/development as well as the 
standard ones.

### Running Tests
Uses the `nose` package to execute test suite.
From the source `hera_stats` directory run: `nosetests`.

### Code Dependencies

* numpy >= 1.15
* pyuvdata (`pip install pyuvdata` or use https://github.com/HERA-Team/pyuvdata.git)
* hera_pspec (https://github.com/HERA-Team/hera_pspec.git)
* hdf5

Optionally, if you want to make use of the Jupyter notebook automation features 
in the `hera_pspec.automate` module:
* jupyter (`pip install jupyter`)

For anaconda users, we suggest using conda to install numpy.


## Running `hera_stats`

There are some Jupyter notebooks in the [`examples/`](examples/) subdirectory 
with examples of how to use various features of `hera_stats`.
