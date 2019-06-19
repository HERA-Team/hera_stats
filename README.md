# ``hera_stats``: HERA statistics and null tests

[![Build Status](https://travis-ci.org/HERA-Team/hera_stats.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_stats)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_stats/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_stats?branch=master)
[![Documentation](https://readthedocs.org/projects/hera-stats/badge/?version=latest)](https://readthedocs.org/projects/hera-stats/badge/?version=latest)

The ``hera_stats`` module provides a collection of functions and container 
objects to help calculate various statistics on sets of delay spectra, and 
manage splits and other aspects of null tests.

For usage examples and documentation, see http://hera-stats.readthedocs.io/en/latest/.

## Installation

### Code Dependencies

* numpy >= 1.15
* pyuvdata (`pip install pyuvdata` or use https://github.com/HERA-Team/pyuvdata.git)
* hera_pspec (https://github.com/HERA-Team/hera_pspec.git)
* hdf5

Optionally, if you want to make use of the Jupyter notebook automation features 
in the `hera_pspec.automate` module:
* jupyter (`pip install jupyter`)

For anaconda users, we suggest using conda to install numpy.

### Installing hera_stats
Clone the repo using
`git clone https://github.com/HERA-Team/hera_stats.git`

Change to the `hera_stats` directory and run `python setup.py install` (or `python setup.py install --user` to install for a single user, without using admin privileges).

## Running `hera_stats`

There are some Jupyter notebooks in the [`examples/`](examples/) subdirectory 
with examples of how to use various features of `hera_stats`.
