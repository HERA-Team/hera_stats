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

## Features

`hera_stats` currently has the following modules:

* `automate`: Functions to replace placeholder variables in template Jupyter notebooks (`jupyter_replace_tags`) and run them programmatically (`jupyter_run_notebook`).
* `average`: More advanced averaging functions for power spectra, currently only cumulative averaging in time or baseline-pair (`average_spectra_cumul`).
* `flag`: Advanced flagging algorithms and utilities, including a way to randomly flag frequency channels (`apply_random_flags`), a convenience function to flag whole ranges of channels at once (`flag_channels`), and an implementation of a 'greedy' flagging algorithm (`construct_factorizable_mask`) that can construct factorizable (in time and frequency) masks that flag as small a total fraction of the data as possible.
* `noise`: Simple empirical noise estimation functions, including a function (`estimate_noise_rms`) to estimate the noise rms by differencing data in the time direction and fit a smooth polynomial model that interpolates over flagged channels.
* `plots`: A wide variety of diagnostic plotting functions. These include a function to generate long waterfalls plots of `nsamples` or `flags` across many files (`long_waterfall`), including summary statistics on the flag fraction for each time/frequency bin.
* `shuffle`: Functions to randomly shuffle data. This includes a function to construct new visibilities by shuffling samples from a set of visibilities within the same redundant baseline group (`shuffle_data_redgrp`).
* `stats`: Various statistical convenience functions to compared jackknife power spectra.
