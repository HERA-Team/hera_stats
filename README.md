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
* more_itertools (`pip install more-itertools`)

Optionally, if you want to make use of the Jupyter notebook automation features
in the `hera_pspec.automate` module:
* jupyter (`pip install jupyter`)

For anaconda users, we suggest using conda to install numpy.


## Running `hera_stats`

There are some Jupyter notebooks in the [`examples/`](examples/) subdirectory
with examples of how to use various features of `hera_stats`.

## Features

`hera_stats` currently has the following modules:

* `automate`: Functions to replace placeholder variables in template Jupyter notebooks (`jupyter_replace_tags`) and run them programmatically (`jupyter_run_notebook`).
* `average`: More advanced averaging functions for power spectra, currently only cumulative averaging in time or baseline-pair (`average_spectra_cumul`) and differencing redundant groups with respect to their average (`redundant_diff`).
* `flag`: Flagging algorithms and utilities, including a way to randomly flag frequency channels (`apply_random_flags`), a convenience function to flag whole ranges of channels at once (`flag_channels`), and an implementation of a 'greedy' flagging algorithm (`construct_factorizable_mask`) that can construct factorizable (in time and frequency) masks that flag as small a total fraction of the data as possible.
* `noise`: Simple empirical noise estimation functions, including a function (`estimate_noise_rms`) to estimate the noise rms by differencing data in the time direction and fit a smooth polynomial model that interpolates over flagged channels.
* `plot`: A wide variety of diagnostic plotting functions. These include a function to generate long waterfalls plots of `nsamples` or `flags` across many files (`long_waterfall`), including summary statistics on the flag fraction for each time/frequency bin.
* `shuffle`: Functions to randomly shuffle data. This includes a function to construct new visibilities by shuffling samples from a set of visibilities within the same redundant baseline group (`shuffle_data_redgrp`).
* `split`: A range of convenience functions for splitting data in various ways.
* `stats`: Various statistical convenience functions to compare jackknife power spectra.
* `bias_jackknife`: Prototype to Chiborg (https://github.com/mwilensky768/chiborg.git)
code that acts as a jackknife test for finding biased subsets within data. The
code in this repo was used for the H1C IDR3 upper limit analysis.
