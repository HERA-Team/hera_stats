import os
import hera_stats as hs
from hera_stats.data import DATA_PATH
import nose.tools as nt
import numpy as np
import copy
import hera_pspec as hp
from pyuvdata import UVData
import shutil
import unittest


def test_make_uvp_data():
    # specify names
    uvp_data_name = "testing_uvp_data.h5"
    jack_data_name = "testing_jack_data.h5"

    hs.testing.make_uvp_data(jack_psc_name=jack_data_name, uvp_psc_name=uvp_data_name, overwrite=True)

    nt.assert_true(os.path.exists(uvp_data_name))
    nt.assert_true(os.path.exists(jack_data_name))

    uvp_psc = hp.PSpecContainer(uvp_data_name, mode='r')
    nt.assert_equal(len(uvp_psc.spectra('IDR2_1')), 2)

    jack_psc = hp.PSpecContainer(jack_data_name, mode='r')
    nt.assert_equal(len(jack_psc.spectra("jackknives")), 75)

    os.remove(uvp_data_name)
    os.remove(jack_data_name)