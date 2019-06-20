import numpy as np
import hera_pspec as hp
import hera_stats as hs
from hera_stats.data import DATA_PATH
import os
import nose.tools as nt
import unittest


class test_testing():

    def setUp(self):
        self.uvp_data_name = os.path.join(DATA_PATH, "testing_uvp_data.h5")
        self.jack_data_name = os.path.join(DATA_PATH, "testing_jack_data.h5")
    
    def tearDown(self):
        if os.path.exists(self.uvp_data_name):
            os.remove(self.uvp_data_name)
        if os.path.exists(self.jack_data_name):
            os.remove(self.jack_data_name)

    def test_make_uvp_data(self):
    
        hs.testing.make_uvp_data(jack_psc_name=self.jack_data_name, 
                                 uvp_psc_name=self.uvp_data_name, 
                                 overwrite=True)
        
        nt.assert_true(os.path.exists(self.uvp_data_name))
        nt.assert_true(os.path.exists(self.jack_data_name))

        uvp_psc = hp.PSpecContainer(self.uvp_data_name, mode='r')
        nt.assert_equal(len(uvp_psc.spectra('IDR2_1')), 2)

        jack_psc = hp.PSpecContainer(self.jack_data_name, mode='r')
        nt.assert_equal(len(jack_psc.spectra("jackknives")), 26)
