import hera_stats as hs
import os
import hera_pspec as hp
from hera_stats.data import DATA_PATH
import nose.tools as nt
import copy
import numpy as np
import shutil
import unittest


jack_data_name = "jkset_jack_data.h5"

def setup_module():
    """ used to make data for the entire test script """
    hs.testing.make_uvp_data(jack_psc_name=jack_data_name, overwrite=True)

def teardown_module():
    """ used to remove data used by the entire test script """
    if os.path.exists(jack_data_name):
        os.remove(jack_data_name)


class Test_JKSet(unittest.TestCase):

    def setUp(self):
        filepath = jack_data_name
        self.pc = hp.container.PSpecContainer(filepath)

    def test_init(self):
        jk = hs.JKSet(self.pc, "spl_ants")
        nt.assert_raises(AssertionError, hs.JKSet, self.pc, "unknown")

    def test_functions(self):
        jk = hs.JKSet(self.pc, "spl_ants")
        jk2 = copy.deepcopy(jk)
        jk2.spectra = jk2.spectra*0 + 1.

        nt.assert_true(jk.shape == (20, 2))
        nt.assert_true(jk.jktype == "spl_ants")
        nt.assert_true(all([attr.shape[:2] == (20, 2) for attr in [jk.spectra, jk.errs, jk.times, jk.integrations, jk.nsamples]]))

        nt.assert_true(np.all(np.moveaxis(jk.spectra, 0, 1) == jk.T().spectra))
        nt.assert_true(jk.flatten().shape == (40,))
        nt.assert_true(jk.flatten() == jk.reshape(40))
        
        nt.assert_false(jk == jk2)
        nt.assert_true(jk == jk.add(jk2, axis=1, inplace=False)[:, :2])
        print jk

        jk2.set_data(jk.spectra, jk.errs, error_field='bs_std')
        nt.assert_true(jk == jk2)
        jk[0, 0]

        jk = hs.JKSet(self.pc, "spl_ants")
        nt.assert_true(jk == hs.JKSet(jk._uvp_list, "spl_ants"))

if __name__ == "__main__":
    setup_module()
    unittest.main()
    teardown_module()

