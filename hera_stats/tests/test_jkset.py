import hera_stats as hs
import os
import hera_pspec as hp
from hera_stats.data import DATA_PATH
import nose.tools as nt
import copy
import numpy as np

class Test_JKSet():

    def setUp(self):
        filepath = os.path.join(DATA_PATH, "test_pc_jackknife.h5")
        self.pc = hp.container.PSpecContainer(filepath)

    def test_init(self):
        jk = hs.JKSet(self.pc, "spl_ants")

        nt.assert_raises(AssertionError, hs.JKSet, self.pc, "unknown")

    def test_functions(self):
        jk = hs.JKSet(self.pc, "spl_ants")
        jk2 = copy.deepcopy(jk)
        jk2.spectra = jk2.spectra*0 + 1.

        nt.assert_true(jk.shape == (40, 2))
        nt.assert_true(jk.jktype == "spl_ants")
        nt.assert_true(all([attr.shape[:2] == (40, 2) for attr in [jk.spectra, jk.errs, jk.times, jk.integrations, jk.nsamples]]))

        nt.assert_true(np.all(np.moveaxis(jk.spectra, 0, 1) == jk.T().spectra))
        nt.assert_true(jk.flatten().shape == (1, 80))
        nt.assert_true(jk.flatten() == jk.reshape(1, 80))
        
        nt.assert_false(jk == jk2)
        jk.add(jk2, axis=1, inplace=False)
        nt.assert_true(jk == jk.add(jk2, axis=1, inplace=False)[:, :2])
        print jk

        jk2.set_data(jk.spectra, jk.errs)
        nt.assert_true(jk == jk2)
        jk[0, 0]

        jk = hs.JKSet(self.pc, "spl_ants")
        nt.assert_true(jk == hs.JKSet(jk._uvp_list, "spl_ants"))