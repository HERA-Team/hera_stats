import hera_stats as hs
import hera_pspec as hp
from pyuvdata import UVData
import nose.tools as nt
import numpy as np
import os
import copy
from hera_stats.data import DATA_PATH
import shutil
import unittest


uvp_data_name = "jk_uvp_data.h5"

def setup_module():
    """ used to make data for the entire test script """
    hs.testing.make_uvp_data(uvp_psc_name=uvp_data_name, overwrite=True)

def teardown_module():
    """ used to remove data used by the entire test script """
    if os.path.exists(uvp_data_name):
        os.remove(uvp_data_name)

class Test_Jackknife(unittest.TestCase):

    def setUp(self):
        self.filepath = uvp_data_name
        pc = hp.container.PSpecContainer(self.filepath, mode='r')
        self.uvp = pc.get_pspec("IDR2_1")[0]
        self.uvp_list = pc.get_pspec("IDR2_1")[0]

    def test_bootstrap_and_save(self):
        np.random.seed(0)
        uvp = self.uvp
        hs.jackknives._bootstrap_single_uvp(uvp, "xx")
        uvpl = hs.jackknives.split_ants(uvp, 1)
        uvplb = hs.jackknives.bootstrap_jackknife(uvpl, "xx")
        if os.path.exists('test_save_jackknife'):
            os.remove('test_save_jackknife')
        pc = hs.PSpecContainer("./test_save_jackknife", "rw")
        hs.jackknives.save_jackknife(pc, uvplb)
        os.remove('test_save_jackknife')

    def test_split_ants(self):
        np.random.seed(0)
        uvpl = hs.jackknives.split_ants([self.uvp], 1, verbose=True)
        nt.assert_raises(AssertionError, hs.jackknives.split_ants, 20000)
        nt.assert_raises(AssertionError, hs.jackknives.split_ants,
                        [self.uvp]*2, 1, True)

    def test_stripe_times(self):
        np.random.seed(0)
        uvp = hs.jackknives.stripe_times(self.uvp, verbose=True)
        uvp = hs.jackknives.stripe_times(self.uvp, 10.)
        uvp = hs.jackknives.stripe_times(self.uvp, [10., 20.])
        nt.assert_equal(np.array(uvp).shape, (2, 2))
        nt.assert_raises(AssertionError, hs.jackknives.stripe_times, np.array([]))

    def test_split_gha(self):
        np.random.seed(0)
        uvp = hs.jackknives.split_gha([self.uvp], bins_list=[1])
        uvp = hs.jackknives.split_gha(self.uvp, bins_list=[np.linspace(228.2, 228.3, 1)], specify_bins=True)  
        nt.assert_raises(AttributeError, hs.jackknives.split_gha, self.uvp, [(0,1,2)], True)

    def test_omit_ants(self):
        np.random.seed(0)
        uvp = hs.jackknives.omit_ants([self.uvp], [37, 38, 39])
        nt.assert_equals(np.asarray(uvp).shape, (1, 3))
        uvp = hs.jackknives.omit_ants([self.uvp], 37)

    def test_sep_files(self):
        np.random.seed(0)
        sep = hs.jackknives.sep_files(self.uvp_list, ["f1", "f2"])


if __name__ == "__main__":
    setup_module()
    unittest.main()
    teardown_module()

