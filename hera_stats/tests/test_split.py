import numpy as np
import hera_stats as hs
import hera_pspec as hp
from pyuvdata import UVData
from hera_stats.data import DATA_PATH
import nose.tools as nt
import os
import unittest


class Test_Jackknife(unittest.TestCase):

    def setUp(self):
        self.filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        pc = hp.container.PSpecContainer(self.filepath, mode='r')
        self.uvp = pc.get_pspec("IDR2_1")[0]
        self.uvp_list = pc.get_pspec("IDR2_1")[0]

    def test_split_ants(self):
        np.random.seed(0)
        uvpl = hs.split.split_ants([self.uvp], 1, verbose=True)
        nt.assert_raises(AssertionError, hs.split.split_ants, 20000)
        nt.assert_raises(AssertionError, hs.split.split_ants,
                        [self.uvp]*2, 1, True)

    def test_stripe_times(self):
        np.random.seed(0)
        uvp = hs.split.stripe_times(self.uvp, verbose=True)
        uvp = hs.split.stripe_times(self.uvp, 10.)
        uvp = hs.split.stripe_times(self.uvp, [10., 20.])
        nt.assert_equal(np.array(uvp).shape, (2, 2))
        nt.assert_raises(AssertionError, hs.split.stripe_times, np.array([]))

    def test_split_gha(self):
        np.random.seed(0)
        uvp = hs.split.split_gha([self.uvp], bins_list=[1])
        uvp = hs.split.split_gha(self.uvp, 
                                 bins_list=[np.linspace(228.2, 228.3, 1)], 
                                 specify_bins=True)  
        nt.assert_raises(AttributeError, hs.split.split_gha, self.uvp, 
                         [(0,1,2)], True)

    def test_omit_ants(self):
        np.random.seed(0)
        uvp = hs.split.omit_ants([self.uvp], [37, 38, 39])
        nt.assert_equals(np.asarray(uvp).shape, (1, 3))
        uvp = hs.split.omit_ants([self.uvp], 37)

    def test_sep_files(self):
        np.random.seed(0)
        sep = hs.split.sep_files(self.uvp_list, ["f1", "f2"])


if __name__ == "__main__":
    unittest.main()

