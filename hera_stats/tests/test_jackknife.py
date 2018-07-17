import hera_stats as hs
import hera_pspec as hp
from pyuvdata import UVData
import nose.tools as nt
import numpy as np
import os
import copy

from hera_stats.data import DATA_PATH

class Test_Jackknife():

    def setUp(self):

        self.filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        pc = hp.container.PSpecContainer(self.filepath)
        self.uvp = pc.get_pspec("IDR2_1")[0]
        self.uvp_list = pc.get_pspec("IDR2_1")[0]

    def test_bootstrap_and_save(self):

        uvp = self.uvp
        hs.jackknives._bootstrap_single_uvp(uvp, "xx")
        uvpl = hs.jackknives.split_ants(uvp, 1)
        uvplb = hs.jackknives.bootstrap_jackknife(uvpl, "xx")

        pc = hs.PSpecContainer("./test_save_jackknife", "rw")
        hs.jackknives.save_jackknife(pc, uvplb)

        os.system("rm -f ./test_save_jackknife")

    def test_split_ants(self):

        uvpl = hs.jackknives.split_ants([self.uvp], 1, verbose=True)

        nt.assert_raises(AssertionError, hs.jackknives.split_ants, 20000)
        nt.assert_raises(AssertionError, hs.jackknives.split_ants,
                        [self.uvp]*2, 1, True)

    def test_stripe_times(self):

        uvp = hs.jackknives.stripe_times(self.uvp, verbose=True)
        uvp = hs.jackknives.stripe_times(self.uvp, 21.)
        uvp = hs.jackknives.stripe_times(self.uvp, [21., 32.])
        nt.assert_equal(np.array(uvp).shape, (2, 2))
        nt.assert_raises(AssertionError, hs.jackknives.stripe_times, np.array([]))

    def test_split_gha(self):
        uvp = hs.jackknives.split_gha([self.uvp], bins_list=[1])
        uvp = hs.jackknives.split_gha(self.uvp, bins_list=[np.linspace(228.2, 228.3, 1)], specify_bins=True)
        
        nt.assert_raises(AttributeError, hs.jackknives.split_gha, self.uvp, [(0,1,2)], True)

    def test_omit_ants(self):

        uvp = hs.jackknives.omit_ants([self.uvp], [0, 1, 2])
        nt.assert_equals(np.asarray(uvp).shape, (1, 3))
        
        uvp = hs.jackknives.omit_ants([self.uvp], 1)
        

    def test_sep_files(self):

        sep = hs.jackknives.sep_files(self.uvp_list, ["f1", "f2"])
        
