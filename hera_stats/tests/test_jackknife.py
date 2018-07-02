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

        self.filepath = os.path.join(DATA_PATH,"gaussian.N18.2018-06-06.06_15_48/")
        self.beampath = os.path.join(DATA_PATH, "NF_HERA_Beams.beamfits")

        uvd = UVData()
        uvd.read_miriad(self.filepath)
        cosmo = hp.conversions.Cosmo_Conversions()
        beam = hp.PSpecBeamUV(self.beampath, cosmo=cosmo)
        ds = hp.PSpecData([uvd, uvd], [None, None], beam=beam)
        blpairs = [p for p in uvd.get_antpairs() if p[0]+1 == p[1]]
        bsl1, bsl2, blpairs = hp.utils.construct_blpairs(blpairs,
                                                         exclude_auto_bls=True,
                                                         exclude_permutations=True)
        self.uvp = ds.pspec(bsl1, bsl2, (0,1), ("XX", "XX"), spw_ranges=[(600,610)], verbose=False)


    def test_bootstrap_and_save(self):

        uvp = self.uvp
        hs.jackknives._bootstrap_single_uvp(uvp, "xx")
        uvpl = hs.jackknives.split_ants(uvp, 1)
        uvplb = hs.jackknives.bootstrap_jackknife(uvpl, "xx")

        pc = hs.PSpecContainer("./test", "rw")
        hs.jackknives.save_jackknife(pc, uvplb)

        os.system("rm -f ./test*")

    def test_split_ants(self):

        uvpl = hs.jackknives.split_ants([self.uvp], 1, verbose=True)

        nt.assert_raises(AssertionError, hs.jackknives.split_ants, 20000)
        nt.assert_raises(AssertionError, hs.jackknives.split_ants,
                        [self.uvp]*2, 1, True)

    def test_stripe_times(self):

        uvp = hs.jackknives.stripe_times(self.uvp, verbose=True)
        uvp = hs.jackknives.stripe_times(self.uvp, 100)
        uvp = hs.jackknives.stripe_times(self.uvp, [100, 200])
        nt.assert_equal(np.array(uvp).shape, (2, 2))
        nt.assert_raises(AssertionError, hs.jackknives.stripe_times, np.array([]))

    def test_split_files(self):

        uvp = [self.uvp]*2
        files = ["thebestfile","thebiggestfile"]
        
        uvpl = hs.jackknives.split_files(uvp, files, "best", verbose=True)

        nt.assert_raises(AttributeError, hs.jackknives.split_files, uvp, files, "whattheheckisthis")
        nt.assert_raises(AttributeError, hs.jackknives.split_files, uvp, files, "18")
        nt.assert_raises(AttributeError, hs.jackknives.split_files, uvp, files, "18",[[0,1]])

    def test_split_gha(self):
        uvp = hs.jackknives.split_gha([self.uvp], bins_list=[10])
        uvp = hs.jackknives.split_gha(self.uvp, bins_list=[np.linspace(228.2, 228.4, 4)], specify_bins=True)
        
        nt.assert_raises(AttributeError, hs.jackknives.split_gha, self.uvp, [(0,1,2)], True)

    def test_omit_ants(self):

        uvp = hs.jackknives.omit_ants([self.uvp], [0, 1, 2])
        nt.assert_equals(np.asarray(uvp).shape, (1, 3))
        
        uvp = hs.jackknives.omit_ants([self.uvp], 1)
        

