import hera_stats as hs
import hera_pspec as hp
import pyuvdata
import nose.tools as nt
import numpy as np
import os
import copy

from hera_stats.data import DATA_PATH

class Test_Jackknife():

    def setUp(self):

        self.filepath = os.path.join(DATA_PATH,"gaussian.N18.2018-06-06.06_15_48/")
        
        hs.utils.shorten([self.filepath])

        self.jk = hs.jackknife()
        self.jk.load_uvd(self.filepath)

        print "Done Loading!"
        self.spw = [(600,625)]
        self.beampath = os.path.join(DATA_PATH, "NF_HERA_Beams.beamfits")
        self.jk.calc_uvp(self.spw,beampath=self.beampath,baseline=(0,1))
        self.jk_small = copy.deepcopy(self.jk)
        self.jk_small.uvd[0].select(antenna_nums=[0,1,2])

    def test_init(self):

        jk = hs.jackknife()
        jk.load_uvd(self.filepath,use_ants=[0,1],verbose=True)
        nt.assert_is_instance(jk.uvd[0], pyuvdata.UVData)
        hs.utils.find_files(DATA_PATH,".jkf")

    def test_calc_uvp(self):

        jk = self.jk_small

        spw = self.spw
        bp = self.beampath
        bsl = (0,1)
        pols = ("XX","XX")

        jk.calc_uvp(spw,beampath=bp,baseline=bsl,pols=pols)

        nt.assert_is_instance(jk.uvp[0], hp.uvpspec.UVPSpec)
        nt.assert_raises(AttributeError, jk.calc_uvp, spw, (0,3), pols, bp, 
                         "blackman-harris")
        nt.assert_raises(ValueError, jk.calc_uvp, spw,bsl, ("YY","YY"), bp, 
                         "blackman-harris")
        nt.assert_raises(AttributeError, jk.calc_uvp, spw, (1,11), pols, bp, 
                         "blackman-harris")

    def test_split_ants(self):

        jk = self.jk

        uvpl, grps, n_pairs = jk.jackknives.split_ants(1)
        nt.assert_true(len(grps[0][0]) == len(grps[0][1]))

        for i,uv in enumerate(uvpl[0]):
            blpairs = np.unique(uv.blpair_array)
            ants = np.array([uv.blpair_to_antnums(b) for b in blpairs]).flatten()
            nt.assert_true(sum([a in grps[0][i] for a in ants]) == len(ants))

    def test_split_times(self):
        jk = self.jk_small
        uvpl, grps, n_pairs = jk.jackknives.split_times()

    def test_split_files(self):

        jk = hs.jackknife()
        jk.uvp = self.jk_small.uvp*2
        jk.jackknives.uvp = jk.uvp
        jk.jackknives.files = ["thebestfile","thebiggestfile"]

        uvpl,grps,n_pairs = jk.jackknives.split_files("best")
        #uvpl,grps,n_pairs = jk.jackknives.split_files(pairs=[[0,1]])
        
        nt.assert_raises(AttributeError, jk.jackknives.split_files,"whattheheckisthis")
        nt.assert_raises(AttributeError, self.jk_small.jackknives.split_files,"18")
        nt.assert_raises(AttributeError, jk.jackknives.split_files,"18",[[0,1]])


    def test_bootstrap(self):

        jk = self.jk
        uvpl,grps,n_pairs = jk.jackknives.split_ants(1)

        dlys,spectra,errs = jk.bootstrap_errs_once(uvpl[0][0])
        nt.assert_equal(dlys.shape,spectra.shape,errs.shape)

        dlys,spectra,errs = jk.bootstrap_errs(uvpl)
        nt.assert_equal(len(spectra[0]),2)

    def test_jackknife_short(self):

        jk = self.jk_small

        jk.clock_reset()
        nt.assert_raises(TypeError, jk.jackknife, "nothing", 1, self.spw, self.beampath)

        dic = jk.jackknife(jk.jackknives.split_times, self.spw, self.beampath, baseline=(0,1),
                           n_boots = 10, returned = True,verbose=True,
                           savename="test")

        nt.assert_true("spectra" in dic.keys())
        nt.assert_true("dlys" in dic.keys())
        nt.assert_true("errs" in dic.keys())