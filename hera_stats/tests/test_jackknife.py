import hera_stats as hs
import hera_pspec as hp
import pyuvdata
import nose.tools as nt
import numpy as np


class Test_Jackknife():

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg",warn=False)

        self.filepath = "./hera_stats/data/gaussian.N18.2018-06-06.06_15_48/"

        self.jk = hs.jackknife()
        self.jk.load_uvd(self.filepath)

        self.spw = [(600,700)]
        self.beampath = "./hera_stats/data/NF_HERA_Beams.beamfits"
        self.jk.calc_uvp(self.spw,beampath=self.beampath,baseline=(0,1))

    def test_init(self):

        jk = hs.jackknife()
        jk.load_uvd(self.filepath,verbose=True)
        nt.assert_is_instance(jk.uvd, pyuvdata.UVData)
        jk.find_files("./hera_stats/data/",".jkf")

    def test_calc_uvp(self):

        jk = self.jk

        spw = self.spw
        bp = self.beampath
        bsl = (0,1)
        pols = ("XX","XX")
        use_ants = None

        jk.calc_uvp(spw,beampath=bp,baseline=bsl,pols=pols,use_ants=use_ants)

        nt.assert_is_instance(jk.uvp, hp.uvpspec.UVPSpec)
        nt.assert_raises(ValueError, jk.calc_uvp, spw, (0,3), pols, bp, 
                         "blackman-harris", use_ants=use_ants)
        nt.assert_raises(ValueError, jk.calc_uvp, spw,bsl, ("YY","YY"), bp, 
                         "blackman-harris", use_ants=use_ants)
        nt.assert_raises(ValueError, jk.calc_uvp, spw,bsl, pols, bp, 
                         "blackman-harris", use_ants=[0,1,2,3,4,5,6])
        nt.assert_raises(ValueError, jk.calc_uvp, spw, (1,11), pols, bp, 
                         "blackman-harris", use_ants=[0,1])

    def test_split_ants(self):

        jk = self.jk

        uvpl, grps = jk.jackknives.split_ants()
        nt.assert_true(len(grps[0]) == len(grps[1]))

        for i,uv in enumerate(uvpl):
            blpairs = np.unique(uv.blpair_array)
            ants = np.array([uv.blpair_to_antnums(b) for b in blpairs]).flatten()
            nt.assert_true(sum([a in grps[i] for a in ants]) == len(ants))

    def test_bootstrap(self):

        jk = self.jk
        uvpl,grps = jk.jackknives.split_ants()

        dlys,spectra,errs = jk.bootstrap_errs_once(uvpl[0])
        nt.assert_equal(dlys.shape,spectra.shape,errs.shape)

        dlys,spectra,errs = jk.bootstrap_errs(uvpl)
        nt.assert_equal(len(spectra),2)

    def test_jackknife_short(self):

        jk = self.jk

        nt.assert_raises(NameError, jk.jackknife, "nothing", 1, self.spw, self.beampath)

        dic = jk.jackknife("split_ants", 1, self.spw, self.beampath, baseline=(0,1),
                           n_boots = 10, returned = True,calc_avspec=True,verbose=True,
                           savename="test")

        nt.assert_true("spectra" in dic.keys())
        nt.assert_true("dlys" in dic.keys())
        nt.assert_true("errs" in dic.keys())