import hera_stats as hs
import matplotlib.pyplot as plt
import nose.tools as nt
import os
from hera_stats.data import DATA_PATH

import matplotlib
import warnings
warnings.simplefilter("ignore",matplotlib.mplDeprecation)

class Test_Plots():

    def setUp(self):
        self.filepath = os.path.join(DATA_PATH,"gauss.spl_ants.Nj40.2018-06-07.04_50_54.jkf")
        plt.ioff()

        pl = hs.plots()
        pl.load_file(self.filepath)
        self.pl = pl

    def test_init(self):

        pl = hs.plots()

        nt.assert_raises(IOError, pl.load_file, "nothing")

        pl.load_file(self.filepath)
        nt.assert_is_instance(pl.stats, hs.jkf_stats)
        nt.assert_is_instance(pl.data, hs.utils.jkfdata)

        nt.assert_raises(AttributeError,pl.sort,100)

    def test_plots(self):

        f = plt.figure()
        ax = f.add_subplot(111,label="h")
        
        pl = self.pl

        pl.plot_spectra(fig=f)
        pl.plot_spectra(fig=f,show_groups=False)
        nt.assert_raises(ValueError,pl.plot_spectra,100,fig=f)

        pl.plot_spectra_errs(fig=f)
        pl.plot_spectra_errs(fig=f,show_groups=False)
        nt.assert_raises(ValueError,pl.plot_spectra_errs,100,fig=f)
        f.clear()

        for mode in ["varsum","weightedsum","raw","norm"]:
            pl.hist_2d(ax=ax,plottype = mode,display_stats=False)
            ax.clear()

        nt.assert_raises(NameError, pl.hist_2d, "nothing")
        pl.plot_kstest(ax=ax)
        pl.plot_anderson(ax=ax)
        pl.plot_av_std(ax=ax)

        ax.clear()
        pl.sort(1)
        fig = None
        ax = None

        pl.plot_spectra(fig=f)
        pl.plot_spectra(fig=f,show_groups=True)
        nt.assert_raises(ValueError,pl.plot_spectra,100,fig=f)

        pl.plot_spectra_errs(fig=f)
        pl.plot_spectra_errs(fig=f,show_groups=True)
        nt.assert_raises(ValueError,pl.plot_spectra_errs,100,fig=f)

        for mode in ["varsum","weightedsum","raw","norm"]:
            pl.hist_2d(ax=ax,plottype = mode, normalize=True,vmax=0.2)

        nt.assert_raises(NameError, pl.hist_2d, "nothing")
        pl.plot_kstest(ax=ax)
        pl.plot_anderson(ax=ax)
        pl.plot_av_std(ax=ax)
        
        pl.sort(None)
        pl.find_outliers()
        pl.find_bad_items()



