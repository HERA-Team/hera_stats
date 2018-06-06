import matplotlib
matplotlib.use("Agg",warn=False)
import hera_stats as hs
import matplotlib.pyplot as plt
import nose.tools as nt


class Test_Plots():

    def setUp(self):
        self.filepath = "./hera_stats/data/gauss.spl_ants.Nj40.2018-06-05.21_21_17.jkf"
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
    
    def test_plots(self):
        
        f = plt.figure()
        
        pl = self.pl
        
        pl.plot_spectra(fig=f)
        plt.close()
        pl.plot_spectra(fig=f,show_groups=True)
        plt.close()
        nt.assert_raises(ValueError,pl.plot_spectra,100,fig=f)
        
        pl.plot_spectra_errs(fig=f)
        plt.close()
        pl.plot_spectra_errs(fig=f,show_groups=True)
        plt.close()
        nt.assert_raises(ValueError,pl.plot_spectra_errs,100,fig=f)
        plt.close()
        
        for mode in ["varsum","weightedsum","raw","norm"]:
            pl.hist_2d(plottype = mode)
            plt.close()
        
        nt.assert_raises(NameError, pl.hist_2d, "nothing")
        plt.close()
        
        pl.plot_kstest()
        plt.close()
        pl.plot_anderson()
        plt.close()
        pl.plot_av_std()
        plt.close()