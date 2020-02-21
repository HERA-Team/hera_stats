import numpy as np
import hera_stats as hs
import matplotlib.pyplot as plt
import hera_pspec as hp
from hera_stats.data import DATA_PATH
from hera_pspec.data import DATA_PATH as PSPEC_DATA_PATH
from pyuvdata import UVData
import matplotlib
import warnings
warnings.simplefilter("ignore", matplotlib.mplDeprecation)
import nose.tools as nt
import os
import unittest


def axes_contains(ax, obj_list):
    """
    Check that a matplotlib.Axes instance contains certain elements.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        Axes instance.
        
    obj_list : list of tuples
        List of tuples, one for each type of object to look for. The tuple 
        should be of the form (matplotlib.object, int), where int is the 
        number of instances of that object that are expected.
    """
    # Get plot elements
    elems = ax.get_children()
    
    # Loop over list of objects that should be in the plot
    contains_all = False
    for obj in obj_list:
        objtype, num_expected = obj
        num = 0
        for elem in elems:
            if isinstance(elem, objtype): num += 1
        if num != num_expected:
            return False
    
    # Return True if no problems found
    return True


class Test_Plot(unittest.TestCase):

    def setUp(self):
        plt.ioff()
        
        # Filenames and other settings
        self.dfiles = ['zen.even.xx.LST.1.28828.uvOCRSA', 
                       'zen.odd.xx.LST.1.28828.uvOCRSA']
        self.dfile_paths = [os.path.join(PSPEC_DATA_PATH, dfile) 
                            for dfile in self.dfiles]
        self.uvh5_files = [ os.path.join(PSPEC_DATA_PATH, 
                                         'zen.2458116.31193.HH.uvh5'),]
        self.baseline = (38, 68, 'xx')
        
        # Load datafiles into UVData objects
        self.d = []
        for dfile in self.dfiles:
            _d = UVData()
            _d.read_miriad(os.path.join(PSPEC_DATA_PATH, dfile))
            self.d.append(_d)
        
        # Load example UVPSpec
        filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        psc = hp.container.PSpecContainer(filepath, mode='r')
        self.uvp = psc.get_pspec("IDR2_1")[0]
        
    
    def test_plot_redgrp_corrmat(self):
        red_grps, red_lens, red_angs = self.uvp.get_red_blpairs()
        grp = red_grps[0]
        if isinstance(grp[0], tuple):
            # FIXME: This would not be necessary if get_red_blpairs() returned 
            # properly-formed blpair integers
            grp = [int("%d%d" % _blp) for _blp in grp]
        
        # Calculate delay spectrum correlation matrix for redundant group and plot
        corr_re, corr_im = hs.stats.redgrp_pspec_covariance(
                                        self.uvp, grp, dly_idx=3, spw=0, 
                                        polpair='xx', mode='corr', verbose=True)
        fig = hs.plot.redgrp_corrmat(corr_re, grp, cmap='RdBu', 
                                     figsize=(30.,20.), line_alpha=0.2)
    
    def test_scatter_bandpowers(self):
        red_grps, red_lens, red_angs = self.uvp.get_red_blpairs()
        n_dlys = self.uvp.get_dlys(0).size
        keys = {
            'spws':     0,
            'blpairs':  red_grps[0],
            'polpairs': self.uvp.get_polpairs(),
        }
        ax = hs.plot.scatter_bandpowers(self.uvp, 2, n_dlys // 2, keys, 
                                        label="All polpairs, redgrp 0")
        # uvp, x_dly, y_dly, keys, operator='abs', label=None, ax=None)
        
        # Missing keys should raise an error
        nt.assert_raises(KeyError, hs.plot.scatter_bandpowers, self.uvp, 
                         2, n_dlys // 2,
                         {'blpairs':  red_grps[0],
                          'polpairs': self.uvp.get_polpairs(),})
        nt.assert_raises(KeyError, hs.plot.scatter_bandpowers, self.uvp, 
                         2, n_dlys // 2,
                         {'spws':     0, 
                          'polpairs': self.uvp.get_polpairs(),})
        nt.assert_raises(KeyError, hs.plot.scatter_bandpowers, self.uvp, 
                         2, n_dlys // 2,
                         {'spws':     0, 
                          'blpairs':  red_grps[0],})
    
    
    def test_long_waterfall(self):
        main_waterfall, freq_histogram, time_histogram, data \
            = hs.plot.long_waterfall(self.d, bl=(38, 68), pol='xx', 
                                     title='Flags Waterfall')
        
        # Make sure the main waterfall has the right number of dividing lines
        if np.round(data.shape[0]/60, 0) == 0: 
            main_waterfall_elems = [(matplotlib.lines.Line2D, 1)]
        else:
            main_waterfall_elems = [(matplotlib.lines.Line2D, \
                                    npround(data.shape[0]/60, 0))]
        nt.assert_true(axes_contains(main_waterfall, main_waterfall_elems))
        
        # Make sure the time graph has the appropriate line element
        time_elems = [(matplotlib.lines.Line2D, 1)]
        nt.assert_true(axes_contains(time_histogram, time_elems))
        
        # Make sure the freq graph has the appropriate line element
        freq_elems = [(matplotlib.lines.Line2D, 1)]
        nt.assert_true(axes_contains(freq_histogram, freq_elems))
        
        # Check to make sure operator kwarg works (i.e. no errors)
        wfall, hfreq, htime, data = hs.plot.long_waterfall(
                                       self.d, bl=(38, 68), pol='xx', 
                                       title='x', mode='data', operator='phase')
        wfall, hfreq, htime, data = hs.plot.long_waterfall(
                                       self.d, bl=(38, 68), pol='xx', 
                                       title='x', mode='data', operator='real')
        wfall, hfreq, htime, data = hs.plot.long_waterfall(
                                       self.d, bl=(38, 68), pol='xx', 
                                       title='x', mode='data', operator='imag')
        nt.assert_raises(ValueError, hs.plot.long_waterfall,
                                       self.d, bl=(38, 68), pol='xx', 
                                       title='x', mode='data', operator='xxx')
        
        # Check to make sure passing filenames instead of UVData works
        wfall, hfreq, htime, data = hs.plot.long_waterfall(
                                     self.dfile_paths, bl=(38, 68), pol='xx', 
                                     title='x', mode='data', file_type='miriad')
        wfall, hfreq, htime, data = hs.plot.long_waterfall(
                                     self.uvh5_files, bl=(23, 24), pol='xx', 
                                     title='x', mode='data', file_type='uvh5')
        
        # Check for input kwarg errors
        # Incorrect input data/filename type (must be list of UVData or str)
        nt.assert_raises(TypeError, hs.plot.long_waterfall,
                                       [1, 2], bl=(38, 68), pol='xx', 
                                       title='x', mode='data')
        # Integer bls not allowed if partial loading being used
        nt.assert_raises(TypeError, hs.plot.long_waterfall,
                                       self.uvh5_files, bl=123124, pol='xx', 
                                       title='x', mode='data', file_type='uvh5')
                                       
if __name__ == "__main__":
    unittest.main()

