import hera_stats as hs
import os
from hera_stats.data import DATA_PATH
import nose.tools as nt
import pickle 

class test_utils():

    def test_functions(self):

        hs.utils.plt_layout(10)

        degs = [d - 50 for d in range(100)]
        hs.utils.bin_wrap(degs, 10)
        hs.utils.bin_wrap(degs, 200)

        nt.assert_true(hs.utils.is_in_wrap(350, 10, 0.2))
        nt.assert_false(hs.utils.is_in_wrap(350, 10, 11))