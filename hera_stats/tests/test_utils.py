import hera_stats as hs
import os
from hera_stats.data import DATA_PATH
import nose.tools as nt
import pickle 

class test_utils():

    def test_functions(self):

        hs.utils.find_files(DATA_PATH, ".py")
        hs.utils.unique_items([[[0,1],[2,3]]])
        hs.utils.plt_layout(10)
        hs.utils.timestamp()