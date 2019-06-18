import hera_stats as hs
import nose.tools as nt
import os
from hera_stats.data import DATA_PATH
from nbconvert.preprocessors import CellExecutionError
import unittest


class test_automate():

    def setUp(self):
        self.template = os.path.join(DATA_PATH, "Example_template_notebook.ipynb")
        self.outfile2 = os.path.join(DATA_PATH, "Example_template_notebook2.ipynb")
        self.outfile3 = os.path.join(DATA_PATH, "Example_template_notebook3.ipynb")
    
    def tearDown(self):
        if os.path.exists(self.outfile2):
            os.remove(self.outfile2)
        if os.path.exists(self.outfile3):
            os.remove(self.outfile3)
    
    def test_jupyter_replace_tags(self):
        
        # Check basic functionality works (with output file)
        nb = hs.automate.jupyter_replace_tags(
                         self.template,  
                         replace={'datafile': 'zen.odd.xx.LST.1.28828.uvOCRSA'},
                         outfile=self.outfile2,
                         overwrite=True,
                         verbose=True )
        nt.assert_equals(nb, None)
        
        # Check basic functionality (no output file, so returns dict)
        nb = hs.automate.jupyter_replace_tags(
                         self.template,  
                         replace={'datafile': 'zen.odd.xx.LST.1.28828.uvOCRSA'},
                         )
        nt.assert_equals(len(nb.keys()), 4)
        
        # Check that overwrite=False works
        nt.assert_raises(OSError, 
                         hs.automate.jupyter_replace_tags,
                         self.template, 
                         replace={'datafile': 'zen.odd.xx.LST.1.28828.uvOCRSA'},
                         outfile=self.outfile2,
                         overwrite=False,
                         verbose=True)
        
        # Check that overwrite=True works
        hs.automate.jupyter_replace_tags(
                         self.template, 
                         replace={'datafile': 'zen.odd.xx.LST.1.28828.uvOCRSA'},
                         outfile=self.outfile2,
                         overwrite=True,
                         verbose=True)

    def test_jupyter_run_notebook(self):
        
        # Check that basic functionality works
        hs.automate.jupyter_run_notebook(fname=self.template, 
                                         outfile=self.outfile3, 
                                         rundir=DATA_PATH)
        
        # Check that example notebook fails and that error is caught
        nb = hs.automate.jupyter_replace_tags(
                         self.template,  
                         replace={'datafile': 'file_doesnt_exist.txt'},
                         )
        nt.assert_raises(CellExecutionError, hs.automate.jupyter_run_notebook, 
                         tree=nb, outfile=self.outfile3, rundir='/tmp')

if __name__ == "__main__":
    unittest.main()

