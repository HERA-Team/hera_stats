# Extra utilities for hera_stats

import ephem

def timestamp():
        
        t = ephem.now()
        timestamp = str(t.datetime()).replace(" ",".").replace(":","_")[:19]
        
        return timestamp
    
    
class jkfdata():
    
    """
    Class for storing jackknife data.
    """
    
    def __init__(self):
        self.loaded = False
    
    def load(self, dic):
        """
        Loads the data using a dictionary.
        
        Parameters
        ----------
        dic: dictionary
            Dictionary containing the data from the jackknife file
        """
        
        self.spectra = dic["spectra"]
        self.dlys = dic["dlys"]
        self.errs = dic["errs"]
        self.grps = dic["grps"]
        self.jkntype = dic["jkntype"]
        self.spw_ranges = dic["spw_ranges"]
        self.times = dic["times"]
        self.n_jacks = len(self.spectra)
        self.dic = dic
        
        self.loaded = True
        
    def __repr__(self):
        
        string = "<hera_stats.utils.jkndata instance at %s>" %hex(id(self))
                  
        if self.loaded == True:
            string +=  ("\n\nJackknife Type: \t\t%s \n" %self.jkntype +
                        "Number of Jackknife Runs: \t%i \n" % self.n_jacks +
                        "Delay Range (ns): \t\t[%i, %i] \n" %(min(self.dlys),max(self.dlys)) + 
                        "Delay Modes: \t\t\t%i \n" %len(self.dlys) +
                        "Number of Times: \t\t%i \n" %len(self.times) + 
                        "Spectral Width Ranges: \t\t" + str(self.spw_ranges))
        else:
            string +=  "\n\nNo Jackknife data loaded. Load through class functions."
            
        return string