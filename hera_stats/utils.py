# Extra utilities for hera_stats

import ephem
import os

class jkfdata():
    def __init__(self):
        """
        Class for storing jackknife data.
        """
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
        self.jackpairs = len(self.spectra[0])
        self.dic = dic
        self.loaded = True

    def validate(self):
        """
        Checks if data is valid.
        """
        if not self.loaded:
            raise AttributeError("No data loaded. Run load function")

    def __repr__(self):

        string = "<hera_stats.utils.jkfdata instance at %s>" %hex(id(self))

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

def find_files(direc,endstring):
    """
    Given a directory path and a search string, finds files. Useful for looking for hera miriad files
    with the right suffixes.

    Parameters:
    ----------
    direc: string
        Directory in which to search for the files.

    endstring: string:
        Search parameter, the ending of the files requested.
    """
    if direc[-1] != "/":
        direc += "/"

    st = endstring

    allf = os.listdir(direc)
    files = []
    for f in allf:
        if f[-len(st):] == st:
            files += [f]

    files = sorted(files)

    return [direc + f for f in files]

def shorten(files):
    """
    Removes the path from the files, returns only filenames.

    Parameters
    ----------
    files: list
        List of filepaths to shorten.

    Returns
    -------
    shortened: list
        List of shortened filepath strings
    """
    sh = []
    for f in files:
        if f[-1] != "/":
            f += "/"
        sh += [f.split("/")[-2]]
    return sh

def unique_items(grps):
    """
    Finds all unique items in a jackknife group list. Use this as opposed to np.unique because
    np.unique cannot do baseline pairs

    Parameters
    ----------
    grps: list
        Groups directly from jaccknife data.

    Returns
    -------
    items: list
        Unique items found in grps.
    """
    unique = []
    for glist in grps:
        for g in glist:
            for item in g:
                if item not in unique:
                    unique += [item]
    return sorted(unique)

def timestamp():
    """
    Returns a timestamp string.
    """
    t = ephem.now()
    timestamp = str(t.datetime()).replace(" ",".").replace(":","_")[:19]
    return timestamp