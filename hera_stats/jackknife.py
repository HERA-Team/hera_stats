# Jackknives for hera data
# Duncan Rocha, June 2018

import numpy as np
import hera_pspec as hp
from pyuvdata import UVData
from hera_cal import redcal
import copy
import os
import pickle

from utils import timestamp
import time

class jackknives():
    
    """
    Class for writing and using jackknives. Class variable is uvp, which each jackknife
    should use and output a pair of uvps, one for each jacknife. Jackknives should
    also return a variable that defines the 
    """

    def __init__(self):
        self.uvp = None
        
    def split_ants(self):

            """
            Splits available antenna into two groups randomly, and returns the UVPspec 
            of each.

            """
            uvp = copy.deepcopy(self.uvp)

            # Load all baselines in uvp
            blns = [uvp.bl_to_antnums(bl) for bl in uvp.bl_array]
            ants = np.unique(blns)

            c = 0
            minlen = 0
            while minlen <= len(blns)//6 or minlen <= 3:
                # Split antenna into two equal groups
                grps = np.random.choice(ants,len(ants)//2*2,replace=False).reshape(2,-1)

                # Find baselines for which both antenna are in a group
                blg = [[],[]]
                for bl in blns:
                    if bl[0] in grps[0] and bl[1] in grps[0]:
                        blg[0] += [bl]
                    elif bl[0] in grps[1] and bl[1] in grps[1]:
                        blg[1] += [bl]

                # Find minimum baseline length
                minlen = min([len(b) for b in blg])

                # If fails to find sufficiently large group enough times, raise excption
                c += 1
                if c == 50:
                    raise Exception("Not enough antenna provided")

            blgroups = []
            for b in blg:
                inds = np.random.choice(range(len(b)),minlen,replace=False)
                blgroups += [[uvp.antnums_to_bl(b[i]) for i in inds]]

            # Split uvp by groups
            uvpl = [uvp.select(bls=b,inplace=False) for b in blgroups]

            self.n_pairs = minlen

            return uvpl,[list(g) for g in grps]


class jackknife():
    
    def __init__(self):
        self.jackknives = jackknives()
        self.__labels = { self.jackknives.split_ants: "spl_ants" }
        self.data_direc = "/lustre/aoc/projects/hera/H1C_IDR2/IDR2_1/"
        pass
    
    def load_uvd(self, filepath, verbose=False):
        """Loads a UVD file to be used for jackknifing. This needs to be done in order for the
        rest of the code to work.
        
        Parameters
        ----------
        filepath: string or list of strings
            If string, loads the UVD miriad file from the specified path. If list, loads all miriad files
            specified.
        
        verbose: boolean, optional
            To print current actions and stopwatch. Default: False.
        """
        t = time.time()
        
        if type(filepath) != list:
            filepath = [filepath]
        
        if verbose:
            print "Loading %i file/s" %len(filepath)
            
        self.__files = filepath
        self.uvd = None
        
        self.uvd = UVData()
        self.uvd.read_miriad(filepath)
        
        tl = time.time()-t
        
        if verbose:
            print "UVD load time: %i min, %.1f sec" %(tl//60,tl%60)
              
        
    def jackknife(self, method, n_jacks, spw_ranges, beampath, baseline=None, pols=("XX","XX"),
                   taper='blackman-harris',
                   use_ants=None,savename=None,n_boots=100,imag=False,bootstrap_times=True,
                   returned=False,
                   calc_avspec=False,verbose=False):
        
        """
        Splits available antenna into two groups using a specified method from hera_stats.jackknife.jackknives,
        runs power spectrum analysis on both groups.
        
        Parameters
        ----------
        method: hera_stats.jackknives method
            The function to use to jackknife. Options: hera_stats.jackknives.split_ants
            
        n_jacks: int
            The amount of times to run the jackknife
            
        spw_ranges: list of tuples
            Spectral windows used to calculate power spectrum.
            
        beampath: str, optional
            The filepath for a beamfits file to use. If None, no beam is used. Default: None.
        
        baseline: tuple, optional
            Any antenna pair that represents the preferred baseline. If None, will select baseline that
            has the most redundant pairs in the data. Default: None.
            
        pols: tuple, optional
            Polarization to use for calculating power spectrum. Default: ("XX","XX").
            
        taper: str, optional
            The taper to pass to the pspec calculation. Default: 'blackman-harris'.
            
        use_ants: list, optional
            List of antenna to use in the calculation. If None, all available antenna are used.
            Default: None.
        
        savename: string, optional
            Optional string that will be placed at the beginning of the .jkf file. Default: None.
        
        n_boots: int, optional
            Number of bootstraps to do to estimate errors. Default: 100.
            
        imag: boolean, optional
            Whether to return the imaginary component of the power spectrum instead of the real. 
            Default: False
            
        calc_avspec: boolean, optional
            If true, also calculates the power spectrum of all data combined (i.e. before jackknife)
            Default: False
            
        returned: boolean, optional
            If true, returns a dictionary with the jackknife data in stead of saving to a file
        
        verbose: boolean, optional
            If true, prints current actions and stopwatch.
        
        
        Returns:
        -------
        dic: dictionary
            Dictionary that is returned if input variable 'returned' is True.

        """
        
        if method == "split_ants" or method == self.jackknives.split_ants:
            function = self.jackknives.split_ants
        else:
            raise NameError("Function not found. Specify by string or jackknife class method")
        
        t = time.time()
        ts = t
        
        # Calculate UVPspecData 
        self.calc_uvp(spw_ranges, baseline=baseline,pols=pols,
                                    beampath=beampath,taper=taper,use_ants=use_ants)
        
        tcalc = time.time() - t
        t = time.time()
            
        spectra,errs,grps,n_pairs = [],[],[],[]
        for i in range(n_jacks):
            
            if verbose:
                print "Jackknife run %i." %i
                
            do_av = (i == n_jacks-1)
            # Run the antenna splitting algorithm
            uvpl, grp = function()
            
            # Calculate delay spectrum and bootsrap errors
            dlys,specs,err = self.bootstrap_errs(uvpl,n_boots=n_boots,imag=imag,bootstrap_times=bootstrap_times)
            
            # Store the information
            spectra += [specs]
            errs += [err]
            grps += [grp]
            n_pairs += [self.jackknives.n_pairs]
            
        tboot = time.time() - t
        t = time.time()
        
        # Save information
        dic = {"dlys": dlys, "spectra": spectra, "errs": errs, "grps":grps,"files":self.__files,
               "times":np.unique(self.uvd.time_array),"n_pairs":n_pairs,
               "spw_ranges":spw_ranges,"taper":taper,"jkntype":self.__labels[function]}
        
        if calc_avspec == True:
            # Calculate the average spectrum
            avspec, averrs = self.bootstrap_errs(self.uvp,n_boots=n_boots,imag=imag)[1:]
            
            dic["avspec"] = avspec,
            dic["averrs"] = averrs
        
        tav = time.time() - t
        
        ttot = time.time() - ts
        
        secmin = lambda x: (x//60,x%60)
        if verbose:
            print ("Time taken:" + 
                    "\nPspec-ing: %i min, %.1f sec" % secmin(tcalc) +
                    "\nBootstrapping: %i min, %.1f sec" % secmin(tboot) +
                    "\nAv. Boots: %i min, %.1f sec" % secmin(tav) +
                    "\nTotal: %i min, %.1f sec" % secmin(ttot))

        if os.path.exists("./data") == False:
            os.mkdir("./data")

        # Make filename
        outname = self.__labels[function] + ".Nj%i." %n_jacks + timestamp() + ".jkf"
        if savename != None:
            outname = savename + "." + outname

        if returned:
            return dic

        # Write
        with open("./data/" + outname, "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
            if verbose: print "Saving to: '%s'" %outname

                
                
    def calc_uvp(self, spw_ranges, baseline=None,pols=("XX","XX"),
                                    beampath=None,taper='blackman-harris',use_ants=None):
        
        """Calculates UVPspecData object using UVData loaded in load_uvd()
        
        Parameters:
        ----------
        
        spw_ranges: list of tuples
            Spectral windows used to calculate power spectrum.
            
        beampath: str, optional
            The filepath for a beamfits file to use. If None, no beam is used. Default: None.
        
        baseline: tuple, optional
            Any antenna pair that represents the preferred baseline. If None, will select baseline that
            has the most redundant pairs in the data. Default: None.
            
        pols: tuple, optional
            Polarization to use for calculating power spectrum. Default: ("XX","XX").
            
        taper: str, optional
            The taper to pass to the pspec calculation. Default: 'blackman-harris'.
            
        use_ants: list, optional
            List of antenna to use in the calculation. If None, all available antenna are used.
            Default: None.
        
        """
        ants = self.uvd.get_ants()
        if sum([b not in ants for b in baseline]) > 0 :
            raise ValueError("Baseline not found in UVData file.")
        
        # If use_ants is not a list of objects, only use those specified
        if type(use_ants) != type(None):
            
            if sum([ua not in ants for ua in use_ants]) > 0:
                raise ValueError("Antenna specified not found in data.")
                
            if sum([b not in use_ants for b in baseline]) > 0:
                raise ValueError("Baseline not in list use_ants.")
            
            # Find the antenna that aren't being used
            notused = ~np.array([a in use_ants for a in ants])
            
            # Show user which are omitted
            print ("Selecting antenna numbers: " + str(sorted(use_ants)) + 
                   "\nThe following antenna have data and" +
                  "will be omitted: " + str(sorted(ants[notused])))
            
            # Modify UVData object to exclude those antenna
            self.uvd.select(antenna_nums=use_ants)
            ants = use_ants
                    
        # If there is a beampath, load up the beamfits file
        if beampath != None:
            cosmo = hp.utils.Cosmo_Conversions()
            beam = hp.pspecbeam.PSpecBeamUV(beampath,cosmo=cosmo)
        
        # If no specified baseline, use the one with the most redundant antenna pairs
        pairs = self.isolate_baseline(self.uvd,baseline)
        
        # Fill PSpecData object
        ds = hp.PSpecData([self.uvd,self.uvd],[None,None],beam=beam)

        # Calculate baseline-pairs using the pairs chosen earlier
        bls1,bls2,blpairs = hp.pspecdata.construct_blpairs(pairs, exclude_auto_bls=True, 
                                                         exclude_permutations=True)

        # Calculate power spectrum
        uvp = ds.pspec(bls1,bls2,dsets=(0,1),pols=pols,spw_ranges=spw_ranges, 
                       input_data_weight='identity',norm='I',taper=taper,
                       verbose=False)
        
        self.uvp = uvp
        self.jackknives.uvp = uvp
        
            
    def find_files(self,direc,endstring):
        
        st = endstring
        
        allf = os.listdir(direc)
        files = []
        for f in allf:
            if f[-len(st):] == st:
                files += [f]

        files = sorted(files)
        
        return [direc + "/" + f for f in files]

    
    def bootstrap_errs_once(self, uvp, pol="xx", n_boots=100,return_all=False,imag=False,
                            bootstrap_times=True):
        
        """
        Uses the bootstrap method to estimate the errors for a PSpecData object. 
        
        Parameters:
        ----------
        uvp: hera_pspec.UVPspecData 
            Object outputted by pspec, contains power spectrum information.
            
        pol: str, optional
            The polarization used in pspec calculations. Default: "xx".
        
        n_boots: int, optional
            How many bootstraps of the data to to to estimate error bars. Default: 100.
            
        return_all: boolean
            Test parameter, returns every bootstrapped spectrum.
        
        version: str
            Test parameter, which version to use.
        
        
        Returns:
        -------
        dlys: list
            Delay modes, x-axis of a power spectrum graph.
        
        avspec: list
            The delay spectrum values at each delay.

        errs: list
            Estimated errors for the delays spectrum at each delay.
            
        """
        
        if imag:
            proj = lambda l: np.abs(l.imag)
        else:
            proj = lambda l: np.abs(l.real)
            
        #proj = lambda l: l.real
        
        # Calculate unique baseline pairs
        pairs = np.unique(uvp.blpair_array)
        blpairs = (uvp.blpair_to_antnums(pair) for pair in pairs)
        blpairs = sorted(blpairs)

        # Calculate the average of all spectra
        avg = uvp.average_spectra([blpairs,],time_avg=True,inplace=False)

        # Create key
        antpair = avg.blpair_to_antnums(avg.blpair_array)
        key = (0,antpair,"xx")

        # Get average spectrum and delay data
        avspec = proj(avg.get_data(key))[0]
        dlys = avg.get_dlys(0)*10**9

        times = np.unique(uvp.time_avg_array)

        allspecs = []
        for i in range(len(blpairs)):
            key=(0,blpairs[i],"xx")

            if bootstrap_times:
                allspecs += list(uvp.get_data(key))
            else:
                allspecs += [np.average(uvp.get_data(key),axis=0)]
                
        lboots = []
        for i in range(n_boots):
            inds = np.random.choice(range(len(allspecs)),len(allspecs),replace=True)
            boot = [allspecs[i] for i in inds]

            bootav = np.average(np.vstack(boot),axis=0)

            lboots += [bootav]

        lboots = np.array(lboots)
        err = np.std(proj(lboots),axis=0)

        # Return everything
        return dlys, avspec, err
    
    def bootstrap_errs(self,uvpl,pol="xx",n_boots=100,imag=False,
                       bootstrap_times=True):
        """
        Calculates the delay spectrum and error bars for every UVPspecData object in a list.
        
        Parameters:
        ----------
        uvpl: list
            List of UVPspecData objects, 
            
        pol: str, optional
            Polarization used to calculate power spectrum. Default: "xx".
        
        n_boots: int, optional
            Number of times to bootstrap error bars. Default: 100.
        
        imag: boolean, optional
            If true, returns the imaginary component of the spectra instead of the real
        
        Returns:
        -------
        dlys: list
            The delays of the power spectra.
            
        avspec: list of lists
            A list of all of the calculated power spectra
        
        errs: list of lists
            A list of all of the calculated errors.
        """
        
        # Turn into list if it is not one
        if type(uvpl) == hp.uvpspec.UVPSpec:
            uvpl = [uvpl]
            
        avspec = []
        errs = []
        
        for uvp in uvpl:
            # Bootstrap errors using other function
            _uvp = copy.deepcopy(uvp)
            dlys,av,er = self.bootstrap_errs_once(_uvp,pol=pol,n_boots=n_boots,imag=imag,
                                                  bootstrap_times=bootstrap_times)
            
            # Add results to list
            avspec += [av]
            errs += [er]
        
        return dlys,avspec,errs
    
    def isolate_baseline(self,uvd,bsl=None):
        
        """
        Gives a list of the available antenna pairs that match the specified baseline.
        if bsl is None, returns list containing highest number of available redundant baselines.
        
        Parameters
        ----------
        uvd: pyuvdata.UVData object
            UVData object containing information necessary.
        
        baseline: tuple
            An example antenna pair of the baseline requested. If None, chooses baseline corresponding 
            to highest number of redundant pairs.
        
        """
        
        # Get antenna with data and positions
        [pos, num] = uvd.get_ENU_antpos(pick_data_ants=True)
        dic = dict(zip(num,pos))
        
        # List the redundant pairs
        redpairs = redcal.get_pos_reds(dic)
            
        if bsl == None:
            # Number of antenna pairs in each redundant group
            numpairs = np.array([len(p) for p in redpairs])
            
            # Find longest rendent antenna pair group. Chooses the smallest baseline if more than one are
            # same length and longest.
            ind = np.where(numpairs == max(numpairs))[0][0]
            
            # Return the group of pairs
            return redpairs[ind]

        # Get all antennas, including those without data
        [pos_nd, num_nd] = uvd.get_ENU_antpos()

        # Make sure both antenna are actually in the list of available antenna and positions
        if sum([bsl[i] in num_nd for i in [0,1]]) != 2:
            raise Exception("Antenna pair not found. Available antennas are: " +
                            str(sorted(num_nd.tolist())))
        
        dic_nd = dict(zip(num_nd,pos_nd))
        
        # All redundant pairs 
        redpairs_nd = redcal.get_pos_reds(dic_nd)
        
        # Reverse the order of the antenna pair and check that too
        bsl_bw = (bsl[1],bsl[0])
            
        # Check each redundant antenna pair group to see if it contains baseline
        matches = [bsl in rp for rp in redpairs_nd]
        matches_bw = [bsl_bw in rp for rp in redpairs_nd]

        # Find where the baseline is contained
        where = np.where(np.array(matches) + np.array(matches_bw) > 0)[0][0]

        # Group of valid baselines is whichever group the baseline is contained within.
        val = redpairs_nd[where]

        # Next look through the redundant pairs of antenna with data
        for pairs in redpairs:
            if pairs[0] in val:
                # If the first antenna pair of the group is in the list of valid antenna pairs, return.
                return pairs
            
        return []
    
    
    
    
        
    
    
    