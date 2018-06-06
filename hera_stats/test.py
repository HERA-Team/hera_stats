# Jackknives for hera data
# Duncan Rocha, June 2018

import numpy as np
import hera_pspec as hp
from pyuvdata import UVData
from hera_cal import redcal
import copy
import os
from utils import timestamp
import pickle

import time


class jackknife():
    def __init__(self):
        self.__labels = { self.split_ants: "spl_ants" }
        self.data_direc = "/lustre/aoc/projects/hera/H1C_IDR2/IDR2_1/"
        pass
    
    def load_uvd(self, filepath):
        
        if type(filepath) != list:
            filepath = [filepath]
            
        self.__files = filepath
        self.uvd = None
        
        self.uvd = UVData()
        self.uvd.read_miriad(filepath)    
        
        
    def jackknife(self,method, n_jacks, spw_ranges, beampath, baseline=None, pols=("XX","XX"),
                   taper='blackman-harris', min_pairs=3,
                   use_ants=None,savename=None,n_boots=100,imag=False,returned=False):
        
        if method == "split_ants" or method == self.split_ants:
            function = self.split_ants
        else:
            raise NameError("Function not found. Specify by string or jackknife class method")
        
        t = time.time()
        ts = t
        tjack = 0
        tboot = 0
        
        spectra,errs,grps = [],[],[]
        for i in range(n_jacks):
            
            do_av = (i == n_jacks-1)
            # Run the antenna splitting algorithm
            uvpl, grp, uvp_avg = function(spw_ranges=spw_ranges,baseline=baseline,min_pairs=3, 
                                    beampath=beampath,return_comb=do_av,taper=taper,use_ants=use_ants)
            
            tjack += time.time() - t
            t = time.time()
            
            # Calculate delay spectrum and bootsrap errors
            dlys,specs,err = self.bootstrap_errs(uvpl,n_boots=n_boots,imag=imag)

            tboot += time.time() - t
            t = time.time()
            
            # Store the information
            spectra += [specs]
            errs += [err]
            grps += [grp]
            
       
        # Calculate the average spectrum
        avspec, averrs = self.bootstrap_errs(uvp_avg,n_boots=n_boots,imag=imag)[1:]
        
        tav = time.time() - t
        
        print "Time taken: \nJackknifing: %.1f \nBootstrapping: %.1f \nAv. Boots: %.1f \nTotal: %.1f" % (tjack,tboot,tav,time.time()-ts)
        
        
        dic = {"dlys": dlys, "spectra": spectra, "errs": errs, "avspec": avspec,
           "averrs": averrs, "grps":grps,"files":self.__files,"times":np.unique(self.uvd.time_array),
           "spw_ranges":spw_ranges,"taper":taper,"jkntype": self.__labels[function]}
        
        if os.path.exists("./data") == False:
            os.mkdir("./data")
            
        outname = self.__labels[function] + ".Nj%i" %n_jacks + timestamp() + ".jkf"
        
        if savename != None:
            outname = savename + "." + outname
        
        if returned:
            return dic
        else:
            with open("./data/" + outname, "wb") as f:
                pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
        
    def find_files(self,direc,endstring):
        
        st = endstring
        
        allf = os.listdir(direc)
        files = []
        for f in allf:
            if f[-len(st):] == st:
                files += [f]

        files = sorted(files)
        
        return [direc + "/" + f for f in files]

    
    def split_ants(self, spw_ranges, beampath=None, baseline=None, pols=("XX","XX"),
                   taper='blackman-harris', min_pairs=3,
                   use_ants=None, return_comb=False):
        
        """
        Splits available antenna into two groups, runs power spectrum analysis on both groups.
        
        Parameters
        ----------
        uvd: pyuvdata.UVData object 
            The UVData object with which to jackknife, loaded with miriad file(s).
            
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
            
        min_pairs: int, optional
            The mininum number of pairs to use to calculate power spectrum. If groups are selected with
            fewer redundant pairs than this, the groups are redrawn. Default: 3.
            
        use_ants: list, optional
            List of antenna to use in the calculation. If None, all available antenna are used.
            Default: None.
        
        return_comb: boolean, optional
            If true, returns the combined power spectrum as a third returned item. Default: False.
            
        
        Returns:
        -------
        uvps: list of 2 UVPspecData
            The two UVPspecs created by the jackknife.
                        
        grps: list
            A list containing the two antenna groups used.
            
        uvp_comb: UVPspecData
            UVPspecData calculated using all the antenna.

        
        """
        
        uvd = copy.deepcopy(self.uvd)
        ants = uvd.get_ants()
        
        # If use_ants is not a list of objects, only use those specified
        if type(use_ants) != type(None):
            
            # Find the antenna that aren't being used
            notused = ~np.array([a in use_ants for a in ants])
            
            # Show user which are omitted
            print ("Selecting antenna numbers: " + str(sorted(use_ants)) + 
                   "\nThe following antenna have data and" +
                  "will be omitted: " + str(sorted(ants[notused])))
            
            # Modify UVData object to exclude those antenna
            uvd.select(antenna_nums=use_ants)
            ants = use_ants
                    
        # If there is a beampath, load up the beamfits file
        if beampath != None:
            cosmo = hp.utils.Cosmo_Conversions()
            beam = hp.pspecbeam.PSpecBeamUV(beampath,cosmo=cosmo)
        
        # If no specified baseline, use the one with the most redundant antenna pairs
        if baseline==None:
            baseline = self.isolate_baseline(uvd)[0]
        
        minlen = 0
        count = 0
        while minlen < min_pairs:
            # Randomly separate antenna into two equally sized groups
            rand_draw = np.random.choice(ants,len(ants)//2*2,replace=False)
            grps = rand_draw.reshape(2,-1)

            # Find redundant baselines within each antenna group
            uvdl = [uvd.select(antenna_nums=g,inplace=False) for g in grps]
            prs = [self.isolate_baseline(u,baseline) for u in uvdl]

            # Find whichever group has fewer redundant antenna pairs
            minlen = min([len(p) for p in prs])

            # Shorten the longer group to the same size of the shorter
            pairs = []
            for p in prs:
                # For each group, draw (minlen) indices randomly and add those to a final list
                inds = np.random.choice(range(len(p)),minlen,replace=False)
                pairs += [[p[i] for i in inds]]

            # If this process fails too many times, there are too few antenna
            if count > 50:
                raise Exception("Too few pairs. Add data that has more antennae.")
            count += 1

        
        # For each pair, calculate the UVPspec object. The return a list of them.
        uvps = []
        for i in [0,1]:
            
            # Load uvd and redundant pairs to use
            _uvd = uvdl[i]
            p = pairs[i]
            
            # Fill PSpecData object
            ds = hp.PSpecData([_uvd,_uvd],[None,None],beam=beam)
            
            # Calculate baseline-pairs using the pairs chosen earlier
            bls1,bls2,blpairs = hp.pspecdata.construct_blpairs(p, exclude_auto_bls=True, 
                                                             exclude_permutations=True)
            
            # Calculate power spectrum
            uvp = ds.pspec(bls1,bls2,dsets=(0,1),pols=pols,spw_ranges=spw_ranges, 
                           input_data_weight='identity',norm='I',taper=taper,
                           verbose=False)
            
            # Save pspec to a list
            uvps += [uvp]
            
        # If combined power spectrum is requested, calculate it too
        if return_comb:
            # Find all redundent pairs in original UVData object
            allpairs = self.isolate_baseline(uvd,baseline)
            
            # Load PSpecData
            ds = hp.PSpecData([uvd,uvd],[None,None],beam=beam)
            
            # Create baseline pairs
            bls1,bls2,blpairs = hp.pspecdata.construct_blpairs(allpairs, 
                                                               exclude_auto_bls=True, 
                                                               exclude_permutations=True)
            # Calculate combined UVPspec
            uvp_comb = ds.pspec(bls1,bls2,dsets=(0,1),pols=pols,spw_ranges=spw_ranges, 
                               input_data_weight='identity',norm='I',taper=taper, 
                               verbose=False)
            
        else:
            uvp_comb = None
        
        # Otherwise, just return the list of UVPSpecs and the groups used
        return uvps, grps, uvp_comb
    
    def bootstrap_errs_once(self, uvp, pol="xx", n_boots=100,return_all=False,imag=False):
        
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
                
        lboots = []
        for i in range(n_boots):
            # Calculate a bootstraped avergae
            boots, _wgts = hp.grouping.bootstrap_average_blpairs([uvp,],
                                                                 [blpairs,],
                                                                 time_avg=False)
            boot = boots[0]
            
            # Add this piece of data to the set
            key = (0,boot.blpair_array[0],pol)
            spectra = proj(boot.get_data(key))
            
            ntimes = len(boot.time_avg_array)
            inds = np.random.choice(range(ntimes),ntimes,replace=True)
            specs = np.vstack([spectra[i] for i in inds])
            
            avspec = np.average(specs,axis=0)
            lboots += [avspec]
                
        if return_all:
            return lboots
        
        # Calculate the standard deviation of each mean
        lboots = np.array(lboots)
        err = np.std(lboots, axis=0,ddof=1)

        # Return everything
        return dlys, avspec, err
    
    def bootstrap_errs(self,uvpl,pol="xx",n_boots=100,returned=False,imag=False):
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
            dlys,av,er = self.bootstrap_errs_once(_uvp,pol=pol,n_boots=n_boots,imag=imag)
            
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
                            str(num_nd.tolist()))
        
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
    
    
    
    
        
    
    
    