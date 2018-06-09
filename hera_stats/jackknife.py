# Jackknives for hera data
# Duncan Rocha, June 2018

import numpy as np
import hera_pspec as hp
from pyuvdata import UVData
from hera_cal import redcal
import copy
import pickle
import os

import utils
import time

class jackknives():
    """
    Class for writing and using jackknives. Class variable is uvp, which each jackknife
    should use and output a pair of uvps, one for each jacknife. Jackknives should
    also return a variable that defines the 
    """
    def split_ants(self,n_jacks=40):
        """
        Splits available antenna into two groups randomly, and returns the UVPspec 
        of each.

        Parameters
        ----------
        n_jacks: int
            Number of times to jackknife the data.

        Returns
        -------
        uvpl: list
            List of hera_pspec.UVPSpecData objects that have been split accordingly.

        grps: list
            A list of the antenna in each group.

        n_pairs: list
            The number of baseline pairs in each UVPspecData.
        """
        if type(self.uvp) == list:
            if len(self.uvp) != 1:
                raise AttributeError("Split_ants can't jackknife multiple files at a time. Only takes" +
                                     "len-1 lists or single UVPSpecData objects. Redo 'load_uvd' and set"+
                                     "combine=False")
            else:
                uvp = self.uvp[0]

        # Load all baselines in uvp
        blns = [uvp.bl_to_antnums(bl) for bl in uvp.bl_array]
        ants = np.unique(blns)

        n_pairs = []
        groups = []
        uvpls = []
        for i in range(n_jacks):

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
            n_pairs += [minlen]
            
            groups += [[list(g) for g in grps]]
            uvpls += [uvpl]

        return uvpls,groups,n_pairs
    
    def split_files(self,identifier=None,pairs=None,all_permutations=True):
        """
        Splits the files into two groups, using one of two methods. One of these must be provided. 
        Providing both will error out.
        
        If an identifier is specified, the jackknife finds the files that have the identifier
        and compares them to those which don't. It automatically does every combination but
        only up to a maximum of n_jacks (not yet tho).
        
        If pairs are specified, which is a list of len-2 lists, the pairs are selected using two indices
        provided. This is essentially a list of pairs of indices to use for selecting files.
        
        Parameters
        ----------
        identifier: string
            String segment to use as a filter for grouping files.

        pairs: list
            List of index pairs, for a more manual selection of file groupings.

        Returns
        -------
        uvpl: list
            List of UVPSpecData objects split accordingly.

        grps: list
            Groups used. Each contains two filepaths.

        n_pairs: list
            Number of baseline pairs used in each jackknife.
        """
        # Sanity checks
        if type(self.uvp) != list:
            raise AttributeError("Split files needs a list of uvp objects.")
        if len(self.uvp) < 2:
            raise AttributeError("Fewer than two files supplied. Make sure combine = False for load_uvd.")
        if identifier != None and pairs != None:
            raise AttributeError("Please only specify either identifier or file pairs.")
        if identifier == None and pairs == None:
            raise AttributeError("No identifier or file pair list specified.")

        uvpl,grps = [],[]
        if type(identifier) == str:

            grp1,grp2,uvp1,uvp2 = [],[],[],[]
            for i,f in enumerate(self.files):
                # Look for identifier in each file, sort accordingly
                if identifier in f:
                    grp1 += [f]
                    uvp1 += [self.uvp[i]]
                else:
                    grp2 += [f]
                    uvp2 += [self.uvp[i]]

            if len(grp1) == 0:
                raise AttributeError("Identifier not found in any filenames loaded.")
            if len(grp2) == 0:
                raise AttributeError("Identifier found in all filenames... Be more strict!")

            # If all permutations are requested, do every possible file matchup.
            if all_permutations:
                for i in range(len(uvp1)):
                    for j in range(len(uvp2)):
                        uvpl += [[uvp1[i],uvp2[j]]]
                        grps += [[grp1[i],grp2[j]]]
            else:
                # Otherwise, do them in order of appearance
                uvpl = [[uvp1[i],uvp2[i]] for i in range(len(uvp1))]
                grps = [[grp1[i],grp2[i]] for i in range(len(grp1))]

        # Split according to index pairs provided
        elif type(pairs) == list:
            for idpair in identifier:
                [i,j] = idpair
                uvpl += [[self.uvp[i],self.uvp[j]]]
                grps += [[self.files[i],self.files[j]]]

        n_pairs = [len(self.uvp[0].bl_array)] * len(uvpl)
        return uvpl,grps,n_pairs

    def split_times(self,binsizes = None):
        """
        Jackknife that splits the UVPSpecData into groups based on the binsize. 

        The binsize is given in seconds, and data is split based on alternating bins, that is, if the time
        array is [1,2,3,4,5,6] and the binsize is 1, then the data will be split into [1,3,5] and [2,4,6].
        if binsize is 2, then data will be split into [1,2,5,6] and [3,4]. Finally, if binsize is 3, you get
        [1,2,3] and [4,5,6]. If no binsize is profided, it will run a jackknife for every valid and unique
        binsize.

        Parameters
        ----------
        binsizes: float or list, optional
            If float, jackknifes a single time using one binsize. If list, jackknives for every binsize
            provided.

        Returns
        -------
        uvpl: list
            List of UVPSpecData objects split accordingly.

        grps: list
            Groups used. Each contains two filepaths.

        n_pairs: list
            Number of baseline pairs used in each jackknife.
        """
        if type(self.uvp) == list:
            uvp = self.uvp[0]
        if type(binsizes) != list:
            binsizes = [binsizes]
        if len(self.uvp) != 1:
            raise AttributeError("Split_times can't jackknife multiple files at a time. Only takes" +
                                 "len-1 lists or single UVPSpecData objects. Redo 'load_uvd' and set "+
                                 "combine=False.")
        times = np.array(sorted(np.unique(uvp.time_avg_array)))
        secs = (times-times[0])*24*3600

        # If no binsize provided, use every binsize possible and unique
        if binsizes == [None]:
            minperiod = secs[1] - secs[0]
            binsizes = [x*minperiod for x in range(1,len(times)//2)]
        
        uvpl = []
        grps = []
        for bs in binsizes:
            # Phase randomly for added effect
            phase = np.random.uniform(0,secs[1])
            select = np.sin(np.pi*(secs + phase)/bs) >= 0

            # Split times into bins
            t1,t2 = times[select],times[~select]
            minlen = min([len(t1),len(t2)])
            t1 = t1[:minlen]
            t2 = t2[:minlen]

            # add to groups and uvpl
            grps += [[list(t1),list(t2)]]
            uvpl += [[uvp.select(times=t,inplace=False) for t in [t1,t2]]]

        n_pairs = [len(self.uvp[0].bl_array)] * len(uvpl)
        return uvpl,grps,None

    def no_jackknife(self):
        """
        No jackknife used at all. No splitting, just one uvp returned.

        Returns:
        -------
        uvpl: list
            List of a single hera_pspec.UVPSpecData.

        grps: list
            A list of the antenna used.

        n_pairs: list
            The number of baseline pairs used.
        """
        # Don't do anything
        blns = [self.uvp.bl_to_antnums(bl) for bl in self.uvp.bl_array]
        ants = np.unique(blns)
        grps = list(ants)
        n_pairs = len(blns)

        return [self.uvp],[grps],[n_pairs]






class jackknife():

    def __init__(self):
        """
        Class for jackkniving hera data.
        """
        self.jackknives = jackknives()
        self.__labels = { self.jackknives.split_ants: "spl_ants", self.jackknives.no_jackknife: "no_jkf",
                          self.jackknives.split_files: "spl_files",self.jackknives.split_times: "spl_times"}
        self.__loadtime = 0.
        self.__boottime = 0.
        self.__tottime = 0.
        self.__calctime = 0.
        pass

    def load_uvd(self, filepath, combine=True,verbose=False,use_ants=None):
        """
        Loads a UVD file to be used for jackknifing. This needs to be done in order for the
        rest of the code to work.

        Parameters
        ----------
        filepath: string or list
            Single filepath or list of filepaths to load.

        combine: boolean, optional
            If False, keeps UVData files separate, so jackknife split_files can be run. Default: True.

        use_ants: list or NoneType, optional
            If provided, uses the antenna listed. Can shorten calculation times. Default: None.

        verbose: boolean, optional
            To print current actions and stopwatch. Default: False.
        """
        t = time.time()

        if type(filepath) != list:
            filepath = [filepath]

        if verbose:
            print "Loading %i file/s" %len(filepath)

        self.files = filepath
        self.jackknives.files = filepath
        self.uvd = None

        if combine:
            # Calculate one UVData using every file provided
            self.uvd = UVData()
            self.uvd.read_miriad(filepath,antenna_nums=use_ants)
            self.uvd = [self.uvd]
        else:
            # Calculate uvData for each file provided
            self.uvd = []
            for fp in filepath:
                uvd = UVData()
                uvd.read_miriad(fp,antenna_nums=use_ants)
                self.uvd += [copy.deepcopy(uvd)]
                uvd = None

        self.__loadtime += time.time()-t

    def jackknife(self, jkf_func, spw_ranges, beampath, baseline=None, pols=("XX","XX"),
                  taper='blackman-harris',savename=None,n_boots=100,imag=False,
                  bootstrap_times=True,returned=False,verbose=False,**kwargs):
        """
        Splits available antenna into two groups using a specified method from hera_stats.jackknife.jackknives,
        runs power spectrum analysis on both groups.

        Parameters
        ----------
        jkf_func: hera_stats.jackknives method
            The function to use to jackknife. All options are stored in hera_stats.jackknife.jackknives.

        spw_ranges: list of tuples
            Spectral windows used to calculate power spectrum.

        beampath: str
            The filepath for a beamfits file to use. If None, no beam is used. Default: None.

        baseline: tuple, optional
            Any antenna pair that represents the preferred baseline. If None, will select baseline that
            has the most redundant pairs in the data. Default: None.

        pols: tuple, optional
            Polarization to use for calculating power spectrum. Default: ("XX","XX").

        taper: str, optional
            The taper to pass to the pspec calculation. Default: 'blackman-harris'.

        savename: string, optional
            Optional string that will be placed at the beginning of the .jkf file. Default: None.

        n_boots: int, optional
            Number of bootstraps to do to estimate errors. Default: 100.

        imag: boolean, optional
            Whether to return the imaginary component of the power spectrum instead of the real. 
            Default: False

        bootstrap_times: boolean, optional
            If true, bootstraps over times. If false, averages over them before bootstrapping.

        returned: boolean, optional
            If true, returns a dictionary with the jackknife data in stead of saving to a file

        verbose: boolean, optional
            If true, prints current actions and stopwatch.

        **kwargs: 
            All other arguments are passed directly on to the jackknife. Look at hera_pspec.jackknife.jackknives
            for more information.

        Returns:
        -------
        dic: dictionary
            Dictionary that is returned if input variable 'returned' is True.
        """
        if type(jkf_func) != type(self.jackknives.split_ants):
            raise TypeError("Jackknife not found.")

        tstart = time.time()

        # Calculate UVPspecData 
        self.calc_uvp(spw_ranges, baseline=baseline,pols=pols,
                                    beampath=beampath,taper=taper)

        # Run jackknife splitting
        uvpl,grps,n_pairs_l = jkf_func(**kwargs)
        n_jacks = len(uvpl)

        # Calculate delay spectrum and bootsrap errors
        dlys,spectra,errs = self.bootstrap_errs(uvpl,n_boots=n_boots,imag=imag,bootstrap_times=bootstrap_times)
        times = [u.time_array for u in self.uvd]

        self.__tottime += time.time() - tstart + self.__loadtime
        secmin = lambda x: (x//60,x%60)
        if verbose:
            # Print time statistics
            print ("Time taken:" + 
                    "\nLoad Time: %i min, %.1f sec" % secmin(self.__loadtime) +
                    "\nPspec-ing: %i min, %.1f sec" % secmin(self.__calctime) +
                    "\nBootstrapping: %i min, %.1f sec" % secmin(self.__boottime) +
                    "\nTotal: %i min, %.1f sec" % secmin(self.__tottime))
        # Make filename
        outname = self.__labels[jkf_func] + ".Nj%i." %n_jacks + utils.timestamp() + ".jkf"
        if savename != None:
            outname = savename + "." + outname

        dic = {"dlys": dlys, "spectra": spectra, "errs": errs, "grps":grps,"files":self.files,
               "times":times,"n_pairs":n_pairs_l,"spw_ranges":spw_ranges,
               "taper":taper,"jkntype":self.__labels[jkf_func]}

        if returned:
            return dic

        if os.path.exists("./data") == False:
            os.mkdir("./data")

        # Write to file
        with open("./data/" + outname, "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
            if verbose: print "Saving to: '%s'" %outname

    def calc_uvp(self, spw_ranges, baseline=None,pols=("XX","XX"),
                                    beampath=None,taper='blackman-harris'):
        """
        Calculates UVPspecData object using UVData loaded in load_uvd()

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
        """
        t = time.time()

        ants = np.unique(np.hstack([u.get_ants() for u in np.hstack(self.uvd)]))

        if baseline != None:
            hbl, nfi = self.hasants(baseline)
            if False in hbl:
                raise AttributeError("Baseline antennae not found in the following files: " + str(nfi))

        self.validate()

        # If there is a beampath, load up the beamfits file
        if beampath != None:
            cosmo = hp.utils.Cosmo_Conversions()
            beam = hp.pspecbeam.PSpecBeamUV(beampath,cosmo=cosmo)

        # If no specified baseline, use the one with the most redundant antenna pairs
        pairs = self.isolate_baseline(self.uvd[0],baseline)

        self.uvp = []
        for uvd in self.uvd:
            # Fill PSpecData object
            ds = hp.PSpecData([uvd,uvd],[None,None],beam=beam)

            # Calculate baseline-pairs using the pairs chosen earlier
            bls1,bls2,blpairs = hp.utils.construct_blpairs(pairs, exclude_auto_bls=True, 
                                                             exclude_permutations=True)

            # Calculate power spectrum
            uvp = ds.pspec(bls1,bls2,dsets=(0,1),pols=pols,spw_ranges=spw_ranges, 
                           input_data_weight='identity',norm='I',taper=taper,
                           verbose=False)

            self.uvp += [copy.deepcopy(uvp)]

        self.__calctime += time.time() - t
        self.jackknives.uvp = self.uvp

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

        imag: boolean, optional
            Whether to return the imaginary component of the power spectrum instead of the real. 
            Default: False

        bootstrap_times: boolean, optional
            If true, generates errors using time axis as well. Othewise, averages over time
            axis before bootstrapping errors.

        Returns:
        -------
        dlys: list
            Delay modes, x-axis of a power spectrum graph.

        avspec: list
            The delay spectrum values at each delay.

        errs: list
            Estimated errors for the delays spectrum at each delay.
        """
        t = time.time()
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

        times = np.unique(uvp.time_avg_array)

        allspecs = []
        for i in range(len(blpairs)):
            key=(0,blpairs[i],"xx")

            if bootstrap_times:
                # Use every spectrum available for bootstrapping
                allspecs += list(uvp.get_data(key))
            else:
                # Average over the time axis before bootstrapping
                allspecs += [np.average(uvp.get_data(key),axis=0)]

        lboots = []
        for i in range(n_boots):
            # Choose spectra indices at random
            inds = np.random.choice(range(len(allspecs)),len(allspecs),replace=True)
            boot = [allspecs[i] for i in inds]

            # Average over these random spectra
            bootav = np.average(np.vstack(boot),axis=0)

            lboots += [bootav]

        # Find standard deviation of all bootstrapped spectra
        lboots = np.array(lboots)
        err = np.std(proj(lboots),axis=0)

        self.__boottime += time.time() - t
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

        bootstrap_times: boolean, optional
            If true, generates errors using time axis as well. Othewise, averages over time
            axis before bootstrapping errors.

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

        for uvp_pair in uvpl:
            sp = []
            e = []
            for uvp in uvp_pair:
                # Bootstrap errors using other function
                _uvp = copy.deepcopy(uvp)
                dlys,av,er = self.bootstrap_errs_once(_uvp,pol=pol,n_boots=n_boots,
                                                      imag=imag,
                                                      bootstrap_times=bootstrap_times)
                # Add results to list
                sp += [av]
                e += [er]

            avspec += [sp]
            errs += [e]

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
        redpairs = redcal.get_pos_reds(dic)

        if bsl == None:
            # Find longest redundant antenna group
            numpairs = np.array([len(p) for p in redpairs])
            ind = np.where(numpairs == max(numpairs))[0][0]
            return redpairs[ind]

        # Get all antennas, including those without data
        [pos_nd, num_nd] = uvd.get_ENU_antpos()

        # Make sure both antenna are in the list of all available antenna and positions
        if sum([b in num_nd for b in bsl]) != 2:
            raise Exception("Antenna pair not found. Available antennas are: " +
                            str(sorted(num_nd.tolist())))

        dic_nd = dict(zip(num_nd,pos_nd))
        redpairs_nd = redcal.get_pos_reds(dic_nd)

        # Reverse the order of the antenna pair
        bsl_bw = (bsl[1],bsl[0])

        # Check each redundant antenna pair group to see if it contains baseline
        matches = [bsl in rp for rp in redpairs_nd]
        matches_bw = [bsl_bw in rp for rp in redpairs_nd]

        # Find where the baseline is contained
        where = np.where(np.array(matches) + np.array(matches_bw) > 0)[0][0]
        val = redpairs_nd[where]

        # Next look through the redundant pairs of antenna with data
        for pairs in redpairs:
            if pairs[0] in val:
                return pairs

        # If nothing found, return empty list
        return []

    def validate(self):
        """
        Validates loaded UVdata files. If one file has antenna that another one does not, uses only
        antenna that are found in every file
        """
        ants = np.unique(np.hstack([u.get_ants() for u in np.hstack(self.uvd)]))
        use = []
        for a in ants:
            ha,nf = self.hasants([a])
            if False not in ha:
                use += [a]
        ha,nf = self.hasants(use)
        for i in range(len(self.uvd)):
            if ha[i] == False:
                self.uvd[i].select(antenna_nums=use)

    def hasants(self, ants):
        """
        Searches which files have ant data inside them.
        
        Parameters
        ----------
        ants: list
            List of antenna to search.

        Returns
        -------
        hasants: list
            List of booleans, true if the file has the antenna specified, false if missing even one.

        notfoundin: list
            List of files in which the antenna inputted were not found.
        """
        myants = [u.get_ants() for u in np.hstack(self.uvd)]
        
        hasants = []
        notfoundin = []
        for i,ma in enumerate(myants):
            # See if ants are in antenna list for specific UVData
            if sum([a not in ma for a in ants]) > 0:
                hasants += [False]
                notfoundin += [self.files[i]]
            else:
                hasants += [True]
        return hasants, notfoundin

    def clock_reset(self):
        """
        Resets the runtime timer.
        """
        self.__loadtime = 0.
        self.__boottime = 0.
        self.__tottime = 0.
        self.__calctime = 0.