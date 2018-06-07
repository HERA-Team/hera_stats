# Statistics and such for hera data
# Duncan Rocha, June 2018

import numpy as np
import pickle as pkl
from scipy import stats as spstats
import utils

class jkf_stats():

    def __init__(self,filepath=None):
        """
        Class for analyzing jackknife data.

        Parameters
        ----------
        filepath: string
            Filepath of the jackknife data file (.jkn) outputted by 
            hera_stats.jackknife methods.
        """
        self.data = utils.jkfdata()

        if filepath != None:
            self.load_file(filepath)

    def load_file(self,filepath):
        """
        Loads a jackknife file (*.jkf) to the jkf_data object

        Parameters
        ----------
        filepath: string
            Filepath of the jackknife file.
        """
        if filepath[-4:] != ".jkf":
            raise IOError("Jackknife file not found at location specified.")

        with open(filepath,"rb") as f:
            dic = pkl.load(f)
            self.data.load(dic)  

    def standardize(self,spectra,errs,method="varsum"):
        """
        Calculates the z scores for a list of split spectra and their errors.

        Parameters
        ----------
        spectra: list (n_jacks x 2 x n_dlys)
            A list of pairs of spectra, to be used to calculate z scores.

        errs: list (n_jacks x 2 x n_dlys)
            List of pairs of errors corresponding to the above spectra
        
        method: string, optional
            Method used to calculate z-scores. 
            Options: ["varsum","weighted"]. Default: varsum.

        Returns
        -------
        zs: list
            Returns zscores for every spectrum pair given.
        """
        self.data.validate()

        if np.array(spectra).shape != np.array(errs).shape:
            raise AttributeError("Spectra pair list and corresponding errors must be \
                                 the same shape")

        if self.data.jackpairs < 2:
            raise AttributeError("Standardization failed because jackknife method not create multiple \
                                  spectra.")

        zs = []
        for i,spec in enumerate(spectra):
            err = errs[i]

            # Calculate z scores using sum of variances
            if method == "varsum":
                comberr = np.sqrt(err[0]**2 + err[1]**2)
                z = (spec[0] - spec[1])/comberr

            # Or Calculate z scores with weighted sum
            elif method == "weightedsum":
                av,unc = self.weighted_sum(spec,err)
                z = (spec[0]-spec[1])/(np.sqrt(2)*unc)

            else:
                raise NameError("Z-score calculation method not recognized")

            zs += [z]

        return zs

    def weighted_sum(self,spectra,errs):
        """
        Calculates the average and standard deviation for each delay mode using a
        weighted sum.

        Parameters
        ----------
        spectra: list (n_spec x n_dlys)
            A list containing all spectra to use in weighted sum

        errs: list of errors
            A list containing all errors on the spectra supplied.

        Returns
        -------
        av: list
            The averages calculated by the weighted sums at each delay mode.

        stdev: list
            The standard deviations for each delay mode.
        """

        spectra = np.array(spectra)
        errs = np.array(errs)

        # Check if spectra are arranged in a list
        if len(spectra) != 2:
            raise AttributeException("Input arrays must be a list of 1-D spectra \
                                    (and list of 1-D errors).")

        # Calculate weighted average and standard deviation.
        cerrs = np.sqrt(1./np.sum(errs**-2,axis=0))
        av = cerrs**2*np.sum(spectra*errs**-2,axis=0)
        stdev = cerrs*np.sqrt(len(spectra))

        return av,stdev

    def kstest(self,asspec=False,norm=(0,1),method="varsum",showmore=False):
        """
        The KS test is a test of normality, in this case, for a gaussian (avg, stdev) 
        as specified by the parameter norm. If the p-value is below the KS stat,
        the null hypothesis (gaussian curve (0,1)) is rejected.

        Parameters
        ----------
        asspec: boolean, optional
            If true, returns the ks test as a spectra, one value for each delay mode.

        norm: len-2 tuple, optional
            (Average, Standard Deviation) of null hypothesis gaussian. Default: (0,1).

        method: string, optional
            Method used to calculate z-scores for Kolmogorov-Smirnov Test. 
            Options: ["varsum","weighted"]. Default: varsum.

        showmore: boolean, optional
            If true, prints information for every delay mode instead of summary. 
            Default: False

        Returns
        -------
        failfrac: float
            The fraction of delay modes that fail the KS test.
        """
        # Calculate zscores
        zs = np.array(self.standardize(self.data.spectra,self.data.errs,method=method))

        ks_l,pval_l = [],[]
        fails = 0.
        for i,zcol in enumerate(zs.T):

            if norm == None:
                norm = (np.average(zcol),np.std(zcol))

            # Do ks test on delay mode
            [ks,pval] = spstats.kstest(zcol,spstats.norm(norm[0],norm[1]).cdf)

            # Save result
            ks_l += [ks]
            pval_l += [pval]
            isfailed = int(pval < ks)
            fails += isfailed

            if showmore:
                st = ["pass","fail"][isfailed]
                print "%i"%self.data.dlys[i], st

        # Return proper data
        if asspec:
            return ks_l,pval_l
        else:
            failfrac = fails/len(self.data.dlys)
            return failfrac

    def anderson(self,asspec=False,method="varsum",showmore=False):
        """
        Does an Anderson-Darling test on the z-scores of the data. Prints results.

        One would expect the fraction of times the null hypothesis is rejected to be
        roughly the same as the confidence level if the distribution is normal, so the 
        observed failure rates should match their respective confidence levels.

        Parameters
        ----------
        asspec: boolean, optional
            If true, returns the ks test as a spectra, one value for each delay mode.

        method: string, optional
            Method used to calculate z-scores for Anderson Darling Test. 
            Options: ["varsum","weighted"]. Default: varsum.

        showmore: boolean, optional
            If true, prints out values neatly as well as returning them

        Returns
        -------
        sigs: list
            Significance levels for the anderson darling test.

        fracs: list
            Fraction of anderson darling failures for each significance level.
        """
        # Calculate z-scores
        zs = np.array(self.standardize(self.data.spectra,self.data.errs,method=method))

        # Calculate Anderson statistic and critical values for each delay mode
        statl = []
        for i,zcol in enumerate(zs.T):
            stat,crit,sig = spstats.anderson(zcol,dist="norm")
            statl += [stat]

        if showmore:
            print "Samples: %i" % len(statl)

        # Print and save failure rates for each significance level
        fracs = []
        for i in range(5):
            frac = float(sum(np.array(statl) >= crit[i]))/len(statl) * 100
            if showmore:
                print ("Significance level: %.1f \tObserved \
                        Failure Rate: %.1f" %(sig[i],frac))
            fracs += [frac]

        # Return if specified
        if asspec:
            return statl, [list(crit)]*len(statl)
        else:
            return list(sig),fracs

    def av_std(self,method="varsum"):
        """
        Gives the average and standard deviation of the data,as a spectrum.

        Parameters
        ----------
        method: string, optional
            Method used to calculate z-scores. 
            Options: ["varsum","weighted"]. Default: varsum.

        Returns
        -------
        average: list
            The average z-score for every delay mode.

        std: list
            The standard deviation of z-scores for every delay mode.
        """
        # Calculate zscores
        zs = np.array(self.standardize(self.data.spectra,self.data.errs,method=method))

        # Return avg and stdev
        return np.average(zs,axis=0), np.std(zs,axis=0)