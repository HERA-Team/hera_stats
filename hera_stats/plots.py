# Plots for visualizing data

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from jkf_stats import jkf_stats
import utils

class plots():

    def __init__(self):
        """
        Class for plotting and analyzing jackknife data.
        """
        self.data = utils.jkfdata()
        self.stats = jkf_stats()
        self.data.sortstring = ""
        self.data.sortitem = None

    def load_file(self,filepath):
        """Load jackknife file.

        Parameters
        ----------
        filepath: str
            The path of the jackknife file (.*jkf) to load.
        """
        if filepath[-4:] != ".jkf":
            raise IOError("Jackknife file not found at location specified.")

        self.stats.load_file(filepath)
        with open(filepath,"rb") as f:
            dic = pkl.load(f)
            self.load_dic(dic)

    def load_dic(self,dic):
        """
        Loads a dictionary of jackknife data.
        
        Parameters
        ----------
        dic: dictionary
            Dictionary outputted by jackknife.
        """
        self.data.load(dic)
        self.stats.data.load(dic)

    def sort(self,item):
        """
        Sort the spectra by specific antenna. Goes through the list of spectra
        pairs and puts whichever spectra has used the specified antenna into the 
        first slot. Output is exactly the same shape as input, but sorted.

        If item == None, it randomizes the order of the spectra, effectively erasing
        any previous actions.

        Parameters
        ----------
        item: int or tuple
            Item to sort spectra by. Can be antenna or baseline.
        """
        self.data.validate()

        # Sanity checks
        if item not in np.array(self.data.grps).flatten() and item != None:
            raise AttributeError("Item not found in groups and not None.")
        if type(self.data.grps[0][0]) == str:
            raise AttributeError("Cannot sort spectra if grouped by files.")

        sort,errs,grps = [],[],[]
        for i in range(self.data.n_jacks):
            # Get jackknife data 
            gr = self.data.grps[i]
            sp = self.data.spectra[i]
            er = self.data.errs[i]

            # Check if item is in either of the current groups. If not, randomly order spectra
            if item in gr[0]:
                i1,i2 = 0,1
            elif item in gr[1]:
                i1,i2 = 1,0
            else:
                [i1,i2] = np.random.choice([0,1],2,replace=False)

            sort += [[sp[i1],sp[i2]]]
            errs += [[er[i1],er[i2]]]
            grps += [[gr[i1],gr[i2]]]

        # Set sortstring for plots to use
        if item != None:
            self.data.sortstring = "(Sorted by Item: %s)" %str(item)
            self.data.sortitem = item
        else:
            self.data.sortstring = ""
            self.data.sortitem = None

        # Save things
        self.data.spectra = sort
        self.data.errs = errs
        self.data.grps = grps
        self.stats.data = self.data

    def find_outliers(self,tolerance="normal",verbose=False):
        """
        Checks for outliers. Does a signifance level failure check for jackknife pairs. 
        standardizes the jackknife data, then rejects spectra if it fails the test.

        This test is basic, but can weed out bad spectra with ease. For each siginicance level, it counts
        the number of points where the zscore exceeds this level. For a normal distribution, one would expect
        the 1-sigma, 2-sigma, and 3-sigma fail fractions to be [0.317, 0.455, 0.0027]. Thus, distributions that 
        are highly nongaussian will greatly exceed these fail fractions.

        Parameters
        ----------
        tolerance: string
            The tolerance level, predetermined but still changeable. Options: "lenient","normal","strict".
            Default: "normal".

        verbose: boolean
            If true, prints out test results.

        Returns
        -------
        outliers: list
            List of jackknife numbers that are outliers. Note that jackknife numbers are different format than
            spectrum indices, greater by 1.
        """
        zs = self.stats.standardize(self.data.spectra,self.data.errs)

        if verbose: print "Looking for Outliers..."
        # Set tolerances
        if tolerance == "normal":
            tols = [0.8,0.3,0.2]
        elif tolerance == "strict":
            tols = [0.6,0.2,0.1]
        elif tolerance == "lenient":
            tols = [0.9,0.5,0.4]
        outl = []
        for i,z in enumerate(zs):
            # Fraction of z scores that are above significance level:
            sig = [float(sum(np.abs(z) > lev))/len(z) for lev in [1,2,3]]

            # True if failure fraction exceeds tolerance
            failed = np.array(sig) > np.array(tols)
            if sum(failed) > 0:
                if verbose:
                    print "%i: \t%r \t%r"%(i+1,sig,list(failed))
                outl += [i+1]
        return outl

    def find_bad_items(self,tolerance="normal",verbose=False):
        """
        Tests the items within each group to see which fail the kstest badly and which do not.
        
        Parameters
        """
        if tolerance == "normal":
            tol = 0.4
        elif tolerance == "strict":
            tol = 0.1
        elif tolerance == "lenient":
            tol = 0.7
        else:
            raise NameError("Tolerance string not recognized. Options: \"lenient\",\"normal\",\"strict\"")

        if verbose: print("Looking for bad items, tolerance: %.2f"%tol)

        # Save last sorted item so sort can reset the order later
        saveitem = self.data.sortitem
        items = utils.unique_items(self.data.grps)
        fails = []
        for item in items:
            # Sort spectra by items, run kstest
            self.sort(item)
            failfrac = self.stats.kstest()

            if failfrac >= tol:
                # If failure, save to list
                if verbose: print("%r\t%.2f"%(item,failfrac))
                fails += [item]

        # Return spectra to original order and return
        self.sort(saveitem)
        return fails

    def __plot_spectrum(self,ax,dlys,spectrum,errs=None):
        """
        Plots a single power spectrum.

        Parameters
        ----------
        ax: matplotlib axis
            Axis on which to plot the spectrum

        dlys: list
            Delays of the spectrum points. 

        spectrum: list
            Power spectrum values.

        errs: list, optional
            If true, plots the errorbars as well
        """
        self.data.validate()

        # If errors are specified, plot those too
        if type(errs) == type(None):
            p, = ax.plot(dlys,spectrum)
        else:
            p = ax.errorbar(dlys,spectrum,errs)

        # Set other parameters
        ax.set_yscale("log")
        ax.set_ylabel(r"P($\tau$)")
        ax.set_xlabel("Delay (ns)")
        ax.set_title("Power Spectrum" + self.data.sortstring)
        ax.grid(True)

        return p

    def plot_spectra(self,n=1,fig=None,show_groups=False,savefile=None):
        """
        Plots a single pairs of power spectra for a splitting jackknife.

        Parameters
        ----------
        n: int, optional
            Index of the spectra pair to use. Default: 1.

        fig: matplotlib.pyplot.figure, optional
            Where to plot the spectra. If None, creates figure. Default: None.

        show_groups: boolean, optional
            Whether to label spectra arbitrarily or by the antenna used. Default: False.

        savefile: str, optional
            If specified, will save the image to a file of this name. Default: None.
        """
        self.data.validate()

        n-=1
        if n < 0 or n >= self.data.n_jacks:
            raise ValueError("Jaccknife number outside of range. \
                            Avail: 0-%i" %self.data.n_jacks)
        if fig == None:
            fig = plt.figure(figsize=(8,5))

        # Create subplot
        ax = fig.add_subplot(111)

        # Use labels of groups 
        if show_groups:
            if type(self.data.grps[0][0]) == list:
                labels=[str(sorted(g)) for g in self.data.grps[n]]
            else:
                labels = [g for g in self.data.grps[n]]
        else:
            labels=["Antenna Group 1", "Antenna Group 2"]

        # Plot power spectrum
        spec = self.data.spectra[n]
        p_l = [self.__plot_spectrum(ax,self.data.dlys,s) for s in spec]

        # Set ylims based on range of the spectrum
        mx,mn = np.max(np.array(spec)),np.min(np.array(spec))
        ax.set_ylim(mn*0.6,mx*5.)

        # Set other graph details
        ax.legend(p_l,labels,fontsize=8,loc=1)
        ax.set_title("Power Spectrum for Jackknife %i/%i " %(n+1,self.data.n_jacks) + 
                     self.data.sortstring)

    def plot_spectra_errs(self,n=1,fig=None,show_groups=False,savefile=None,plot_zscores=True,zlim=5):
        """
        Plots a pair of spectra, their errors, and normalized residuals.

        Parameters
        ----------
        n: int, optional
            Number (out of total) position of the spectra pair to use [1 - n_jacks]. Default: 1.

        fig: matplotlib figure, optional
            Where to plot the spectra. If None, creates figure. Default: None.

        show_groups: boolean, optional
            Whether to label spectra arbitrarily or by the antenna used. Default: False.

        savefile: str, optional
            If specified, will save the image to a file of this name. Default: None.
        """
        self.data.validate()
        if self.data.jackpairs < 2:
            plot_zscores = False

        n-=1
        if n < 0 or n >= self.data.n_jacks:
            raise ValueError("Jaccknife number outside of range. \
                            Avail: 0-%i" %self.data.n_jacks)

        if fig == None:
            fig = plt.figure(figsize=(8,5))

        if plot_zscores:
            ax = fig.add_axes([0.1,0.3,0.8,0.7])
        else:
            ax = fig.add_subplot(111)

        # Choose labels appropriately
        if show_groups:
            if type(self.data.grps[0][0]) == list:
                labels=[str(sorted(g)) for g in self.data.grps[n]]
            else:
                labels = [g for g in self.data.grps[n]]
        else:
            labels=["Antenna Group 1", "Antenna Group 2"]

        # Plot spectra with errorbars
        spec,er = self.data.spectra[n],self.data.errs[n]
        p_l = [self.__plot_spectrum(ax,self.data.dlys,spec[i],er[i]) for i in range(len(spec))]

        # Set ylims based on range of power spectrum
        mx,mn = np.max(np.array(spec)),np.min(np.array(spec))
        ax.set_ylim(mn*0.6,mx*4.)

        # Set other details
        ax.legend(p_l,labels,fontsize=8,loc=1)
        ax.set_title("Power Spectrum for Jackknife %i/%i " %(n+1,self.data.n_jacks) + 
                    self.data.sortstring)

        if plot_zscores:
            # Create second plot for z scores
            ax2 = fig.add_axes([0.1,0.1,0.8,0.2])

            # plot z scores as scatterplot
            stds = self.stats.standardize(self.data.spectra,self.data.errs)[n]
            ax2.scatter(self.data.dlys,stds,marker="+",color="green")
            xlims = ax.get_xlim() # Save axis limite

            # Plot zero line 
            ax2.hlines([0],[-10000],[10000],linestyles="--",alpha=0.6)

            # Calculate yticks.
            zmt = zlim//2*2
            ticks = range(-zmt,zmt+1,2)

            # Reinstate limits and set other paramters
            ax2.set_ylim(-zlim,zlim)
            ax2.set_xlim(xlims[0],xlims[1])
            ax2.set_yticks(ticks)
            ax2.set_xlabel("Delay (ns)")
            ax2.set_ylabel("Z-Score")
            ax2.grid()

        # Save if necessary
        if savefile != None:
            fig.savefig(savefile)

    def hist_2d(self,plottype="varsum",ax=None,ybins=40,display_stats=True,
                vmax=None,normalize=False,returned=False):
        """
        Plots 2d power spectrum for the data.

        Parameters
        ----------
        plottype: string, optional
            Type of plot. Options: ["varsum","raw","norm","weightedsum"]. 
            Default: "varsum"

        ax: matplotlib axis, optional
            If specified, plots the heatmap on this axis.

        ybins: int or list, optional
            If int, the number of ybins to use. If list, the entries are used
            as bin locations. Note: "raw" requires logarithmic binning.
            Default: 40

        display_stats: boolean, optional
            If true, plots the average and 1-sigma confidence interval of the data
            Default: True

        vmax: float or int, optional
            If set, sets the value for the maximum color. Useful if comparing 
            data sets. Default: None.

        returned: boolean, optional
            If true, returns the histogram as well as plotting it. Default: False.
        """
        self.data.validate()

        if ax == None:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(111)

        if plottype == "varsum":
            # Plot zscores using sum of variances 
            data = self.stats.standardize(self.data.spectra,self.data.errs)
            ylims = (-5,5)

        elif plottype == "raw":
            # Plot raw spectra, combine jackknife pairs
            data = np.vstack([np.vstack(sp) for sp in self.data.spectra])
            ylims = (np.log10(np.min(self.data.spectra)),
                     np.log10(np.max(self.data.spectra)))
            ax.set_yscale("log")

        elif plottype == "norm":
            # Plot normed difference, that is, spectra devided by average
            allspecs = np.vstack(self.data.spectra)
            data = allspecs/np.average(allspecs,axis=0)
            ylims = (np.min(data),np.max(data))

        elif plottype == "weightedsum":
            # Plot zscores using sum of variances method
            stds = self.stats.standardize(self.data.spectra,
                                          self.data.errs,method="weightedsum")
            data = np.vstack(stds)
            ylims = (-5,5)
        else: 
            # Otherwise, type not recognized
            raise NameError("Plot type not recognized")

        # Make bins
        if type(ybins) == int:
            ybins = np.linspace(ylims[0],ylims[1],ybins+1)

            if plottype == "raw":
                ybins = 10**ybins

        xbins = np.linspace(min(self.data.dlys)-1,
                            max(self.data.dlys)+1,
                            len(self.data.dlys)+1)

        x = np.hstack([self.data.dlys]*len(data))
        y = np.hstack(data)

        if display_stats:
            # Plot average and stdev if requested
            avs = np.average(data,axis=0)
            stdevs = np.std(data,axis=0)
            av, = ax.plot(self.data.dlys,avs,c="black",lw=2)
            sigma, = ax.plot(self.data.dlys,avs+stdevs,c="red",ls="--",lw=2)
            ax.plot(self.data.dlys,avs-stdevs,c="red",ls="--",lw=2)
            ax.legend([av,sigma],["Average","1-Sigma"],loc=1)
            ax.set_ylim(min(ybins),max(ybins))

        # Calculate histogram
        hist,xpts,ypts = np.histogram2d(x,y,bins=[xbins,ybins])
        
        if normalize == True:
            hist /= len(data)
        if vmax == None:
            vmax = np.max(hist)/2

        # Plot hist
        c = ax.pcolor(xpts,ypts,hist.T,vmax=vmax)
        plt.colorbar(c,ax=ax)

        # Set labels
        ax.set_title("Jackknife Data Histogram " + self.data.sortstring, fontsize=16)
        ax.set_xlabel("Delay (ns)")

        if plottype == "norm":
            ax.set_ylabel("Normalized Difference")
        elif plottype == "raw":
            ax.set_ylabel(r"P($\tau$)")
        else: 
            ax.set_ylabel("Z-Score")

        if returned:
            return hist

    def plot_kstest(self,ax=None,norm=(0,1), method="varsum"):
        """
        Plots the Kolmogorov-Smirnov test for each delay mode.

        The KS test is a test of normality, in this case, for a gaussian (avg, stdev) 
        as specified by the parameter norm. If the p-value is below the KS stat,
        the null hypothesis (gaussian curve (0,1)) is rejected.

        Parameters
        ----------
        ax: matplotlib axis, optional
            The axis on which to plot the ks test. Default: None.

        norm: len-2 tuple, optional
            (Average, Standard Deviation) of null hypothesis gaussian. Default: (0,1).

        method: string, optional
            Method used to calculate z-scores for Kolmogorov-Smirnov Test. 
            Options: ["varsum","weighted"]. Default: varsum.
        """
        self.data.validate()

        if ax == None:
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)

        # Get ks information
        ks,pvals = self.stats.kstest(norm=norm,method=method,asspec=True)

        # Plot it
        p2, = ax.plot(self.data.dlys,pvals)
        p1, = ax.plot(self.data.dlys,ks)  

        ax.legend([p1,p2],["ks-stat","p-val"])
        ax.set_xlabel("Delay (ns)")
        ax.set_title("Kolmogorov-Smirnov Test by Delay Bin " + self.data.sortstring)
        ax.set_ylim(0,1)
        ax.grid(True)

    def plot_anderson(self,ax=None, method="varsum"):
        """
        Plots the Anderson-Darling test for the normality of each delay mode. 

        Confidence levels are plotted as horizontal colored lines. The 
        numbers on the left hand side of are the observed rates 
        at which the levels are exceeded. These are similar to alpha levels, so if the 
        Anderson Darling statistic surpasses the confidence level, it indicates a 
        rejection of the null hypothesis (a normal distribution) with a certainty of 
        the confidence level exceeded.

        One would expect the fraction of times the null hypothesis is rejected to be
        roughly the same as the confidence level if the distribution is normal, so the 
        observed failure rates should match their respective confidence levels.

        Parameters
        ----------
        ax: matplotlib axis, optional
            The axis on which to plot the Anderson Darling Test. Default: None.

        method: string, optional
            Method used to calculate z-scores for Anderson Darling Test. 
            Options: ["varsum","weighted"]. Default: varsum.
        """
        self.data.validate()

        if ax == None:
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)

        # Get anderson statistics
        stat,crit = self.stats.anderson(True,method=method)

        # Plot them
        p1 = ax.plot(self.data.dlys,stat)
        lp = ax.plot(self.data.dlys,crit)[::-1]
        sigs = ["1%","2.5%","5%","10%","15%"]

        # Plot significance and failure info
        ax.legend(p1+lp,["Stat"]+sigs,loc=1)
        fails = [sum(np.array(stat) > c) for c in crit[0]]
        [ax.text(-5000,crit[0][i]+0.02,"%.1f"%fails[i]+"%") for i in range(5)]

        ax.set_xlabel("Delay (ns)")
        ax.set_title("Anderson Darling Test by Delay Bin " + self.data.sortstring)
        ax.set_ylim(0,1.15)
        ax.grid(True)

    def plot_av_std(self,ax=None,method="varsum"):
        """
        Plots the average and standard deviation of the data as a spectrum.

        Parameters
        ----------
        ax: matplotlib axis, optional
            Axis on which to plot average and stdev. Default: None.

        method: string, optional
            Method used to calculate z-scores. 
            Options: ["varsum","weighted"]. Default: varsum.
        """
        self.data.validate()

        if ax == None:
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)

        # Get average and stdev
        av,std = self.stats.av_std(method=method)

        # plot av and stdev and set other parameters
        p1, = ax.plot(self.data.dlys,av)
        p2, = ax.plot(self.data.dlys,std)
        ax.legend([p1,p2],["Avg","Stdev"])
        ax.set_title("Standard Deviation and Average " + self.data.sortstring)
        ax.set_xlabel("Delay (ns)")
        ax.set_ylim(-1,3)
        ax.grid(True)