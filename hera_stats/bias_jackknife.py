import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.sparse import block_diag
from scipy.integrate import quad_vec
from scipy.special import comb
from collections import namedtuple
from itertools import combinations, chain
from more_itertools import set_partitions
from more_itertools.recipes import powerset
import copy

gauss_prior = namedtuple("gauss_prior", ["mean", "std"])
multi_gauss_prior = namedtuple("multi_gauss_prior", ["mean", "cov"])

class bandpower():

    def __init__(self, simulate=True, mean=1, std=0.5, bias=0, num_pow=4,
                 num_draw=int(1e6), bp_meas=None):
        """
        Container for holding bandpower draws and related metadata.

        Parameters:
            simulate: Whether to simulate bandpower draws
            mean: The unbiased mean of the bandpower draws
            std: The standard deviation of the bandpower draws
            bias: An optional bias to add to the mean
            num_pow: Number of bandpowers to draw in a single group
            num_draw: Number of draws to do of length num_pow
            bp_meas: Some measured bandpowers, if not simulating.

        Attributes:
            bp_draws: An array of draws from a gaussian with parameters
                described by the class parameters, OR an array of measured
                bandpowers.
        """

        neither = ((bp_meas is None) and (not simulate))
        both = ((bp_meas is not None) and simulate)
        if neither:
            raise ValueError("User must supply measured bandpowers if simulate is False")
        elif both:
            raise ValueError("Measured bandpowers have been supplied and simulate has been set to True."
                             "User must either supply measured bandpowers or simulate them.")

        if not isinstance(num_draw, int):
            print("Casting num_draw parameter as an integer")
            num_samp = int(num_draw)

        if not isinstance(num_pow, int):
            print("Casting num_pow parameter as an integer")
            num_pow = int(num_pow)

        if "__iter__" in dir(std):
            if not isinstance(std, np.ndarray):
                print("Casting std parameter as an array")
                std = np.array(std)

        self.num_pow = num_pow
        self.num_draw = num_draw
        self.dat_shape = (num_draw, num_pow)

        self.mean = mean
        self.std = std
        self.bias = bias

        if simulate:
            self.simulated = True
            self.bp_draws = np.random.normal(loc=self.mean + self.bias,
                                             scale=self.std,
                                             size=self.dat_shape)
        else:
            self.simulated = False
            if self.num_draw > 1:
                raise ValueError("num_draw must be set to 1 for measured bandpowers.")
            bp_meas_arr = np.array(bp_meas)
            self.bp_draws = bp_meas_arr[np.newaxis, :]
            shape_match = self.bp_draws.shape == self.dat_shape
            if not shape_match:
                raise ValueError("User must supply 1-dimensional input for bp_meas of length num_pow.")


class bias_jackknife():

    def __init__(self, bp_obj, bp_prior_mean=1, bp_prior_std=0.5,
                 bias_prior_mean=0, bias_prior_std=10, bias_prior_corr=1,
                 hyp_prior=None, analytic=True, mode='full'):
        """
        Class for containing jackknife parameters and doing calculations of
        various test statistics.

        Parameters:
            bp_obj: A bandpower object that holds the bandpower draws
                with which to work.
            bp_pior_mean: Mean parameter for the bandpower prior
            bp_prior_std: Standard deviation for bandpower prior
            bias_prior_mean: Mean parameter for bias prior
            bias_prior_std: Standard deviation for bias prior
            bias_prior_corr: Correlation coefficient for bias prior in correlated hypothesis
            hyp_prior: Prior probability of each hypothesis, in the order (null, uncorrelated bias, correlated bias)
            analytic: Whether to use analytic result for likelihood computation
        """
        if mode not in ['full', 'diagonal', 'ternary', 'binary']:
            raise ValueError("mode must be 'full', 'diagonal', 'ternary', or 'binary'")
        self.mode = mode
        self.bp_obj = copy.deepcopy(bp_obj)
        self.bp_prior = gauss_prior(bp_prior_mean, bp_prior_std)
        self.num_hyp = self.get_num_hyp()
        bias_prior_mean, bias_prior_cov = self._get_bias_mean_cov(bias_prior_mean,
                                                                  bias_prior_std,
                                                                  bias_prior_corr)
        if hyp_prior is None:  # Default to flat
            self.hyp_prior = self._get_default_prior()
        elif not np.isclose(np.sum(hyp_prior), 1):
            raise ValueError("hyp_prior does not sum close to 1, which can result in faulty normalization.")
        elif len(hyp_prior) != self.num_hyp:
            raise ValueError("hyp_prior length does not match hypothesis set length. Check mode keyword.")
        else:
            self.hyp_prior = hyp_prior
        self.bias_prior = multi_gauss_prior(bias_prior_mean, bias_prior_cov)

        self.analytic = analytic
        self.noise_cov = self._get_noise_cov()

        self.like = self.get_like()
        self.evid = self.get_evidence()
        self.post = self.get_post()

    def _get_default_prior(self):
        hyp_prior = np.zeros(self.num_hyp)
        hyp_prior[0] = 0.5
        hyp_prior[1:] = 0.5 / (self.num_hyp - 1)
        return(hyp_prior)

    def get_num_hyp(self):
        """
        Fun fact: these are called bell numbers in the full case. For N
        bandpowers, we actually want the N+1th bell number, which is the total
        number of ways of partitioning a set with N+1 elements.
        """
        if self.mode == 'full':
            M = self.bp_obj.num_pow + 1
            B = np.zeros(M + 1, dtype=int)
            B[0] = 1  # NEED THE SEED
            for n in range(M):
                for k in range(n + 1):
                    B[n + 1] += comb(n, k, exact=True) * B[k]

            num_hyp = B[M]
        elif self.mode == 'diagonal':
            num_hyp = 2**(self.bp_obj.num_pow)
        elif self.mode == 'ternary':
            num_hyp = 3
        elif self.mode == 'binary':
            num_hyp = 2
        return(num_hyp)

    def get_like(self):
        """
        Get the likelihoods for each of the null hypotheses.
        """

        like = np.zeros([self.num_hyp, self.bp_obj.num_draw])
        for hyp_ind in range(self.num_hyp):
            if self.analytic:
                like[hyp_ind] = self._get_like_analytic(hyp_ind)
            else:
                like[hyp_ind] = self._get_like_num(hyp_ind)

        return(like)

    def _get_bias_mean_cov(self, bias_prior_mean, bias_prior_std, bias_prior_corr):

        bias_cov_shape = [self.num_hyp, self.bp_obj.num_pow, self.bp_obj.num_pow]
        bias_cov = np.zeros(bias_cov_shape)

        ###
        # The matrices need to be transitive - this should be all of them. #
        ###
        diag_val = bias_prior_std**2
        off_diag_val = bias_prior_corr * diag_val
        hyp_ind = 0
        if self.mode in ["full", "diagonal"]:
            for diag_on in powerset(range(self.bp_obj.num_pow)):
                N_on = len(diag_on)
                if N_on == 0:  # Null hypothesis - all 0 cov. matrix
                    hyp_ind += 1
                elif self.mode == "full":
                    parts = set_partitions(diag_on)  # Set of partitionings
                    for part in parts:  # Loop over partitionings
                        bias_cov[hyp_ind, diag_on, diag_on] = diag_val
                        for sub_part in part:  # Loop over compartments to correlate them
                            off_diags = combinations(sub_part, 2)  # Get off-diagonal indices for this compartment
                            for pair in off_diags:  # Fill off-diagonal indices for this compartment
                                bias_cov[hyp_ind, pair[0], pair[1]] = off_diag_val
                                bias_cov[hyp_ind, pair[1], pair[0]] = off_diag_val
                        hyp_ind += 1
                else:  # Mode must be diagonal
                    bias_cov[hyp_ind, diag_on, diag_on] = diag_val
                    hyp_ind += 1
        elif self.mode in ["ternary", "binary"]:
            diag_inds = np.arange(self.bp_obj.num_pow)
            bias_cov[1] = diag_val * np.eye(self.bp_obj.num_pow)
            if self.mode == "ternary":
                bias_cov[2] = (1 - bias_prior_corr) * bias_cov[1] + off_diag_val * np.ones(bias_cov_shape[1:])

        bias_mean = np.zeros([self.num_hyp, self.bp_obj.num_pow])
        bias_mean_vec = np.repeat(bias_prior_mean, self.bp_obj.num_pow)
        bias_mean[1:] = np.repeat(bias_mean_vec[np.newaxis, :], self.num_hyp - 1,
                                  axis=0)

        return(bias_mean, bias_cov)

    def _get_noise_cov(self):
        # Assuming a scalar
        if hasattr(self.bp_obj.std, "__iter__"):  # Assume a vector
            noise_cov = np.diag(self.bp_obj.std**2)
        else:
            noise_cov = np.diag(np.repeat(self.bp_obj.std**2, self.bp_obj.num_pow))
        return(noise_cov)

    def _get_mod_var_mean_cov_sum(self, hyp_ind):
        cov_sum = self.noise_cov + self.bias_prior.cov[hyp_ind]
        cov_sum_inv = np.linalg.inv(cov_sum)
        mod_var = 1 / np.sum(cov_sum_inv)
        mod_mean = mod_var * np.sum((self.bp_obj.bp_draws - self.bias_prior.mean[hyp_ind]) @ cov_sum_inv, axis=1)

        return(mod_var, mod_mean, cov_sum)

    def _get_like_analytic(self, hyp_ind):

        mod_var, mod_mean, cov_sum = self._get_mod_var_mean_cov_sum(hyp_ind)

        # Use log to avoid dividing by 0
        gauss1 = norm.logpdf(mod_mean, loc=self.bp_prior.mean,
                             scale=np.sqrt(mod_var + self.bp_prior.std**2))
        try:
            gauss2 = multivariate_normal.logpdf(self.bp_obj.bp_draws,
                                                mean=self.bias_prior.mean[hyp_ind],
                                                cov=cov_sum)
        except ValueError:
            print(self.bias_prior.cov[hyp_ind])
            print(cov_sum)
            raise ValueError("Bad matrix. Clues printed out.")
        gauss3 = norm.logpdf(mod_mean, loc=0, scale=np.sqrt(mod_var))
        like = np.exp(gauss1 + gauss2 - gauss3)

        return(like)

    def _get_integr(self, hyp_ind):

        _, _, cov_sum = self._get_mod_var_mean_cov_sum(hyp_ind)

        def integrand(x):
            gauss_1 = multivariate_normal.pdf(self.bp_obj.bp_draws - self.bias_prior.mean[hyp_ind],
                                              mean=x * np.ones(self.bp_obj.num_pow),
                                              cov=cov_sum)
            gauss_2 = norm.pdf(x, loc=self.bp_prior.mean, scale=self.bp_prior.std)

            return(gauss_1 * gauss_2)

        return(integrand)

    def _get_like_num(self, hyp_ind):

        integrand_func = self._get_integr(hyp_ind)

        integral = quad_vec(integrand_func, -np.inf, np.inf)[0]

        return(integral)

    def get_evidence(self):
        evid = self.hyp_prior @ self.like
        return(evid)

    def get_post(self):
        # Transpose to make shapes conform to numpy broadcasting
        post = (self.like.T * self.hyp_prior).T / self.evid
        return(post)

    def gen_bp_mix(self, num_draw):
        """
        Generate a mixture of bandpower objects in accordance with the priors.

        Args:
            num_draw: How many bandpowers to simulate per hypothesis
        """
        bp_list = []
        mean = np.random.normal(loc=self.bp_prior.mean, scale=self.bp_prior.std,
                                size=[num_draw, self.num_hyp, self.bp_obj.num_pow])
        bias = np.random.multivariate_normal(mean=self.bias_prior.mean,
                                             cov=block_diag(self.bias_prior.cov),
                                             size=num_draw).reshape([num_draw, self.num_hyp, self.bp_obj.num_pow])
        std = self.bp_obj.std
        for hyp in range(self.num_hyp):
            bp = bandpower(mean=mean[:, hyp, :], std=std, num_draw=num_draw,
                           bias=bias[:, hyp, :], num_pow=self.bp_obj.num_pow)
            bp_list.append(bp)

        return(bp_list)
