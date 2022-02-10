import numpy as np
from scipy.stats import norm
from scipy.integrate import quad_vec
from collections import namedtuple
import copy

gauss_prior = namedtuple("gauss_prior", ["mean", "std"])


class bandpower():

    def __init__(self, mean=1, std=0.5, bias=0, num_pow=4, num_draw=int(1e6)):
        """
        Container for holding bandpower draws and related metadata.

        Parameters:
            mean: The unbiased mean of the bandpower draws
            std: The standard deviation of the bandpower draws
            bias: An optional bias to add to the mean
            num_pow: Number of bandpowers to draw in a single group
            num_draw: Number of draws to do of length num_pow

        Attributes:
            bp_draws: An array of draws from a gaussian with parameters
                described by the class parameters
        """
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

        self.bp_draws = np.random.normal(loc=self.mean + self.bias,
                                         scale=self.std,
                                         size=self.dat_shape)


class bias_jackknife():

    def __init__(self, bp_obj, bp_prior_mean=1, bp_prior_std=0.5,
                 bias_prior_mean=0, bias_prior_std=10, p_bad=0.5, analytic=True):
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
            p_bad: Prior probability of an epoch with at least one biased draw
            analytic: Whether to use analytic result for likelihood computation
        """

        self.bp_prior = gauss_prior(bp_prior_mean, bp_prior_std)
        self.bias_prior = gauss_prior(bias_prior_mean, bias_prior_std)

        if (p_bad > 0) and (p_bad < 1):
            self.null_prior = np.array([p_bad, 1 - p_bad])
        else:
            raise ValueError("p_bad keyword must lie strictly between 0 and 1 (boundaries excluded).")

        self.bp_obj = copy.deepcopy(bp_obj)
        self.analytic = analytic

        self.like = self.get_like()
        self.evid = self.get_evidence()
        self.post = self.get_post()

    def get_like(self):
        """
        Get the likelihoods for each of the null hypotheses.

        Args:
            analytic: Whether to use the analytic result. If False, scipy's
                quadrature integration methods are used.
        """

        like = np.zeros([2, self.bp_obj.num_draw])
        for null_cond in [0, 1]:
            if self.analytic:
                like[null_cond] = self._get_like_analytic(null_cond)
            else:
                like[null_cond] = self._get_like_num(null_cond)

        return(like)

    def _get_mod_var_mean_gauss_2(self, null_cond, debug=False):
        cov_sum = self.bp_obj.std**2 + (1 - null_cond) * self.bias_prior.std**2  # Add term if not null_cond
        mod_bp = np.copy(self.bp_obj.bp_draws) - (1 - null_cond) * self.bias_prior.mean  # Subtract term if not null_cond
        if isinstance(self.bp_obj.std, np.ndarray):
            if debug:
                print("Getting params in array case.")
            mod_var = 1 / np.sum(1 / cov_sum)
            mod_mean = mod_var * np.sum(mod_bp / cov_sum, axis=1)
            gauss_2_arg = 0.5 * (np.sum(mod_bp**2 / cov_sum, axis=1) - mod_mean**2 / mod_var)
            gauss_2_prefac = np.sqrt(2 * np.pi * mod_var) / np.sqrt(np.prod(2 * np.pi * cov_sum))
        else:
            if debug:
                print("Getting params in non-array case.")
            mod_var = cov_sum / self.bp_obj.num_pow
            mod_mean = np.mean(mod_bp, axis=1)
            gauss_2_arg = 0.5 * np.var(mod_bp, axis=1, ddof=False) / mod_var
            gauss_2_prefac = np.sqrt(2 * np.pi * mod_var) / np.sqrt(2 * np.pi * cov_sum)**self.bp_obj.num_pow

        return(mod_var, mod_mean, gauss_2_arg, gauss_2_prefac)

    def _get_like_analytic(self, null_cond):

        mod_var, mod_mean, gauss_2_arg, gauss_2_prefac = self._get_mod_var_mean_gauss_2(null_cond)

        gauss_1_loc = self.bp_prior.mean
        gauss_1_scale = np.sqrt(mod_var + self.bp_prior.std**2)

        val = norm.pdf(mod_mean, loc=gauss_1_loc, scale=gauss_1_scale) * np.exp(-gauss_2_arg) * gauss_2_prefac

        return(val)

    def _get_integr(self, null_cond):
        if null_cond:
            gauss_1_loc = self.bp_obj.bp_draws
            gauss_1_scale = self.bp_obj.std
        else:
            gauss_1_loc = self.bp_obj.bp_draws - self.bias_prior.mean
            gauss_1_scale = np.sqrt(self.bp_obj.std**2 + self.bias_prior.std**2)

        gauss_2_loc = self.bp_prior.mean
        gauss_2_scale = self.bp_prior.std

        def integrand(x):
            gauss_1 = np.prod(norm.pdf(x, loc=gauss_1_loc, scale=gauss_1_scale), axis=1)
            gauss_2 = norm.pdf(x, loc=gauss_2_loc, scale=gauss_2_scale)

            return(gauss_1 * gauss_2)

        return(integrand)

    def _get_like_num(self, null_cond):

        integrand_func = self._get_integr(null_cond)

        integral = quad_vec(integrand_func, -np.inf, np.inf)[0]

        return(integral)

    def get_evidence(self):
        evid = self.null_prior @ self.like
        return(evid)

    def get_post(self):
        # Transpose to make shapes conform to numpy broadcasting
        post = (self.like.T * self.null_prior).T / self.evid
        return(post)
