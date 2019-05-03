
import numpy as np


def estimate_noise_rms(uvd, bls, fit_poly=True, order=2):
    """
    Estimate noise RMS per frequency channel by taking a simple standard 
    deviation of differences along the time axis. The real and imaginary parts 
    are treated independently. A smooth polynomial model fit can be returned if 
    desired.
    
    Parameters
    ----------
    uvd : UVData
        UVData file containing data to fit.
    
    bls : list of int/tuple
        List of baselines/baseline keys to calculate the noise rms for.
    
    fit_poly : bool, optional
        Fit a polynomial to the standard deviations as a function of frequency. 
        Default: True.
    
    order : int, optional
        If fit_poly is True, order of the polynomial to fit to the standard 
        deviations. Default: 2.
    
    Returns
    -------
    noise_rms : array_like, complex
        1D array of noise RMS (in the same units as the input data) as a 
        function of frequency. Shape is (Nbls, Nfreq-1); the differences are 
        evaluated at the point *between* frequency channel centers.
    
    rms_model : array_like, complex, optional
        If poly_fit is True, return the smooth polynomial fit to the noise rms. 
        Shape is (Nbls, Nfreq), i.e. the rms model is evaluated *at* frequency 
        channel centers.
    """
    # Loop over baseline keys
    noise_rms, rms_model = [], []
    for bl in bls:
        
        # Get data
        data = uvd.get_data(bls)
    
        # Calculate sigma_rms as function of freq, for real and imag parts
        sigma_noise = 0; sigma_model = 0
        for fn, factor in ((np.real, 1.), (np.imag, 1.j)):
            
            # Difference along time ax then take std. dev. (also along time axis)
            sigma_rms = np.std(np.diff(fn(data), axis=0), axis=0)
            
            # Get indices of masked data
            idxs = np.where(sigma_rms != 0.) # flagged channels have diff = 0
            sigma_rms[np.where(sigma_rms == 0.)] = np.nan # set flagged channel rms = nan
            sigma_noise += factor * sigma_rms
            
            # Fit polynomial to sigma_rms if requested
            if fit_poly:
                freq_chans = np.arange(sigma_rms.size + 1)
                diff_chans = 0.5 * (freq_chans[1:] + freq_chans[:-1])
                coeff_n = np.polyfit(diff_chans[idxs], sigma_rms[idxs], deg=order)
                sigma_model += factor * np.poly1d(coeff_n)(freq_chans)
        
        # Append results to list
        noise_rms.append(sigma_noise)
        if fit_poly: rms_model.append(sigma_model)
        
    # Convert to arrays and return
    if fit_poly: return np.array(noise_rms), np.array(rms_model)
    return np.array(sigma_rms)
    
