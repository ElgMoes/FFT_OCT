import numpy as np
from scipy.optimize import curve_fit

def noiseBehaviour(rms, noise, noiseSamples=1000, unc=None):
    """ noiseBehaviour(rms, noise, noiseSamples=1000)
    essentially, a linear fit to rms vs noise
    returns the value from rms = 1, which is also the slope
    """
    
    def lin(x, k):
        return k*x
    
    if unc is None:
        # if uncertainty is not given, try to estimate from sample size
        unc = noise/np.sqrt(noiseSamples*(noiseSamples-1))
    
    k, kv = curve_fit(lin, rms, noise, p0=noise[-1]/rms[-1], sigma=unc)
       
    return k[0], np.sqrt(kv[0,0])