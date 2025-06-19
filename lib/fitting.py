import math
import numpy as np
from scipy.optimize import curve_fit

import lib.fitting_routines as fr

def FFT_peakFit(fdata, method, peak_index = None): 
    """ 
    Finding the peak in the fft of data (fdata) using a given method; unless
    the index of the peak is given, the position of the maximum magnitude of the
    signal is used
    """
    
    k = []
    
    if hasattr(fdata, "__len__") and hasattr(fdata[0],"__len__"):
        # list of lists (or array)
        for ii in range(len(fdata)):
            k.append(FFT_peakFit(fdata[ii], method, peak_index))
        return np.array(k)
    
    if type(method) is not str and hasattr(method, "__len__"):
        for ii in range(len(method)):
            k.append(FFT_peakFit(fdata, method[ii], peak_index))
        return np.array(k)
    
    if peak_index is None: # in case we want to override for testing
        peak_index = np.argmax(np.abs(fdata))

    assert((peak_index>2) and (peak_index < len(fdata)-3)), f'maximum position {peak_index} too close to start or end of data.'
    
    y1, y2, y3 = fdata[peak_index-1:peak_index+2]
    # initial peak fit is three-point around maximum; 
    # python ranges are non-inclusive, thus the +2
    
    d = 0    
    match method.lower():
        case "maximumpixel":
            pass
        case "quadratic": # weighted average
            d = fr.quadraticDelta(abs(y1),abs(y2),abs(y3))
        case "barycentric": # weighted average
            a1 = abs(y1); a2 = abs(y2); a3=abs(y3)
            d = (a3 - a1) / (a1 + a2 + a3)
        case "jains":
            d = fr.jainsDelta(abs(y1),abs(y2),abs(y3))
        case "jacobsen":
            d = fr.jacobsenDelta(y1, y2, y3)
        case "jacobsenmod":
            d = fr.jacobsenDelta(y1, y2, y3)
            w = abs(d)    # linear weight (0 at integer, 1/2 at half-integ)
            d *= 1-w
            if d < 0:
                d += w*(fr.jacobsenDelta(fdata[peak_index-2], y1, y2) - 1)
            elif d>0:
                d += w*(fr.jacobsenDelta(y2, y3, fdata[peak_index+2]) + 1)
        case "candan":
            d = fr.candanDelta(y1, y2, y3, len(fdata))
        case "macleod":
            d = fr.macLeodDelta(y1, y2, y3)            
        case "quinns2nd":
            d = fr.quinnsDelta(y1,y2,y3)
        case "gaussian":
            res = Gauss_fit(fdata, peak_index=peak_index)
            d = res[1] - peak_index
        case _:
            print(f"unknown method {method} supplied.")
            d = np.nan
            
    k = peak_index + d + 1 # We add 1 because we feed in fdata[1:LD], shifting it all by 1 
    
    return k

def Gauss_fit(fdata, N = 7, peak_index=None):
    
    popt = []
    
    if hasattr(fdata, "__len__") and hasattr(fdata[0],"__len__"):
        # list of lists (or array)
        for ii in range(len(fdata)):
            popt.append(Gauss_fit(fdata[ii], N, peak_index))
        return popt
    
    def gauss(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/2/sigma**2)
    
    if peak_index is None: # in case we want to override for testing
         peak_index = np.argmax(np.abs(fdata))
    pl = peak_index - math.floor(N/2)
    pr = peak_index + math.floor((N-1)/2) # for even N, range is asymmetric
    
    assert((pl>0) and (pr < len(fdata))), f'maximum position {peak_index} too close to start or end of data.'
    
    x = np.arange(pl,pr+1)
    y = abs(fdata[pl:pr+1])
    
    popt, pcov = curve_fit(gauss,x,y,p0=[y.max(),peak_index,1], maxfev=20000)
    return np.append(popt, np.sqrt((y**2).sum()))