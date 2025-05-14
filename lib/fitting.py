import math
import numpy as np
from scipy.optimize import curve_fit
import tqdm
from scipy.fftpack import fft
import scipy.stats as stats

useMP = False
if useMP:
    import multiprocessing as mp

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
            
    k = peak_index + d
    
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

def singlenoisyFit(data, rms = 0.1, Nrepeat = 1000, method = "JacobsenMod"):
    """ singlenoisyFit(data, rms = 0.1, Nrepeat = 1000, method = "JacobsenMod"):
        noise fit of a single data set with a single rms
        noise is defined by rms (normal distributed)
        Nrepeat is the number of noisy datasets generated from data and rms
        method can be a string, or a list of strings.
        
        this fit has been moved into a separate function so it can be more
        easily paralellised (with package multiprossing)
    """
    if type(method) is not str and hasattr(method, "__len__"):
        Nm = len(method)  
    else:
        method = [method]
        Nm = 1
    def offsetEst(fft_data, d = 0.75):
        Nd = len(fft_data)
        
        ind = slice(int(d*0.5*Nd), int(0.5*Nd))
        return fft_data[ind].mean()
    
    # generate data with random noise, calculate fft
    #dm = [data + np.random.normal(0, rms, size=(len(data),)) for ii in range(Nrepeat)]
    #fdm = fft(dm)
    
    kk = []
    k = np.zeros((Nrepeat, Nm))
    lD = len(data)
    Nf = int(np.floor(lD/2))
    #lD = len(data)
    for dd in range(Nrepeat):
       fdm = fft(data + np.random.normal(0,rms,size=(lD,)))[0:Nf]
       for mm in range(Nm):    # loop over different methods
            k[dd,mm] = FFT_peakFit(fdm, method[mm])
    for mm in range(Nm):
        kk.append(stats.describe(k[:,mm]))
    
    return kk

def noisyFit(data, rms = 0.1, Nrepeat = 1000, method = "JacobsenMod", nProc=None):
    """ generate and fit noisy data
    this is a wrapper that calls singleNoisyFit and allows to fit several
    rms and methods on the same data set.
    if possible and selected, it will parallelise the work load.
    """
    
    Nd = 1; Nr = 1; Nm = 1;
    
    if hasattr(data, "__len__") and hasattr(data[0],"__len__"):
        # list of lists (or array)
        Nd = len(data)
    else:
        data = [data]
    if hasattr(rms, "__len__"):
        Nr = len(rms)
    else:
        rms = [rms]
    if type(method) is not str and hasattr(method, "__len__"):
        Nm = len(method)  
    else:
        method = [method]
        
    km = np.zeros((Nd, Nr, Nm))
    kstd = np.zeros((Nd, Nr, Nm))
    
    kskew = np.zeros((Nd, Nr, Nm));
    kkurt = np.zeros((Nd, Nr, Nm));
    
    pool = None
    if useMP and (Nd*Nr*Nm > 100):  # minimum complexity requirement
        #memPinst = 8*Nrepeat*len(data[0])
        #maxProcess = int(np.floor(32e9/memPinst))
        maxProcess = mp.cpu_count()
        pool = mp.Pool(processes = min(maxProcess, mp.cpu_count()-2))
        
    print("generating noise")
    progbar = tqdm.tqdm(total = Nr*Nd, smoothing = 0.025, unit='job')
    
    def callbackProgress(*a):
        progbar.update()
        
    if pool is not None:
        res = [[ pool.apply_async(singlenoisyFit, (dd, rr, Nrepeat, method), callback=callbackProgress) for rr in rms] for dd in data]
        pool.close()
        pool.join()
        
    for dd in range(Nd):    # different data sets (base sine)
        for rr in range(Nr):    # different noise level
            if pool is None:    # not parallel -> calculate
                kk = singlenoisyFit(data[dd], rms[rr], Nrepeat, method)
            else:               # parallel -> get results
                kk = res[dd][rr].get()
            km[dd,rr,:] = [ kk[ii].mean for ii in range(len(kk)) ]
            kstd[dd,rr,:] = np.sqrt([ kk[ii].variance for ii in range(len(kk)) ])
            kskew[dd,rr,:] = [ kk[ii].skewness for ii in range(len(kk)) ]
            kkurt[dd,rr,:] = [ kk[ii].kurtosis for ii in range(len(kk)) ]
            progbar.update()
            
    return (km, kstd, kskew, kkurt), (Nd, rms, method)