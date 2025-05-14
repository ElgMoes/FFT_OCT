#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:47:10 2024

@author: Gerhard Blab

routines for FFT peak fitting; 
contributions by Miriam Voots, Bram Haasnoot, and Saban Caliscan
example script for arXiv 
"""

# general use packages & plotting
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy.fftpack import fft
from scipy.optimize import curve_fit
import scipy.stats as stats

import time as tm
import itertools as it

# parallel computing for noise analysis on more than one CPU/core
useMP = True
if useMP:
    import multiprocessing as mp
# progress bar for noise analysis (takes a while ...)
import tqdm
# store results from noise analysis
import pickle


def uniquify(path, prefix = ''):
    """
    for a given path/filename, create a unique name by adding
    zero-padded numbers and prefix if needed
    """
    
    filename, extension = os.path.splitext(path)
    counter = 1
    
    while os.path.exists(path):
        path = f'{filename}_{prefix}{counter:04d}{extension}'
        counter += 1
        
    return path


## peak fitting routines -- these calculate the delta for each method
def quadraticDelta(a1, a2, a3):
    return (a3 - a1) / (2 * (2 * a2 - a1 - a3))

def barycentricDelta(a1, a2, a3):
    return (a3 - a1)/(a1 + a2 + a3)

def candanDelta(y1, y2, y3, N):
    cor = np.pi/N
    return np.tan(cor)/cor * jacobsenDelta(y1, y2,y3)

def jacobsenDelta(y1, y2, y3):
    return ((y1-y3)/(2*y2-y1-y3)).real

def macLeodDelta(y1, y2, y3):
    R1 = (y1 * y2.conjugate()).real
    R2 = abs(y2)**2
    R3 = (y3 * y2.conjugate()).real
    
    gamma = (R1-R3)/(2*R2+R1+R3);
    return (math.sqrt(1+8*gamma**2)-1)/(4*gamma)

def jainsDelta(a1, a2, a3):
    if (a1 > a3):
        a = a2/a1
        d = a/(1+a)-1
    else:
        a = a3/a2
        d = a/(1+a)
    return d

def quinnsDelta(y1, y2, y3):
    def tau(x):
        return  0.25 * np.log10(3 * x ** 2 + 6 * x + 1) - np.sqrt(6) / 24 * np.log10((x + 1 - np.sqrt(2 / 3)) / (x + 1 + np.sqrt(2 / 3)))
            
    y2r = y2.real; y2i = y2.imag; y2m =abs(y2)**2;

    ap = (y3.real * y2r + y3.imag * y2i) / y2m;
    dp = -ap/(1-ap);
    am = (y1.real * y2r + y1.imag * y2i) / y2m;
    dm = am/(1-am);
    d = (dp+dm)/2+tau(dp*dp)-tau(dm*dm);
    return d


def formalMethodName(method = None):
    if (type(method) != str):
        return "method must be string"
    
    match method.lower():
        case "maximumpixel":
            return "maximum pixel"
        case "quadratic": # weighted average
            return "Quadratic approximation"
        case "barycentric": # weighted average
            return "Barycentric approximation"
        case "jains":
            return "Jain's method"
        case "jacobsen":
            return "Jacobsen's method"
        case "jacobsenmod":
            return "modified Jacobsen's"
        case "macleod":
            return "MacLeod's method"    
        case "candan":
            return "Candan"
        case "quinns2nd":
            return r"Quinn's $2^\mathrm{nd}$ estimator"
        case "gaussian":
            return "Gaussian fit"
        case _:
            return 'unknown method {method}'

        
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
            d = quadraticDelta(abs(y1),abs(y2),abs(y3))
        case "barycentric": # weighted average
            a1 = abs(y1); a2 = abs(y2); a3=abs(y3)
            d = (a3 - a1) / (a1 + a2 + a3)
        case "jains":
            d = jainsDelta(abs(y1),abs(y2),abs(y3))
        case "jacobsen":
            d = jacobsenDelta(y1, y2, y3)
        case "jacobsenmod":
            d = jacobsenDelta(y1, y2, y3)
            w = abs(d)    # linear weight (0 at integer, 1/2 at half-integ)
            d *= 1-w
            if d < 0:
                d += w*(jacobsenDelta(fdata[peak_index-2], y1, y2) - 1)
            elif d>0:
                d += w*(jacobsenDelta(y2, y3, fdata[peak_index+2]) + 1)
        case "candan":
            d = candanDelta(y1, y2, y3, len(fdata))
        case "macleod":
            d = macLeodDelta(y1, y2, y3)            
        case "quinns2nd":
            d = quinnsDelta(y1,y2,y3)
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

        
def generateColourmap(N=10, c1 = None, c2 = None):
    """Generate a N-level colourmap from Red to Blue
    """
    if c1 is None:
        c1 = np.array([1,0,0])
    if c2 is None:
        c2 = np.array([0,0,1])
        
    if (N==1):
        return [c1]
    
    clrs = []
    for ii in range(N):
        w = ii/(N-1)
        clrs.append(w*c1+(1-w)*c2)
    return clrs


def longHandDFT(data, f=None):
    """ 
    calculate DFT for arbitray frequencies (note the lack of the second
    "F" -- this method is no longer 'fast'!)
    """
    
    if f is None:
        fd = fft(data)
    else:
        if hasattr(data, "__len__") and hasattr(data[0],"__len__"):
            # list of lists (or array)
            Nd = len(data)
        else:
            data = [data]
            Nd = 1
            
        if not hasattr(f, "__len__"):
            f = [f]
            
        fd = np.zeros(shape=(Nd, len(f)), dtype = np.complex64)
        
        N = len(data[0])    # length of original data
        base = -2j*np.pi*np.arange(N)/N # fixed part in calculation of dFT
        
        for ii in range(len(f)):
            expbase = np.exp(base * f[ii])  # frequency/k dependent part of dFT
            for jj in range(Nd):    
                fd[jj][ii] = np.sum(data[jj] * expbase) # dFT (not fast anymore)
                
    return fd

    
def generateData(N = 700, f = 9, phase = 0):
    """ create a data set (pure sine) of length N, frequency f, 
    phase, and optionally noise (rms, gaussian noise)
    f, phase, and noise may be lists, in which case a list of data sets 
    is returned; 
    """
    data = []
    param = []
  
    if hasattr(f, "__len__"):   # probably a list or matrix
        for fval in f:
            d, p = generateData(N, fval, phase)
            data.append(d)
            param.append(p)
        return data, param
      
    if hasattr(phase, "__len__"):
         for phaseval in phase:
             d, p = generateData(N, f, phaseval)
             data.append(d)
             param.append(p)
         return data, param
     
    # if non of the above, calculate single-parameter sine
    data = np.sin(2*np.pi*f/N*np.arange(1, N+1) + phase)   
    # to maintain compatibility with the original matlab test code
           
    return data, (f, phase)


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
    
 
def makeComparisonPlot(fvals, data, methods, title = None):
    """Make a comparison plot of different estimators
    """
    
    f,(ax1, ax2) = plt.subplots(2,1)
    #cval = generateColourmap(len(data))
    
    if isinstance(methods, (str)):  # make a 2D array out of data
        methods = [methods]
        data = data.reshape(len(data),1)
        
    for ii in range(len(methods)):
        ki = np.array(data[:,ii])
        bias = ki - fvals
        ax1.plot(fvals, ki, label=formalMethodName(methods[ii]))
        ax2.plot(fvals, bias)
        
        am = abs(bias)
        mf = am.max()
        avgf = am.mean()
        stdf = am.std()
        mfpos = fvals[np.argmax(am)]
        print(f'{methods[ii]}: maximum bias is {mf:.6f} at position {mfpos:.2f}; std of bias is {bias.std():.6f}')
        print(f'     -- <|bias|> is {avgf:.6f}, std(|bias|) is {stdf:.6f}')
    ax1.set_ylabel('peak position')
    ax1.legend(fontsize=8)
    ax2.set_ylabel('bias')
    ax2.set_xlabel('frequency ($\\Delta f$)')
    
    if title is not None:
        ax1.set_title(title)
    
    return f, (ax1, ax2)


def makeDataPlot(xvals = None, data = None, fvals = None, 
                 fdata = None, fmt='.', xl=None, yl=None):
    
    if hasattr(data, '__len__') and not hasattr(data[0],'__len__'):
        data = [data]   # only one data set -- ok, deal with it!
        
    plotFFT = (fdata is not None)
    
    if plotFFT:
        f, (ax1, ax2) = plt.subplots(2,1, dpi=600)
    else:
        f, ax1 = plt.subplots(1,1, dpi=600)
        ax2 = None
        
    if xvals is None:  
        xvals = inclusiveRange(0,1,N=len(data[0]))
        
    if fvals is None:
        fvals = slice(2,13)
        
    cval = generateColourmap(len(data))
    for ii in range(len(data)):
        ax1.plot(xvals, data[ii], fmt, color = cval[ii])
        if plotFFT:
            ax2.plot(np.arange(fvals.start,fvals.stop, fvals.step),\
                     abs(fdata[ii])[fvals], color = cval[ii])
        
    ax1.set_xlabel('time (s)' if xl is None else xl)
    ax1.set_ylabel('amplitude' if yl is None else yl )
    if plotFFT:
        ax2.set_xlabel('frequency ($\\Delta f$)')
        ax2.set_ylabel("|FFT|")
    
    return f, (ax1, ax2)


def compareNoisePlots(fvals, dataMean, dataStd, param, dataC = None, biasType="max"):
    """compareNoisePlot(dataMean, dataStd, param, dataC, biasType)
        plots rms vs noise (bias/std) and compares different methods
        valid biasType are 'max', 'mean', and 'both'
    """
    Nd, Nr, Nm = dataMean.shape
    doublePlot = (biasType == "both")
    mx = []
    mn = []
    sx = []
    sd = []
    
    rvals = param[1]
    for rr in range(Nr):
        # generate sub-set of data for constant RMS
        maxb = []
        meanb = []
        maxs = []
        means = []
        biasR = np.array([abs(dataMean[:,rr,mm]-fvals) for mm in range(Nm)]).transpose()
        stdR = dataStd[:,rr,:]
        for mm in range(Nm):
            maxb.append(biasR[:,mm].max())
            meanb.append(biasR[:,mm].mean())
            maxs.append(stdR[:,mm].max())
            means.append(stdR[:,mm].mean())
        mx.append(maxb)
        mn.append(meanb)
        sx.append(maxs)
        sd.append(means)
        
    mx = np.array(mx)
    mn = np.array(mn)
    sx = np.array(sx)
    sd = np.array(sd)
    
    fig, ax = plt.subplots(1, 2 if doublePlot else 1)
    colours = generateColourmap(N=Nm)
    markers = it.cycle("o*Dv<PX")
    # position of certain S/N values on the oritinal X axis
    newTickPositions = 1/(np.sqrt(2) * np.array([1/np.sqrt(2), 1, 1.5, 2, 5]))
    def makeSecondXAxis(axOrig, newTicks, label = None):
        """make a secondary X axis with (non-unformly) spaced ticks indicating S/N
        """
        axNew = axOrig.twiny()
        def tickLabel(x):
            V = 1/(np.sqrt(2)*x)
            return [f'{v:.2f}' for v in V]
        axNew.set_xticks(newTicks)
        axNew.set_xticklabels(tickLabel(newTicks))
        axNew.set_xlim(axOrig.get_xlim())
        if label is not None:
            axNew.set_xlabel(label)
        return axNew
    
    if Nr>5: #make a slice
        useSlices = True
        Nstep = int(np.floor(Nr/6))
    else:
        ind =slice(0,Nr)
    for rr in range(Nm):
        if useSlices:
            ind = slice((rr%Nstep)+2, Nr, Nstep)
        method = formalMethodName(param[2][rr])
        marker = next(markers)
        match biasType:
            case "both":
                ax[0].plot(rvals, mx[:,rr], linestyle='-', marker=marker, color=colours[rr], label=method)
                ax[0].plot(rvals, sx[:,rr], linestyle=':', color=colours[rr])
                ax[0].plot(rvals[ind], sx[ind,rr], marker, markeredgecolor=colours[rr])
                ax[1].plot(rvals, mn[:,rr], linestyle='-.', marker=marker, color=colours[rr], label=method)
                ax[1].plot(rvals, sd[:,rr], linestyle=':', color=colours[rr])
                ax[1].plot(rvals[ind], sd[ind,rr], marker, markeredgecolor=colours[rr])
            case "mean":
                ax.plot(rvals, mn[:,rr], linestyle='-', marker=marker, color=colours[rr], label=method)
                ax.plot(rvals, sd[:,rr], linestyle=':', color=colours[rr])
                ax.plot(rvals[ind], sd[ind,rr],  marker, color=colours[rr])
            case _:
                ax.plot(rvals, mx[:,rr], linestyle='-', marker=marker, color=colours[rr], label=method)
                ax.plot(rvals, sx[:,rr], linestyle=':', color=colours[rr])
                ax.plot(rvals[ind], sx[ind,rr], marker, color=colours[rr])
    if doublePlot:
        for ii in range(2):
            ax[ii].set_xlabel('rms noise')
            # ax[ii].set_xlim([0,1.5])
            makeSecondXAxis(ax[ii], newTickPositions, label='S/N')
            if (Nm==1):
                ax[ii].set_title("comparison max/mean bias" if ii==0 else method)
            else:
                #ax[ii].set_title("comparison of methods")
                ax[ii].legend()
                
        ax[0].set_ylabel('maximum -- average bias, : std of bias')
        ax[1].set_ylabel('mean -- average bias, : std of bias')
        
    else:
        ax.set_xlabel('rms noise')
        # ax.set_xlim([0, 1.5])
        makeSecondXAxis(ax, newTickPositions, label='S/N')
        ylbl = ('maximum' if biasType=="max" else 'mean')+' -- average, : std of bias' 
        ax.set_ylabel(ylbl)
        if (Nm == 1):
            ax.set_title(method)
        else:
            #ax.set_title("comparison of methods")
            ax.legend(fontsize='x-small', loc='lower right')
    
    return fig, ax


def noiseBehaviour(rms, noise, unc=None):
    """ noiseBehaviour(rms, noise)
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


def makeNoisePlot(fvals, dataMean, dataStd, param, rind = None, dataC = None):
    """ generate a noise plot for the noise data in dataMean and dataStd, both 
    of which are 3 dimensional arrays [frequency, rms, method]
    values of rms and methods can be found in param (output of the noisy data
                                                     generation function)
    rind gives the indices in the rms axis for which plots are performed
    dataC (if given) is a tuple of skewness and kurtosis.
    """
    Nd, Nr, Nm = dataMean.shape
    f = []
    a = []
    p = []
    
    if (Nd>1) and (Nr>1) and (Nm>1):    # that is too much
        # we have both mutliple rms and multiple methods -> one figure(set) per method
        for mm in range(Nm):
            dM = dataMean[:,:,[mm]]
            dS = dataStd[:,:,[mm]]
            if dataC is not None:
                dSk = dataC[0][:,:,[mm]]
                dKu = dataC[1][:,:,[mm]]
                dC = (dSk, dKu)
            else:
                dC = None
            fig, ax, paramOut = makeNoisePlot(fvals, dM, dS, \
                                 (param[0], param[1], [param[2][mm]]), rind, dC)
            f.extend(fig)
            a.extend(ax)
            p.extend(paramOut)
        
        return f, a, p
    
    
    if (Nr>1):  # plot rms noise vs bias/std
        fig, ax = plt.subplots(1,1)
        f.append(fig)
        a.append(ax)
        mk = []; ms = []; xs =[]
        
        for ii in range(Nr):
            mk.append(abs(dataMean[:,ii,0]-fvals).max())
            ms.append(abs(dataStd[:,ii,0]).mean())
            xs.append(abs(dataStd[:,ii,0]).max())
        method = param[2][0]
        rms = param[1]
        
        ax.plot(rms, mk, 'b-o', label='bias')
        k, kerr = noiseBehaviour(rms, ms)
        print(fr'{method}: $\kappa$ mean = {k:.3e} ± {kerr:.3e}')
        k, kerr = noiseBehaviour(rms, xs)
        print(fr'{method}: rms $\kappa$ max = {k:.3e} ± {kerr:.3e}')
        ax.plot(rms, xs, 'b:*', label=fr'max std ($\kappa$ = {k:.3f})')
        
        ax.set_xlabel('rms noise')
        ax.set_ylabel('maximum bias/standard deviation')
        ax.legend()
        ax.set_title(method)
        
        p.append((0,method))
        
        # make noise vs freqency plots for relevant rms (default: 3)
        if rind is None:
            if Nr<4:
                rind = range(Nr)
            else:
                rind= [0, int(np.floor(Nr/3)), int(np.floor(2/3*Nr)), Nr-1]
        for r in rind:
            dM = dataMean[:,[r],:]
            dS = dataStd[:,[r],:]
            dC = (dataC[0][:,[r],:], dataC[1][:,[r],:])
            fig, ax, paramOut = makeNoisePlot(fvals, dM, dS, \
                          (param[0], [param[1][r]], param[2]), None, dC)
            f.extend(fig)
            a.extend(ax)
            p.extend(paramOut)
        
    else:
        fig, ax = plt.subplots(Nm, 1 if dataC is None else 2, dpi=600)
        f.append(fig)
        a.append(ax)
        if Nm==1:
            ax = [ax]
    
        for ii, method in enumerate(param[2]):
           
            if dataC is not None:
                ax1 = ax[ii][0]
            else:
                ax1 = ax[ii]
                
            rms = param[1][0]
            p.append((rms, method))
          
            # in this branch we only have one rms value (-> second index 0)
            dM = dataMean[:,0,ii]
            dS = dataStd[:,0,ii]
            
            bias = dM - fvals
            ax1.plot(fvals, bias, 'b.')
            #ax1.plot(fvals, bias+dS, 'b:', fvals, bias-dS, 'b:')
            #ax1.plot(fvals, dS, 'b:', fvals, dS, 'b:')
            ax1.plot(fvals, bias+2*dS, 'b:', fvals, bias-2*dS, 'b:')
            ax1.set_xlabel('input frequency ($\\Delta f$)')
            ax1.set_ylabel('bias')
            fmethod = formalMethodName(method)
            ax1.set_title(f'{fmethod:s} - (rms={rms:.2f})' if (dataC is None) else f'{fmethod:s}')
            #ax1.legend(['average', 'std'])
            #ax1.legend(['average', '95% confidence interval'])
            print(f'{method:s} with rms {rms:.2f}: maximum bias is {abs(dM-fvals).max():.5f}; max std of bias is {dS.max():.5f}')
            
            if dataC is not None:
                ax2 = ax[ii][1]
                dSkew = dataC[0][:,0,ii]
                dKurt = dataC[1][:,0,ii]
                ax2.plot(fvals, dSkew, 'b-', label='skewness')
                ax2.plot(fvals, dKurt, 'r-', label='kurtosis')
                ax2.set_xlabel('input frequency ($\\Delta f$)')
                #ax2.set_ylabel('skewness/kurtosis')
                ax2.yaxis.set_label_position('right')
                ax2.yaxis.tick_right()
                ax2.set_title(f'(rms={rms:.2f})')
                ax2.legend()
            
        else:
            pass # to be written -- plogenerateDatat of rms vs mean/max error?
        
    return f, a, p


def makeKappaPlot(fC, fDelta, stdData, param):
    """ generate a kappa plot for the kappa data in kappa, a 3 dimensional 
    array [frequency, rms, method]   
    values of rms and methods can be found in param (output of the noisy data
                                                     generation function)
    """
    Nf, Nr, Nm = stdData.shape
    Nd = len(fDelta)
    Nc = len(fC)
    f = []
    a = []
    p = []
    c = []
    
    if (Nm>1):    
        # we have multiple methods -> one figure per method
        for mm in range(Nm):
            dS = stdData[:,:,[mm]]
                        
            fig, ax, paramOut, cM = makeKappaPlot(fC, fDelta, dS, \
                                 (param[0], param[1], [param[2][mm]]))
            f.extend(fig)
            a.extend(ax)
            p.extend(paramOut)
            c.append(cM)
        return f, a, p, c
    
    fig, ax = plt.subplots(1,1)
    f.append(fig)
    a.append(ax)
    
    method = param[2][0]
    rmsval = param[1]
    
    kappa = np.zeros(shape = (Nc, Nd*Nr))
    for rr in range(Nr):
        ## TODO: try a + b x?
        stdData /= rmsval[rr]    # calculate kappa estimates (std = kappa * rms)
        
    for ff in range(Nc):
        # all kappa values for a central frequency
        kappa[ff, :] = np.reshape(stdData[Nd*ff:Nd*(ff+1),:,0], (Nd*Nr,))
        
    mk = kappa.mean(axis=1)
    sk = kappa.std(axis=1)/np.sqrt(Nd*Nr)   # rough estimation
    
    cM = np.zeros(shape=(Nc, Nc))
    vk = sk**2; # variance
    for ii in range(Nc-1):
        for jj in range(ii+1, Nc):
            sij = np.sqrt(vk[ii]+vk[jj])
            delta = np.abs(mk[ii]-mk[jj])/sij
            cM[ii,jj] = delta # delta >2 indicates significant difference
            cM[jj,ii] = delta
            
    merr = np.array([mk-kappa.min(axis=1), kappa.max(axis=1)-mk])
    
    ax.errorbar(fC, mk, yerr=merr, fmt='b-o', label=r'$\kappa_\mathrm{mean}$')
   
    ax.set_xlabel(r'centre frequency ($\Delta f$)')
    ax.set_ylabel(r'$\kappa_\mathrm{mean}$')
    #ax.legend()
    ax.set_title(method)
    
    p.append((0,method))
    
    print(f'kappa values for method: {method}')
    
    for ii in range(Nc):
        print(f'  {fC[ii]:.0f}:\t {mk[ii]:.3e} ± {sk[ii]:.3e}')
    # this should be the same as calculating over all kappa?   
    print(f' \t average: {mk.mean():.3e} ± {sk.mean()/np.sqrt(Nc):.3e}')   
        
    return f, a, p, cM


def inclusiveRange(start = 0, end = 1, step = None, N = 100):
    """generate linear ranges which include the final value
    """
    if step is None:
        step = (end - start)/(N-1)
    data = list(np.arange(start, end, step))
    # test whether end value is near-integer steps away from start
    intOff = (end-start)/step
    intOff -= round(intOff)
    if abs(intOff)<1e-10:
        data.append(end)
    return data


def saveFigure(fig, filename):

    if directory is None:
        d = ''
    else:
        d = directory + '/'
        
    fig.savefig(d+filename, dpi=300, bbox_inches='tight')


## start script / figure generation 

saveFigures = True
directory = None

# be aware calculation of noise properties can take significant time!
generateNoise = False
noiseAnalysis = False


# be aware calculation of kappa can take significant time!
generateKappaData = False
kappaAnalysis = False

#NdataPoints = 700 # roughly number of pixels in raw wavelength-spectrum 
NdataPoints = 2048 # resampled signal in k-space
centralFrequency = 7    # based on Pegah et al. (5 µm beads)
#centralFrequency = 50  # check for different frequency
noiseSamples = 1000


#directory = f'simResults_{NdataPoints}_{centralFrequency:.1f}_{noiseSamples:d}'
directory = 'presentation'
if not os.path.exists(directory):
    os.makedirs(directory)
    
fRange =  inclusiveRange(centralFrequency-0.6,centralFrequency+0.6,N=1001)
noiseRange = inclusiveRange(0.02, 0.5, 0.02)

## figure DFT vs FFT
freq = [centralFrequency+delta for delta in [-0.4,0,0.4]]
dfrange = inclusiveRange(centralFrequency-3,centralFrequency+3,0.01)
ffrange = np.arange(min(dfrange), max(dfrange)+1)
fslice=slice(400,810,200)
colours='rgb'
data, param = generateData(NdataPoints, f=freq)
dfdata = longHandDFT(data, dfrange)
ffdata = fft(data)

fig,ax = plt.subplots(1)

for ii in range(len(dfdata)):
    dd = abs(dfdata)
    ax.plot(dfrange, dd[ii], colours[ii]+'-', label=f'dFT@{freq[ii]:.1f}')
    #ax.plot(dfrange[fslice], dd[ii][fslice], colours[ii]+'o', label='')            
    ax.plot(ffrange, abs(ffdata[ii,(centralFrequency-3):(centralFrequency+4)]), colours[ii]+'*')#, label='FFT')
fig.legend()
ax.set_xlabel('input frequency ($\\Delta f$)')
ax.set_ylabel('magnitude fourier transform')

if saveFigures:
    saveFigure(fig, 'fig_DFT_FFT.png')


## compare methods
data, param = generateData(N=NdataPoints, f = fRange)
fdata = fft(data)

makeDataPlot(data = data[::100], fdata = fdata[::100], fmt='-')

Ngauss = 5
k = np.array(Gauss_fit(fdata, N=Ngauss))

fig, ax = makeComparisonPlot(fRange, k[:,1], "Gaussian")

if saveFigures:
    saveFigure(fig, 'gaussian.png')
    
fig1,ax1=plt.subplots(1)
ax1.plot(fRange, k[:,0], label="Gaussian fit")
ax1.plot(fRange, k[:,3], label=f'RMS FFT (N={Ngauss})')
ax1.set_xlabel('input frequency ($\\Delta f$)')
ax1.set_ylabel('amplitude of component (AU)')
ax1.legend()

fig2,ax2=plt.subplots(1)
ax2.plot(fRange, k[:,2], label="fit width")
ax2.set_xlabel('input frequency ($\\Delta f$)')
ax2.set_ylabel('width of Gaussian $\\sigma$')


# comparision plots with a  3 methods each
methods = ["Quadratic", "Barycentric", "Gaussian"]
k = np.array(FFT_peakFit(fdata, methods))
fig,_ = makeComparisonPlot(fRange, k, methods)
if saveFigures:
    saveFigure(fig,"stdFits.png")


methods = ["Jains", "MacLeod"] # , "Candan"]
k = np.array(FFT_peakFit(fdata, "Jains"))
fig0,_ = makeComparisonPlot(fRange, k, "Jains")
k = np.array(FFT_peakFit(fdata, "MacLeod"))
fig1,_ = makeComparisonPlot(fRange, k, "MacLeod")
if saveFigures:
    saveFigure(fig0, "Jain.png")
    saveFigure(fig1, "MacLeod.png")
    


methods = ["Jacobsen", "JacobsenMod","Quinns2nd"]
k = np.array(FFT_peakFit(fdata, methods))
fig, _ = makeComparisonPlot(fRange, k, methods)
if saveFigures:
    saveFigure(fig,"QuinnJacobsen.png")


if noiseAnalysis:
    # test (some) methods for noise
    rmsvals = np.arange(0.05, 1.2, 0.05);
    methods = ['Quadratic', 'MacLeod', 'Quinns2nd', 'Jacobsen', 'JacobsenMod']    
    ## generate and save data
    if generateNoise:
        start_time = tm.time()
        print('starting data generation (noise analysis)')
        kk, paramF = noisyFit(data, rms = rmsvals, Nrepeat = noiseSamples, method = methods)
        end_time = tm.time()
        delta = end_time - start_time
        str_time = f'{(delta%60):.2f} s'
        if (delta >= 60):
            delta = int(np.floor(delta/60))
            str_time = f'{(delta%60):d} m ' + str_time
            if delta >= 60:
                delta = int(np.floor(delta/60))
                str_time = f'{delta:d} h ' + str_time
        print(f'data generation (noise analysis) finished in {str_time:s}.')
        km, ks, kskew, kkurt = kk
        if directory is None:
            directory = '.'
        for ii, k in enumerate((km, ks, kskew, kkurt)):
            np.save(f'{directory}/moment{ii:d}.dat', k)
        with open(f'{directory}/param.dat', 'wb') as f:
            pickle.dump(paramF, f)
        generateNoise = False # only once per session
    else:
        ## or read data from last run
        km = np.load('moment0.dat.npy')
        ks = np.load('moment1.dat.npy')
        kskew = np.load('moment2.dat.npy')
        kkurt = np.load('moment3.dat.npy')
        with open('param.dat', 'rb') as f:
            paramF = pickle.load(f)
        
    #rind = None # chose automagically (up to 4)
    rind = [0, 3, 6, 9, 14, 22]
    fig, ax, param = makeNoisePlot(fRange, km, ks, paramF, rind, (kskew, kkurt))
    
    if saveFigures:
        for ii in range(len(fig)):
            if hasattr(fig[ii],'__len__'):
                for jj in range(len(fig[ii])):
                    pp = param[ii][jj]
                    rms = pp[0]
                    method = pp[1]
                    saveFigure(fig[ii][jj], f'noise_{method:s}_{rms:.3f}.png')    
            else:
                rms = param[ii][0]
                method = param[ii][1]
                saveFigure(fig[ii], f'noise_{method:s}_{rms:.3f}.png')
    
    fig, ax = compareNoisePlots(fRange, km, ks, paramF, biasType="max")
    ax.set_yscale("log")
    if saveFigures:
        saveFigure(fig, 'noiseComparison.png') 
    fig, ax = compareNoisePlots(fRange, km, ks, paramF, biasType="mean")
    ax.set_yscale("log")
    if saveFigures:
        saveFigure(fig, 'noiseComparisonMean.png')
    

if kappaAnalysis:

    rmsvals = [0.3535]  # S/N ~ 2
    methods = ['Quadratic', 'MacLeod', 'Quinns2nd', 'Jacobsen', 'JacobsenMod']    
    
    #deltaF = [ -0.25, -0,1, 0, 0.1, 0.25 ]
    Nd = 40
    deltaF = [(ii/Nd-1/2) for ii in range(Nd+1)]
    centerF = [7, 50, 100, 150, 200, 250, 300, 350, 400]
    
    fvals = [ff + dd for ff in centerF for dd in deltaF]
    
    data, param = generateData(N=NdataPoints, f = fvals)
    fdata = fft(data)
    
    if generateKappaData:
        start_time = tm.time()
        print('starting data generation (kappa analysis)')
        kk, paramK = noisyFit(data, rms = rmsvals, Nrepeat = noiseSamples, method = methods)
        end_time = tm.time()
        delta = end_time - start_time
        str_time = f'{(delta%60):.2f} s'
        if (delta >= 60):
            delta = int(np.floor(delta/60))
            str_time = f'{(delta%60):d} m ' + str_time
            if delta >= 60:
                delta = int(np.floor(delta/60))
                str_time = f'{delta:d} h ' + str_time
        print(f'data generation (kappa analysis) finished in {str_time:s}.')
        km, ks, kskew, kkurt = kk
        if directory is None:
            directory = '.'
        for ii, k in enumerate((km, ks, kskew, kkurt)):
            np.save(f'{directory}/kappa_moment{ii:d}.dat', k)
        #paramK.append(centerF)
        #paramK.append(deltaF)
        with open(f'{directory}/kappa_param.dat', 'wb') as f:
            pickle.dump(paramK, f)
        generateKappaData = False # only once per session
        
    else:
        ## or read data from last run
        km = np.load('kappa_moment0.dat.npy')
        ks = np.load('kappa_moment1.dat.npy')
        kskew = np.load('kappa_moment2.dat.npy')
        kkurt = np.load('kappa_moment3.dat.npy')
        with open('kappa_param.dat', 'rb') as f:
            paramK = pickle.load(f)

    fig, ax, param, cM = makeKappaPlot(centerF, deltaF, ks, paramK)
    if saveFigures:
        for ii in range(len(fig)):
            if hasattr(fig[ii],'__len__'):
                for jj in range(len(fig[ii])):
                    pp = param[ii][jj]
                    method = pp[1]
                    saveFigure(fig[ii][jj], f'kappa_{method:s}.png')    
            else:
                method = param[ii][1]
                saveFigure(fig[ii], f'kappa_{method:s}.png')
                
                

## EOF