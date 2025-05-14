import numpy as np
import time as tm
import pickle

import lib.fitting as fitting
import lib.plotting as plotting

def noiseAnalysis(data, noiseSamples, fRange, rmsvals, methods, generateNoise=True, directory=None, saveFigures=False, useMP=False):
    """
    A function to analyse noise and it's inpacts on the methods we use to find the peak

    Parameters
    ----------
    data : NDArray[Any] | list
        The raw data (excluding noise) of the sinusoid

    noiseSamples : int
        Determines how many times we create new data with unique noise

    fRange : list[floating[Any]]
        The range of frequencies the sinusoid moves over

    rmsvals : 1DArray[Floating[Any]]
        List of values for amplitudes of noise

    methods : list[str]
        List of methods we use

    generateNoise : bool
        Whether to generate new noise (default True)

    directory : str
        The directory you want to save figures to (default None)

    saveFigures : bool
        Whether or not you want to save the figures (default False)

    useMP : bool
        Use multi-processing

    Returns
    -------
    It saves figures in a directory of the methods used on a signal with noise for each method over a range of different noise amplitudes
    """
    ## generate and save data
    if generateNoise:
        start_time = tm.time()
        print('starting data generation (noise analysis)')
        kk, paramF = fitting.noisyFit(data, rms = rmsvals, Nrepeat = noiseSamples, method = methods, useMP=useMP)
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
    fig, ax, param = plotting.makeNoisePlot(fRange, km, ks, paramF, noiseSamples,rind, (kskew, kkurt))

    if saveFigures:
        for ii in range(len(fig)):
            if hasattr(fig[ii],'__len__'):
                for jj in range(len(fig[ii])):
                    pp = param[ii][jj]
                    rms = pp[0]
                    method = pp[1]
                    plotting.saveFigure(fig[ii][jj], f'noise_{method:s}_{rms:.3f}.png', directory)    
            else:
                rms = param[ii][0]
                method = param[ii][1]
                plotting.saveFigure(fig[ii], f'noise_{method:s}_{rms:.3f}.png', directory)

    fig, ax = plotting.compareNoisePlots(fRange, km, ks, paramF, biasType="max")
    ax.set_yscale("log")
    if saveFigures:
        plotting.saveFigure(fig, 'noiseComparison.png', directory) 
    fig, ax = plotting.compareNoisePlots(fRange, km, ks, paramF, biasType="mean")
    ax.set_yscale("log")
    if saveFigures:
        plotting.saveFigure(fig, 'noiseComparisonMean.png', directory)

def kappaAnalysis(data, noiseSamples, centerF, deltaF, rmsvals, methods, generateKappaData=True, directory=None, saveFigures=False, useMP=False):
    """
    A function to analyse kappa and it's inpacts on the methods we use to find the peak

    Parameters
    ----------
    data : NDArray[Any] | list
        The raw data (excluding noise) of the sinusoid

    noiseSamples : int
        Determines how many times we create new data with unique noise

    centerF: list[int]
        Center frequency

    deltaF : list[float]
        Bandwidth around center frequency

    rmsvals : 1DArray[Floating[Any]]
        List of values for amplitudes of noise

    methods : list[str]
        List of methods we use

    generateNoise : bool
        Whether to generate new noise (default True)

    directory : str
        The directory you want to save figures to (default None)

    saveFigures : bool
        Whether or not you want to save the figures (default False)

    useMP : bool
        Use multi-processing

    Returns
    -------
    It saves figures in a directory of the methods used on a signal with noise for each method over a range of different noise amplitudes over a frequency bandwidth
    """
    if generateKappaData:
        start_time = tm.time()
        print('starting data generation (kappa analysis)')
        kk, paramK = fitting.noisyFit(data, rms = rmsvals, Nrepeat = noiseSamples, method = methods, useMP=useMP)
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

    fig, ax, param, cM = plotting.makeKappaPlot(centerF, deltaF, ks, paramK)
    if saveFigures:
        for ii in range(len(fig)):
            if hasattr(fig[ii],'__len__'):
                for jj in range(len(fig[ii])):
                    pp = param[ii][jj]
                    method = pp[1]
                    plotting.saveFigure(fig[ii][jj], f'kappa_{method:s}.png', directory)    
            else:
                method = param[ii][1]
                plotting.saveFigure(fig[ii], f'kappa_{method:s}.png', directory)
