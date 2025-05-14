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
from scipy.fftpack import fft
import time as tm

# store results from noise analysis
import pickle

import lib.generate as gen
import lib.helper_functions as helper
import lib.plotting as plotting
import lib.fitting as fitting

## start script / figure generation 

saveFigures = True
directory = None

# be aware calculation of noise properties can take significant time!
generateNoise = True
noiseAnalysis = True


# be aware calculation of kappa can take significant time!
generateKappaData = False
kappaAnalysis = False

#NdataPoints = 700 # roughly number of pixels in raw wavelength-spectrum 
NdataPoints = 2048 # resampled signal in k-space
centralFrequency = 7    # based on Pegah et al. (5 µm beads)
#centralFrequency = 50  # check for different frequency
noiseSamples = 10
    
fRange = helper.inclusiveRange(centralFrequency-0.6,centralFrequency+0.6,N=1001)
noiseRange = helper.inclusiveRange(0.02, 0.5, 0.02)

## figure DFT vs FFT
freq = [centralFrequency+delta for delta in [-0.4,0,0.4]]
dfrange = helper.inclusiveRange(centralFrequency-3,centralFrequency+3,0.01)
ffrange = np.arange(min(dfrange), max(dfrange)+1)
fslice=slice(400,810,200)
colours='rgb'
data, param = gen.generateData(NdataPoints, f=freq)
dfdata = helper.longHandDFT(data, dfrange)
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
    plotting.saveFigure(fig, 'fig_DFT_FFT.png')


## compare methods
data, param = gen.generateData(N=NdataPoints, f = fRange)
fdata = fft(data)

plotting.makeDataPlot(data = data[::100], fdata = fdata[::100], fmt='-')

Ngauss = 5
k = np.array(fitting.Gauss_fit(fdata, N=Ngauss))

fig, ax = plotting.makeComparisonPlot(fRange, k[:,1], "Gaussian")

if saveFigures:
    plotting.saveFigure(fig, 'gaussian.png')
    
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
k = np.array(fitting.FFT_peakFit(fdata, methods))
fig,_ = plotting.makeComparisonPlot(fRange, k, methods)
if saveFigures:
    plotting.saveFigure(fig,"stdFits.png")


methods = ["Jains", "MacLeod"] # , "Candan"]
k = np.array(fitting.FFT_peakFit(fdata, "Jains"))
fig0,_ = plotting.makeComparisonPlot(fRange, k, "Jains")
k = np.array(fitting.FFT_peakFit(fdata, "MacLeod"))
fig1,_ = plotting.makeComparisonPlot(fRange, k, "MacLeod")
if saveFigures:
    plotting.saveFigure(fig0, "Jain.png")
    plotting.saveFigure(fig1, "MacLeod.png")
    


methods = ["Jacobsen", "JacobsenMod","Quinns2nd"]
k = np.array(fitting.FFT_peakFit(fdata, methods))
fig, _ = plotting.makeComparisonPlot(fRange, k, methods)
if saveFigures:
    plotting.saveFigure(fig,"QuinnJacobsen.png")


if noiseAnalysis:
    # test (some) methods for noise
    rmsvals = np.arange(0.05, 1.2, 0.05);
    methods = ['Quadratic', 'MacLeod', 'Quinns2nd', 'Jacobsen', 'JacobsenMod']    
    ## generate and save data
    if generateNoise:
        start_time = tm.time()
        print('starting data generation (noise analysis)')
        kk, paramF = fitting.noisyFit(data, rms = rmsvals, Nrepeat = noiseSamples, method = methods)
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
                    plotting.saveFigure(fig[ii][jj], f'noise_{method:s}_{rms:.3f}.png')    
            else:
                rms = param[ii][0]
                method = param[ii][1]
                plotting.saveFigure(fig[ii], f'noise_{method:s}_{rms:.3f}.png')
    
    fig, ax = plotting.compareNoisePlots(fRange, km, ks, paramF, biasType="max")
    ax.set_yscale("log")
    if saveFigures:
        plotting.saveFigure(fig, 'noiseComparison.png') 
    fig, ax = plotting.compareNoisePlots(fRange, km, ks, paramF, biasType="mean")
    ax.set_yscale("log")
    if saveFigures:
        plotting.saveFigure(fig, 'noiseComparisonMean.png')
    

if kappaAnalysis:

    rmsvals = [0.3535]  # S/N ~ 2
    methods = ['Quadratic', 'MacLeod', 'Quinns2nd', 'Jacobsen', 'JacobsenMod']    
    
    #deltaF = [ -0.25, -0,1, 0, 0.1, 0.25 ]
    Nd = 40
    deltaF = [(ii/Nd-1/2) for ii in range(Nd+1)]
    centerF = [7, 50, 100, 150, 200, 250, 300, 350, 400]
    
    fvals = [ff + dd for ff in centerF for dd in deltaF]
    
    data, param = gen.generateData(N=NdataPoints, f = fvals)
    fdata = fft(data)
    
    if generateKappaData:
        start_time = tm.time()
        print('starting data generation (kappa analysis)')
        kk, paramK = fitting.noisyFit(data, rms = rmsvals, Nrepeat = noiseSamples, method = methods)
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
                    plotting.saveFigure(fig[ii][jj], f'kappa_{method:s}.png')    
            else:
                method = param[ii][1]
                plotting.saveFigure(fig[ii], f'kappa_{method:s}.png')
                
                

## EOF