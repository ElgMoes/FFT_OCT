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
import numpy as np
from scipy.fftpack import fft
import os
from tqdm import tqdm


import lib.generate as gen
import lib.helper_functions as helper

import plotter
import analysis


#%% start script / figure generation

poisson_offset = 500
poisson_modulation = 50

frequencies = 101

saveFigures = True
directory = 'presentation'
if not os.path.exists(directory):
    os.makedirs(directory)

# be aware calculation of noise properties can take significant time!
noiseAnalysis = True
generateNoise = True

# be aware calculation of kappa can take significant time!
kappaAnalysis = False
generateKappaData = False

NdataPoints = 2048 # resampled signal in k-space
centralFrequency = 7    # based on Pegah et al. (5 µm beads)
noiseSamples = 10
    
fRange = helper.inclusiveRange(centralFrequency-0.6,centralFrequency+0.6,N=frequencies)
noiseRange = helper.inclusiveRange(0.02, 0.5, 0.02)

useMP = False

#%%
plotter.plotDFT_FFT(centralFrequency, NdataPoints, directory, saveFigures)

"compare methods"
# Generate data
data, param = gen.generateData(N=NdataPoints, f = fRange)
fdata = fft(data-np.mean(data, axis=1, keepdims=True))

plotter.comparisonGaussian(data, fdata, fRange, directory, saveFigures)

# comparision plots with a  3 methods each
methods = ["Quadratic", "Barycentric", "Gaussian"]
plotter.compareMethods(fdata, fRange, methods, saveFigures, "stdFits", directory)

methods = ["MacLeod"] # , "Candan", "Jains"]
plotter.compareMethods(fdata, fRange, methods, saveFigures, "Jains_MacLeod", directory)
    
methods = ["Jacobsen", "JacobsenMod"]#,"Quinns2nd"]
plotter.compareMethods(fdata, fRange, methods, saveFigures, "QuinnJacobsen", directory)

agenerateNoise = False
#%%
if noiseAnalysis:
    import poisson_analysis as panalyse
    methods = ['Quadratic', 'MacLeod', 'Quinns2nd', 'Jacobsen', 'JacobsenMod']
    N_methods = len(methods)

    if generateNoise:
        N_data = 1001
        for f in tqdm(range(len(fRange))):
            data[f] = panalyse.singleNoise(N_methods, noiseSamples, NdataPoints, fRange[f], methods, poisson_offset, poisson_modulation)

        # initializing arrays to store statistical values
        # mean and standard deviation
        kmean = np.zeros(shape=(N_data, N_methods))
        kstd = np.zeros(shape=(N_data, N_methods))
           
        # Skewness and kurtosis
        kskewness = np.zeros(shape=(N_data, N_methods))
        kkurtosis = np.zeros(shape=(N_data, N_methods))

        for method in range(N_methods):
            for Nd in range(N_data):
                kmean[Nd, N_methods] = [ data[method][i].mean for i in range(len(data)) ]
                kstd[Nd, N_methods] = np.sqrt([ data[i].variance for i in range(len(data)) ])
                kskewness[Nd, N_methods] = [ data[i].skewness for i in range(len(data)) ]
                kkurtosis[Nd, N_methods] = [ data[i].kurtosis for i in range(len(data)) ]

    rmsvals = [0.1, 0.2, 0.5]

    #analysis.noiseAnalysis(data, noiseSamples, fRange, rmsvals, methods, generateNoise, directory, saveFigures, useMP)
    

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

    analysis.kappaAnalysis(data, noiseSamples, centerF, deltaF, rmsvals, methods, generateKappaData, directory, saveFigures, useMP)