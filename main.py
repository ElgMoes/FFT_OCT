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


import lib.generate as gen
import lib.helper_functions as helper

import plotter
import analysis


#%% start script / figure generation 
saveFigures = True
directory = 'presentation'
if not os.path.exists(directory):
    os.makedirs(directory)

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

useMP = True

#%%
plotter.plotDFT_FFT(centralFrequency, NdataPoints, directory, saveFigures)

"compare methods"
# Generate data
data, param = gen.generateData(N=NdataPoints, f = fRange)
fdata = fft(data)

plotter.comparisonGaussian(data, fdata, fRange, directory, saveFigures)

# comparision plots with a  3 methods each
methods = ["Quadratic", "Barycentric", "Gaussian"]
plotter.compareMethods(fdata, fRange, methods, saveFigures, "stdFits", directory)

methods = ["Jains", "MacLeod"] # , "Candan"]
plotter.compareMethods(fdata, fRange, methods, saveFigures, "Jains_MacLeod", directory)
    
methods = ["Jacobsen", "JacobsenMod","Quinns2nd"]
plotter.compareMethods(fdata, fRange, methods, saveFigures, "QuinnJacobsen", directory)


#%%
if noiseAnalysis:
    # test (some) methods for noise
    data, param = gen.generateData(N=NdataPoints, f = fRange)
    
    rmsvals = np.arange(0.05, 1.2, 0.05);
    methods = ['Quadratic', 'MacLeod', 'Quinns2nd', 'Jacobsen', 'JacobsenMod']

    analysis.noiseAnalysis(data, noiseSamples, fRange, rmsvals, methods, generateNoise, directory, saveFigures, useMP)
    

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