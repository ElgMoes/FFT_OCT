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
import matplotlib.pyplot as plt
import pickle

import lib.generate as gen
import lib.helper_functions as helper

import plotter
import analysis

#%% start script / figure generation

poisson_offset = 500
poisson_modulation = 50

frequencies = 1001

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
noiseSamples = 1000
    
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
        N_data = frequencies

        used_parameters = (N_data, methods)
        iterations = frequencies * noiseSamples * N_methods

        with tqdm(total = iterations) as pbar:
            for f in range(len(fRange)):
                data[f] = panalyse.singleNoise(N_methods, noiseSamples, NdataPoints, fRange[f], methods, poisson_offset, poisson_modulation, pbar)
        # initializing arrays to store statistical values
        # mean and standard deviation
        kmean = np.zeros(shape=(N_data, N_methods))
        kstd = np.zeros(shape=(N_data, N_methods))
           
        # Skewness and kurtosis
        kskewness = np.zeros(shape=(N_data, N_methods))
        kkurtosis = np.zeros(shape=(N_data, N_methods))


        for Nd in range(N_data):
            data_per_frequency = data[Nd]
            kmean[Nd, :] = [ data_per_frequency[i].mean for i in range(len(data_per_frequency)) ]
            kstd[Nd, :] = np.sqrt([ data_per_frequency[i].variance for i in range(len(data_per_frequency)) ])
            kskewness[Nd, :] = [ data_per_frequency[i].skewness for i in range(len(data_per_frequency)) ]
            kkurtosis[Nd, :] = [ data_per_frequency[i].kurtosis for i in range(len(data_per_frequency)) ]

        directory = "noise_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if directory is None:
            directory = '.'
        for ii, k in enumerate((kmean, kstd, kskewness, kkurtosis)):
            names = ['kmean', 'kstd', 'kskewness', 'kkurtosis']
            np.save(f'{directory}/noise_{names[ii]}.dat', k)

        with open(f'{directory}/param.dat', 'wb') as f:
            pickle.dump(used_parameters, f)
        generateNoise = False # only once per session
    else:
        ## or read data from last run
        kmean = np.load(f'{directory}/noise_kmean.dat.npy')
        kstd = np.load(f'{directory}/noise_kstd.dat.npy')
        kskewness = np.load(f'{directory}/noise_kskewness.dat.npy')
        kkurtosis = np.load(f'{directory}/noise_kkurtosis.dat.npy')
        with open(f'{directory}/param.dat', 'rb') as f:
            used_parameters = pickle.load(f)

    directory = "noise_plots"
    if not os.path.exists(directory):
        os.makedirs(directory)

    #plotting
    kmean_bias = np.zeros(shape=(N_data, N_methods))
    for method in range(len(methods)):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        kmean_bias[:, method] = kmean[:, method] - fRange
        ax1.plot(fRange, kmean_bias[:, method], label=f"Mean bias {methods[method]}")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Bias")
        ax1.set_title(f"Bias of {methods[method]} per frequency")
        ax1.fill_between(fRange, kmean_bias[:, method] - kstd[:, method], kmean_bias[:, method] + kstd[:, method], alpha=0.3)

        ax2.plot(fRange, kskewness[:, method], label=f"Skewness {methods[method]}", color="green")
        ax2.plot(fRange, kkurtosis[:, method], label=f"Kurtosis {methods[method]}", color="red")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_title(f"Skewnes and kurtosis of {methods[method]}")

        fig.legend()
        fig.tight_layout()
        fig.savefig(f"{directory}/{methods[method]}.png")
    

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