#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:47:10 2024

@author: Gerhard Blab

routines for FFT peak fitting; 
contributions by Miriam Voots, Bram Haasnoot, and Saban Caliscan
example script for arXiv 
"""
def main():
    # general use packages & plotting
    import numpy as np
    from scipy.fftpack import fft
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import pickle
    import gc

    import lib.generate as gen
    import lib.helper_functions as helper

    import plotter

#%% start script / figure generation
    def print_startup_info():
        print("="*50)
        print("Starting analysis with parameters:")
        print(f"working directory   : {os.getcwd()}")
        print(f"poisson offset      : {poisson_offset}")
        print(f"poisson modulation  : {poisson_modulation}")
        print(f"data points         : {NdataPoints}")
        print(f"central frequency   : {centralFrequency}")
        print(f"frequency difference: {frequencyDiv}")
        print(f"frequencies         : {frequencies}")
        print(f"save figures        : {saveFigures}")
        print(f"noise analysis      : {noiseAnalysis}")
        print(f"generate noise      : {generateNoise}")
        print(f"kappa analysis      : {kappaAnalysis}")
        print(f"generate kappa data : {generateKappaData}")
        print(f"noise samples       : {noiseSamples}")
        print(f"Use multiprocessing : {useMP}")
        print("="*50)

    poisson_offset = 500
    poisson_modulation = 50

    frequencies = 1001

    saveFigures = True

    # be aware calculation of noise properties can take significant time!
    noiseAnalysis = True
    generateNoise = True

    # be aware calculation of kappa can take significant time!
    kappaAnalysis = False
    generateKappaData = False

    NdataPoints = 2048 # resampled signal in k-space
    centralFrequency = 7    # based on Pegah et al. (5 µm beads)
    frequencyDiv = 0.6
    noiseSamples = 100
        
    fRange = helper.inclusiveRange(centralFrequency-frequencyDiv,centralFrequency+frequencyDiv,N=frequencies)
    noiseRange = helper.inclusiveRange(0.02, 0.5, 0.02)

    useMP = True
    debug = False

    print_startup_info()

    directory = 'presentation'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")

    #%%
    print("plotting DFT vs FFT")
    plotter.plotDFT_FFT(centralFrequency, NdataPoints, directory, saveFigures)

    "compare methods"
    # Generate data
    data, param = gen.generateData(N=NdataPoints, f = fRange)
    fdata = fft(data-np.mean(data, axis=1, keepdims=True))

    print("comparing: Gaussian")
    plotter.comparisonGaussian(data, fdata, fRange, directory, saveFigures)

    # comparision plots with a  3 methods each
    methods = ["Quadratic", "Barycentric", "Gaussian"]
    print(f"comparing: {', '.join(methods)}")
    plotter.compareMethods(fdata, fRange, methods, saveFigures, "stdFits", directory)

    methods = ["MacLeod"] # , "Candan", "Jains"]
    print(f"comparing: {', '.join(methods)}")
    plotter.compareMethods(fdata, fRange, methods, saveFigures, "Jains_MacLeod", directory)
        
    methods = ["Jacobsen", "JacobsenMod"]#,"Quinns2nd"]
    print(f"comparing: {', '.join(methods)}")
    plotter.compareMethods(fdata, fRange, methods, saveFigures, "QuinnJacobsen", directory)

    #%%
    
    if noiseAnalysis:
        import noiseAnalysis.poissonAnalysis as panalyse
        import noiseAnalysis.gaussAnalysis as ganalyse
        methods = ['Quadratic', 'MacLeod', 'Quinns2nd', 'Jacobsen', 'JacobsenMod']
        N_methods = len(methods)
        N_data = frequencies

        directory = "noise_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")
        
        pmean, pstd, pskewness, pkurtosis = panalyse.poissonAnalysis(NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, fRange, poisson_offset, poisson_modulation, noiseAnalysis, directory, useMP, debug)

        gmean, gstd, gskewness, gkurtosis = ganalyse.gaussAnalysis(NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, fRange, poisson_offset, poisson_modulation, noiseAnalysis, directory, useMP, debug)

        directory = "noise_plots"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")

        #plotting
        kmean_bias = np.zeros(shape=(N_data, N_methods))
        gmean_bias = np.zeros(shape=(N_data, N_methods))
        for method in range(len(methods)):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, figsize=(20,20))
            kmean_bias[:, method] = pmean[:, method] - fRange
            gmean_bias[:, method] = gmean[:, method] - fRange
            ax1.plot(fRange, kmean_bias[:, method], label=f"Mean bias {methods[method]}")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Bias")
            ax1.set_title(f"Bias of {methods[method]} per frequency\nPoisson noise")
            ax1.fill_between(fRange, kmean_bias[:, method] - pstd[:, method], kmean_bias[:, method] + pstd[:, method], alpha=0.3)

            ax3.plot(fRange, gmean_bias[:, method])
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Bias")
            ax3.set_title(f"Bias of {methods[method]} per frequency\nGauss noise")
            ax3.fill_between(fRange, gmean_bias[:, method] - gstd[:, method], gmean_bias[:, method] + gstd[:, method], alpha=0.3)

            ax2.plot(fRange, pkurtosis[:, method], label=f"Kurtosis {methods[method]}", color="red")
            ax2.plot(fRange, pskewness[:, method], label=f"Skewness {methods[method]}", color="green")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_title(f"Skewnes and kurtosis of {methods[method]}\nPoisson noise")

            ax4.plot(fRange, gkurtosis[:, method])
            ax4.plot(fRange, gskewness[:, method])
            ax4.set_xlabel("Frequency (Hz)")
            ax4.set_title(f"Skewnes and kurtosis of {methods[method]}\nGauss noise")

            fig.legend()
            fig.savefig(f"{directory}/{methods[method]}_f{centralFrequency}_o{poisson_offset}_m{poisson_modulation}.png")
            print(f"Saved noise analysis of {methods[method]} to /{directory}/{methods[method]}_f{centralFrequency}_o{poisson_offset}_m{poisson_modulation}.png")
    os._exit(0)

if __name__ == "__main__":
    import multiprocessing as mp
    main()
