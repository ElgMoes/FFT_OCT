#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:47:10 2024

@author: Gerhard Blab

routines for FFT peak fitting; 
contributions by Miriam Voots, Bram Haasnoot, and Saban Caliscan
example script for arXiv 
"""
def main(modulation, offset, methods):
    # general use packages & plotting
    import numpy as np
    from scipy.fftpack import fft

    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import pickle
    import gc
    from statistics import geometric_mean

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

    poisson_offset = offset
    poisson_modulation = modulation

    gauss_noise_std = 0.35*poisson_modulation

    frequencies = 1001

    saveFigures = True

    # be aware calculation of noise properties can take significant time!
    noiseAnalysis = True
    generateNoise = True

    gauss_analysis = False

    # be aware calculation of kappa can take significant time!
    kappaAnalysis = False
    generateKappaData = False

    NdataPoints = 2048 # resampled signal in k-space
    centralFrequency = 7    # based on Pegah et al. (5 µm beads)
    frequencyDiv = 0.6
    noiseSamples = 10 # 2500 for 1% error in std, 163 for 5%, 50 for 10%, 1000 gives 1.7%

    compare_methods = False
        
    fRange = helper.inclusiveRange(centralFrequency-frequencyDiv,centralFrequency+frequencyDiv,N=frequencies)
    noiseRange = helper.inclusiveRange(0.02, 0.5, 0.02)

    useMP = True
    debug = True
    SNRanalysis = True

    print_startup_info()

    directory = 'presentation'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")

    #%%
    print("plotting DFT vs FFT")
    plotter.plotDFT_FFT(centralFrequency, NdataPoints, directory, saveFigures)

    if compare_methods:
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
        N_methods = len(methods)
        N_data = frequencies

        directory = "noise_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")
        
        pmean, pstd, pskewness, pkurtosis, SNR = panalyse.poissonAnalysis(NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, fRange, poisson_offset, poisson_modulation, noiseAnalysis, directory, useMP, debug)

        directory = "noise_plots"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")

        #plotting
        kmean_bias = np.zeros(shape=(N_data, N_methods))
        if SNRanalysis:
            rms_pstd = np.zeros(shape=(len(methods)))
            var_pstd = np.zeros(shape=(len(methods)))
            for method in range(len(methods)):
                rms_pstd[method] = np.sqrt(np.mean(pstd[:, method]**2))
                var_pstd[method] = np.var(pstd[:, method])
            return rms_pstd, var_pstd

        else:
            for method in range(len(methods)):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, figsize=(20,20))
                kmean_bias[:, method] = pmean[:, method] - fRange
                ax1.plot(fRange, kmean_bias[:, method], label=f"Mean bias {methods[method]}")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Bias")
                ax1.set_title(f"Bias of {methods[method]} per frequency\nPoisson noise")
                ax1.fill_between(fRange, kmean_bias[:, method] - pstd[:, method], kmean_bias[:, method] + pstd[:, method], alpha=0.3)

                ax1.text(
                    0.02, 0.98, f"SNR = {SNR:.5f} dB",
                    transform=ax1.transAxes,  # relative coordinates (0 to 1)
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

                ax2.plot(fRange, pkurtosis[:, method], label=f"Kurtosis {methods[method]}", color="red")
                ax2.plot(fRange, pskewness[:, method], label=f"Skewness {methods[method]}", color="green")
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_title(f"Skewnes and kurtosis of {methods[method]}\nPoisson noise")

                mean_pstd = np.mean(pstd[:, method])
                rms_pstd = np.sqrt(np.mean(pstd[:, method]**2))
                gmean_pstd = geometric_mean(pstd[:, method])
                ax3.plot(fRange, pstd[:, method])
                ax3.text(
                    0.02, 0.02, f"rms = {rms_pstd:.5f}\nmean = {mean_pstd:.5f}\ngmean = {gmean_pstd:.5f}",
                    transform=ax3.transAxes,  # relative coordinates (0 to 1)
                    fontsize=12,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

                fig.legend()
                fig.savefig(f"{directory}/{methods[method]}_f{centralFrequency}_s{noiseSamples}_o{poisson_offset}_m{poisson_modulation}.png")
                print(f"Saved noise analysis of {methods[method]} to /{directory}/{methods[method]}_f{centralFrequency}_s{noiseSamples}_o{poisson_offset}_m{poisson_modulation}.png")

if __name__ == "__main__":
    import multiprocessing as mp
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    
    methods = ['macleod', 'jacobsenmod', 'quinns2nd']
    offset = 500
    SNR = np.linspace(-10, 20, 6)
    modulations = []
    rms_per_snr = []
    var_per_snr = []
    for i, snr in enumerate(SNR):
        modulations.append(float(np.sqrt(2 * offset * 10**(snr / 10))))
    for i in range(len(modulations)):
        snr_rms, snr_var = main(modulations[i], offset, methods)
        rms_per_snr.append(snr_rms)
        var_per_snr.append(snr_var)
    rms_per_snr_T = np.array(rms_per_snr).T
    var_per_snr_T = np.array(var_per_snr).T

    fig, ax1 = plt.subplots(1, 1, constrained_layout=True, figsize=(10,10), facecolor="#737373")
    for i in range(len(rms_per_snr_T)):
        ax1.plot(SNR, rms_per_snr_T[i], label=f"{methods[i]}", linewidth=2.5)
    ax1.set_xlabel("SNR in dB", fontsize=20)
    ax1.set_ylabel("rms", fontsize=20)
    plt.rcParams.update({'font.size': 20})
    ax1.set_facecolor("#737373")
    yticks = 10.0 ** np.arange(-3, -0.75, 0.25)
    ax1.set_yticks(yticks)
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax1.get_yaxis().set_major_formatter(formatter)
    ax1.grid(True)
    ax1.legend()
    ax1.set_yscale('log')
    fig.savefig("snr/rmsplot.png")
    os._exit(0)
