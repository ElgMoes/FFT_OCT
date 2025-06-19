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
        print(f"noise samples       : {noiseSamples}")
        print(f"Use multiprocessing : {useMP}")
        print("="*50)

    poisson_offset = offset
    poisson_modulation = modulation

    frequencies = 1001

    saveFigures = True

    # be aware calculation of noise properties can take significant time!
    noiseAnalysis = True
    generateNoise = True

    NdataPoints = 2048 # resampled signal in k-space
    centralFrequency = 7    # based on Pegah et al. (5 µm beads)
    frequencyDiv = 0.6
    noiseSamples = 2500 # 2500 for 1% error in std, 163 for 5%, 50 for 10%, 1000 gives 1.7%

    compare_methods = True
        
    fRange = helper.inclusiveRange(centralFrequency-frequencyDiv,centralFrequency+frequencyDiv,N=frequencies)

    useMP = True
    debug = False
    SNRanalysis = True
    forPoster = True
    print_startup_info()

    #%%
    directory = 'poster'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")

    if compare_methods:
        "compare methods"
        # Generate data
        data, param = gen.generateData(N=NdataPoints, f = fRange)
        fdata = fft(data-np.mean(data, axis=1, keepdims=True))

        methods = ["MacLeod", "JacobsenMod", "Quinns2nd"]
        method_names = ["MacLeod", "Jacobsen modified", r"Quinns $2^{nd}$"]
        colors = ["#38B6FF", "#FF8100", "#00EE00"]
        print(f"comparing: {', '.join(methods)}")
        plotter.compareMethods(fdata, fRange, methods, saveFigures, "bias-comparison", directory, method_names=method_names, colors=colors)

    #%%
    
    if noiseAnalysis:
        import noiseAnalysis.poissonAnalysis as panalyse
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
            if forPoster:
                idx = (np.abs(np.array(modulations) - 50)).argmin()
                if modulation == modulations[idx]:
                    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                    kmean_bias[:, 0] = pmean[:, 0] - fRange
                    ax.plot(fRange, kmean_bias[:, 0], label=f"Mean bias {methods[0]}", color="#38B6FF")
                    ax.set_xlabel("Frequency (Hz)", fontsize=20)
                    ax.set_ylabel("Bias (Hz)", fontsize=20)
                    plt.rcParams.update({'font.size': 20})
                    ax.fill_between(fRange, kmean_bias[:, 0] - pstd[:, 0], kmean_bias[:, 0] + pstd[:, 0], alpha=0.25, color="#38B6FF", label="Std bias MacLeod")
                    ax.text(
                        0.01, 0.98, f"SNR = {SNR:.2f} dB",
                        transform=ax.transAxes,  # relative coordinates (0 to 1)
                        fontsize=12,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    )
                    ax.legend()
                    fig.savefig(f"poster/example_{methods[0]}.png")
                    plt.close()

            rms_pstd = np.zeros(shape=(len(methods)))
            var_pstd = np.zeros(shape=(len(methods)))
            for method in range(len(methods)):
                rms_pstd[method] = np.sqrt(np.mean(pstd[:, method]**2))
                var_pstd[method] = np.var(pstd[:, method])
            return rms_pstd, var_pstd

        else:
            for method in range(len(methods)):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,20))
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
                ax2.set_title(f"Skewnes and kurtosis of {methods[method]}")

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
                plt.close()
                print(f"Saved noise analysis of {methods[method]} to /{directory}/{methods[method]}_f{centralFrequency}_s{noiseSamples}_o{poisson_offset}_m{poisson_modulation}.png")

if __name__ == "__main__":
    import multiprocessing as mp
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    methods = ['macleod', 'jacobsenmod', 'quinns2nd']
    offset = 500
    SNR = np.linspace(-10, 20, 51)
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

    fig, ax1 = plt.subplots(1, 1, figsize=(16,9), facecolor="#FFFFFF")
    method_names = ["MacLeod", "Jacobsen modified", r"Quinns $2^{nd}$"]
    colors = ["#38B6FF", "#FF8100", "#00EE00"]
    for i in range(len(rms_per_snr_T)):
        ax1.plot(SNR, rms_per_snr_T[i], label=f"{method_names[i]}", linewidth=5, color=colors[i])
    ax1.set_xlabel("SNR (dB)", fontsize=20)
    ax1.set_ylabel("RMS", fontsize=20)
    plt.rcParams.update({'font.size': 20})
    ax1.set_facecolor("#FFFFFF")
    ax1.set_ylim(1e-3, 1e-1)
    ax1.grid(True)
    ax1.legend()
    ax1.set_yscale('log')
    fig.savefig("poster/rmsplot.png")
    plt.close()
    os._exit(0)
