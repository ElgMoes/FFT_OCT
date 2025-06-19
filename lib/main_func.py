import numpy as np
import matplotlib.pyplot as plt
from statistics import geometric_mean
import os

import lib.noiseAnalysis.poissonAnalysis as panalyse

def plot_example_MacLeod(fRange, method, SNR, poisson_mean, poisson_std):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    mean_bias = np.zeros_like(poisson_mean)
    mean_bias[:, 0] = poisson_mean[:, 0] - fRange
    ax.plot(fRange, mean_bias[:, 0], label=f"Mean bias {method}", color="#38B6FF")
    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel("Bias (Hz)", fontsize=20)
    plt.rcParams.update({'font.size': 20})
    ax.fill_between(fRange, mean_bias[:, 0] - poisson_std[:, 0], mean_bias[:, 0] + poisson_std[:, 0], alpha=0.25, color="#38B6FF", label=f"Std bias {method}")
    ax.text(
        0.01, 0.98, f"SNR = {SNR:.2f} dB",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    ax.legend()
    fig.savefig(f"poster/example_{method}.png")
    plt.close()
    print(f"saved example plot to poster/example_{method}.png")

def SNRanalysis(noise_parameters, poisson_data, modulations, SNR):
    (poisson_mean, poisson_std, poisson_skewness, poisson_kurtosis) = poisson_data
    (NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, centralFrequency, fRange, poisson_offset, poisson_modulation) = noise_parameters

    idx = (np.abs(np.array(modulations) - 50)).argmin()
    if poisson_modulation == modulations[idx]:
        plot_example_MacLeod(fRange, methods[0], SNR, poisson_mean, poisson_std)

    rms_poisson_std = np.zeros(shape=(len(methods)))
    var_poisson_std = np.zeros(shape=(len(methods)))
    for method in range(len(methods)):
        rms_poisson_std[method] = np.sqrt(np.mean(poisson_std[:, method]**2))
        var_poisson_std[method] = np.var(poisson_std[:, method])
    return rms_poisson_std, var_poisson_std

def plot_bias_and_deviations(noise_parameters, poisson_data, method, SNR):
    directory = "noise_plots"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")

    (poisson_mean, poisson_std, poisson_skewness, poisson_kurtosis) = poisson_data
    (NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, centralFrequency, fRange, poisson_offset, poisson_modulation) = noise_parameters
    mean_bias = np.zeros(shape=(N_data, N_methods))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,20))
    mean_bias[:, method] = poisson_mean[:, method] - fRange
    ax1.plot(fRange, mean_bias[:, method], label=f"Mean bias {methods[method]}")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Bias")
    ax1.set_title(f"Bias of {methods[method]} per frequency\nPoisson noise")
    ax1.fill_between(fRange, mean_bias[:, method] - poisson_std[:, method], mean_bias[:, method] + poisson_std[:, method], alpha=0.3)
    ax1.text(
        0.02, 0.98, f"SNR = {SNR:.5f} dB",
        transform=ax1.transAxes,  # relative coordinates (0 to 1)
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    ax2.plot(fRange, poisson_kurtosis[:, method], label=f"Kurtosis {methods[method]}", color="red")
    ax2.plot(fRange, poisson_skewness[:, method], label=f"Skewness {methods[method]}", color="green")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title(f"Skewnes and kurtosis of {methods[method]}")
    mean_pstd = np.mean(poisson_std[:, method])
    rms_pstd = np.sqrt(np.mean(poisson_std[:, method]**2))
    gmean_pstd = geometric_mean(poisson_std[:, method])
    ax3.plot(fRange, poisson_std[:, method])
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

def noiseAnalysis(noise_params, generateNoise, useMP, SNRanalyse, modulations, run_nr):
    (NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, centralFrequency, fRange, poisson_offset, poisson_modulation) = noise_params

    poisson_data, SNR = panalyse.poissonAnalysis(noise_params, generateNoise, useMP, run_nr)

    if SNRanalyse:
        rms_pstd, var_pstd = SNRanalysis(noise_params, poisson_data, modulations, SNR)
        return rms_pstd, var_pstd
    else:
        for method in range(len(methods)):
            plot_bias_and_deviations(noise_params, poisson_data, method, SNR)