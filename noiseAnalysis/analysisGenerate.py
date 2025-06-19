import numpy as np
from scipy.fftpack import fft
import scipy.stats as stats
import matplotlib.pyplot as plt

import lib.generate as gen
import lib.fitting as fitting


def poissonSingleNoise(N_methods, noiseSamples, NdataPoints, fRange, methods, poisson_offset, poisson_modulation, debug=False):
    # test (some) methods for noise #TODO
    # initialise arrays to store found data
    found_peak = np.zeros(shape=(N_methods, noiseSamples))

    for sample in range(noiseSamples):
        for method in range(N_methods):
            singleData, _, olddata = gen.poissonData(N=NdataPoints, f = fRange, offset=poisson_offset, modulation=poisson_modulation)
                                                      
            LD = int(np.floor(NdataPoints/2))
            fdata = fft(singleData)

            signal = olddata*poisson_modulation
            noise = olddata*poisson_modulation+poisson_offset - singleData
            signal_power = np.mean(signal**2)
            noise_power = np.mean(noise**2)

            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)

            if debug == True:
                if method == 0 and sample == 0 and fRange < 7.51 and fRange > 7.49:
                    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=(20, 10))
                    ax1.plot(singleData, label="data with noise")
                    ax1.plot(olddata*poisson_modulation+poisson_offset, label="pure sine")
                    ax2.plot(olddata*poisson_modulation+poisson_offset - singleData, label="just noise")
                    ax1.legend()
                    ax2.legend()

                    oldfdata = fft(olddata*poisson_modulation+poisson_offset)
                    noisefdata = fft(olddata*poisson_modulation+poisson_offset - singleData)
                    fig2, ax3 = plt.subplots()
                    ax3.plot(np.abs(fdata[1:25])/np.max(fdata[4:10]), label = "fourier with noise")
                    #ax3.plot(np.abs(oldfdata[1:100]), label = "fourier without noise")
                    #ax3.plot(np.abs(noisefdata[1:100]), label = "fourier of noise")
                    ax3.legend()
                    fig.savefig("debug/debug_data.png")
                    fig2.savefig("debug/debug_fdata.png")
            if method == 0 and fRange < 7.51 and fRange > 7.49:
                if poisson_modulation < 51 and poisson_modulation > 50:
                    fig1, ax1= plt.subplots(1, 1, figsize=(16, 9))
                    ax1.plot(singleData[0:769], label="Data with noise", color="#38B6FF")
                    ax1.set_xlabel("Time (ms)", fontsize=20)
                    ax1.set_ylabel("Photon (count)", fontsize=20)
                    plt.rcParams.update({'font.size': 20})
                    fig1.savefig("poster/datavisual.png")

            found_peak[method, sample] = fitting.FFT_peakFit(fdata[1:LD], methods[method])

    statistics = []
    for method in range(N_methods):
        desc = stats.describe(found_peak[method, :])
        statistics.append((float(desc.mean), float(desc.variance), float(desc.skewness), float(desc.kurtosis)))
    return statistics, snr_db

def gaussSingleNoise(N_methods, noiseSamples, NdataPoints, fRange, methods, poisson_offset, poisson_modulation, noise_std, debug=False):
    # test (some) methods for noise #TODO
    # initialise arrays to store found data
    found_peak = np.zeros(shape=(N_methods, noiseSamples))

    for sample in range(noiseSamples):
        for method in range(N_methods):
            singleData, _, olddata = gen.gaussData(N=NdataPoints, f = fRange, offset=poisson_offset, modulation=poisson_modulation, noise_std=noise_std)
                                                      
            LD = int(np.floor(NdataPoints/2))
            fdata = fft(singleData)

            found_peak[method, sample] = fitting.FFT_peakFit(fdata[1:LD], methods[method])

    statistics = []
    for method in range(N_methods):
        desc = stats.describe(found_peak[method, :])
        statistics.append((float(desc.mean), float(desc.variance), float(desc.skewness), float(desc.kurtosis)))
    return statistics