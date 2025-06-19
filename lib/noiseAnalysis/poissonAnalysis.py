import numpy as np
import os
from tqdm import tqdm
import pickle
import gc

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
                    print("saved visual plot to poster/datavisual.png")

            found_peak[method, sample] = fitting.FFT_peakFit(fdata[1:LD], methods[method])

    statistics = []
    for method in range(N_methods):
        desc = stats.describe(found_peak[method, :])
        statistics.append((float(desc.mean), float(desc.variance), float(desc.skewness), float(desc.kurtosis)))
    return statistics, snr_db


def poissonAnalysis(noise_parameters, generateNoise, useMP, run_nr):
        (NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, centralFrequency, fRange, poisson_offset, poisson_modulation) = noise_parameters

        directory = "noise_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")

        if generateNoise:
            used_parameters = (N_data, methods)
            iterations = frequencies

            pool = None
            data = []
            if useMP and (iterations > 100):
                import multiprocessing as mp
                maxProcess = mp.cpu_count()
                pool = mp.Pool(processes = min(maxProcess, mp.cpu_count()))
                if min(maxProcess, mp.cpu_count()) > 1:
                    print(f"generating poisson noise on {min(maxProcess, mp.cpu_count())} cores ({run_nr})")
                else:
                    print("generating poisson noise on 1 core")
                progbar = tqdm(total = iterations, smoothing = 0.025, unit='job')
            else:
                print("generating poisson noise on 1 core")
                
            def callbackProgress(*a):
                progbar.update(1)
            
            if pool is not None:
                res = [ pool.apply_async(poissonSingleNoise, (N_methods, noiseSamples, NdataPoints, f, methods, poisson_offset, poisson_modulation), callback=callbackProgress) for f in fRange]
                pool.close()
                pool.join()
                gc.collect()

            if pool is None:
                with tqdm(total = iterations) as pbar:
                    for f in range(len(fRange)):
                        full_data.append(poissonSingleNoise(N_methods, noiseSamples, NdataPoints, fRange[f], methods, poisson_offset, poisson_modulation))
                        pbar.update(1)
            else:   
                full_data = [res[f].get() for f in range(len(fRange))]

            data, snr_list = zip(*full_data)

            SNR = np.mean(snr_list)

            # initializing arrays to store statistical values
            # mean and standard deviation
            mean = np.zeros(shape=(N_data, N_methods))
            std = np.zeros(shape=(N_data, N_methods))
            
            # Skewness and kurtosis
            skewness = np.zeros(shape=(N_data, N_methods))
            kurtosis = np.zeros(shape=(N_data, N_methods))


            for Nd in range(N_data):
                try:
                    data_per_frequency = data[Nd]
                    mean[Nd, :] = [ data_per_frequency[i][0] for i in range(len(data_per_frequency)) ]
                    std[Nd, :] = np.sqrt([ data_per_frequency[i][1] for i in range(len(data_per_frequency)) ])
                    skewness[Nd, :] = [ data_per_frequency[i][2] for i in range(len(data_per_frequency)) ]
                    kurtosis[Nd, :] = [ data_per_frequency[i][3] for i in 
                    range(len(data_per_frequency)) ]
                except Exception as e:
                    print(f"{e} at loop nr {Nd} in storing statistics")

            if directory is None:
                directory = '.'
            for ii, k in enumerate((mean, std, skewness, kurtosis)):
                names = ['kmean', 'kstd', 'kskewness', 'kkurtosis']
                np.save(f'{directory}/noise_{names[ii]}.dat', k)

            with open(f'{directory}/param.dat', 'wb') as f:
                pickle.dump(used_parameters, f)
            generateNoise = False # only once per session
        else:
            ## or read data from last run
            print("Reading data from last run")
            try:
                mean = np.load(f'{directory}/noise_kmean.dat.npy')
                std = np.load(f'{directory}/noise_kstd.dat.npy')
                skewness = np.load(f'{directory}/noise_kskewness.dat.npy')
                kurtosis = np.load(f'{directory}/noise_kkurtosis.dat.npy')
                with open(f'{directory}/param.dat', 'rb') as f:
                    used_parameters = pickle.load(f)
            except Exception as e:
                print(f"Failed opening file -> {e}")
                exit()

        return (mean, std, skewness, kurtosis), SNR