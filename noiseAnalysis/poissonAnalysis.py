import analysis as analyse
import numpy as np
import os
from tqdm import tqdm
import pickle
import gc

def poissonAnalysis(NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, fRange, poisson_offset, poisson_modulation, generateNoise, directory, useMP, debug):
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
                    print(f"generating poisson noise on {min(maxProcess, mp.cpu_count())} cores")
                else:
                    print("generating poisson noise on 1 core")
                progbar = tqdm(total = iterations, smoothing = 0.025, unit='job')
            else:
                print("generating poisson noise on 1 core")
                
            def callbackProgress(*a):
                progbar.update(1)
            
            if pool is not None:
                res = [ pool.apply_async(analyse.poissonSingleNoise, (N_methods, noiseSamples, NdataPoints, f, methods, poisson_offset, poisson_modulation, debug), callback=callbackProgress) for f in fRange]
                pool.close()
                pool.join()
                gc.collect()

            if pool is None:
                with tqdm(total = iterations) as pbar:
                    for f in range(len(fRange)):
                        data.append(analyse.poissonSingleNoise(N_methods, noiseSamples, NdataPoints, fRange[f], methods, poisson_offset, poisson_modulation, debug))
                        pbar.update(1)
            else:   
                data = [res[f].get() for f in range(len(fRange))]

            # initializing arrays to store statistical values
            # mean and standard deviation
            kmean = np.zeros(shape=(N_data, N_methods))
            kstd = np.zeros(shape=(N_data, N_methods))
            
            # Skewness and kurtosis
            kskewness = np.zeros(shape=(N_data, N_methods))
            kkurtosis = np.zeros(shape=(N_data, N_methods))


            for Nd in range(N_data):
                try:
                    data_per_frequency = data[Nd]
                    kmean[Nd, :] = [ data_per_frequency[i][0] for i in range(len(data_per_frequency)) ]
                    kstd[Nd, :] = np.sqrt([ data_per_frequency[i][1] for i in range(len(data_per_frequency)) ])
                    kskewness[Nd, :] = [ data_per_frequency[i][2] for i in range(len(data_per_frequency)) ]
                    kkurtosis[Nd, :] = [ data_per_frequency[i][3] for i in 
                    range(len(data_per_frequency)) ]
                except Exception as e:
                    print(f"{e} at loop nr {Nd} in storing statistics")

            if directory is None:
                directory = '.'
            for ii, k in enumerate((kmean, kstd, kskewness, kkurtosis)):
                names = ['kmean', 'kstd', 'kskewness', 'kkurtosis']
                np.save(f'{directory}/noise_{names[ii]}.dat', k)
                print(f"saved data to /{directory}/noise_{names[ii]}.dat.npy")

            with open(f'{directory}/param.dat', 'wb') as f:
                pickle.dump(used_parameters, f)
                print(f"saved data to {directory}/param.dat")
            generateNoise = False # only once per session
        else:
            ## or read data from last run
            print("Reading data from last run")
            try:
                kmean = np.load(f'{directory}/noise_kmean.dat.npy')
                kstd = np.load(f'{directory}/noise_kstd.dat.npy')
                kskewness = np.load(f'{directory}/noise_kskewness.dat.npy')
                kkurtosis = np.load(f'{directory}/noise_kkurtosis.dat.npy')
                with open(f'{directory}/param.dat', 'rb') as f:
                    used_parameters = pickle.load(f)
            except Exception as e:
                print(f"Failed opening file -> {e}")
                exit()

        return kmean, kstd, kskewness, kkurtosis