import numpy as np
from scipy.fftpack import fft
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt

import lib.generate as gen
import lib.fitting as fitting


def singleNoise(N_methods, noiseSamples, NdataPoints, fRange, methods, poisson_offset, poisson_modulation):
    # test (some) methods for noise #TODO
    # initialise arrays to store found data
    data = []
    found_peak = np.zeros(shape=(N_methods, noiseSamples))

    for sample in range(noiseSamples):
        for method in range(N_methods):
            singleData, singleParam = gen.poissonData(N=NdataPoints, f = fRange, offset=poisson_offset, modulation=poisson_modulation)
                                                      
            LD = int(np.floor(NdataPoints/2))
            fdata = fft(singleData)
            found_peak[method, sample] = fitting.FFT_peakFit(fdata[0:LD], methods[method])

    for method in range(N_methods):
        data.append(stats.describe(found_peak[method, :]))
    return data