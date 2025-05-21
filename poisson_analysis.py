import numpy as np
from scipy.fftpack import fft
import scipy.stats as stats

import lib.generate as gen
import lib.fitting as fitting


def singleNoise(N_methods, noiseSamples, NdataPoints, fRange, methods, poisson_offset, poisson_modulation):
    # test (some) methods for noise #TODO
    # initialise arrays to store found data
    found_peak = np.zeros(shape=(N_methods, noiseSamples))

    for sample in range(noiseSamples):
        for method in range(N_methods):
            singleData, _ = gen.poissonData(N=NdataPoints, f = fRange, offset=poisson_offset, modulation=poisson_modulation)
                                                      
            LD = int(np.floor(NdataPoints/2))
            fdata = fft(singleData)
            found_peak[method, sample] = fitting.FFT_peakFit(fdata[0:LD], methods[method])

    statistics = []
    for method in range(N_methods):
        desc = stats.describe(found_peak[method, :])
        statistics.append((float(desc.mean), float(desc.variance), float(desc.skewness), float(desc.kurtosis)))
    return statistics