import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

import lib.helper_functions as helper
import lib.generate as gen

offset = 5
modulation = 1
print(f"SNR = {np.sqrt(modulation)}")

def genPhotons(sine_wave):
    expected_photon_counts = sine_wave * modulation

    photon_counts = np.random.poisson(expected_photon_counts + offset)
    #gaussian_noise = np.random.normal(expected_photon_counts + offset, modulation/2)
    #photon_counts_gauss = gaussian_noise
    photon_noise = photon_counts - expected_photon_counts

    return photon_counts#, photon_counts_gauss

#NdataPoints = 700 # roughly number of pixels in raw wavelength-spectrum 
NdataPoints = 2048 # resampled signal in k-space
centralFrequency = 7    # based on Pegah et al. (5 µm beads)
#centralFrequency = 50  # check for different frequency
noiseSamples = 10
fRange = helper.inclusiveRange(centralFrequency-0.6,centralFrequency+0.6,N=1001)

olddata, param = gen.generateData(N=NdataPoints, f = fRange)

# Let's plot 5 evenly spaced samples out of the 1001 frequencies
sample_indices = [0]#, 250, 500, 750, 1000]

plt.figure(figsize=(12, 6))

data = []
gauss_data = []
fdata = []
for i, sample in enumerate(sample_indices):
    dataPoints, gaussPoints = genPhotons(olddata[sample])
    data.append(dataPoints)
    gauss_data.append(gaussPoints)
    fdata.append(fft(dataPoints-np.mean(dataPoints)))
    #fdata = fft(olddata)
    plt.plot(data[i], label=f'f poisson = {param[sample][0]:.2f}')
    plt.plot(gauss_data[i], label=f'f gauss = {param[sample][0]:.2f}')

peak_index = np.argmax(np.abs(fdata))
print(peak_index)

plt.title('Sample Integer-Valued Sine Waves')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (Integer)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"gauss_poisson/noise_{offset}_{modulation}.png")
plt.show()