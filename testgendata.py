import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

import lib.helper_functions as helper
import lib.generate as gen

def genPhotons(sine_wave):
    # 3. Convert to intensity (e.g., power)
    intensity = sine_wave**2

    # 4. Normalize and scale to photon rate (e.g., mean photons per time bin)
    max_photon_rate = 2  # max photons per time bin
    expected_photon_counts = intensity / np.max(intensity) * max_photon_rate

    # 5. Simulate Poisson-distributed photon counts
    photon_counts = np.random.poisson(expected_photon_counts)
    photon_noise = photon_counts - expected_photon_counts

    return photon_counts

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
fdata = []
for i, sample in enumerate(sample_indices):
    dataPoints = (genPhotons(olddata[sample]))
    data.append(dataPoints)
    fdata.append(fft(dataPoints-np.mean(dataPoints)))
    #fdata = fft(olddata)
    plt.plot(fdata[i], label=f'f = {param[sample][0]:.2f}')

peak_index = np.argmax(np.abs(fdata))
print(peak_index)

plt.title('Sample Integer-Valued Sine Waves')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (Integer)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()