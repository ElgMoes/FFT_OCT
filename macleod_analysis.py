import numpy as np
import matplotlib.pyplot as plt
import pickle
import lib.helper_functions as helper

directory = "noise_data"

kmean = np.load(f'{directory}/noise_kmean.dat.npy')
kstd = np.load(f'{directory}/noise_kstd.dat.npy')

print(np.shape(kmean))

frequencies = 1001
centralFrequency = 20    # based on Pegah et al. (5 µm beads)
frequencyDiv = 10

fRange = helper.inclusiveRange(centralFrequency-frequencyDiv,centralFrequency+frequencyDiv,N=frequencies)

kmean_bias = kmean[:,0] - fRange

plt.figure()
plt.plot(fRange, kmean_bias)
plt.fill_between(fRange, kmean_bias - kstd[:,0], kmean_bias + kstd[:,0], alpha=0.3)
plt.show()