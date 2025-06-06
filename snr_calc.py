import numpy as np

offset = 500
SNR = np.linspace(-10, 20, 11)

modulation = []

for i, snr in enumerate(SNR):
    modulation.append(float(np.sqrt(2 * offset * 10**(snr / 10))))

print(modulation)
