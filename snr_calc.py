import numpy as np

offset = 500
modulations = []
rms_per_snr = []
var_per_snr = []
SNR = np.linspace(-10, 20, 101)
for i, snr in enumerate(SNR):
    modulations.append(float(np.sqrt(2 * offset * 10**(snr / 10))))

print(modulations)
