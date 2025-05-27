import numpy as np

def generateData(N = 700, f = 9, phase = 0):
    """ create a data set (pure sine) of length N, frequency f, 
    phase, and optionally noise (rms, gaussian noise)
    f, phase, and noise may be lists, in which case a list of data sets 
    is returned; 
    """
    data = []
    param = []
  
    if hasattr(f, "__len__"):   # probably a list or matrix
        for fval in f:
            dataPoint, paramPoint = generateData(N, fval, phase)
            data.append(dataPoint)
            param.append(paramPoint)
        return data, param
      
    if hasattr(phase, "__len__"):
         for phaseval in phase:
             dataPoint, paramPoint = generateData(N, f, phaseval)
             data.append(dataPoint)
             param.append(paramPoint)
         return data, param
     
    # if non of the above, calculate single-parameter sine
    data = np.sin(2*np.pi*f/N*np.arange(1, N+1) + phase)
    # to maintain compatibility with the original matlab test code
           
    return data, (f, phase)

def genPhotons(sine_wave, offset, modulation):
    expected_photon_counts = sine_wave * modulation
    photon_counts = np.random.poisson(expected_photon_counts + offset)
    return photon_counts

def poissonData(N = 700, f=9, phase=0, offset=500, modulation=50):
    olddata, param = generateData(N=N, f = f, phase=phase)

    data = genPhotons(olddata, offset, modulation)

    return data, param, olddata

# TODO not working

def gaussData(N=700, f=9, phase=0, offset=500, modulation=50,  noise_std=0.35*50):
    olddata, param = generateData(N=N, f = f, phase=phase)
    data = []
    for i in range(len(olddata)):
        data.append(olddata*modulation+offset)
    noise = np.random.normal(0, noise_std, size=len(data))
    noisy_data = data + noise
    intdata = np.rint(noisy_data).astype(int)
    
    return intdata, param