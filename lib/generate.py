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

def genPhotons(sine_wave):
    # 3. Convert to intensity (e.g., power)
    intensity = sine_wave**2

    # 4. Normalize and scale to photon rate (e.g., mean photons per time bin)
    max_photon_rate = 20  # max photons per time bin
    expected_photon_counts = intensity / np.max(intensity) * max_photon_rate

    # 5. Simulate Poisson-distributed photon counts
    photon_counts = np.random.poisson(expected_photon_counts)
    photon_noise = photon_counts - expected_photon_counts

    return photon_counts

def poissonData(N = 700, f=9, phase=0):
    olddata, param = generateData(N=N, f = f, phase=phase)

    data = []
    for i in range(len(olddata)):
        data.append(genPhotons(olddata[i]))

    return data, param