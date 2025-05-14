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
            d, p = generateData(N, fval, phase)
            data.append(d)
            param.append(p)
        return data, param
      
    if hasattr(phase, "__len__"):
         for phaseval in phase:
             d, p = generateData(N, f, phaseval)
             data.append(d)
             param.append(p)
         return data, param
     
    # if non of the above, calculate single-parameter sine
    data = np.sin(2*np.pi*f/N*np.arange(1, N+1) + phase)   
    # to maintain compatibility with the original matlab test code
           
    return data, (f, phase)