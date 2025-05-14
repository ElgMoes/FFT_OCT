import os
import numpy as np
from scipy.fftpack import fft

def uniquify(path, prefix = ''):
    """
    for a given path/filename, create a unique name by adding
    zero-padded numbers and prefix if needed
    """
    
    filename, extension = os.path.splitext(path)
    counter = 1
    
    while os.path.exists(path):
        path = f'{filename}_{prefix}{counter:04d}{extension}'
        counter += 1
        
    return path

        
def generateColourmap(N=10, c1 = None, c2 = None):
    """Generate a N-level colourmap from Red to Blue
    """
    if c1 is None:
        c1 = np.array([1,0,0])
    if c2 is None:
        c2 = np.array([0,0,1])
        
    if (N==1):
        return [c1]
    
    clrs = []
    for ii in range(N):
        w = ii/(N-1)
        clrs.append(w*c1+(1-w)*c2)
    return clrs

def inclusiveRange(start = 0, end = 1, step = None, N = 100):
    """generate linear ranges which include the final value
    """
    if step is None:
        step = (end - start)/(N-1)
    data = list(np.arange(start, end, step))
    # test whether end value is near-integer steps away from start
    intOff = (end-start)/step
    intOff -= round(intOff)
    if abs(intOff)<1e-10:
        data.append(end)
    return data

def longHandDFT(data, f=None):
    """ 
    calculate DFT for arbitray frequencies (note the lack of the second
    "F" -- this method is no longer 'fast'!)
    """
    
    if f is None:
        fd = fft(data)
    else:
        if hasattr(data, "__len__") and hasattr(data[0],"__len__"):
            # list of lists (or array)
            Nd = len(data)
        else:
            data = [data]
            Nd = 1
            
        if not hasattr(f, "__len__"):
            f = [f]
            
        fd = np.zeros(shape=(Nd, len(f)), dtype = np.complex64)
        
        N = len(data[0])    # length of original data
        base = -2j*np.pi*np.arange(N)/N # fixed part in calculation of dFT
        
        for ii in range(len(f)):
            expbase = np.exp(base * f[ii])  # frequency/k dependent part of dFT
            for jj in range(Nd):    
                fd[jj][ii] = np.sum(data[jj] * expbase) # dFT (not fast anymore)
                
    return fd