import matplotlib.pyplot as plt
import numpy as np

import lib.fitting as fitting
import os

def compareMethods(fdata, fRange, methods, image_name=None, method_names=[], colors=[]):
    directory = 'poster'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"created new directory \'{directory}\' for storing images at {os.getcwd()}/")
    
    k = np.array(fitting.FFT_peakFit(fdata, methods))
    fig, _ = makeComparisonPlot(fRange, k - 1, methods, method_names=method_names, colors=colors) # -1 to compensate slice
    saveFigure(fig,f"{image_name}.png", directory)
    plt.close()
    print(f"saved {image_name}.png to /{directory}/")

def makeComparisonPlot(fvals, data, methods, title = None, method_names=[], colors=[]):
    f, ax1 = plt.subplots(1,1, figsize=(16, 9))
    plt.rcParams.update({'font.size': 20})
        
    for ii in range(len(methods)):
        ki = np.array(data[:,ii])
        bias = ki - fvals
        ax1.plot(fvals, bias, linewidth=5, label=f"{method_names[ii]}", color=colors[ii])

    ax1.set_ylabel('Bias (Hz)', fontsize=20)
    ax1.set_xlabel('Frequency (Hz)', fontsize=20)
    ax1.legend()
    
    if title is not None:
        ax1.set_title(title)
    
    return f, ax1

def saveFigure(fig, filename, directory):
    if directory is None:
        d = ''
    else:
        d = directory + '/'
        
    fig.savefig(d+filename, dpi=300, bbox_inches='tight')
    plt.close()

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
