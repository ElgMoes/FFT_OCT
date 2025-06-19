import matplotlib.pyplot as plt
import numpy as np


def makeComparisonPlot(fvals, data, methods, title = None, method_names=[], colors=[]):
    """Make a comparison plot of different estimators
    """
    
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
    #directory = f'simResults_{NdataPoints}_{centralFrequency:.1f}_{noiseSamples:d}'
    if directory is None:
        d = ''
    else:
        d = directory + '/'
        
    fig.savefig(d+filename, dpi=300, bbox_inches='tight')
    plt.close()