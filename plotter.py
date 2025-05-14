import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

import lib.generate as gen
import lib.helper_functions as helper
import lib.plotting as plotting
import lib.fitting as fitting

def plotDFT_FFT(centralFrequency, NdataPoints, directory=None, saveFigures=False):
    """
    Plots the figure for comparing discrete Fourier Transform and Fast Fourier Transform

    Parameters
    ----------
    centralFrequency : int
        The frequency to check around

    NdataPoints : int
        Amount of datapoints, resolution

    directory : str
        Directory to save the plot to (default None)


    Returns
    -------
    A plot named 'fig_DFT_FFT.png' in the given directory
    """
    freq = [centralFrequency+delta for delta in [-0.4,0,0.4]]
    dfrange = helper.inclusiveRange(centralFrequency-3,centralFrequency+3,0.01)
    ffrange = np.arange(min(dfrange), max(dfrange)+1)
    fslice=slice(400,810,200)
    colours='rgb'
    data, param = gen.generateData(NdataPoints, f=freq)
    dfdata = helper.longHandDFT(data, dfrange)
    ffdata = fft(data)

    fig,ax = plt.subplots(1)

    for ii in range(len(dfdata)):
        dd = abs(dfdata)
        ax.plot(dfrange, dd[ii], colours[ii]+'-', label=f'dFT@{freq[ii]:.1f}')
        #ax.plot(dfrange[fslice], dd[ii][fslice], colours[ii]+'o', label='')            
        ax.plot(ffrange, abs(ffdata[ii,(centralFrequency-3):(centralFrequency+4)]), colours[ii]+'*', label='FFT')
    fig.legend()
    ax.set_xlabel('input frequency ($\\Delta f$)')
    ax.set_ylabel('magnitude fourier transform')

    if saveFigures:
        plotting.saveFigure(fig, 'fig_DFT_FFT.png', directory)

def comparisonGaussian(data, fdata, fRange, directory=None, saveFigures=False):
    """
    Plots a comparison between the found peak and actual peak

    Parameters
    ----------
    data : NDArray[Any] | list
        Datapoints

    fdata : Any
        Fourier transform of data

    fRange : list[floating[Any]]
        Range of frequencies

    directory : str
        Directory to save the plot to (default None)

    saveFigures : bool
        Whether or not to save figures in the directory (default False)

    Returns
    -------
    Currently nothing #TODO
    """
    plotting.makeDataPlot(data = data[::100], fdata = fdata[::100], fmt='-')

    Ngauss = 5
    k = np.array(fitting.Gauss_fit(fdata, N=Ngauss))

    fig, ax = plotting.makeComparisonPlot(fRange, k[:,1], "Gaussian")

    if saveFigures:
        plotting.saveFigure(fig, 'gaussian.png', directory)
        
    fig1,ax1=plt.subplots(1)
    ax1.plot(fRange, k[:,0], label="Gaussian fit")
    ax1.plot(fRange, k[:,3], label=f'RMS FFT (N={Ngauss})')
    ax1.set_xlabel('input frequency ($\\Delta f$)')
    ax1.set_ylabel('amplitude of component (AU)')
    ax1.legend()

    fig2,ax2=plt.subplots(1)
    ax2.plot(fRange, k[:,2], label="fit width")
    ax2.set_xlabel('input frequency ($\\Delta f$)')
    ax2.set_ylabel('width of Gaussian $\\sigma$')

def compareMethods(fdata, fRange, methods, saveFigures=False, image_name=None, directory=None):
    k = np.array(fitting.FFT_peakFit(fdata, methods))
    fig,_ = plotting.makeComparisonPlot(fRange, k, methods)
    if saveFigures:
        plotting.saveFigure(fig,f"{image_name}.png", directory)