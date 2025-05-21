import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from scipy.optimize import curve_fit

import lib.fitting_routines as fr
import lib.helper_functions as helper

def noiseBehaviour(rms, noise, noiseSamples=1000, unc=None):
    """ noiseBehaviour(rms, noise, noiseSamples=1000)
    essentially, a linear fit to rms vs noise
    returns the value from rms = 1, which is also the slope
    """
    
    def lin(x, k):
        return k*x
    
    if unc is None:
        # if uncertainty is not given, try to estimate from sample size
        unc = noise/np.sqrt(noiseSamples*(noiseSamples-1))
    
    k, kv = curve_fit(lin, rms, noise, p0=noise[-1]/rms[-1], sigma=unc)
       
    return k[0], np.sqrt(kv[0,0])

def makeComparisonPlot(fvals, data, methods, title = None):
    """Make a comparison plot of different estimators
    """
    
    f,(ax1, ax2) = plt.subplots(2,1)
    #cval = generateColourmap(len(data))
    
    if isinstance(methods, (str)):  # make a 2D array out of data
        methods = [methods]
        data = data.reshape(len(data),1)
        
    for ii in range(len(methods)):
        ki = np.array(data[:,ii])
        bias = ki - fvals
        ax1.plot(fvals, ki, label=fr.formalMethodName(methods[ii]))
        ax2.plot(fvals, bias)
        
        am = abs(bias)
        mf = am.max()
        avgf = am.mean()
        stdf = am.std()
        mfpos = fvals[np.argmax(am)]
        #print(f'{methods[ii]}: maximum bias is {mf:.6f} at position {mfpos:.2f}; std of bias is {bias.std():.6f}')
        #print(f'     -- <|bias|> is {avgf:.6f}, std(|bias|) is {stdf:.6f}')
    ax1.set_ylabel('peak position')
    ax1.legend(fontsize=8)
    ax2.set_ylabel('bias')
    ax2.set_xlabel('frequency ($\\Delta f$)')
    
    if title is not None:
        ax1.set_title(title)
    
    return f, (ax1, ax2)


def makeDataPlot(xvals = None, data = None, fvals = None, 
                 fdata = None, fmt='.', xl=None, yl=None):
    
    if hasattr(data, '__len__') and not hasattr(data[0],'__len__'):
        data = [data]   # only one data set -- ok, deal with it!
        
    plotFFT = (fdata is not None)
    
    if plotFFT:
        f, (ax1, ax2) = plt.subplots(2,1, dpi=600)
    else:
        f, ax1 = plt.subplots(1,1, dpi=600)
        ax2 = None
        
    if xvals is None:  
        xvals = helper.inclusiveRange(0,1,N=len(data[0]))
        
    if fvals is None:
        fvals = slice(2,13)
        
    cval = helper.generateColourmap(len(data))
    for ii in range(len(data)):
        ax1.plot(xvals, data[ii], fmt, color = cval[ii])
        if plotFFT:
            ax2.plot(np.arange(fvals.start,fvals.stop, fvals.step),\
                     abs(fdata[ii])[fvals], color = cval[ii])
        
    ax1.set_xlabel('time (s)' if xl is None else xl)
    ax1.set_ylabel('amplitude' if yl is None else yl )
    if plotFFT:
        ax2.set_xlabel('frequency ($\\Delta f$)')
        ax2.set_ylabel("|FFT|")
    
    return f, (ax1, ax2)


def compareNoisePlots(fvals, dataMean, dataStd, param, dataC = None, biasType="max"):
    """compareNoisePlot(dataMean, dataStd, param, dataC, biasType)
        plots rms vs noise (bias/std) and compares different methods
        valid biasType are 'max', 'mean', and 'both'
    """
    Nd, Nr, Nm = dataMean.shape
    doublePlot = (biasType == "both")
    mx = []
    mn = []
    sx = []
    sd = []
    
    rvals = param[1]
    for rr in range(Nr):
        # generate sub-set of data for constant RMS
        maxb = []
        meanb = []
        maxs = []
        means = []
        biasR = np.array([abs(dataMean[:,rr,mm]-fvals) for mm in range(Nm)]).transpose()
        stdR = dataStd[:,rr,:]
        for mm in range(Nm):
            maxb.append(biasR[:,mm].max())
            meanb.append(biasR[:,mm].mean())
            maxs.append(stdR[:,mm].max())
            means.append(stdR[:,mm].mean())
        mx.append(maxb)
        mn.append(meanb)
        sx.append(maxs)
        sd.append(means)
        
    mx = np.array(mx)
    mn = np.array(mn)
    sx = np.array(sx)
    sd = np.array(sd)
    
    fig, ax = plt.subplots(1, 2 if doublePlot else 1)
    colours = helper.generateColourmap(N=Nm)
    markers = it.cycle("o*Dv<PX")
    # position of certain S/N values on the oritinal X axis
    newTickPositions = 1/(np.sqrt(2) * np.array([1/np.sqrt(2), 1, 1.5, 2, 5]))
    def makeSecondXAxis(axOrig, newTicks, label = None):
        """make a secondary X axis with (non-unformly) spaced ticks indicating S/N
        """
        axNew = axOrig.twiny()
        def tickLabel(x):
            V = 1/(np.sqrt(2)*x)
            return [f'{v:.2f}' for v in V]
        axNew.set_xticks(newTicks)
        axNew.set_xticklabels(tickLabel(newTicks))
        axNew.set_xlim(axOrig.get_xlim())
        if label is not None:
            axNew.set_xlabel(label)
        return axNew
    
    if Nr>5: #make a slice
        useSlices = True
        Nstep = int(np.floor(Nr/6))
    else:
        ind =slice(0,Nr)
    for rr in range(Nm):
        if useSlices:
            ind = slice((rr%Nstep)+2, Nr, Nstep)
        method = fr.formalMethodName(param[2][rr])
        marker = next(markers)
        match biasType:
            case "both":
                ax[0].plot(rvals, mx[:,rr], linestyle='-', marker=marker, color=colours[rr], label=method)
                ax[0].plot(rvals, sx[:,rr], linestyle=':', color=colours[rr])
                ax[0].plot(rvals[ind], sx[ind,rr], marker, markeredgecolor=colours[rr])
                ax[1].plot(rvals, mn[:,rr], linestyle='-.', marker=marker, color=colours[rr], label=method)
                ax[1].plot(rvals, sd[:,rr], linestyle=':', color=colours[rr])
                ax[1].plot(rvals[ind], sd[ind,rr], marker, markeredgecolor=colours[rr])
            case "mean":
                ax.plot(rvals, mn[:,rr], linestyle='-', marker=marker, color=colours[rr], label=method)
                ax.plot(rvals, sd[:,rr], linestyle=':', color=colours[rr])
                ax.plot(rvals[ind], sd[ind,rr],  marker, color=colours[rr])
            case _:
                ax.plot(rvals, mx[:,rr], linestyle='-', marker=marker, color=colours[rr], label=method)
                ax.plot(rvals, sx[:,rr], linestyle=':', color=colours[rr])
                ax.plot(rvals[ind], sx[ind,rr], marker, color=colours[rr])
    if doublePlot:
        for ii in range(2):
            ax[ii].set_xlabel('rms noise')
            # ax[ii].set_xlim([0,1.5])
            makeSecondXAxis(ax[ii], newTickPositions, label='S/N')
            if (Nm==1):
                ax[ii].set_title("comparison max/mean bias" if ii==0 else method)
            else:
                #ax[ii].set_title("comparison of methods")
                ax[ii].legend()
                
        ax[0].set_ylabel('maximum -- average bias, : std of bias')
        ax[1].set_ylabel('mean -- average bias, : std of bias')
        
    else:
        ax.set_xlabel('rms noise')
        # ax.set_xlim([0, 1.5])
        makeSecondXAxis(ax, newTickPositions, label='S/N')
        ylbl = ('maximum' if biasType=="max" else 'mean')+' -- average, : std of bias' 
        ax.set_ylabel(ylbl)
        if (Nm == 1):
            ax.set_title(method)
        else:
            #ax.set_title("comparison of methods")
            ax.legend(fontsize='x-small', loc='lower right')
    
    return fig, ax


def makeNoisePlot(fvals, dataMean, dataStd, param, noiseSamples, rind = None, dataC = None):
    """ generate a noise plot for the noise data in dataMean and dataStd, both 
    of which are 3 dimensional arrays [frequency, rms, method]
    values of rms and methods can be found in param (output of the noisy data
                                                     generation function)
    rind gives the indices in the rms axis for which plots are performed
    dataC (if given) is a tuple of skewness and kurtosis.
    """
    Nd, Nr, Nm = dataMean.shape
    f = []
    a = []
    p = []
    
    if (Nd>1) and (Nr>1) and (Nm>1):    # that is too much
        # we have both mutliple rms and multiple methods -> one figure(set) per method
        for mm in range(Nm):
            dM = dataMean[:,:,[mm]]
            dS = dataStd[:,:,[mm]]
            if dataC is not None:
                dSk = dataC[0][:,:,[mm]]
                dKu = dataC[1][:,:,[mm]]
                dC = (dSk, dKu)
            else:
                dC = None
            fig, ax, paramOut = makeNoisePlot(fvals, dM, dS, \
                                 (param[0], param[1], [param[2][mm]]), noiseSamples,rind, dC)
            f.extend(fig)
            a.extend(ax)
            p.extend(paramOut)
        
        return f, a, p
    
    
    if (Nr>1):  # plot rms noise vs bias/std
        fig, ax = plt.subplots(1,1)
        f.append(fig)
        a.append(ax)
        mk = []; ms = []; xs =[]
        
        for ii in range(Nr):
            mk.append(abs(dataMean[:,ii,0]-fvals).max())
            ms.append(abs(dataStd[:,ii,0]).mean())
            xs.append(abs(dataStd[:,ii,0]).max())
        method = param[2][0]
        rms = param[1]
        
        ax.plot(rms, mk, 'b-o', label='bias')
        k, kerr = noiseBehaviour(rms, ms, noiseSamples)
        print(fr'{method}: $\kappa$ mean = {k:.3e} ± {kerr:.3e}')
        k, kerr = noiseBehaviour(rms, xs, noiseSamples)
        print(fr'{method}: rms $\kappa$ max = {k:.3e} ± {kerr:.3e}')
        ax.plot(rms, xs, 'b:*', label=fr'max std ($\kappa$ = {k:.3f})')
        
        ax.set_xlabel('rms noise')
        ax.set_ylabel('maximum bias/standard deviation')
        ax.legend()
        ax.set_title(method)
        
        p.append((0,method))
        
        # make noise vs freqency plots for relevant rms (default: 3)
        if rind is None:
            if Nr<4:
                rind = range(Nr)
            else:
                rind= [0, int(np.floor(Nr/3)), int(np.floor(2/3*Nr)), Nr-1]
        for r in rind:
            dM = dataMean[:,[r],:]
            dS = dataStd[:,[r],:]
            dC = (dataC[0][:,[r],:], dataC[1][:,[r],:])
            fig, ax, paramOut = makeNoisePlot(fvals, dM, dS, \
                          (param[0], [param[1][r]], param[2]), noiseSamples,None, dC)
            f.extend(fig)
            a.extend(ax)
            p.extend(paramOut)
        
    else:
        fig, ax = plt.subplots(Nm, 1 if dataC is None else 2, dpi=600)
        f.append(fig)
        a.append(ax)
        if Nm==1:
            ax = [ax]
    
        for ii, method in enumerate(param[2]):
           
            if dataC is not None:
                ax1 = ax[ii][0]
            else:
                ax1 = ax[ii]
                
            rms = param[1][0]
            p.append((rms, method))
          
            # in this branch we only have one rms value (-> second index 0)
            dM = dataMean[:,0,ii]
            dS = dataStd[:,0,ii]
            
            bias = dM - fvals
            ax1.plot(fvals, bias, 'b.')
            #ax1.plot(fvals, bias+dS, 'b:', fvals, bias-dS, 'b:')
            #ax1.plot(fvals, dS, 'b:', fvals, dS, 'b:')
            ax1.plot(fvals, bias+2*dS, 'b:', fvals, bias-2*dS, 'b:')
            ax1.set_xlabel('input frequency ($\\Delta f$)')
            ax1.set_ylabel('bias')
            fmethod = fr.formalMethodName(method)
            ax1.set_title(f'{fmethod:s} - (rms={rms:.2f})' if (dataC is None) else f'{fmethod:s}')
            #ax1.legend(['average', 'std'])
            #ax1.legend(['average', '95% confidence interval'])
            #print(f'{method:s} with rms {rms:.2f}: maximum bias is {abs(dM-fvals).max():.5f}; max std of bias is {dS.max():.5f}')
            
            if dataC is not None:
                ax2 = ax[ii][1]
                dSkew = dataC[0][:,0,ii]
                dKurt = dataC[1][:,0,ii]
                ax2.plot(fvals, dSkew, 'b-', label='skewness')
                ax2.plot(fvals, dKurt, 'r-', label='kurtosis')
                ax2.set_xlabel('input frequency ($\\Delta f$)')
                #ax2.set_ylabel('skewness/kurtosis')
                ax2.yaxis.set_label_position('right')
                ax2.yaxis.tick_right()
                ax2.set_title(f'(rms={rms:.2f})')
                ax2.legend()
            
        else:
            pass # to be written -- plogenerateDatat of rms vs mean/max error?
        
    return f, a, p


def makeKappaPlot(fC, fDelta, stdData, param):
    """ generate a kappa plot for the kappa data in kappa, a 3 dimensional 
    array [frequency, rms, method]   
    values of rms and methods can be found in param (output of the noisy data
                                                     generation function)
    """
    Nf, Nr, Nm = stdData.shape
    Nd = len(fDelta)
    Nc = len(fC)
    f = []
    a = []
    p = []
    c = []
    
    if (Nm>1):    
        # we have multiple methods -> one figure per method
        for mm in range(Nm):
            dS = stdData[:,:,[mm]]
                        
            fig, ax, paramOut, cM = makeKappaPlot(fC, fDelta, dS, \
                                 (param[0], param[1], [param[2][mm]]))
            f.extend(fig)
            a.extend(ax)
            p.extend(paramOut)
            c.append(cM)
        return f, a, p, c
    
    fig, ax = plt.subplots(1,1)
    f.append(fig)
    a.append(ax)
    
    method = param[2][0]
    rmsval = param[1]
    
    kappa = np.zeros(shape = (Nc, Nd*Nr))
    for rr in range(Nr):
        ## TODO: try a + b x?
        stdData /= rmsval[rr]    # calculate kappa estimates (std = kappa * rms)
        
    for ff in range(Nc):
        # all kappa values for a central frequency
        kappa[ff, :] = np.reshape(stdData[Nd*ff:Nd*(ff+1),:,0], (Nd*Nr,))
        
    mk = kappa.mean(axis=1)
    sk = kappa.std(axis=1)/np.sqrt(Nd*Nr)   # rough estimation
    
    cM = np.zeros(shape=(Nc, Nc))
    vk = sk**2; # variance
    for ii in range(Nc-1):
        for jj in range(ii+1, Nc):
            sij = np.sqrt(vk[ii]+vk[jj])
            delta = np.abs(mk[ii]-mk[jj])/sij
            cM[ii,jj] = delta # delta >2 indicates significant difference
            cM[jj,ii] = delta
            
    merr = np.array([mk-kappa.min(axis=1), kappa.max(axis=1)-mk])
    
    ax.errorbar(fC, mk, yerr=merr, fmt='b-o', label=r'$\kappa_\mathrm{mean}$')
   
    ax.set_xlabel(r'centre frequency ($\Delta f$)')
    ax.set_ylabel(r'$\kappa_\mathrm{mean}$')
    #ax.legend()
    ax.set_title(method)
    
    p.append((0,method))
    
    print(f'kappa values for method: {method}')
    
    for ii in range(Nc):
        print(f'  {fC[ii]:.0f}:\t {mk[ii]:.3e} ± {sk[ii]:.3e}')
    # this should be the same as calculating over all kappa?   
    print(f' \t average: {mk.mean():.3e} ± {sk.mean()/np.sqrt(Nc):.3e}')   
        
    return f, a, p, cM

def saveFigure(fig, filename, directory):
    #directory = f'simResults_{NdataPoints}_{centralFrequency:.1f}_{noiseSamples:d}'
    if directory is None:
        d = ''
    else:
        d = directory + '/'
        
    fig.savefig(d+filename, dpi=300, bbox_inches='tight')