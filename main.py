def main():
    import numpy as np
    from scipy.fftpack import fft

    import lib.generate as gen
    import lib.helper_functions as helper
    

    #%% start script / figure generation
    SNRanalyse = True

    frequencies = 1001
    NdataPoints = 2048

    centralFrequency = 7    # based on Pegah et al. (5 µm beads)
    frequencyDiv = 0.6
    fRange = helper.inclusiveRange(centralFrequency-frequencyDiv,centralFrequency+frequencyDiv,N=frequencies)

    compare_methods = True

    #%% Compare the best 3 methods without noise
    if compare_methods:
        data, _params = gen.generateData(N=NdataPoints, f = fRange)
        fdata = fft(data-np.mean(data, axis=1, keepdims=True))

        methods = ["MacLeod", "JacobsenMod", "Quinns2nd"]
        method_names = ["MacLeod", "Jacobsen modified", r"Quinns $2^{nd}$"]
        colors = ["#38B6FF", "#FF8100", "#00EE00"]
        print(f"comparing: {', '.join(methods)}")
        helper.compareMethods(fdata, fRange, methods, image_name="bias-comparison", method_names=method_names, colors=colors)

    if SNRanalyse:
        single_params = (NdataPoints, None, frequencies, centralFrequency, fRange, None, None)
        SNR_main(single_params)
    else:
        offset = 500
        modulation = 50
        single_params = (NdataPoints, methods, frequencies, centralFrequency, fRange, offset, modulation)
        single_main(single_params, generateNoise=True, useMP=True, SNRanalyse=False, modulations=[modulation])


def single_main(single_params, generateNoise, useMP, SNRanalyse, modulations, loop=1):
    import lib.main_func as mf
    
    (NdataPoints, methods, frequencies, centralFrequency, fRange, offset, modulation) = single_params

    poisson_offset = offset
    poisson_modulation = modulation

    generateNoise = True
    noiseSamples = 100 # 2500 for 1% error in std, 163 for 5%, 50 for 10%, 1000 gives 1.7%

    N_methods = len(methods)
    N_data = frequencies

    useMP = True #!

    noise_params = (NdataPoints, noiseSamples, N_data, methods, N_methods, frequencies, centralFrequency, fRange, poisson_offset, poisson_modulation)

    run_number = f"{loop}/{len(modulations)}"

    if SNRanalyse:
        rms_pstd, var_pstd = mf.noiseAnalysis(noise_params, generateNoise, useMP, SNRanalyse, modulations, run_number)
        return rms_pstd, var_pstd
    else:
        mf.noiseAnalysis(noise_params, generateNoise, useMP, SNRanalyse, modulations, run_number)

def SNR_main(single_params):
    (NdataPoints, methods, frequencies, centralFrequency, fRange, offset, modulation) = single_params
    import numpy as np
    import matplotlib.pyplot as plt
    
    methods = ['macleod', 'jacobsenmod', 'quinns2nd']
    offset = 500
    SNR = np.linspace(-10, 20, 21)
    modulations = []
    rms_per_snr = []
    var_per_snr = []
    for i, snr in enumerate(SNR):
        modulations.append(float(np.sqrt(2 * offset * 10**(snr / 10))))
    for i in range(len(modulations)):
        single_params = (NdataPoints, methods, frequencies, centralFrequency, fRange, offset, modulations[i])
        snr_rms, snr_var = single_main(single_params, generateNoise=True, useMP=True, SNRanalyse=True, modulations=modulations, loop=i+1)
        rms_per_snr.append(snr_rms)
        var_per_snr.append(snr_var)
    rms_per_snr_T = np.array(rms_per_snr).T
    var_per_snr_T = np.array(var_per_snr).T

    fig, ax1 = plt.subplots(1, 1, figsize=(16,9), facecolor="#FFFFFF")
    method_names = ["MacLeod", "Jacobsen modified", r"Quinns $2^{nd}$"]
    colors = ["#38B6FF", "#FF8100", "#00EE00"]
    for i in range(len(rms_per_snr_T)):
        ax1.plot(SNR, rms_per_snr_T[i], label=f"{method_names[i]}", linewidth=5, color=colors[i])
    ax1.set_xlabel("SNR (dB)", fontsize=20)
    ax1.set_ylabel("RMS", fontsize=20)
    plt.rcParams.update({'font.size': 20})
    ax1.set_facecolor("#FFFFFF")
    ax1.set_ylim(1e-3, 1e-1)
    ax1.grid(True)
    ax1.legend()
    ax1.set_yscale('log')
    fig.savefig("poster/rmsplot.png")
    plt.close()
    print(f"saved rms plot to poster/rmsplot.png")

if __name__ == "__main__":
    import os
    main()
    
    os._exit(0)
