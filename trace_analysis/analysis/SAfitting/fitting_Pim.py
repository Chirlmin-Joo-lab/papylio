# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:42:29 2019

@author: pimam
"""
if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path, PureWindowsPath
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = Path(__file__).parents[3]
    sys.path.insert(0, str(p))
#    mainPath = PureWindowsPath('F:\\20191121_dcas9_tracrRNA_DNA6_20_2\\#2_strept_1nMdCas9Sigma_10nMDNA6_movies\\peaks5collect4_red\\analysisfiles')

    from trace_analysis import Experiment
    from trace_analysis.analysis.SAfitting import fitting_function as fitfunc
    import numpy as np
    import pandas as pd
    import matplotlib.pylab as plt
    import seaborn as sns
    sns.set(style="dark")
    sns.set_color_codes()

#    # Import data and prepare for fitting
#    exp = Experiment(mainPath)
#    file = exp.files[0]
#    filename = './'+file.name+'_dwells_data.xlsx'
#    print(filename)
#    data = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})
#
#    if len(exp.files) > 1:  # time of traces should be of the same length
#        for file in exp.files[1:]:
#            filename = './'+file.name+'_dwells_data.xlsx'
#            print(filename)
#            data2 = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})
#            data = data.append(data2, ignore_index=True)

#    dwelltype = 'offtime'
#    dwells_all = []
#    dwells = data[dwelltype].values
#    dwells = dwells[~np.isnan(dwells)]
#    dwells_all.append(dwells)
#    dwells_all = np.concatenate(dwells_all)

    filename = '2exp_N=10000_rep=1_tau1=10_tau2=200_a=0.5'
    dwells_all = np.load('./data/2exp_N=10000_rep=1_tau1=10_tau2=200_a=0.5.npy')
    dwells_all = dwells_all[0]

    # Start fitting
    mdl = '2Exp'
    include_over_Tmax = True
    Nfits = 3
    bootstrap = True
    boot_repeats = 40
    fitdata = fitfunc.fitting(dwells_all, mdl, Nfits, include_over_Tmax, bootstrap, boot_repeats)
    if bootstrap is True:
        fitdata.to_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}_Nfits{Nfits}.csv', index=False)
    else:
        fitdata.to_csv(f'{mdl}_inclTmax_{include_over_Tmax}_Nfits{Nfits}.csv', index=False)

#    newdata = pd.read_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}.csv')

    # Getting measures and plotting the parameter values found
    taubnd = 100
    fitP1 = []
    fittau1 = []
    fittau2 = []
    for i in range(0, len(fitdata['tau1'])):
        if fitdata['tau1'][i] > taubnd:
            fittau2.append(fitdata['tau1'][i])
            fitP1.append(1-fitdata['P1'][i])
        else:
            fittau1.append(fitdata['tau1'][i])
            fitP1.append(fitdata['P1'][i])
        if fitdata['tau2'][i] > taubnd:
            fittau2.append(fitdata['tau2'][i])
        else:
            fittau1.append(fitdata['tau2'][i])

    P1_avg = np.average(fitP1)
    tau1_avg = np.average(fittau1)
    tau2_avg = np.average(fittau2)
    P1_std = np.std(fitP1)
    tau1_std = np.std(fittau1)
    tau2_std = np.std(fittau2)
    Nbins = 20

    plt.figure()
    plt.hist(fitP1, bins=Nbins)
    plt.vlines(P1_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(P1_avg))
    plt.title(f'Fit values for P1 Nfits: {boot_repeats} Nbins: {Nbins}')
    plt.legend()
    plt.figure()
    plt.hist(fittau1, bins=Nbins)
    plt.vlines(tau1_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(tau1_avg))
    plt.title(rf'Fit values for $\tau$1 Nfits: {boot_repeats} Nbins: {Nbins}')
    plt.legend()
    plt.figure()
    plt.hist(fittau2, bins=Nbins)
    plt.vlines(tau2_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(tau2_avg))
    plt.title(rf'Fit values for $\tau$2 Nfits: {boot_repeats} Nbins: {Nbins}')
    plt.legend()

    # Quick plot of best fit with histogram
    plt.figure()
    Tmax = dwells_all.max()
    Tcut = Tmax - 10
    dwells = dwells_all[dwells_all < Tcut]
    Ncut = dwells_all[dwells_all >= Tcut].size
    print(Ncut)
    timearray = np.linspace(0, Tmax, 1000)
    values, bins = np.histogram(dwells, bins=60, density=True)
    centers = (bins[1:] + bins[:-1]) / 2.0
    plt.plot(centers, values, '.', label=f'Dwells')
    if mdl == '2Exp':
        bestparam = [fitdata['P1'][-1], fitdata['tau1'][-1], fitdata['tau2'][-1]]
        plt.title('Best double exponential fit found Nbins=60 N='+'{0:.0f}'.format(len(dwells_all)))
        bestfit, Pcutbest = fitfunc.P2expcut(timearray, bestparam, Tcut, Ncut)
        plt.plot(timearray, bestfit, label='P1:'+"{0:.2f}".format(bestparam[0])
                 + "\n"+r'$\tau$1:'+"{0:.1f}".format(bestparam[1])+"\n"
                 + r'$\tau$2:'+"{0:.1f}".format(bestparam[2]))
    else:
        if len(fitdata['tau1']) > 1:
            bestparam = fitdata['tau1'][-1]
        else:
            bestparam = fitdata['tau1']
        plt.title('Best single exponential fit Nbins=60 N='+'{0:.0f}'.format(len(dwells_all)))
        bestfit, MLtau = fitfunc.ML1expcut(dwells, Tcut, Ncut)
        plt.plot(timearray, bestfit, label=r'$\tau$1:'+"{0:.1f}".format(MLtau)
                 + "\n Ncut:"+str(Ncut))
    plt.xlabel('dwell time (sec)')
    plt.ylabel('log prob. density')
    plt.legend()




#    # Plot data with double and single exponential fit
#    plt.figure()
#    plt.semilogy(centers, values, '.', label=f'Dwells with Ncut:{Ncut}')
#    plt.semilogy(timearray, bestfit, label='P1:'+"{0:.2f}".format(bestparams[0])+"\n"+r'$\tau$1:'+"{0:.1f}".format(bestparams[1])+"\n"+r'$\tau$2:'+"{0:.1f}".format(bestparams[2]))
#    singlexp = 1/avg_dwells*np.exp(-timearray/avg_dwells)
#    plt.plot(timearray, singlexp, 'orange', label = rf'$\tau$:{avg_dwells:.1f}')
#    plt.xlabel('dwell time (sec)')
#    plt.ylabel('log prob. density')
#    plt.legend(fontsize='x-large')
#  #  plt.savefig(f'{len(exp.files)}files_1_2expfit__compared.png', dpi=200)

