# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:35:16 2020

@author: pimam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trace_analysis.analysis import common_PDF

def comparative_plot(dwells, name, dist='offtime', trace='red', binsize='auto',
         scale='log', style='dots', color='from_trace', fit_result=None):

    if fit_result is not None:
        tcut = fit_result.tcut[0]
        Tmax = fit_result.Tmax[0]
        Ncut = fit_result.Ncut[0]
        if Ncut > 0:
            dwells = dwells[dwells <= Tmax]
    else:
        Tmax = dwells.max()
        tcut = dwells.min()
        Ncut = 0

    try:
        bsize = float(binsize)
        if scale == 'Log-Log':
            bin_edges = 10**(np.arange(np.log10(min(dwells)), np.log10(max(dwells)) + bsize, bsize))
        else:
            bin_edges = np.arange(min(dwells), max(dwells) + bsize, bsize)
    except ValueError:
        if binsize == 'Auto':
            binsize = 'auto'
        bin_edges = binsize

    values, bins = np.histogram(dwells, bins=bin_edges, density=True)

    # Determine position of bins
    if scale == 'Log-Log':
        centers = (bins[1:] * bins[:-1])**0.5  # geometric average of bin edges
    else:
        centers = (bins[1:] + bins[:-1]) / 2.0

    # combine bins until they contain at least one data point (for y-log plots)
    if scale in ['Log', 'Log-Log']:
        izeros = np.where(values == 0)[0]
        print('izeros', izeros)
        j = 0
        while j < len(izeros):
            i = j
            j += 1
            while j < len(izeros) and izeros[j] - izeros[j-1] == 1:
                j += 1
            values[izeros[i]:(izeros[i]+j-i+1)] = np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1)

    plt.figure(f'Histogram {trace} {dist}s {name}', figsize=(4, 3), dpi=200)

    if color == 'from_trace':
        if dist == 'offtime':
            color = 'r'*(trace == 'red') + 'g'*(trace == 'green') + \
                    'b'*(trace == 'FRET') + 'sandybrown'*(trace == 'total')
        if dist == 'ontime':
            color = 'firebrick'*(trace == 'red') + 'olive'*(trace == 'green') + \
                    'darkviolet'*(trace == ' FRET') + 'saddlebrown'*(trace == 'total')

    label = f'{dist} pdf, N={dwells.size}'
    if tcut > 0:
        label = label + f', tcut={tcut:.2f}'
    if Ncut > 0:
        label = label + f', Ncut={int(Ncut)}'

    if style == 'dots':
        plt.plot(centers, values, '.', color=color, label=label)
    if style == 'bars':
        plt.bar(centers, values, color=color, label=label,
                width=(bins[1] - bins[0]))
    if style == 'line':
        plt.plot(centers, values, '-', lw=2, color=color, label=label)

    if fit_result is not None:
        print(fit_result)
        idx_mdl = np.where(~pd.isnull(fit_result.model))[0]
        BICs = np.zeros(3)
        for i in idx_mdl:
            if fit_result.model[i] == '1Exp':
#                BICs[0] = fit_result.BIC[i]
                tau, error = fit_result.value[i], fit_result.error[i]
                time, fit = common_PDF.Exp1(tau,
                                            tcut=tcut, Tmax=Tmax)
                label = f'\n tau={tau:.1f}'
                if error != 0:
                    label += f'$\pm$ {error:.1f}'

            if fit_result.model[i] == '2Exp':
                BICs[0] = fit_result.BIC[i]
                p1, errp1 = fit_result.value[i], fit_result.error[i]
                tau1, err1 = fit_result.value[i+1], fit_result.error[i+1]
                tau2, err2 = fit_result.value[i+2], fit_result.error[i+2]
                print(f'errors: ', errp1, err1, err2)
                time, fit = common_PDF.Exp2(p1, tau1, tau2,
                                            tcut=tcut, Tmax=Tmax)
                label = f'\n p1={p1:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}'

            if fit_result.model[i] == '3Exp':
                BICs[1] = fit_result.BIC[i]
                p1, errp1 = fit_result.value[i], fit_result.error[i]
                p2, errp2 = fit_result.value[i+1], fit_result.error[i+1]
                tau1, err1 = fit_result.value[i+2], fit_result.error[i+2]
                tau2, err2 = fit_result.value[i+3], fit_result.error[i+3]
                tau3, err3 = fit_result.value[i+4], fit_result.error[i+4]
                print(f'errors: ', errp1, errp2, err1, err2, err3)
                time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3,
                                            tcut, Tmax)
                label = f'\n p1={p1:.2f}, p2={p2:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}'

            if fit_result.model[i] == '4Exp':
                BICs[2] = fit_result.BIC[i]
                p1, errp1 = fit_result.value[i], fit_result.error[i]
                p2, errp2 = fit_result.value[i+1], fit_result.error[i+1]
                p3, errp3 = fit_result.value[i+2], fit_result.error[i+2]
                tau1, err1 = fit_result.value[i+3], fit_result.error[i+3]
                tau2, err2 = fit_result.value[i+4], fit_result.error[i+4]
                tau3, err3 = fit_result.value[i+5], fit_result.error[i+5]
                tau4, err4 = fit_result.value[i+6], fit_result.error[i+6]
                print(f'errors: ', errp1, errp2, errp3, err1, err2, err3, err4)
                time, fit = common_PDF.Exp4(p1, p2, p3, tau1, tau2, tau3, tau4,
                                            tcut=tcut, Tmax=Tmax)
                label = f'\n p1={p1:.2f}, p2={p2:.2f}, p3={p3:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}, tau4={int(tau4)}'

            plt.plot(time, fit, label=f'{fit_result.model[i]}fit{label}')

    numdim = np.arange(len(BICs))
    BICs = BICs - BICs.min()
    print('BICS',BICs)
    
    if scale in ['Log', 'Log-Log']:
        plt.yscale('log')

    if scale == 'Log-Log':
        plt.xscale('log')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'{dist} (s)')
    plt.tight_layout()
    
    plt.figure('BICplot')
    numdim = [2, 3, 4]#np.arange(2, len(BICs), dtype=int)
    BICs = BICs - BICs.min()
    plt.plot(numdim, BICs, '-r')
    plt.xticks(numdim)
    plt.xlabel('Number of binding phases')
    plt.ylabel('$\Delta$BIC')

    
    plt.show()
    return

def plot_bootstrap(bootresults, bestfit, Nbins, model='N'):
     # Getting measures and plotting the parameter values found
    Nfits = np.size(bootresults, 0)
    print('Nfits', Nfits)
    bootP1 = None
    bootP2 = None
    bootP3 = None
    boottau1 = None
    boottau2 = None
    boottau3 = None
    boottau4 = None
    avg_array = np.zeros(7)
    std_array = np.zeros(7)

    if model == '2Exp':
        bootP1 = bootresults[:,0][~np.isnan(bootresults[:,0])]
        boottau1 = bootresults[:,1][~np.isnan(bootresults[:,1])]
        boottau2 = bootresults[:,2][~np.isnan(bootresults[:,2])]
        bootNcut = bootresults[:,3][~np.isnan(bootresults[:,3])]

        avg_array[0] = [np.average(bootP1)]
        avg_array[3:5] = [np.average(boottau1), np.average(boottau2)]
        std_array[0] = [np.std(bootP1)]
        std_array[3:5] = [np.std(boottau1), np.std(boottau2)]

        boot_results = pd.DataFrame({'p1': bootP1, 'tau1': boottau1,
                                     'tau2': boottau2, 'Ncut': bootNcut})
        boot_stats = pd.DataFrame({'avg': avg_array, 'std': std_array})
        boot_results =pd.concat([boot_results, boot_stats], axis=1)

    if model == '3Exp':
#        i = np.where(bootresults[:,4] > 590)[0]
#        bootresults[i,:] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        bootP1 = bootresults[:,0][~np.isnan(bootresults[:,0])]
        bootP2 = bootresults[:,1][~np.isnan(bootresults[:,1])]
        boottau1 = bootresults[:,2][~np.isnan(bootresults[:,2])]
        boottau2 = bootresults[:,3][~np.isnan(bootresults[:,3])]
        boottau3 = bootresults[:,4][~np.isnan(bootresults[:,4])]
        bootNcut = bootresults[:,5][~np.isnan(bootresults[:,5])]

        avg_array[:2] = [np.average(bootP1), np.average(bootP2)]
        avg_array[3:6] = [np.average(boottau1), np.average(boottau2),
                         np.average(boottau3)]
        std_array[:2] = [np.std(bootP1), np.std(bootP2)]
        std_array[3:6] = [np.std(boottau1), np.std(boottau2), np.std(boottau3)]

        boot_results = pd.DataFrame({'p1': bootP1, 'p2': bootP2,
                                     'tau1': boottau1, 
                                     'tau2': boottau2, 'tau3': boottau3,
                                     'Ncut': bootNcut})
        boot_stats = pd.DataFrame({'avg': avg_array, 'std': std_array})
        boot_results =pd.concat([boot_results, boot_stats], axis=1)

    print(boot_results)

    if avg_array[0] != 0:  # p1
        plt.figure()
        plt.hist(bootP1, bins=Nbins)
        plt.vlines(avg_array[0], 0, round(Nbins/2), label=f'avg:{avg_array[0]:.2f} std:{std_array[0]:.2f}')
        plt.title(f'Fit values for P1 Nfits: {Nfits} Nbins: {Nbins}')
        plt.legend()
    if avg_array[1] != 0:  # p2
        plt.figure()
        plt.hist(bootP2, bins=Nbins)
        plt.vlines(avg_array[1], 0, round(Nbins/2), label=f'avg:{avg_array[1]:.2f} std:{std_array[1]:.2f}')
        plt.title(f'Fit values for P2 Nfits: {Nfits} Nbins: {Nbins}')
        plt.legend()
    if avg_array[2] != 0:  # p3
        plt.figure()
        plt.hist(bootP3, bins=Nbins)
        plt.vlines(avg_array[2], 0, round(Nbins/2), label=f'avg:{avg_array[2]:.2f} std:{std_array[2]:.2f}')
        plt.title(f'Fit values for P3 Nfits: {Nfits} Nbins: {Nbins}')
        plt.legend()
    if avg_array[3] != 0:  # tau1
        plt.figure()
        plt.hist(boottau1, bins=Nbins)
        plt.vlines(avg_array[3], 0, round(Nbins/2), label=f'avg:{avg_array[3]:.1f} std:{std_array[3]:.2f}')
        plt.title(rf'Fit values for $\tau$1 Nfits: {Nfits} Nbins: {Nbins}')
        plt.legend()
    if avg_array[4] != 0: # tau2
        plt.figure()
        plt.hist(boottau2, bins=Nbins)
        plt.vlines(avg_array[4], 0, round(Nbins/2), label=f'avg:{avg_array[4]:.0f} std:{std_array[4]:.1f}')
        plt.title(rf'Fit values for $\tau$2 Nfits: {Nfits} Nbins: {Nbins}')
        plt.legend()
    if avg_array[5] != 0:  # tau3
        plt.figure()
        plt.hist(boottau3, bins=Nbins)
        plt.vlines(avg_array[5], 0, round(Nbins/2), label=f'avg:{avg_array[5]:.0f} std:{std_array[5]:.1f}')
        plt.title(rf'Fit values for $\tau$3 Nfits: {Nfits} Nbins: {Nbins}')
        plt.legend()
    if avg_array[6] != 0:  # tau4
        plt.figure()
        plt.hist(boottau4, bins=Nbins)
        plt.vlines(avg_array[6], 0, round(Nbins/2), label=f'avg:{avg_array[6]:.0f} std:{avg_array[6]:.1f}')
        plt.title(rf'Fit values for $\tau$4 Nfits: {Nfits} Nbins: {Nbins}')
        plt.legend()

    return boot_results
