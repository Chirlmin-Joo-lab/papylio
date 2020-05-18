# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:35:16 2020

@author: pimam
"""
import numpy as np
import matplotlib.pyplot as plt
from trace_analysis.analysis import common_PDF

def comparative_plot(dwells, name, dist='offtime', trace='red', binsize='auto',
         scale='log', style='dots', color='from_trace', fit_result=None):

    if fit_result is not None:
        tcut = fit_result.tcut[0]
        Tmax = fit_result.Tmax[0]
        Ncut = fit_result.Ncut[0]
        dwells = dwells[dwells <= Tmax]
    else:
        Tmax = dwells.max()
        tcut = dwells.min()

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
        if fit_result.model[0] == '1Exp':
            tau, error = fit_result.value[0], fit_result.error[0]

            print(f'plotting 1Exp fit')
            time, fit = common_PDF.Exp1(tau,
                                        tcut=tcut, Tmax=Tmax)
            label = f'\n tau={tau:.1f}'
            if error != 0:
                label += f'$\pm$ {error:.1f}'

        elif fit_result.model[0] == '2Exp':
            p1, errp1 = fit_result.value[0], fit_result.error[0]
            tau1, err1 = fit_result.value[1], fit_result.error[1]
            tau2, err2 = fit_result.value[2], fit_result.error[2]
            print(f'errors: ', errp1, err1, err2)
            time, fit = common_PDF.Exp2(p1, tau1, tau2,
                                        tcut=tcut, Tmax=Tmax)
            label = f'\n p1={p1:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}'

        elif fit_result.model[0] == '3Exp':
            p1, errp1 = fit_result.value[0], fit_result.error[0]
            p2, errp2 = fit_result.value[1], fit_result.error[1]
            tau1, err1 = fit_result.value[2], fit_result.error[2]
            tau2, err2 = fit_result.value[3], fit_result.error[3]
            tau3, err3 = fit_result.value[4], fit_result.error[4]
            print(f'errors: ', errp1, errp2, err1, err2, err3)
            time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3,
                                        tcut=tcut, Tmax=Tmax)
            label = f'\n p1={p1:.2f}, p2={p2:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}'

        elif fit_result.model[0] == '4Exp':
            p1, errp1 = fit_result.value[0], fit_result.error[0]
            p2, errp2 = fit_result.value[1], fit_result.error[1]
            p3, errp3 = fit_result.value[2], fit_result.error[2]
            tau1, err1 = fit_result.value[3], fit_result.error[3]
            tau2, err2 = fit_result.value[4], fit_result.error[4]
            tau3, err3 = fit_result.value[5], fit_result.error[5]
            tau4, err4 = fit_result.value[6], fit_result.error[6]
            print(f'errors: ', errp1, errp2, errp3, err1, err2, err3, err4)
            time, fit = common_PDF.Exp4(p1, p2, p3, tau1, tau2, tau3, tau4,
                                        tcut=tcut, Tmax=Tmax)
            label = f'\n p1={p1:.2f}, p2={p2:.2f}, p3={p3:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}, tau4={int(tau4)}'

        plt.plot(time, fit, color='k', label=f'{fit_result.model[0]}fit{label}')

    if scale in ['Log', 'Log-Log']:
        plt.yscale('log')

    if scale == 'Log-Log':
        plt.xscale('log')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'{dist} (s)')
    plt.tight_layout()
    plt.show()
    return