# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:32:58 2019

@author: ikatechis
"""

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    import SAfitting
    import common_PDF
else:
    from trace_analysis.analysis import SAfitting_based_on_Z as SAfitting
    from trace_analysis.analysis import common_PDF
# import SAfitting
sns.set(style="ticks")
sns.set_color_codes()


def analyze(dwells_data, dataset_name, dist, configuration):
    conf = configuration
    # find the Tmax until which data is selected
    d = apply_config_to_data(dwells_data, dist, conf)
    figures = []
    fit_data = []
    keys_with_data = []  # keys refer to 'red', 'green', 'total', 'FRET'
    for key in d.keys():
        if d[key].empty:  # check if the dataframe is empty
            print(f'{dist} dataFrame for {key} is empty')
            continue
        dwells = d[key].loc[:, dist].values
        dwells = dwells[dwells > 0]
        print(np.size(dwells), 'dwells selected')
        if conf['FitBool']:
            if conf['tcutBool']:
                dwells, t_cut = Short_time_cutoff(dwells)
            else:
                t_cut = []
            fit_res = fit(dwells, model=conf['model'],
                          dataset_name=dataset_name,
                          Nfits=int(conf['Nfits']),
                          include_over_Tmax=conf['TmaxBool'],
                          bootstrap=conf['BootBool'],
                          boot_repeats=int(conf['BootRepeats']))
            fit_data.append(fit_res)
        else:
            fit_res = None
        print(f'plotting {key} {dist}')
        figure = plot(dwells, dataset_name, dist, trace=key,
                      binsize=conf['binsize'], t_cut=t_cut,
                      scale=conf['scale'], style=conf['PlotType'],
                      fit_result=fit_res)
        figures.append(figure)
        keys_with_data.append(key)

    if fit_data != []:
        fit_data = pd.concat(fit_data, axis=1, keys=keys_with_data)
    return dwells, figures, fit_data


def Short_time_cutoff(dwells):
    t_cut = 0.6  # a bit higher than Nyquist (2*exposure time)
    short_dwells = np.where(dwells < t_cut)[0]
    print('Short dwells cut:', len(short_dwells))
    dwells_cut = dwells
    dwells_cut[short_dwells] = t_cut
    return dwells_cut, t_cut


def fit(dwells, model='1Exp', dataset_name='Dwells', Nfits=1,
        include_over_Tmax=True, bootstrap=True, boot_repeats=100):

    if model == '1Exp+2Exp':
        fit_result = []
        for model in ['1Exp', '2Exp']:
            result, boots = SAfitting.fit(dwells, model, dataset_name, Nfits,
                                   include_over_Tmax, bootstrap, boot_repeats)
            fit_result.append(result)
        fit_result = pd.concat(fit_result, axis=1, ignore_index=True)
        return fit_result

    fit_result, boots = SAfitting.fit(dwells, model, dataset_name, Nfits,
                                   include_over_Tmax, bootstrap, boot_repeats)
    # print(fit_result)
    return fit_result


def plot(dwells, name, dist='offtime', trace='red', binsize='auto',
         t_cut=None,
         scale='log', style='dots', color='from_trace', fit_result=None):
    if t_cut is not None:
        dwells = dwells[dwells >= t_cut]

    if fit_result is not None:
        if fit_result.Ncut[0] > 0:
            Tcut = dwells.max() - 5  # 5 sec is kind of arbitrary here
            dwells = dwells[dwells < Tcut]

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
    print('values', np.sum(values*np.diff(bins)))
    # Determine position of bins
    if scale == 'Log-Log':
        centers = (bins[1:] * bins[:-1])**0.5  # geometric average of bin edges
    else:
        centers = (bins[1:] + bins[:-1]) / 2.0

    print('bin edges', bin_edges)
    # resolution correction bins
    if scale == 'Log-Log':
        t_reso = 0.3
        ismbin = np.where(np.diff(bin_edges) < t_reso)[0]
        ismbin = np.concatenate((ismbin, [ismbin[-1] + 1]))  # diff only gives index of first compared
        print('small bins ', ismbin)
         
        j = 0
        while j < len(ismbin):
            i = j
            j += 1
            while j < len(ismbin) and centers[ismbin[j]] - centers[ismbin[i]] < t_reso:
                j += 1
            print('startbin ', centers[ismbin[i]])
            print('endbin ', centers[ismbin[i]+j-i])
            print('values ', values[ismbin[i]:(ismbin[i]+j-i)])
#           print('mean value', np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1))
            values[ismbin[i]:(ismbin[i]+j-i)] = np.sum(values[ismbin[i]:(ismbin[i]+j-i)])/(j-i)

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
            # print('jstart ', izeros[i])
            # print('jend ', izeros[i]+(j-i))
            # print('values ', values[izeros[i]:(izeros[i]+j-i+1)])
            # print('mean value', np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1))
            values[izeros[i]:(izeros[i]+j-i+1)] = np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1)

    fig = plt.figure(f'Histogram {trace} {dist}s {name}', figsize=(4, 3), dpi=200)

    if color == 'from_trace':
        if dist == 'offtime':
            color = 'r'*(trace == 'red') + 'g'*(trace == 'green') + \
                    'b'*(trace == 'FRET') + 'sandybrown'*(trace == 'total')
        if dist == 'ontime':
            color = 'firebrick'*(trace == 'red') + 'olive'*(trace == 'green') + \
                    'darkviolet'*(trace == ' FRET') + 'saddlebrown'*(trace == 'total')

    label = f'{dist} pdf, N={dwells.size}'
    if t_cut:
        label = label + f', tcut={t_cut:.1f}'
    if style == 'dots':
        plt.plot(centers, values, '.', color=color, label=label)
    if style == 'bars':
        plt.bar(centers, values, color=color, label=label,
                width=(bins[1] - bins[0]))
    if style == 'line':
        plt.plot(centers, values, '-', lw=2, color=color, label=label)

    if fit_result is not None:
        if fit_result.model[0] == '1Exp':
            tau = fit_result.value[0]
            error = fit_result.error[0]
            Ncut = fit_result.Ncut[0]
            print(f'plotting 1Exp fit')
            time, fit = common_PDF.Exp1(tau, Tmin=min(dwells),
                                        Tmax=centers[-1])
            label = f'\n tau={tau:.1f}'
            if error != 0:
                label += f'$\pm$ {error:.1f}'

        elif fit_result.model[0] == '2Exp':
            p, errp = fit_result.value[0], fit_result.error[0]
            tau1, err1 = fit_result.value[1], fit_result.error[1]
            tau2, err2 = fit_result.value[2], fit_result.error[2]
            Ncut = fit_result.Ncut[0]
            print(fit_result)
            print(f'errors: ', errp, err1, err2)
            time, fit = common_PDF.Exp2(p, tau1, tau2, Tmin=min(dwells),
                                        Tmax=centers[-1])
            label = f'\n p={p:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}'

        elif fit_result.model[0] == '3Exp':
            p1, errp1 = fit_result.value[0], fit_result.error[0]
            p2, errp2 = fit_result.value[1], fit_result.error[1]
            tau1, err1 = fit_result.value[2], fit_result.error[2]
            tau2, err2 = fit_result.value[3], fit_result.error[3]
            tau3, err3 = fit_result.value[4], fit_result.error[4]
            Ncut = fit_result.Ncut[0]
            print(fit_result)
            print(f'errors: ', errp1, errp2, err1, err2, err3)
            time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3,
                                        Tmin=min(dwells), Tmax=centers[-1])
            label = f'\n p1={p1:.2f}, p2={p2:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}'

        if fit_result.Ncut[0] > 0:
            label = f', Ncut={int(Ncut)}' + label

        plt.plot(time, fit, color='k', label=f'{fit_result.model[0]}fit{label}')

    if scale in ['Log', 'Log-Log']:
        plt.yscale('log')

    if scale == 'Log-Log':
        plt.xscale('log')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'{dist} (s)')
    # plt.locator_params(axis='y', nbins=3)
    plt.tight_layout()
    plt.show()
    return fig


def apply_config_to_data(dwells_data, dist, config):
    d = dwells_data
    # Select the requested sides
    side_list = ['l'*bool(config['side']['left']),
                 'm'*bool(config['side']['middle']),
                 'r'*bool(config['side']['right'])]

    if dist == 'offtime':
        d = d[d.side.isin(side_list)]
    if dist == 'ontime':
        d = d[d.onside.isin(side_list)]
    # apply min, max conditions
    if config['max'] in ['Max', 'max']:
        d = d[d[dist] > float(config['min'])]
    else:
        d = d[d[dist] > float(config['min'])]
        d = d[d[dist] < float(config['max'])]

    data = {}

    for key in config['trace'].keys():
        if config['trace'][key]:
            data[key] = d[d['trace'] == key]
        else:
            pass

    return data


if __name__ == '__main__':
    filename = 'C:/Users/iason/Desktop/traceanalysis/trace_analysis/traces/'
    filename += 'hel0_dwells_data.xlsx'

    data = pd.read_excel(filename, index_col=[0, 1], dtype={'kon' :np.str})
    print(data.shape)
    config = {'trace': {'red': True, 'green': False, 'total': False, 'FRET': False},
         'side': {'left': True, 'middle': True, 'right': True},
         'min': '0', 'max': 'max',
         'scale': 'Normal',
         'PlotType': 'dots',
         'binsize': 'auto',
         'FitBool': True,
         'TmaxBool': False,
         'BootBool': False,
         'model': '2Exp',
         'Nfits': '1',
         'BootRepeats': '5'}

    result = analyze(data, 'test', 'offtime', config)
