import numpy as np #scientific computing with Python
import matplotlib.pyplot as plt

from trace_analysis.molecule import Molecule

def histogram(molecules, axis=None, bins=100, parameter='E', molecule_averaging=False, makeFit=False, collection_name='', **kwargs):
    if not molecules: return None
    if not axis:
        axis = plt.gca()
        axis.cla()
    #    if not isinstance(input,list): input = [input]
    #
    #    molecules = list()
    #
    #    for i in input:
    #        if isinstance(i, Molecule):
    #            molecules.append(i)
    #        else:
    #            molecules.append(i.molecules)

    # data = np.concatenate([molecule.intensity[0,:] for molecule in molecules])
    # axis.hist(data,100)
    # data = np.concatenate([molecule.E() for molecule in molecules])

    if parameter is 'E':
        if molecule_averaging:
            data = np.array([np.mean(molecule.E()) for molecule in molecules])
        else:
            data = np.concatenate([molecule.E() for molecule in molecules])
        histogram_FRET(data, bins=bins, axis=axis, **kwargs)

    axis.set_title(f'{parameter} histogram - {collection_name} \n Bins: {bins} - Number of molecules: {len(molecules)} - Molecule averaging: {molecule_averaging}')

    if makeFit:
        fit_hist(data, axis)

def histogram_FRET(data, axis, **kwargs):
    axis.hist(data, range=(0, 1), **kwargs)
    axis.set_xlim((0, 1))
    axis.set_xlabel('FRET')
    axis.set_ylabel('Count')

def fit_hist(data, axis):
    hist, bin_edges = np.histogram(data, 100, range=(0, 1))
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2

    # plt.plot(bin_centers,hist)

    from scipy.signal import butter
    from scipy.signal import filtfilt
    b, a = butter(2, 0.2, 'low')
    output_signal = filtfilt(b, a, hist)
    plt.plot(bin_centers, output_signal)

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(output_signal, prominence=5, width=7)  # prominence=1
    plt.plot(bin_centers[peaks], hist[peaks], "x")

    def func(x, a, b, c, d, e, f):
        return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d * np.exp(-(x - e) ** 2 / (2 * f ** 2))

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, bin_centers, hist, method='trf',
                           p0=[hist[peaks[0]], bin_centers[peaks[0]], 0.1, hist[peaks[1]], bin_centers[peaks[1]], 0.1],
                           bounds=(0, [np.inf, 1, 1, np.inf, 1, 1]))

    axis.plot(bin_centers, func(bin_centers, *popt))
    # plt.plot(bin_centers,func(bin_centers, 10000,0.18,0.1,5000,0.5,0.2))

# uniqueFileNames = list(set([re.search('hel[0-9]*',fileName).group() for fileName in fileNames]))
