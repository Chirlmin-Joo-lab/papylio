# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:24:46 2018

@author: ivoseverins
"""

# Use the following lines on Mac
from sys import platform
if platform == "darwin":
    from matplotlib import use
    use('WXAgg')

import os
import re # Regular expressions
import warnings
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pyplot as plt
from autothresholdAnalysis import stepfinder
from pathlib import Path # For efficient path manipulation
import pickle
import time
import sys


class Experiment:
    def __init__(self, mainPath):  # Exposure_time is needed until the automatic reading from a .log file is implemented
        self.name = os.path.basename(mainPath)
        self.mainPath = Path(mainPath).absolute()
        self.files = list()
        self.Ncolours = 2

        os.chdir(mainPath)  # this should be avoided. We can just add the path so the working directory is not changed
#        sys.path.append(self.mainPath)
        self.addAllFilesInMainPath()

    @property
    def molecules(self):
        return [molecule for file in self.files for molecule in file.molecules]

    @property
    def selectedFiles(self):
        return [file for file in self.files if file.isSelected]
    @property
    def selectedMoleculesInSelectedFiles(self):
        return [molecule for file in self.selectedFiles for molecule in file.selectedMolecules]
    @property
    def selectedMoleculesInAllFiles(self):
        return [molecule for file in self.files for molecule in file.selectedMolecules]

    def addAllFilesInMainPath(self):
                ## Find all unique files in all subfolders

        # Get all the files in all subfolders of the mainpath and remove their suffix (extensions)
        # Also make sure only relevant files are included (exclude folders, averaged tif files, data folders and .dat files)
        files = [p.relative_to(self.mainPath).with_suffix('') for p in self.mainPath.glob('**/*')
                    if  (p.is_file() &
                        ('_' not in p.name) &
                        #('\\.' not in str(p.with_suffix(''))) & # Can be removed, line below is better  - Ivo
                        ('.' not in [s[0] for s in p.parts]) &
                        (p.suffix not in ['.dat','.db', '.ini'])
                        )
                    ]
        for i, file in enumerate(files):
            if (file.name == 'Spooled files'):
               files[i] = files[i].parent

        uniqueFiles = np.unique(files)

        for file in uniqueFiles:
            self.addFile(file)


#        # Only works/tested for files in the main folder for now
#        for root, dirs, fileNames in os.walk('.', topdown=False):
#
#            for fileName in fileNames:
#                    print(root)
#                    print(fileName)
#
#                    self.addFile(root, fileName)
#
#            for name in dirs:
#                warnings.warn('Import of files in subfolders does not work properly yet')
#                print(os.path.join(root, name))


    def addFile(self, relativeFilePath):
        relativeFilePath = Path(relativeFilePath)


        # if there is no extension, add all files with the same name with all extensions
        # if there is an extension just add that file if the filename is the same




        #fileNameSearch = re.search('(hel[0-9]*)(.*\..*)', fileName)

        #if fileNameSearch:
        #    name, extension = fileNameSearch.groups()

        #    fullPath = os.path.join(relativePath, name)

            # Test whether file is already in experiment
        for file in self.files:
            if file.relativeFilePath == relativeFilePath:
                file.update()
                break
        else:
            self.files.append(File(relativeFilePath, self))

            # If not found: add file and extension, or if the file is already there then add the extention to it.
            # If the file and extension are already imported, display a warning message.
#            if not foundFile:
#                self.files.append(File(relativePath, name, self, self.exposure_time))
#                self.files[-1].addExtension(extension)
#            else:
#                if extension not in foundFile.extensions:
#                    foundFile.addExtension(extension)
#                else:
#                    warnings.warn(fullPath + extension + ' already imported')
#        else:
#            warnings.warn('FileName ' + fileName + ' not added, filename should contain hel...')


    def histogram(self, axis = None, fileSelection = False, moleculeSelection = False, makeFit = False):
        #files = [file for file in exp.files if file.isSelected]
        #files = self.files

        if (fileSelection & moleculeSelection):
            histogram([molecule for file in self.selectedFiles for molecule in file.selectedMolecules], axis, makeFit)
        elif (fileSelection & (not moleculeSelection)):
            histogram([molecule for file in self.selectedFiles for molecule in file.molecules], axis, makeFit)
        elif ((not fileSelection) & moleculeSelection):
            histogram([molecule for file in self.files for molecule in file.selectedMolecules], axis, makeFit)
        else:
            histogram([molecule for file in self.files for molecule in file.molecules], axis, makeFit)


    def select(self):
        for molecule in self.molecules:
            molecule.plot()
            input("Press enter to continue")

class File:
    def __init__(self, relativeFilePath, experiment):
        relativeFilePath = Path(relativeFilePath)
        self.experiment = experiment


        self.relativePath = relativeFilePath.parent
        self.name = relativeFilePath.name
        self.extensions = list()

        self.molecules = list()
        self.exposure_time = None #Here the exposure time is given but it should be found from the log file if possible

        self.isSelected = False

        self.findAndAddExtensions()

    @property
    def time(self):  # the time axis of the experiment
        return np.arange(0, self.Nframes)*self.exposure_time

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.name})')

    @property
    def relativeFilePath(self):
        return os.path.join(self.relativePath, self.name)

    @property
    def coordinates(self):
        return np.concatenate([[molecule.coordinates[0, :]
                                for molecule in self.molecules]])
    @property
    def selectedMolecules(self):
        return [molecule for molecule in self.molecules if molecule.isSelected]

    def findAndAddExtensions(self):
        foundFiles = [file.name for file in self.experiment.mainPath.joinpath(self.relativePath).glob(self.name+'*')]
        foundExtensions = [file[len(self.name):] for file in foundFiles]

        # For the special case of a sifx file, which is located inside a folder
        if '' in foundExtensions: foundExtensions[foundExtensions.index('')] = '.sifx'

        newExtensions = [extension for extension in foundExtensions if extension not in self.extensions]
        self.extensions = self.extensions + newExtensions
        for extension in newExtensions: self.importExtension(extension)

    def importExtension(self, extension):

        #print(f.relative_to(self.experiment.mainPath))

        #self.extensions.append(extension)
        #print(extension)
        importFunctions = {'.pks'        : self.importPksFile,
                           '.traces'     : self.importTracesFile,
                           '.sifx'       : self.importSifxFile,
                           '.log'        : self.importExposuretime,
                           '.sim'        : self.importSimFile
                           }

        importFunctions.get(extension, self.noneFunction)()

    def noneFunction(self):
        return

    def importSifxFile(self):
        print('sifx')

    def importExposuretime(self):
        self.exposure_time = np.genfromtxt(f'{self.relativeFilePath}.log', max_rows=1)[2]

    def importPksFile(self):
        # Background value stored in pks file is not imported yet
        Ncolours = self.experiment.Ncolours

   #     pks = np.genfromtxt(self.name + '.pks', delimiter='      ')  #MD190104 you get an error when using 6 spaces for tab
        try:
            pks = np.genfromtxt(str(self.relativeFilePath) + '.pks')  #MD190104 By default, any consecutive whitespaces act as delimiter.
            Ntraces = np.shape(pks)[0]
            if not self.molecules:
                for molecule in range(0, Ntraces, Ncolours):
                    self.addMolecule()

            for i, molecule in enumerate(self.molecules):
                molecule.coordinates = pks[(i*Ncolours):((i+1)*Ncolours), 1:3]

        except ValueError: #assuming the .pks file is from the Han lab programm
            Ntraces = int(np.genfromtxt(str(self.relativeFilePath) + '.pks', skip_header=5, max_rows=1))
            for molecule in enumerate(self.molecules):
                molecule.coordinates = [pks[0][0], pks[0][2], pks[0][5], pks[0][7]]

#        for trace in range(0, Ntraces, Ncolours):
#            coordinates = pks[trace:(trace+Ncolours),1:3]
#            self.addMolecule(coordinates)


    def importTracesFile(self):
        Ncolours = self.experiment.Ncolours
        file = open(str(self.relativeFilePath) + '.traces', 'r')
        self.Nframes = np.fromfile(file, dtype=np.int32, count=1).item()
        Ntraces = np.fromfile(file, dtype=np.int16, count=1).item()
        if not self.molecules:
            for molecule in range(0, Ntraces, Ncolours):
                self.addMolecule()

        rawData = np.fromfile(file, dtype=np.int16, count=self.Nframes * Ntraces)
        orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, self.Nframes), order = 'F')

        for i, molecule in enumerate(self.molecules):
            molecule.intensity = orderedData[:,i,:]
        file.close()

    def importSimFile(self):
        file = open(self.name + '.sim', 'rb')
        self.data = pickle.load(file)
        red, green  = self.data['red'], self.data['green']
        Ntraces = red.shape[0]
        self.Nframes = red.shape[1]

        if not self.molecules:
            for molecule in range(0, Ntraces):
                self.addMolecule()

        for i, molecule in enumerate(self.molecules):
            molecule.intensity = np.vstack((green[i], red[i]))
        file.close()

    def addMolecule(self):
        self.molecules.append(Molecule(self))
        self.molecules[-1].index = len(self.molecules)  # this is the molecule number

    def histogram(self):
        histogram(self.molecules)

    def importExcel(self, filename=None):
        if filename is None:
            filename = self.name+'_steps_data.xlsx'
        try:
            steps_data = pd.read_excel(filename, index_col=[0,1],
                                            dtype={'kon':np.str})       # reads from the 1st excel sheet of the file
        except FileNotFoundError:
            return
        molecules = steps_data.index.unique(0)
        indices = [int(m.split()[-1]) for m in molecules]
        for mol in self.molecules:
            if mol.index not in indices:
                continue
            mol.steps = steps_data.loc[f'mol {mol.index}']
            if 'kon' in mol.steps.columns:
                k = [int(i) for i in mol.steps.kon[0]]
                mol.kon_boolean = np.array(k).astype(bool).reshape((3,3))
        return steps_data

    def savetoExcel(self, filename=None, save=True):
        if filename is None:
            filename = self.name+'_steps_data.xlsx'
        # Concatenate all steps dataframes that are not None
        mol_data = [mol.steps for mol in self.molecules if mol.steps is not None]
        if not mol_data:
            print(f'no data to save for {self.name}')
            return
        keys = [f'mol {mol.index}' for mol in self.molecules if mol.steps is not None]
        steps_data = pd.concat(mol_data, keys=keys, sort=False)
        if save:
            print("data saved in: " + filename)
            writer = pd.ExcelWriter(filename)
            steps_data.to_excel(writer, self.name)
            writer.save()
        return steps_data

    def autoThreshold(self, trace_name, threshold=100, max_steps=20,
                      only_selected=False, kon_str='000000000'):
        nam = trace_name
        start = time.time()
        for mol in self.molecules:

            trace = mol.I(0)*int((nam == 'green')) + \
                    mol.I(1)*int((nam == 'red')) +\
                     mol.E()*int((nam == 'E'))  # Here no offset corrections are applied yet

            d = mol.find_steps(trace)
            frames = d['frames']
            times = frames*self.exposure_time
            times = np.sort(times)
            mol.steps = pd.DataFrame({'time': times, 'trace': nam,
                                  'state': 1, 'method': 'thres',
                                'thres': threshold, 'kon': kon_str})
        filename = self.name+'_steps_data.xlsx'
        print(f'Analysis time: {time.time() - start} sec')
        data = self.savetoExcel(filename)
        return data

    def select(self, axis=None):
        for molecule in self.molecules:
            molecule.plot(axis=axis)
            input("Press enter to continue")


class Molecule:

    def __init__(self, file):
        self.file = file
        self.index = None
        self.coordinates = None
        self.intensity = None

        self.isSelected = False

        self.steps = None  #Defined in other classes as: pd.DataFrame(columns=['frame', 'trace', 'state', 'method','thres'])
        self.kon_boolean = None  # 3x3 matrix that is indicates whether the kon will be calculated from the beginning, in-between molecules or for the end only

    def I(self, emission, Ioff=0):
        return self.intensity[emission, :] - Ioff

    def E(self, Imin=0, alpha=0, Iroff=0, Igoff=0):
        red = np.copy(self.I(1, Ioff=Iroff))
        green = self.I(0, Ioff=Igoff)
        np.putmask(red, red<Imin, 0)  # the mask makes all elements of acceptor that are below the Imin zero, for E caclulation
        E =  (red - alpha*green) / (green + red - alpha*green)
        E = np.nan_to_num(E)  # correct for divide with zero = None values
        return E

    def plot(self):
        plt.plot(self.intensity[0,:], 'g')
        plt.plot(self.intensity[1,:], 'r')
        plt.show()
#MD190104: why not add a subplot with FRET here as well, to match with display Matlab?
    @property
    def find_steps(self):
        return stepfinder


def histogram(input, nbins, axis, makeFit = False):
    if not input: return None
    if not axis: axis = plt.gca()
    #    if not isinstance(input,list): input = [input]
#
#    molecules = list()
#
#    for i in input:
#        if isinstance(i, Molecule):
#            molecules.append(i)
#        else:
#            molecules.append(i.molecules)
    molecules = input
    data = np.concatenate([molecule.E() for molecule in molecules])
    axis.hist(data, nbins, range = (0,1))

    if makeFit:
        fit_hist(data, axis)


def fit_hist(data, nbins, axis):

    hist, bin_edges = np.histogram(data, nbins, range = (0,1))
    bin_centers = (bin_edges[0:-1]+bin_edges[1:])/2

    #plt.plot(bin_centers,hist)

    from scipy.signal import butter
    from scipy.signal import filtfilt
    b, a = butter(2, 0.2, 'low')
    output_signal = filtfilt(b, a, hist)
    plt.plot(bin_centers, output_signal)

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(output_signal, prominence=5, width=7) #prominence=1
    plt.plot(bin_centers[peaks], hist[peaks], "x")

    def func(x, a, b, c, d, e, f):
        return a * np.exp(-(x-b)**2/(2*c**2)) + d * np.exp(-(x-e)**2/(2*f**2))

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, bin_centers, hist, method = 'trf',
                           p0 = [hist[peaks[0]],bin_centers[peaks[0]],0.1,hist[peaks[1]],bin_centers[peaks[1]],0.1], bounds = (0,[np.inf,1,1,np.inf,1,1]))

    axis.plot(bin_centers,func(bin_centers, *popt))
    #plt.plot(bin_centers,func(bin_centers, 10000,0.18,0.1,5000,0.5,0.2))














#uniqueFileNames = list(set([re.search('hel[0-9]*',fileName).group() for fileName in fileNames]))

"""

for root, dirs, files in os.walk('..', topdown=False):
	for name in files:
		print(os.path.join(root, name))
		print(root)
		print(name)

	for name in dirs:
		print(os.path.join(root, name))



for root, dirs, files in os.walk('..', topdown=False):
	print(files)

"""

