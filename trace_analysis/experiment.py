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

import os # Miscellaneous operating system interfaces - to be able to switch from Mac to Windows
from pathlib import Path # For efficient path manipulation
import yaml
import numpy as np
import matplotlib.pyplot as plt #Provides a MATLAB-like plotting framework

from trace_analysis.file import File
from trace_analysis.molecule import Molecule
from trace_analysis.plotting import histogram

import re # Regular expressions
import warnings

#import matplotlib.pyplot as plt #Provides a MATLAB-like plotting framework
#import itertools #Functions creating iterators for efficient looping
#np.seterr(divide='ignore', invalid='ignore')
#import pandas as pd
#from threshold_analysis_v2 import stepfinder
#import pickle


class Experiment:
    def __init__(self, mainPath, colours = ['g','r'], import_all = True):
        self.name = os.path.basename(mainPath)
        self.mainPath = Path(mainPath).absolute()
        self.files = list()
        self._colours = np.atleast_1d(np.array(colours))
        self._Ncolours = len(colours)
        self._pairs = [[c1, c2] for i1, c1 in enumerate(colours) for i2, c2 in enumerate(colours) if i2 > i1]
        self.import_all = import_all

        # Load custom config file or otherwise load the default config file
        if self.mainPath.joinpath('config.yml').is_file():
            self.import_config_file()
        else:
            with Path(__file__).with_name('default_configuration.yml').open('r') as yml_file:
                self.configuration = yaml.load(yml_file, Loader=yaml.SafeLoader)
            self.export_config_file()

        os.chdir(mainPath)

        self.addAllFilesInMainPath()

    @property
    def colours(self):
        return self._colours

    @colours.setter
    def colours(self, colours):
        self._colours = np.atleast_1d(np.array(colours))
        self._Ncolours = len(colours)
        self._pairs = [[c1, c2] for i1, c1 in enumerate(colours) for i2, c2 in enumerate(colours) if i2 > i1]

    @property
    def Ncolours(self):
        return self._Ncolours

    @property
    def pairs(self):
        return self._pairs

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

    def import_config_file(self):
        with self.mainPath.joinpath('config.yml').open('r') as yml_file:
            self.configuration = yaml.load(yml_file, Loader=yaml.SafeLoader)

    def export_config_file(self):
        with self.mainPath.joinpath('config.yml').open('w') as yml_file:
             yaml.dump(self.configuration, yml_file, sort_keys = False)

    def addAllFilesInMainPath(self):
        ## Find all unique files in all subfolders
        
        # Get all the files in all subfolders of the mainpath and remove their suffix (extensions)
        # Also make sure only relevant files are included (exclude folders, averaged tif files, data folders and .dat files)
        files = [p.relative_to(self.mainPath).with_suffix('') for p in self.mainPath.glob('**/*')
                    if  (p.is_file() & 
                        ('_' not in p.name) &
                        ('Analysis ' not in str(p)) &
                        #('\\.' not in str(p.with_suffix(''))) & # Can be removed, line below is better  - Ivo
                        ('.' not in [s[0] for s in p.parts]) &
                        (p.suffix not in ['.dat','.db', '.ini','.py','.yml'])
                        )
                    ]

        # Since sifx files made using spooling are all called 'Spooled files' the parent folder is used as file instead of the sifx file
        for i, file in enumerate(files):
            if (file.name == 'Spooled files'):
               files[i] = files[i].parent

        uniqueFiles = np.unique(files)

        for file in uniqueFiles:
            self.addFile(file)

    def addFile(self, relativeFilePath):
        relativeFilePath = Path(relativeFilePath)

        # if there is no extension, add all files with the same name with all extensions
        # if there is an extension just add that file if the filename is the same
        
        # Test whether file is already in experiment
        for file in self.files:
            if file.relativeFilePath == relativeFilePath:
                file.findAndAddExtensions()
                break
        else:
            new_file = File(relativeFilePath, self)
            if new_file.extensions:
                self.files.append(new_file)
            
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

    def boxplot_number_of_molecules(self):
        fig, ax = plt.subplots(figsize = (8,1.5))
        pointCount = [len(file.molecules) for file in self.files]
        plt.boxplot(pointCount, vert=False, labels = [''], widths = (0.8))
        plt.xlabel('Count')
        plt.title('Molecules per file')
        plt.tight_layout()

        fig.savefig(self.mainPath.joinpath('number_of_molecules.pdf'), bbox_inches='tight')
        fig.savefig(self.mainPath.joinpath('number_of_molecules.png'), bbox_inches='tight')

    def select(self):
        for molecule in self.molecules:
            molecule.plot()
            input("Press enter to continue")




