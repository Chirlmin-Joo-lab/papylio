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
    """ Main experiment class

    Class containing all the files in an experiment.
    In fact it can contain any collection of files.

    .. warning:: Only works with one or two colours.

    Attributes
    ----------
    name : str
        Experiment name based on the name of the main folder
    mainPath : str
        Absolute path to the main experiment folder
    files : list of :obj:`File`
        Files
    import_all : bool
        If true, then all files in the main folder are automatically imported. \n
        If false, then files are detected, but not imported.
    """

    def __init__(self, mainPath, colours=['g','r'], import_all=True):
        """Init method for the Experiment class

        Loads config file if it locates one in the main directory, otherwise it exports the default config file to the main directory.
        Scans all directory in the main directory recursively and imports all found files (if import_all is set to `True`).

        Parameters
        ----------
        mainPath : str
            Absolute path to the main experiment folder
        colours : list of str
            Colours used in the experiment
        import_all : bool
            If true, then all files in the main folder are automatically imported. \n
            If false, then files are detected, but not imported.
        """
        self.name = os.path.basename(mainPath)
        self.mainPath = Path(mainPath).absolute()
        self.files = list()
        self.import_all = import_all

        self._colours = np.atleast_1d(np.array(colours))
        self._Ncolours = len(colours)
        self._pairs = [[c1, c2] for i1, c1 in enumerate(colours) for i2, c2 in enumerate(colours) if i2 > i1]

        # Load custom config file or otherwise load the default config file
        if self.mainPath.joinpath('config.yml').is_file():
            self.import_config_file()
        else:
            with Path(__file__).with_name('default_configuration.yml').open('r') as yml_file:
                self.configuration = yaml.load(yml_file, Loader=yaml.SafeLoader)
            try:
                self.export_config_file()
            except:
                FileNotFoundError
                pass

        os.chdir(mainPath)

        self.addAllFilesInMainPath()

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')

    @property
    def colours(self):
        """list of str : Colours used in the experiment.

        Setting the colours will automatically update pairs.
        """
        return self._colours

    @colours.setter
    def colours(self, colours):
        self._colours = np.atleast_1d(np.array(colours))
        self._Ncolours = len(colours)
        self._pairs = [[c1, c2] for i1, c1 in enumerate(colours) for i2, c2 in enumerate(colours) if i2 > i1]

    @property
    def Ncolours(self):
        """int : Number of colours used in the experiment (read-only)"""
        return self._Ncolours

    @property
    def pairs(self):
        """list of list of str : List of colour pairs"""
        return self._pairs

    @property
    def molecules(self):
        """list of Molecule : List of all molecules in the experiment"""
        return [molecule for file in self.files for molecule in file.molecules]

    @property
    def selectedFiles(self):
        """list of File : List of selected files"""
        return [file for file in self.files if file.isSelected]

    @property
    def selectedMoleculesInSelectedFiles(self):
        """list of Molecule : List of selected molecules in selected files"""
        return [molecule for file in self.selectedFiles for molecule in file.selectedMolecules]

    @property
    def selectedMoleculesInAllFiles(self):
        """list of Molecule : List of selected molecules in all files"""
        return [molecule for file in self.files for molecule in file.selectedMolecules]

    def import_config_file(self):
        """Import configuration file from main folder into the configuration property."""
        with self.mainPath.joinpath('config.yml').open('r') as yml_file:
            self.configuration = yaml.load(yml_file, Loader=yaml.SafeLoader)

    def export_config_file(self):
        """Export from the configuration property into the configuration file in main folder"""
        with self.mainPath.joinpath('config.yml').open('w') as yml_file:
             yaml.dump(self.configuration, yml_file, sort_keys = False)

    def addAllFilesInMainPath(self):
        """Find unique files in all subfolders and add them to the experiment

        Get all files in all subfolders of the mainpath and remove their suffix (extensions), and add them to the experiment.

        Note
        ----
        Non-relevant files are excluded e.g. files with underscores or 'Analysis' in their name, or files with dat, db,
        ini, py and yml extensions.

        Note
        ----
        Since sifx files made using spooling are all called 'Spooled files' the parent folder is used as file instead of the sifx file

        """

        files = [p.relative_to(self.mainPath).with_suffix('') for p in self.mainPath.glob('**/*')
                    if  (p.is_file() &
                        ('_' not in p.name) &
                        ('Analysis ' not in str(p)) &
                        #('\\.' not in str(p.with_suffix(''))) & # Can be removed, line below is better  - Ivo
                        ('.' not in [s[0] for s in p.parts]) &
                        (p.suffix not in ['.dat','.db', '.ini','.py','.yml'])
                        )
                    ]

        for i, file in enumerate(files):
            if (file.name == 'Spooled files'):
               files[i] = files[i].parent

        uniqueFiles = np.unique(files)

        for file in uniqueFiles:
            self.addFile(file)

    def addFile(self, relativeFilePath):
        """Add a file to the experiment

        Add the file to the experiment only if the file object has found and imported relevant extensions .
        If the file is already present in experiment, then try to find and import new extensions.

        Parameters
        ----------
        relativeFilePath : pathlib.Path or str
            Path with respect to the main experiment path

        """
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

    def histogram(self, axis = None, bins = 100, parameter = 'E', molecule_averaging = False,
                  fileSelection = False, moleculeSelection = False, makeFit = False, export=False, **kwargs):
        """FRET histogram of all molecules in the experiment or a specified selection

        Parameters
        ----------
        axis : matplotlib.axis
            Axis to use for histogram plotting
        bins : int
            Number of bins
        parameter : str
            Parameter to be used for histogram I or E
        molecule_averaging : bool
            If True an time average of the trace is used
        fileSelection : bool
            If True the histogram is made only using selected files.
        moleculeSelection : bool
            If True the histogram is made only using selected molecules.
        makeFit : bool
            If True perform Gaussian fitting.
        export : bool
            If True the graph is exported.
        **kwargs
            Arbitrary keyword arguments.

        """
        #files = [file for file in exp.files if file.isSelected]
        #files = self.files

        if (fileSelection & moleculeSelection):
            molecules = [molecule for file in self.selectedFiles for molecule in file.selectedMolecules]
        elif (fileSelection & (not moleculeSelection)):
            molecules = [molecule for file in self.selectedFiles for molecule in file.molecules]
        elif ((not fileSelection) & moleculeSelection):
            molecules = [molecule for file in self.files for molecule in file.selectedMolecules]
        else:
            molecules = [molecule for file in self.files for molecule in file.molecules]

        histogram(molecules, axis=axis, bins=bins, parameter=parameter, molecule_averaging=molecule_averaging, makeFit=makeFit, collection_name=self, **kwargs)
        if export: plt.savefig(self.mainPath.joinpath(f'{self.name}_{parameter}_histogram').with_suffix('.png'))

    def boxplot_number_of_molecules(self):
        """Boxplot of the number of molecules in each file"""
        fig, ax = plt.subplots(figsize = (8,1.5))
        pointCount = [len(file.molecules) for file in self.files]
        plt.boxplot(pointCount, vert=False, labels = [''], widths = (0.8))
        plt.xlabel('Count')
        plt.title('Molecules per file')
        plt.tight_layout()

        fig.savefig(self.mainPath.joinpath('number_of_molecules.pdf'), bbox_inches='tight')
        fig.savefig(self.mainPath.joinpath('number_of_molecules.png'), bbox_inches='tight')

    def select(self):
        """Simple method to look through all molecules in the experiment

        Plots a molecule. If enter is pressed the next molecule is shown.

        """
        for molecule in self.molecules:
            molecule.plot()
            input("Press enter to continue")




