# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:24:46 2018

@author: ivoseverins
"""
# Use the following lines on Mac
# from sys import platform
# if platform == "darwin":
#     from matplotlib import use
#     use('WXAgg')
import PySide2
import os  # Miscellaneous operating system interfaces - to be able to switch from Mac to Windows
from pathlib import Path  # For efficient path manipulation

import tqdm
import yaml
import numpy as np
import pandas as pd
# import wx

###################################################
## To enable interactive plotting with PySide2 in PyCharm 2022.3
import PySide2
import sys
sys.modules['PyQt5'] = sys.modules['PySide2']
from matplotlib import use
use('Qt5Agg')
###################################################

import matplotlib.pyplot as plt  # Provides a MATLAB-like plotting framework
import xarray as xr
from collections import UserDict
import re
import tifffile

from papylio.file import File
# from papylio.molecule import Molecules
from papylio.file_collection import FileCollection
from papylio.plotting import histogram
from papylio.movie.movie import Movie
# from papylio.plugin_manager import PluginManager
# from papylio.plugin_manager import PluginMetaClass
from papylio.plugin_manager import plugins

import re  # Regular expressions
import warnings
from nd2reader import ND2Reader


# import matplotlib.pyplot as plt #Provides a MATLAB-like plotting framework
# import itertools #Functions creating iterators for efficient looping
# np.seterr(divide='ignore', invalid='ignore')
# import pandas as pd
# from threshold_analysis_v2 import stepfinder
# import pickle

class Configuration(UserDict):
    # Ruamel yaml parser may be better for preserving comments
    # https://sourceforge.net/projects/ruamel-yaml/
    def __init__(self, filepath):
        self.reload_block = 0
        self.filepath = Path(filepath)
        self.previous_file_modification_time = 0

        # Load custom config file or otherwise load the default config file
        if self.filepath.is_file():
            self.load()
        else:
            filepath = Path(__file__).with_name('default_configuration.yml')
            with filepath.open('r') as yml_file:
                self._data = yaml.load(yml_file, Loader=yaml.SafeLoader)
            # self.load(Path(__file__).with_name('default_configuration.yml'))
            self.save()

    @property
    def data(self):
        self.load()
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def file_modification_time(self):
        return self.filepath.stat().st_mtime

    def __enter__(self):
        self.load()
        self.reload_block += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reload_block -= 1

    @property
    def reload(self):
        return self.reload_block == 0

    def __getitem__(self, item):
        return self.data[item]

    def load(self, filepath=None):
        if self.reload:
            file_modification_time = self.file_modification_time
            if file_modification_time != self.previous_file_modification_time:
                if filepath is None:
                    filepath = self.filepath
                with filepath.open('r') as yml_file:
                    self._data = yaml.load(yml_file, Loader=yaml.SafeLoader)
                self.previous_file_modification_time = file_modification_time

    def save(self):
        with self.filepath.open('w') as yml_file:
            yaml.dump(self._data, yml_file, sort_keys=False)

# from PyQt5.QtWidgets import QFileDialog
def get_QApplication():
    from PySide2 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then a QApplication is created
        app = QtWidgets.QApplication([])
    return app

def get_path(main_window):
    # if not 'app' in globals().keys():
    #     global app
    #     app = wx.App(None)
    # dlg = wx.DirDialog(None, message="Choose a folder", defaultPath="")
    # if dlg.ShowModal() == wx.ID_OK:
    #     path = dlg.GetPath()
    # else:
    #     path = None
    # dlg.Destroy()

    app = get_QApplication()
    from PySide2.QtWidgets import QFileDialog, QMainWindow
    if main_window is None:
        main_window = QMainWindow()
    path = QFileDialog.getExistingDirectory(main_window, 'Choose directory')
    return path

@plugins
class Experiment:
    """ Main experiment class

    Class containing all the files in an experiment.
    In fact it can contain any collection of files.

    .. warning:: Only works with one or two channels.

    Attributes
    ----------
    name : str
        Experiment name based on the name of the main folder
    main_path : str
        Absolute path to the main experiment folder
    files : list of :obj:`File`
        Files
    import_all : bool
        If true, then all files in the main folder are automatically imported. \n
        If false, then files are detected, but not imported.
    """
    # TODO: Add presets for specific microscopes
    def __init__(self, main_path=None, channels=['g', 'r'], import_all=True, main_window=None, perform_logging=True):
        """Init method for the Experiment class

        Loads config file if it locates one in the main directory, otherwise it exports the default config file to the main directory.
        Scans all directory in the main directory recursively and imports all found files (if import_all is set to `True`).

        Parameters
        ----------
        main_path : str
            Absolute path to the main experiment folder
        channels : list of str
            Channels used in the experiment
        import_all : bool
            If true, then all files in the main folder are automatically imported. \n
            If false, then files are detected, but not imported.
        """
        if main_path is None:
            main_path = get_path(main_window)
            if main_path is None:
                raise ValueError('No folder selected')

        self.name = os.path.basename(main_path)
        self.main_path = Path(main_path).absolute()
        self.files = FileCollection()
        self.import_all = import_all
        self.perform_logging = perform_logging

        self._channels = np.atleast_1d(np.array(channels))
        self._number_of_channels = len(channels)
        self._pairs = [[c1, c2] for i1, c1 in enumerate(channels) for i2, c2 in enumerate(channels) if i2 > i1]

        # Load custom config file or otherwise load the default config file
        self.configuration = Configuration(self.main_path.joinpath('config.yml'))

        os.chdir(main_path)

        # file_paths = self.find_file_paths()
        # self.add_files(file_paths, test_duplicates=False)

        with self.configuration:
            self.add_files(self.main_path, test_duplicates=False)

        self.common_image_corrections = xr.Dataset()
        self.load_darkfield_correction()
        self.load_flatfield_correction()

        # Find mapping file
        for file in self.files:
            if file.mapping is not None:
                file.use_mapping_for_all_files()
                break

        print('\nInitialize experiment: \n' + str(self.main_path))

    def __getstate__(self):
        d = self.__dict__.copy()
        # d.pop('files')
        d['files'] = []
        excluded_keys = ['files', 'sequencing_data', '_tile_mappings'] #TODO: Move sequencing related terms to sequencing.py
        d = {key: value for key, value in self.__dict__.items() if key not in excluded_keys}
        d['_do_not_update'] = None # This is for parallelization in Collection
        return d

    def __setstate__(self, dict):
        self.__dict__.update(dict)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')

    @property
    def channels(self):
        """list of str : Channels used in the experiment.

        Setting the channels will automatically update pairs.
        """
        return self._channels

    @channels.setter
    def channels(self, channels):
        self._channels = np.atleast_1d(np.array(channels))
        self._number_of_channels = len(channels)
        self._pairs = [[c1, c2] for i1, c1 in enumerate(channels) for i2, c2 in enumerate(channels) if i2 > i1]

    @property
    def number_of_channels(self):
        """int : Number of channels used in the experiment (read-only)"""
        return self._number_of_channels

    @property
    def pairs(self):
        """list of list of str : List of channel pairs"""
        return self._pairs

    # @property
    # def molecules(self):
    #     """list of Molecule : List of all molecules in the experiment"""
    #     return Molecules.sum([file.molecules for file in self.files])

    @property
    def selectedFiles(self):
        """list of File : List of selected files"""
        return self.files[self.files.isSelected]

    # @property
    # def selectedMoleculesInSelectedFiles(self):
    #     """list of Molecule : List of selected molecules in selected files"""
    #     return [molecule for file in self.selectedFiles for molecule in file.selectedMolecules]

    # @property
    # def selectedMoleculesInAllFiles(self):
    #     """list of Molecule : List of selected molecules in all files"""
    #     return [molecule for file in self.files for molecule in file.selectedMolecules]

    @property
    def mapping_file(self):
        for file in self.files:
            if file.is_mapping_file:
                return file

    @property
    def analysis_path(self):
        analysis_path = self.main_path.joinpath('Analysis')
        analysis_path.mkdir(parents=True, exist_ok=True)
        return analysis_path

    @property
    def file_paths(self):
        return [file.relativeFilePath for file in self.files]

    @property
    def nc_file_paths(self):
        return [file.relativeFilePath.with_suffix('.nc') for file in self.files if '.nc' in file.extensions]

    def find_file_paths_and_extensions(self, paths):
        """Find unique files in all subfolders and add them to the experiment

        Get all files in all subfolders of the main_path and remove their suffix (extensions), and add them to the experiment.

        Note
        ----
        Non-relevant files are excluded e.g. files with underscores or 'Analysis' in their name, or files with dat, db,
        ini, py and yml extensions.

        Note
        ----
        Since sifx files made using spooling are all called 'Spooled files' the parent folder is used as file instead of the sifx file

        """

        if isinstance(paths, str) or isinstance(paths, Path):
            #paths = paths.glob('**/*')
                #'**/?*.*')  # At least one character in front of the extension to prevent using hidden folders

            # The following approach is faster than checking each file separately using is_file() for network drives. (not tested for regular drives)
            files_and_folders = set(paths.glob('**/*'))
            folders = set(paths.glob('**'))
            paths = files_and_folders - folders

        file_paths_and_extensions = \
            [[p.relative_to(self.main_path).with_suffix(''), p.suffix]
             for p in paths
             if (
                     # Use only files
                     #p.is_file() &
                     # Exclude stings in filename
                     all(name not in p.with_suffix('').name for name in
                         self.configuration['files']['excluded_names']) &
                     # Exclude strings in path
                     all(path not in str(p.relative_to(self.main_path).parent) for path in
                         self.configuration['files']['excluded_paths']) &
                     # Exclude hidden folders
                     ('.' not in [s[0] for s in p.parts]) &
                     # Exclude file extensions
                     (p.suffix[1:] not in self.configuration['files']['excluded_extensions'])
             )
             ]

        # TODO: Test spooled file and nd2 file import
        new_file_paths_and_extensions = []
        for i, (file_path, extensions) in enumerate(file_paths_and_extensions):
            if (file_path.name == 'Spooled files'):
                new_file_paths_and_extensions.append([file_path.parent, extensions])
                # file_paths_and_extensions[i, 0] = file_paths_and_extensions[i, 0].parent
            elif '.nd2' in extensions and not 'fov' in str(file_path):
                from papylio.movie.movie import Movie
                nd2_movie = Movie(file_path.with_suffix('.nd2'))
                if nd2_movie.number_of_fov > 1:  # if the file is nd2 with multiple field of views
                    for fov_id in range(nd2_movie.number_of_fov):
                        new_path = Path(str(file_path) + f'_fov{fov_id:03d}')
                        new_file_paths_and_extensions.append([new_path, extensions])
                        # fov_info['fov_chosen'] = fov_id
                        # new_file = File(new_path, self, fov_info=fov_info.copy())
                        # if new_file.extensions:
                        #     self.files.append(new_file)
                else:
                    new_file_paths_and_extensions.append([file_path, extensions])
            else:
                new_file_paths_and_extensions.append([file_path, extensions])
        file_paths_and_extensions = new_file_paths_and_extensions

        file_paths_and_extensions = np.array(file_paths_and_extensions)

        file_paths_and_extensions = file_paths_and_extensions[file_paths_and_extensions[:, 0].argsort()]
        unique_file_paths, indices = np.unique(file_paths_and_extensions[:, 0], return_index=True)
        extensions_per_filepath = np.split(file_paths_and_extensions[:, 1], indices[1:])

        return unique_file_paths, extensions_per_filepath

    def add_files(self, paths, test_duplicates=True):
        """Find unique files in all subfolders and add them to the experiment

        Get all files in all subfolders of the main_path and remove their suffix (extensions), and add them to the experiment.

        Note
        ----
        Non-relevant files are excluded e.g. files with underscores or 'Analysis' in their name, or files with dat, db,
        ini, py and yml extensions.

        Note
        ----
        Since sifx files made using spooling are all called 'Spooled files' the parent folder is used as file instead of the sifx file

        """
        file_paths_and_extensions = self.find_file_paths_and_extensions(paths)

        for file_path, extensions in tqdm.tqdm(zip(*file_paths_and_extensions), 'Import files',
                                               total=(len(file_paths_and_extensions[0]))):
            if not test_duplicates or (file_path.absolute().relative_to(self.main_path) not in self.file_paths):
                self.files.append(File(file_path, extensions, self, perform_logging=self.perform_logging))
            else:
                i = self.file_paths.find(file_path.absolute().relative_to(self.main_path))
                self.files[i].add_extensions(extensions)

        # nd2_file = list(self.main_path.glob(str(relativeFilePath) + '.nd2'))
        # fov_info = {'number_of_fov': 1}  # fov=Field of View
        # if nd2_file:  # check if the nd2 file has multiple fov data,
        #     fov_info = self.get_fov_from_nd2(nd2_file[0])
        #
        # if fov_info['number_of_fov'] > 1:  # if the file is nd2 with multiple field of views
        #     for fov_id in range(fov_info['number_of_fov']):
        #         new_path = Path(str(relativeFilePath) + f'_fov{fov_id:03d}')
        #         fov_info['fov_chosen'] = fov_id
        #         new_file = File(new_path, self, fov_info=fov_info.copy())
        #         if new_file.extensions:
        #             self.files.append(new_file)



    # def add_files(self, file_paths, test_duplicates=True):
    #     for file_path in file_paths:
    #         self.add_file(file_path, test_duplicates)
    #
    #     for file in self.files:
    #         if file.mapping is not None:
    #             file.use_mapping_for_all_files()
    #             break

    # def add_file(self, file_path, test_duplicates=True):
    #     """Add a file to the experiment
    #
    #     Add the file to the experiment only if the file object has found and imported relevant extensions .
    #     If the file is already present in experiment, then try to find and import new extensions.
    #
    #     Parameters
    #     ----------
    #     relativeFilePath : pathlib.Path or str
    #         Path with respect to the main experiment path
    #
    #     """
    #     # Perhaps move this conversion to relative file path to File
    #     relative_file_path = Path(file_path).absolute().relative_to(self.main_path)
    #
    #     # if there is no extension, add all files with the same name with all extensions
    #     # if there is an extension just add that file if the filename is the same
    #
    #     # Test whether file is already in experiment
    #     # for file in self.files:
    #     #     if file.relativeFilePath == relativeFilePath:
    #     #         file.findAndAddExtensions()
    #     #         break
    #     # else:
    #     #     new_file = File(relativeFilePath, self)
    #     #     if new_file.extensions:
    #     #         self.files.append(new_file)
    #
    #
    #     if not test_duplicates or (relative_file_path not in self.file_paths):
    #         self.files.append(File(relative_file_path, self))
    #     else:
    #         i = self.file_paths.find(file_path.absolute().relative_to(self.main_path))
    #         self.files[i].findAndAddExtensions()

    # def load_flatfield_correction(self):
    #     file_paths = list(self.main_path.glob('flatfield*'))
    #     if file_paths:
    #         movie = self.files[0].movie
    #         flatfield_correction = xr.DataArray(np.ones((movie.number_of_illuminations,
    #                                             movie.height, movie.width)), # perhaps make the movie width and height equal to the channel width and height
    #                               dims=('illumination', 'y', 'x'),
    #                               coords={'illumination': movie.illumination_indices})
    #         for file_path in file_paths:
    #             flatfield = tifffile.imread(file_path)
    #             _, _, illumination_indices, _ = movie.image_type_from_filename(file_path.name)
    #             flatfield_correction[dict(illumination=illumination_indices)] = flatfield
    #
    #         self.files.movie.flatfield_correction = flatfield_correction
    #     else:
    #         self.files.movie.flatfield_correction = None

    def determine_flatfield_and_darkfield_corrections(self, files, method='BaSiC', illumination_index=0, frame_index=0,
                                                      estimate_darkfield=True, **kwargs):
        from papylio.movie.basic_shading_correction import spatial_shading_correction

        darkfield, flatfield = spatial_shading_correction(files.movie, method=method,
                                                          illumination_index=illumination_index,
                                                          frame_index=frame_index,
                                                          estimate_darkfield=estimate_darkfield, **kwargs)
        if estimate_darkfield:
            tifffile.imwrite(self.main_path / f'darkfield_i{illumination_index}.tif', darkfield.astype('float32'), imagej=True)
            self.load_darkfield_correction()
        tifffile.imwrite(self.main_path / f'flatfield_i{illumination_index}.tif', flatfield.astype('float32'), imagej=True)
        self.load_flatfield_correction()

    def load_flatfield_correction(self):
        file_paths = list(self.main_path.glob('flatfield*'))
        if file_paths:
            movie = self.files[0].movie
            flatfield_correction = xr.DataArray(np.ones((movie.number_of_illuminations, movie.number_of_channels,
                                                         movie.channels[0].height, movie.channels[0].width)),
                                                # perhaps make the movie width and height equal to the channel width and height
                                                dims=('illumination', 'channel', 'y', 'x'),
                                                coords={'illumination': movie.illumination_indices,
                                                        'channel': movie.channel_indices})
            for file_path in file_paths:
                flatfield = tifffile.imread(file_path)
                image_info = Movie.image_info_from_filename(file_path.name)
                illumination_index = image_info['illumination_index']
                channel_indices = movie.channel_indices
                flatfield_correction[dict(illumination=illumination_index, channel=channel_indices)] = \
                    movie.separate_channels(flatfield)

            self.common_image_corrections['flatfield_correction'] = flatfield_correction
        else:
            self.common_image_corrections = self.common_image_corrections.drop_vars('flatfield_correction', errors='ignore')

        self.add_common_image_corrections_to_movies()

    def load_darkfield_correction(self):
        file_paths = list(self.main_path.glob('darkfield*'))
        if file_paths:
            movie = self.files[0].movie
            darkfield_correction = xr.DataArray(np.ones((movie.number_of_illuminations, movie.number_of_channels,
                                                         movie.channels[0].height, movie.channels[0].width)),
                                                # perhaps make the movie width and height equal to the channel width and height
                                                dims=('illumination', 'channel', 'y', 'x'),
                                                coords={'illumination': movie.illumination_indices,
                                                        'channel': movie.channel_indices})
            # for file_path in file_paths:
            darkfield = tifffile.imread(file_paths[0])
            for illumination_index in darkfield_correction.illumination.values:
                # image_info = Movie.image_info_from_filename(file_path.name)
                # illumination_index = image_info['illumination_index']
                channel_indices = movie.channel_indices
                darkfield_correction[dict(illumination=illumination_index, channel=channel_indices)] = \
                    movie.separate_channels(darkfield)

            self.common_image_corrections['darkfield_correction'] = darkfield_correction
        else:
            self.common_image_corrections = self.common_image_corrections.drop_vars('darkfield_correction', errors='ignore')

        self.add_common_image_corrections_to_movies()

    def add_common_image_corrections_to_movies(self):
        self.files.movie._common_corrections = self.common_image_corrections

    # def show_flatfield_and_darkfield_corrections(self, name='', save=True):
    #     pass

    # def load_darkfield_correction(self):
    #     file_paths = list(self.main_path.glob('darkfield*'))
    #     if file_paths:
    #         movie = self.files[0].movie
    #         darkfield_correction = xr.DataArray(np.zeros((movie.number_of_illuminations,
    #                                             movie.height, movie.width)), # perhaps make the movie width and height equal to the channel width and height
    #                                    dims=('illumination', 'y', 'x'),
    #                                    coords={'illumination': movie.illumination_indices})
    #         for file_path in file_paths:
    #             darkfield = tifffile.imread(file_path)
    #             _, _, illumination_indices, _ = movie.image_type_from_filename(file_path.name)
    #             darkfield_correction[dict(illumination=illumination_indices)] = darkfield
    #
    #         self.files.movie.darkfield_correction = darkfield_correction
    #     else:
    #         self.files.movie.darkfield_correction = None

    def histogram(self, axis=None, bins=100, parameter='E', molecule_averaging=False,
                  fileSelection=False, moleculeSelection=False, makeFit=False, export=False, **kwargs):
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
        # files = [file for file in exp.files if file.isSelected]
        # files = self.files

        if (fileSelection & moleculeSelection):
            molecules = [molecule for file in self.selectedFiles for molecule in file.selectedMolecules]
        elif (fileSelection & (not moleculeSelection)):
            molecules = [molecule for file in self.selectedFiles for molecule in file.molecules]
        elif ((not fileSelection) & moleculeSelection):
            molecules = [molecule for file in self.files for molecule in file.selectedMolecules]
        else:
            molecules = [molecule for file in self.files for molecule in file.molecules]

        histogram(molecules, axis=axis, bins=bins, parameter=parameter, molecule_averaging=molecule_averaging,
                  makeFit=makeFit, collection_name=self, **kwargs)
        if export: plt.savefig(self.main_path.joinpath(f'{self.name}_{parameter}_histogram').with_suffix('.png'))

    def boxplot_number_of_molecules(self):
        """Boxplot of the number of molecules in each file"""
        fig, ax = plt.subplots(figsize=(8, 1.5))
        pointCount = [len(file.molecules) for file in self.files]
        plt.boxplot(pointCount, vert=False, labels=[''], widths=(0.8))
        plt.xlabel('Count')
        plt.title('Molecules per file')
        plt.tight_layout()

        fig.savefig(self.main_path.joinpath('number_of_molecules.pdf'), bbox_inches='tight')
        fig.savefig(self.main_path.joinpath('number_of_molecules.png'), bbox_inches='tight')

    def select(self):
        """Simple method to look through all molecules in the experiment

        Plots a molecule. If enter is pressed the next molecule is shown.

        """
        for molecule in self.molecules:
            molecule.plot()
            input("Press enter to continue")

    def print_files(self):
        self.files.print()

    def plot_trace(self, files=None, query={}, **kwargs):
        # from papylio.trace_plot import TraceAnalysisFrame

        if files is None:
            files = self.files

        file_paths = [file.relativeFilePath.with_suffix('.nc') for file in files if '.nc' in file.extensions]

        with xr.open_mfdataset(file_paths, concat_dim='molecule', combine='nested') as ds:
            ds_sel = ds.query(query)  # HJ1_WT, HJ7_G116T
            # if not 'app' in globals().keys():
            #     global app
            #     app = wx.App(None)
            # app = wit.InspectableApp()
            # frame = TraceAnalysisFrame(None, ds_sel, "Sample editor")
            # frame.molecules = exp.files[1].molecules
            # print('test')
            # import wx.lib.inspection
            # wx.lib.inspection.InspectionTool().Show()
            # app.MainLoop()
            from papylio.trace_plot import TracePlotWindow
            TracePlotWindow(dataset=ds_sel, save_path=None, **kwargs)

    def export_number_of_molecules_per_file(self):
        df = pd.DataFrame(columns=['Number of molecules'])
        for i, file in enumerate(self.files):
            n = str(file.relativeFilePath)
            try:
                nms = file.number_of_molecules
            except FileNotFoundError:
                nms = -1
            df.loc[n] = nms
        df.to_excel(self.main_path.joinpath('number_of_molecules'))
