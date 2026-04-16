import PySide6
import os
from pathlib import Path

import tqdm
import yaml
import numpy as np
import pandas as pd

###################################################
## To enable interactive plotting with PySide2 in PyCharm 2022.3
import PySide6
import sys
sys.modules['PyQt6'] = sys.modules['PySide6']
from matplotlib import use
use('qtagg')
###################################################

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import xarray as xr
from collections import UserDict
import re
import tifffile
import matchpoint as mp

from papylio.file import File
from papylio.file_collection import FileCollection
from papylio.plotting import histogram
from papylio.movie.movie import Movie
# from papylio.plugin_manager import PluginManager
# from papylio.plugin_manager import PluginMetaClass
from papylio.plugin_manager import plugins

def get_QApplication():
    """Get or create a PySide2 QApplication instance.

    Returns the existing QApplication instance if one exists, otherwise
    creates and returns a new one.

    Returns
    -------
    PySide2.QtWidgets.QApplication
        The QApplication instance
    """
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then a QApplication is created
        app = QtWidgets.QApplication([])
    return app

def get_path(main_window):
    """Open a file dialog to select a directory.

    Opens a platform-native directory selection dialog for the user to choose
    a folder. Returns the selected path or None if cancelled.

    Parameters
    ----------
    main_window : PySide6.QtWidgets.QMainWindow, optional
        Parent window for the dialog. If None, a new QMainWindow is created

    Returns
    -------
    str
        Absolute path to selected directory, or empty string if cancelled
    """
    app = get_QApplication()
    from PySide6.QtWidgets import QFileDialog, QMainWindow
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
    def __init__(self, main_path=None, channels=['g', 'r'], import_all=True, main_window=None, perform_logging=True, use_colorblind_friendly_colors=True):
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

        ### CONFIGURATION ###
        self.excluded_extensions = ['pdf', 'dat', 'db', 'py', 'yml', 'png', 'pdf', 'xlsx', 'md', 'txt']
        self.included_extensions = ['.nc']
        self.filename_suffixes = ['_ave', '_max', '_corrections', '_dwells', '_dwell_analysis']
        self.filename_suffixes += ['_sequencing_data', '_sequencing_match'] # TODO: Move sequencing related terms to sequencing.py
        self.excluded_names = ['darkfield', 'flatfield']
        self.excluded_paths = ['Analysis', 'Sequencing data', 'Results']
        #####################


        set_default_matplotlib_colors(use_colorblind_friendly_colors)

        ### MICROSCOPE ###
        self._channels = np.atleast_1d(np.array(channels))
        self.microscope_name = 'TIR-T'
        ##################

        self.movie_extensions = list(Movie.type_dict().keys())
        self.included_extensions += self.movie_extensions


        self._number_of_channels = len(channels)
        self._pairs = [[c1, c2] for i1, c1 in enumerate(channels) for i2, c2 in enumerate(channels) if i2 > i1]

        os.chdir(main_path)

        # file_paths = self.find_file_paths()
        # self.add_files(file_paths, test_duplicates=False)

        self.add_files(self.main_path, test_duplicates=False)

        self.common_image_corrections = xr.Dataset()
        self.load_darkfield_correction()
        self.load_flatfield_correction()

        # Find mapping file
        # for file in self.files:
        #     if file.mapping is not None:
        #         #TODO: Check whether we want to log this or not. If so, then the mapping file name should be logged.
        #         file.use_mapping_for_all_files(perform_logging=False)
        #         break
        self.load_mappings()

        print('\nInitialize experiment: \n' + str(self.main_path))

    def __getstate__(self):
        """Prepare object for pickling by excluding non-serializable attributes.

        Returns a dictionary of the object's state excluding files and other
        attributes that cannot be properly serialized. This is used for
        parallelization and object persistence.

        Returns
        -------
        dict
            Dictionary with object state minus excluded keys
        """
        d = self.__dict__.copy()
        # d.pop('files')
        d['files'] = []
        excluded_keys = ['files', 'sequencing_data', '_tile_mappings'] #TODO: Move sequencing related terms to sequencing.py
        d = {key: value for key, value in self.__dict__.items() if key not in excluded_keys}
        d['_do_not_update'] = None # This is for parallelization in Collection
        return d

    def __setstate__(self, dict):
        """Restore object state from pickled state dictionary.

        Parameters
        ----------
        dict : dict
            State dictionary from __getstate__
        """
        self.__dict__.update(dict)

    def __repr__(self):
        """Return string representation of the Experiment object.

        Returns
        -------
        str
            String representation in format 'Experiment(name)'
        """
        return (f'{self.__class__.__name__}({self.name})')

    @property
    def channels(self):
        """list of str : Channels used in the experiment.

        Setting the channels will automatically update pairs.
        """
        return self._channels

    @channels.setter
    def channels(self, channels):
        """Set channels used in the experiment.

        Automatically updates the number of channels and channel pairs when
        channels are modified.

        Parameters
        ----------
        channels : list or array-like
            Channel identifiers (typically strings like 'g', 'r')
        """
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
    def selected_files(self):
        """list of File : List of selected files"""
        return self.files[self.files.is_selected]

    def load_mappings(self):
        mappings = []
        for filepath in self.main_path.glob('channel_mapping*.nc'):
            mappings.append(mp.MatchPoint.load(filepath))
        if len(mappings) == (self.number_of_channels-1):
            for file in self.files:
                file.mappings = mappings

    @property
    def analysis_path(self):
        """pathlib.Path : Path to the Analysis folder.

        Creates the Analysis folder in the main experiment directory if it
        does not exist, then returns the path.
        """
        analysis_path = self.main_path.joinpath('Analysis')
        analysis_path.mkdir(parents=True, exist_ok=True)
        return analysis_path

    @property
    def file_paths(self):
        """list of pathlib.Path : List of relative file paths for all files in experiment"""
        return [file.relative_filepath for file in self.files]

    @property
    def nc_file_paths(self):
        """list of pathlib.Path : List of relative NetCDF file paths for all files in experiment"""
        return [file.relative_filepath.with_suffix('.nc') for file in self.files if '.nc' in file.extensions]

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
        main_path = Path(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20230912 - Objective-type TIRF (BN)')
        remove_pattern_filename = re.compile("|".join(map(re.escape, self.filename_suffixes)))
        #
        # filepaths_and_extensions = set()
        #
        # stack = [main_path]
        # while stack:
        #     dir_path = stack.pop()
        #     dir_path_relative = Path(dir_path).relative_to(main_path)
        #     with os.scandir(dir_path) as dir_items:
        #         for item in dir_items:
        #             if item.is_dir():
        #                 stack.append(item.path)
        #             elif item.is_file():
        #                 filepath = Path(item.path)
        #                 if filepath.suffix in self.included_extensions:
        #                     # filename = remove_pattern_filename.sub('', filepath.name)
        #                     filename = filepath.name
        #                     for pattern in self.filename_suffixes:
        #                         filename = filename.replace(pattern, '')
        #                     filepaths_and_extensions.add(dir_path_relative / filename)

        #TODO: Things like _dwells.nc are now added as extensions, not sure whether we need extensions at all actually.

        from collections import defaultdict
        filepaths_and_extensions = defaultdict(set)

        stack = [main_path]
        while stack:
            dir_path = stack.pop()
            dir_path_relative = Path(dir_path).relative_to(main_path)
            if any(excluded_path in str(dir_path_relative) for excluded_path in self.excluded_paths):
                continue
            with os.scandir(dir_path) as dir_items:
                for item in dir_items:
                    if item.is_dir():
                        stack.append(item.path)
                    elif item.is_file():
                        filepath = Path(item.path)
                        if filepath.suffix in self.included_extensions:
                            # filename = remove_pattern_filename.sub('', filepath.name)
                            filename = filepath.stem
                            extension = filepath.suffix
                            if any(excluded_name in filename for excluded_name in self.excluded_names):
                                continue
                            for filename_suffix in self.filename_suffixes:
                                if filename_suffix in filename:
                                    filename = filename.replace(filename_suffix, '')
                                    extension = filename_suffix + extension
                            filepaths_and_extensions[dir_path_relative / filename].add(extension)


        # if isinstance(paths, str) or isinstance(paths, Path):
        #     #paths = paths.glob('**/*')
        #         #'**/?*.*')  # At least one character in front of the extension to prevent using hidden folders
        #
        #     # The following approach is faster than checking each file separately using is_file() for network drives. (not tested for regular drives)
        #     files_and_folders = set(paths.glob('**/*'))
        #     folders = set(paths.glob('**'))
        #     paths = files_and_folders - folders
        #
        # file_paths_and_extensions = \
        #     [[p.relative_to(self.main_path).with_suffix(''), p.suffix]
        #      for p in paths
        #      if (
        #              # Use only files
        #              #p.is_file() &
        #              # Exclude stings in filename
        #              all(name not in p.with_suffix('').name for name in
        #                  self.excluded_names) &
        #              # Exclude strings in path
        #              all(path not in str(p.relative_to(self.main_path).parent) for path in
        #                  self.excluded_paths) &
        #              # Exclude hidden folders
        #              ('.' not in [s[0] for s in p.parts]) &
        #              # Exclude file extensions
        #              (p.suffix[1:] not in self.excluded_extensions)
        #      )
        #      ]

        # TODO: Test spooled file and nd2 file import
        new_filepaths_and_extensions = defaultdict(set)
        for i, (filepath, extensions) in enumerate(filepaths_and_extensions.items()):
            if (filepath.name == 'Spooled files'):
                new_filepaths_and_extensions.append([filepath.parent, extensions])
                # file_paths_and_extensions[i, 0] = file_paths_and_extensions[i, 0].parent
            elif '.nd2' in extensions and not 'fov' in str(filepath):
                from papylio.movie.movie import Movie
                nd2_movie = Movie(filepath.with_suffix('.nd2'))
                if nd2_movie.number_of_fov > 1:  # if the file is nd2 with multiple field of views
                    for fov_id in range(nd2_movie.number_of_fov):
                        new_path = Path(str(filepath) + f'_fov{fov_id:03d}')
                        new_filepaths_and_extensions[filepath] = extensions
                        # fov_info['fov_chosen'] = fov_id
                        # new_file = File(new_path, self, fov_info=fov_info.copy())
                        # if new_file.extensions:
                        #     self.files.append(new_file)
                else:
                    new_filepaths_and_extensions[filepath] = extensions
            else:
                new_filepaths_and_extensions[filepath] = extensions
        filepaths_and_extensions = new_filepaths_and_extensions

        return filepaths_and_extensions

        # filepaths_and_extensions = np.array(filepaths_and_extensions)
        #
        # file_paths_and_extensions = filepaths_and_extensions[filepaths_and_extensions[:, 0].argsort()]
        # unique_file_paths, indices = np.unique(file_paths_and_extensions[:, 0], return_index=True)
        # extensions_per_filepath = np.split(file_paths_and_extensions[:, 1], indices[1:])

        # unique_filepaths = list(filepaths_and_extensions.keys())
        # extensions_per_filepath = list(filepaths_and_extensions.values())
        #
        # return unique_filepaths, extensions_per_filepath

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

        for file_path, extensions in tqdm.tqdm(file_paths_and_extensions.items(), 'Import files'):
            if not test_duplicates or (file_path.absolute().relative_to(self.main_path) not in self.file_paths):
                self.files.append(File(file_path, extensions, self, perform_logging=self.perform_logging))
            else:
                i = self.file_paths.find(file_path.absolute().relative_to(self.main_path))
                self.files[i].add_extensions(extensions)


    def determine_flatfield_and_darkfield_corrections(self, files, method='BaSiC', illumination_index=0, frame_index=0,
                                                      estimate_darkfield=True, **kwargs):
        """Determine and save flatfield and darkfield corrections for image shading.

        Calculates spatial shading corrections using the specified method and saves
        them as TIFF files in the main experiment directory. Automatically loads the
        corrections into the experiment.

        Parameters
        ----------
        files : FileCollection
            Collection of files to use for determining corrections
        method : str, optional
            Correction method to use (default: 'BaSiC')
        illumination_index : int, optional
            Index of the illumination pattern to correct (default: 0)
        frame_index : int, optional
            Frame index to use for correction calculation (default: 0)
        estimate_darkfield : bool, optional
            If True, also estimate and save darkfield correction (default: True)
        **kwargs
            Additional keyword arguments passed to spatial_shading_correction
        """
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
        """Load flatfield correction images from disk.

        Searches for flatfield correction TIFF files in the main experiment directory.
        If found, loads them and adds them to the common image corrections dataset.
        Updates all movie objects with the loaded corrections.
        """
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
                    movie.separate_channels(flatfield, movie.channel_rows, movie.channel_columns)

            self.common_image_corrections['flatfield_correction'] = flatfield_correction
        else:
            self.common_image_corrections = self.common_image_corrections.drop_vars('flatfield_correction', errors='ignore')

        self.add_common_image_corrections_to_movies()

    def load_darkfield_correction(self):
        """Load darkfield correction images from disk.

        Searches for darkfield correction TIFF files in the main experiment directory.
        If found, loads them and adds them to the common image corrections dataset.
        Updates all movie objects with the loaded corrections.
        """
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
                    movie.separate_channels(darkfield, movie.channel_rows, movie.channel_columns)

            self.common_image_corrections['darkfield_correction'] = darkfield_correction
        else:
            self.common_image_corrections = self.common_image_corrections.drop_vars('darkfield_correction', errors='ignore')

        self.add_common_image_corrections_to_movies()

    def add_common_image_corrections_to_movies(self):
        """Apply common image corrections to all movie objects.

        Sets the common image corrections (flatfield, darkfield) to all movie
        objects in the experiment so they can be applied during image processing.
        """
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
        # files = [file for file in exp.files if file.is_selected]
        # files = self.files

        if (fileSelection & moleculeSelection):
            molecules = [molecule for file in self.selected_files for molecule in file.selectedMolecules]
        elif (fileSelection & (not moleculeSelection)):
            molecules = [molecule for file in self.selected_files for molecule in file.molecules]
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

    def print_files(self):
        """Print a summary of all files in the experiment.

        Outputs a formatted representation of all files currently loaded
        in the experiment.
        """
        self.files.print()

    def plot_trace(self, files=None, query={}, **kwargs):
        """Plot molecule traces with interactive visualization.

        Opens an interactive window to visualize and inspect molecule traces
        from NetCDF files. Allows filtering of molecules using a query.

        Parameters
        ----------
        files : FileCollection, optional
            Files to plot traces from. If None, uses all files in experiment
        query : dict, optional
            Query dictionary to filter molecules (default: {})
        **kwargs
            Additional keyword arguments passed to TracePlotWindow
        """
        # from papylio.trace_plot import TraceAnalysisFrame

        if files is None:
            files = self.files

        file_paths = [file.relative_filepath.with_suffix('.nc') for file in files if '.nc' in file.extensions]

        with xr.open_mfdataset(file_paths, concat_dim='molecule', combine='nested') as ds:
            ds_sel = ds.query(query)  # HJ1_WT, HJ7_G116T
            from papylio.trace_plot import TracePlotWindow
            TracePlotWindow(dataset=ds_sel, save_path=None, **kwargs)

    def export_number_of_molecules_per_file(self):
        """Export the number of molecules in each file to an Excel spreadsheet.

        Creates an Excel file containing a summary of the number of molecules
        detected in each file of the experiment. Missing files are marked with -1.
        The file is saved as 'number_of_molecules.xlsx' in the main experiment directory.
        """
        df = pd.DataFrame(columns=['Number of molecules'])
        for i, file in enumerate(self.files):
            n = str(file.relative_filepath)
            try:
                nms = file.number_of_molecules
            except FileNotFoundError:
                nms = -1
            df.loc[n] = nms
        df.to_excel(self.main_path.joinpath('number_of_molecules'))

def set_default_matplotlib_colors(colorblind_friendly=True):
    if colorblind_friendly:
        # Replace default green and red colors with colorblind colors
        mcolors._colors_full_map['green'] = mcolors._colors_full_map['g'] = '#1EC509'
        mcolors._colors_full_map['red'] = mcolors._colors_full_map['r'] = '#DA0031'
        mcolors._colors_full_map['blue'] = mcolors._colors_full_map['b'] = '#0043FF'
    else:
        mcolors._colors_full_map['green'] = mcolors._colors_full_map['g'] = '#1f77b4'
        mcolors._colors_full_map['red'] = mcolors._colors_full_map['r'] = '#FF0000'
        mcolors._colors_full_map['blue'] = mcolors._colors_full_map['b'] = '#0000FF'
