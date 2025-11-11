import numpy as np
from pathlib import Path
import h5py
import re
import pandas as pd
import netCDF4
import tabulate
# import h5netcdf.legacyapi as netCDF4
# import h5netcdf
import tqdm
from contextlib import ExitStack
import xarray as xr
import re
import matplotlib.pyplot as plt

from papylio.plugins.sequencing.plotting import plot_cluster_locations_per_tile
#
# class DatasetPointer:
#     class_attributes = ['file_path', '_dataset']
#
#     def __init__(self, file_path):
#         self.__dict__['file_path'] = file_path
#         self.__dict__['_dataset'] = None
#
#     def __enter__(self):
#         self._dataset = xr.open_dataset(self.file_path, engine='netcdf4')
#         return self._dataset
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self._dataset.close()
#         self._dataset = None
#
#     def __repr__(self):
#         with self as ds:
#             return str(ds)
#
#     def __getattr__(self, item):
#         print(item)
#         with self as ds:
#             return ds[item].load()
#
#     def __setattr__(self, key, value):
#         if key in DatasetPointer.class_attributes:
#             super().__setattr__(key, value)
#         else:
#             print('__setattr__', key, value)
#             value = xr.DataArray(value)
#             value.name = key
#             value.to_netcdf(self.file_path, mode='a', engine='h5netcdf')
# For testing
# import pytest
# from trace_analysis.plugins.sequencing.sequencing_data import DatasetPointer
#
#
# def test_DatasetPointer(shared_datadir):
#     ds = DatasetPointer(shared_datadir / 'BN_TIRF_sequencing/HJ_scan TIRF 561 0300_sequencing_data.nc')
#     ds.tile



# Update class with new sam analysis function below
class SequencingData:

    reagent_kit_info = {'v2':       {'number_of_tiles': 14, 'number_of_surfaces': 2},
                        'v2_micro': {'number_of_tiles':  4, 'number_of_surfaces': 2},
                        'v2_nano':  {'number_of_tiles':  2, 'number_of_surfaces': 1},
                        'v3':       {'number_of_tiles': 19, 'number_of_surfaces': 2}}

    # @classmethod
    # def load(cls):
    #     cls()

    def __init__(self, file_path=None, dataset=None, name='', reagent_kit='v3', load=True, file_kwargs={}, save_path=None):
        if file_path is not None:
            file_path = Path(file_path)
            if file_path.suffix == '.nc':
                # with xr.open_dataset(file_path.with_suffix('.nc'), engine='h5netcdf') as dataset:
                #     self.dataset = dataset.load().set_index({'sequence': ('tile','x','y')})
                if load:
                    self.dataset = xr.load_dataset(file_path.with_suffix('.nc'), engine='netcdf4')
                else:
                    self.dataset = xr.open_dataset(file_path.with_suffix('.nc'), engine='netcdf4')#, chunks=10000)
                    # self.dataset = DatasetPointer(file_path.with_suffix('.nc'))
                # self.dataset = self.dataset.set_index({'sequence': ('tile', 'x', 'y')})
                # self.dataset.coordsupdate
                # self.dataset.update(
                #     {'sequence': pd.MultiIndex.from_frame(self.dataset[['tile', 'x', 'y']].to_pandas())})

            else:
                data = pd.read_csv(file_path, delimiter='\t')
                data.columns = data.columns.str.lower()
                data = data.set_index(['tile', 'x', 'y'])
                data = data.drop_duplicates() # TODO: It would be better to do this when converting the sam file.
                data.index.name = 'sequence' # Do this after drop_duplicates!
                self.dataset = xr.Dataset(data) #.reset_index('sequence', drop=True)

                if name == '':
                    name = Path(file_path).name

        elif dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError('Either file_path or data should be given')

        self.name = name
        if save_path is None:
            self.save_path = file_path.parent
        else:
            self.save_path = save_path

        self.reagent_kit = reagent_kit
        self.reagent_kit_info = SequencingData.reagent_kit_info[reagent_kit]

        self._tiles = None


    def __getattr__(self, item):
        #try:
        # print('seqdata=' + item)
        if 'dataset' in self.__dict__.keys() and hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        #except AttributeError:
        #    super().__getattribute__(item)
        else:
            raise AttributeError
            # super(SequencingData, self).__getattribute__(item)

        # try:
        #     return getattr(self.dataset, item)
        # except AttributeError:
        #     super().__getattribute__(item)

    def __getitem__(self, item):
        return SequencingData(dataset=self.dataset[dict(sequence=item)], reagent_kit=self.reagent_kit, save_path=self.save_path)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')

    def __len__(self):
        return len(self.dataset.sequence)

    @property
    def coordinates(self):
        return xr.DataArray([self.dataset['x'], self.dataset['y']], dims=('dimension','sequence'),
                            coords={'dimension': ['x','y'], 'sequence': self.dataset.sequence, 'tile': self.dataset.tile}, name='coordinates').T
        # return self.dataset[['x','y']].to_array(dim='dimension', name='coordinates').transpose('sequence',...)
            # xr.DataArray(self.data[['x','y']], dims=['sequence', 'dimension'], name='coordinates')\
            # .reset_index('sequence', drop=True)

    @property
    def tile_numbers(self):
        return np.unique(self.dataset.tile)

    # @property
    # def tiles(self):
    #     if not self._tiles:
    #         # Perhaps more elegant if this returns SequencingData objects [25-10-2021 IS]
    #         self._tiles = [Tile(tile, tile_coordinates) for tile, tile_coordinates in self.coordinates.groupby('tile')]
    #     return self._tiles

    def sel(self, *args, **kwargs):
        return SequencingData(dataset=self.dataset.sel(*args, **kwargs))

    # def classify_sequences(self, variable=''):

    def reference_distribution(self, save=True, report=True):
        da_reference_name = self.dataset.reference_name.load()
        da_sequence_subset = self.dataset.sequence_subset.load()

        reference_names, counts = np.unique(da_reference_name, return_counts=True)

        analysis_reference = xr.Dataset(coords={'reference_name': reference_names})
        analysis_reference['reference_count'] = xr.DataArray(counts, dims='reference_name')
        analysis_reference['reference_fraction'] = xr.DataArray(counts / counts.sum(), dims='reference_name')

        full_subset_counts = np.zeros(len(reference_names), dtype='int64')
        for i, reference_name in enumerate(reference_names):
            da_sequence_subset_sel = da_sequence_subset.sel(sequence=da_reference_name == reference_name)
            full_subset_counts[i] = (~da_sequence_subset_sel.str.contains('-')).sum().item()
        analysis_reference['full_subset_count'] = xr.DataArray(full_subset_counts, dims='reference_name')
        analysis_reference['full_subset_fraction'] = analysis_reference['full_subset_count'] / analysis_reference[
            'reference_count']

        # reference_names[reference_names=='*'] = 'Unmapped'
        # analysis_reference.reindex({'reference': reference_names})
        if save:
            analysis_reference.to_netcdf(self.save_path / 'reference_distribution.nc')

        if report:
            string = 'Mapped sequences \n================\n\n' + \
                     tabulate.tabulate(analysis_reference.to_pandas(),
                                       headers=["Reference name", "Reference\nCount", "\nPercentage",
                                                "Full subset\nCount", "\nPercentage"],
                                       floatfmt=(None, ".0f", ".2%", ".0f", ".2%"))
            print(string)
            self.add_to_report_file(string)

        return analysis_reference

    def show_reference_distribution(self, save=True):
        analysis_reference = self.reference_distribution(save=save)

        fig, ax = plt.subplots(layout='tight', figsize=(analysis_reference.reference_name.size,3))
        ax.bar(analysis_reference['reference_name'], analysis_reference['reference_count'], fc='grey')
        total = analysis_reference['reference_count'].sum().item()
        secax = ax.secondary_yaxis('right', functions=(lambda x: x/total, lambda x: x*total))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
        ax.set_xlabel('Reference')
        ax.set_ylabel('Count')
        secax.set_ylabel('Fraction')
        ax.set_title('Reference distribution')

        if hasattr(analysis_reference, 'full_subset_count'):
            ax.bar(analysis_reference['reference_name'], analysis_reference['full_subset_count'], fc='grey', ec='white', lw=0, label='Complete subset', hatch='/////')
            ax.set_ylim(0, ax.get_ylim()[1]*1.2)
            ax.legend(frameon=False, loc='upper right') # bbox_to_anchor=(1, 1),

        if save:
            fig.savefig(self.save_path / 'reference_distribution.png')
            fig.savefig(self.save_path / 'reference_distribution.pdf')

    def distribution(self, variable='reference_name', save=True, report=True, remove_empty_strings=False):
        da_variable = self.dataset[variable].load()
        da_sequence_subset = self.dataset.sequence_subset.load()

        if remove_empty_strings:
            selection = da_variable != ''
            da_variable = da_variable[selection]
            da_sequence_subset = da_sequence_subset[selection]

        names, counts = np.unique(da_variable, return_counts=True)

        ds = xr.Dataset(coords={variable: names})
        ds['count'] = xr.DataArray(counts, dims=variable)
        ds['fraction'] = xr.DataArray(counts / counts.sum(), dims=variable)

        full_subset_counts = np.zeros(len(names), dtype='int64')
        for i, name in enumerate(names):
            da_sequence_subset_sel = da_sequence_subset.sel(sequence=da_variable == name)
            full_subset_counts[i] = (~da_sequence_subset_sel.str.contains('-')).sum().item()
        ds['full_subset_count'] = xr.DataArray(full_subset_counts, dims=variable)
        ds['full_subset_fraction'] = ds['full_subset_count'] / ds['count']

        # reference_names[reference_names=='*'] = 'Unmapped'
        # analysis_reference.reindex({'reference': reference_names})
        if save:
            ds.to_netcdf(self.save_path / f'{variable}_distribution.nc')

        if report:
            string = 'Mapped sequences \n================\n\n' + \
                     tabulate.tabulate(ds.to_pandas(),
                                       headers=[variable, "Count", "\nPercentage",
                                                "Full subset\nCount", "\nPercentage"],
                                       floatfmt=(None, ".0f", ".2%", ".0f", ".2%"))
            print(string)
            self.add_to_report_file(string)

        return ds

    def show_distribution(self, variable='reference_name', save=True):
        ds = self.distribution(variable=variable, save=save)

        #TODO: Move to plotting.py ?
        fig, ax = plt.subplots(layout='tight', figsize=(ds[variable].size+1, 3))
        ds[variable] = ds[variable].astype('U').str.replace('_','\n')
        ax.bar(ds[variable], ds['count'], fc='grey')
        total = ds['count'].sum().item()
        secax = ax.secondary_yaxis('right', functions=(lambda x: x / total, lambda x: x * total))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.set_xlabel(variable)
        ax.set_ylabel('Count')
        secax.set_ylabel('Fraction')
        ax.set_title(f'{variable} distribution')

        if hasattr(ds, 'full_subset_count'):
            ax.bar(ds[variable], ds['full_subset_count'], fc='grey', ec='white',
                   lw=0, label='Complete subset', hatch='/////')
            ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
            ax.legend(frameon=False, loc='upper right')  # bbox_to_anchor=(1, 1),

        if save:
            fig.savefig(self.save_path / f'{variable}_distribution.png')
            fig.savefig(self.save_path / f'{variable}_distribution.pdf')

    def get_sequences(self, reference_name=None, variable='read1_sequence', remove_incomplete_sequences=True):
        dataset = self.dataset
        if reference_name is not None:
            dataset = dataset.sel(sequence=self.dataset.reference_name == reference_name)

        sequences = dataset[variable].load()

        if remove_incomplete_sequences:
            is_complete = ~np.array(['-' in sequence for sequence in sequences.values])
            sequences = sequences[is_complete]

        return sequences

    def sequence_count(self, variable='sequence_subset',  remove_incomplete_sequences=True, save=True):
        sequences = self.get_sequences(variable=variable, remove_incomplete_sequences=remove_incomplete_sequences)
        n, c = np.unique(sequences, return_counts=True)
        sequence_count = xr.DataArray(c, coords={variable: n})

        sequence_count.name = f'{self.name} - sequence_count - {variable}'
        sequence_count.attrs['variable'] = variable
        sequence_count.attrs['remove_incomplete_sequences'] = str(remove_incomplete_sequences)

        if save:
            sequence_count.to_netcdf(self.save_path / f'{self.name} - sequence_count - {variable}.nc')

        return sequence_count

    def base_composition(self, variable='read1_sequence', positions=None, remove_incomplete_sequences=True, save=True):
        sequences = self.get_sequences(variable=variable, remove_incomplete_sequences=remove_incomplete_sequences)
        if positions is None:
            positions = np.arange(len(sequences[0].item()))

        base_count = xr.DataArray(0, dims=['position', 'base'],
                                  coords={'position': positions, 'base': ['A', 'T', 'C', 'G']})

        for p in tqdm.tqdm(base_count.position.values):
            base_count[dict(position=p)] = \
                [(sequences.str.get(p) == b).sum().item() for b in base_count.base.values]

        base_fractions = base_count / len(sequences)

        base_composition = xr.Dataset()
        base_composition['base_count'] = base_count
        base_composition['base_fraction'] = base_fractions

        base_composition.attrs['variable'] = variable
        base_composition.attrs['remove_incomplete_sequences'] = str(remove_incomplete_sequences)

        if save:
            base_composition.to_netcdf(self.save_path / f'{self.name} - base_composition - {variable}.nc')

        return base_composition

    def show_base_composition(self, save=True, **kwargs):
        base_composition = self.base_composition(**kwargs)
        import logomaker

        figure, axis = plt.subplots(figsize=(min(1+base_composition.position.size/2, 10), 2.5), layout='constrained')
        nn_logo = logomaker.Logo(base_composition.base_fraction.to_pandas(), stack_order='fixed', ax=axis)
        axis.set_ylabel('Fraction')
        axis.set_xlabel('Position')
        variable = base_composition.attrs['variable']
        axis.set_title(f'{self.name} - base composition - {variable}')
        if save:
            figure.savefig(self.save_path / f'{self.name} - base_composition - {variable}.png')
            figure.savefig(self.save_path / f'{self.name} - base_composition - {variable}.pdf')

    def basepair_count(self, variable='read1_sequence', positions=None, remove_incomplete_sequences=True, save=True):
        filepath = self.save_path / f'{self.name} - basepair_count - {variable}.nc'
        if filepath.exists():
            basepair_count = xr.load_dataarray(filepath, engine='netcdf4')
            return basepair_count

        sequences = self.get_sequences(variable=variable, remove_incomplete_sequences=remove_incomplete_sequences)
        if positions is None:
            positions = np.arange(len(sequences[0].item()))

        bases = ['A', 'T', 'C', 'G']
        basepair_count = xr.DataArray(0, dims=['position_0', 'position_1', 'base_0', 'base_1'],
                                      coords={'position_0': positions, 'position_1': positions, 'base_0': bases,
                                              'base_1': bases})
        basepair_count.name = f'{self.name} - basepair_count - {variable}'
        basepair_count.attrs['variable'] = variable
        basepair_count.attrs['remove_incomplete_sequences'] = str(remove_incomplete_sequences)


        for position_0 in tqdm.tqdm(positions):
            bases_position_0 = sequences.str.get(position_0)
            for position_1 in positions:
                bases_position_1 = sequences.str.get(position_1)
                for base_0 in bases:
                    for base_1 in bases:
                        basepair_count.loc[
                            dict(position_0=position_0, position_1=position_1, base_0=base_0, base_1=base_1)] = \
                            ((bases_position_0 == base_0) & (bases_position_1 == base_1)).sum()
        if save:
            basepair_count.to_netcdf(self.save_path / (basepair_count.name + '.nc'))

        return basepair_count

    def new_report_file(self):
        report = open(self.save_path / 'report.txt', 'w')
        report.close()

    def add_to_report_file(self, string):
        report = open(self.save_path / 'report.txt', 'a')
        report.write(string)
        report.close()

    def plot_cluster_locations_per_tile(self, save_filepath=None):
        # TODO: Fix bug self.dataset[['x','y']]
        plot_cluster_locations_per_tile(self.dataset[['tile','x','y']], **self.reagent_kit_info, save_filepath=save_filepath)

    def save(self, filepath):
        self.dataset.reset_index('sequence').to_netcdf(filepath, engine='netcdf4', mode='w')

class Tile:
    def __init__(self, number, coordinates):
        self.name = str(number)
        self.number = number
        self.coordinates = coordinates

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')



def read_sam(sam_filepath, add_aligned_sequence=False, extract_sequence_subset=False, read_name='read1'):
    sam_filepath = Path(sam_filepath)

    # Check number of header lines:
    with Path(sam_filepath).open('r') as file:
        number_of_header_lines = 0
        while file.readline()[0] == '@':
            number_of_header_lines += 1

    df = pd.read_csv(sam_filepath,
                     delimiter='\t', skiprows=number_of_header_lines, usecols=range(11), header=None,
                     names=['sequence_identifier', 'sam_flag', 'reference_name', 'position', 'mapping_quality', 'cigar_string',
                            'mate_reference_name', 'mate_position', 'template_length', name+'_sequence', name+'_quality'])
    df = df.drop_duplicates(subset='sequence_identifier', keep='first', ignore_index=False)
    df.index.name = 'sequence'
    df_split = df['sequence_identifier'].str.split(':', expand=True)
    df_split.columns = ['instrument', 'run', 'flowcell', 'lane', 'tile', 'x', 'y']
    df_split = df_split.astype({'run': int, 'lane': int, 'tile': int, 'x': int, 'y': int})
    df_joined = df_split.join(df.iloc[:, 1:])

    if add_aligned_sequence:
        df_joined[[read_name+'_sequence_aligned', read_name+'_quality_aligned']] = \
            df_joined.apply(get_aligned_sequence_from_row, axis=1, result_type='expand', name=name)

    if extract_sequence_subset:
        df_joined['sequence_subset'] = extract_positions(df_joined[read_name+'_sequence_aligned'],
                                                         extract_sequence_subset)
        df_joined['quality_subset'] = extract_positions(df_joined[read_name+'_quality_aligned'],
                                                        extract_sequence_subset)

    return df_joined


def read_sam_header(sam_filepath):
    header_dict = {'HD': [], 'SQ': [], 'RG': [], 'PG':[], 'CO':[]}
    with Path(sam_filepath).open('r') as file:
        number_of_header_lines = 0
        # read_line = file.readline().split('\t')
        while (read_line := file.readline().rstrip('\n').split('\t'))[0][0] == '@':
        # while read_line[0][0] == '@':
            line_data = {item.split(':')[0]: item.split(':')[1] for item in read_line[1:]}
            header_dict[read_line[0][1:]].append(line_data)
            # read_line = file.readline().split('\t')

            number_of_header_lines += 1

    return number_of_header_lines, header_dict
#
# def number_of_lines(filepath):
#     def _make_gen(reader):
#         while True:
#             b = reader(2 ** 16)
#             if not b: break
#             yield b
#
#     with open(filepath, "rb") as f:
#         count = sum(buf.count(b"\n") for buf in tqdm.tqdm(_make_gen(f.raw.read)))
#     return count

def number_of_lines(filepath):
    return sum(1 for line in tqdm.tqdm(filepath.open('r')))

def number_of_sequence_alignments(sam_filepath):
    return sum(1 if line[0] != '@' else 0 for line in tqdm.tqdm(sam_filepath.open('r')))

def bitwise_flag(flag, bit):
    flag = np.array(flag)
    return (flag & (2**bit)).astype(bool)

def number_of_primary_sequence_alignments(sam_filepath):
    sam_filepath = Path(sam_filepath)
    compiled_regex = re.compile('(?<=\t)\d*(?=\t)')
    is_primary_alignment_list = []
    for line in tqdm.tqdm(sam_filepath.open('r'), 'Determine number of primary alignments'):
        if line[0] != '@': # Remove header lines
            is_primary_alignment = not bitwise_flag(int(compiled_regex.search(line).group()), 11)
            is_primary_alignment_list.append(is_primary_alignment)
    return np.sum(is_primary_alignment_list)

    # return sum(bitwise_flag(int(compiled_regex.search(line).group()), 11)
    #            if line[0]!='@' else 0 for line in tqdm.tqdm(sam_filepath.open('r')))


def parse_sam(sam_filepath, read_name='read1', remove_duplicates=True, add_aligned_sequence=False, extract_sequence_subset=False,
              chunksize=10000, write_csv=False, write_nc=True, write_filepath=None):
    sam_filepath = Path(sam_filepath)

    # Check number of header lines:
    number_of_header_lines, header_dict = read_sam_header(sam_filepath)

    name_dict = {'*': 0, '=': 1}
    name_dict.update({SQ_dict['SN']:i+2 for i, SQ_dict in enumerate(header_dict['SQ'])})

    max_reference_length = np.array([int(d['LN']) for d in header_dict['SQ']]).max()

    number_of_sequences = number_of_primary_sequence_alignments(sam_filepath)

    end_index = 0

    with ExitStack() as stack:

        with pd.read_csv(sam_filepath,
                         delimiter='\t', skiprows=number_of_header_lines, usecols=range(11), header=None,
                         names=['sequence_identifier', 'sam_flag', 'reference_name', 'position', 'mapping_quality',
                                'cigar_string', 'mate_reference_name', 'mate_position', 'template_length',
                                read_name + '_sequence', read_name + '_quality'],
                         chunksize=chunksize) as reader:

            for i, chunk in tqdm.tqdm(enumerate(reader), 'Parse sam file', total=number_of_sequences//chunksize+1):
                df_chunk = chunk
                df_chunk.index.name = 'sequence'
                if remove_duplicates:
                    df_chunk = df_chunk[~bitwise_flag(df_chunk.sam_flag, 11)]
                    # df_chunk = df_chunk[df_chunk.sam_flag//2048 == 0]
                df_split = df_chunk['sequence_identifier'].str.split(':', expand=True)
                df_split.columns = ['instrument', 'run', 'flowcell', 'lane', 'tile', 'x', 'y']
                df_split = df_split.astype({'run': int, 'lane': int, 'tile': int, 'x': int, 'y': int})
                df = df_split.join(df_chunk.iloc[:, 1:])

                if add_aligned_sequence:
                    df[[read_name+'_sequence_aligned', read_name+'_quality_aligned']] = \
                        df.apply(get_aligned_sequence_from_row, axis=1, result_type='expand', read_name=read_name,
                                 reference_range=(0, max_reference_length))

                if extract_sequence_subset:
                    df['sequence_subset'] = extract_positions(df[read_name+'_sequence_aligned'], extract_sequence_subset)
                    df['quality_subset'] = extract_positions(df[read_name+'_quality_aligned'], extract_sequence_subset)

                if write_filepath is None:
                    write_filepath = sam_filepath.with_suffix('')

                if write_csv:
                    if i == 0:
                        csv_filepath = write_filepath.with_suffix('.csv')
                        df.to_csv(csv_filepath, header=True, mode='w')
                    else:
                        df.to_csv(csv_filepath, header=False, mode='a')

                if write_nc:
                    if i == 0:
                        nc_filepath = write_filepath.with_suffix('.nc')
                        # with netCDF4.Dataset(nc_filepath, 'w'):
                        #     nc_file.createDimension('sequence', None)

                        nc_file = stack.enter_context(netCDF4.Dataset(nc_filepath, 'w'))
                        nc_file.createDimension('sequence', number_of_sequences)
                        for name, datatype in df.dtypes.items():
                            if name in ['instrument','run','flowcell']:
                                setattr(nc_file, name, df[name][0])
                            elif name in ['reference_name', 'mate_reference_name']:
                                # nc_file.createVariable(name, np.uint8, ('sequence',))
                                # nc_file[name].enum_dict = name_dict
                                size = np.array(list(name_dict.keys())).astype('S').itemsize
                                create_string_variable_in_nc_file(nc_file, name, dimensions=('sequence',), size=size)
                            elif (read_name+'_sequence') in name or (read_name+'_quality') in name or 'subset' in name:
                                size = len(df[name][0])
                                create_string_variable_in_nc_file(nc_file, name, dimensions=('sequence',), size=size)
                            elif name in ['cigar_string']:
                            # if datatype == np.dtype('O'):
                                # nc_file.createDimension(name + '_size', None)
                                # nc_file.createVariable(name, 'S1', ('sequence', name + '_size'), chunksizes=(10000, 1))
                                # nc_file[name]._Encoding = 'utf-8'
                                create_string_variable_in_nc_file(nc_file, name, dimensions=('sequence',), chunksizes=(1, 25))
                            else:
                                nc_file.createVariable(name, datatype, ('sequence', ))

                    start_index = end_index
                    end_index = start_index + len(df)
                    for name, datatype in df.dtypes.items():
                        # print(name)
                        if name in ['instrument', 'run', 'flowcell']:
                            continue
                        elif datatype == np.dtype('O'):
                            size = np.max([2, nc_file.dimensions[name+'_size'].size, df[name].str.len().max()])
                            nc_file[name][start_index:end_index] = df[name].values.astype(f'S{size}')
                            # nc_file[name][old_size:, :] = df[name].values.astype(f'S{size}').view('S1').reshape(-1, size)
                        else:
                            nc_file[name][start_index:end_index] = df[name].values

                    # To set the coords for xarray.
                    nc_file.setncattr('coordinates', 'tile x y')

            # Use this if xarray should open the file with standard datatype "|S" instead of "object"
            # for name, datatype in df.dtypes.items():
            #     print(name)
            #     if datatype == np.dtype('O'):
            #         delattr(nc_file[name], '_Encoding')


                    #
                    #     nc_file['cigar_string'][:] = df[0:200].cigar_string.astype('S').values
                    #     # delattr(nc_file['cigar_string'], '_Encoding')
                    #     nc_file.close()
                    #
                    #     test = xr.load_dataset(write_filepath.with_suffix('.nc'))
                    #
                    #
                    #     nc_file = stack.enter_context(h5netcdf.File(nc_filepath, 'a'))
                    #     nc_file.dimensions = {'sequence': None}
                    #     for name, datatype in df.dtypes.items():
                    #         if datatype == np.dtype('O'):
                    #             # datatype = h5py.string_dtype(encoding='utf-8')
                    #             datatype = 'S10'
                    #         else:
                    #             nc_file.create_variable(name, ('sequence',), data=None, dtype=datatype, chunks=(chunksize,))
                    # old_size = nc_file.dimensions['sequence'].size
                    # new_size = old_size + len(df)
                    # nc_file.resize_dimension('sequence', new_size)
                    # added_data_slice = slice(old_size, new_size)
                    # for name in df.columns:
                    #     nc_file[name][added_data_slice] = df[name].values

    #
    # encoding = {key: {'dtype': '|S'} for key in ds.keys() if ds[key].dtype == 'object'}
    #
    # keys = ['instrument', 'run', 'flowcell', 'lane', 'tile', 'x', 'y', 'sam_flag', 'contig_name', 'first_base_position', 'mapping_quality', 'cigar_string', 'mate_name', 'mate_position', 'template_length', 'read_sequence', 'read_quality', 'read_sequence_aligned', 'read_quality_aligned', 'sequence_subset', 'quality_subset', 'index1_sequence', 'index1_quality']
    # with xr.open_dataset(
    #         r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220607 - Sequencer (MiSeq)\Analysis\sequencing_data.nc') as ds:
    #     keys = list(ds.keys())
    # for key in keys:
    #     print(key)
    #     with xr.open_dataset(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220607 - Sequencer (MiSeq)\Analysis\sequencing_data.nc') as ds:
    #         da = ds[key].load()
    #         if da.dtype == 'object':
    #             encoding = {key: {'dtype': '|S'}}
    #         else:
    #             encoding = {}
    #         da.to_netcdf(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220607 - Sequencer (MiSeq)\Analysis\sequencing_data_S.nc', engine='h5netcdf', mode='a', encoding=encoding)


# for key in ds.keys():
#     if ds[key].dtype == 'object':
#         print(key)


def fastq_data(fastq_filepath, read_name):
    tile_list = []
    x_list = []
    y_list = []
    sequence = []
    quality = []
    expr = re.compile('[:@ \n]')
    with Path(fastq_filepath).open('r') as fq_file:
        for line_index, line in enumerate(tqdm.tqdm(fq_file)):
            if line_index % 4 == 0:
                name = line.strip()
                instrument, run, flowcell, lane, tile, x, y = expr.split(name)[1:8]
                # sequence_index = numbered_index.loc[dict(sequence=(int(tile), int(x), int(y)))].item()
                tile_list.append(int(tile))
                x_list.append(int(x))
                y_list.append(int(y))

            if line_index % 4 == 1:
                sequence.append(line.strip())
            if line_index % 4 == 3:
                quality.append(line.strip())
    ds = xr.Dataset({read_name+'_sequence': ('sequence', sequence), read_name+'_quality': ('sequence', quality)},
                      coords={'tile': ('sequence', tile_list), 'x': ('sequence', x_list), 'y': ('sequence', y_list)})
    return ds


def fastq_generator(fastq_filepath, chunksize):
    expr = re.compile('[:@ \n]')
    data = {'instrument': [], 'run': [], 'flowcell': [], 'lane': [],
            'tile': [], 'x': [], 'y': [], 'sequence': [], 'quality': []} # Perhaps better to use a numpy struct array
    current_chunk_size = 0
    with Path(fastq_filepath).open('r') as fq_file:
        for line_index, line in enumerate(fq_file):
            if line_index % 4 == 0:
                name = line.strip()
                instrument, run, flowcell, lane, tile, x, y = expr.split(name)[1:8]
                data['instrument'].append(instrument)
                data['run'].append(run)
                data['flowcell'].append(flowcell)
                data['lane'].append(lane)

                # sequence_index = numbered_index.loc[dict(sequence=(int(tile), int(x), int(y)))].item()
                data['tile'].append(int(tile))
                data['x'].append(int(x))
                data['y'].append(int(y))

            if line_index % 4 == 1:
                data['sequence'].append(line.strip())
            if line_index % 4 == 3:
                data['quality'].append(line.strip())
                # yield instrument_name, run_id, flowcell_id, lane, tile, x, y, sequence, quality

                current_chunk_size += 1

            if current_chunk_size == chunksize:
                yield data
                data = {'instrument': [], 'run': [], 'flowcell': [], 'lane': [],
                        'tile': [], 'x': [], 'y': [], 'sequence': [], 'quality': []}
                current_chunk_size = 0

        yield data # Final round

# def add_sequence_data_to_dataset(nc_filepath, fastq_filepath, name):
#     with xr.open_dataset(nc_filepath, engine='h5netcdf') as ds:
#         sequence_multiindex = ds[['tile', 'x', 'y']].load().set_index({'sequence':('tile','x','y')})#.indexes['sequence']
#
#     ds = fastq_data(fastq_filepath)
#     ds = ds.rename_vars({'read_sequence':f'{name}_sequence', 'read_quality': f'{name}_quality'})
#     ds = ds.set_index({'sequence': ('tile','x','y')})
#     ds, = xr.align(ds, indexes=sequence_multiindex.indexes, copy=False)
#     ds.reset_index('sequence',drop=True)[[f'{name}_sequence', f'{name}_quality']].to_netcdf(nc_filepath, mode='a', engine='h5netcdf')


def create_string_variable_in_nc_file(nc_file, variable_name, size=None, **kwargs):
    kwargs['dimensions'] += (variable_name + '_size',)
    # if not 'chunksizes' in kwargs.keys():
    #     kwargs['chunksizes'] = (10000, 1)
    nc_file.createDimension(variable_name + '_size', size)
    nc_file.createVariable(variable_name, 'S1', **kwargs)
    # nc_file[variable_name][:] = np.repeat(fill_value*size, nc_file.dimensions[kwargs['dimensions'][0]].size).astype(f'S{dtype_size}')
    # if fill_value is not None:
    #     nc_file[variable_name][:, :] = fill_value.encode()
    nc_file[variable_name]._Encoding = 'utf-8'



def add_sequence_data_to_dataset(nc_filepath, fastq_filepath, read_name, chunksize=10000):
    with netCDF4.Dataset(nc_filepath, 'a') as nc_file:
        nc_file['lane'][-1] = nc_file['lane'][-1]
        # To prevent that unlimited sequence dim is resized after reopening
        # Can probably be removed once libnetcdf 4.9 can be used.

    # import h5netcdf.legacyapi as h5netcdf
    # with h5netcdf.Dataset(nc_filepath, 'a') as nc_file:
        end_index = 0
        for i, sequence_data in tqdm.tqdm(enumerate(fastq_generator(fastq_filepath, chunksize=chunksize)), f'Add {read_name} to sequencing dataset', total=len(nc_file.dimensions['sequence'])//chunksize+1):
            if i == 0:
                size = len(sequence_data['sequence'][0])
                sequence_variable_name = read_name + '_sequence'
                create_string_variable_in_nc_file(nc_file, sequence_variable_name, dimensions=('sequence',), size=size)

                quality_variable_name = read_name + '_quality'
                create_string_variable_in_nc_file(nc_file, quality_variable_name, dimensions=('sequence',), size=size)

            start_index = end_index
            end_index = start_index + len(sequence_data['instrument'])
            # sequence_slice = slice(i * chunksize, (i + 1) * len(sequence_data))
            sequence_slice = slice(start_index, end_index)

            if (nc_file['tile'][sequence_slice] == sequence_data['tile']).all() & \
                    (nc_file['x'][sequence_slice] == sequence_data['x']).all() & \
                    (nc_file['y'][sequence_slice] == sequence_data['y']).all():
                nc_file[sequence_variable_name][sequence_slice, :] = np.array(sequence_data['sequence']).astype('S')
                nc_file[quality_variable_name][sequence_slice, :] = np.array(sequence_data['quality']).astype('S')
                pass
            else:
                raise ValueError()
#
#
# import xarray as xr
# from tqdm import tqdm
# nc_filepath = sam_filepath.with_suffix('.nc')
# with xr.open_dataset(nc_filepath, engine='h5netcdf') as ds:
#     sequence_multiindex = ds[['tile_number','x','y']].set_index({'sequence':('tile_number','x','y')}).indexes['sequence']
# numbered_index = xr.DataArray(np.arange(len(sequence_multiindex)), dims=('sequence',), coords={'sequence': sequence_multiindex})
# #
# # reference_array = ds[['tile','x','y']].to_array().T
# #
# # index1_sequence = xr.DataArray(np.empty(len(ds.sequence), dtype=str), dims=('sequence',), coords={'sequence': ds.sequence})
# # index1_quality = xr.DataArray(np.empty(len(ds.sequence), dtype=str), dims=('sequence',), coords={'sequence': ds.sequence})
# fastq_filepath = r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20211011 - Sequencer (MiSeq)\Analysis\Index1.fastq'
# with h5netcdf.File(nc_filepath, 'a') as nc_file:
#     if not 'index1_sequence' in nc_file:
#         nc_file.create_variable('index1_sequence', ('sequence',),  h5py.string_dtype(encoding='utf-8'))
#     if not 'index1_quality' in nc_file:
#         nc_file.create_variable('index1_quality', ('sequence',),  h5py.string_dtype(encoding='utf-8'))
#     sequence_index = 0
#     with Path(fastq_filepath).open('r') as fq_file:
#         for line_index, line in enumerate(tqdm(fq_file)):
#             if line_index % 4 == 0:
#                 name = line.strip()
#                 instrument_name, run_id, flowcell_id, lane, tile, x, y = re.split('[:@ \n]', name)[1:8]
#                 #sequence_index = numbered_index.loc[dict(sequence=(int(tile), int(x), int(y)))].item()
#
#             if line_index % 4 == 1:
#                 nc_file['index1_sequence'][sequence_index] = line.strip()
#             if line_index % 4 == 3:
#                 nc_file['index1_quality'][sequence_index] = line.strip()
#                 sequence_index += 1





        #index1_sequence.loc[dict(sequence=(int(tile), int(x), int(y)))] = fq_file.readline().strip()
        #fq_file.readline()
        #index1_quality.loc[dict(sequence=(int(tile), int(x), int(y)))] = fq_file.readline().strip()

def get_aligned_sequence(read_sequence, read_quality, cigar_string, position, reference_range=None):
    if reference_range is None:
        reference_range = (0, len(read_sequence))
    output_length = reference_range[1]-reference_range[0]

    if cigar_string == '*':
        return ('-'*output_length,' '*output_length)

    # cigar_string.split('(?<=M|I|D|N|S|H|P|=|X)')
    # split_cigar_string = re.findall(r'[0-9]*[MIDNSHP=X]', cigar_string)
    cigar_string_split = [(int(s[:-1]), s[-1]) for s in re.findall(r'[0-9]*[MIDNSHP=X]', cigar_string)]
    read_index = 0 # in read_sequence
    aligned_sequence = ''
    aligned_quality = ''

    if cigar_string_split[0][1] == 'S':
        read_index += cigar_string_split[0][0]
        cigar_string_split.pop(0)

    aligned_sequence += '-' * (position - 1)
    aligned_quality += ' ' * (position - 1)
    # read_index = position - 1

    for length, code in cigar_string_split:
        if code in ['M','=','X']:
            aligned_sequence += read_sequence[read_index:read_index+length]
            aligned_quality += read_quality[read_index:read_index+length]
            read_index += length
        elif code == 'I':
            read_index += length
        elif code == 'S':
            aligned_sequence += '-'*length
            aligned_quality += ' ' * length
            read_index += length
        elif code in ['D','N']:
            aligned_sequence += '-' * length
            aligned_quality += ' ' * length

    length_difference = len(aligned_sequence) - output_length
    if length_difference > 0:
        aligned_sequence = aligned_sequence[:output_length]
        aligned_quality = aligned_quality[:output_length]
    elif length_difference < 0:
        aligned_sequence += (-length_difference) * '-'
        aligned_quality += (-length_difference) * ' '

    return aligned_sequence, aligned_quality

def get_aligned_sequence_from_row(df_row, read_name, reference_range=None):
    sequence = df_row[read_name+'_sequence']
    quality = df_row[read_name+'_quality']
    cigar_string = df_row['cigar_string']
    position = df_row['position']
    return get_aligned_sequence(sequence, quality, cigar_string, position, reference_range)

########### Code to dynamically get aligned bases froj
# def get_aligned_position(index, cigar_string, first_base_position):
#     if cigar_string == '*':
#         return -1
#
#     # cigar_string.split('(?<=M|I|D|N|S|H|P|=|X)')
#     # split_cigar_string = re.findall(r'[0-9]*[MIDNSHP=X]', cigar_string)
#     cigar_string_split = [(int(s[:-1]), s[-1]) for s in re.findall(r'[0-9]*[MIDNSHP=X]', cigar_string)]
#     query_index = 0 # in read_sequence
#     reference_index = first_base_position-1
#     for length, code in cigar_string_split:
#         if code in ['M','I','S','=','X']:
#             query_index += length
#         if code in ['M','D','N','=','X']:
#             reference_index += length
#         #print(f'Reference index: {reference_index}')
#         #print(f'Query index: {query_index}')
#         if reference_index > index:
#             return query_index-reference_index+index
#     return -1
#
# def get_aligned_position_from_row(df_row, index):
#     cigar_string = df_row['cigar_string']
#     first_base_position = df_row['first_base_position']
#     return get_aligned_position(index, cigar_string, first_base_position)
#
#     a = np.array(re.findall(r'[0-9]*(?=[MIS=X])', cigar_string)).astype(int).cumsum()
#     b = np.array([first_base_position-1]+re.findall(r'[0-9]*(?=[MDN=X])', cigar_string)).astype(int).cumsum()
#     return aligned_sequence, aligned_quality


def extract_positions(series, indices):
    for i, index in enumerate(indices):
        if i == 0:
            combined = series.str.get(index)
        else:
            combined += series.str.get(index)
    return combined


def make_sequencing_dataset(file_path, index1_file_path=None, remove_duplicates=True, add_aligned_sequence=True,
                            extract_sequence_subset=False, chunksize=10000):

    file_path = Path(file_path)

    nc_file_path = file_path.with_name('sequencing_data.nc')
    if file_path.suffix == '.sam':
        parse_sam(file_path, remove_duplicates=remove_duplicates, add_aligned_sequence=add_aligned_sequence,
                  extract_sequence_subset=extract_sequence_subset, chunksize=chunksize, write_csv=False, write_nc=True,
                  write_filepath=nc_file_path)
    else:
        raise ValueError('Wrong file type')

    if index1_file_path is not None:
        index1_file_path = Path(index1_file_path)
        if index1_file_path.suffix == '.fastq':
            add_sequence_data_to_dataset(nc_file_path, index1_file_path, 'index1')
        else:
            raise ValueError('Wrong file type for index1')

    return nc_file_path



# TODO: Add to sam to nc file conversion
import re
import tqdm

def determine_read_end_position(cigar_string, reference_start_postion, reference_length):
    cigar_string_split = [(int(s[:-1]), s[-1]) for s in re.findall(r'[0-9]*[MIDNSHP=X]', cigar_string)]
    reference_index = reference_start_postion
    read_index = 0
    for length, code in cigar_string_split:
        if code in ['M',  'D', 'N', '=', 'X']:
            reference_index += length
        if code in ['M', 'I', 'S', '=', 'X']:
            read_index += length
        if reference_index >= reference_length:
            return read_index-(reference_index-reference_length)+1
    return reference_length

def determine_read_end_positions(sequencing_data, reference_length, chunk_size=10000):
    end_positions = np.zeros(len(sequencing_data)).astype('int16')
    for j in tqdm.tqdm(list(range(0,len(sequencing_data),chunk_size))):
        ds = sequencing_data.dataset.isel(sequence=slice(j, j+chunk_size))
        cigar_strings = ds.cigar_string.load()
        reference_start_positions = ds.position.load()
        for i, (cigar_string, reference_start_position) in enumerate(zip(cigar_strings, reference_start_positions)):
            end_positions[j+i] = determine_read_end_position(cigar_string.item(), reference_start_position.item(), reference_length)
    return end_positions


def sequence_correspondence(sequences, references, minimum_match=None):
    reference_names = np.array(list(references.keys()))
    reference_barcodes = np.array(list(references.values()))
    length = len(sequences[0])
    sequences_split = sequences.astype('S').view('S1').reshape(-1,length)
    reference_barcodes_split = reference_barcodes.astype('S').view('S1').reshape(-1,length)
    N_matched = (sequences_split[:, None, :] == reference_barcodes_split[None, :, :]).sum(axis=-1)
    correspondence = reference_names[N_matched.argmax(axis=1)]
    score = N_matched.max(axis=1)
    if minimum_match is not None:
        correspondence[score < minimum_match] = ''
    return correspondence, score

#
# def write_string_variable_to_nc_file(nc_file, name, array, indices=None):
#     from papylio.plugins.sequencing.sequencing_data import create_string_variable_in_nc_file
#
#     if indices is None:
#         indices = slice(None)
#
#     array = array.astype('S')
#     size = int(str(array.dtype)[2:])
#     create_string_variable_in_nc_file(nc_file, name, dimensions=('sequence',), size=size)
#     nc_file[name][indices] = array

if __name__ == '__main__':
    file_path = r'J:\Ivo\20211011 - Sequencer (MiSeq)\Analysis\sequencing_data_MapSeq.csv'
    seqdata = SequencingData(file_path=file_path)
    seqdata.plot_cluster_locations_per_tile(save_filepath=r'J:\Ivo\20211011 - Sequencer (MiSeq)\Analysis\Mapping_seqquences_per_tile.png')

    file_path = r'J:\Ivo\20211011 - Sequencer (MiSeq)\Analysis\sequencing_data_HJ_general.csv'
    seqdata_HJ = SequencingData(file_path=file_path)



    # New analysis
    sam_filepath = Path(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20211011 - Sequencer (MiSeq)\Analysis\Alignment.sam')
    # sequencing_data = read_sam(sam_filepath)

    extract_sequence_subset = [30, 31, 56, 57, 82, 83, 108, 109]
    parse_sam(sam_filepath, remove_duplicates=True, add_aligned_sequence=True,
              extract_sequence_subset=extract_sequence_subset,
              chunksize=10000, write_csv=False, write_nc=True)

    nc_filepath = sam_filepath.with_suffix('.nc')
    fastq_filepath = r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20211011 - Sequencer (MiSeq)\Analysis\Index1.fastq'
    add_sequence_data_to_dataset(nc_filepath, fastq_filepath, 'index1')
