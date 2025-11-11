# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:59:54 2018

@author: Ivo Severins
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import re
import gc
import copy
from tabulate import tabulate
from pathlib import Path
import pandas as pd
try:
    import logomaker
except ModuleNotFoundError:
    pass

import matplotlib.path as pth

class FastqData:
    def __init__(self, path):
        self.path = Path(path)
        self.write_path = self.path.parent
        self.adapter_sequences = ['AATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT', 'ATCTCGTATGCCGTCTTCTGCTTG']
        self._text_file = None
        self._tiles = []
        self.is_reverse_complement = False

        file = open(path, 'r')

        lineCount = 0
        recordCount = 0
        
        instrument = list()
        run = list()
        flowcell = list()
        lane = list()
        tile = list()
        x = list()
        y = list()
        sample = list()

        sequence = list()
        quality = list()
        
        
        for line in file:
            if lineCount % 4 == 0: 
                splitHeader = re.split('[@: -]',line.strip())
                
                instrument.append(splitHeader[1])
                run.append(splitHeader[2])
                flowcell.append(splitHeader[4])
                lane.append(int(splitHeader[5]))
                tile.append(int(splitHeader[6]))
                x.append(int(splitHeader[7]))
                y.append(int(splitHeader[8]))
                sample.append(int(splitHeader[12]))
                
            if lineCount % 4 == 1: 
                sequence.append(line.strip())
            if lineCount % 4 == 3: 
                quality.append(line.strip())
                       
                recordCount +=1
        
            lineCount +=1
        
        
        file.close()

        self.name = np.array(['']*len(sequence))
        self.instrument = np.array(instrument)
        self.run = np.array(run)
        self.flowcell = np.array(flowcell)
        self.lane = np.array(lane)
        self.tile = np.array(tile)
        self.x = np.array(x)
        self.y = np.array(y)
        self.sample = np.array(sample)
        self.sequence = np.array(sequence, dtype = bytes).view('S1').reshape((len(sequence),-1))
        self.quality = np.array(quality, dtype = bytes).view('S1').reshape((len(quality),-1))

        self.selection

        self.write_to_text_file('Total number of sequences: ' + str(self.sequence.shape[0]) + '\n\n')

    @property
    def tile_numbers(self):
        return np.unique(self.tile)

    @property
    def tiles(self):
        if not self._tiles:
            self._tiles = [self.get_tile_object(tile_number) for tile_number in self.tile_numbers]
        return self._tiles

    @property
    def coordinates(self):
        return np.vstack([self.x, self.y]).T

    @property
    def text_file(self):
        if not self._text_file:
            self._text_file = self.write_path.joinpath('Output.txt')
            with self._text_file.open('w') as f:
                f.write('Analysis\n\n')
        return self._text_file

    def write_to_text_file(self, input_text):
        with self.text_file.open('a') as f:
            f.write(input_text + '\n')

    def export_fastq(self, filepath = None):
        if not filepath: filepath = self.write_path

        with filepath.with_suffix('.fastq').open('w') as f:
            for i in np.arange(len(self)):
                f.write(f"@{self.instrument[i]}:{self.run[i]}:000000000-{self.flowcell[i]}:{self.lane[i]}:"
                        f"{self.tile[i]}:{self.x[i]}:{self.y[i]} 1:N:0:{self.sample[i]}\n"
                        f"{self.sequence[i].tostring().decode('utf-8')}\n"
                        f"+\n"
                        f"{self.quality[i].tostring().decode('utf-8')}\n"
                        )

    def __len__(self):
        return self.sequence.shape[0]

    def __getitem__(self, item):
        new = copy.copy(self)
        new.name = self.name[item]
        new.instrument = self.instrument[item]
        new.run = self.run[item]
        new.flowcell = self.flowcell[item]
        new.lane = self.lane[item]
        new.tile = self.tile[item]
        new.x = self.x[item]
        new.y = self.y[item]
        new.sample = self.sample[item]
        new.sequence = self.sequence[item, :]
        new.quality = self.quality[item, :]

        new._tiles = []
        return new

    def __add__(self, other):
        new = copy.copy(self)
        new.name = np.append(new.name, other.name)
        new.instrument = np.append(new.instrument, other.instrument)
        new.run = np.append(new.run, other.run)
        new.flowcell = np.append(new.flowcell, other.flowcell)
        new.lane = np.append(new.lane, other.lane)
        new.tile = np.append(new.tile, other.tile)
        new.x = np.append(new.x, other.x)
        new.y = np.append(new.y, other.y)
        new.sample = np.append(new.sample, other.sample)
        new.sequence = np.vstack([new.sequence, other.sequence])
        new.quality = np.vstack([new.quality, other.quality])

        new._tiles = []
        return new

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def select(self, indices, copyData=False):
        if copyData:
            selection = copy.deepcopy(self)
        else:
            selection = self

        selection.name = self.name[indices]
        selection.instrument = self.instrument[indices]
        selection.run = self.run[indices]
        selection.flowcell = self.flowcell[indices]
        selection.lane = self.lane[indices]
        selection.tile = self.tile[indices]
        selection.x = self.x[indices]
        selection.y = self.y[indices]
        selection.sample = self.sample[indices]
        selection.sequence = self.sequence[indices,:]
        selection.quality = self.quality[indices,:]

        selection._tiles = []
        
        if copyData:
            return selection

    def selection(self, save=False, **kwargs):
        selection = np.zeros((self.sequence.shape[0], len(kwargs)), dtype=bool)
        for i, (key, value) in enumerate(kwargs.items()):
            if key == 'sequence':
                selection[:, i] = np.all(self.sequence[:, 0:len(value)] == np.array(list(value), dtype = bytes), axis = 1)
            elif key in ['x', 'y']:
                selection[:, i] = np.all(np.vstack([getattr(self, key) > np.min(value), getattr(self, key) < np.max(value)]), axis=0)
            elif key == 'coordinates_within_vertices':
                selection[:, i] = pth.Path(value).contains_points(self.coordinates)
            elif key == 'boolean_selection':
                selection[:, i] = value
            elif key == 'in_name':
                if not isinstance(value, list):
                    value = [value]
                selection[:, i] = [any(test_string in name for test_string in value)
#                                    if name is not None else False
                                   for name in self.name]
            else:
                selection[:, i] = getattr(self, key) == value

        return np.all(selection, axis=1)

    def get_selection(self, **kwargs):
        selection = self.selection(**kwargs)
        return self[selection]

    def reverse_complement(self):
        condlist = [self.sequence == b'A', self.sequence == b'T',
                    self.sequence == b'C', self.sequence == b'G']
        choicelist = [b'T', b'A', b'G', b'C']
        self.sequence = np.fliplr(np.select(condlist, choicelist))
        self.quality = np.fliplr(self.quality)
        self.is_reverse_complement = not self.is_reverse_complement

    def base_count(self):
        base_count = pd.DataFrame()
        for b in ['A', 'C', 'T', 'G']:
            base_count[b] = np.sum(self.sequence == b.encode(), axis=0)
        return base_count

    def logo_plot(self, start=None, end=None, row_length=None, figure=None, save=False, title='', **kwargs):
        base_count = self.base_count()[slice(start, end)]
        if not row_length:
            row_length=len(base_count)
        n_rows = len(base_count) // row_length
        if not figure:
            figure = plt.figure(figsize=(row_length/5,2*n_rows))

        figure.subplots(n_rows,1, sharey=True)
        axes = figure.axes

        for i in np.arange(n_rows):
            logo_object = logomaker.Logo(base_count[(i*row_length):((i+1)*row_length)], ax=axes[i], **kwargs)
            axes[i].set_ylabel('Count')

        axes[0].set_title(title)
        axes[-1].set_xlabel('Position')


        figure.tight_layout(pad=1)
        if save:
            title = title.replace('>','gt').replace('<','st')
            figure.savefig(f'{title}_seqlogo.png')

        return logo_object

    def sequence_density(self, expected_seq=None, start=None, end=None, row_length=None, figure=None, save=False, title=''):
        # TODO: Replace with function in plotting_sequences.py

        sequence = self.sequence[~np.any(self.sequence == b'N', axis=1)]
        # sequence_int = sequence.view('uint8')
        # sequence_df = pd.DataFrame(sequence_int).iloc[:, slice(start, end)]
        sequence_df = pd.DataFrame(sequence)
        expected_seq = expected_seq[slice(start, end)]

        total_length = sequence_df.shape[1]

        if not row_length:
            row_length = total_length

        n_rows = total_length // row_length
        if not figure:
            figure = plt.figure(figsize=(row_length/5,2*n_rows))

        figure.subplots(n_rows, 1, sharey=False)
        axes = figure.axes

        bases = [b'A', b'T', b'C',b'G']
        index = [(b1,b2) for b1 in bases for b2 in bases]
        # out = pd.DataFrame(index=index)
        for r in np.arange(n_rows):
            for i in np.arange((r*row_length),((r+1)*row_length)-1):
                print(i)
                # out[i] = (test.iloc[:,i:(i+2)].value_counts(normalize=True))
                base_transition_count = (sequence_df.iloc[:, i:(i + 2)].value_counts(normalize=True)).reindex(index)
                ys = base_transition_count.index.to_list()
                zs = base_transition_count.values
                x = (i, i + 1)
                for y, z in zip(ys, zs):
                    axes[r].plot(x, y, lw=z * 10, c='b', solid_capstyle='round')
                # ax.plot((i, i + 1), [0, 0])
            # expected_seq=sequences_per_read['Read1']['HJ1']
            if expected_seq:
                # axis.set_xticklabels(list(expected_seq))
                # axis.set_xticks(np.arange(len(expected_seq)))
                xs = np.where(np.array(list(expected_seq)) != 'N')[0]
                x0 = 0
                for s in re.split('N', expected_seq):
                    if s:
                        x = xs[x0:(x0 + len(s))]
                        y = np.array(list(s), dtype='S1')#.view('uint8')
                        axes[r].plot(x, y, c='r')
                        x0 += len(s)
            axes[r].set_xlim(r*row_length, (r+1)*row_length-1)
            axes[r].set_ylabel('Base')
            # axes[r].xaxis.set_minor_locator(plt.MultipleLocator(1))

        axes[0].set_title(title)
        axes[-1].set_xlabel('Position')

        # for axis in axes:
        #     bases = list('ATGC')
        #     axis.set_yticks(np.array(bases, dtype='S1').view('uint8'))
        #     axis.set_yticklabels(bases)

        figure.tight_layout(pad=1)
        if save:
            title = title.replace('>','gt').replace('<','st')
            figure.savefig(f'{title}_seqdensity.png')

    def number_of_matches(self, sequence):
        # sequence must be a string
        # return np.sum(self.sequence[:,0:len(sequence)]==np.array(list(sequence), dtype = bytes),1)

        sequence_bytes = np.array(list(sequence), dtype=bytes)
        indices_not_N = np.where(sequence_bytes != np.array('N', dtype=bytes))[0]
        number_of_Ns = len(sequence)-len(indices_not_N)

        return np.sum(self.sequence[:, indices_not_N] == sequence_bytes[indices_not_N], 1) + number_of_Ns

    def number_of_matches_with_sequence_dict(self, sequence_dict):
        number_of_matches = [pd.Series(self.number_of_matches(sequence), name=sequence_name) for
                             sequence_name, sequence in sequence_dict.items()]
        number_of_matches = (pd.concat(number_of_matches, keys=sequence_dict.keys(), names='Sequence', axis=1)
                             .rename_axis('Sequence index').astype(pd.Int64Dtype()))
        return number_of_matches

    def number_of_mismatches_with_sequence_dict(self, sequence_dict):
        number_of_matches = self.number_of_matches_with_sequence_dict(sequence_dict)
        number_of_mismatches = number_of_matches.copy()
        for sequence in number_of_matches.columns:
            number_of_mismatches[sequence] = len(sequence_dict[sequence]) - number_of_matches[sequence]
        return number_of_mismatches

    def matches_per_tile(self, sequence):
        if len(sequence) < 5: return

        header = [''] + list(self.tile_numbers)
        emptyRow = ['' for i in np.arange(9)]

        number_of_matches = self.number_of_matches(sequence)

        allClusters = ['All clusters'] + [np.sum(self.selection(tile = tile)) for tile in self.tile_numbers]
        fullMatchCount = ['Full matches'] + [np.sum(self.tile[number_of_matches == (len(sequence))] == tile) for tile in self.tile_numbers]
        oneMisMatch = ['1 mismatch'] + [np.sum(self.tile[number_of_matches == (len(sequence) - 1)] == tile) for tile in self.tile_numbers]
        twoMisMatches = ['2 mismatches'] + [np.sum(self.tile[number_of_matches == (len(sequence) - 2)] == tile) for tile in self.tile_numbers]
        threeMisMatches = ['3 mismatches'] + [np.sum(self.tile[number_of_matches == (len(sequence) - 3)] == tile) for tile in self.tile_numbers]
        fourMisMatches = ['4 mismatches'] + [np.sum(self.tile[number_of_matches == (len(sequence) - 4)] == tile) for tile in self.tile_numbers]
        fiveMisMatches = ['5 mismatches'] + [np.sum(self.tile[number_of_matches == (len(sequence) - 5)] == tile) for tile in self.tile_numbers]

        # lessThan3mismatches = ['<=2 mismatches'] + [np.sum(data.tile[Nmatch > 47] == tile) for tile in self.tile_numbers]]

        table = tabulate([allClusters,
                          emptyRow,
                          fullMatchCount,
                          oneMisMatch,
                          twoMisMatches,
                          threeMisMatches,
                          fourMisMatches,
                          fiveMisMatches,
                          emptyRow
                          ], header)

        print(table)
        self.write_to_text_file('Matches per tile with sequence: ' + sequence + '\n\n')
        self.write_to_text_file(table)

    def classify(self, sequences_dict, criteria_dict):
        df = self.number_of_mismatches_with_sequence_dict(sequences_dict)

        selection = pd.DataFrame(index=df.index)
        for sequence_name, criterion in criteria_dict.items():
            selection.loc[df.query(criterion, engine='python').index, sequence_name] = True

            self.name = selection.apply(lambda x: ', '.join(x.index[x == True]), axis=1).to_numpy()

    def get_tile_object(self, tile):
        x = self.x[self.selection(tile=tile)]
        y = self.y[self.selection(tile=tile)]
        return Tile(tile, np.transpose([x,y]))
        # tile = Tile('2102', np.loadtxt(path.joinpath('2102.loc'))) If we later want to get it from the file

    def export_positions_per_tile(self):
        # Export clusterpositions
        for tile in self.tile_numbers:
            x = self.x[self.selection(tile=tile)]
            y = self.y[self.selection(tile=tile)]
            np.savetxt(self.write_path.joinpath(str(tile)+'.loc'), np.transpose([x,y]), fmt='%u', delimiter='\t')

    def show_tiles(self):
        # Figure with matching cluster positions
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex='all', sharey='all')
        for i, tile in enumerate(self.tile_numbers):
            ax = axes.flatten()[i]
            x = self.x[self.selection(tile = tile)]
            y = self.y[self.selection(tile = tile)]
            ax.scatter(x, y, c='k', marker='.')
            ax.set_title('Tile ' + str(tile))
            ax.set_aspect('equal')

            ax.set_xlim([0, 31000])
            ax.set_ylim([0, 31000])

            if i in [4, 5, 6, 7]: ax.set_xlabel('x (FASTQ)')
            if i in [0, 4]: ax.set_ylabel('y (FASTQ)')

        fig.savefig(self.write_path.joinpath('clusterPositionsWithMatchingSequence.pdf'), bbox_inches='tight')
        fig.savefig(self.write_path.joinpath('clusterPositionsWithMatchingSequence.png'), bbox_inches='tight')


class Tile:
    def __init__(self, number, coordinates):
        self.name = str(number)
        self.number = number
        self.coordinates = coordinates

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')

#https://stackoverflow.com/questions/9476797/how-do-i-create-character-arrays-in-numpy

#dataDictionary = {
#		'instrument': instrument,
#		'run': run,
#		'flowcell': flowcell,
#		'lane': lane,
#		'tile': tile,
#		'x': x,
#		'y': y,
#		'sample': sample,
#		'sequence': sequence,
#		'quality': quality
#		}
#
#
#df = pd.DataFrame(columns=['instrument','run','flowcell','lane','tile','x','y','sample','sequence','quality'])
#df = pd.DataFrame(dataDictionary)

# A way to get out specific sequences
#regex = re.compile('(\w){0}AA(\w){49}')
#len(list(filter(regex.match, sequence)))


#
#
#[df['sequence'][:][i] == df['sequence'][1][i] for i in range(len(df['sequence'][1]))]
#
#gc.collect()
#
#
#seqList = list(map(list, sequence))
#seqArray = np.array(list(map(np.array,seqList)))
#
#compSeq = np.array(list('ACTGTTTTTTTTTTTTTTTTACTACCTCTTTTTTTTTTTTTTT'))
#df['mmCountCompSeq'] = np.sum(seqArray[:,0:43]==compSeq, axis=1)
#df['firstMmPosition'] = np.logical_not((seqArray[:,0:43]==compSeq)).argmax(axis=1)
#
#
#df[df['mmCountCompSeq']==42]['firstMmPosition'].plot.hist(bins = 43)


# =============================================================================
# def matchCount(row):
# 	return sum([x==y for x,y in zip(row['sequence'],'CCTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCTTTTTC')])
# 
# 
# df.apply(lambda row: matchCount(row), axis=1)
# =============================================================================

# =============================================================================
# 
# 
# def matchCount(row):
#	return sum([x==y for x,y in zip(row['sequence'],'CCTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCTTTTTC')])
# test = list()
# for seq in sequence:
# 	test.append(sum([x==y for x,y in zip(seq,'GATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCTTTTTC')]))
# 	
# 	
# 		
# test = list()
# for seq in sequence:
# 	test.append([ord(char) for char in seq])
# 	
# a = np.array(test)
# 
# b = np.sum(a[1,:]==a,axis=1)
# =============================================================================

# =============================================================================
# df.sequence.str[1:3]
# 
# 
# 
# a = np.array([list(x) for x in df.sequence[1:3].str[1:40]])
# np.sum(df.sequence.str[1:3]==['T','T'],axis=1)
# =============================================================================





# =============================================================================
# 
# from Bio import SeqIO
# #record_dict = SeqIO.to_dict(SeqIO.parse(r"C:\Users\Ivo Severins\Desktop\Sequencing data\20180705\Undetermined_S0_L001_I1_001.fastq","fastq"))
# 

# 
# 
# 
# records = list(SeqIO.parse(path+file, "fastq"))
# 
# x = [o.id[33:38] for o in records]
# 
# 
# x = [re.search('(?=:)...',o.id) for o in records]
# 
# re.split(
# 
# 
# test = [o.letter_annotations['phred_quality'] for o in records]
# 
# 
# 
# 
# 
# 
# file = open(path+file, 'r') 
# print file.readline(): 
# =============================================================================






