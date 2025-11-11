import re
import numpy as np
import xarray as xr


def datatype_conversion(dtype=list):
    def datatype_conversion_decorator(func):
        def wrapper(sequences, *args, **kwargs):
            if isinstance(sequences, xr.DataArray):
                coords = sequences.coords
                sequences = dtype(sequences.values)
                return xr.DataArray(func(sequences, *args, **kwargs), coords=coords)
            elif isinstance(sequences, np.ndarray):
                sequences = dtype(sequences)
                return np.array(func(sequences, *args, **kwargs))
            else:
                sequences = dtype(sequences)
                return list(func(sequences, *args, **kwargs))
        return wrapper
    return datatype_conversion_decorator


@datatype_conversion()
def fraction_GC(sequences):
   return [len(re.findall('G|C', sequence))/len(sequence) for sequence in sequences]

@datatype_conversion()
def number_of_neighboring_bases(sequences, base_type):
    if base_type == 'purine':
        search_string = 'AG|GA|AA|GG'
    elif base_type == 'pyrimidine':
        search_string = 'TC|CT|TT|CC'
    elif len(base_type) == 1:
        search_string = base_type + base_type
    return [len(re.findall(search_string, sequence + sequence[0])) for sequence in sequences]

@datatype_conversion()
def number_of_bases(sequences, base_type, positions='all'):
    if not positions=='all':
        sequences_new = []
        sequences_new.append([])
    if base_type == 'purine':
        search_string = 'AG|GA|AA|GG'
    elif base_type == 'pyrimidine':
        search_string = 'TC|CT|TT|CC'
    elif len(base_type) == 1:
        search_string = base_type + base_type
    return [len(re.findall(search_string, sequence + sequence[0])) for sequence in sequences]

def get_bases(bases):
    if bases == 'all':
        bases = 'ATCG' # np.array(['A', 'T', 'C', 'G'])
    elif bases == 'purines':
        bases = 'AG' #np.array(['A', 'G'])
    elif bases == 'pyrimidines':
        bases = 'TC' #np.array(['T', 'C'])

    # if return_type == str:
    #     ''.join(list(bases))
    # else:
    return np.array(list(bases))

import itertools
def get_base_combinations(base_combinations='basepaired'):
    if base_combinations in ['all','purines','pyrimidines']:
        base_combinations = np.array(list(itertools.product(get_bases(base_combinations),repeat=2)))
    elif base_combinations == 'basepaired':
        base_combinations = sequences_to_sequence_array(np.array(['AT', 'TA', 'GC', 'CG']))
    elif isinstance(base_combinations, str):
        base_combinations = sequences_to_sequence_array([base_combinations])
    else:
        base_combinations = sequences_to_sequence_array(base_combinations)
    return base_combinations

def sequences_to_sequence_array(sequences):
    return np.atleast_1d(sequences).astype('U').view('U1').reshape(-1, len(sequences[0]))

@datatype_conversion(np.array)
def base_count(sequences, positions='all', bases='all'):
    # TODO: make it possible to
    if positions == 'all':
        positions = np.arange(len(sequences[0]))
    else:
        positions = np.array(positions)

    sequence_array = sequences_to_sequence_array(sequences)

    return (sequence_array[:, positions, None] == bases[None, None, :]).any(axis=2).sum(axis=1)


def base_combination_presence(sequences, position_0, position_1, base_combinations):
    base_combinations = get_base_combinations(base_combinations)
    sequence_array = sequences_to_sequence_array(sequences)
    return (sequence_array[:, [position_0, position_1], None] == base_combinations.T[None, :, :]).all(axis=1).any(axis=1)

    # base_combinations_per_pair = np.stack(
    #     [get_base_combinations(base_combinations).T for base_combinations in base_combinations_per_pair])
@datatype_conversion(np.array)
def base_combination_count(sequences, position_pairs, base_combinations_per_pair):
    if isinstance(base_combinations_per_pair, str):
        base_combinations_per_pair = [base_combinations_per_pair] * len(position_pairs)
    return np.array([base_combination_presence(sequences, *position_pair, base_combinations) \
        for position_pair, base_combinations in zip(position_pairs, base_combinations_per_pair)]).sum(axis=0)

    # base_combinations_per_pair = np.stack([get_base_combinations(base_combinations).T for base_combinations in base_combinations_per_pair])
    # sequence_array = sequences_to_sequence_array(sequences)
    # return (sequence_array[:, position_pairs, None] == base_combinations_per_pair[None, :, :, :]).all(axis=2).sum(axis=2)
    #
    #
    #     base_combination_count = \
    #         np.array([(base_count(sequences, positions=position_pair, bases=bases,
    #                               count_basepairs=count_basepairs) == 2).astype(int)
    #                   for position_pair in position_pairs])
    #     return base_combination_count.sum(axis=0)


# @datatype_conversion(np.array)
# def base_combination_count(sequences, position_pairs, bases='purines'):
#     base_combination_count = \
#         np.array([(base_count(sequences, positions=position_pair, bases=bases, count_basepairs=count_basepairs)==2).astype(int)
#                   for position_pair in position_pairs])
#     return base_combination_count.sum(axis=0)
