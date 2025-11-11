import numpy as np
import xarray as xr
#
# stacking_energies = xr.DataArray(np.nan, dims=('reference', 'from_base', 'to_base'),
#                     coords={'reference': ['Protozanova2004', 'Punnoose2023'],
#                             'from_base': ['A','T','G','C'], 'to_base': ['A','T','G','C']})
#
#
# stacking_energies.loc[dict(reference='Protozanova2004')] = \
#     [[-1.11, -1.34, -1.06, -1.81],
#      [-0.19, -1.11, -0.55, -1.43],
#      [-1.43, -1.81, -1.44, -2.17],
#      [-0.55, -1.06, -0.91, -1.44]]
#
# stacking_energies.loc[dict(reference='Punnoose2023')] = \
#     [[-2.3, -1.5, -2.3, -1.9],
#      [-1.5, -0.8, -1.7, -0.5],
#      [-2.3, -1.7, -1.8, -2.1],
#      [-1.9, -0.5, -2.1, -0.6]]
#
# stacking_energies_flat = stacking_energies.stack(base_combination=('from_base', 'to_base'))
# stacking_energies_flat = stacking_energies_flat.reset_index('base_combination') \
#     .assign_coords(base_combination=('base_combination', [i+j for i,j in stacking_energies_flat.indexes['base_combination']]))


#
# def calculate_stacking_energy(sequence_subsets):
#
#     states = list(from_to_positions_per_state.keys())
#     energies = xr.DataArray(np.nan, dims=('sequence_subset', 'reference', 'state'),
#                             coords={'sequence_subset': sequence_subsets, 'state': states,
#                                     'reference': stacking_energies.reference})
#     for sequence_subset in sequence_subsets:
#         for state in (0,1):
#             from_tos = [(sequence_subset[i], sequence_subset[j]) for i, j in from_to_positions_per_state[state]]
#             energy = stacking_energies.stack(from_to=('from_base', 'to_base')).sel(from_to=from_tos).sum(dim='from_to')
#             energies.loc[dict(sequence_subset=sequence_subset, state=state)] = energy



def energy_dict_to_dataarray(energy_dict):
    return xr.DataArray(list(energy_dict.values()), dims=('base_combination',), coords={'base_combination': list(energy_dict.keys())})

stacking_energies_flat = xr.Dataset()

energy_dict = {'AA': -1.11, 'AT': -1.34, 'AG': -1.06, 'AC': -1.81,
               'TA': -0.19, 'TT': -1.11, 'TG': -0.55, 'TC': -1.43,
               'GA': -1.43, 'GT': -1.81, 'GG': -1.44, 'GC': -2.17,
               'CA': -0.55, 'CT': -1.06, 'CG': -0.91, 'CC': -1.44}
stacking_energies_flat['Protozanova2004'] = energy_dict_to_dataarray(energy_dict)

energy_dict = {'AA': -1.49, 'AT': -1.72, 'AG': -1.44, 'AC': -2.19,
               'TA': -0.57, 'TT': -1.49, 'TG': -0.93, 'TC': -1.81,
               'GA': -1.81, 'GT': -2.19, 'GG': -1.82, 'GC': -2.55,
               'CA': -0.93, 'CT': -1.44, 'CG': -1.29, 'CC': -1.82}
stacking_energies_flat['Krueger2006'] = energy_dict_to_dataarray(energy_dict)

energy_dict = {'AA': -1.36, 'AC': -2.03, 'AG': -1.60, 'AT': -2.35,
               'CA': -0.81, 'CC': -1.64, 'CG': -2.06, 'CT': -1.60,
               'GA': -1.39, 'GC': -3.42, 'GG': -1.64, 'GT': -2.03,
               'TA': -1.01, 'TC': -1.39, 'TG': -0.81, 'TT': -1.36}
stacking_energies_flat['Kilchherr2016'] = energy_dict_to_dataarray(energy_dict)


energy_dict = {'GA': -2.3, 'AG': -2.3, 'AA': -2.3, 'GG': -1.8, 'GC': -2.1, 'CG': -2.1, 'AC': -1.9, 'CA': -1.9,
               'GT': -1.7, 'TG': -1.7, 'AT': -1.5, 'TA': -1.5, 'TT': -0.8, 'CC': -0.6, 'CT': -0.5, 'TC': -0.5}
stacking_energies_flat['Punnoose2023'] = energy_dict_to_dataarray(energy_dict)

energy_dict = {'AA': -2.38, 'TT': -1.52, 'AT': -2.17, 'TA': -1.52, 'GG': -1.62, 'CC': -1.89, 'GC': -2.99, 'CG': -1.42,
                'TG': -1.42, 'CA': -1.45, 'AG': -1.74, 'CT': -1.19, 'TC': -2.19, 'GA': -2.16, 'GT': -2.06, 'AC': -3.57}
stacking_energies_flat['Banerjee2022'] = energy_dict_to_dataarray(energy_dict)

energy_dict = {'AA': -2.30, 'TT': -1.31, 'AT': -1.96, 'TA': -1.37, 'GG': -1.62, 'CC': -1.60, 'GC': -2.61, 'CG': -1.24,
                'TG': -1.26, 'CA': -1.29, 'AG': -1.71, 'CT': -0.95, 'TC': -1.92, 'GA': -2.00, 'GT': -1.80, 'AC': -3.22}
stacking_energies_flat['Banerjee2023'] = energy_dict_to_dataarray(energy_dict)

stacking_energies_flat = stacking_energies_flat.to_array('reference')

sequence_subsets = np.array(['TTAGCCGA', 'AATCGGCT', 'GGCGCCGC'])

from_to_positions_per_state = {0: ((0, 1), (2, 7), (6, 3), (4, 5)), 1: ((2, 3), (4, 1), (0, 5), (6, 7))}
states = [0,1]


def migrate_junction(sequence_subset, penultimate_bases='CCGCGGCG', step=-1):
    if step == -1: #Horizontal
        return penultimate_bases[0] + sequence_subset[0] + sequence_subset[3] + penultimate_bases[3] + \
               penultimate_bases[4] + sequence_subset[4] + sequence_subset[7] + penultimate_bases[7]
    elif step == 0:
        return sequence_subset
    elif step == 1: # Vertical
        return sequence_subset[1] + penultimate_bases[1] + penultimate_bases[2] + sequence_subset[2] + \
               sequence_subset[5] + penultimate_bases[5] + penultimate_bases[6] + sequence_subset[6]

base_pairs = ['AT','TA','CG','GC']
def check_basepairing(sequence_subset):
    if (sequence_subset[1:3] in base_pairs) and (sequence_subset[3:5] in base_pairs) and \
        (sequence_subset[5:7] in base_pairs) and ((sequence_subset[7] + sequence_subset[0]) in base_pairs):
        return True
    else:
        return False


def migration_options(sequence_subset, penultimate_bases='CCGCGGCG', return_all=False):
    migration_option_list = []
    for step in [-1, 0, 1]:
       sequence_subset_migrated = migrate_junction(sequence_subset, penultimate_bases=penultimate_bases, step=step)
       if check_basepairing(sequence_subset_migrated):
           migration_option_list.append(sequence_subset_migrated)
       elif return_all:
           migration_option_list.append(None)
    return migration_option_list

def migration_sequence_subsets(sequence_subsets, penultimate_bases='CCGCGGCGC'):
    migration_sequences = np.array([migration_options(sequence_subset, penultimate_bases=penultimate_bases, return_all=True)
                                    for sequence_subset in sequence_subsets])
    migration_sequences[migration_sequences == None] = ''
    migration_sequences = migration_sequences.astype('U')
    return migration_sequences

import itertools
bases = 'ATCG'
base_combinations = [''.join(i) for i in itertools.product(bases, repeat=2)]

def base_combination_per_position(sequence_subsets):
    base_combinations_per_position = xr.DataArray('', dims=('sequence_subset', 'position'),
                                          coords={'sequence_subset': sequence_subsets, 'position': np.arange(4)})

    for sequence_subset in sequence_subsets:
        for state in states:
            base_combinations_per_position.loc[dict(sequence_subset=sequence_subset, state=state)] = \
                [sequence_subset[i] + sequence_subset[j] for i, j in from_to_positions_per_state[state]]

def base_combination_count(sequence_subsets, penultimate_bases=None):
    base_combination_count = xr.DataArray(0, dims=('sequence_subset', 'base_combination', 'state','stack_position'),
                                          coords={'sequence_subset': sequence_subsets,
                                                  'base_combination': base_combinations,
                                                  'state': np.arange(2),
                                                  'stack_position': np.arange(4)})

    for sequence_subset in sequence_subsets:
        for state in states:
            for stack_position, positions in enumerate(from_to_positions_per_state[state]):
                # from_tos = [sequence_subset[i]+sequence_subset[j] for i, j in from_to_positions_per_state[state]]
                # base_combination, count = np.unique(from_tos, return_counts=True)
                # if state==0:
                #     count = -count
                base_combination = sequence_subset[positions[0]] + sequence_subset[positions[1]]
                base_combination_count.loc[dict(sequence_subset=sequence_subset,
                                                base_combination=base_combination,
                                                state=state, stack_position=stack_position)] += 1

                if penultimate_bases is not None:
                    penultimate_base_combination_0 = penultimate_bases[positions[0]] + sequence_subset[positions[0]]
                    penultimate_base_combination_1 = sequence_subset[positions[1]] + penultimate_bases[positions[1]]
                    base_combination_count.loc[dict(sequence_subset=sequence_subset,
                                                    base_combination=penultimate_base_combination_0,
                                                    state=state, stack_position=stack_position)] += 1
                    base_combination_count.loc[dict(sequence_subset=sequence_subset,
                                                    base_combination=penultimate_base_combination_1,
                                                    state=state, stack_position=stack_position)] += 1
    return base_combination_count



def calculate_total_stacking_energy(sequence_subsets, penultimate_bases=None):
    return (base_combination_count(sequence_subsets, penultimate_bases=penultimate_bases)
            .sum('stack_position').diff('state').squeeze(drop=True) * stacking_energies_flat).sum('base_combination')


# sequence_subsets = np.array(['TTAGCCGA', 'AATCGGCT', 'GGCGCCGC'])
# penultimate_bases = 'CCGCGGCGC'


def stacking_energies(sequence_subsets,penultimate_bases = 'CCGCGGCG'):
    return (base_combination_count(sequence_subsets, penultimate_bases=penultimate_bases)*stacking_energies_flat)

def total_stacking_energies(sequence_subsets, penultimate_bases = 'CCGCGGCG'):
    return stacking_energies(sequence_subsets, penultimate_bases).sum('base_combination').sum('stack_position')

def inner_stacking_energies(sequence_subsets, penultimate_bases = 'CCGCGGCG'):
    return stacking_energies(sequence_subsets, penultimate_bases).sel(stack_position=[1,2]).sum('base_combination').sum('stack_position')

def inner_stacking_energies_minimum(sequence_subsets, penultimate_bases='CCGCGGCG'):
    return stacking_energies(sequence_subsets, penultimate_bases).sel(stack_position=[1, 2]).sum('base_combination').min('stack_position')

    #
    #
    #
    # sequence_subset_split = np.array(list(sequence_subset))
    # if state == 0:
    #     sequence_subset_split = np.array(list(sequence_subset))
    #     from_bases = sequence_subset_split[[0,2,6,4]]
    #     to_bases = sequence_subset_split[[1,7,3,5]]
    #
    #     from_tos = [(sequence_subset[i], sequence_subset[j]) for i,j in ((0,1),(2,7),(6,3),(4,5))]
    #
    #
    #
    # if state == 1:
    #     from_bases = sequence_subset_split[[2,4,0,6]]
    #     to_bases = sequence_subset_split[[3,1,5,7]]
    # # energies_from_ref = stacking_energies.sel(reference=)
    # # np.array([stacking_energies.sel(from_base=from_base, to_base=to_base) for from_base, to_base in zip(from_bases, to_bases)])
    # #
    # from_tos = np.stack([from_bases, to_bases]).T
    # energy_for_subset = stacking_energies.stack(from_to=('from_base', 'to_base')).sel(from_to=[tuple(ft) for ft in from_tos])
    # return energy_for_subset.sum(dim='from_to')
