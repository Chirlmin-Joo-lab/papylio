import itertools
import numpy as np

def all_sequence_subsets():
    bases = ['A', 'T', 'C', 'G']
    return np.array([''.join(bpc) for bpc in itertools.product(*[bases]*8)])

def all_basepaired_subsets():
    basepairs = ['AT', 'TA', 'CG', 'GC']
    return np.array([''.join(bpc)[1:] + ''.join(bpc)[0] for bpc in itertools.product(*[basepairs]*4)])

def basepaired_subsets():
    basepairs = ['AT', 'TA', 'CG', 'GC']
    basepaired_subsets = []
    for i0, bp0 in enumerate(basepairs):
        for i1, bp1 in enumerate(basepairs):
            for i2, bp2 in enumerate(basepairs):
                for i3, bp3 in enumerate(basepairs):
                    bpc = bp0 + bp1 + bp2 + bp3
                    basepaired_subsets.append(''.join(bpc)[1:] + ''.join(bpc)[0])
    return basepaired_subsets




def rotationally_symmetric_subsets(sequence_subset):
    return list(set([sequence_subset[i*2:] + sequence_subset[:i*2] for i in range(4)]))

def rotationally_symmetric_subset_categories(sequence_subsets):
    sequence_subsets = list(sequence_subsets)
    index = 0
    subset_category_dict = {}
    while len(sequence_subsets) > 0:
        sss = rotationally_symmetric_subsets(sequence_subsets[0])
        for ss in sss:
            subset_category_dict[ss] = index
            if ss in sequence_subsets:
                sequence_subsets.remove(ss)
        index += 1
    return subset_category_dict

def rotationally_symmetric_subset_groups():
    sequence_subsets = all_basepaired_subsets()
    sequence_subsets = list(sequence_subsets)
    index = 0
    subset_groups = []
    while len(sequence_subsets) > 0:
        sss = rotationally_symmetric_subsets(sequence_subsets[0])
        subset_groups.append(sss)
        for ss in sss:
            sequence_subsets.remove(ss)
    return subset_groups


def sequence_subset_with_comparable_structure(sequence_subset):
    # sequence_subset = sequence_subset[7] + sequence_subset[:7]
    sequence_subsets = []
    for i in range(4):
        sequence_subsets.append(sequence_subset[i*2:] + sequence_subset[:i*2])
    for ss in sequence_subsets.copy():
        change_base_within_basepair = str.maketrans('CG', 'GC')
        sequence_subsets.append(ss.translate(change_base_within_basepair))
    for ss in sequence_subsets.copy():
        change_base_within_basepair = str.maketrans('AT', 'TA')
        sequence_subsets.append(ss.translate(change_base_within_basepair))
    for ss in sequence_subsets.copy():
        switch_base_pairs = str.maketrans('ATCG', 'GCAT')
        sequence_subsets.append(ss.translate(switch_base_pairs))
    return set(sequence_subsets)


def sequence_subset_structure_category():
    subset_category_dict = {}
    sequence_subsets = basepaired_subsets()

    structure_index = 0
    while len(sequence_subsets) > 0:
        ss_with_structure = sequence_subset_with_comparable_structure(sequence_subsets[0])
        for ss in ss_with_structure:
            subset_category_dict[ss] = structure_index
            sequence_subsets.remove(ss)
        structure_index += 1
    return subset_category_dict


def unique_subset_structures():
    unique_subset_category_dict = {}
    for key, value in sequence_subset_structure_category().items():
        if value not in unique_subset_category_dict.values():
            unique_subset_category_dict[key] = value
    return unique_subset_category_dict
