import numpy as np

def generate_sequences(base_composition):
    """

    Parameters
    ----------
    base_composition : list of str
        A list with the length of the sequence, where each entry indicates the possible bases at that position.
        pairI where I is an integer indicates that the base at the position of pairI should basepair with position I.
        E.g. ['N', 'AC', 'GC', 'ACTG', 'pair1']

    Returns
    -------
    sequences : numpy.array
        Array of sequences
    """

    base_composition_2 = []
    for bases in base_composition:
        if bases == 'N':
            bases = 'ACTG'
        elif bases.startswith('pair'):
            bases = '-'

        base_composition_2.append(list(bases))

    sequences = np.array(np.meshgrid(*base_composition_2)).T.reshape(-1, len(base_composition_2))

    for i, bases in enumerate(base_composition):
        if bases.startswith('pair'):
            sequences[:,i] = convert_bases(sequences[:,int(bases[4:])], conversion='basepair')
    return sequences.view(f'U{len(base_composition_2)}').squeeze()

def convert_bases(bases, conversion='basepair', **conversion_dict):
    if conversion == 'basepair':
        conversion_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    return np.array([conversion_dict[base] for base in bases])


    # DNA_bases = ['A','T','C','G']
    # complementary_DNA_bases =
    #
    # condlist = [bases == base for base in
    #
    # np.select(condlist, choicelist, 55)