import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_sequence_density(sequences, expected_seq=None, start=None, end=None, is_circular=False,
                          row_length=None, figure=None, save=False, title=''):
    sequences = sequences.view('U1').reshape(-1, len(sequences[0]))
    if is_circular:
        sequences = np.hstack([sequences, sequences[:,0:1]])
    sequence = sequences[~np.any(sequences == 'N', axis=1)]
    # sequence_int = sequence.view('uint8')
    # sequence_df = pd.DataFrame(sequence_int).iloc[:, slice(start, end)]
    sequence_df = pd.DataFrame(sequence)
    if expected_seq is not None:
        expected_seq = expected_seq[slice(start, end)]

    total_length = sequence_df.shape[1]

    if not row_length:
        row_length = total_length

    n_rows = total_length // row_length
    if figure is None:
        figure = plt.figure(figsize=(np.max([row_length / 4, 6]), 3 * n_rows), layout='constrained')

    figure.subplots(n_rows, 1, sharey=False)
    axes = figure.axes

    bases = ['A', 'T', 'C', 'G']
    index = [(b1, b2) for b1 in bases for b2 in bases]
    # out = pd.DataFrame(index=index)
    for r in np.arange(n_rows):
        for i in np.arange((r * row_length), ((r + 1) * row_length) - 1):
            # print(i)
            # out[i] = (test.iloc[:,i:(i+2)].value_counts(normalize=True))
            base_transition_count = (sequence_df.iloc[:, i:(i + 2)].value_counts(normalize=True)).reindex(index)
            ys = base_transition_count.index.to_list()
            zs = base_transition_count.values
            x = (i, i + 1)
            for y, z in zip(ys, zs):
                axes[r].plot(x, y, lw=z * 10, c='b', solid_capstyle='round')
            # ax.plot((i, i + 1), [0, 0])
        # expected_seq=sequences_per_read['Read1']['HJ1']
        if expected_seq is not None:
            # axis.set_xticklabels(list(expected_seq))
            # axis.set_xticks(np.arange(len(expected_seq)))
            xs = np.where(np.array(list(expected_seq)) != 'N')[0]
            x0 = 0
            for s in re.split('N', expected_seq):
                if s:
                    x = xs[x0:(x0 + len(s))]
                    y = np.array(list(s), dtype='S1')  # .view('uint8')
                    axes[r].plot(x, y, c='r')
                    x0 += len(s)
        axes[r].set_xlim(r * row_length, (r + 1) * row_length - 1)
        axes[r].set_ylabel('Base')
        # axes[r].xaxis.set_minor_locator(plt.MultipleLocator(1))

    axes[0].set_title(title)
    axes[-1].set_xlabel('Position')
    if is_circular:
        xticklabels = np.arange(sequences.shape[1])
        xticklabels[-1] = 0
        axes[-1].set_xticks(np.arange(len(xticklabels)))
        axes[-1].set_xticklabels(xticklabels)

    # for axis in axes:
    #     bases = list('ATGC')
    #     axis.set_yticks(np.array(bases, dtype='S1').view('uint8'))
    #     axis.set_yticklabels(bases)

    # figure.tight_layout(pad=1)
    if save:
        title = title.replace('>', 'gt').replace('<', 'st')
        figure.savefig(f'{title}_seqdensity.png')