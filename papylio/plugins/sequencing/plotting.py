import math
import numpy as np
import xarray as xr
from pathlib2 import Path


###################################################
## To enable interactive plotting with PySide2 in PyCharm 2022.3
import PySide2
import sys
sys.modules['PyQt5'] = sys.modules['PySide2']
from matplotlib import use
use('Qt5Agg')
###################################################

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable




# ## Generated dataset
# sequence = 'AAAA'
# sequences = [sequence]
# bases = 'ACTG'
# for i in range(len(sequence)):
#     for base in bases:
#         new = sequence[0:i] + base + sequence[i + 1:len(sequence)]
#         if new != sequence:
#             sequences.append(new)
# da = xr.DataArray(np.random.rand(len(sequences)), coords={'sequence': sequences})


def single_mutations(sequence):
    sequences = [] # [sequence]
    bases = 'ACTG'
    for i in range(len(sequence)):
        for base in bases:
            new = sequence[0:i] + base + sequence[i + 1:len(sequence)]
            if new != sequence:
                sequences.append(new)
    return sequences

def plot_single_mutation_data(sequence, da):
    sequences = [sequence] + single_mutations(sequence)
    da = da.sel(sequence=sequences)

    fig, ax = plt.subplots(figsize=(1+len(sequence)*0.75, 3), tight_layout=True)
    width = 0.25
    position = np.arange(len(sequence))
    x = np.hstack([-1, np.vstack([position - width, position, position + width]).T.flatten()])
    ax.bar(x, da.values, width=width)

    ticklabels = ['WT'] + [label for p in position for label in list(bases.replace(sequence[p], ''))]
    ax.set_xticks(x, ticklabels)
    ax.set_ylabel(da.name)

    for p in position:
        # for xi in x[p*3+1:(p+1)*3+1]:
        # ax.annotate(seq[p], xy=(xi,-20), xytext=(p,-20), xycoords=('data', 'axes points'), textcoords=('data','offset points'),
        #             ha='center', va='bottom', arrowprops={'width': 0.3, 'headwidth': 3, 'headlength':2})
        # text = seq[p]
        for xi in [-1, 0, 1]:
            if xi == 0:
                text = sequence[p]
            else:
                text = ' '
            ax.annotate(text, xy=(p + xi * width * 2.3 / 3, -17), xytext=(p, -20), xycoords=('data', 'axes points'),
                        textcoords=('data', 'offset points'),
                        ha='center', va='bottom', arrowprops={'width': 0.2, 'headwidth': 3, 'headlength': 2})

def double_mutations(sequence, add_reference=False):
    ## Generated dataset
    # sequence = 'GGCGCCGC'
    sequences = []
    if add_reference:
        sequences.append(sequence)
    bases = 'ACTG'

    for i in range(len(sequence)):
        for j in range(len(sequence)):
            if i == j:
                for base in bases:
                    if base != sequence[i]:
                        print(i, j, base)
                        new = sequence[0:i] + base + sequence[(i+1):len(sequence)]
                        sequences.append(new)
            elif i < j:
                for base1 in bases:
                    for base2 in bases:
                        if base1 != sequence[i] and base2 != sequence[j]:
                            print(i, j, base1, base2)
                            new = sequence[0:i] + base1 + sequence[(i+1):j] + base2 + sequence[(j + 1):len(sequence)]
                            sequences.append(new)

    return sequences


def plot_double_mutations(sequence, da, da_annotation=None, save=False, save_path=None, **kwargs):
    # sequences = [sequence] + double_mutations(sequence)
    bases = 'ACTG'
    data = np.zeros((len(sequence)*3, len(sequence)*3))
    data_annotation = np.zeros((len(sequence)*3, len(sequence)*3))
    data[:] = np.nan
    data_annotation[:] = np.nan
    bases_axes = []

    for i in range(len(sequence)):
        for j in range(len(sequence)):
            if i == j:
                k = 0
                for base in bases:
                    if base != sequence[i]:
                        new = sequence[0:i] + base + sequence[(i+1):len(sequence)]
                        # sequences.append(new)
                        data[i*3+k,i*3+k] = da.sel(sequence=new).item()
                        if da_annotation is not None:
                            data_annotation[i * 3 + k, i * 3 + k] = da_annotation.sel(sequence=new).item()
                        bases_axes.append(base)
                        # print(i, j, k, base)
                        k += 1
            elif i < j:
                k1 = 0
                for base1 in bases:
                    if base1 != sequence[i]:
                        k2 = 0
                        for base2 in bases:
                             if base2 != sequence[j]:
                                new = sequence[0:i] + base1 + sequence[(i+1):j] + base2 + sequence[(j + 1):len(sequence)]
                                # sequences.append(new)
                                data[j*3+k2, i*3+k1] = da.sel(sequence=new).item()
                                if da_annotation is not None:
                                    data_annotation[j * 3 + k2, i * 3 + k1] = da_annotation.sel(sequence=new).item()
                                # print(i, j, k1, k2, base1, base2)
                                k2 += 1
                        k1 += 1

    figsize = 1+len(sequence)*0.75
    fig, ax = plt.subplots(figsize=(figsize, figsize), tight_layout=True)

    if sequence in da.sequence:
        data[0,data.shape[1]//2] = da.sel(sequence=sequence).item()
        if da_annotation is not None:
            data_annotation[0, data.shape[1] // 2] = da_annotation.sel(sequence=sequence).item()

    # Plot the heatmap
    im = ax.imshow(data, cmap="Greens", **kwargs)


    # Create colorbar
    cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax.transAxes)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.set_ylabel(da.name.capitalize().replace('_',' '), rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=bases_axes)
    ax.set_yticks(np.arange(data.shape[0]), labels=bases_axes)
    #
    # # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(0, data.shape[1] + 1, 3) - .5, minor=True)
    ax.set_yticks(np.arange(0, data.shape[0] + 1, 3) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(sequence)


    position = np.arange(len(sequence))
    # x = np.hstack([np.vstack([position-1, position, position+1]).T.flatten()])

    for p in position:
        #for xi in x[p*3+1:(p+1)*3+1]:
            # ax.annotate(seq[p], xy=(xi,-20), xytext=(p,-20), xycoords=('data', 'axes points'), textcoords=('data','offset points'),
            #             ha='center', va='bottom', arrowprops={'width': 0.3, 'headwidth': 3, 'headlength':2})
        text = sequence[p]
        for xi in [-1,0,1]:
            if xi == 0:
                text = sequence[p]+str(p+1)
                # ax.annotate(p+1, xy=(p * 3 + 1, -57), xycoords=('data', 'axes points'), ha='center', va='bottom')
                # ax.annotate(p + 1, xy=(-57, p * 3 + 1), xycoords=('axes points', 'data'), ha='left', va='center')
            else:
                text = '    '
            ax.annotate(text, xy=(p*3+1+xi*2.3/3, -17), xytext=(p*3+1, -20), xycoords=('data', 'axes points'),
                        textcoords=('data', 'offset points'),
                        ha='center', va='bottom', arrowprops={'width': 0.2, 'headwidth': 3, 'headlength': 2}, in_layout=True)
            ax.annotate(text, xy=(-17, p * 3 + 1 + xi * 2.3 / 3), xytext=(-25, p * 3 + 1), xycoords=('axes points', 'data'),
                        textcoords=('offset points', 'data'),
                        ha='left', va='center', arrowprops={'width': 0.2, 'headwidth': 3, 'headlength': 2}, in_layout=True)


    ax.set_xlabel('Mutation 1', ha='center', labelpad=25)
    ax.set_ylabel('Mutation 2', labelpad=30)
    # ax.xaxis.set_label_coords(0.5, -0.12)
    # ax.yaxis.set_label_coords(-0.13, 0.5)

    if da_annotation is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data_annotation[i, j]
                if ~np.isnan(value):
                    text = ax.text(j, i, value.astype(int),
                                   ha="center", va="center", color="k")

    if save and save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path / (sequence + '_double_mutations_' + da.name + '.pdf'))
        fig.savefig(save_path / (sequence + '_double_mutations_' + da.name + '.png'))

    return fig, ax



def plot_sequencing_match(match, write_path, title, filename, unit = 'um', MiSeq_pixels_to_um = None, Fluo_pixels_to_um = None, save=True):
    # TODO: Update for new tile and sequencing matches
    source = match.source
    source_in_destination = match.transform_coordinates(source)
    destination = match.destination
    destination_in_source = match.transform_coordinates(destination, inverse=True)

    source_vertices = match.source_vertices
    destination_vertices = match.transform_coordinates(match.source_vertices)

    if unit == 'um':
        source = Fluo_pixels_to_um(source)
        source_in_destination = MiSeq_pixels_to_um(source_in_destination)
        destination = MiSeq_pixels_to_um(destination)
        destination_in_source = Fluo_pixels_to_um(destination_in_source)

        source_vertices = Fluo_pixels_to_um(source_vertices)
        destination_vertices = MiSeq_pixels_to_um(destination_vertices)

    # fig = plt.figure(figsize = (8,4))
    # ax1, ax2 = fig.subplots(1, 2)

    fig, ax1 = plt.subplots(figsize = (8,4))
    fig.subplots_adjust(0.05,0.05,0.95,0.93)

    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes('right', size='60%', pad=0.5)
    #
    # from mpl_toolkits.axes_grid1 import ImageGrid
    #
    # fig = plt.figure(figsize = (8,4))
    # grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.1, add_all=True, label_mode='L')
    #
    # ax1 = grid[0]
    # ax2 = grid[1]

    ax1.scatter(source_in_destination[:,0],source_in_destination[:,1],c='#40A535',marker = 'x')
    ax1.scatter(destination[:,0],destination[:,1], marker = '.', facecolors = 'k', edgecolors='k')
    ax1.set_facecolor('white')
    ax1.set_aspect('equal')


    ax1.set_xlim([0, 30000])
    ax1.set_ylim([0, 30000])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax1.ticklabel_format(axis='both', style='sci', scilimits=(5,6), useOffset=None, useLocale=None, useMathText=True)

    if unit == 'um':
        start = 0
        end = 1001
        stepsize = 250

        ax1.set_xlim([0, 1000])
        ax1.set_ylim([0, 1000])
        ax1.set_xlabel('x (\u03BCm)')
        ax1.set_ylabel('y (\u03BCm)')
        ax1.xaxis.set_ticks(np.arange(start, end, stepsize))
        ax1.yaxis.set_ticks(np.arange(start, end, stepsize))

        ax1.ticklabel_format(axis='both', style='sci', scilimits=(0,4), useOffset=None, useLocale=None, useMathText=True)

    p = patches.Polygon(destination_vertices,
                        linewidth=2, edgecolor='#40A535', facecolor='none', fill='false'
                        )

    ax1.add_patch(p)


    #plt.tight_layout()

    # fig.savefig(write_path.joinpath(name + '.pdf'), bbox_inches='tight')
    # fig.savefig(write_path.joinpath(name + '.png'), bbox_inches='tight', dpi=1000)


    ax2.scatter(source[:,0],source[:,1],c='#40A535',marker = 'x')
    ax2.scatter(destination_in_source[:,0],destination_in_source[:,1], marker = '.', facecolors = 'k', edgecolors='k')
    ax2.set_facecolor('white')
    ax2.set_aspect('equal')

    image_size = np.array([np.min(source_vertices, axis=0),np.max(source_vertices, axis=0)])

    ax2.set_xlim([image_size[0,0], image_size[1,0]])
    ax2.set_ylim([image_size[0,1], image_size[1,1]])

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    ax2.ticklabel_format(axis='both', style='sci', scilimits=(5,6), useOffset=None, useLocale=None, useMathText=True)

    if unit == 'um':
        # ax2.set_xlim([image_size[0, 0], image_size[1, 0]])
        # ax2.set_ylim([image_size[0, 1], image_size[1, 1]])
        ax2.set_xlabel('x (\u03BCm)')
        ax2.set_ylabel('y (\u03BCm)')

        ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,4), useOffset=None, useLocale=None, useMathText=True)

    # if ax1pos is not None:
    #     ax1.set_position(ax1pos)
    #     ax2.set_position(ax2pos)

    for spine in ax2.spines.values():
        spine.set_edgecolor('#40A535')

    def connect_vertices_in_axis(verticesA, verticesB, axisA, axisB, **kwargs):
        for vertexA, vertexB in zip(verticesA, verticesB):
            con = ConnectionPatch(xyA=vertexA, xyB=vertexB, coordsA="data", coordsB="data",
                                  axesA=axisA, axesB=axisB, **kwargs)
            axisB.add_artist(con)

    connect_vertices_in_axis(source_vertices, destination_vertices, ax2, ax1)

    for artist in ax1.artists:
        artist.set_linestyle((0,(5,5)))
        artist.set_linewidth(0.5)
        artist.set_edgecolor('grey')

    fig.suptitle(title, fontsize='medium')

    plt.show()

    if save:
        n = filename.replace('\\', '_')
        fig.savefig(write_path.joinpath(n + '.pdf'), bbox_inches='tight')
        fig.savefig(write_path.joinpath(n + '.png'), bbox_inches='tight', dpi=250)

    return ax1, ax2



# Show all matched files in tiles
def plot_matched_files_in_tile(files, show_file_coordinates=False, show_file_vertices=True, unit='um', save=False):
    # TODO: Update for new tile and sequencing matches

    def MiSeq_pixels_to_um(pixels):
        return 958 / 2800 * (pixels - 1000) / 10

    for tile in files[0].experiment.sequencing_data_for_mapping.tiles:
        files_on_tile = [file for file in files if file.sequencing_match.tile == tile.number]
        print(len(files_on_tile))

        tile_coordinates = tile.coordinates

        if unit == 'um':
            tile_coordinates = MiSeq_pixels_to_um(tile_coordinates)

        figure = plt.figure()
        axis = figure.gca()

        axis.scatter(tile_coordinates[:,0], tile_coordinates[:,1], marker = '.', facecolors = 'k', edgecolors='k')
        axis.set_facecolor('white')
        axis.set_aspect('equal')

        axis.set_xlim([0, 30000])
        axis.set_ylim([0, 30000])
        axis.set_xlabel('x (FASTQ)')
        axis.set_ylabel('y (FASTQ)')

        axis.ticklabel_format(axis='both', style='sci', scilimits=(5,6), useOffset=None, useLocale=None, useMathText=True)

        for file in files_on_tile:
            if show_file_coordinates:
                coordinates = file.sequencing_match.source_to_destination
                if unit == 'um':
                    coordinates = MiSeq_pixels_to_um(coordinates)
                axis.scatter(coordinates[:, 0], coordinates[:, 1], c='#40A535', marker='x')

            # figure.gca().scatter(vertices[:, 0], vertices[:, 1], c='g')
            if show_file_vertices:
                vertices = file.sequencing_match.transform_coordinates(file.movie.channels[1].vertices)
                if unit == 'um':
                    vertices = MiSeq_pixels_to_um(vertices)
                p = patches.Polygon(vertices,
                                    linewidth=2, edgecolor='#40A535', facecolor='none', fill='false'
                                    )

                axis.add_patch(p)

        axis.set_title(f'Tile {tile.name}')

        if unit == 'um':
            start = 0
            end = 1001
            stepsize = 250

            axis.set_xlim([0, 1000])
            axis.set_ylim([0, 1000])
            axis.set_xlabel('x (\u03BCm)')
            axis.set_ylabel('y (\u03BCm)')
            axis.xaxis.set_ticks(np.arange(start, end, stepsize))
            axis.yaxis.set_ticks(np.arange(start, end, stepsize))

            axis.ticklabel_format(axis='both', style='sci', scilimits=(0, 4), useOffset=None, useLocale=None,
                                 useMathText=True)

        if save:
            figure.tight_layout()
            figure.savefig(f'Matched_files_in_tile_{tile.name}.png', bbox_inches='tight', dpi=250)

def plot_cluster_locations_per_tile(dataset, number_of_tiles=19, number_of_surfaces=2, save_filepath=None):
    # TODO: Update for new tile and sequencing matches
    # df should contain Tile number, x and y

    number_of_rows = number_of_surfaces * math.ceil(number_of_tiles / 10)
    number_of_columns = np.min([10, number_of_tiles])
    figure, axes = plt.subplots(number_of_rows, number_of_columns, sharex=True, sharey=True,
                                figsize=(number_of_columns*2, number_of_rows*2))

    for ax in axes.flatten():
        row_index, column_index = np.where(axes == ax)
        row_index = row_index[0]
        column_index = column_index[0]

        side_index = row_index // 2
        tile_index = column_index + row_index % 2 * 10
        tile_number = (side_index+1)*1000+100+tile_index+1

        # print(side_index, tile_index, row_index, column_index, tile_number)

        if tile_index+1 > number_of_tiles:
            ax.set_visible(False)
            continue

        dataset_tile = dataset[{'sequence': dataset.tile == tile_number}]

        dataset_tile.to_dataframe().plot.scatter(x='x', y='y', ax=ax, marker='.', s=7)
        ax.set_title(tile_number)
        ax.set_aspect('equal')
        ax.set_xlabel('x (sequencer)')
        ax.set_ylabel('y (sequencer)')


    figure.tight_layout()
    if save_filepath:
        figure.savefig(save_filepath, dpi=300)