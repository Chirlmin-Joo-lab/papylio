import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import itertools
import tqdm
from matplotlib import patches
from papylio.plugins.holliday_junction.sequence_generation import all_basepaired_subsets


def basepair_count_per_position(basepair_count, save_path):
    title = (basepair_count.name.replace('basepair_count','basepair_count_per_position'))

    basepair_count_stacked = basepair_count.stack(basepair=('base_0', 'base_1'))
    basepair_count_stacked['basepair'] = xr.concat([basepair_count_stacked.base_0, basepair_count_stacked.base_1],
                                                   dim='dummy').str.join(dim='dummy').values

    basepaired_position = xr.concat(
        [basepair_count_stacked.sel(position_0=p0, position_1=p1) for p0, p1 in [[7, 0], [1, 2], [3, 4], [5, 6]]],
        dim='position')

    cm = plt.colormaps['tab20c']
    figure, axis = plt.subplots(figsize=(10,3), layout='constrained')
    axis.cla()
    x = np.arange(len(basepaired_position.position)) * (len(basepaired_position.basepair) + 4)
    labels = []
    x_ticks = []
    for i, basepair in enumerate(basepaired_position.basepair.values):
        axis.bar(x + i, basepaired_position.sel(basepair=basepair), width=1, label=basepair, color=cm.colors[i])
        labels += [basepair[0] + '\n' + basepair[1]] * len(x)
        x_ticks += (x+i).tolist()
    # ax.legend()
    axis.set_xticks(x_ticks, labels)
    axis.set_ylabel('Count')
    axis.set_title(title)

    axis.annotate('Basepair 8-1', xy=(8,-40), xycoords=('data','axes points'), ha='center')
    axis.annotate('Basepair 2-3', xy=(28,-40), xycoords=('data','axes points'), ha='center')
    axis.annotate('Basepair 4-5', xy=(48,-40), xycoords=('data','axes points'), ha='center')
    axis.annotate('Basepair 6-7', xy=(68,-40), xycoords=('data','axes points'), ha='center')
    savefile_path = save_path / title
    figure.savefig(savefile_path.with_suffix('.png'))
    figure.savefig(savefile_path.with_suffix('.pdf'))

def basepaired_sequence_subset_count(sequence_subset_count, save_path):
    title = (sequence_subset_count.name.replace('sequence_count', 'basepaired_sequence_count'))
    variable = sequence_subset_count.attrs['variable']

    sequence_subset_count2 = sequence_subset_count.reindex(**{variable: all_basepaired_subsets(), 'fill_value': 0})

    # bases = ['A', 'T', 'C', 'G']
    basepairs = ['AT', 'TA', 'CG', 'GC']
    # possible_sequence_subsets = [''.join(ss) for ss in
    #                              itertools.product(bases, basepairs, bases, bases, basepairs, bases)]
    # possible_sequence_subset_count = sequence_subset_count.reindex(sequence_subset=possible_sequence_subsets,
    #                                                                fill_value=0)

    sequence_subset_count_bp = xr.DataArray(0, dims=('bp0', 'bp1', 'bp2', 'bp3'),
                                            coords={'bp0': basepairs, 'bp1': basepairs, 'bp2': basepairs,
                                                    'bp3': basepairs}).astype(sequence_subset_count2.dtype)
    for bpc in tqdm.tqdm(itertools.product(basepairs, basepairs, basepairs, basepairs)):
        sequence_subset_count_bp.loc[dict(bp0=bpc[0], bp1=bpc[1], bp2=bpc[2], bp3=bpc[3])] = \
            sequence_subset_count2.sel(**{variable: (''.join(bpc)[1:] + ''.join(bpc)[0])})

    figure, axis = plt.subplots(figsize=(7.2, 6.5))#, layout='tight')
    # axis = axes[0]
    axis.cla()
    data = sequence_subset_count_bp.stack(bp02=('bp0', 'bp2')).stack(bp13=('bp1', 'bp3')).T
    image = axis.imshow(data.values, vmin=0, vmax=sequence_subset_count_bp.mean() * 2, cmap='coolwarm')
    # axis.images[0].set_data(data.values)

    for x, bp in zip(np.arange(0, 16, 4) + 1.5, basepairs):
        axis.annotate(bp[::-1], xy=(x, 16), xycoords=('data', 'data'), ha='center', va='center')
    for y, bp in zip(np.arange(0, 16, 4) + 1.5, basepairs):
        axis.annotate(bp, xy=(-1, y), xycoords=('data', 'data'), ha='center', va='center')
    for x, bp in zip(np.arange(0, 16), basepairs * 4):
        axis.annotate(bp, xy=(x, -1), xycoords=('data', 'data'), ha='center', va='center')
    for y, bp in zip(np.arange(0, 16), basepairs * 4):
        axis.annotate(bp, xy=(16, y), xycoords=('data', 'data'), ha='center', va='center')

    axis.set_xticks(np.arange(-0.5, 16, 4)[1:-1], minor=False)
    # axis.set_xticks(np.arange(0,16), basepairs*4, minor=True)
    axis.set_yticks(np.arange(-0.5, 16, 4)[1:-1], minor=False)
    # axis.set_yticks(np.arange(0,16), basepairs*4, minor=True)
    axis.tick_params(which="major", top=False, labeltop=False, right=False, labelright=False,
                     bottom=False, labelbottom=False, left=False, labelleft=False)
    axis.grid(which='major', color="w", linestyle='-', linewidth=2)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.annotate('Basepair 1-8', xy=(7.5, 17), xycoords=('data', 'data'), ha='center', va='center')
    axis.annotate('Basepair 2-3', xy=(-2, 7.5), xycoords=('data', 'data'), ha='center', va='center',
                  rotation='vertical')
    axis.annotate('Basepair 4-5', xy=(7.5, -2), xycoords=('data', 'data'), ha='center', va='center')
    axis.annotate('Basepair 6-7', xy=(17, 7.5), xycoords=('data', 'data'), ha='center', va='center',
                  rotation='vertical')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", "4%", pad="15%")
    figure.colorbar(image, cax=cax)

    # figure.colorbar(image, aspect=30)
    # cax = figure.axes[-1]
    cax.set_ylabel('Count')
    cax.axes.ticklabel_format(scilimits=(0,0))
    for spine in cax.spines.values():
        spine.set_visible(False)

    fontsize = 8
    title2 = title + ' - scaled_mean_basepaired'
    figure.suptitle(title2, fontsize=fontsize)
    savefile_path = save_path / title2
    figure.savefig(savefile_path.with_suffix('.png'))
    figure.savefig(savefile_path.with_suffix('.pdf'))

    image.set_clim(sequence_subset_count_bp.min(), np.percentile(sequence_subset_count.values, 20))
    title2 = title + ' - scaled_bottom_20%_of_all'
    figure.suptitle(title2, fontsize=fontsize)
    savefile_path = save_path / title2
    figure.savefig(savefile_path.with_suffix('.png'))
    figure.savefig(savefile_path.with_suffix('.pdf'))

    image.set_clim(sequence_subset_count_bp.min(), sequence_subset_count.mean().item() * 2)
    title2 = title + ' - scaled_mean_of_all'
    figure.suptitle(title2, fontsize=fontsize)
    savefile_path = save_path / title2
    figure.savefig(savefile_path.with_suffix('.png'))
    figure.savefig(savefile_path.with_suffix('.pdf'))

    image.set_clim(sequence_subset_count_bp.min(), sequence_subset_count_bp.max())
    title2 = title + ' - scaled_min_max_basepaired'
    figure.suptitle(title2, fontsize=fontsize)
    savefile_path = save_path / title2
    figure.savefig(savefile_path.with_suffix('.png'))
    figure.savefig(savefile_path.with_suffix('.pdf'))




# def plot_basepaired_holliday_junction(data, save_path, **imshow_kwargs):
#
#     data = data.reindex(**{'sequence_subset': all_basepaired_subsets(), 'fill_value': np.nan})
#
#     # bases = ['A', 'T', 'C', 'G']
#     basepairs = ['AT', 'TA', 'CG', 'GC']
#
#     data_bp = xr.DataArray(0, dims=('bp0', 'bp1', 'bp2', 'bp3'),
#                            coords={'bp0': basepairs, 'bp1': basepairs, 'bp2': basepairs,
#                                    'bp3': basepairs}).astype(data.dtype)
#     for bpc in tqdm.tqdm(itertools.product(basepairs, basepairs, basepairs, basepairs)):
#         data_bp.loc[dict(bp0=bpc[0], bp1=bpc[1], bp2=bpc[2], bp3=bpc[3])] = \
#             data.sel(**{'sequence_subset': (''.join(bpc)[1:] + ''.join(bpc)[0])})
#
#     figure, axis = plt.subplots(figsize=(7.9, 6.5), layout='constrained')
#     # axis = axes[0]
#     axis.cla()
#     data_bp_stacked = data_bp.stack(bp02=('bp0', 'bp2')).stack(bp13=('bp1', 'bp3')).T
#     image = axis.imshow(data_bp_stacked.values, cmap='coolwarm', **imshow_kwargs)
#     # axis.images[0].set_data(data.values)
#
#     for x, bp in zip(np.arange(0, 16, 4) + 1.5, basepairs):
#         axis.annotate(bp[::-1], xy=(x, 16), xycoords=('data', 'data'), ha='center', va='center')
#     for y, bp in zip(np.arange(0, 16, 4) + 1.5, basepairs):
#         axis.annotate(bp, xy=(-1, y), xycoords=('data', 'data'), ha='center', va='center')
#     for x, bp in zip(np.arange(0, 16), basepairs * 4):
#         axis.annotate(bp, xy=(x, -1), xycoords=('data', 'data'), ha='center', va='center')
#     for y, bp in zip(np.arange(0, 16), basepairs * 4):
#         axis.annotate(bp, xy=(16, y), xycoords=('data', 'data'), ha='center', va='center')
#
#     axis.set_xticks(np.arange(-0.5, 16, 4)[1:-1], minor=False)
#     # axis.set_xticks(np.arange(0,16), basepairs*4, minor=True)
#     axis.set_yticks(np.arange(-0.5, 16, 4)[1:-1], minor=False)
#     # axis.set_yticks(np.arange(0,16), basepairs*4, minor=True)
#     axis.tick_params(which="major", top=False, labeltop=False, right=False, labelright=False,
#                      bottom=False, labelbottom=False, left=False, labelleft=False)
#     axis.grid(which='major', color="w", linestyle='-', linewidth=2)
#     for spine in axis.spines.values():
#         spine.set_visible(False)
#     axis.annotate('Basepair 1-8', xy=(7.5, 17), xycoords=('data', 'data'), ha='center', va='center')
#     axis.annotate('Basepair 2-3', xy=(-2, 7.5), xycoords=('data', 'data'), ha='center', va='center',
#                   rotation='vertical')
#     axis.annotate('Basepair 4-5', xy=(7.5, -2), xycoords=('data', 'data'), ha='center', va='center')
#     axis.annotate('Basepair 6-7', xy=(17, 7.5), xycoords=('data', 'data'), ha='center', va='center',
#                   rotation='vertical')
#
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     divider = make_axes_locatable(axis)
#     cax = divider.append_axes("right", "4%", pad="15%")
#     figure.colorbar(image, cax=cax)
#
#     # figure.colorbar(image, aspect=30)
#     # cax = figure.axes[-1]
#     cax.set_ylabel(data.name)
#     cax.axes.ticklabel_format(scilimits=(0,0))
#     for spine in cax.spines.values():
#         spine.set_visible(False)
#
#     title = 'basepaired_HJ_' + data.name
#
#     fontsize = 8
#     # title2 = title + ' - scaled_mean_basepaired'
#     figure.suptitle(title, fontsize=fontsize)
#     savefile_path = save_path / title
#     figure.savefig(savefile_path.with_suffix('.png'))
#     figure.savefig(savefile_path.with_suffix('.pdf'))
#
#     # image.set_clim(sequence_subset_count_bp.min(), np.percentile(sequence_subset_count.values, 20))
#     # title2 = title + ' - scaled_bottom_20%_of_all'
#     # figure.suptitle(title2, fontsize=fontsize)
#     # savefile_path = save_path / title2
#     # figure.savefig(savefile_path.with_suffix('.png'))
#     # figure.savefig(savefile_path.with_suffix('.pdf'))
#     #
#     # image.set_clim(sequence_subset_count_bp.min(), sequence_subset_count.mean().item() * 2)
#     # title2 = title + ' - scaled_mean_of_all'
#     # figure.suptitle(title2, fontsize=fontsize)
#     # savefile_path = save_path / title2
#     # figure.savefig(savefile_path.with_suffix('.png'))
#     # figure.savefig(savefile_path.with_suffix('.pdf'))
#     #
#     # image.set_clim(sequence_subset_count_bp.min(), sequence_subset_count_bp.max())
#     # title2 = title + ' - scaled_min_max_basepaired'
#     # figure.suptitle(title2, fontsize=fontsize)
#     # savefile_path = save_path / title2
#     # figure.savefig(savefile_path.with_suffix('.png'))
#     # figure.savefig(savefile_path.with_suffix('.pdf'))
#
#


def plot_holliday_junction(data, size=1, name=None, s2max=None, geometry='square',
                                      axis_facecolor="lightgrey", save_path=None, vmin=None, vmax=None, axis=None, cmap=None):

    if name is None:
        name = data.name

    if data.ndim == 1:
        data = data.expand_dims('loop_dim', 0)

    data = xr.Dataset(dict(data=data))
    if np.isscalar(size):
        data['size'] = xr.DataArray(size, coords=data.data.coords)
    elif isinstance(size, (list, tuple)):
        data['size'] = xr.DataArray(np.full_like(data.data, np.array(size)[None,:].T, dtype='object'), dims=data.data.dims)
    elif isinstance(size, xr.DataArray):
        if size.ndim == 1:
            size = size.expand_dims({data.data.dims[0]: data.data.shape[0]}, 0)
        data['size'] = size

    if isinstance(geometry, str):
        data['geometry'] = xr.DataArray(geometry, coords=data.data.coords)
    elif isinstance(geometry, (list, tuple)):
        data['geometry'] = xr.DataArray(np.full_like(data.data, np.array(geometry)[None,:].T, dtype='object'), dims=data.data.dims)
    elif isinstance(geometry, xr.DataArray):
        if geometry.ndim == 1:
            geometry = geometry.expand_dims({data.data.dims[0]: data.data.shape[0]}, 0)
        data['geometry'] = geometry

    # data = data.reindex(**{'sequence_subset': all_basepaired_subsets(), 'fill_value': np.nan})
    data['sequence_subset'] = data['sequence_subset'].astype('U')

    # bases = ['A', 'T', 'C', 'G']
    basepairs = ['AT', 'TA', 'CG', 'GC']

    bp0 = np.char.add(data.sequence_subset.str[7:8].values, data.sequence_subset.str[0:1].values)
    bp1 = data.sequence_subset.str[1:3].values
    bp2 = data.sequence_subset.str[3:5].values
    bp3 = data.sequence_subset.str[5:7].values
    data_bp = data.assign_coords(bp=('sequence_subset',pd.MultiIndex.from_arrays([bp0, bp1, bp2, bp3], names=['bp0','bp1','bp2','bp3'])))
    data_bp = data_bp.swap_dims(sequence_subset='bp').unstack('bp')
    # data_bp = data_bp.sel(bp0=basepairs, bp1=basepairs, bp2=basepairs, bp3=basepairs)


    # data_bp = xr.DataArray(0, dims=('bp0', 'bp1', 'bp2', 'bp3'),
    #                        coords={'bp0': basepairs, 'bp1': basepairs, 'bp2': basepairs,
    #                                'bp3': basepairs}).astype(data.dtype)
    # for bpc in tqdm.tqdm(itertools.product(basepairs, basepairs, basepairs, basepairs)):
    #     data_bp.loc[dict(bp0=bpc[0], bp1=bpc[1], bp2=bpc[2], bp3=bpc[3])] = \
    #         data.sel(**{'sequence_subset': (''.join(bpc)[1:] + ''.join(bpc)[0])})
    if axis is None:
        figure, axis = plt.subplots(figsize=(7.9, 6.5), layout='constrained')
    else:
        figure = axis.figure
    # axis = axes[0]
    axis.cla()
    data_bp_stacked = data_bp.stack(bp13=('bp1', 'bp3')).stack(bp02=('bp0', 'bp2'))
    data_bp_stacked = data_bp_stacked.sel(bp02=~data_bp_stacked.data.isnull().all('bp13').values.squeeze())
    data_bp_stacked = data_bp_stacked.sel(bp13=~data_bp_stacked.data.isnull().all('bp02').values.squeeze())
    # image = axis.imshow(data_bp_stacked.values, cmap='coolwarm', **imshow_kwargs)

    data_bp_stacked2 = data_bp_stacked.copy()
    data_bp_stacked2 = data_bp_stacked2.assign_coords(
        dict(bp02i=('bp02', np.arange(len(data_bp_stacked.bp02))), bp13i=('bp13', np.arange(len(data_bp_stacked.bp13)))))\
        .unstack(('bp13', 'bp02')).stack(bp0123=('bp0', 'bp1', 'bp2', 'bp3'))
    axis.cla()
    # axis.scatter(x=data_bp_stacked2.bp02i, y=data_bp_stacked2.bp13i, c=data_bp_stacked2.data.values, cmap='coolwarm',
    #              vmin=0, vmax=1, s=data_bp_stacked2.size_data.values, marker='s')

    x = data_bp_stacked2.bp02i.values
    y = data_bp_stacked2.bp13i.values

    selection = (~np.isnan(x)) & (~np.isnan(y))
    x = x[selection]
    y = y[selection]
    from matplotlib.colors import Normalize
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = data_bp_stacked2.data.max().item()
    norm = Normalize(vmin=vmin, vmax=vmax)
    # cmap = 'coolwarm'
    if cmap is None:
        cmap = plt.get_cmap('coolwarm').copy()
        cmap.set_bad([1, 1, 1, 1])
    from matplotlib.cm import ScalarMappable
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    # c = plt.get_cmap('coolwarm')(norm(data_bp_stacked2.data.values))

    for i in range(data_bp_stacked2.data.shape[0]):
        c = scalar_mappable.to_rgba(data_bp_stacked2.data[i].values)[selection]

        s = np.sqrt(data_bp_stacked2.size[i].values)[selection]
        if s2max is None:
            s2max = np.nanmax(s)**2
        s = np.minimum(s/np.sqrt(s2max),1) #s/s.max()

        g = data_bp_stacked2.geometry[i].values[selection]

        for xi, yi, ci, si, gi in zip(x,y,c,s,g):
            shape_kwargs = dict(facecolor=ci, linewidth=None)
            if gi == 'square':
                rect = plt.Rectangle([xi - si / 2, yi - si / 2], si, si, **shape_kwargs)
            elif gi == 'circle':
                rect = plt.Circle([xi, yi], radius=si/2, **shape_kwargs)
            elif gi == 'diamond':
                si2 = si/np.sqrt(2)
                rect = plt.Rectangle([xi, yi - si / 2], si2, si2, angle=45, **shape_kwargs)
            elif gi == 'triangle_top_left':
                rect = plt.Polygon([[xi - si / 2, yi - si / 2],
                                    [xi - si / 2, yi + si / 2],
                                    [xi + si / 2, yi - si / 2]], closed=True, **shape_kwargs)
            elif gi == 'triangle_bottom_right':
                rect = plt.Polygon([[xi + si / 2, yi + si / 2],
                                    [xi - si / 2, yi + si / 2],
                                    [xi + si / 2, yi - si / 2]], closed=True, **shape_kwargs)
            elif gi == 'semi_circle_top_left':
                rect = patches.Wedge([xi, yi], r=si / 2, theta1=-225, theta2=-45, **shape_kwargs)
            elif gi == 'semi_circle_bottom_right':
                rect = patches.Wedge([xi, yi], r=si / 2, theta1=-45, theta2=-225, **shape_kwargs)
            else:
                rect = None
            if rect is not None:
                axis.add_patch(rect)
    axis.set_aspect(1)
    half_shape_size = np.nanmax(s)/2
    axis.set_xlim(np.nanmin(x)-half_shape_size, np.nanmax(x)+half_shape_size)
    axis.set_ylim(np.nanmin(y)-half_shape_size, np.nanmax(y)+half_shape_size)
    # axis.autoscale_view()

    # axis.images[0].set_data(data.values)


    x_len = len(np.unique(data_bp_stacked2.bp02i))
    y_len = len(np.unique(data_bp_stacked2.bp13i))

    bp0_labels, bp0_labels_x = np.unique(data_bp_stacked.bp0, return_counts=True)
    bp0_edges = np.hstack([[0], bp0_labels_x.cumsum()])
    bp0_labels_x = bp0_labels_x.cumsum() - bp0_labels_x / 2 - 0.5
    for label, x in zip(bp0_labels, bp0_labels_x):
        axis.annotate(label, xy=(x, y_len), xycoords=('data', 'data'), ha='center', va='center')

    bp1_labels, bp1_labels_y = np.unique(data_bp_stacked.bp1, return_counts=True)
    bp1_edges = np.hstack([[0], bp1_labels_y.cumsum()])
    bp1_labels_y = bp1_labels_y.cumsum() - bp1_labels_y / 2 - 0.5
    for label, y in zip(bp1_labels, bp1_labels_y):
        axis.annotate(label, xy=(-1, y), xycoords=('data', 'data'), ha='center', va='center')

    bp2_labels = data_bp_stacked.bp2.values
    bp2_labels_x = np.arange(len(bp2_labels))
    for label, x in zip(bp2_labels, bp2_labels_x):
        axis.annotate(label, xy=(x, -1), xycoords=('data', 'data'), ha='center', va='center')

    bp3_labels = data_bp_stacked.bp3.values
    bp3_labels_y = np.arange(len(bp3_labels))
    for label, y in zip(bp3_labels, bp3_labels_y):
        axis.annotate(label, xy=(x_len, y), xycoords=('data', 'data'), ha='center', va='center')


    axis.set_xticks(bp0_edges-0.5, minor=False)
    axis.set_yticks(bp1_edges-0.5, minor=False)
    axis.tick_params(which="major", top=False, labeltop=False, right=False, labelright=False,
                     bottom=False, labelbottom=False, left=False, labelleft=False)
    axis.grid(which='major', color="w", linestyle='-', linewidth=2)
    # axis.grid(which='major', color="k", linestyle='-', linewidth=2)

    axis.set_xticks(bp2_labels_x, minor=True)
    axis.set_yticks(bp3_labels_y, minor=True)
    axis.tick_params(which="minor", top=False, labeltop=False, right=False, labelright=False,
                     bottom=False, labelbottom=False, left=False, labelleft=False)
    axis.grid(which='minor', color="k", linestyle='-', linewidth=0)

    #
    # x_len = len(np.unique(data_bp_stacked2.bp02i))
    # x_interval = len(np.unique(data_bp_stacked2.bp2))
    #
    # y_len = len(np.unique(data_bp_stacked2.bp13i))
    # y_interval = len(np.unique(data_bp_stacked2.bp3))
    #
    # for i in np.arange(0, x_len, x_interval):
    #     x = i + x_interval/2-0.5
    #     bp = data_bp_stacked2.bp0[i].item()
    #     axis.annotate(bp, xy=(x, y_len), xycoords=('data', 'data'), ha='center', va='center')
    # for i in np.arange(0, y_len, y_interval):
    #     y = i + y_interval/2-0.5
    #     bp = data_bp_stacked2.bp1[i].item()
    #     axis.annotate(bp, xy=(-1, y), xycoords=('data', 'data'), ha='center', va='center')
    # for i in np.arange(0, x_len):
    #     x = i
    #     bp = data_bp_stacked2.bp2[i].item()
    #     axis.annotate(bp, xy=(x, -1), xycoords=('data', 'data'), ha='center', va='center')
    # for i in np.arange(0, y_len):
    #     y = i
    #     bp = data_bp_stacked2.bp3[i].item()
    #     axis.annotate(bp, xy=(x_len, y), xycoords=('data', 'data'), ha='center', va='center')
    #
    # axis.set_xticks(np.arange(-0.5, x_len, x_interval)[1:-1], minor=False)
    # axis.set_yticks(np.arange(-0.5, y_len, y_interval)[1:-1], minor=False)
    # axis.tick_params(which="major", top=False, labeltop=False, right=False, labelright=False,
    #                  bottom=False, labelbottom=False, left=False, labelleft=False)
    # axis.grid(which='major', color="w", linestyle='-', linewidth=2)
    # # axis.grid(which='major', color="k", linestyle='-', linewidth=2)
    #
    # axis.set_xticks(np.arange(-0.5,x_len + 0.5), minor=True)
    # axis.set_yticks(np.arange(-0.5,y_len + 0.5), minor=True)
    # axis.tick_params(which="minor", top=False, labeltop=False, right=False, labelright=False,
    #                  bottom=False, labelbottom=False, left=False, labelleft=False)
    # axis.grid(which='minor', color="k", linestyle='-', linewidth=0)

    for spine in axis.spines.values():
        spine.set_visible(False)
        spine.set_lw(2)

    axis.annotate('Base pair 7-8', xy=(x_len/2-0.5, y_len+1), xycoords=('data', 'data'), ha='center', va='center')
    axis.annotate('Base pair 1-2', xy=(-2, y_len/2-0.5), xycoords=('data', 'data'), ha='center', va='center',
                  rotation='vertical')
    axis.annotate('Base pair 3-4', xy=(x_len/2-0.5, -2), xycoords=('data', 'data'), ha='center', va='center')
    axis.annotate('Base pair 5-6', xy=(x_len+1, y_len/2-0.5), xycoords=('data', 'data'), ha='center', va='center',
                  rotation=270)

    axis.set_facecolor(axis_facecolor)
    axis.invert_yaxis()

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axis)
    bbox = axis.get_window_extent()
    width, height = bbox.width, bbox.height
    s = 16 / width
    p = 60 / width
    cax = divider.append_axes("right", f"{s*100}%", pad=f"{p*100}%")
    figure.colorbar(scalar_mappable, cax=cax)

    # figure.colorbar(image, aspect=30)
    # cax = figure.axes[-1]
    if 'unit' in data.data.attrs and data.data.attrs['unit'] !='':
        unit_string = f' ({data.data.attrs["unit"]})'
    else:
        unit_string = ''
    cax.set_ylabel(name + unit_string)
    # cax.axes.ticklabel_format(scilimits=(0,0))
    for spine in cax.spines.values():
        spine.set_visible(False)

    title = 'HJ_' + name

    fontsize = 8
    # title2 = title + ' - scaled_mean_basepaired'
    figure.suptitle(title, fontsize=fontsize)
    if save_path is not None:
        savefile_path = save_path / title
        figure.savefig(savefile_path.with_suffix('.png'))
        figure.savefig(savefile_path.with_suffix('.pdf'))



def plot_basepaired_holliday_junction(data, size=1, name=None, s2max=None, geometry='square',
                                      axis_facecolor="lightgrey", save_path=None, vmin=None, vmax=None, axis=None, cmap=None):

    if name is None:
        name = data.name

    if data.ndim == 1:
        data = data.expand_dims('loop_dim', 0)

    data = xr.Dataset(dict(data=data))
    if np.isscalar(size):
        data['size'] = xr.DataArray(size, coords=data.data.coords)
    elif isinstance(size, (list, tuple)):
        data['size'] = xr.DataArray(np.full_like(data.data, np.array(size)[None,:].T, dtype='object'), dims=data.data.dims)
    elif isinstance(size, xr.DataArray):
        if size.ndim == 1:
            size = size.expand_dims({data.data.dims[0]: data.data.shape[0]}, 0)
        data['size'] = size

    if isinstance(geometry, str):
        data['geometry'] = xr.DataArray(geometry, coords=data.data.coords)
    elif isinstance(geometry, (list, tuple)):
        data['geometry'] = xr.DataArray(np.full_like(data.data, np.array(geometry)[None,:].T, dtype='object'), dims=data.data.dims)
    elif isinstance(geometry, xr.DataArray):
        if geometry.ndim == 1:
            geometry = geometry.expand_dims({data.data.dims[0]: data.data.shape[0]}, 0)
        data['geometry'] = geometry

    data = data.reindex(**{'sequence_subset': all_basepaired_subsets(), 'fill_value': np.nan})
    data['sequence_subset'] = data['sequence_subset'].astype('U')

    # bases = ['A', 'T', 'C', 'G']
    basepairs = ['AT', 'TA', 'CG', 'GC']

    bp0 = np.char.add(data.sequence_subset.str[7:8].values, data.sequence_subset.str[0:1].values)
    bp1 = data.sequence_subset.str[1:3].values
    bp2 = data.sequence_subset.str[3:5].values
    bp3 = data.sequence_subset.str[5:7].values
    data_bp = data.assign_coords(bp=('sequence_subset',pd.MultiIndex.from_arrays([bp0, bp1, bp2, bp3], names=['bp0','bp1','bp2','bp3'])))
    data_bp = data_bp.swap_dims(sequence_subset='bp').unstack('bp')
    data_bp = data_bp.sel(bp0=basepairs, bp1=basepairs, bp2=basepairs, bp3=basepairs)


    # data_bp = xr.DataArray(0, dims=('bp0', 'bp1', 'bp2', 'bp3'),
    #                        coords={'bp0': basepairs, 'bp1': basepairs, 'bp2': basepairs,
    #                                'bp3': basepairs}).astype(data.dtype)
    # for bpc in tqdm.tqdm(itertools.product(basepairs, basepairs, basepairs, basepairs)):
    #     data_bp.loc[dict(bp0=bpc[0], bp1=bpc[1], bp2=bpc[2], bp3=bpc[3])] = \
    #         data.sel(**{'sequence_subset': (''.join(bpc)[1:] + ''.join(bpc)[0])})
    if axis is None:
        figure, axis = plt.subplots(figsize=(7.9, 6.5), layout='constrained')
    else:
        figure = axis.figure
    # axis = axes[0]
    axis.cla()
    data_bp_stacked = data_bp.stack(bp13=('bp1', 'bp3')).stack(bp02=('bp0', 'bp2'))
    # image = axis.imshow(data_bp_stacked.values, cmap='coolwarm', **imshow_kwargs)

    data_bp_stacked2 = data_bp_stacked.copy()
    data_bp_stacked2 = data_bp_stacked2.assign_coords(dict(bp02i=('bp02', np.arange(16)), bp13i=('bp13', np.arange(16))))\
        .unstack(('bp13', 'bp02')).stack(bp0123=('bp0', 'bp1', 'bp2', 'bp3'))
    axis.cla()
    # axis.scatter(x=data_bp_stacked2.bp02i, y=data_bp_stacked2.bp13i, c=data_bp_stacked2.data.values, cmap='coolwarm',
    #              vmin=0, vmax=1, s=data_bp_stacked2.size_data.values, marker='s')

    x = data_bp_stacked2.bp02i.values
    y = data_bp_stacked2.bp13i.values
    from matplotlib.colors import Normalize
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = data_bp_stacked2.data.max().item()
    norm = Normalize(vmin=vmin, vmax=vmax)
    # cmap = 'coolwarm'
    if cmap is None:
        cmap = plt.get_cmap('coolwarm').copy()
        cmap.set_bad([1, 1, 1, 1])
    from matplotlib.cm import ScalarMappable
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    # c = plt.get_cmap('coolwarm')(norm(data_bp_stacked2.data.values))

    for i in range(data_bp_stacked2.data.shape[0]):
        c = scalar_mappable.to_rgba(data_bp_stacked2.data[i].values)

        s = np.sqrt(data_bp_stacked2.size[i].values)
        if s2max is None:
            s2max = np.nanmax(s)**2
        s = np.minimum(s/np.sqrt(s2max),1) #s/s.max()

        g = data_bp_stacked2.geometry[i].values

        for xi, yi, ci, si, gi in zip(x,y,c,s,g):
            shape_kwargs = dict(facecolor=ci, linewidth=None)
            if gi == 'square':
                rect = plt.Rectangle([xi - si / 2, yi - si / 2], si, si, **shape_kwargs)
            elif gi == 'circle':
                rect = plt.Circle([xi, yi], radius=si/2, **shape_kwargs)
            elif gi == 'diamond':
                si2 = si/np.sqrt(2)
                rect = plt.Rectangle([xi, yi - si / 2], si2, si2, angle=45, **shape_kwargs)
            elif gi == 'triangle_top_left':
                rect = plt.Polygon([[xi - si / 2, yi - si / 2],
                                    [xi - si / 2, yi + si / 2],
                                    [xi + si / 2, yi - si / 2]], closed=True, **shape_kwargs)
            elif gi == 'triangle_bottom_right':
                rect = plt.Polygon([[xi + si / 2, yi + si / 2],
                                    [xi - si / 2, yi + si / 2],
                                    [xi + si / 2, yi - si / 2]], closed=True, **shape_kwargs)
            elif gi == 'semi_circle_top_left':
                rect = patches.Wedge([xi, yi], r=si / 2, theta1=-225, theta2=-45, **shape_kwargs)
            elif gi == 'semi_circle_bottom_right':
                rect = patches.Wedge([xi, yi], r=si / 2, theta1=-45, theta2=-225, **shape_kwargs)

            axis.add_patch(rect)
    axis.set_aspect(1)
    half_shape_size = np.nanmax(s)/2
    axis.set_xlim(x.min()-half_shape_size, x.max()+half_shape_size)
    axis.set_ylim(y.min()-half_shape_size, y.max()+half_shape_size)
    # axis.autoscale_view()

    # axis.images[0].set_data(data.values)

    for x, bp in zip(np.arange(0, 16, 4) + 1.5, basepairs):
        # axis.annotate(bp[::-1], xy=(x, 16), xycoords=('data', 'data'), ha='center', va='center')
        axis.annotate(bp, xy=(x, 16), xycoords=('data', 'data'), ha='center', va='center')
    for y, bp in zip(np.arange(0, 16, 4) + 1.5, basepairs):
        axis.annotate(bp, xy=(-1, y), xycoords=('data', 'data'), ha='center', va='center')
    for x, bp in zip(np.arange(0, 16), basepairs * 4):
        axis.annotate(bp, xy=(x, -1), xycoords=('data', 'data'), ha='center', va='center')
    for y, bp in zip(np.arange(0, 16), basepairs * 4):
        axis.annotate(bp, xy=(16, y), xycoords=('data', 'data'), ha='center', va='center')

    axis.set_xticks(np.arange(-0.5, 16, 4)[1:-1], minor=False)
    axis.set_yticks(np.arange(-0.5, 16, 4)[1:-1], minor=False)
    axis.tick_params(which="major", top=False, labeltop=False, right=False, labelright=False,
                     bottom=False, labelbottom=False, left=False, labelleft=False)
    axis.grid(which='major', color="w", linestyle='-', linewidth=2)
    # axis.grid(which='major', color="k", linestyle='-', linewidth=2)

    axis.set_xticks(np.arange(-0.5,16.5), minor=True)
    axis.set_yticks(np.arange(-0.5,16.5), minor=True)
    axis.tick_params(which="minor", top=False, labeltop=False, right=False, labelright=False,
                     bottom=False, labelbottom=False, left=False, labelleft=False)
    axis.grid(which='minor', color="k", linestyle='-', linewidth=0)

    for spine in axis.spines.values():
        spine.set_visible(False)
        spine.set_lw(2)

    axis.annotate('Base pair 7-8', xy=(7.5, 17), xycoords=('data', 'data'), ha='center', va='center')
    axis.annotate('Base pair 1-2', xy=(-2, 7.5), xycoords=('data', 'data'), ha='center', va='center',
                  rotation='vertical')
    axis.annotate('Base pair 3-4', xy=(7.5, -2), xycoords=('data', 'data'), ha='center', va='center')
    axis.annotate('Base pair 5-6', xy=(17, 7.5), xycoords=('data', 'data'), ha='center', va='center',
                  rotation=270)

    axis.set_facecolor(axis_facecolor)
    axis.invert_yaxis()

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", "4%", pad="15%")
    figure.colorbar(scalar_mappable, cax=cax)

    # figure.colorbar(image, aspect=30)
    # cax = figure.axes[-1]
    if 'unit' in data.data.attrs and data.data.attrs['unit'] !='':
        unit_string = f' ({data.data.attrs["unit"]})'
    else:
        unit_string = ''
    cax.set_ylabel(name + unit_string)
    # cax.axes.ticklabel_format(scilimits=(0,0))
    for spine in cax.spines.values():
        spine.set_visible(False)

    title = 'basepaired_HJ_' + name

    fontsize = 8
    # title2 = title + ' - scaled_mean_basepaired'
    figure.suptitle(title, fontsize=fontsize)
    if save_path is not None:
        savefile_path = save_path / title
        figure.savefig(savefile_path.with_suffix('.png'))
        figure.savefig(savefile_path.with_suffix('.pdf'))


def format_sequence_subset(sequence_subset):
    ss = sequence_subset
    return np.array(
    [f'  ||  ',
    f'  {ss[3]}{ss[4]}  ',
    f'-{ss[2]}  {ss[5]}-',
    f'-{ss[1]}  {ss[6]}-',
    f'  {ss[0]}{ss[7]}  ',
    f'  ||  ']
    )

def format_sequence_subsets(sequence_subsets):
    spacer = np.array(['  ']*6)
    sequence_subsets_formatted = np.array([format_sequence_subset(ss) for ss in sequence_subsets])
    spacers = np.array([spacer]*len(sequence_subsets))

    final = char_add([np.char.add(*s) for s in zip(sequence_subsets_formatted, spacers)])
    final = '\n'.join(final)
    return final

# From itertools recipes
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def print_sequence_subsets(sequence_subsets, rows=None):
    if isinstance(sequence_subsets, str):
        sequence_subsets = [sequence_subsets]
    if rows is not None:
        sequence_subsets = batched(sequence_subsets, rows)
    else:
        sequence_subsets = [sequence_subsets]
    for sequence_subsets_row in sequence_subsets:
        print(format_sequence_subsets(sequence_subsets_row), '\n\n')


def char_add(strings):
    for i, s in enumerate(strings):
        if i == 0:
            final = s
        else:
            final = np.char.add(final, s)
    return final


def shorthand_notation(sequences):
    return [s[0::2] for s in sequences]


def purine_pyrimidine_sequence(sequences):
    translation_table = str.maketrans('AGCT', 'RRYY')
    return [s.translate(translation_table) for s in sequences]


def roll(sequences, n):
    for i in range(n):
        sequences = [s[1:] + s[0] for s in sequences]
    return sequences


def roll_multiple(sequences, n):
    ss = [sequences]
    for i in range(n):
        sequences = roll(sequences, 1)
        ss.append(sequences)
    return ss


def purine_pyrimidine_classes():
    return {1: 'RRRR', 2: 'YRRR', 3: 'YYRR', 4: 'YRYR', 5: 'YYYR', 6: 'YYYY'}


def purine_pyrimidine_classification(sequence_subsets, xarray=False):
    shorthand_purine_pyrimidine = shorthand_notation(purine_pyrimidine_sequence(sequence_subsets))
    shorthand_purine_pyrimidine_roll = np.array(roll_multiple(shorthand_purine_pyrimidine, 3)).T

    purine_pyrimidine_classification = np.zeros(len(shorthand_purine_pyrimidine_roll)).astype('int8')

    for index, pp_sequence in purine_pyrimidine_classes().items():
        purine_pyrimidine_classification[(shorthand_purine_pyrimidine_roll == pp_sequence).any(axis=1)] = index

    if xarray:
        purine_pyrimidine_classification = xr.DataArray(purine_pyrimidine_classification,
                                                        coords=dict(sequence_subset=sequence_subsets))

    return purine_pyrimidine_classification


def variant_score(variant):
    score = 0
    #     for i, l in enumerate(variant):
    #         if l == 'C':
    #             score += (i+1) * 1000
    #         elif l == 'T':
    #             score += (i+1) * 100
    #         elif l == 'G':
    #             score += (i+1) * 10
    #         elif l == 'A':
    #             score += (i+1) * 1
    order = {'C': 0, 'T': 1, 'A': 10, 'G': 11}
    for i, base in enumerate(variant):
        #         if i == 0 and base in ['T', 'C']:
        #             score -= 1
        if i == 0:
            score += order[base]
        #         elif i==3:
        #             score += order[::-1].index(base)
        elif i > 0:
            score += (order[previous_base] - order[base]) ** 3
        #         print(score)
        previous_base = base

    return score


def shorthand_sequence_rotationally_symmetric(sequence_subsets, xarray=False):
    invariant_shorthand = []
    for sequence_subset in sequence_subsets:
        variants = roll_multiple(shorthand_notation([sequence_subset]), 3)
        variant_scores = []
        for v in variants:
            #             print('test', variant_score(v[0]))
            variant_scores.append(variant_score(v[0]))
        selected_variant = variants[np.argmin(variant_scores)][0]
        invariant_shorthand.append(selected_variant)

    if xarray:
        invariant_shorthand = xr.DataArray(invariant_shorthand, coords=dict(sequence_subset=sequence_subsets))
    return invariant_shorthand

# def shorthand_sequence_rotationally_symmetric(sequence_subsets, xarray=False):
    # shorthand_purine_pyrimidine = shorthand_notation(purine_pyrimidine_sequence(sequence_subsets))
    # shorthand_purine_pyrimidine_roll = np.array(roll_multiple(shorthand_purine_pyrimidine, 3)).T
    # shorthand_purine_pyrimidine_roll

    # pps = np.array(list(purine_pyrimidine_classes().values()))

    # roll_indices = (shorthand_purine_pyrimidine_roll[:,:,None] == pps[None,None,:]).argmax(axis=-1).argmax(axis=-1)
    # shorthand_sequence_rotationally_symmetric = \
    #     np.array(roll_multiple(shorthand_notation(sequence_subsets), 3)).T[np.arange(len(roll_indices)),roll_indices]

    # if xarray:
    #     shorthand_sequence_rotationally_symmetric = xr.DataArray(shorthand_sequence_rotationally_symmetric,
    #                                                         coords=dict(sequence_subset=sequence_subsets))

    #     return shorthand_sequence_rotationally_symmetric