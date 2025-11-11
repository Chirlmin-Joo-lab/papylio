import numpy as np
import matplotlib.pyplot as plt
from objectlist import ObjectList
from scipy import stats
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

from objectlist import ObjectList
from matchpoint.core import distance_threshold_from_number_of_matches


# TODO: Make sure that the collection can only contain MatchPoint objects
class MappingCollection(ObjectList):
    dimension_number = {0: 0, 1: 1, 'x': 0, 'y': 1}
    dimension_string = {0: 'x', 1: 'y', 'x': 'x', 'y': 'y'}

    @property
    def translations(self):
        return np.vstack(self.translation)

    @property
    def unit_string(self):

        if self[0].destination_unit is not None and len(np.unique(self.destination_unit)) == 1:
            destination_unit_string = ' (' + self[0].destination_unit + ')'
        else:
            destination_unit_string = ''

        unit_string = {'translation': destination_unit_string, 'rotation': ' (rad)', 'scale': '',
                       'shear': ' (rad)'}

        return unit_string

    def plot_parameter(self, parameter, dimension=0, label_name='Label', indices=None, figure=None, save=False, **kwargs):
        data = np.vstack(getattr(self, parameter))[:, self.dimension_number[dimension]]

        all_indices = range(len(self))
        if indices is None:
            indices = all_indices

        if figure is None:
            figure, axis = plt.subplots()
        else:
            axis = figure.gca()

        axis.scatter(indices, data[indices], **kwargs)
        axis.set_xlabel(label_name)
        axis.set_ylabel(parameter.capitalize() + ' ' + self.dimension_string[dimension] + self.unit_string[parameter])
        axis.set_xticks(all_indices)
        axis.set_xticklabels(self.label, rotation=45, ha="right")

        figure.tight_layout()
        if save:
            figure.savefig(self[0].save_path.joinpath(f'{parameter.capitalize()}-{self.dimension_string[dimension]}.png'))

        return figure, axis

    def scatter_parameters(self, parameter_x, parameter_y, dimension_x=0, dimension_y=0, indices=None, figure=None, save=False, **kwargs):
        data_x = np.vstack(getattr(self, parameter_x))[:, self.dimension_number[dimension_x]]
        data_y = np.vstack(getattr(self, parameter_y))[:, self.dimension_number[dimension_y]]

        if indices is None:
            indices = np.arange(len(data_x))

        if figure is None:
            figure, axis = plt.subplots()
        else:
            axis = figure.gca()

        axis.scatter(data_x[indices], data_y[indices], **kwargs)
        axis.set_xlabel(parameter_x.capitalize() + ' ' + self.dimension_string[dimension_x] + self.unit_string[parameter_x])
        axis.set_ylabel(parameter_y.capitalize() + ' ' + self.dimension_string[dimension_y] + self.unit_string[parameter_y])
        for label, x, y in zip(self.label[indices], data_x[indices], data_y[indices]):
            axis.annotate(label, (x, y))

        figure.tight_layout()
        if save:
            figure.savefig(self[0].save_path.joinpath(f'{parameter_y.capitalize()}-{self.dimension_string[dimension_y]}_vs_'
                                                      f'{parameter_x}-{self.dimension_string[dimension_x]}.png'))

        return figure, axis

    # def plot_translations(self, dimension=None, save=False, **kwargs):
    #     unit_string = self.destination_unit_string
    #
    #     fig, ax = plt.subplots()
    #     if dimension is None:
    #         ax.scatter(*self.translations.T)
    #         ax.set_xlabel('Translation x' + unit_string)
    #         ax.set_ylabel('Translation y' + unit_string)
    #         for label, translation in zip(self.label, self.translations):
    #             ax.annotate(label, translation)
    #     elif dimension in ['x', 0]:
    #         ax.scatter(range(len(self)), self.translations[:, 0])
    #         ax.set_xlabel('Label')
    #         ax.set_ylabel('Translation x' + unit_string)
    #         ax.set_xticks(range(len(self)))
    #         ax.set_xticklabels(self.label, rotation=45, ha="right")
    #     elif dimension in ['y', 1]:
    #         ax.scatter(range(len(self)), self.translations[:,1])
    #         ax.set_xlabel('Label')
    #         ax.set_ylabel('Translation y' + unit_string)
    #         ax.set_xticks(range(len(self)))
    #         ax.set_xticklabels(self.label, rotation=45, ha="right")
    #     else:
    #         raise ValueError("Dimension can be 'x', 'y' or None.")
    #
    #     fig.tight_layout()
    #     if save:
    #         fig.savefig(self[0].save_path.joinpath(f'Tile_translations.png'), **kwargs)
    #
    #     return fig, ax

    def fit_translations(self, indices, save=False):
        all_indices = np.arange(len(self))
        if indices is None:
            indices = all_indices
        res = stats.linregress(*self.translations[indices].T)
        figure, axis = self.scatter_parameters('translation', 'translation', 'x', 'y')
        self.scatter_parameters('translation', 'translation', 'x', 'y', indices=indices, figure=figure, c='orange')
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        x = np.array(xlim)
        axis.plot(x, res.slope * x + res.intercept, c='orange')
        axis.set_xlim(xlim), axis.set_ylim(ylim)
        figure.tight_layout()
        if save:
            figure.savefig(self[0].save_path.joinpath(f'Translations_fit.png'))

        # Plot and fit x translation vs tile number
        res_x = stats.linregress(indices, self.translations[indices, 0])
        figure, axis = self.plot_parameter('translation', dimension='x')
        self.plot_parameter('translation', dimension='x', indices=indices, figure=figure, c='orange')
        axis.plot(all_indices, res_x.slope * all_indices + res_x.intercept)
        if save:
            figure.savefig(self[0].save_path.joinpath(f'Translation-x_vs_index_fit.png'))

        # Plot and fit y translation vs tile number
        res_y = stats.linregress(indices, self.translations[indices, 1])
        figure, axis = self.plot_parameter('translation', dimension='y')
        self.plot_parameter('translation', dimension='y', indices=indices, figure=figure, c='orange')
        axis.plot(all_indices, res_y.slope * all_indices + res_y.intercept)
        if save:
            figure.savefig(self[0].save_path.joinpath(f'Translation-y_vs_index_fit.png'))

        return res_x, res_y

    def estimate_translations(self, **kwargs):
        res_x, res_y = self.fit_translations(**kwargs)

        for i, mapping in enumerate(self):
            mapping.initial_transformation = type(mapping.transformation)(matrix=mapping.transformation.params) # To make a copy
            new_translation = [res_x.slope * i + res_x.intercept, res_y.slope * i + res_y.intercept]
            mapping.transformation.params[0:2, 2] = new_translation
            mapping.calculate_inverse_transformation()

    # TODO: Extract parameters / statistics
    # columns = pd.MultiIndex.from_product([['Single-molecule', 'Sequencing'], ['Total', 'Matched', 'Fraction']])
    # df = pd.DataFrame(columns=columns)
    # for tile, mapping in tile_mappings_calculated.items():
    #     mapping.source_distance_threshold = 5  # um
    #     pm = mapping.number_of_matched_points
    #     fs, fd = mapping.fraction_of_points_matched('source')
    #     df.loc[tile, :] = [len(mapping.source_cropped), pm, fs, len(mapping.destination), pm, fd]
    # df.index.name = 'Tile'
    # df.to_excel(analysis_path.joinpath(f'Fraction_of_points_matched_source_threshold_5um.xlsx'))

    def find_distance_threshold(self, method='single_match_optimization', **kwargs):
        if method == 'single_match_optimization':
            self.single_match_optimization(**kwargs)
        else:
            raise ValueError('Unknown method')

    def single_matches_over_radius(self, maximum_radius=20, number_of_steps=100):
        radii = np.linspace(0, maximum_radius, number_of_steps)
        number_of_pairs = self.number_of_single_matches_for_radii(radii)
        number_of_pairs_summed = np.vstack(number_of_pairs).sum(axis=0)
        return radii, number_of_pairs_summed

    def single_match_optimization(self, maximum_radius=20, number_of_steps=100, plot=True):
        radii, number_of_pairs = self.single_matches_over_radius(maximum_radius, number_of_steps)
        self.destination_distance_threshold = distance_threshold_from_number_of_matches(radii, number_of_pairs, plot=plot)
