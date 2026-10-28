# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:44:52 2018

@author: ivoseverins
"""
# Required for interactive Matplotlib plotting with PySide2 in some PyCharm setups.
import sys
import json

import PySide2
sys.modules['PyQt5'] = sys.modules['PySide2']

import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
from PySide2.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QHeaderView, QTreeView
from PySide2.QtGui import QStandardItemModel, QStandardItem
from PySide2.QtCore import Qt, QModelIndex
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class Plot_ConfigurationModel(QStandardItemModel):
    """
    Custom model that only allows reordering of top-level rows.
    Disallows dropping into child items.
    """

    def flags(self, index):
        """Return item flags, allowing drag only for top-level items."""
        default_flags = super().flags(index)

        # Only top-level items can be dragged
        if not index.parent().isValid():# & index.column() == 0:
            return default_flags | Qt.ItemIsDragEnabled & ~Qt.ItemIsDropEnabled
        else:
            # Children cannot be dragged or accept drops
            return default_flags & ~Qt.ItemIsDropEnabled & ~Qt.ItemIsDragEnabled

    def supportedDropActions(self):
        """Return supported drop actions for drag-and-drop operations."""
        return Qt.MoveAction

    def dropMimeData(self, data, action, row, column, parent):
        """Handle drop events, allowing only root-level drops."""
        # Only allow drops at root level
        if parent.isValid():
            return False

        self.blockSignals(True)
        result = super().dropMimeData(data, action, row, 0, parent)
        self.blockSignals(False)
        return result

class Plot_Configuration(QWidget):
    """Configuration widget for trace plots.

    Provides a tree-view UI to enable/disable trace variables, set plot ranges,
    colors and per-illumination options. Updates the main canvas when
    settings change.
    """
    def __init__(self, parent, canvas, initial_plot_settings=None):

        super().__init__(parent=parent)
        self.canvas = canvas
        self.view = QTreeView()
        self.model = Plot_ConfigurationModel()
        self.model.setHorizontalHeaderLabels(["Variable", ""])

        self.model.itemChanged.connect(self._on_item_change)
        self.model.rowsRemoved.connect(self._on_rows_changed)

        self.view.setModel(self.model)
        self.view.setDragEnabled(True)
        self.view.setAcceptDrops(True)
        self.view.setDropIndicatorShown(True)
        self.view.setDefaultDropAction(Qt.MoveAction)
        self.view.setDragDropMode(QTreeView.InternalMove)

        self.view.setAlternatingRowColors(True)
        self.view.setRootIsDecorated(True)
        self.view.header().setStretchLastSection(True)
        self.view.header().setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents
        )

        self._dataset = None
        self._trace_variables = []

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

        self.plot_settings = initial_plot_settings

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset
        self._trace_variables_dataset = [name for name, da in self._dataset.data_vars.items() if
                                         da.dims and da.dims[0] == "molecule" and da.dims[-1] == "frame"]

        self._add_missing_plot_settings_from_dataset()

        self._add_plot_settings_to_model()

        self._enable_dataset_variables()
        self.canvas.plot_settings = self.plot_settings

        self.parent().setFocus()

    def _enable_trace_variable(self, variable):
        """Enable the specified trace variable in the model."""
        self.model.blockSignals(True)
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item.text() == variable:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
        self.model.blockSignals(False)

    def _disable_trace_variable(self, variable):
        """Disable the specified trace variable in the model."""
        self.model.blockSignals(True)
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item.text() == variable:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        self.model.blockSignals(False)

    def _disable_all_rows(self):
        """Disable all rows in the model."""
        self.model.blockSignals(True)
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        self.model.blockSignals(False)

    def _enable_dataset_variables(self):
        """Enable only the trace variables available in the current dataset."""
        self._disable_all_rows()
        for var in self._trace_variables_dataset:
            self._enable_trace_variable(var)

    def _add_missing_plot_settings_from_dataset(self):
        """Add default plot settings for variables not yet configured."""
        #TODO: variables live on in plotconfig even when cleared
        plot_settings = self.plot_settings
        for var in set(self._trace_variables_dataset).union(set(plot_settings.keys())):
            if var not in plot_settings:
                if 'plot_settings' in self.dataset[var].attrs:
                    plot_settings[var] = json.loads(self.dataset[var].attrs['plot_settings'])
                else:
                    plot_settings[var] = {}

            if 'active' not in plot_settings[var]:
                if var in ['intensity', 'FRET']:
                    plot_settings[var]['active'] = True
                else:
                    plot_settings[var]['active'] = False
            if 'n_bins' not in plot_settings[var]:
                plot_settings[var]['n_bins'] = 200

            # Plot range text
            if 'plot_range' not in plot_settings[var]:
                if 'FRET' in var:
                    plot_settings[var]['plot_range'] = (-0.05, 1.05)
                    plot_settings[var]['n_bins'] = 200
                elif 'classification' in var:
                    plot_settings[var]['plot_range'] = (self.dataset[var].min().round().item()-0.5,
                                                        self.dataset[var].max().round().item()+0.5)
                else:
                    plot_settings[var]['plot_range'] = (self.dataset[var].min().round().item(),
                                                        self.dataset[var].max().round().item())
                if 'n_bins' not in plot_settings[var]:

                        plot_settings[var]['n_bins'] = 200

            # Color column
            if 'color' not in plot_settings[var]:
                if 'channel' in self.dataset[var].dims:
                    plot_settings[var]['color'] = (('g','r') + ('k',)*10)[0:len(self.dataset.channel)]
                elif 'FRET' in var:
                    plot_settings[var]['color'] = ('b',)
                else:
                    plot_settings[var]['color'] = ('k',)
            elif isinstance(plot_settings[var]['color'], str):
                plot_settings[var]['color'] = (plot_settings[var]['color'],)

            if 'axis' not in plot_settings[var]:
                plot_settings[var]['axis'] = var

            if 'secondary' not in plot_settings[var]:
                plot_settings[var]['secondary'] = False

            if 'intensity' in var or var == 'FRET':
                illuminations = np.unique(self.dataset.illumination)
                if len(illuminations) > 1:
                    plot_settings[var]['split_illuminations'] = False
                    for illumination in illuminations:
                        plot_settings[var][f'illumination_{illumination}'] = True



        # Normalize order values
        ordered_variables = sorted(plot_settings.keys(),
                              key=lambda v: plot_settings[v].get('order', 1000))
        plot_settings_ordered = {}
        for i, plot_variable in enumerate(ordered_variables):
            plot_settings[plot_variable]['order'] = i
            plot_settings_ordered[plot_variable] = plot_settings[plot_variable]

        self.plot_settings = plot_settings_ordered

    def _apply_row_spanning_for_plot_variables(self):
        """Apply row spanning for plot variables to display them properly."""
        for row in range(self.model.rowCount()):
            self.view.setFirstColumnSpanned(row, QModelIndex(), True)
        self.view.doItemsLayout()  # Important to refresh treeview, otherwise it is not stay up to date with the model.

    def _add_plot_settings_to_model(self):
        """Add all plot settings to the model tree view."""
        for plot_variable, plot_settings_of_variable in self.plot_settings.items():
            self._add_plot_settings_of_variable_to_model(plot_variable, plot_settings_of_variable)

        self._apply_row_spanning_for_plot_variables()

    def _get_or_create_name_item(self, plot_variable):
        """Get existing plot variable item or create a new one."""
        # Look for existing item
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item and item.text() == plot_variable:
                return item

        # Not found → create new
        self._trace_variables.append(plot_variable)
        name_item = QStandardItem(plot_variable)
        name_item.setEditable(False)
        name_item.setCheckable(True)
        name_item.setDropEnabled(False)
        empty_item = QStandardItem()
        empty_item.setDropEnabled(False)
        self.model.appendRow([name_item, empty_item])
        return name_item

    def _add_plot_settings_of_variable_to_model(self, plot_variable, plot_settings):
        """Add configuration items for a specific plot variable to the model."""
        self.model.blockSignals(True)

        name_item = self._get_or_create_name_item(plot_variable)

        if plot_settings['active']:
            name_item.setCheckState(Qt.Checked)
        else:
            name_item.setCheckState(Qt.Unchecked)

        # Find current settings for variable
        current_settings = []
        for row in range(name_item.rowCount()):
            current_settings.append(name_item.child(row, 1).data(Qt.UserRole))

        if 'plot_range' not in current_settings:
            plot_range = plot_settings['plot_range']

            plot_range_low_text_item = QStandardItem("Y min")
            plot_range_low_text_item.setEditable(False)
            plot_range_low_item = QStandardItem(str(plot_range[0]))
            plot_range_low_item.setEditable(True)
            plot_range_low_item.setData('plot_range', Qt.UserRole)
            name_item.appendRow([plot_range_low_text_item, plot_range_low_item])

            plot_range_high_text_item = QStandardItem("Y max")
            plot_range_high_text_item.setEditable(False)
            plot_range_high_item = QStandardItem(str(plot_range[1]))
            plot_range_high_item.setEditable(True)
            plot_range_high_item.setData('plot_range', Qt.UserRole)

            name_item.appendRow([plot_range_high_text_item, plot_range_high_item])

        if 'n_bins' not in current_settings:
            bins = plot_settings['n_bins']

            plot_range_bins_text_item = QStandardItem("n_bins")
            plot_range_bins_text_item.setEditable(False)

            plot_range_bins_item = QStandardItem(str(bins))
            plot_range_bins_item.setEditable(True)
            plot_range_bins_item.setData('n_bins', Qt.UserRole)

            name_item.appendRow([plot_range_bins_text_item, plot_range_bins_item])


        if 'color' not in current_settings:
            color_text_item = QStandardItem("Color(s)")
            color_text_item.setEditable(False)

            color_string = ', '.join(plot_settings['color'])
            color_item = QStandardItem(color_string)
            color_item.setEditable(True)
            color_item.setData('color', Qt.UserRole)

            name_item.appendRow([color_text_item, color_item])

        if 'axis' not in current_settings:
            axis_text_item = QStandardItem("Axis")
            axis_text_item.setEditable(False)

            axis_item = QStandardItem(plot_settings['axis'])
            axis_item.setEditable(True)
            axis_item.setData('axis', Qt.UserRole)

            name_item.appendRow([axis_text_item, axis_item])

        if 'secondary' not in current_settings:
            secondary_text_item = QStandardItem("Secondary axis")
            secondary_text_item.setEditable(False)

            secondary_checkbox = QStandardItem()
            secondary_checkbox.setCheckable(True)
            if plot_settings['secondary']:
                secondary_checkbox.setCheckState(Qt.Checked)
            else:
                secondary_checkbox.setCheckState(Qt.Unchecked)
            secondary_checkbox.setEditable(False)
            secondary_checkbox.setData('secondary', Qt.UserRole)

            name_item.appendRow([secondary_text_item, secondary_checkbox])

        if 'split_illuminations' in plot_settings:
            if 'split_illuminations' not in current_settings:
                split_illuminations_text_item = QStandardItem("Split illuminations")
                split_illuminations_text_item.setEditable(False)

                split_illuminations_checkbox = QStandardItem()
                split_illuminations_checkbox.setCheckable(True)
                split_illuminations_checkbox.setCheckable(True)
                if plot_settings['split_illuminations']:
                    split_illuminations_checkbox.setCheckState(Qt.Checked)
                else:
                    split_illuminations_checkbox.setCheckState(Qt.Unchecked)
                split_illuminations_checkbox.setEditable(False)
                split_illuminations_checkbox.setData('split_illuminations', Qt.UserRole)

                name_item.appendRow([split_illuminations_text_item, split_illuminations_checkbox])

            for illumination in np.unique(self.dataset.illumination):
                if f'illumination_{illumination}' not in current_settings:
                    illumination_text_item = QStandardItem(f"Illumination {illumination}")
                    illumination_text_item.setEditable(False)

                    illumination_checkbox = QStandardItem()
                    illumination_checkbox.setCheckable(True)
                    if plot_settings[f'illumination_{illumination}']:
                        illumination_checkbox.setCheckState(Qt.Checked)
                    else:
                        illumination_checkbox.setCheckState(Qt.Unchecked)
                    illumination_checkbox.setEditable(False)
                    illumination_checkbox.setData(f'illumination_{illumination}', Qt.UserRole)

                    name_item.appendRow([illumination_text_item, illumination_checkbox])

        self.model.blockSignals(False)

    def _on_item_change(self, item):
        """Handle changes to plot configuration items and update settings."""
        if item.column() == 0:
            # self._update_order_from_model()
            variable_name = item.model().item(item.row(), 0).text()
            active = bool(item.checkState())
            if active is not self.plot_settings[variable_name]['active']:
                self.plot_settings[variable_name]['active'] = active
                self.canvas.plot_settings = self.plot_settings
            # self.parent().molecule = self.parent().molecule
        elif item.column() == 1:# and item.text() is not '':
            variable_name = item.model().item(item.parent().row(), 0).text()
            if item.data(Qt.UserRole)  == 'plot_range':
                plot_range = tuple(float(item.parent().child(i,1).text()) for i in [0,1])
                self.plot_settings[variable_name][item.data(Qt.UserRole)] = plot_range
                self.canvas.set_plot_range(variable_name, plot_range)
                # self.parent().molecule = self.parent().molecule
            elif item.data(Qt.UserRole) == 'color':
                color = tuple(item.text().replace(' ','').split(','))
                self.plot_settings[variable_name][item.data(Qt.UserRole)] = color
                self.canvas.set_plot_color(variable_name, color)
            elif item.data(Qt.UserRole) == 'axis':
                self.plot_settings[variable_name][item.data(Qt.UserRole)] = item.text()
                self.canvas.plot_settings = self.plot_settings
            elif item.data(Qt.UserRole) == 'secondary':
                secondary = bool(item.checkState())
                self.plot_settings[variable_name][item.data(Qt.UserRole)] = secondary
                self.canvas.plot_settings = self.plot_settings
            elif item.data(Qt.UserRole) == 'split_illuminations':
                split_illuminations = bool(item.checkState())
                self.plot_settings[variable_name][item.data(Qt.UserRole)] = split_illuminations
                self.canvas.plot_settings = self.plot_settings
            elif item.data(Qt.UserRole).startswith('illumination'):
                illumination = bool(item.checkState())
                self.plot_settings[variable_name][item.data(Qt.UserRole)] = illumination
                self.canvas.plot_settings = self.plot_settings
            elif item.data(Qt.UserRole)  == 'n_bins':
                n_bins = int(item.parent().child(2,1).text())
                self.plot_settings[variable_name][item.data(Qt.UserRole)] = n_bins
                self.canvas.plot_settings = self.plot_settings

        self.parent().setFocus()

    def _on_rows_changed(self):
        """Handle reordering of rows after drag-and-drop operations."""
        self._apply_row_spanning_for_plot_variables()
        self._update_order_from_model()

    def _update_order_from_model(self):
        """
        Update plot_settings order after rows are reordered by drag & drop.
        """
        plot_settings_new = {}
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            var_name = item.text()
            if var_name in self.plot_settings:
                plot_settings_new[var_name] = self.plot_settings[var_name]
                plot_settings_new[var_name]['order'] = row

        self.plot_settings = plot_settings_new
        self.canvas.plot_settings = self.plot_settings

        self.parent().setFocus()

class HistogramPlotWindow(QWidget):
    """Interactive window for plotting histograms of selected molecules."""

    def __init__(
            self,
            file=None,
            plot_settings=None,
            width=8,
            height=None,
            file_path=None,
            parent=None,
            show=True,
            **kwargs
    ):
        super().__init__(parent=parent)





        if plot_settings is None:
            plot_settings = {
                'intensity': {
                    'active': True,
                    'n_bins': 200,
                    'color': ('g', 'r')
                },
                'FRET': {
                    'active': True,
                    'plot_range': (-0.05, 1.05),
                    'n_bins': 200,
                    'color': ('b',)
                }
            }
        #force write FRET settings (to avoid auto-scaling on outliers)
        plot_settings['FRET']['plot_range'] = (-0.05, 1.05)
        plot_settings['FRET']['n_bins'] = 200
        #bins=y_range = np.linspace(plot_settings['FRET']['plot_range'][0],
        #                              plot_settings['FRET']['plot_range'][1],
        #                              plot_settings['FRET']['n_bins'] + 1)
        #plot_settings['FRET']['n_bins'] = bins

        # Store the initial file for assignment after the UI is constructed.
        self._file = None
        self._dataset = None
        self.file_path = file_path

        # 0 = unselected
        # 1 = all
        # 2 = selected
        self._selection_state = 1

        # Create canvas
        self.canvas = HistogramPlotCanvas(
            parent=self,
            width=width,
            height=height or 7,
            dpi=100
        )

        # Selection controls
        layout_bar = QHBoxLayout()
        layout_bar.addWidget(QLabel('N_molecules:'))

        self.number_of_molecules_label = QLabel('0')
        self.number_of_molecules_label.setFixedWidth(70)
        layout_bar.addWidget(self.number_of_molecules_label)

        self.selected_molecules_checkbox = QCheckBox()
        self.selected_molecules_checkbox.setTristate(True)
        self.selected_molecules_checkbox.setCheckState(
            Qt.PartiallyChecked
        )
        self.selected_molecules_checkbox.stateChanged.connect(
            self.on_selected_molecules_checkbox_state_change
        )
        self.selected_molecules_checkbox.setFocusPolicy(Qt.NoFocus)

        layout_bar.addWidget(QLabel('Selected'))
        layout_bar.addWidget(self.selected_molecules_checkbox)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(layout_bar)
        layout.addWidget(self.canvas)

        # Re-use the existing configuration panel
        self.plot_configuration = Plot_Configuration(
            parent=self,
            canvas=self.canvas,
            initial_plot_settings=plot_settings
        )
        self.plot_configuration.setMinimumWidth(250)

        layout_main = QHBoxLayout()
        layout_main.addLayout(layout, stretch=4)
        layout_main.addWidget(
            self.plot_configuration,
            stretch=1
        )

        self.setLayout(layout_main)

        # Give canvas access to the window
        self.canvas.parent_window = self

        # Set the initial file only after canvas and plot_configuration exist.
        # if file is not None:
        #     self.set_file(file)
        # else:
        #     self.setDisabled(True)
        self.file = file

        if show:
            self.show()

    # ------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------

    @property
    def selection_state(self):
        return self._selection_state

    @selection_state.setter
    def selection_state(self, value):
        self._selection_state = value
        # self.set_selection()
        if self.file is not None:
            self.canvas.update_histograms()

    def on_selected_molecules_checkbox_state_change(
            self,
            selection_state
    ):
        self.selection_state = selection_state
        self.selected_molecules_checkbox.clearFocus()
    #
    # def set_selection(self):
    #     """Determine which molecules are included in the histogram."""
    #     if  self.file is None:
    #         self.molecule_indices = []
    #         self.number_of_molecules_label.setText('0')
    #         return
    #
    #     selected = self.file.selected
    #
    #     if self.selection_state == 0:
    #         # Unselected molecules
    #         self.molecule_indices = (
    #             selected.molecule
    #             .sel(molecule=~selected.values)
    #             .values
    #         )
    #
    #     elif self.selection_state == 1:
    #         # All molecules
    #         self.molecule_indices = selected.molecule.values
    #
    #     elif self.selection_state == 2:
    #         # Selected molecules
    #         self.molecule_indices = (
    #             selected.molecule
    #             .sel(molecule=selected.values)
    #             .values
    #         )
    #
    #     else:
    #         raise ValueError(
    #             f'Unknown selection_state {self.selection_state}'
    #         )
    #
    #     self.number_of_molecules_label.setText(
    #         str(len(self.molecule_indices))
    #     )

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        self._file = file
        if file is None:
            self.dataset = None
            self.canvas.figure.clear()
            self.canvas.histogram_axes = {}
            self.canvas.draw()
            self.setDisabled(True)
        else:
            self.dataset = file.dataset
            self.plot_configuration.dataset = file.dataset
            self.canvas.plot_settings = (
                self.plot_configuration.plot_settings
            )

            self.canvas.file = file
            self.canvas.update_histograms()

            # """Assign a file and refresh the canvas."""
            # self.file = file
            # self.canvas.file = file
            #
            # if self.dataset is not None:
            #
            # else:

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        if value is not None and (hasattr(value, 'frame') or hasattr(value, 'time')):
            self._dataset = value
            self._dataset['selected'] = self._dataset.selected.astype('bool')
            if 'intensity' in self._dataset:
                self._dataset['intensity_total'] = self._dataset['intensity'].sum('channel')

            self.plot_configuration.dataset = self._dataset
            #self.set_selection()
            self.setDisabled(False)
        else:
            self._dataset = None
            self.setDisabled(True)
        self.molecule_index = 0

class HistogramPlotCanvas(FigureCanvasQTAgg):
    """Canvas for histograms of a selection of molecules."""

    def __init__(
            self,
            parent=None,
            width=8,
            height=7,
            dpi=100
    ):
        self.figure = Figure(
            figsize=(width, height),
            dpi=dpi,
            tight_layout=True
        )

        super().__init__(self.figure)

        self.parent_window = parent
        self.file = None
        self._plot_settings = {}

        self.histogram_axes = {}
        self.selection_state = None

    # ------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------

    @property
    def plot_settings(self):
        return self._plot_settings

    @plot_settings.setter
    def plot_settings(self, value):
        self._plot_settings = {
            variable: settings
            for variable, settings in value.items()
            if settings.get('active', False)
        }

        self.init_plots()

    @property
    def plot_variables(self):
        return list(self.plot_settings.keys())

    # ------------------------------------------------------------
    # Plot initialization
    # ------------------------------------------------------------

    def init_plots(self):
        """Create one histogram axis for each active variable."""

        self.figure.clf()

        self.histogram_axes = {}

        variables = self.plot_variables

        if not variables:
            self.draw()
            return

        axes = self.figure.subplots(
            len(variables),
            1,
            squeeze=False
        ).flatten()

        for axis, variable in zip(axes, variables):

            self.histogram_axes[variable] = axis

            plot_settings = self.plot_settings[variable]

            if 'plot_range' in plot_settings:
                axis.set_xlim(
                    plot_settings['plot_range']
                )

            axis.set_xlabel(variable)
            axis.set_ylabel('Count')
            self.update_histograms()

    # ------------------------------------------------------------
    # Histogram drawing
    # ------------------------------------------------------------

    def update_histograms(self):
        """Redraw histograms using the current molecule selection."""

        if self.file is None :
            return
        fret = getattr(self.file.dataset, "FRET", None)
        if fret is None:
            return
        if not self.histogram_axes:
            return

        for variable, axis in self.histogram_axes.items():

            plot_settings = self.plot_settings[variable]

            axis.clear()

            if 'n_bins' in plot_settings and 'plot_range' in plot_settings:
                bin_axis = np.linspace(plot_settings['plot_range'][0],
                                         plot_settings['plot_range'][1],
                                         plot_settings['n_bins'] + 1)
            else:
                bin_axis=200

            self.file.show_histogram(
                variable=variable,
                axis=axis,
                bins=bin_axis
                #selected=self.selection_state  TODO: change in show_hist
            )

            if 'plot_range' in plot_settings:
                axis.set_xlim(
                    plot_settings['plot_range']
                )

            if 'color' in plot_settings:
                colors = plot_settings['color']

                # show_histogram may create several artists
                # depending on the variable/channel.
                for i, artist in enumerate(axis.patches):
                    if colors:
                        artist.set_facecolor(
                            colors[i % len(colors)]
                        )

            axis.set_xlabel(variable)
            axis.set_ylabel('Count')

        self.draw()

    # ------------------------------------------------------------
    # Configuration callbacks
    # ------------------------------------------------------------

    def set_plot_range(self, plot_variable, plot_range):

        self.update_histograms()

        if plot_variable not in self.histogram_axes:
            return

        axis = self.histogram_axes[plot_variable]

        axis.set_xlim(
            plot_range[0],
            plot_range[1]
        )

        self.draw()

    def set_plot_color(self, plot_variable, colors):

        if plot_variable not in self.histogram_axes:
            return

        axis = self.histogram_axes[plot_variable]

        for i, artist in enumerate(axis.patches):
            artist.set_facecolor(
                colors[i % len(colors)]
            )

        self.draw()

if __name__ == "__main__":

    import papylio as pp
    exp = pp.Experiment(r'C:\Users\jkerssemakers\OneDrive - Delft University of Technology\Documents\GitHub\Papylio example dataset_flat')
    file = exp.files[1]
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)

    frame = HistogramPlotWindow(file=file)
    app.exec_()
