# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:44:52 2018

@author: ivoseverins
"""
from typing import Optional

# import wx
# import wx.lib.mixins.inspection as wit
# import sys
# print('PyQt5', sys.modules.get("PyQt5.QtCore"))
# print('PySide2', sys.modules.get("PySide2.QtCore"))

###################################################
## To enable interactive plotting with PySide2 in PyCharm 2022.3
import PySide2
import sys
sys.modules['PyQt5'] = sys.modules['PySide2']
import matplotlib
matplotlib.use('Qt5Agg')
###################################################

import matplotlib.pyplot as plt

# from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
# from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
# from matplotlib.backends.backend_qtagg import FigureCanvas
# from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
from pathlib2 import Path

from PySide2.QtWidgets import (QMainWindow, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QCheckBox, QLabel,
                               QTableWidget, QTableWidgetItem, QHeaderView, QTreeView, QStyledItemDelegate,
                               QAbstractItemView)
from PySide2.QtGui import QStandardItemModel, QStandardItem, QColor
from PySide2.QtGui import QKeySequence, QCloseEvent, QDragMoveEvent
from PySide2.QtCore import Qt, QModelIndex, QTimer

import sys
import time

import numpy as np

import netCDF4
import json

# from matplotlib.backends.qt_compat import QtWidgets
from PySide2 import QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure



class TracePlotWindow(QWidget):
    def __init__(self, dataset=None,
                 plot_settings=None,
                 width=14, height=None, dataset_path=None, save_path=None, parent=None,
                 show=True, split_illuminations=False, **kwargs):

        if plot_settings is None:
            plot_settings = {'intensity': {'active': True, 'plot_range': (0, 10000), 'color': ('g', 'r')},
                             'FRET': {'active': True, 'plot_range': (-0.05, 1.05), 'color': ('b')}}

        # To accomodate old arguments
        if 'plot_variables' in kwargs:
            plot_settings = {}
            for plot_variable, ylim, color in zip(kwargs['plot_variables'], kwargs['ylims'], kwargs['colours']):
                plot_settings[plot_variable] = dict(active=True, plot_range=ylim, color=color)
            import warnings
            warnings.simplefilter("always", DeprecationWarning)

            warnings.warn(
                "Use of `plot_variables`, `ylims` and `colours` arguments is depricated,"
                "use `plot_settings` instead, "
                "e.g. `plot_settings = {'intensity': {'active': True, 'plot_range': (0, 10000), 'color': ('g', 'r')}, "
                "'FRET': {'active': True, 'plot_range': (-0.05, 1.05), 'color': ('b')}}`",
                DeprecationWarning,
                stacklevel=2,
            )

        self.split_illuminations = split_illuminations

        if height is None:
            height = max(len(plot_settings) * 3.5, 9)

        from papylio.experiment import get_QApplication
        #TODO: Use selection only if it is present.
        app = get_QApplication()

        super().__init__()

        self.parent = parent

        self.setWindowTitle("Traces")

        if save_path is None:
            self.save_path = save_path
        else:
            self.save_path = Path(save_path)

        # self._dataset = dataset

        self.canvas = TracePlotCanvas(self, width=width, height=height, dpi=100)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()

        layout_bar = QHBoxLayout()
        layout_bar.addWidget(toolbar, 0.5)

        self.molecule_index_field = QLineEdit()
        self.molecule_index_field.setFixedWidth(70)

        layout_bar.addWidget(self.molecule_index_field, 0.05)
        layout_bar.addWidget(QLabel(' out of '), 0.05)
        self.number_of_molecules_label = QLabel('0')
        self.number_of_molecules_label.setFixedWidth(70)
        layout_bar.addWidget(self.number_of_molecules_label, 0.15)
        self._selection_state = 1
        self.selected_molecules_checkbox = QCheckBox()
        self.selected_molecules_checkbox.setTristate(True)
        self.selected_molecules_checkbox.setCheckState(Qt.PartiallyChecked)
        self.selected_molecules_checkbox.stateChanged.connect(self.on_selected_molecules_checkbox_state_change)
        self.selected_molecules_checkbox.setFocusPolicy(Qt.NoFocus)


        layout_bar.addWidget(QLabel('Selected'),0.1)
        layout_bar.addWidget(self.selected_molecules_checkbox, 0.15)

        self.molecule_index_field.returnPressed.connect(self.set_molecule_index_from_molecule_index_field)
        self.molecule_index_field.returnPressed.connect(self.deactivate_line_edit)


        layout.addLayout(layout_bar)
        layout.addWidget(self.canvas)

        # self.setLayout(layout)
        # Create a placeholder widget to hold our toolbar and canvas.
        # widget = QWidget()
        # widget.setLayout(layout)
        # self.setCentralWidget(widget)

        self.plot_configuration = PlotConfiguration(parent=self, canvas=self.canvas, initial_plot_settings=plot_settings)
        self.plot_configuration.setMinimumWidth(250)

        layout_main = QHBoxLayout()
        layout_main.addLayout(layout, stretch=4)
        layout_main.addWidget(self.plot_configuration, stretch=1)
        self.setLayout(layout_main)

        self.dataset_path = dataset_path
        if self.dataset_path is not None:
            self.dataset_path = Path(self.dataset_path)

        self.dataset = dataset

        if show:
            self.show()
            app.exec_()

        self.setFocus()

    def closeEvent(self, event: QCloseEvent):
        self.save_plot_settings()
        self.save_selection()

    def save_plot_settings(self):
        if self.dataset_path is not None:
            with netCDF4.Dataset(self.dataset_path, "a") as nc:
                for variable, plot_settings in self.plot_configuration.plot_settings.items():
                    nc[variable].setncattr("plot_settings", json.dumps(plot_settings))

    def save_selection(self):
        if self.dataset_path is not None:
            self.dataset.selected.astype('bool').to_netcdf(self.dataset_path, engine='netcdf4', mode='a')

    def deactivate_line_edit(self):
            self.molecule_index_field.clearFocus()  # Clear the focus from the line edit

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
            self.set_selection()
            self.setDisabled(False)
        else:
            self._dataset = None
            self.setDisabled(True)
        self.molecule_index = 0

    @property
    def selection_state(self):
        return self._selection_state

    @selection_state.setter
    def selection_state(self, selection_state):
        self._selection_state = selection_state
        self.set_selection()

    def on_selected_molecules_checkbox_state_change(self, selection_state):
        self.selection_state = selection_state
        self.selected_molecules_checkbox.clearFocus()

    def set_selection(self):
        if self.selection_state == 0:
            self.dataset_molecule_indices_to_show = self.dataset.molecule.sel(molecule=~self.dataset.selected).values
        elif self.selection_state == 1:
            self.dataset_molecule_indices_to_show = self.dataset.molecule.values
        elif self.selection_state == 2:
            self.dataset_molecule_indices_to_show = self.dataset.molecule.sel(molecule=self.dataset.selected).values
        else:
            raise ValueError(f'Unknown selection_state {self.selection_state}')

    @property
    def molecule_index(self):
        return self._molecule_index

    @molecule_index.setter
    def molecule_index(self, molecule_index):
        self._molecule_index = molecule_index
        if self.dataset is not None and self.number_of_molecules_to_show > 0:
            self.molecule = self.dataset.isel(molecule=self.dataset_molecule_index)
        else:
            self.molecule = None
        self.molecule_index_field.setText(str(molecule_index))
        self.molecule_index_field.setFocusPolicy(Qt.ClickFocus)

    @property
    def dataset_molecule_index(self):
        return self.dataset_molecule_indices_to_show[self._molecule_index]

    @property
    def dataset_molecule_indices_to_show(self):
        return self._dataset_molecule_indices_to_show

    @dataset_molecule_indices_to_show.setter
    def dataset_molecule_indices_to_show(self, dataset_molecule_indices_to_show):
        self._dataset_molecule_indices_to_show = dataset_molecule_indices_to_show
        self.number_of_molecules_label.setText(f'{self.number_of_molecules_to_show}')
        self.molecule_index = 0

    @property
    def number_of_molecules_to_show(self):
        return len(self.dataset_molecule_indices_to_show)

    def set_molecule_index_from_molecule_index_field(self):
        self.molecule_index = int(self.molecule_index_field.text())

    def next_molecule(self):
        if (self.molecule_index+1) < self.number_of_molecules_to_show:
            self.molecule_index += 1

    def previous_molecule(self):
        if self.molecule_index > 0:
            self.molecule_index -= 1

    def update_current_molecule(self):
        self.molecule_index = self.molecule_index

    @property
    def molecule(self):
        return self.canvas.molecule

    @molecule.setter
    def molecule(self, molecule):
        self.canvas.molecule = molecule

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Right: # Right arrow
            self.next_molecule()
        elif key == Qt.Key_Left: # Left arrow
            self.previous_molecule()
        elif key == Qt.Key_Space: # Spacebar
            self.dataset.selected[dict(molecule=self.dataset_molecule_index)] = ~self.dataset.selected[dict(molecule=self.dataset_molecule_index)]
            self.update_current_molecule()
        elif key == Qt.Key_S: # S
            self.canvas.save()

    # def selected_molecules_checkbox_state_changed(self, state):
    #     show_selected_mapping = {0: False, 1: None, 2: True}
    #     self.show_selected = show_selected_mapping[state]
    #     self.canvas.init_plot_artists()
    #     print('test')


class PlotConfigurationModel(QStandardItemModel):
    """
    Custom model that only allows reordering of top-level rows.
    Disallows dropping into child items.
    """

    def flags(self, index):
        default_flags = super().flags(index)

        # Only top-level items can be dragged
        if not index.parent().isValid():# & index.column() == 0:
            return default_flags | Qt.ItemIsDragEnabled & ~Qt.ItemIsDropEnabled
        else:
            # Children cannot be dragged or accept drops
            return default_flags & ~Qt.ItemIsDropEnabled & ~Qt.ItemIsDragEnabled

    def supportedDropActions(self):
        return Qt.MoveAction

    def dropMimeData(self, data, action, row, column, parent):
        # Only allow drops at root level
        if parent.isValid():
            return False

        self.blockSignals(True)
        result = super().dropMimeData(data, action, row, 0, parent)
        self.blockSignals(False)
        return result

class PlotConfiguration(QWidget):
    def __init__(self, parent, canvas, initial_plot_settings=None):

        super().__init__(parent=parent)

        self.canvas = canvas

        self.view = QTreeView()
        self.model = PlotConfigurationModel()
        self.model.setHorizontalHeaderLabels(["Variable", ""])
        self.view.setColumnWidth(0, 200)

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
        self.model.blockSignals(True)
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item.text() == variable:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
        self.model.blockSignals(False)

    def _disable_trace_variable(self, variable):
        self.model.blockSignals(True)
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item.text() == variable:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        self.model.blockSignals(False)

    def _disable_all_rows(self):
        self.model.blockSignals(True)
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        self.model.blockSignals(False)

    def _enable_dataset_variables(self):
        self._disable_all_rows()
        for var in self._trace_variables_dataset:
            self._enable_trace_variable(var)

    def _add_missing_plot_settings_from_dataset(self):
        plot_settings = self.plot_settings

        for var in self._trace_variables_dataset:
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

            # Plot range text
            if 'plot_range' not in plot_settings[var]:
                if 'FRET' in var:
                    plot_settings[var]['plot_range'] = (-0.05, 1.05)
                elif 'classification' in var:
                    plot_settings[var]['plot_range'] = (self.dataset[var].min().round().item()-0.5,
                                                        self.dataset[var].max().round().item()+0.5)
                else:
                    plot_settings[var]['plot_range'] = (self.dataset[var].min().round().item(),
                                                        self.dataset[var].max().round().item())

            # Color column
            if 'color' not in plot_settings[var]:
                if 'channel' in self.dataset[var].dims:
                    plot_settings[var]['color'] = (('g','r') + ('k',)*10)[0:len(self.dataset.channel)]
                elif 'FRET' in var:
                    plot_settings[var]['color'] = ('b',)
                else:
                    plot_settings[var]['color'] = ('k',)

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
        for row in range(self.model.rowCount()):
            self.view.setFirstColumnSpanned(row, QModelIndex(), True)
        self.view.doItemsLayout()  # Important to refresh treeview, otherwise it is not stay up to date with the model.

    def _add_plot_settings_to_model(self):
        for plot_variable, plot_settings_of_variable in self.plot_settings.items():
            self._add_plot_settings_of_variable_to_model(plot_variable, plot_settings_of_variable)

        self._apply_row_spanning_for_plot_variables()

    def _get_or_create_name_item(self, plot_variable):
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

        self.parent().setFocus()

    def _on_rows_changed(self):
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

        # Update canvas with new ordering
        # self.canvas.plot_settings = self.plot_settings

        # Refocus parent window
        # self.parent().setFocus()

from dataclasses import dataclass
from matplotlib.artist import Artist
@dataclass
class TraceArtist:
    plot_variable: str
    illumination: int
    axis_name: str
    secondary: bool = False
    plot_artists: Optional[list[Artist]] = None
    histogram_artists: Optional[list[Artist]] = None

    def update(self, plot_settings, ys):
        for plot_artist, y in zip(self.plot_artists, ys):
            plot_artist.set_ydata(y)

        for histogram_artist, y in zip(self.histogram_artists, ys):
            n, _ = np.histogram(y, 50, range=plot_settings['plot_range']) # range=self.plot_axes[plot_variable].get_ylim())
            for count, bar in zip(n, histogram_artist):
                bar.set_width(count)
            # TODO: When you shift the view, change the y positions of the bars to the new view, if possible. use set_y

    def set_color(self, colors):
        for plot_artist, color in zip(self.plot_artists, colors):
            plot_artist.set_color(color)
        for histogram_artist, color in zip(self.histogram_artists, colors):
            for bar in histogram_artist:
                bar.set_facecolor(color)

    def show(self, show=True):
        for plot_artist in self.plot_artists:
            plot_artist.set_alpha(int(show))
        for histogram_artist in self.histogram_artists:
            for bar in histogram_artist:
                bar.set_alpha(int(show)*0.5)


class TracePlotCanvas(FigureCanvasQTAgg):
    # Kader om plot als geselecteerd
    # Autosave function
    def __init__(self, parent=None, width=14, height=7, dpi=100):
        self.figure = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi, constrained_layout=False, tight_layout=True)  # , figsize=(2, 2))
        # self.figure.subplots_adjust(top=0.95, left=0.05, right=0.95, bottom=0.05, hspace=0.05, wspace=0.05)
        super().__init__(self.figure)
        self.parent_window = parent

        self._molecule = None

        self._plot_settings = {}

        self._trace_artists = []

        self.plot_axes = {}
        self.histogram_axes = {}

    def _remove_blit_manager(self):
        if hasattr(self, "bm"):
            try:
                self.mpl_disconnect(self.bm.cid)
            except Exception:
                pass
            del self.bm

    @property
    def plot_settings(self):
        return self._plot_settings

    @plot_settings.setter
    def plot_settings(self, plot_settings):
        self._trace_artists = []
        plot_settings = {pv: ps for pv, ps in plot_settings.items() if 'active' in ps and ps['active']}
        data_vars_names = list(self.dataset.data_vars.keys())
        plot_settings = {pv: ps for pv, ps in plot_settings.items() if pv in data_vars_names}
        self._plot_settings = plot_settings
        self.init_plots()

    @property
    def plot_variables(self):
        return list(self.plot_settings.keys())

    @property
    def dataset(self):
        return self.parent_window.dataset

    @property
    def trace_artists(self):
        if not self._trace_artists:
            trace_artists = []
            for plot_variable in self.plot_variables:
                plot_settings = self.plot_settings[plot_variable]
                axis_setting = plot_settings.get('axis', plot_variable)
                # Support multiple axes separated by commas
                axes = [a.strip() for a in axis_setting.split(',')]
                secondary = plot_settings.get('secondary', False)
                for axis in axes:
                    if 'intensity' in plot_variable or 'FRET' in plot_variable:
                        if 'split_illuminations' in plot_settings and plot_settings['split_illuminations']:
                            for key, value in plot_settings.items():
                                if key.startswith('illumination') and value:
                                    illumination = int(key.replace('illumination_', ''))
                                    axis_name = axis + f'_i{illumination}'
                                    trace_artists.append(TraceArtist(plot_variable=plot_variable, illumination=illumination, axis_name=axis_name, secondary=secondary))
                                    # artist_info.append(dict(plot_variable=plot_variable, illumination=illumination, axis_name=axis_name))
                        else:
                            trace_artists.append(TraceArtist(plot_variable=plot_variable, illumination=None, axis_name=axis, secondary=secondary))
                    else:
                        trace_artists.append(TraceArtist(plot_variable=plot_variable, illumination=None, axis_name=axis, secondary=secondary))
                            # artist_info.append(dict(plot_variable=plot_variable, illumination=None, axis_name=plot_variable))
            self._trace_artists = trace_artists

        return self._trace_artists

    def get_trace_artists_with_attribute(self, attribute_name, value):
        return [trace_artist for trace_artist in self.trace_artists if getattr(trace_artist, attribute_name) == value]

    def get_axis_names_with_plot_variable(self, plot_variable):
        trace_artists = self.get_trace_artists_with_attribute('plot_variable', plot_variable)
        axis_names = [trace_artist.axis_name for trace_artist in trace_artists]
        return axis_names

    @property
    def axis_names(self):
        return list(dict.fromkeys([trace_artist.axis_name for trace_artist in self.trace_artists])) # Same as np.unique

    def init_plots(self):
        # Remove current blitmanager
        self._remove_blit_manager()

        self.figure.clf()
        axis_names = self.axis_names
        grid = self.figure.add_gridspec(len(axis_names), 2, width_ratios=[10, 1]) #, height_ratios=(2, 7),
                         # left=0.1, right=0.9, bottom=0.1, top=0.9,
                         # wspace=0.05, hspace=0.05)

        self.plot_axes = {}
        self.twin_axes = {}
        self.histogram_axes = {}

        # self.plot_artists = {}
        # self.histogram_artists = {}

        for i, axis_name in enumerate(axis_names):
            plot = self.figure.add_subplot(grid[i, 0])
            histogram = self.figure.add_subplot(grid[i, 1], sharey=plot)

            if i > 0:
                plot.sharex(self.plot_axes[axis_names[0]])
                histogram.sharex(self.histogram_axes[axis_names[0]])

            if i < len(axis_names) - 1:
                plot.tick_params(labelbottom=False)
                histogram.tick_params(labelbottom=False)

            if i == len(axis_names) - 1:
                if 'time' in self.dataset.coords.keys():
                    plot.set_xlabel(f'Time ({self.dataset.time.units})')
                else:
                    plot.set_xlabel('Frame')

            import re
            # Find the first plot_variable that uses this axis
            # We use the original axis name for lookup in plot_settings, 
            # stripping the illumination suffix if present.
            axis_base_name = re.sub(r"_i\d+", "", axis_name)
            
            # Prefer the plot_variable that is exactly the same as axis_base_name if it exists
            if axis_base_name in self.plot_settings:
                plot_variable_for_settings = axis_base_name
            else:
                # Otherwise just take the first one that maps to this axis
                plot_variable_for_settings = next(
                    (pv for pv, ps in self.plot_settings.items() if axis_base_name in [a.strip() for a in ps.get('axis', pv).split(',')]),
                    axis_base_name
                )
            
            plot_settings = self.plot_settings.get(plot_variable_for_settings, {'plot_range': (0, 1)})

            plot.set_ylim(plot_settings['plot_range'])
            plot.set_ylabel(axis_name[0].upper() + axis_name[1:].replace('_', '\n'))

            histogram.get_yaxis().set_visible(False)

            self.plot_axes[axis_name] = plot
            self.histogram_axes[axis_name] = histogram
            self.twin_axes[axis_name] = None

        self.init_plot_artists()

        # self.draw()

        #self.figure, self.axes = mpl.figure.Figure().subplots(2,1)

    def init_plot_artists(self):
        self._remove_blit_manager()
        for i, trace_artist in enumerate(self.trace_artists):
            # self.plot_axes[plot_variable].cla()

            data_array = self.dataset[trace_artist.plot_variable]

            # For excluding nan values
            dims_without_frame = set(data_array.dims).difference({'frame'})
            frame_not_nan = ~data_array.isnull().all(dim=dims_without_frame)
            data_array = data_array.sel(frame=frame_not_nan)
            data_array_molecule = data_array.sel(molecule=0)

            if trace_artist.illumination is not None:
                data_array_molecule = data_array_molecule.sel(frame=data_array_molecule.illumination == trace_artist.illumination)

            if 'time' in data_array_molecule.coords.keys():
                x = data_array_molecule.time  # self.dataset.time[frame_not_nan]
            else:
                x = data_array_molecule.frame  # self.dataset.frame[frame_not_nan]

            plot_settings = self.plot_settings[trace_artist.plot_variable]

            self.init_plot_artist(trace_artist, plot_settings, x, data_array_molecule)

            if i == 0:
                self.title_artist = self.plot_axes[trace_artist.axis_name].set_title('Init')

        # self.artists += [self.intensity_plot.plot(g, c='g')]
        # self.artists += [self.intensity_plot.plot(r, c='r')]
        # self.artists += [self.FRET_plot.plot(e, c='b')]
        # self.artists += [[self.intensity_plot.set_title('test')]]
        # self.artists += [self.intensity_histogram.hist(g, bins=100, orientation='horizontal',
        #                                                range=self.intensity_plot.get_ylim(), color='g', alpha=0.5)[2]]
        # self.artists += [self.intensity_histogram.hist(r, bins=100, orientation='horizontal',
        #                                                range=self.intensity_plot.get_ylim(), color='r', alpha=0.5)[2]]
        # self.artists += [self.FRET_histogram.hist(e, bins=100, orientation='horizontal',
        #                                           range=self.FRET_plot.get_ylim(), color='b')[2]]

        # self.axes[1].plot(molecule.E(), animate=True)
        artists = [self.title_artist] + \
                  [plot_artist for trace_artist in self.trace_artists for plot_artist in trace_artist.plot_artists] + \
                  [bar for trace_artist in self.trace_artists for histogram_artist in trace_artist.histogram_artists for bar in histogram_artist]

        self.bm = BlitManager(self, artists)
        self.molecule = self.molecule
        self.draw()
        # self.show_artists(show=True, draw=True)

    def init_plot_artist(self, trace_artist, plot_settings, x, y):
        axis = self.plot_axes[trace_artist.axis_name]
        if trace_artist.secondary:
            if self.twin_axes[trace_artist.axis_name] is None:
                self.twin_axes[trace_artist.axis_name] = axis.twinx()
            axis = self.twin_axes[trace_artist.axis_name]
            axis.set_ylim(plot_settings['plot_range'])

        trace_artist.plot_artists = axis.plot(x, y.T)
        # molecule.intensity.plot.line(x='frame', ax=self.plot_axes[plot_variable], color=self.parent_window.colours[i])
        histogram_artists = (
            self.histogram_axes[trace_artist.axis_name].hist(y.T, bins=50, orientation='horizontal',
                                                             # range=self.plot_axes[plot_variable].get_ylim(),
                                                             range=plot_settings['plot_range'],
                                                             color=plot_settings['color'], alpha=0.5))[2]
        if not isinstance(histogram_artists, list):
            histogram_artists = [histogram_artists]

        trace_artist.histogram_artists = histogram_artists

        trace_artist.set_color(plot_settings['color'])

    def show_artists(self, show, draw=True):
        for trace_artist in self.trace_artists:
            trace_artist.show(show)

        if draw:
            self.draw()

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, molecule):
        previous_molecule = self._molecule
        self._molecule = molecule

        if molecule is None and previous_molecule is not None:
            self.show_artists(False, draw=True)
            return
        elif molecule is None and previous_molecule is None:
            return
        elif molecule is not None and previous_molecule is None:
            self.show_artists(True, draw=False)

        self._molecule['file'] = self._molecule['file'].astype(str)

        # g = molecule.intensity.sel(channel=0).values
        # r = molecule.intensity.sel(channel=1).values
        # e = molecule.FRET.values

        # if not self.plot_artists:
        #     self.init_plot_artists()

        # for axis in self.axes:
        #     axis.cla()

        illumination_per_frame = molecule.illumination.values

        for i, trace_artist in enumerate(self.trace_artists):
            # self.plot_axes[plot_variable].cla()
            data = np.atleast_2d(molecule[trace_artist.plot_variable])

            # For excluding nan values (can go wrong when trace contains nans that are not present in all molecules)
            data = data[:, ~np.isnan(data).all(axis=0)]

            if self._molecule.selected.item():
                selection_string = ' | Selected'
            else:
                selection_string = ''

            self.title_artist.set_text(f'# {self.parent_window.molecule_index} of {len(self.dataset.molecule)} | File: {molecule.file.values} | Molecule: {molecule.molecule_in_file.values}' + selection_string)#| Sequence: {molecule.sequence_name.values}')
            self.title_artist.set_text(
                f'File: {molecule.file.values} | Molecule: {molecule.molecule_in_file.values}' + selection_string)  # | Sequence: {molecule.sequence_name.values}')

            plot_settings = self.plot_settings[trace_artist.plot_variable]

            if trace_artist.illumination is not None:
                data = data[:, illumination_per_frame == trace_artist.illumination]

            trace_artist.update(plot_settings, data)





        # self.artists[0][0].set_ydata(g)
        # self.artists[1][0].set_ydata(r)
        # self.artists[2][0].set_ydata(e)
        # self.artists[3][0].set_text(molecule.sequence_name.values)
        # n, _ = np.histogram(g, 100, range=self.intensity_plot.get_ylim())
        # for count, artist in zip(n, self.artists[4]):
        #     artist.set_width(count)
        # n, _ = np.histogram(r, 100, range=self.intensity_plot.get_ylim())
        # for count, artist in zip(n, self.artists[5]):
        #     artist.set_width(count)
        # n, _ = np.histogram(e, 100, range=self.FRET_plot.get_ylim())
        # for count, artist in zip(n, self.artists[6]):
        #     artist.set_width(count)
        #     #for count, rect in zip(n, bar_container.patches):
        # tell the blitting manager to do its thing
        self.bm.update()

    def set_plot_range(self, plot_variable, plot_range):
        axis_names = self.get_axis_names_with_plot_variable(plot_variable)
        secondary = self.plot_settings[plot_variable].get('secondary', False)
        for axis_name in axis_names:
            if secondary and self.twin_axes.get(axis_name) is not None:
                self.twin_axes[axis_name].set_ylim(plot_range[0], plot_range[1])
            else:
                self.plot_axes[axis_name].set_ylim(plot_range[0], plot_range[1])
        self.init_plot_artists()
        # self.init_plots() # Perhaps this can be init_plot_artists only, but then probably the blit background needs to be updated.

        # self.draw()  # full redraw
        # self.bm.on_draw(None)

    def set_plot_color(self, plot_variable, colors):
        trace_artists = self.get_trace_artists_with_attribute('plot_variable', plot_variable)
        for trace_artist in trace_artists:
            trace_artist.set_color(colors)

        self.draw()

    def save(self):
        save_path = self.parent_window.save_path
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            file_name = self.molecule.file.item().replace('\\' ,' - ')+f' - mol {self.molecule.molecule_in_file.item()}.png'
            file_path = save_path.joinpath(file_name)
            self.figure.savefig(file_path, bbox_inches='tight')
        else:
            raise ValueError('No save_path set')

class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        # cv.flush_events()

# class MainWindow(wx.Frame):
#    def __init__(self, parent, title):
#        wx.Frame.__init__(self, parent, title=title, size=(300, 700))
#        self.parent = parent
#        self.panel = TraceAnalysisPanel(parent=self)
#        # self.Bind(wx.EVT_CLOSE, self.OnClose)
#        self.Show()



if __name__ == "__main__":

    # # Check whether there is already a running QApplication (e.g., if running
    # # from an IDE).
    # qapp = QtWidgets.QApplication.instance()
    # if not qapp:
    #     qapp = QtWidgets.QApplication(sys.argv)
    #
    # app = ApplicationWindow()
    # app.show()
    # app.activateWindow()
    # app.raise_()
    # qapp.exec_()


    import papylio as pp
    exp = pp.Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
    ds = exp.files[0].dataset

    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    frame = TracePlotWindow(ds)
        #, "Sample editor", plot_variables=['intensity', 'FRET'],  # 'classification'],
        #          ylims=[(0, 1000), (0, 1), (-1,2)], colours=[('g', 'r'), ('b'), ('k')])

    app.exec_()

    # # exp = pp.Experiment(r'D:\20200918 - Test data\Single-molecule data small')
    # #exp = pp.Experiment(r'P:\SURFdrive\Promotie\Data\Test data')
    # # exp = pp.Experiment(r'/Users/ivoseverins/SURFdrive/Promotie/Data/Test data')
    # # print(exp.files)
    # # m = exp.files[1].molecules[0]
    # # print(exp.files[2])
    # import xarray as xr
    # #file_paths = [p for p in exp.nc_file_paths if '561' in str(p)]
    # file_paths = [exp.nc_file_paths[0]]
    # with xr.open_mfdataset(file_paths, concat_dim='molecule', combine='nested') as ds:
    #     # ds_sel = ds.sel(molecule=ds.sequence_name=='HJ7_G')# .reset_index('molecule', drop=True) # HJ1_WT, HJ7_G116T
    #     app = wx.App(False)
    #     # app = wit.InspectableApp()
    #     frame = TraceAnalysisFrame(None, ds, "Sample editor", plot_variables=['intensity', 'FRET'], #'classification'],
    #              ylims=[(0, 1000), (0, 1), (-1,2)], colours=[('g', 'r'), ('b'), ('k')])
    #     # frame.molecules = exp.files[1].molecules
    #     print('test')
    #     import wx.lib.inspection
    #     wx.lib.inspection.InspectionTool().Show()
    #     app.MainLoop()





# Add time to existing .nc file
# for file in exp.files:
#     with xr.open_dataset(file.absoluteFilePath.with_suffix('.nc')) as ds:
#         i = ds.intensity.load()
#     test = i.assign_coords(time=file.movie.time)
#     test.to_netcdf(file.absoluteFilePath.with_suffix('.nc'), engine='h5netcdf', mode='a')


#
# from matplotlib import use
# use('TkAgg')
#
# import papylio as pp
# exp = pp.Experiment(r'D:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
# #exp = pp.Experiment(r'J:\Ivo\20200221 - Magnetic tweezers setup (Old)\Data')
# # exp.files[-2].perform_mapping()
# # exp.files[-2].mapping.show_mapping_transformation()

# class B:
#     def __init__(self):
#         print('Badd')
#         super().__init__()
#
#
#
# class A:
#     def __init__(self):
#         print('A')
#
#
# def test(c):
#     return type(c.__name__, (c,B),{})
#
#
# @test
# class Bo(A):
#     def __init__(self):
#         print('Bo')
#         super().__init__()



# class B:
#     def __init__(self):
#         print('Badd')
#         super().__init__()
#
# # class PluginMetaClass(type):
# #     def __new__(cls, clsname, bases, attrs):
# #         bases_base = tuple(base for base in bases if not base.__name__ is clsname)
# #         attrs.pop('__qualname__')
# #         cls_base = type(clsname+'_base', bases_base, attrs)
# #         bases_main = tuple(base for base in bases if base.__name__ is clsname) + (cls_base,)
# #         return super().__new__(cls, clsname, bases_main, {})
# class PluginMetaClass(type):
#     def __new__(cls, clsname, bases_base, attrs):
#         # bases_base = tuple(base for base in bases if not base.__name__ is clsname)
#         attrs_base = attrs.copy()
#         attrs_base.pop('__qualname__')
#         #attrs_base.pop('__module__')
#         #attrs_base.pop('__classcell__')
#         cls_base = super().__new__(cls, clsname, bases_base, attrs_base)
#         #cls_base = type(clsname, bases_base, attrs)
#         added_bases = (B,)
#         bases_main = added_bases + (cls_base,)
#         test = super().__new__(cls, clsname+'main', bases_main,{})
#         print('test')
#         return test
#
# class A:
#     def __init__(self):
#         print('A')
#
# class Bo(A, metaclass=PluginMetaClass):
#     def __init__(self):
#         print('Bo')
#         super().__init__()
#



# exp = pp.Experiment(r'P:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
# # exp = pp.Experiment(r'D:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
# exp.files[-1].use_mapping_for_all_files()



# def add_class_to_class(base_class):
#     def add_class_to_class_decorator(added_class):
#         base_class.__bases__ += (added_class,)
#     return add_class_to_class_decorator
#
# @add_class_to_class(pp.File)
# class ExperimentPlugIn():
#     def test(self):
#         print(self.name)



# exp.files[0].find_coordinates()
#
#
# # #exp = pp.Experiment(r'D:\ivoseverins\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20191209 - Single-molecule setup (TIR-I)')
# # exp.files[0].perform_mapping(transformation_type='nonlinear')
# #
# import matplotlib.pyplot as plt
# figure = plt.figure()
# #exp.files[0].show_average_image(figure=figure)
# plt.imshow(exp.files[0].movie.maximum_projection_image)
# exp.files[0].show_coordinates(figure=figure)
# #exp.files[0].mapping.show_mapping_transformation(figure=figure)



# exp.files[-1].use_mapping_for_all_files()

from papylio.plotting import histogram
# exp.files[7].histogram(bins = 100, molecule_averaging=True, export=True)
# exp.histogram(bins = 100, molecule_averaging=True, export=True)
#
# import sys
# #sys.path.append(r'D:\ivoseverins\SURFdrive\Promotie\Code\Python\fastqAnalysis')
# sys.path.append(r'D:\SURFdrive\Promotie\Code\Python\fastqAnalysis')
#
# from papylio.traceAnalysisCode import Experiment
# from fastqAnalysis import FastqData
#
# from pathlib import Path # For efficient path manipulation
#
# path = Path(r'G:\Ivo\20190918 - Sequencer (MiSeq)\Analysis')
# #path = 'D:\\ivoseverins\\Desktop\\Sequencing data\\20180705\\'
# #path = 'C:\\Users\\Ivo Severins\\Desktop\\Sequencing data\\20180705\\'
# fileName = r'One_S1_L001_R1_001.fastq'
#
#
# data = FastqData(path.joinpath(fileName))
#
# data.selection(sequence = 'AA')
#
# data.matches_per_tile(sequence = 'TATCTGTATAATGAGAAATATGGAGTACAATTTTTTTTTTTTTTTTTTTT')









#import wx
#
#
#class OtherFrame(wx.Frame):
#    """
#    Class used for creating frames other than the main one
#    """
#
#    def __init__(self, title, parent=None):
#        wx.Frame.__init__(self, parent=parent, title=title)
#        self.Show()
#
#
#class MyPanel(wx.Panel):
#
#    def __init__(self, parent):
#        wx.Panel.__init__(self, parent)
#
#        btn = wx.Button(self, label='Create New Frame')
#        btn.Bind(wx.EVT_BUTTON, self.on_new_frame)
#        self.frame_number = 1
#
#    def on_new_frame(self, event):
#        title = 'SubFrame {}'.format(self.frame_number)
#        frame = OtherFrame(title=title)
#        self.frame_number += 1
#
#
#class MainFrame(wx.Frame):
#
#    def __init__(self):
#        wx.Frame.__init__(self, None, title='Main Frame', size=(800, 600))
#        panel = MyPanel(self)
#        self.Show()
#
#
#if __name__ == '__main__':
#    app = wx.App(False)
#    frame = MainFrame()
#    app.MainLoop()


# #!/usr/bin/env python
# import wx
# import wx.dataview
# import wx.lib.agw.aui as aui
# import os
#
# import wx.lib.agw.customtreectrl as CT
# #from traceAnalysisCode import Experiment
# import wx.lib.agw.hypertreelist as HTL
#
#
# import matplotlib as mpl
# from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
# from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
#
# from matplotlib import use
# use('WXAgg')
# from matplotlib import pyplot as plt
# #import matplotlib.pyplot as plt
#
#
#
# class MyFrame(wx.Frame):
#     """ We simply derive a new class of Frame. """
#     def __init__(self, parent, title):
#         wx.Frame.__init__(self, parent, title=title, size=(400,400))
#         tree_list = HTL.HyperTreeList(self)
#
#         tree_list.AddColumn("First column")
#
#         root = tree_list.AddRoot("Root")
#
#         parent = tree_list.AppendItem(root, "First child")
#         child = tree_list.AppendItem(parent, "First Grandchild")
#
#         tree_list.AppendItem(root, "Second child", ct_type=1)
#         self.Show(True)
#
# app = wx.App(False)
# frame = MyFrame(None, 'Small editor')
# app.MainLoop()