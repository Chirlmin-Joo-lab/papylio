import sys
import json
from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QTreeView, QApplication, QMainWindow, \
    QPushButton, QTabWidget, QTableWidget, QComboBox, QLineEdit
from PySide2.QtGui import QStandardItem, QStandardItemModel
from PySide2.QtCore import Qt

import numpy as np

class SelectionWidget(QWidget):
    def __init__(self, parent=None):
        super(SelectionWidget, self).__init__(parent)

        self.parent = parent

        self.tree_view = QTreeView(self)
        self.model = QStandardItemModel()
        self.root = self.model.invisibleRootItem()
        self.model.setHorizontalHeaderLabels(['Variable', 'Channel', 'Aggregator', 'Operator', 'Threshold', 'Count'])
        self.tree_view.setModel(self.model)

        self.tree_view.setColumnWidth(0, 150)
        self.tree_view.setColumnWidth(1,100)

        self.model.itemChanged.connect(self.on_item_change)

        # variable_item = QStandardItem()
        # type_item = QStandardItem()
        # comparison_item = QStandardItem()
        # value_item = QStandardItem()
        # add_button_item = QStandardItem()
        # remove_button_item = QStandardItem()
        # self.root.appendRow([
        #     variable_item,
        #     type_item,
        #     comparison_item,
        #     value_item,
        #     add_button_item,
        #     remove_button_item,
        # ])
        # variable_item.setCheckable(True)

        # parentItem = self.root.child(self.root.rowCount() - 1)
        # testitem = QStandardItem('10')
        # selection1 = parentItem.appendRow([
        #     testitem,
        # ])



        #

        variable_combobox = QComboBox()
        variables = ['intensity', 'intensity_total', 'FRET']
        variable_combobox.addItems(variables)

        channel_combobox = QComboBox()
        channels = ['', '0', '1']
        channel_combobox.addItems(channels)

        aggregator_combobox = QComboBox()
        aggregators = ['mean', 'median', 'min', 'max']
        aggregator_combobox.addItems(aggregators)

        operator_combobox = QComboBox()
        operators = ['<', '>']
        operator_combobox.addItems(operators)

        threshold_lineedit = QLineEdit()

        add_button = QPushButton('Add')
        #
        # def add_function():
        #
        #     self.generate_selection(variable_combobox.currentText(),
        #                             channel_combobox.currentText(),
        #                             aggregator_combobox.currentText(),
        #                             operator_combobox.currentText(),
        #                             float(threshold_lineedit.text()))
        # add_button.clicked.connect(add_function)
        add_button.clicked.connect(self.add_selection)


        clear_button = QPushButton('Clear all')
        clear_button.clicked.connect(self.clear_selections)

        apply_to_selected_files_button = QPushButton('Apply to selected files')
        apply_to_selected_files_button.clicked.connect(self.apply_to_selected_files)


        self.add_selection_layout = QHBoxLayout()
        # self.add_selection_layout.addWidget(variable_combobox,1)
        # self.add_selection_layout.addWidget(channel_combobox,1)
        # self.add_selection_layout.addWidget(aggregator_combobox,1)
        # self.add_selection_layout.addWidget(operator_combobox,1)
        # self.add_selection_layout.addWidget(threshold_lineedit,1)
        self.add_selection_layout.addWidget(add_button)
        self.add_selection_layout.addWidget(clear_button)
        self.add_selection_layout.addWidget(apply_to_selected_files_button)

        selection_layout = QVBoxLayout()
        selection_layout.addWidget(self.tree_view)
        selection_layout.addLayout(self.add_selection_layout)

        self.setLayout(selection_layout)

        self.tree_view.setFixedWidth(700)
        #
        # self.add_button = QPushButton('Add')
        # self.add_button.clicked.connect(self.add_selection)
        # selection_layout = QVBoxLayout()
        # selection_layout.addWidget(self.tree_view)
        # selection_layout.addWidget(self.add_button)

        self.setLayout(selection_layout)

        self.update_final_selection = True
        self._file = None

    def on_item_change(self, item):
        if self.update_final_selection:
            selection_names = []
            for i in range(self.model.rowCount()):
                item = self.model.item(i)
                if item.checkState() == Qt.Checked:
                    selection_names.append(self.model.item(i).data())
            self.file.apply_selections(*selection_names)
            self.refresh_selections()
            self.parent.update_plots()

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        self._file = file
        self.update_final_selection = False
        self.refresh_selections()
        self.update_final_selection = True
        # self.refresh_add_panel()

    def clear_selections(self):
        self.file.clear_selections()
        self.refresh_selections()

    def refresh_selections(self):
        self.root.removeRows(0, self.root.rowCount())
        if self.file is not None and '.nc' in self.file.extensions:
            self.setDisabled(False)
            for name, selection in self.file.selections.items():
                if not selection.attrs:
                    row_data = [name[10:], '', '', '', '']
                else:
                    columns = ['variable', 'channel', 'aggregator', 'operator', 'threshold']
                    configuration = json.loads(selection.attrs['configuration'])
                    row_data = [configuration[c] for c in columns]
                row_data.append(selection.sum().item())
                items = [QStandardItem(str(d)) for d in row_data]
                items[0].setCheckable(True)
                items[0].setData(name)
                if 'configuration' in self.file.selected.attrs.keys():
                    if np.isin(name, json.loads(self.file.selected.attrs['configuration'])):
                        items[0].setCheckState(Qt.Checked)
                    else:
                        items[0].setCheckState(Qt.Unchecked)
                self.root.appendRow(items)

            items = [QStandardItem('') for _ in range(6)]
            self.root.appendRow(items)

            row_data = ['', '', '', '', 'Selected', str(self.file.number_of_selected_molecules)]
            items = [QStandardItem(str(d)) for d in row_data]
            self.root.appendRow(items)

            row_data = ['', '', '', '', 'Total', str(self.file.number_of_molecules)]
            items = [QStandardItem(str(d)) for d in row_data]
            self.root.appendRow(items)
        else:
            self.setDisabled(True)

    def add_selection(self):
        items = [QStandardItem(None) for _ in range(self.root.columnCount())]
        row_index = self.root.rowCount()-3
        # self.root.appendRow(items)
        self.root.insertRow(row_index, items)
        self.update_selection(row_index=row_index)

    def update_selection(self, row_index):
        i = row_index

        # row_items = self.root.takeRow(i)

        variable_item = self.root.child(i, 0)
        variable_combobox = QComboBox()
        variables = ['intensity', 'intensity_total', 'FRET']
        variable_combobox.addItems(variables)
        current_variable = variable_item.text()
        if current_variable != '':
            variable_combobox.setCurrentIndex(variables.index(variable_item.text()))
        self.tree_view.setIndexWidget(variable_item.index(), variable_combobox)

        channel_item = self.root.child(i, 1)
        channel_combobox = QComboBox()
        channels = ['', '0', '1']
        channel_combobox.addItems(channels)
        current_channel = channel_item.text()
        if current_channel != '':
            channel_combobox.setCurrentIndex(channels.index(channel_item.text()))
        self.tree_view.setIndexWidget(channel_item.index(), channel_combobox)

        aggregator_item = self.root.child(i, 2)
        aggregator_combobox = QComboBox()
        aggregators = ['mean', 'median', 'min', 'max']
        aggregator_combobox.addItems(aggregators)
        current_aggregator = aggregator_item.text()
        if current_aggregator != '':
            aggregator_combobox.setCurrentIndex(variables.index(aggregator_item.text()))
        self.tree_view.setIndexWidget(aggregator_item.index(), aggregator_combobox)

        operator_item = self.root.child(i, 3)
        operator_combobox = QComboBox()
        operators = ['<', '>']
        operator_combobox.addItems(operators)
        current_operator = operator_item.text()
        if current_operator != '':
            operator_combobox.setCurrentIndex(variables.index(operator_item.text()))
        self.tree_view.setIndexWidget(operator_item.index(), operator_combobox)

        threshold_item = self.root.child(i, 4)
        threshold_lineedit = QLineEdit()
        threshold_lineedit.setText(threshold_item.text())
        self.tree_view.setIndexWidget(threshold_item.index(), threshold_lineedit)

        apply_button_item = self.root.child(i, 5)
        apply_button = QPushButton('Apply')
        apply_function = lambda: self.generate_selection(variable_combobox.currentText(),
                                                         channel_combobox.currentText(),
                                                         aggregator_combobox.currentText(),
                                                         operator_combobox.currentText(),
                                                         float(threshold_lineedit.text()))
        apply_button.clicked.connect(apply_function)
        self.tree_view.setIndexWidget(apply_button_item.index(), apply_button)

        # remove_button_item = self.root.child(i, 5)
        # remove_button = QPushButton('Remove')
        # self.tree_view.setIndexWidget(remove_button_item.index(), remove_button)

    def apply_to_selected_files(self):
        self.file.copy_selections_to_selected_files()

    def generate_selection(self, variable, channel, aggregator, operator, threshold):

        # variable = variable.lower().replace(' ','_')
        # #TODO: Link these to available channels somehow
        # if variable[-6:] == '_green':
        #     channel = 0
        #     variable = variable[:-6]
        # elif variable[-4:] == '_red':
        #     channel = 1
        #     variable = variable[:-4]
        # else:
        #     channel = None

        self.file.create_selection(variable, channel, aggregator, operator, threshold)
        self.refresh_selections()



        # print(variable, ttype, operator, value)

            # variable_item = QStandardItem()
            # type_item = QStandardItem()
            # comparison_item = QStandardItem()
            # value_item = QStandardItem()
            # add_button_item = QStandardItem()
            # remove_button_item = QStandardItem()
            #
            #     variable_item,
            #     type_item,
            #     comparison_item,
            #     value_item,
            #     add_button_item,
            #     remove_button_item,
            # ])


        #     if selection.attrs
    #         name_item = QStandardItem(selection.selection.item()[10:].replace('_',' ').capitalize())
    #         count_item = QStandardItem(str(selection.sum('molecule').item()))
    #
    #         self.root.appendRow([
    #             name_item,
    #             count_item,
    #         ])
    # #
    # def refresh_add_panel(self):
    #     print('test')


        #
        # self.tree_view.expandAll()



## Old


# import sys
#
# # import contextlib
# # import io
# # import numpy as np
# # from pathlib import Path # For efficient path manipulation
# # import trace_analysis as ta
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import tqdm
# # import time
# # import xarray as xr
# # from skimage.transform import AffineTransform
# # from tabulate import tabulate
#
# # from trace_analysis.plugins.holliday_junction.holliday_junction import *
# import sys
# from PySide2.QtWidgets import (QApplication, QMainWindow, QTreeView, QListView, QFileSystemModel,
#                                QVBoxLayout, QHBoxLayout, QWidget, QStyledItemDelegate, QComboBox)
# from PySide2.QtCore import Qt
# from PySide2.QtGui import QStandardItem, QStandardItemModel
# from PySide2.QtWidgets import QTreeWidget, QTreeWidgetItem, QComboBox, QTextEdit, QLineEdit, QPushButton
#
# class TreeViewExample(QMainWindow):
#     def __init__(self):
#         super(TreeViewExample, self).__init__()
#
#         self.setWindowTitle("QTreeView Example")
#         self.setGeometry(100, 100, 800, 600)
#
#         # Create a QTreeView and set up a QFileSystemModel as the model
#
#         # # Set the custom delegate for a specific column (e.g., column 1)
#         # child1 = QTreeWidgetItem(["Field 1", "<Free Text>"])
#         # treeWidget = QTreeWidget(self)
#         # treeWidget.setItemWidget(child1, 1, QComboBox())
#         # self.root.addChild(child1)
#
#         self.selection = SelectionWidget()
#
#         # Set up the layout
#         layout = QVBoxLayout(self)
#         layout.addWidget(self.selection)
#
#         central_widget = QWidget(self)
#         central_widget.setLayout(layout)
#         self.setCentralWidget(central_widget)
#
# class SelectionWidget(QWidget):
#     def __init__(self, parent=None):
#         super(SelectionWidget, self).__init__(parent)
#
#         self.tree_view = QTreeView(self)
#         self.model = QStandardItemModel()
#         self.root = self.model.invisibleRootItem()
#         self.model.setHorizontalHeaderLabels(['Variable', 'Channel', 'Aggregator', 'Operator', 'Threshold', 'Count'])
#         self.tree_view.setModel(self.model)
#
#         self.tree_view.setColumnWidth(0, 150)
#         self.tree_view.setColumnWidth(1,100)
#
#
#         # variable_item = QStandardItem()
#         # type_item = QStandardItem()
#         # comparison_item = QStandardItem()
#         # value_item = QStandardItem()
#         # add_button_item = QStandardItem()
#         # remove_button_item = QStandardItem()
#         # self.root.appendRow([
#         #     variable_item,
#         #     type_item,
#         #     comparison_item,
#         #     value_item,
#         #     add_button_item,
#         #     remove_button_item,
#         # ])
#         # variable_item.setCheckable(True)
#
#         # parentItem = self.root.child(self.root.rowCount() - 1)
#         # testitem = QStandardItem('10')
#         # selection1 = parentItem.appendRow([
#         #     testitem,
#         # ])
#
#
#
#         #
#
#         variable_combobox = QComboBox()
#         variables = ['intensity', 'intensity_total', 'FRET']
#         variable_combobox.addItems(variables)
#
#         channel_combobox = QComboBox()
#         channels = ['', '0', '1']
#         channel_combobox.addItems(channels)
#
#         aggregator_combobox = QComboBox()
#         aggregators = ['mean', 'median', 'min', 'max']
#         aggregator_combobox.addItems(aggregators)
#
#         operator_combobox = QComboBox()
#         operators = ['<', '>']
#         operator_combobox.addItems(operators)
#
#         threshold_lineedit = QLineEdit()
#
#         add_button = QPushButton('Add')
#
#         def add_function():
#
#             self.generate_selection(variable_combobox.currentText(),
#                                     channel_combobox.currentText(),
#                                     aggregator_combobox.currentText(),
#                                     operator_combobox.currentText(),
#                                     float(threshold_lineedit.text()))
#         add_button.clicked.connect(add_function)
#
#         clear_button = QPushButton('Clear all')
#         clear_button.clicked.connect(self.clear_selections)
#
#
#         self.add_selection_layout = QHBoxLayout()
#         self.add_selection_layout.addWidget(variable_combobox,1)
#         self.add_selection_layout.addWidget(channel_combobox,1)
#         self.add_selection_layout.addWidget(aggregator_combobox,1)
#         self.add_selection_layout.addWidget(operator_combobox,1)
#         self.add_selection_layout.addWidget(threshold_lineedit,1)
#         self.add_selection_layout.addWidget(add_button,0.5)
#         self.add_selection_layout.addWidget(clear_button,0.5)
#
#         selection_layout = QVBoxLayout()
#         selection_layout.addWidget(self.tree_view)
#         selection_layout.addLayout(self.add_selection_layout)
#
#         self.setLayout(selection_layout)
#
#         self.tree_view.setFixedWidth(700)
#         #
#         # self.add_button = QPushButton('Add')
#         # self.add_button.clicked.connect(self.add_selection)
#         # selection_layout = QVBoxLayout()
#         # selection_layout.addWidget(self.tree_view)
#         # selection_layout.addWidget(self.add_button)
#
#         self.setLayout(selection_layout)
#
#         self._file = None
#
#     @property
#     def file(self):
#         return self._file
#
#     @file.setter
#     def file(self, file):
#         self._file = file
#         self.refresh_selections()
#         # self.refresh_add_panel()
#
#     def clear_selections(self):
#         self.file.clear_selections()
#         self.refresh_selections()
#
#     def refresh_selections(self):
#         self.root.removeRows(0,self.root.rowCount())
#         for name, selection in self.file.selections_dataset.items():
#             if not selection.attrs:
#                 row_data = [name[10:], '', '', '', '']
#             else:
#                 columns = ['variable', 'channel', 'aggregator', 'operator', 'threshold']
#                 row_data = [selection.attrs[c] for c in columns]
#             row_data.append(selection.sum().item())
#             items = [QStandardItem(str(d)) for d in row_data]
#             self.root.appendRow(items)
#             items[0].setCheckable(True)
#
#     def add_selection(self):
#         items = [QStandardItem(None) for _ in range(self.root.columnCount())]
#         self.root.appendRow(items)
#         self.update_selection(row_index=self.root.rowCount()-1)
#
#     def update_selection(self, row_index):
#         i = row_index
#
#         # row_items = self.root.takeRow(i)
#
#         variable_item = self.root.child(i, 0)
#         variable_combobox = QComboBox()
#         variables = ['intensity', 'intensity_total', 'FRET']
#         variable_combobox.addItems(variables)
#         current_variable = variable_item.text()
#         if current_variable != '':
#             variable_combobox.setCurrentIndex(variables.index(variable_item.text()))
#         self.tree_view.setIndexWidget(variable_item.index(), variable_combobox)
#
#         channel_item = self.root.child(i, 1)
#         channel_combobox = QComboBox()
#         channels = ['', '0', '1']
#         channel_combobox.addItems(variables)
#         current_channel = channel_item.text()
#         if current_channel != '':
#             channel_combobox.setCurrentIndex(channels.index(channel_item.text()))
#         self.tree_view.setIndexWidget(channel_item.index(), channel_combobox)
#
#         aggregator_item = self.root.child(i, 2)
#         aggregator_combobox = QComboBox()
#         aggregators = ['mean', 'median', 'min', 'max']
#         aggregator_combobox.addItems(aggregators)
#         current_aggregator = aggregator_item.text()
#         if current_aggregator != '':
#             aggregator_combobox.setCurrentIndex(variables.index(aggregator_item.text()))
#         self.tree_view.setIndexWidget(aggregator_item.index(), aggregator_combobox)
#
#         operator_item = self.root.child(i, 3)
#         operator_combobox = QComboBox()
#         operators = ['<', '>']
#         operator_combobox.addItems(operators)
#         current_operator = operator_item.text()
#         if current_operator != '':
#             operator_combobox.setCurrentIndex(variables.index(operator_item.text()))
#         self.tree_view.setIndexWidget(operator_item.index(), operator_combobox)
#
#         threshold_item = self.root.child(i, 4)
#         threshold_lineedit = QLineEdit()
#         threshold_lineedit.setText(threshold_item.text())
#         self.tree_view.setIndexWidget(threshold_item.index(), threshold_lineedit)
#
#         apply_button_item = self.root.child(i, 5)
#         apply_button = QPushButton('Apply')
#         apply_function = lambda: self.generate_selection(variable_combobox.currentText(),
#                                                          aggregator_combobox.currentText(),
#                                                          channel_combobox.currentText(),
#                                                          operator_combobox.currentText(),
#                                                          float(threshold_lineedit.text()))
#         apply_button.clicked.connect(apply_function)
#         self.tree_view.setIndexWidget(apply_button_item.index(), apply_button)
#
#         # remove_button_item = self.root.child(i, 5)
#         # remove_button = QPushButton('Remove')
#         # self.tree_view.setIndexWidget(remove_button_item.index(), remove_button)
#
#     def generate_selection(self, variable, channel, aggregator, operator, threshold):
#
#         # variable = variable.lower().replace(' ','_')
#         # #TODO: Link these to available channels somehow
#         # if variable[-6:] == '_green':
#         #     channel = 0
#         #     variable = variable[:-6]
#         # elif variable[-4:] == '_red':
#         #     channel = 1
#         #     variable = variable[:-4]
#         # else:
#         #     channel = None
#
#         self.file.add_selection(variable, channel, aggregator, operator, threshold)
#         self.refresh_selections()
#
#
#
#         # print(variable, ttype, operator, value)
#
#             # variable_item = QStandardItem()
#             # type_item = QStandardItem()
#             # comparison_item = QStandardItem()
#             # value_item = QStandardItem()
#             # add_button_item = QStandardItem()
#             # remove_button_item = QStandardItem()
#             #
#             #     variable_item,
#             #     type_item,
#             #     comparison_item,
#             #     value_item,
#             #     add_button_item,
#             #     remove_button_item,
#             # ])
#
#
#         #     if selection.attrs
#     #         name_item = QStandardItem(selection.selection.item()[10:].replace('_',' ').capitalize())
#     #         count_item = QStandardItem(str(selection.sum('molecule').item()))
#     #
#     #         self.root.appendRow([
#     #             name_item,
#     #             count_item,
#     #         ])
#     # #
#     # def refresh_add_panel(self):
#     #     print('test')
#
#
#         #
#         # self.tree_view.expandAll()
#
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = TreeViewExample()
#     window.show()
#
#     from trace_analysis import Experiment
#     test_path = r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\BN_TIRF'
#     exp = Experiment(test_path)
#
#     window.selection.file = exp.files[-1]
#
#     sys.exit(app.exec_())
