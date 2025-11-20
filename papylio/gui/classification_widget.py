import sys
from PySide2.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QLineEdit, QPushButton, QFormLayout, QSpinBox, QDoubleSpinBox,
    QTreeView, QMainWindow, QMessageBox, QCheckBox
)

from PySide2.QtGui import QStandardItem, QStandardItemModel
from PySide2.QtCore import Qt

from papylio.analysis.classification_simple import classify_threshold
from papylio.analysis.hidden_markov_modelling import classify_hmm

import numpy as np
import inspect
import json
import typing

class ClassificationWidget(QWidget):
    def __init__(self, parent=None):
        super(ClassificationWidget, self).__init__(parent)

        self.methods = {}
        self._file = None# name -> function

        main_layout = QVBoxLayout(self)

        # --- Classification name input ---
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. HMM")
        name_layout.addWidget(self.name_edit)
        main_layout.addLayout(name_layout)

        # --- Method selector ---
        method_layout = QHBoxLayout()
        self.method_selector = QComboBox()
        self.method_selector.currentTextChanged.connect(self._update_method_panel)
        method_layout.addWidget(QLabel("Method:"))
        method_layout.addWidget(self.method_selector)
        main_layout.addLayout(method_layout)

        # --- Variable selector ---
        variable_layout = QHBoxLayout()
        self.variable_selector = QLineEdit()
        variable_layout.addWidget(QLabel("Variable:"))
        variable_layout.addWidget(self.variable_selector)
        main_layout.addLayout(variable_layout)

        # --- Dynamic options container ---
        self.stack = QWidget()
        self.stack_layout = QVBoxLayout(self.stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stack)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Classify")
        self.clear_button = QPushButton("Clear")
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        main_layout.addLayout(button_layout)

        # --- Results table ---
        self.tree_view = QTreeView(self)
        self.model = QStandardItemModel()
        self.root = self.model
        # self.root = self.model.invisibleRootItem()
        self.model.setHorizontalHeaderLabels(["", "States", "Name", "Method", "Variable", "Select", "Parameters"])
        self.tree_view.setModel(self.model)

        self.model.itemChanged.connect(self.on_item_changed)

        # --- Column sizing ---
        self.tree_view.setColumnWidth(0, 20)    # Checkbox
        self.tree_view.setColumnWidth(1, 60)    # States
        self.tree_view.setColumnWidth(2, 100)   # Name
        self.tree_view.setColumnWidth(3, 100)   # Method
        self.tree_view.setColumnWidth(4, 100)   # Variable
        self.tree_view.setColumnWidth(5, 100)   # Select
        self.tree_view.setColumnWidth(6, 250)   # Parameters

        main_layout.addWidget(self.tree_view)
        # self.tree_view.setColumnWidth(0, 150)
        # self.tree_view.setColumnWidth(1,100)
        # self.model.itemChanged.connect(self.on_item_change)

        # --- Connect actions ---
        self.run_button.clicked.connect(self._run_classification)
        self.clear_button.clicked.connect(self._clear_results)

        self.method_forms = {}  # method_name -> (widget, inputs)
        self.setLayout(main_layout)

        self.register_method('threshold', classify_threshold)
        self.register_method('hmm', classify_hmm)

        self.refresh_classifications()

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        self._file = file
        self.refresh_classifications()
        # self.update_final_selection = False
        # self.refresh_selections()
        # self.update_final_selection = True
        # self.refresh_add_panel()

    # -------------------------------------------------------------------------
    # Register methods dynamically and create their forms
    # -------------------------------------------------------------------------
    def register_method(self, name, func):
        """Register a classification method, introspect arguments, and build a form."""

        self.methods[name] = func

        # --- build the form for the function ---
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        inputs = {}

        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param_name in ['traces', 'classification', 'selection']:
                continue

            default = param.default if param.default is not inspect.Parameter.empty else None
            annotation = param.annotation

            # Pick appropriate input type
            if annotation == int or isinstance(default, int):
                widget = QSpinBox()
                widget.setRange(-1_000_000, 1_000_000)
                if default is not None:
                    widget.setValue(default)
            elif annotation == float or isinstance(default, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setDecimals(6)
                if default is not None:
                    widget.setValue(default)
            else:
                widget = QLineEdit()
                if default not in (None, inspect.Parameter.empty):
                    widget.setText(str(default))

            form.addRow(f"{param_name}:", widget)
            inputs[param_name] = widget

        self.method_forms[name] = (form_widget, inputs)
        self.method_selector.addItem(name)

        # First registered method becomes default
        if self.method_selector.count() == 1:
            self._update_method_panel(name)

    def _update_method_panel(self, name):
        # Clear the old form
        for i in reversed(range(self.stack_layout.count())):
            widget = self.stack_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Add new form
        if name in self.method_forms:
            form_widget, _ = self.method_forms[name]
            self.stack_layout.addWidget(form_widget)

    def _run_classification(self):
        method_name = self.method_selector.currentText()
        if not method_name:
            QMessageBox.warning(self, "No method", "Please select a classification method.")
            return

        variable_name = self.variable_selector.text()
        if not variable_name:
            QMessageBox.warning(self, "No variable", "Please select a variable used for classification.")
            return

        name = self.name_edit.text().strip() or f"Unnamed_{method_name}"
        func = self.methods[method_name]
        _, inputs = self.method_forms[method_name]

        # Collect args
        kwargs = {}
        for pname, widget in inputs.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                val = widget.value()
            else:
                val = widget.text()
                try:
                    val = float(val) if "." in val else int(val)
                except ValueError:
                    pass
            kwargs[pname] = val

        try:
            self.file.create_classification(method_name, variable_name, select=None, name=name, classification_kwargs=kwargs,
                                            apply=None)
            name = 'classification_' + name
            self.add_classification(name, self.file.classifications[name])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

    def add_classification(self, name, classification):
        if not classification.attrs:
            row_data = ['', '', name[len('classification_'):], '', '', '', '']
        else:
            configuration = json.loads(classification.attrs['configuration'])
            classification_type = configuration.get('classification_type', '')
            variable = configuration.get('variable', '')
            select = configuration.get('select', '')
            parameters = configuration.get('classification_kwargs', {})
            row_data = [
                '',
                '',
                name[len('classification_'):],
                classification_type,
                variable,
                select,
                json.dumps(parameters)
            ]

        items = [QStandardItem(str(d)) for d in row_data]
        items[0].setCheckable(True)
        # items[2].setData(name)

        self.root.appendRow(items)
        if 'configuration' in self.file.classification.attrs.keys():
            configuration = json.loads(self.file.classification.attrs['configuration'])
            if name in configuration:
                items[0].setCheckState(Qt.Checked)
                # text_edit.setText(str(configuration[name]))
            else:
                items[0].setCheckState(Qt.Unchecked)

    def refresh_classifications(self):
        self.root.removeRows(0, self.root.rowCount())
        if self.file is not None and '.nc' in self.file.extensions:
            self.setDisabled(False)
            for name, classification in self.file.classifications.items():
                self.add_classification(name, classification)
        else:
            self.setDisabled(True)


    def _clear_results(self):
        self.file.clear_classifications()
        self.refresh_classifications()

    def get_checked_classifications(self):
        pass

    def on_item_changed(self, item):
        """Triggered when a checkbox in the table is toggled."""
        # Only respond if this is the Name column (the one with checkboxes)
        row = item.row()
        column = item.column()

        if column in [0, 1]:
            self.apply_classifications()
        # name = item.data()  # stored classification name
        # if not name:
        #     return
        #
        # checked = (item.checkState() == Qt.Checked)
        #
        # # Example: print or update your file selections
        # print(f"Checkbox toggled for '{name}': {'Checked' if checked else 'Unchecked'}")
        #
        # # If you want to apply logic to your file object:
        # if self.file is not None:
        #     try:
        #         # Example: toggle selection in your file object
        #         selected = set(json.loads(self.file.selected.attrs.get('configuration', '[]')))
        #         if checked:
        #             selected.add(name)
        #         else:
        #             selected.discard(name)
        #         self.file.selected.attrs['configuration'] = json.dumps(list(selected))
        #         print(f"Updated selected configurations: {selected}")
        #     except Exception as e:
        #         print(f"Error updating selection for {name}: {e}")


    def apply_classifications(self):
        apply_classifications_configuration = {}
        for row in range(self.model.rowCount()):
            item_checkbox = self.model.item(row, 0)

            if item_checkbox.checkState() == Qt.Checked:

                classification_states = self.model.item(row, 1).text()
                classification_name = 'classification_' + self.model.item(row, 2).text()
                if classification_states in [None, '']:
                    classification_states = np.unique(self.file.classifications[classification_name]).astype(int).tolist()
                    self.model.blockSignals(True)
                    self.model.item(row, 1).setText(str(classification_states)[1:-1])
                    self.model.blockSignals(False)
                elif isinstance(classification_states, str):
                    classification_states = [int(x.strip()) for x in classification_states.split(',')]
                apply_classifications_configuration[classification_name] = classification_states

        self.file.apply_classifications(**apply_classifications_configuration)




    #
    #
    #     classification_type_combobox = QComboBox()
    #     classification_types = ['threshold', 'filter', 'hmm']
    #     classification_type_combobox.addItems(classification_types)
    #
    #     channel_combobox = QComboBox()
    #     channels = ['', '0', '1']
    #     channel_combobox.addItems(channels)
    #
    #     aggregator_combobox = QComboBox()
    #     aggregators = ['mean', 'median', 'min', 'max']
    #     aggregator_combobox.addItems(aggregators)
    #
    #     operator_combobox = QComboBox()
    #     operators = ['<', '>']
    #     operator_combobox.addItems(operators)
    #
    #     threshold_lineedit = QLineEdit()
    #
    #     add_button = QPushButton('Add')
    #     #
    #     # def add_function():
    #     #
    #     #     self.generate_selection(variable_combobox.currentText(),
    #     #                             channel_combobox.currentText(),
    #     #                             aggregator_combobox.currentText(),
    #     #                             operator_combobox.currentText(),
    #     #                             float(threshold_lineedit.text()))
    #     # add_button.clicked.connect(add_function)
    #     add_button.clicked.connect(self.add_selection)
    #
    #
    #     clear_button = QPushButton('Clear all')
    #     clear_button.clicked.connect(self.clear_selections)
    #
    #     apply_to_selected_files_button = QPushButton('Apply to selected files')
    #     apply_to_selected_files_button.clicked.connect(self.apply_to_selected_files)
    #
    #
    #     self.add_selection_layout = QHBoxLayout()
    #     # self.add_selection_layout.addWidget(variable_combobox,1)
    #     # self.add_selection_layout.addWidget(channel_combobox,1)
    #     # self.add_selection_layout.addWidget(aggregator_combobox,1)
    #     # self.add_selection_layout.addWidget(operator_combobox,1)
    #     # self.add_selection_layout.addWidget(threshold_lineedit,1)
    #     self.add_selection_layout.addWidget(add_button)
    #     self.add_selection_layout.addWidget(clear_button)
    #     self.add_selection_layout.addWidget(apply_to_selected_files_button)
    #
    #     selection_layout = QVBoxLayout()
    #     selection_layout.addWidget(self.tree_view)
    #     selection_layout.addLayout(self.add_selection_layout)
    #
    #     self.setLayout(selection_layout)
    #
    #     self.tree_view.setFixedWidth(700)
    #     #
    #     # self.add_button = QPushButton('Add')
    #     # self.add_button.clicked.connect(self.add_selection)
    #     # selection_layout = QVBoxLayout()
    #     # selection_layout.addWidget(self.tree_view)
    #     # selection_layout.addWidget(self.add_button)
    #
    #     self.setLayout(selection_layout)
    #
    #     self.update_final_selection = True
    #     self._file = None
    #
    # def on_item_change(self, item):
    #     if self.update_final_selection:
    #         selection_names = []
    #         for i in range(self.model.rowCount()):
    #             item = self.model.item(i)
    #             if item.checkState() == Qt.Checked:
    #                 selection_names.append(self.model.item(i).data())
    #         self.file.apply_selections(selection_names)
    #         self.refresh_selections()
    #
    # @property
    # def file(self):
    #     return self._file
    #
    # @file.setter
    # def file(self, file):
    #     self._file = file
    #     self.update_final_selection = False
    #     self.refresh_selections()
    #     self.update_final_selection = True
    #     # self.refresh_add_panel()
    #
    # def clear_selections(self):
    #     self.file.clear_selections()
    #     self.refresh_selections()
    #
    # def refresh_selections(self):
    #     self.root.removeRows(0, self.root.rowCount())
    #     if self.file is not None and '.nc' in self.file.extensions:
    #         self.setDisabled(False)
    #         for name, selection in self.file.selections_dataset.items():
    #             if not selection.attrs:
    #                 row_data = [name[10:], '', '', '', '']
    #             else:
    #                 columns = ['variable', 'channel', 'aggregator', 'operator', 'threshold']
    #                 row_data = [selection.attrs[c] for c in columns]
    #             row_data.append(selection.sum().item())
    #             items = [QStandardItem(str(d)) for d in row_data]
    #             items[0].setCheckable(True)
    #             items[0].setData(name)
    #             if 'selection_names' in self.file.selected.attrs.keys():
    #                 if np.isin(name, self.file.selected.attrs['selection_names']):
    #                     items[0].setCheckState(Qt.Checked)
    #                 else:
    #                     items[0].setCheckState(Qt.Unchecked)
    #             self.root.appendRow(items)
    #
    #         items = [QStandardItem('') for _ in range(6)]
    #         self.root.appendRow(items)
    #
    #         row_data = ['', '', '', '', 'Selected', str(self.file.number_of_selected_molecules)]
    #         items = [QStandardItem(str(d)) for d in row_data]
    #         self.root.appendRow(items)
    #
    #         row_data = ['', '', '', '', 'Total', str(self.file.number_of_molecules)]
    #         items = [QStandardItem(str(d)) for d in row_data]
    #         self.root.appendRow(items)
    #     else:
    #         self.setDisabled(True)
    #
    # def add_selection(self):
    #     items = [QStandardItem(None) for _ in range(self.root.columnCount())]
    #     row_index = self.root.rowCount()-3
    #     # self.root.appendRow(items)
    #     self.root.insertRow(row_index, items)
    #     self.update_selection(row_index=row_index)
    #
    # def update_selection(self, row_index):
    #     i = row_index
    #
    #     # row_items = self.root.takeRow(i)
    #
    #     variable_item = self.root.child(i, 0)
    #     variable_combobox = QComboBox()
    #     variables = ['intensity', 'intensity_total', 'FRET']
    #     variable_combobox.addItems(variables)
    #     current_variable = variable_item.text()
    #     if current_variable != '':
    #         variable_combobox.setCurrentIndex(variables.index(variable_item.text()))
    #     self.tree_view.setIndexWidget(variable_item.index(), variable_combobox)
    #
    #     channel_item = self.root.child(i, 1)
    #     channel_combobox = QComboBox()
    #     channels = ['', '0', '1']
    #     channel_combobox.addItems(channels)
    #     current_channel = channel_item.text()
    #     if current_channel != '':
    #         channel_combobox.setCurrentIndex(channels.index(channel_item.text()))
    #     self.tree_view.setIndexWidget(channel_item.index(), channel_combobox)
    #
    #     aggregator_item = self.root.child(i, 2)
    #     aggregator_combobox = QComboBox()
    #     aggregators = ['mean', 'median', 'min', 'max']
    #     aggregator_combobox.addItems(aggregators)
    #     current_aggregator = aggregator_item.text()
    #     if current_aggregator != '':
    #         aggregator_combobox.setCurrentIndex(variables.index(aggregator_item.text()))
    #     self.tree_view.setIndexWidget(aggregator_item.index(), aggregator_combobox)
    #
    #     operator_item = self.root.child(i, 3)
    #     operator_combobox = QComboBox()
    #     operators = ['<', '>']
    #     operator_combobox.addItems(operators)
    #     current_operator = operator_item.text()
    #     if current_operator != '':
    #         operator_combobox.setCurrentIndex(variables.index(operator_item.text()))
    #     self.tree_view.setIndexWidget(operator_item.index(), operator_combobox)
    #
    #     threshold_item = self.root.child(i, 4)
    #     threshold_lineedit = QLineEdit()
    #     threshold_lineedit.setText(threshold_item.text())
    #     self.tree_view.setIndexWidget(threshold_item.index(), threshold_lineedit)
    #
    #     apply_button_item = self.root.child(i, 5)
    #     apply_button = QPushButton('Apply')
    #     apply_function = lambda: self.generate_selection(variable_combobox.currentText(),
    #                                                      channel_combobox.currentText(),
    #                                                      aggregator_combobox.currentText(),
    #                                                      operator_combobox.currentText(),
    #                                                      float(threshold_lineedit.text()))
    #     apply_button.clicked.connect(apply_function)
    #     self.tree_view.setIndexWidget(apply_button_item.index(), apply_button)
    #
    #     # remove_button_item = self.root.child(i, 5)
    #     # remove_button = QPushButton('Remove')
    #     # self.tree_view.setIndexWidget(remove_button_item.index(), remove_button)
    #
    # def apply_to_selected_files(self):
    #     self.file.copy_selections_to_selected_files()
    #
    # def generate_selection(self, variable, channel, aggregator, operator, threshold):
    #
    #     # variable = variable.lower().replace(' ','_')
    #     # #TODO: Link these to available channels somehow
    #     # if variable[-6:] == '_green':
    #     #     channel = 0
    #     #     variable = variable[:-6]
    #     # elif variable[-4:] == '_red':
    #     #     channel = 1
    #     #     variable = variable[:-4]
    #     # else:
    #     #     channel = None
    #
    #     self.file.add_selection(variable, channel, aggregator, operator, threshold)
    #     self.refresh_selections()


