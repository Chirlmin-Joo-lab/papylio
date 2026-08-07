"""Main GUI application and image canvas widget.

Defines the main application window and image canvas used by the Papylio GUI.
"""

import sys
import PySide6
import platform
from PySide2.QtCore import Signal
import sys
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QTreeView, QApplication, QMainWindow, \
                              QPushButton, QTabWidget, QHeaderView, QTableWidget, QComboBox, QLineEdit
from PySide6.QtGui import QStandardItem, QStandardItemModel, QIcon
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

import matplotlib.pyplot as plt
import numpy as np

import papylio as pp
from papylio import Experiment, File
from papylio.trace_plot import TracePlotWindow
from papylio.gui.setup_widget import SetUpWidget
from papylio.gui.selection_widget import SelectionWidget
from papylio.gui.classification_widget import ClassificationWidget
from papylio.gui.mapping_widget import MappingWidget
from papylio.gui.extraction_widget import ExtractionWidget
from papylio.gui.kinetics_widget import KineticsWidget
from papylio.gui.histogram_widget import HistogramWidget
from papylio.gui.common_layouts import HelpDialog
from papylio.gui.image_widget import ImageWidget, ImageCanvas

class MainWindow(QMainWindow):
    """Main application window.

    This class defines the main window of the Papylio GUI, including the
    file tree, image canvas, and various control widgets for
    interacting with the data and configuring the experiment.
    """
    pass_selected_config_to_gui_fields = Signal(int)  # send index of tab to activate
    pass_setup_to_config_on_refresh = Signal(int)

    def __init__(self, main_path=None):
        super().__init__()

        system = platform.system()
        if system == "Windows":
            extension = 'ico'
        else:  # macOS or Linux
            extension = "png"
        self.setWindowIcon(QIcon("icon." + extension))
        self.setWindowTitle("Papylio v" + pp.__version__)

        self.tree = QTreeView(self)
        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        self.model = QStandardItemModel()
        self.root = self.model.invisibleRootItem()
        self.model.setHorizontalHeaderLabels(['Name', 'Count'])
        #self.tree.header().setDefaultSectionSize(180)
        self.tree.setModel(self.model)

        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setColumnHidden(1, True)
        self.tree.setFixedWidth(400)
        self.update = True

        self.model.itemChanged.connect(self.onItemChange)

        # right side has a viewing pane (top) and
        # viewing panes
        self.top_tabs = QTabWidget()
        self.top_tabs.setTabPosition(QTabWidget.North)
        self.top_tabs.setMovable(False)
        self.top_tabs.setDocumentMode(True)
        self.traces = TracePlotWindow(parent=self, width=4, height=5, show=False)
        self.image = ImageWidget(parent=self)
        self.histograms=HistogramWidget(parent=self)
        self.kinetics_results = QWidget()
        self.top_tabs.addTab(self.image, 'Image')
        self.top_tabs.addTab(self.traces, 'Traces')
        self.top_tabs.addTab(self.histograms, 'Histograms')


        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)

        # tab are set to which bottom widget is used:
        self.setup_widget = SetUpWidget(parent=self)
        tabs.addTab(self.setup_widget, 'Start')
        #mapping:
        self.mapping_widget = MappingWidget(parent=self)
        self.mapping_widget.request_top_tab_change.connect(self.top_tabs.setCurrentIndex)
        tabs.addTab(self.mapping_widget, 'Mapping')
        #extraction:
        self.extraction_widget = ExtractionWidget(parent=self, top_tabs=self.top_tabs)
        self.extraction_widget.request_top_tab_change.connect(self.top_tabs.setCurrentIndex)
        tabs.addTab(self.extraction_widget, 'Extraction')
        #selection:
        self.selection_widget = SelectionWidget(parent=self)
        self.selection_widget.request_top_tab_change.connect(self.top_tabs.setCurrentIndex)
        tabs.addTab(self.selection_widget, 'Selection')
        #classification:
        self.classification_widget = ClassificationWidget(parent=self)
        self.classification_widget.classificationChanged.connect(self.update_plots)
        self.classification_widget.request_top_tab_change.connect(self.top_tabs.setCurrentIndex)
        tabs.addTab(self.classification_widget, 'Classification')
        #time analysis:
        self.kinetics_widget = KineticsWidget(parent=self)
        tabs.addTab(self.kinetics_widget, 'Kinetics')
        tabs.currentChanged.connect(self.setTabFocus)

        self.pass_selected_config_to_gui_fields.connect(self.extraction_widget.set_buttons_from_selected_file)
        self.pass_setup_to_config_on_refresh.connect(self.setup_widget.pass_buttons_to_config_for_setup)

        self.tab_widgets = [self.image, self.histograms, self.traces, self.setup_widget, self.mapping_widget, self.extraction_widget,
                            self.selection_widget, self.classification_widget, self.kinetics_widget]

        # refresh & tree
        experiment_layout = QVBoxLayout()
        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.refresh)
        experiment_layout.addWidget(refresh_button)
        experiment_layout.addWidget(self.tree)

        top_layout = QVBoxLayout()
        top_layout.addWidget(self.top_tabs)
        # ... a bottom pane (pipeline)
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(tabs)

        # build main panel
        # full left is file tree
        left_layout = QVBoxLayout()
        left_layout.addLayout(experiment_layout)

        right_layout = QVBoxLayout()
        right_layout.addLayout(top_layout)
        right_layout.addLayout(bottom_layout)

        super_layout = QHBoxLayout()
        super_layout.addLayout(left_layout)
        super_layout.addLayout(right_layout)

        central_widget = QWidget()
        central_widget.setLayout(super_layout)

        self.setCentralWidget(central_widget)
        self.show()
        self.showMaximized()

        self.experiment = pp.Experiment(main_path, main_window=self)
        self.addExperiment(self.experiment)
        self.setup_widget.experiment = self.experiment
        self.traces.save_path = self.experiment.analysis_path.joinpath('Trace_plots')

    def keyPressEvent(self, e):
        self.traces.keyPressEvent(e)

    def setTabFocus(self, e):
        if e == 0:
            self.image.setFocus()
        if e == 1:
            self.traces.setFocus()

    def onItemChange(self, item):
        if isinstance(item.data(), File):
            file = item.data()
            file.is_selected = (True if item.checkState() == Qt.Checked else False)
            print(f'{file}: {file.is_selected}')
        else:
            self.update = False
            for i in range(item.rowCount()):
                item.child(i).setCheckState(item.checkState())
            self.update = True
        if self.update:
            self.update_plots()
            self.update_settings()

    def update_settings(self):
        #this collects the configuration of the first selected file
        #and passes the contents  to all relevant settings buttons
        selected_files = self.experiment.selectedFiles + [None]
        if selected_files[0] is not None:
            self.pass_selected_config_to_gui_fields.emit(1)

    def update_plots(self):
        selected_files = self.experiment.selectedFiles + [None]
        for widget in self.tab_widgets:
            widget.file = selected_files[0]
        # self.image.file = selected_files[0]
        # if selected_files[0] is not None:
        #     self.traces.file = selected_files[0]
        #     self.selection_widget.file = selected_files[0]
        #     self.classification_widget.file = selected_files[0]
        # else:
        #     self.traces.file = None
        #     self.selection_widget.file = None
        #     self.classification_widget.file = None

    def addExperiment(self, experiment):
        self.root.appendRow([
            QStandardItem(experiment.name),
            QStandardItem(0),
        ])
        experimentNode = self.root.child(self.root.rowCount() - 1)
        for file in experiment.files:
            print('addfile' + file.name)
            self.addFile(file, experimentNode)

        self.tree.expandAll()

        print('add')

    def addFile(self, file, experimentNode):
        folders = file.relativePath.parts

        parentItem = experimentNode
        parentItem.setCheckable(True)
        for folder in folders:

            # Get the folderItems and folder names for the current folderItem
            nodeItems = [parentItem.child(i) for i in range(parentItem.rowCount())]  # if item.type == 'folder']
            nodeItemNames = [item.text() for item in nodeItems]

            if folder not in nodeItemNames:
                # Add new item for the folder and set parentItem to this item
                parentItem.appendRow([
                    QStandardItem(folder),
                    QStandardItem(0),
                ])
                parentItem = parentItem.child(parentItem.rowCount() - 1)
                parentItem.setCheckable(True)
            else:
                # Set parent item to the found folderItem
                parentItem = nodeItems[nodeItemNames.index(folder)]

        parentItem.appendRow([
            QStandardItem(file.name),
            QStandardItem(0),
        ])
        item = parentItem.child(parentItem.rowCount() - 1)
        item.setCheckable(True)
        if file.is_selected:
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)
        item.setData(file)
        # self.FileItems.append(item)

        # self.insertDataIntoColumns(item)

        return item

    def refresh(self):
        #TODO: pass settings from setup widget to config before doing this one
        self.pass_setup_to_config_on_refresh.emit(1)
        self.root.removeRows(0, 1)
        self.experiment = Experiment(self.experiment.main_path)
        self.addExperiment(self.experiment)


if __name__ == '__main__':
    from multiprocessing import Process, freeze_support

    freeze_support()
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    app.exec_()


