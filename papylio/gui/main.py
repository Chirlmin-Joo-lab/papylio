import sys
import PySide2
import platform

import sys
from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QTreeView, QApplication, QMainWindow, \
    QPushButton, QTabWidget, QTableWidget, QComboBox, QLineEdit
from PySide2.QtGui import QStandardItem, QStandardItemModel, QIcon
from PySide2.QtCore import Qt

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
import numpy as np

import papylio as pp
from papylio import Experiment, File
from papylio.trace_plot import TracePlotWindow
from papylio.gui.selection_widget import SelectionWidget

# class TreeNode:
#     def __init__(self, node_object, parent=None):
#         self.parent = parent
#         if isinstance(node_object, Experiment):
#             self.experiment = node_object
#             self.name = self.experiment.name
#             self.type = 'experiment'
#         elif isinstance(node_object, str):
#             self.name = node_object
#             self.type = 'folder'
#         elif isinstance(node_object, File):
#             self.file = node_object
#             self.name = self.file.name
#             self.type = 'file'
#
#         self.children = []
#
#     def data(self, column):
#         # if column == 0:
#         return self.columnValues[column]
#         # else:
#         #     return ''
#         # return self._data[column]
#
#     def appendChild(self, node_object):
#         node = TreeNode(node_object, self)
#         self.children.append(node)
#         return node
#
#     def child(self, row):
#         return self.children[row]
#
#     def childrenCount(self):
#         return len(self.children)
#
#     def hasChildren(self):
#         if len(self.children) > 0:
#             return True
#         return False
#
#     def row(self):
#         if self.parent is not None:
#             return self.parent.children.index(self)
#         else:
#             return 0
#
#     @property
#     def columnValues(self):
#         return [self.name]
#
#     def columnCount(self):
#         return len(self.columnValues)
#
#     def __repr__(self):
#         return f'TreeNode: {self.name}'
#
#
# class TreeModel(QAbstractItemModel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         # column_names = ['Column1','Column2']
#         self.root = TreeNode('Name')
#         self.createData()
#         print('t')
#
#     def createData(self):
#         for x in ['a','b','c']:
#             self.root.appendChild(x)
#         for y in ['q','r','s']:
#             self.root.child(0).appendChild(y)
#         for z in ['d','e','f']:
#             self.root.child(2).appendChild(z)
#
#     def addExperiment(self, experiment):
#         # experiment = Experiment(r'D:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
#         #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
#         experimentNode = self.root.appendChild(experiment)
#         for file in experiment.files:
#             print('addfile'+file.name)
#             self.addFile(file, experimentNode)
#
#         print('add')
#
#     def addFile(self, file, experimentNode):
#         # pass
#
#         folders = file.relativePath.parts
#
#         #nodeItemNames = [item.GetText() for item in experimentNode.children if item.GetData() == None]
#
#         parentItem = experimentNode
#         for folder in folders:
#
#             # Get the folderItems and folder names for the current folderItem
#             nodeItems = [item for item in parentItem.children if item.type == 'folder']
#             nodeItemNames = [item.name for item in nodeItems]
#
#             if folder not in nodeItemNames:
#                 # Add new item for the folder and set parentItem to this item
#                 parentItem = parentItem.appendChild(folder)
#             else:
#                 # Set parent item to the found folderItem
#                 parentItem = nodeItems[nodeItemNames.index(folder)]
#
#         item = parentItem.appendChild(file)
#         #self.FileItems.append(item)
#
#         # self.insertDataIntoColumns(item)
#
#         return item
#
#     def columnCount(self, index=QtCore.QModelIndex()):
#         if index.isValid():
#             return index.internalPointer().columnCount()
#         else:
#             return self.root.columnCount()
#
#     def rowCount(self, index=QtCore.QModelIndex()):
#         if index.row() > 0:
#             return 0
#         if index.isValid():
#             item = index.internalPointer()
#         else:
#             item = self.root
#         return item.childrenCount()
#
#     def index(self, row, column, index=QtCore.QModelIndex()):
#         if not self.hasIndex(row, column, index):
#             return QtCore.QModelIndex()
#         if not index.isValid():
#             item = self.root
#         else:
#             item = index.internalPointer()
#
#         child = item.child(row)
#         if child:
#             return self.createIndex(row, column, child)
#         return QtCore.QMOdelIndex()
#
#     def parent(self, index):
#         if not index.isValid():
#             return QtCore.QModelIndex()
#         item = index.internalPointer()
#         if not item:
#             return QtCore.QModelIndex()
#
#         parent = item.parent
#         if parent == self.root:
#             return QtCore.QModelIndex()
#         else:
#             return self.createIndex(parent.row(), 0, parent)
#
#     def hasChildren(self, index):
#         if not index.isValid():
#             item = self.root
#         else:
#             item = index.internalPointer()
#         return item.hasChildren()
#
#     def data(self, index, role=QtCore.Qt.DisplayRole):
#        if index.isValid() and role == QtCore.Qt.DisplayRole:
#             return index.internalPointer().data(index.column())
#        elif not index.isValid():
#             return self.root.getData()
#
#     def headerData(self, section, orientation, role):
#         if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
#             return self.root.data(section)
#
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         # model = QFileSystemModel()
#         # model.setRootPath(QDir.currentPath())
#
#
#
#         self.model = TreeModel()
#
#         self.tree = QTreeView()
#         self.tree.setModel(self.model)
#
#         from papylio import Experiment
#         experiment = Experiment(r'D:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
#         #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
#         #self.model.addExperiment(experiment)
#
#         self.setCentralWidget(self.tree)


class MainWindow(QMainWindow):
    # def __init__(self):
    #     super().__init__()
    #     # model = QFileSystemModel()
    #     # model.setRootPath(QDir.currentPath())
    #
    #
    #
    #     self.model = TreeModel()
    #
    #     self.tree = QTreeView()
    #     self.tree.setModel(self.model)
    #
    #      #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
    #     #self.model.addExperiment(experiment)
    #
    #     self.setCentralWidget(self.tree)

    def __init__(self, main_path=None):
        super().__init__()

        system = platform.system()
        if system == "Windows":
            extension = 'ico'
        else:  # macOS or Linux
            extension = "png"
        self.setWindowIcon(QIcon("icon."+extension))
        self.setWindowTitle("Papylio v" + pp.__version__ )

        self.tree = QTreeView(self)
        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        self.model = QStandardItemModel()
        self.root = self.model.invisibleRootItem()
        self.model.setHorizontalHeaderLabels(['Name', 'Count'])
        self.tree.header().setDefaultSectionSize(180)
        self.tree.setModel(self.model)

        self.tree.setFocusPolicy(Qt.NoFocus)
        self.tree.setFixedWidth(256)
        self.update = True

        self.model.itemChanged.connect(self.onItemChange)


        self.image_canvas = ImageCanvas(self, width=5, height=4, dpi=100)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        image_toolbar = NavigationToolbar(self.image_canvas, self)

        image_layout = QVBoxLayout()
        image_layout.addWidget(image_toolbar)
        image_layout.addWidget(self.image_canvas)

        # Create a placeholder widget to hold our toolbar and canvas.
        self.image = QWidget()
        self.image.setLayout(image_layout)

        controls_layout = QGridLayout()
        controls_layout.setAlignment(Qt.AlignTop)

        # controls_layout.addWidget(QLabel('Minimum intensity difference'), 0, 0)
        # mid = QLineEdit(str(self.experiment.configuration['find_coordinates']['peak_finding']['minimum_intensity_difference']))
        # mid.textChanged.connect(self.midChange)
        # controls_layout.addWidget(mid, 0, 1)

        perform_mapping_button = QPushButton('Perform mapping')
        perform_mapping_button.clicked.connect(self.perform_mapping)
        controls_layout.addWidget(perform_mapping_button, 1, 0, 1, 2)

        find_molecules_button = QPushButton('Find coordinates')
        find_molecules_button.clicked.connect(self.find_coordinates)
        controls_layout.addWidget(find_molecules_button, 2, 0, 1, 2)

        extract_traces_button = QPushButton('Extract traces')
        extract_traces_button.clicked.connect(self.extract_traces)
        controls_layout.addWidget(extract_traces_button, 3, 0, 1, 2)

        self.controls = QWidget()
        self.controls.setLayout(controls_layout)
        self.controls.setMinimumWidth(200)


        extraction_layout = QHBoxLayout()
        extraction_layout.addWidget(self.image)
        extraction_layout.addWidget(self.controls)




        # self.selection = QTableWidget()
        # self.selection.setRowCount(5)
        # self.selection.setColumnCount(4)


        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)

        tab1 = QWidget(self)
        tab1.setLayout(extraction_layout)
        tabs.addTab(tab1, 'Movie')
        self.traces = TracePlotWindow(parent=self, width=4, height=3, show=False)
        tabs.addTab(self.traces, 'Traces')
        self.selection = SelectionWidget(parent=self)
        tabs.addTab(self.selection, 'Selection (beta)')
        tabs.currentChanged.connect(self.setTabFocus)

        experiment_layout = QVBoxLayout()

        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.refresh)
        experiment_layout.addWidget(refresh_button)

        experiment_layout.addWidget(self.tree)

        layout = QHBoxLayout()
        layout.addLayout(experiment_layout)
        layout.addWidget(tabs)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()

        # self.experiment = Experiment(
        #     r'D:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
        self.experiment = pp.Experiment(main_path, main_window=self)
        self.addExperiment(self.experiment)
        self.traces.save_path = self.experiment.analysis_path.joinpath('Trace_plots')

    def keyPressEvent(self, e):
        self.traces.keyPressEvent(e)

    def setTabFocus(self, e):
        if e == 0:
            self.image.setFocus()
        if e == 1:
            self.traces.setFocus()

    def midChange(self, input):
        input = int(input)
        self.experiment.configuration['find_coordinates']['peak_finding']['minimum_intensity_difference'] = input
        self.experiment.configuration.save()

    def perform_mapping(self, t):
        print(t)
        selected_files = self.experiment.selectedFiles
        if selected_files:
            selected_files.serial.perform_mapping()
            self.image_canvas.refresh()
            plt.show()

    def find_coordinates(self):
        selected_files = self.experiment.selectedFiles
        if selected_files:
            selected_files.movie.determine_spatial_background_correction(use_existing=True)
            selected_files.find_coordinates()
            self.image_canvas.refresh()
            self.update_plots()

    def extract_traces(self):
        selected_files = self.experiment.selectedFiles
        if selected_files:
            selected_files.extract_traces()
            # self.image_canvas.refresh()
            self.update_plots()

    def onItemChange(self, item):
        if isinstance(item.data(), File):
            file = item.data()
            file.isSelected = (True if item.checkState() == Qt.Checked else False)
            print(f'{file}: {file.isSelected}')

        else:
            self.update = False
            for i in range(item.rowCount()):
                item.child(i).setCheckState(item.checkState())
            self.update = True

        if self.update:
            self.update_plots()

    def update_plots(self):
        selected_files = self.experiment.selectedFiles + [None]
        self.image_canvas.file = selected_files[0]
        if selected_files[0] is not None:
            self.traces.dataset = selected_files[0].dataset
            self.selection.file = selected_files[0]
        else:
            self.traces.dataset = None
            self.selection.file = None

    def addExperiment(self, experiment):

        # experiment = Experiment(r'D:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
        #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy')
        self.root.appendRow([
                QStandardItem(experiment.name),
                QStandardItem(0),
            ])
        experimentNode = self.root.child(self.root.rowCount() - 1)
        for file in experiment.files:
            print('addfile'+file.name)
            self.addFile(file, experimentNode)

        self.tree.expandAll()

        print('add')

    def addFile(self, file, experimentNode):
        folders = file.relativePath.parts

        parentItem = experimentNode
        parentItem.setCheckable(True)
        for folder in folders:

            # Get the folderItems and folder names for the current folderItem
            nodeItems = [parentItem.child(i) for i in range(parentItem.rowCount())]# if item.type == 'folder']
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
        if file.isSelected:
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)
        item.setData(file)
        #self.FileItems.append(item)

        # self.insertDataIntoColumns(item)

        return item

    def refresh(self):
        self.root.removeRows(0, 1)
        self.experiment = Experiment(self.experiment.main_path)
        self.addExperiment(self.experiment)


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None, width=14, height=7, dpi=100):
        self.figure = mpl.figure.Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)  # , figsize=(2, 2))
        super().__init__(self.figure)
        self.parent = parent

        # self.axis = self.figure.gca()

        self._file = None

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        if file is not None and file is not self._file:
            self._file = file
            self.refresh()
        elif file is None:
            self._file = None
            self.figure.clf()
            self.draw()

    def refresh(self):
        self.figure.clf()
        self._file.movie.determine_spatial_background_correction(use_existing=True)
        self._file.show_coordinates_in_image(figure=self.figure)
        self.draw()


if __name__ == '__main__':
    from multiprocessing import Process, freeze_support
    freeze_support()

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()
