import sys

from PySide2.QtCore import QDir
from PySide2.QtWidgets import QTreeView, QFileSystemModel, QApplication, QLabel, QMainWindow, QMenu
from PySide2.QtCore import QAbstractItemModel, QModelIndex
from PySide2 import QtCore

from trace_analysis import Experiment, File

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
#         # experiment = Experiment(r'D:\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
#         #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
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
#         from trace_analysis import Experiment
#         experiment = Experiment(r'D:\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
#         #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
#         #self.model.addExperiment(experiment)
#
#         self.setCentralWidget(self.tree)


import sys
from collections import deque
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *


class FileItemModel(QStandardItemModel):
    def __init__(self):
        super().__init__()


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
    #      #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
    #     #self.model.addExperiment(experiment)
    #
    #     self.setCentralWidget(self.tree)

    def __init__(self):
        super().__init__()

        from trace_analysis import Experiment
        experiment = Experiment(
            r'D:\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')

        self.tree = QTreeView(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        self.model = FileItemModel()
        self.root = self.model.invisibleRootItem()
        self.model.setHorizontalHeaderLabels(['Name', 'Count'])
        self.tree.header().setDefaultSectionSize(180)
        self.tree.setModel(self.model)
        self.addExperiment(experiment)
        self.tree.expandAll()

        self.model.itemChanged.connect(self.onItemChange)

        self.setCentralWidget(self.tree)

    def onItemChange(self, item):
        if isinstance(item.data(), File):
            file = item.data()
            file.isSelected = (True if item.checkState() == Qt.Checked else False)
            print(f'{file}: {file.isSelected}')
        else:
            for i in range(item.rowCount()):
                item.child(i).setCheckState(item.checkState())


    def addExperiment(self, experiment):

        # experiment = Experiment(r'D:\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
        #experiment = Experiment(r'C:\Users\ivoseverins\surfdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy')
        self.root.appendRow([
                QStandardItem(experiment.name),
                QStandardItem(0),
            ])
        experimentNode = self.root.child(self.root.rowCount() - 1)
        for file in experiment.files:
            print('addfile'+file.name)
            self.addFile(file, experimentNode)

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

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
