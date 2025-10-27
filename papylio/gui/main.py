import sys
import PySide2
import platform
from PySide2.QtWidgets import QWidget, QCheckBox,QHBoxLayout, QVBoxLayout, QGridLayout, QTreeView, QApplication, QMainWindow, \
    QPushButton, QTabWidget, QTableWidget, QComboBox, QLineEdit
from PySide2.QtGui import QStandardItem, QStandardItemModel, QIcon
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib as mpl
import matplotlib.pyplot as plt
from networkx.algorithms.approximation.matching import min_maximal_matching

import papylio as pp
from papylio import Experiment, File
#separately defined layouts
from papylio.trace_plot import TracePlotWindow
from papylio.gui.alignment_widget import AlignmentWidget
from papylio.gui.selection_widget import SelectionWidget

class MainWindow(QMainWindow):
    def __init__(self, main_path=None):
        super().__init__()

        system = platform.system()
        if system == "Windows":
            extension = 'ico'
        else:  # macOS or Linux
            extension = "png"
        self.setWindowIcon(QIcon("icon."+extension))
        self.setWindowTitle("Papylio v" + pp.__version__ )
        
        from papylio import Experiment
        
        # jk note: here you may just use a fixed path for quick editing
        if 0: #jk's easy-to_find UglySwitch
            # experiment = Experiment(r'C:\Users\myname\personalpaths\Papylio example dataset')
            self.experiment = Experiment(
                 r'C:\Users\jkerssemakers\OneDrive - Delft University of Technology\ChJ_recent\Papylio example dataset')
        else:
            self.experiment = Experiment(main_path)

        #I. set up the left section of the GUI,
        # displaying a tree of selectable files in th dataset-directory:
        self.tree = QTreeView(self)
        self.model = QStandardItemModel()
        self.root = self.model.invisibleRootItem()
        self.model.setHorizontalHeaderLabels(['Name', 'Count'])
        self.tree.header().setDefaultSectionSize(180)
        self.tree.setModel(self.model)

        self.tree.setFocusPolicy(Qt.NoFocus)
        self.tree.setFixedWidth(300)
        self.update = True
        self.model.itemChanged.connect(self.onItemChange)


        #set the overall size
        self.image_canvas = ImageCanvas(self, width=8, height=5, dpi=100)

        #menus following Papylio-by-script:
        #Channel_alignment (current label: 'movie')
        #Image_correction (inactive)
        #Molecule_localization (inactive)
        #Trace_extraction (inactive)
        #Trace_evaluation (current label: 'Traces')
        #Trace_correction (inactive)
        #Molecule_selection (current label: 'Selection (beta)' )
        #Trace_classification (tbd)
        #Kinetics_quantification (tbd)

        #Current menus:
        #II. Build the 'extraction' tab, containing an image display (including a toolbar)  and action buttons.
        # a. Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second.
        image_toolbar = NavigationToolbar(self.image_canvas, self)
        image_layout = QVBoxLayout()
        image_layout.addWidget(image_toolbar)
        image_layout.addWidget(self.image_canvas)

        # b. Create a placeholder widget to hold our toolbar and the canvas as defined above.
        #toolbar is the column of control buttons to the right
        self.image = QWidget()
        self.image.setLayout(image_layout)
        
        #set the current menu layouts
        alignment_controls_layout = QGridLayout()
        alignment_controls_layout.setAlignment(Qt.AlignTop)

        localization_controls_layout = QGridLayout()
        localization_controls_layout.setAlignment(Qt.AlignTop)

        extraction_controls_layout = QGridLayout()
        extraction_controls_layout.setAlignment(Qt.AlignTop)

        #trace_selection_layout=SelectionWidget()

        # c. The  main action buttons are defined here (for now, they are grouped together:
        # set buttons:
        perform_mapping_button = QPushButton('Perform mapping')
        perform_mapping_button.setToolTip("Use this once for the bead slide")
        perform_mapping_button.clicked.connect(self.perform_mapping)

        find_molecules_button = QPushButton('Find coordinates')
        find_molecules_button.setToolTip("select movie(s) in tree pane and press to obtain XY per molecule")
        find_molecules_button.clicked.connect(self.find_coordinates)

        extract_traces_button = QPushButton('Extract traces')
        extract_traces_button.setToolTip("select movie(s) in tree pane and press to obtain trace per molecule")
        extract_traces_button.clicked.connect(self.extract_traces)

        #for now, we place buttons at their original location (in the 'movie' tab)
        alignment_controls_layout.addWidget(perform_mapping_button, 1, 0, 1, 2)
        alignment_controls_layout.addWidget(find_molecules_button, 2, 0, 1, 2)
        alignment_controls_layout.addWidget(extract_traces_button, 3, 0, 1, 2)

        #set control panel:
        self.alignment_controls = QWidget()
        self.alignment_controls.setLayout(alignment_controls_layout)
        self.alignment_controls.setMinimumWidth(200)

        #combine image and control panel:
        alignment_layout = QHBoxLayout()
        alignment_layout.addWidget(self.image)
        alignment_layout.addWidget(self.alignment_controls)

        # temporally inactive tabs, following the papylio script menu:
        if 0:
            #localize:
            # buttons
            find_molecules_button = QPushButton('go: coordinates')
            find_molecules_button.setToolTip("select movie(s) in tree pane and press to obtain XY per molecule")
            find_molecules_button.clicked.connect(self.find_coordinates)
            localization_controls_layout.addWidget(find_molecules_button, 2, 0, 1, 2)
            # set control panel:
            self.localization_controls = QWidget()
            self.localization_controls.setLayout(localization_controls_layout)
            self.localization_controls.setMinimumWidth(200)
            # build control panel:
            localization_layout = QHBoxLayout()
            localization_layout.addWidget(self.localization_controls)

            # extract
            # buttons:
            extract_traces_button = QPushButton('go: traces')
            extract_traces_button.setToolTip("select movie(s) in tree pane and press to obtain trace per molecule")
            extract_traces_button.clicked.connect(self.extract_traces)
            extraction_controls_layout.addWidget(extract_traces_button, 3, 0, 1, 2)
            # set control panel:
            self.extraction_controls = QWidget()
            self.extraction_controls.setLayout(extraction_controls_layout)
            self.extraction_controls.setMinimumWidth(200)
        if 0:  # inactive tab, following the papylio script menu:
            # build control panel:
            extraction_layout = QHBoxLayout()
            extraction_layout.addWidget(self.extraction_controls)

        # build tabs:
        #we distinguish 'basic' tabs that are always visible,
        # and 'advanced' tabs for more elaborate control
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)

        # Channel_alignment (later might transfer 'movie' functionality to separate widget)
        #self.alignment = AlignmentWidget() [later]
        tab_alignment = QWidget(self)
        tab_alignment.setLayout(alignment_layout)
        tabs.addTab(tab_alignment, 'Movie')
        tab_i = tabs.indexOf(tab_alignment)
        tabs.setTabToolTip(tab_i, "Align the two color channels using one reference image")

        advanced = False
        # Image_correction (advanced) [new]
        if advanced:  #placeholder for conditional 'advanced' menu addin
            self.image_correction = QWidget(self)
            tabs.addTab(self.image_correction, 'Correct I')
            tab_i = tabs.indexOf(self.image_correction)
            tabs.setTabToolTip(tab_i, "Set background corrections for spot detection")

        if 0: #inactive tab, following the papylio script menu:
            # Molecule_localization (basic):
            self.molecule_localization = QWidget(self)
            self.molecule_localization.setLayout(localization_layout)
            tabs.addTab(self.molecule_localization, 'Localize')
            tab_i = tabs.indexOf(self.molecule_localization)
            tabs.setTabToolTip(tab_i, "find XY coordinates of molecules")

        if 0: #inactive tab, following the papylio script menu:
            # Trace_extraction (basic)
            self.trace_extraction = QWidget(self)
            self.trace_extraction.setLayout(extraction_layout)
            tabs.addTab(self.trace_extraction, 'Extract')
            tab_i = tabs.indexOf(self.trace_extraction)
            tabs.setTabToolTip(tab_i, "Get intensity traces per molecule")

        #Trace_evaluation (basic)
        self.trace_evaluation = TracePlotWindow(parent=self, width=4, height=3, show=False,
                                      save_path=self.experiment.analysis_path.joinpath('Trace_plots'))
        tabs.addTab(self.trace_evaluation, 'Traces')
        tab_i = tabs.indexOf(self.trace_evaluation)
        tabs.setTabToolTip(tab_i, "Inspect and select traces")


        # Trace_correction: (advanced)
        if 0: #inactive tab, following the papylio script menu:
            tab_i += 1
            self.trace_correction = QWidget(self)
            tabs.addTab(self.trace_correction, 'Correct II')
            tab_i = tabs.indexOf(self.trace_correction)
            tabs.setTabToolTip(tab_i, "Scale intensity of traces per molecule")

        # trace_selection (basic):
        self.trace_selection = SelectionWidget()
        tab_i += 1
        tabs.addTab(self.trace_selection, 'Select')
        tabs.currentChanged.connect(self.setTabFocus)
        tab_i = tabs.indexOf(self.trace_selection)
        tabs.setTabToolTip(tab_i, "Set threshold parameters for auto-selection of traces")

        if 0:  # inactive tab, following the papylio script menu:
            # Trace_classification: (basic)
            tab_i += 1
            self.trace_classification = QWidget(self)
            tabs.addTab(self.trace_classification, 'Classify')
            tab_i = tabs.indexOf(self.trace_classification)
            tabs.setTabToolTip(tab_i, "Extract molecule states using HMM or other methods")

            # Kinetics_quantification: KIQ (basic)
            tab_i += 1
            self.trace_quantification = QWidget(self)
            tabs.addTab(self.trace_quantification, 'Quantify')
            tab_i = tabs.indexOf(self.trace_quantification)
            tabs.setTabToolTip(tab_i, "Obtain kinetic parameters of molecule states")

        # V. build an 'experiment' pane with tree and refresh button in vertical order:
        refresh_button = QPushButton('Refresh')
        refresh_button.setToolTip("TO DO: refresh adds status color to filenames")
        refresh_button.clicked.connect(self.refresh)

        experiment_layout = QVBoxLayout()
        experiment_layout.addWidget(refresh_button)
        experiment_layout.addWidget(self.tree)

        # VI. now, assemble the various panels
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
            self.trace_evaluation.setFocus()

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
            self.trace_evaluation.dataset = selected_files[0].dataset
            self.trace_selection.file = selected_files[0]
        else:
            self.trace_evaluation.dataset = None
            self.trace_selection.file = None

    def addExperiment(self, experiment):
        #jk_note: when uncommenting a path here, code basically auto-loads:
        # experiment = Experiment(r'C:\Users\myname\personalpaths\Papylio example dataset')
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
        #self.insertDataIntoColumns(item)

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
