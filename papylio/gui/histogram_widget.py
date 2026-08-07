import sys
import json
from PySide2.QtWidgets import QHBoxLayout,  \
    QPushButton, QTabWidget, QComboBox, QSizePolicy
from PySide2.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide2.QtCore import Qt

from papylio import File
from papylio.gui.common_layouts import (HelpDialog,
                                        build_control_layouts, make_push_button)
from papylio.plotting import wysiwyg_export
from matplotlib.figure import Figure
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.fig_histogram_1D_FRET = Figure(figsize=(14, 3))
        self.histogram_canvas_1D_FRET = HistogramCanvas_1D(
            self.fig_histogram_1D_FRET,
            variable = 'FRET',
            bins = np.arange(-0.05, 1.06, 0.01)
        )

        self.fig_histogram_1D_Intensity = Figure(figsize=(14, 3))
        self.histogram_canvas_1D_Intensity = HistogramCanvas_1D(
            self.fig_histogram_1D_Intensity,
            variable='intensity',
            bins=200
        )

        # self.fig_histogram_2D = Figure(figsize=(14, 3))
        # self.histogram_canvas_2D = HistogramCanvas_2D(self.fig_histogram_2D)

        # tab layout
        histogram_tab_layout = QHBoxLayout()
        # histogram_tab_layout.addWidget(self.histogram_canvas_2D)
        histogram_tab_layout.addWidget(self.histogram_canvas_1D_Intensity)
        histogram_tab_layout.addWidget(self.histogram_canvas_1D_FRET)


        # other
        other_graph_layout = QHBoxLayout()


        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)

        # tab1 = QWidget(self)
        # tab1.setLayout(histogram_tab_layout)
        # tabs.addTab(tab1, 'Histogram')
        # tab2 = QWidget(self)
        # tab2.setLayout(other_graph_layout)
        # tabs.addTab(tab2, 'Other')

        # self.kinetics_widget = QWidget()
        # kinetics_layout = QHBoxLayout()
        # kinetics_layout.addWidget(tabs)
        # self.kinetics_widget.setLayout(kinetics_layout)
        self.setLayout(histogram_tab_layout)

        self.file = None

        # Create a placeholder widget to hold our toolbar and canvas.
        #self.setLayout(histogram_layout)

    @property
    def file(self):
        return self.image_canvas.file

    @file.setter
    def file(self, file):
        self.histogram_canvas_1D_FRET.file = file
        self.histogram_canvas_1D_Intensity.file = file
        # self.histogram_canvas_2D.file = file
        if file is None:
            self.setDisabled(True)
        else:
            self.setDisabled(False)

class HistogramCanvas_1D(FigureCanvas):
    def __init__(self, figure=None, parent=None, width=14, height=7, dpi=100, variable='FRET',bins=100):

        if figure is None:
            figure = mpl.figure.Figure(figsize=(width, height), dpi=dpi,
                                        constrained_layout=True)  # , figsize=(2, 2))
        self.fig_histogram_1D = figure

        super().__init__(self.fig_histogram_1D)
        self.parent = parent
        self._file = None
        self.variable = variable
        self.bins = bins

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        if file is not None and file is not self._file and file.dataset is not None:

            # use getattr so variable can change dynamically
            if getattr(file.dataset, self.variable) is not None:
                self._file = file
                self.refresh()
            else:
                self._file = None
                self.figure.clf()
                self.draw()

        elif file is None:
            self._file = None
            self.figure.clf()
            self.draw()

    def refresh(self):
        self.fig_histogram_1D.clear()
        axis = self.fig_histogram_1D.subplots(1, 1, sharex=True)
        self.file.show_histogram(
            variable=self.variable,
            axis=axis,
            bins=self.bins
        )
        self.draw()

class HistogramCanvas_2D(FigureCanvas):
    def __init__(self, parent=None, width=14, height=7, dpi=100):
        self.fig_histogram_2D = mpl.figure.Figure(figsize=(width, height), dpi=dpi,
                                                  constrained_layout=True)  # , figsize=(2, 2))
        super().__init__(self.fig_histogram_2D)
        self.parent = parent
        self._file = None

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        # TODO: this condition should be expanded: file.data (?) should exist
        if file is not None and file is not self._file and file.dataset is not None:
            if file.dataset.FRET is not None:
                self._file = file
                self.refresh()
            else:
                self._file = None
                self.figure.clf()
                self.draw()
        elif file is None:
            self._file = None
            self.figure.clf()
            self.draw()

    def refresh(self):
        # 1D-histogram
        self.fig_histogram_2D.clear()
        axis = self.fig_histogram_2D.subplots(1, 1, sharex=True)
        self.file.histogram_2D_intensity_per_channel(
            ax=axis,
            bins=50,
            show_marginal=False
        )
        self.draw()
