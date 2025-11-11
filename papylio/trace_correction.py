import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from papylio.file import calculate_FRET, calculate_stoichiometry, calculate_intensity_total

from PySide2 import QtWidgets
from PySide2.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout, QLineEdit, QLabel
from PySide2.QtGui import QKeySequence
from PySide2.QtCore import Qt

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure, GridSpec

import sys
import time

import numpy as np

# from matplotlib.backends.qt_compat import QtWidgets
# from matplotlib.backends.backend_qtagg import (
#     FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
# from matplotlib.figure import Figure

def trace_correction(intensity, background_correction=None, alpha_correction=None,
                     gamma_correction=None):
    intensity = intensity.copy()
    if background_correction is not None:
        intensity[dict(channel=0)] -= background_correction[0]
        intensity[dict(channel=1)] -= background_correction[1]
        intensity.attrs['background_correction'] = background_correction
    if gamma_correction is not None:
        intensity[dict(channel=0)] *= gamma_correction
        intensity.attrs['gamma_correction'] = gamma_correction
    if alpha_correction is not None:
        intensity[dict(channel=0)] += alpha_correction * intensity[dict(channel=0)]
        intensity[dict(channel=1)] -= alpha_correction * intensity[dict(channel=0)]
        intensity.attrs['alpha_correction'] = alpha_correction
    # if delta_correction is not None:
    #     intensity[dict(channel=0)] *= self.delta_correction

    # if beta_correction is not None:
    #     intensity[dict(channel=0)] *= self.beta_correction
    return intensity


# if beta_correction is not None:
#     intensity[dict(channel=0)] *= self.beta_correction


class TraceCorrectionWindow(QWidget):
    def __init__(self, intensity_raw, show=True):
        from papylio.experiment import get_QApplication
        app = get_QApplication()

        super().__init__()

        self.setWindowTitle("Trace correction")

        self.setFixedHeight(600)
        self.setFixedWidth(1500)

        self.intensity_raw = intensity_raw #.values.flatten()
        self.background_correction_green = 0
        self.background_correction_red = 0
        self.alpha_correction = 0
        self.gamma_correction = 1


        self.histogram_intensity_green_intensity_red_kwargs = \
            dict(bins=(1000,1000))

        self.histogram_FRET_intensity_total_kwargs = \
            dict(bins=(1000,1000), range=((-0.05,1.05),(-np.max(self.intensity_total)*0.05,np.max(self.intensity_total))))

        self.histogram_FRET_stoichiometry_kwargs = \
            dict(bins=(1000,1000), range=((-0.05,1.05),(-0.05,1.05)))


        illumination_indices = np.unique(self.intensity_raw.illumination)

        if len(illumination_indices) == 1:
            figure = plt.figure(figsize=(9, 6), layout='constrained')
            self.gs = figure.add_gridspec(1, 2)
        elif len(illumination_indices) == 2:
            figure = plt.figure(figsize=(14, 6), layout='constrained')
            self.gs = figure.add_gridspec(1, 3)
        else:
            raise NotImplementedError('More than two illuminations are not yet supported, feel free to add this feature yourself!')

        # figure = plt.figure(figsize=(14, 6), layout='constrained')
        # self.gs = figure.add_gridspec(1, 3)

        self.axis_G_R = figure.add_subplot(self.gs[0,0])
        self.axis_G_R.set_box_aspect(1)
        self.axis_G_R.set_aspect(1)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self.axis_E_T = figure.add_subplot(self.gs[0,1])
        self.axis_E_T.set_box_aspect(1)
        # divider_E_T = make_axes_locatable(self.axis_E_T)
        # self.axis_E = divider_E_T.append_axes("top", size="25%", pad=0)
        # self.axis_T = divider_E_T.append_axes("right", size="25%", pad=0)


        if len(illumination_indices) == 2:
            self.axis_E_S = figure.add_subplot(self.gs[0, 2], sharex=self.axis_E_T)
            self.axis_E_S.set_box_aspect(1)
            # divider_E_S = make_axes_locatable(self.axis_E_S)
            # self.axis_S = divider_E_S.append_axes("right", size="25%", pad=0)
        else:
            self.axis_E_S = None
            # self.axis_S = None

        # for axis in [self.ax]

        self.canvas = FigureCanvas(figure)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        figure_layout = QtWidgets.QVBoxLayout()
        figure_layout.addWidget(NavigationToolbar(self.canvas, self))
        figure_layout.addWidget(self.canvas)

        control_layout = QtWidgets.QGridLayout()
        corrections = ['background_correction_green', 'background_correction_red', 'alpha_correction', 'gamma_correction']
        self.text_boxes = {}
        for i, correction in enumerate(corrections):
            label = control_layout.addWidget(QLabel(correction), i, 0)
            line_edit = QLineEdit(str(getattr(self, correction)))
            # line_edit.textChanged.connect(lambda text: setattr(self, correction, float(text)))
            self.text_boxes[correction] = line_edit
            line_edit.editingFinished.connect(self.setCorrectionsFromTextBoxes)
            # line_edit.editingFinished.connect(self.plot_histogram_FRET_intensity_total)
            control_layout.addWidget(line_edit, i, 1)


        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)
        main_layout.addLayout(figure_layout)
        main_layout.addLayout(control_layout)

        # dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # layout.addWidget(dynamic_canvas)
        # layout.addWidget(NavigationToolbar(dynamic_canvas, self))




        # self.axis.hist2d(np.zeros((self.FRET.size, self.intensity_total.size)),
        #                  cmin=1)
        # H, x, y = self.histogram_FRET_intensity_total()
        # H[H==0] = np.nan
        # self.axis.imshow(H, origin='lower')
        # self.axis.pcolormesh()
        # fig, ax = plt.subplots()
        # ax.hist2d(self.FRET.values.flatten(), self.intensity_total.values.flatten(), cmin=1, **self.histogram_FRET_intensity_total_kwargs)
        # plt.show()
        # self.plot_histogram_FRET_intensity_total()
        # self.axis.set_xlabel('FRET')
        # self.axis.set_ylabel('Intensity total')
        # self._dynamic_ax = dynamic_canvas.figure.subplots()
        # t = np.linspace(0, 10, 101)
        # # Set up a Line2D.
        # self._line, = self._dynamic_ax.plot(t, np.sin(t + time.time()))
        # self._timer = dynamic_canvas.new_timer(50)
        # self._timer.add_callback(self._update_canvas)
        # self._timer.start()

        if show:
            self.show()
            self.activateWindow()
            self.raise_()
            app.exec_()

        self.update_canvas()

    @property
    def intensity(self):
        return trace_correction(self.intensity_raw, (self.background_correction_green, self.background_correction_red),
                                self.alpha_correction, self.gamma_correction)

    @property
    def intensity_total(self):
        return calculate_intensity_total(self.intensity).sel(frame=self.intensity.illumination==0)

    @property
    def FRET(self):
        return calculate_FRET(self.intensity).sel(frame=self.intensity.illumination==0)

    @property
    def stoichiometry(self):
        return calculate_stoichiometry(self.intensity)

    # def histogram_FRET_intensity_total(self):
    #     FRET = self.FRET.values.flatten()
    #     intensity_total = self.intensity_total.values.flatten()
    #     H, xedges, yedges = np.histogram2d(FRET, intensity_total, **self.histogram_FRET_intensity_total_kwargs)
    #     xcenters = (xedges[1:] + xedges[:-1])/2
    #     ycenters = (yedges[1:] + yedges[:-1]) / 2
    #     return H.T, xcenters, ycenters

    def setCorrectionsFromTextBoxes(self):
        for correction, line_edit in self.text_boxes.items():
            setattr(self, correction, float(line_edit.text()))
            print(float(line_edit.text()))

        self.update_canvas()

    def plot_histogram_intensity_green_intensity_red(self):
        print(self.background_correction_green, self.background_correction_red, self.alpha_correction, self.gamma_correction)

        axis = self.axis_G_R

        intensity = self.intensity

        axis.cla()
        intensity_green = intensity.sel(channel=0, frame=intensity.illumination==0).values.flatten()
        intensity_red = intensity.sel(channel=1, frame=intensity.illumination==0).values.flatten()
        axis.hist2d(intensity_green, intensity_red, cmin=1, **self.histogram_intensity_green_intensity_red_kwargs)
        axis.set_xlabel('Intensity green')
        axis.set_ylabel('Intensity red')
        (xmin, xmax) = axis.get_xlim()
        (ymin, ymax) = axis.get_ylim()
        axis.hlines(y=0, xmin=xmin, xmax=xmax)
        axis.vlines(x=0, ymin=ymin, ymax=ymax)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)


    def plot_histogram_FRET_intensity_total(self):
        print(self.background_correction_green, self.background_correction_red, self.alpha_correction, self.gamma_correction)

        axis = self.axis_E_T

        axis.cla()
        FRET = self.FRET.values.flatten()
        intensity_total = self.intensity_total.values.flatten()
        axis.hist2d(FRET, intensity_total, cmin=1, **self.histogram_FRET_intensity_total_kwargs)
        axis.set_xlabel('FRET')
        axis.set_ylabel('Intensity total')
        (xmin, xmax), (ymin, ymax) = self.histogram_FRET_intensity_total_kwargs['range']
        axis.hlines(y=0, xmin=xmin, xmax=xmax)
        axis.vlines(x=[0,0.5,1], ymin=ymin, ymax=ymax)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)



    def plot_histogram_FRET_stoichiometry(self):

        axis = self.axis_E_S

        axis.cla()
        FRET = self.FRET.values.flatten()
        stoichiometry = self.stoichiometry.values.flatten()
        axis.hist2d(FRET, stoichiometry, cmin=1, **self.histogram_FRET_stoichiometry_kwargs)
        axis.set_xlabel('FRET')
        axis.set_ylabel('Stoichiometry')
        axis.figure.canvas.draw()
        (xmin, xmax), (ymin, ymax) = self.histogram_FRET_stoichiometry_kwargs['range']
        axis.hlines(y=[0,0.5,1], xmin=xmin, xmax=xmax)
        axis.vlines(x=[0,0.5,1], ymin=ymin, ymax=ymax)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)


    def update_canvas(self):
        self.plot_histogram_intensity_green_intensity_red()
        self.plot_histogram_FRET_intensity_total()
        if self.axis_E_S is not None:
            self.plot_histogram_FRET_stoichiometry()
        self.canvas.draw()

        # t = np.linspace(0, 10, 101)
        # # Shift the sinusoid as a function of time.
        # self._line.set_data(t, np.sin(t + time.time()))
        # self._line.figure.canvas.draw()

#
# class TraceCorrectionWindow(QWidget):
#     def __init__(self, traces=None, parent=None, show=True):
#         from papylio.experiment import get_QApplication
#         app = get_QApplication()
#
#         super().__init__()
#         self.parent = parent
#
#         self.traces = traces
#
#
#
#         if show:
#             self.show()
#             self.activateWindow()
#             self.raise_()
#             app.exec_()
#
#         self.setWindowTitle("Trace correction")
#
#         layout = QVBoxLayout(self)
#
#         static_canvas = FigureCanvasQTAgg(Figure(figsize=(5, 3)))
#         # Ideally one would use self.addToolBar here, but it is slightly
#         # incompatible between PyQt6 and other bindings, so we just add the
#         # toolbar as a plain widget instead.
#         layout.addWidget(NavigationToolbar(static_canvas, self))
#         layout.addWidget(static_canvas)
#
#         static_canvas.draw()
#         dynamic_canvas = FigureCanvasQTAgg(Figure(figsize=(5, 3)))
#         layout.addWidget(dynamic_canvas)
#         layout.addWidget(NavigationToolbar(dynamic_canvas, self))
#
#         self._static_ax = static_canvas.figure.subplots()
#         t = np.linspace(0, 10, 501)
#         self._static_ax.plot(t, np.tan(t), ".")
#
#         self._dynamic_ax = dynamic_canvas.figure.subplots()
#         t = np.linspace(0, 10, 101)
#         # Set up a Line2D.
#         self._line, = self._dynamic_ax.plot(t, np.sin(t + time.time()))
#         self._timer = dynamic_canvas.new_timer(50)
#         self._timer.add_callback(self._update_canvas)
#         self._timer.start()
#
#     def _update_canvas(self):
#         t = np.linspace(0, 10, 101)
#         # Shift the sinusoid as a function of time.
#         self._line.set_data(t, np.sin(t + time.time()))
#         self._line.figure.canvas.draw()
#
#
# if __name__ == "__main__":
#     w = TraceCorrectionWindow()
