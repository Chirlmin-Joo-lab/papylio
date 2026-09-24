import json
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from matplotlib.figure import Figure
import matplotlib as mpl
# from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from papylio.movie.movie import Movie

import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# class ImageWidget(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.image_canvas = ImageCanvas(self, width=4, height=4, dpi=100)
#
#         # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
#         image_toolbar = NavigationToolbar(self.image_canvas, self)
#         image_layout = QVBoxLayout()
#         image_layout.addWidget(image_toolbar)
#         image_layout.addWidget(self.image_canvas)
#
#         # Create a placeholder widget to hold our toolbar and canvas.
#         self.setLayout(image_layout)
#
#     @property
#     def file(self):
#         return self.image_canvas.file
#
#     @file.setter
#     def file(self, file):
#         self.image_canvas.file = file
#         if file is None:
#             self.setDisabled(True)
#         else:
#             self.setDisabled(False)

from papylio.file import show_single_image

class ImageWidgetSingle(QWidget):
    """A QSlider that controls a line plot embedded in a matplotlib canvas."""

    def __init__(self, image, parent=None, figure=None):
        super().__init__(parent)

        # --- Matplotlib figure/canvas ---

        self.figure = figure if figure is not None else Figure(figsize=(4, 4))#, dpi=100)
        self.axes = None
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # self.ax = self.figure.add_subplot(111)

        # self.x = np.linspace(0, 2 * np.pi, 500)
        # (self.line,) = self.ax.plot(self.x, np.sin(self.x))
        # self.ax.set_ylim(-1.2, 1.2)
        # self.ax.set_xlabel("x")
        # self.ax.set_ylabel("sin(f · x)")
        # self.figure.tight_layout()

        # --- Slider ---
        self.slider = QSlider(Qt.Horizontal)

        self.slider_label = QLabel()
        self.slider_label.setMinimumWidth(80)
    #
        # --- Layout ---
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(self.slider_label)
        #
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(slider_row)

        # --- Signals ---
        self.slider.valueChanged.connect(self.update_frame)

        self.image = image


    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if isinstance(image, Movie):
            number_of_frames = image.number_of_frames
        else:
            if image.ndim == 3:
                image = image[None, :, :, :]
            number_of_frames = image.shape[0]
        self._image = image

        if self.axes is None:
            _, self.axes = show_single_image(self.get_frame(0), figure=self.figure)
        self.set_slider_range(number_of_frames)

    def get_frame(self, value: int):
        if isinstance(self.image, Movie):
            return self.image.read_frame(value)#, apply_corrections=False)
        else:
            return self.image[value]

    def set_slider_range(self, number_of_frames):
        self.slider.setRange(0, number_of_frames-1)  # frequency
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(0)
        self.update_frame(self.slider.value())

    def update_frame(self, value: int):
        for channel, axis in enumerate(self.axes):
            axis.images[0].set_data(self.get_frame(value)[channel])
        self.slider_label.setText(f"f = {value}")
        self.canvas.draw_idle()



#
# class ImageCanvas(FigureCanvas):
#     """Image canvas widget.
#
#     This class defines the canvas used to display images in the Papylio
#     GUI. It is responsible for rendering the image data and updating the
#     display when the underlying data changes.
#     """
#     def __init__(self, parent=None, width=14, height=7, dpi=100):
#         self.figure = mpl.figure.Figure(figsize=(width, height), dpi=dpi,
#                                         constrained_layout=True)  # , figsize=(2, 2))
#         super().__init__(self.figure)
#         self.parent = parent
#         self._file = None
#
#     @property
#     def file(self):
#         return self._file
#
#     @file.setter
#     def file(self, file):
#         if file is not None and file is not self._file:
#             self._file = file
#             self.refresh()
#         elif file is None:
#             self._file = None
#             self.figure.clf()
#             self.draw()
#
#     def refresh(self):
#         self.figure.clf()
#         self.file.movie.determine_spatial_background_correction(use_existing=True)
#         if self.file.coordinates is not None and 'configuration' in self.file.coordinates.attrs:
#             self.file.experiment.configuration['projection_image'] = json.loads(self.file.coordinates.attrs['configuration'])['projection_image']
#         self.file.show_coordinates_in_image(figure=self.figure)
#         self.draw()

