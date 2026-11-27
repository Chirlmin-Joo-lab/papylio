import json
from PySide6.QtWidgets import QWidget, QSpinBox, QVBoxLayout, QLabel, QHBoxLayout, QFormLayout, QVBoxLayout
from matplotlib.figure import Figure
import matplotlib as mpl
# from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from papylio.movie.movie import Movie

from papylio.gui.common_layouts import (Expander, HelpDialog,Group_Box,
                                        build_control_layouts,make_push_button,
                                        build_form,build_parameters_input, get_button_value,
                                        deep_get_config)
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

from papylio.movie.movie import show_single_image

class ImageWidgetSingle(QWidget):
    """A QSlider that controls a line plot embedded in a matplotlib canvas."""

    def __init__(self, image, parent=None, figure=None):
        from papylio.experiment import get_QApplication
        # TODO: Use selection only if it is present.
        app = get_QApplication()
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

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(slider_row)

        # --- Signals ---
        self.slider.valueChanged.connect(self.update_frame)

        self.image = image

        self.show()
        app.exec_()

        # TODO: Reimplement this when adding coordinates
        # image_top_bar = QHBoxLayout()
        # image_top_bar.addWidget(image_toolbar, 0.5)
        #
        # #add highlight
        # self.button_molecule_index = QSpinBox()
        # self.button_molecule_index.setRange(0, 5000)
        # self.button_molecule_index.setValue(0)
        # self.button_molecule_index.valueChanged.connect(self.image_canvas.set_highlighted_molecule)
        #
        # image_top_bar.addWidget(QLabel("highlighted molecule"), 0.1)
        # image_top_bar.addWidget(self.button_molecule_index, 0.15)
        #
        # imaging_controls = build_control_layouts([
        #     make_push_button('Help', self.show_image_help, None)])
        # image_top_bar.addWidget(imaging_controls)
        #
        # #todo: if this image tab is popped up,
        # # ..refresh it (to have last molecule there but not do this while scrolling traces)
        # #(now it is only when selected file is swapped)


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
        for channel, axis in enumerate(self.axes.flatten()):
            axis.images[0].set_data(self.get_frame(value)[channel])
        self.slider_label.setText(f"f = {value}")
        self.canvas.draw_idle()

    def show_image_help(self):
        help_text = """
                <html>
                  <body style="font-family: sans-serif; font-size: 10pt;">
    
                    <h2>Image</h2>
    
                    <p>
                      Shows the current primary image file with detected molecules (if detection was performed)
                    </p>
    
                    <p>
                    <ol>
                      <li> Green spots are currently accepted spots from the 'Selection' menu below</li>
                      <li> Red spots are currently rejected spots from the 'Selection' menu below</li>
                      <li> One 'highlighted' spot is shown in magenta with its index listed below </li>
                      <li> This value is also updated from the index field of the 'Traces' tab - after pressing Enter</li>
                    </ul>
    
                    </p>
    
                  </body>
                </html>
                """
        self.help_dialog = HelpDialog(self, help_text)
        # dialog.exec_()  # modal
        self.help_dialog.show()

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
#         self.highlighted_molecule=None
#         self._file = None
#
#     @property
#     def file(self):
#         return self._file
# #
#     @file.setter
#     def file(self, file):
#         if file is not None and file is not self._file:
#             self._file = file
#             self.refresh()
#         elif file is None:
#             self._file = None
#             self.figure.clf()
#             self.draw()
#         self.highlighted_molecule = None
#
#
#     def set_highlighted_molecule(self, value):
#         self.highlighted_molecule = value
#         self.refresh()
#
#     def refresh(self):
#         self.figure.clf()
#         if self.file is None:
#             return
#         self.file.movie.determine_spatial_background_correction(use_existing=True)
#         if self.file.coordinates is not None and 'configuration' in self.file.coordinates.attrs:
#             self.file.experiment.configuration['projection_image'] = \
#             json.loads(self.file.coordinates.attrs['configuration'])['projection_image']
#         highlighted = [False] * self.file.number_of_molecules
#         if self.highlighted_molecule is not None:
#             highlighted[self.highlighted_molecule] = True
#             self.parent.button_molecule_index.setValue(self.highlighted_molecule)
#         self.file.show_coordinates_in_image(figure=self.figure, highlighted=highlighted)
#         self.draw()