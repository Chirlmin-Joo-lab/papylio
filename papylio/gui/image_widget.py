import json
from PySide2.QtWidgets import QWidget, QLabel, QVBoxLayout
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_canvas = ImageCanvas(self, width=4, height=4, dpi=100)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        image_toolbar = NavigationToolbar(self.image_canvas, self)
        image_layout = QVBoxLayout()
        image_layout.addWidget(image_toolbar)
        image_layout.addWidget(self.image_canvas)

        # Create a placeholder widget to hold our toolbar and canvas.
        self.setLayout(image_layout)

    @property
    def file(self):
        return self.image_canvas.file

    @file.setter
    def file(self, file):
        self.image_canvas.file = file
        if file is None:
            self.setDisabled(True)
        else:
            self.setDisabled(False)

class ImageCanvas(FigureCanvas):
    """Image canvas widget.

    This class defines the canvas used to display images in the Papylio
    GUI. It is responsible for rendering the image data and updating the
    display when the underlying data changes.
    """
    def __init__(self, parent=None, width=14, height=7, dpi=100):
        self.figure = mpl.figure.Figure(figsize=(width, height), dpi=dpi,
                                        constrained_layout=True)  # , figsize=(2, 2))
        super().__init__(self.figure)
        self.parent = parent
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
        self.file.movie.determine_spatial_background_correction(use_existing=True)
        if self.file.coordinates is not None and 'configuration' in self.file.coordinates.attrs:
            self.file.experiment.configuration['projection_image'] = json.loads(self.file.coordinates.attrs['configuration'])['projection_image']
        self.file.show_coordinates_in_image(figure=self.figure)
        self.draw()