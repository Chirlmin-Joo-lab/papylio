import json
from PySide2.QtWidgets import QWidget, QSpinBox, QVBoxLayout, QLabel, QHBoxLayout, QFormLayout
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from papylio.gui.common_layouts import (Expander, HelpDialog,Group_Box,
                                        build_control_layouts,make_push_button,
                                        build_form,build_parameters_input, get_button_value,
                                        deep_get_config)

class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_canvas = ImageCanvas(self, width=8, height=8, dpi=100)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        image_toolbar = NavigationToolbar(self.image_canvas, self)

        image_top_bar = QHBoxLayout()
        image_top_bar.addWidget(image_toolbar, 0.5)

        #add highlight
        self.button_molecule_index = QSpinBox()
        self.button_molecule_index.setRange(0, 5000)
        self.button_molecule_index.setValue(0)
        self.button_molecule_index.valueChanged.connect(self.image_canvas.set_highlighted_molecule)

        image_top_bar.addWidget(QLabel("highlighted molecule"), 0.1)
        image_top_bar.addWidget(self.button_molecule_index, 0.15)

        imaging_controls = build_control_layouts([
            make_push_button('Help', self.show_image_help, None)])
        image_top_bar.addWidget(imaging_controls)

        image_layout = QVBoxLayout()
        image_layout.addLayout(image_top_bar)
        image_layout.addWidget(self.image_canvas)

        #todo: if this image tab is popped up,
        # ..refresh it (to have last molecule there but not do this while scrolling traces)
        #(now it is only when selected file is swapped)

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
        self.highlighted_molecule=None
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

    def set_highlighted_molecule(self, value):
        self.highlighted_molecule=value
        self.refresh()



    def refresh(self):
        self.figure.clf()
        if self.file is None:
            return
        self.file.movie.determine_spatial_background_correction(use_existing=True)
        if self.file.coordinates is not None and 'configuration' in self.file.coordinates.attrs:
            self.file.experiment.configuration['projection_image'] = json.loads(self.file.coordinates.attrs['configuration'])['projection_image']
        highlighted= [False] * self.file.number_of_molecules
        if self.highlighted_molecule is not None:
            highlighted[self.highlighted_molecule] = True
            self.parent.button_molecule_index.setValue(self.highlighted_molecule)
        self.file.show_coordinates_in_image(figure=self.figure, highlighted=highlighted)
        self.draw()

