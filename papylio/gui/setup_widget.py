
from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
    QComboBox, QLineEdit, QSpinBox, QFormLayout, QButtonGroup, QRadioButton, QLabel, QMessageBox
from PySide2.QtCore import Qt, Signal
from papylio import File
from papylio.gui.common_layouts import (Expander, HelpDialog,Group_Box,
                                        build_control_layouts,make_push_button,
                                        build_form,build_parameters_input, get_button_value)
from papylio.movie.movie import Channel
#for registry:
from papylio.peak_finding import (find_peaks_absolute_threshold,
                                  find_peaks_adaptive_threshold,
                                  find_peaks_local_maximum,
                                  find_peaks_local_maximum_auto,
                                  find_peaks_relative_local_maximum)

#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

class SetUpWidget(QWidget):
    def __init__(self, parent=None):
        super(SetUpWidget, self).__init__(parent)

        # self.parent = parent

        # 1 movie settings box ---------------------------------
        frame_movie = Group_Box(title="Movie", highlight=False)
        frame_movie.setToolTip('"User alert: see help or pop-up. Settings source: local config.yml')
        frame_movie.setStyleSheet("""
        QGroupBox {
            border: 1px solid red;
            border-radius: 5px;
            margin-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        """)

        movie_setup_layout = QFormLayout(frame_movie)
        # 1.1 channels:
        self.button_movie_rotation = QSpinBox()
        self.button_movie_rotation.setRange(-3, 3)
        self.button_movie_rotation.setValue(0)
        self.button_movie_rotation.valueChanged.connect(self.pass_buttons_to_config_for_setup)
        self.button_movie_rotation.setToolTip('If changed, remove all analyzed data and projection images')
        movie_setup_layout.addRow("Rotation", self.button_movie_rotation)
        # movie_setup_layout.addRow("", QLabel("If changed, remove all analyzed data and projection images"))

        self.number_of_channels_selector_1 = QRadioButton("1")
        self.number_of_channels_selector_1.setToolTip('This needs to be manually set every time on reopening the gui')

        self.number_of_channels_selector_2 = QRadioButton("2")
        self.number_of_channels_selector_2.setChecked(True)
        self.number_of_channels_selector_2.setToolTip('This needs to be manually set every time on reopening the gui')

        self.channel_selector = QButtonGroup()
        self.channel_selector.addButton(self.number_of_channels_selector_1, 1)
        self.channel_selector.addButton(self.number_of_channels_selector_2, 2)
        # self.channel_selector.setToolTip("Choose number of channels")
        self.channel_selector.buttonClicked.connect(self.on_channel_selection_changed)

        channel_layout = QHBoxLayout()
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.addWidget(self.number_of_channels_selector_1)
        channel_layout.addWidget(self.number_of_channels_selector_2)

        movie_setup_layout.addRow("Number of channels", channel_layout)


        advanced_layout = QHBoxLayout()
        advanced_layout.setAlignment(Qt.AlignLeft)
        advanced_layout.addWidget(frame_movie)

        # build panel layout:
        # setup_advanced = Expander("Advanced")
        # setup_advanced.setContentLayout(advanced_layout)

        start_help_button = build_control_layouts([
            make_push_button('Help', self.show_main_help, None)])

        start_tab_layout = QVBoxLayout()
        #TODO: development: keep invisible as long as it doesn't function:
        # start_tab_layout.addWidget(setup_advanced)
        start_tab_layout.addLayout(advanced_layout)
        start_tab_layout.addStretch()
        start_tab_layout.addWidget(start_help_button)
        self.setLayout(start_tab_layout)

        self.file = None
        self.experiment = None

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        self._file = file
        # if file is None:
        #     self.setDisabled(True)
        # else:
        #     self.setDisabled(False)
        #     self.update_button_settings()

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        if experiment is not None:
            self.update_button_settings()

    def update_button_settings(self):
        self.button_movie_rotation.blockSignals(True)
        self.button_movie_rotation.setValue(self.experiment.configuration['movie']['rot90'])
        self.button_movie_rotation.blockSignals(False)

    def pass_buttons_to_config_for_setup(self, value):
    #line up 'classic config' with buttons in gui.
        self.experiment.configuration['movie']['rot90'] = value
        self.experiment.configuration.save()
        self.experiment.files.movie.rot90 = value

        QMessageBox.warning(self, "Rotation change", "User alert: please first unselect all files, then delete all previously generated projection images [*ave*.tif] and datafiles [.nc]. See also 'Help'")
        # self.file.movie.rot90 = self.button_movie_rotation.value()

    def on_channel_selection_changed(self, button):
        id = self.channel_selector.id(button)
        if id == 1:
            for file in self.experiment.files:
                file.movie.channels = [Channel(file.movie, 'green', 'g', other_names=['donor', 'd'])]
                file.movie.channel_arrangement = [[[0, ]]]
        if id == 2:
            for file in self.experiment.files:
                file.movie.channels = [Channel(file.movie, 'green', 'g', other_names=['donor', 'd']),
                                       Channel(file.movie, 'red', 'r', other_names=['acceptor', 'a'])]
                file.movie.channel_arrangement = [[[0, 1]]]

    def show_main_help(self):
        help_text = """
                <html>
                  <body style="font-family: sans-serif; font-size: 10pt;">

                    <h2>Welcome</h2>

                    <p>
                      This gui is based on the Papylio framework
                    </p>

                    <ul>
                      <li>Select and view movies and variables in the top panel</li>
                      <li>Walk the pipeline via the tabs in the bottom panel</li>
                    </ul>

                    <p>
                      For background, see
                      <a href="https://papylio.readthedocs.io/en/stable/user_guide/index.html">
                        Papylio documentation
                      </a>.
                    </p>

                    <h3>General tips</h3>

                    <p>
                      <ul>
                        <li>Hover over box for settings source [1]. </li>
                        <li>Hover over buttons for help notes.</li>
                        <li>Find more detailed info under the 'Help' buttons per tab</li>
                    </ul>
                    
                    [1] Note: as-loaded settings may have different sources. All used settings can be found per movie in the corresponding .log files

                    </p>
                    
                    <h3>Known Issues</h3>

                    <p>
                    For known issues in Papylio, see
                      <a href="https://github.com/Chirlmin-Joo-lab/papylio/issues">
                        Papylio Issues
                      </a>.

                    
                    <h3>This tab (Start)</h3>
                    <p>
                      Settings in this panel require extra actions to take effect: 
                    </p>

                    <p>
                      <ol>
                        <li> Delete all previously generated projection images [*ave*.tif] </li>
                        <li> Delete all previously generated related data files [.nc]</li>
                        <li> Change settings (immediately written to config.yml)</li>
                        <li> Unselect-select or 'refresh' </li>
                      </ol>
                    </p>
                    
                    </p>

                  </body>
                </html>
                """
        self.help_dialog = HelpDialog(self, help_text)
        # dialog.exec_()  # modal
        self.help_dialog.show()
