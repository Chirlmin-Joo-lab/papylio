
import json
from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
    QComboBox, QLineEdit, QSpinBox, QFormLayout
from PySide2.QtCore import Qt, Signal
from papylio import File
from papylio.gui.common_layouts import (Expander, HelpDialog,Group_Box,
                                        build_control_layouts,make_push_button,
                                        build_form,build_parameters_input, get_button_value,
                                        deep_get_config)
#for registry:
from papylio.peak_finding import (find_peaks_absolute_threshold,
                                  find_peaks_adaptive_threshold,
                                  find_peaks_local_maximum,
                                  find_peaks_local_maximum_auto,
                                  find_peaks_relative_local_maximum)

#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

class ExtractionWidget(QWidget):
    request_top_tab_change = Signal(int)  # send index of tab to activate

    def __init__(self, top_tabs, parent=None):
        super(ExtractionWidget, self).__init__(parent)
        #self.top_tabs = top_tabs
        self.parent = parent
        self.methods_spot_detection = {}
        self.method_forms_spot_detection = {}  # method_name -> (widget, inputs)

        #1 general settings box ---------------------------------
        frame_general = Group_Box(title="General", highlight=False)
        frame_general.setToolTip('settings source: last run selected primary file')
        advanced_general_layout = QFormLayout(frame_general)
        #1.1 channels:
        self.button_general_channel = QComboBox()
        self.button_general_channel.addItems(['donor', 'acceptor', 'both'])
        advanced_general_layout.addRow("channels", self.button_general_channel)
        #1.2 illuminations
        self.button_general_illumination = QSpinBox()
        self.button_general_illumination.setValue(0)
        advanced_general_layout.addRow("illumination:", self.button_general_illumination)
        #1.3 method (discarded):
        #self.button_general_method = QComboBox()
        #self.button_general_method.addItems(['by_channel', 'average_channels', 'sum_channels'])
        #advanced_general_layout.addRow("channels:", self.button_general_method)

        #2. projection image box--------------------------------------
        #note that it is a subsection 'proj_image'
        frame_projection = Group_Box(title="Projection Image",highlight=False)
        frame_projection.setToolTip('settings source: last run selected primary file')
        advanced_projection_layout = QFormLayout(frame_projection)
        #2.1 type
        self.button_projection_image_type = QComboBox()
        self.button_projection_image_type.addItems(['average', 'maximum'])
        advanced_projection_layout.addRow("projection type:", self.button_projection_image_type)
        #2.2 frame range
        self.button_projection_image_frame_range = QLineEdit()
        self.button_projection_image_frame_range.setText("[0, 20]")
        advanced_projection_layout.addRow("frame range:", self.button_projection_image_frame_range)
        #2.3 illumination
        self.button_projection_image_illumination= QSpinBox()
        self.button_projection_image_illumination.setValue(0)
        advanced_projection_layout.addRow("illumination:", self.button_projection_image_illumination)

        #3. peakfinding (dynamic form building)
        # build a flexible form for the spot_detection channel:-------------------------
        group_spot_detection = Group_Box(title="Spot Detection")
        group_spot_detection.setToolTip('settings source: function defaults of selected method')
        form_spot_detection = QFormLayout(group_spot_detection)
        # --- Method selector spot_detection---
        self.method_selector_spot_detection = QComboBox()
        self.method_selector_spot_detection.setToolTip("Choose peak_find_method")
        self.method_selector_spot_detection.currentTextChanged.connect(self._update_method_panel_spot_detection)
        form_spot_detection.addRow("Method:", self.method_selector_spot_detection)
        # --- Dynamic options container spot_detection ---
        self.stack_spot_detection = QWidget()
        self.stack_spot_detection_layout = QVBoxLayout(self.stack_spot_detection)
        self.stack_spot_detection_layout.setContentsMargins(0, 0, 0, 0)
        form_spot_detection.addRow("Options:", self.stack_spot_detection)

        #4. coordinate optimization box-------------------------------
        frame_coord_opt = Group_Box(title="Coordinate Optimization",highlight=False)
        frame_coord_opt.setToolTip('settings source: hardwired here in GUI')
        advanced_coord_opt_layout = QFormLayout(frame_coord_opt)
        #4.1 margin
        self.button_coordinates_within_margin = QSpinBox()
        self.button_coordinates_within_margin.setValue(10)
        advanced_coord_opt_layout.addRow("within_margin:",
                                         self.button_coordinates_within_margin)
        #4.2 Gauss fit width
        self.button_coordinates_after_gaussian_fit_width = QSpinBox()
        self.button_coordinates_after_gaussian_fit_width.setValue(3)
        advanced_coord_opt_layout.addRow("gaussian_fit:" ,self.button_coordinates_after_gaussian_fit_width)


        #5. extract_traces box-------------------------------
        frame_extract_traces = Group_Box(title="Extract Traces")
        frame_extract_traces.setToolTip('settings source: hardwired here in GUI')
        advanced_extract_traces_layout = QFormLayout(frame_extract_traces)
        #5.1 obsoletes/not used:

        #background_correction = None,
        #alpha_correction = None,
        #gamma_correction = None

        #5.2 mask
        self.button_extract_mask_size = QLineEdit()
        self.button_extract_mask_size.setPlaceholderText("1")
        self.button_extract_mask_size.setText("1")
        self.button_extract_mask_size.setToolTip("float number or presets: TIR-T, TIR-V, TIR-S 1.5x 2x2, TIR-S 1x 2x2, BN-TIRF")
        advanced_extract_traces_layout.addRow("mask size:", self.button_extract_mask_size)
        #5.3 neighbourhood_size: 11
        self.button_extract_neighbourhood_size= QSpinBox()
        self.button_extract_neighbourhood_size.setValue(11)
        advanced_extract_traces_layout.addRow("neighbourhood_size:", self.button_extract_neighbourhood_size)

        #add general frame to tab layout
        # this one collects the various settings frames in horizontal fashion
        advanced_layout = QHBoxLayout()
        advanced_layout.setAlignment(Qt.AlignLeft)
        advanced_layout.addWidget(frame_general)
        advanced_layout.addWidget(frame_projection)
        advanced_layout.addWidget(group_spot_detection)
        advanced_layout.addWidget(frame_coord_opt)
        advanced_layout.addWidget(frame_extract_traces)

        #build panel layout:
        # extraction_advanced = Expander("Advanced")
        # extraction_advanced.setContentLayout(advanced_layout)


        #main action:
        self.extract_button = make_push_button(
            'Extract traces',
            self.extract_traces,
            "extraction of traces (multiple file select)"
        )

        self.find_coordinates_button= make_push_button(
            'Find coordinates',
            self.find_coordinates,"finding of coordinates (multiple file select)"
        )

        extraction_controls = build_control_layouts([
            self.find_coordinates_button,
            self.extract_button,
            make_push_button('Help', self.show_help, None)])

        #collect:
        extraction_controls_layout = QVBoxLayout()
        # extraction_controls_layout.addWidget(extraction_advanced)
        extraction_controls_layout.addLayout(advanced_layout, stretch=0)
        extraction_controls_layout.addStretch(1)
        extraction_controls_layout.addWidget(extraction_controls)

        # #pack in widget:
        # self.extraction_controls = QWidget()
        # self.extraction_controls.setLayout(extraction_controls_layout)
        # self.extraction_controls.setMinimumWidth(150)
        #
        # #add all to tab:
        # extraction_tab_layout = QHBoxLayout()
        # extraction_tab_layout.addWidget(self.extraction_controls)
        #
        # self.setLayout(extraction_tab_layout)

        self.setLayout(extraction_controls_layout)

        #collect peak finding methods for building flexible GUI forms
        self.register_method_spot_detection('local-maximum-auto', find_peaks_local_maximum_auto)  #default  in GUI
        self.register_method_spot_detection('absolute-threshold', find_peaks_absolute_threshold)
        self.register_method_spot_detection('adaptive-threshold', find_peaks_adaptive_threshold)
        self.register_method_spot_detection('local-maximum', find_peaks_local_maximum)
        self.register_method_spot_detection('relative-local-maximum', find_peaks_relative_local_maximum)

        self.file = None

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        self._file = file
        if file is None:
            self.setDisabled(True)
        else:
            self.setDisabled(False)

    # -------------------------------------------------------------------------
    # Register methods dynamically and create their forms
    # -------------------------------------------------------------------------
    def register_method_spot_detection(self, name, func):
        """Register a peak finding method, introspect arguments,
        and build forms for spot_detection and acceptor channels"""

        #spot_detection-------------------------:
        form_widget_spot_detection, inputs_spot_detection = build_form(func)
        self.methods_spot_detection[name] = func
        self.method_forms_spot_detection[name] = (form_widget_spot_detection, inputs_spot_detection)
        self.method_selector_spot_detection.addItem(name) #add options to appropriate selector box
        if self.method_selector_spot_detection.count() == 1: # First registered method becomes default
            self._update_method_panel_spot_detection(name)


    def _update_method_panel_spot_detection(self, name):
        # Clear the old form
        for i in reversed(range(self.stack_spot_detection_layout.count())):
            widget = self.stack_spot_detection_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        # Add new form
        if name in self.method_forms_spot_detection:
            form_widget, _ = self.method_forms_spot_detection[name]
            self.stack_spot_detection_layout.addWidget(form_widget)


    def find_coordinates(self):
        #switch top_tab:
        self.request_top_tab_change.emit(0)

        selected_files = self.parent.experiment.selectedFiles
        if selected_files:
            #write button values to config
            # spot_detection for extraction (Gui_box 3):
            method_name_spot_detection = self.method_selector_spot_detection.currentText()
            _, inputs_spot_detection = self.method_forms_spot_detection[method_name_spot_detection]
            # kwargs for peak finding
            spot_detection_kwargs = build_parameters_input(method_name_spot_detection, inputs_spot_detection)

            find_coordinates_config=self.pass_buttons_to_config_for_find_coordinates()
            find_coordinates_config['peak_finding'] = spot_detection_kwargs
            config_find_coordinates = {**find_coordinates_config}
            selected_files.movie.determine_spatial_background_correction(use_existing=True)
            selected_files.find_coordinates(**config_find_coordinates)
            self.parent.image.image_canvas.refresh()
            self.parent.update_plots()

    def extract_traces(self):
        self.request_top_tab_change.emit(1) #set viewer to 'traces'
        #TODO: Better to pass the experiment directly to this widget, instead of going through the parent. Also adjust in other places.

        selected_files = self.parent.experiment.selectedFiles
        #here, pass button values to config
        config_extraction = self.pass_buttons_to_config_for_extraction()
        if selected_files:
            selected_files.extract_traces(**config_extraction)
            # self.image_canvas.refresh()
            self.parent.update_plots()

    def set_buttons_from_selected_file(self):
        #this function loads the settings used earlier for the primary file
        # currently the first of selected) and passes these to the appropriate GUI button
        selected_file = self.parent.experiment.selectedFiles[0]
        config_str = selected_file.coordinates.attrs.get('configuration')

        if config_str is not None:
            file_config = json.loads(config_str)
            #1. general
            #note: below button value-to-get was stored as single-element-list

            channels = deep_get_config(file_config, ['channels'])
            if len(channels) == 2:
                self.button_general_channel.setCurrentText('both')
            else:
                self.button_general_channel.setCurrentText((deep_get_config(file_config, ['channels'])[0]))
            self.button_general_illumination.setValue(deep_get_config(file_config, ["illumination"]))
            #self.button_general_method.setCurrentText(deep_get_config(file_config, ["method"]))
            #2. projection
            self.button_projection_image_type.setCurrentText(deep_get_config(file_config,["projection_image", "projection_type"]))
            self.button_projection_image_frame_range.setText(repr(deep_get_config(file_config,["projection_image", "frame_range"])))
            self.button_projection_image_illumination.setValue(deep_get_config(file_config,["projection_image", "illumination"]))

            #3. peak finding:-------------------------
            # TODO find out how w/dynamic : (probably need to re-set box first based on method, then allocate by input index
            #"peak_finding": {
            #"peak_finding"    "method": "local-maximum-auto",
            #"peak_finding"    "fraction_difference": 0.5,
            #"peak_finding"    "filter_neighbourhood_size_min": 10,
            #"peak_finding"    "filter_neighbourhood_size_max": 5

            # 4. coordinate optimization box-------------------------------
            self.button_coordinates_within_margin.setValue(deep_get_config(file_config,
                ["coordinate_optimization","coordinates_within_margin", "margin"]))
            self.button_coordinates_after_gaussian_fit_width.setValue(deep_get_config(file_config,
                ["coordinate_optimization", "coordinates_after_gaussian_fit", "gaussian_width"]))


    def pass_buttons_to_config_for_find_coordinates(self):
    #line up 'classic config' with buttons in gui.
        config_find_coordinates=self.parent.experiment.configuration['find_coordinates']
        #TODO: remove this pre-loading of config
        #1. general box:
        channels = get_button_value(self.button_general_channel)
        if channels == 'both':
            config_find_coordinates['channels'] = ['donor', 'acceptor']
            config_find_coordinates['method'] = 'sum_channels'
        else:
            config_find_coordinates['channels'] = [channels]
            config_find_coordinates['method'] = 'by_channel'
        config_find_coordinates['illumination'] = get_button_value(
            self.button_general_illumination)
        #config_find_coordinates['method'] = get_button_value(
            #self.button_general_method)

        #2. projection image box--------------------------------------
        config_find_coordinates['projection_image']['projection_type'] = get_button_value(
        self.button_projection_image_type)
        config_find_coordinates['projection_image']['frame_range'] = get_button_value(
        self.button_projection_image_frame_range)
        config_find_coordinates['projection_image']['illumination'] = get_button_value(
        self.button_projection_image_illumination)

        #3. peakfinding (from dynamic form building, see extraction def)
        #TODO : how to link file-loaded settings to a dynamic container?

        #4. coordinate optimization box-------------------------------
        config_find_coordinates['coordinate_optimization']['coordinates_within_margin']['margin'] = get_button_value(
        self.button_coordinates_within_margin)
        config_find_coordinates['coordinate_optimization']['coordinates_after_gaussian_fit']['gaussian_width']  = get_button_value(
        self.button_coordinates_after_gaussian_fit_width)

        return config_find_coordinates

    def pass_buttons_to_config_for_extraction(self):
        # Later to be replaced by direct kwargs output for 'extraction'
        #5. extract_traces box-------------------------------
        #config_extraction = self.parent.experiment.configuration['trace_extraction']
        config_extraction = {'mask_size': get_button_value(
            self.button_extract_mask_size), 'neighbourhood_size': get_button_value(
            self.button_extract_neighbourhood_size)}

        return config_extraction

    def show_help(self):
        help_text = """
            <html>
              <body style="font-family: sans-serif; font-size: 10pt;">

                <h2>Extraction</h2>

                <p>
                  Find spot coordinates and extract traces.
                </p>

                <ol>
                  <li> select one or more files </li>
                  <li> hover over group boxes to inspect settings source </li>
                  <li> adapt settings if wished </li>
                  <li> press 'Find coordinates' </li>
                  <li> press 'Extract traces' </li>
                  <li> progress can be seen in Python console </li>
                  
                </ul>

                <p>
                  For more help, see
                  <a href="https://papylio.readthedocs.io/en/stable/user_guide/molecule_localization.html">
                    Molecule Localization in Papylio
                  </a>.
                </p>

                <h3>Example</h3>

                <p>
                  Settings examples can be found in the documentation.
                </p>

              </body>
            </html>
            """

        self.help_dialog = HelpDialog(self, help_text)
        # dialog.exec_()  # modal
        self.help_dialog.show()
