
from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
    QComboBox, QLineEdit, QSpinBox, QFormLayout, QButtonGroup, QRadioButton, QCheckBox, QMessageBox
from PySide2.QtCore import Qt, Signal
from papylio import File
from papylio.gui.common_layouts import (Expander, HelpDialog,Group_Box,
                                        build_control_layouts,make_push_button,
                                        build_form,build_parameters_input, get_button_value)

#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

#for dynamic method registry:
from scipy.ndimage import gaussian_filter,median_filter,minimum_filter

"""
There are three types of background subtraction that can be performed. 
A temporal background subtraction that corrects variations in background over time 
(but not in x and y); 

a spatial background correction that corrects variations over the x and y 
(but not over time); 

and a general background correction, which is a single value for the entire movie. 

Any of these can be applied. 
If more than one are used, they have to be determined in the order 
1) temporal, 
2) spatial, 
3) single_value
 
... as this is also the order in which they are applied. 
Determining the corrections in a different order, 
will delete the corrections that should have been determined later.
(from Papylio docs, IS)

"""
class MovieCorrectionsWidget(QWidget):
    def __init__(self, parent=None):
        super(MovieCorrectionsWidget, self).__init__(parent)

        self.parent = parent
        self.methods_spatial_background = {}
        self.method_forms_spatial_background = {}


        #movie corrections box 1: temporal ---------------------------------
        self.frame_temporal_correction = Group_Box(title="Temporal", highlight=False)
        self.frame_temporal_correction.setCheckable(True)
        self.frame_temporal_correction.setChecked(False)

        self.frame_temporal_correction.setToolTip('"see help')
        temporal_correction_layout = QFormLayout(self.frame_temporal_correction)
        #method:
        self.method_temporal = QComboBox()
        self.method_temporal.setToolTip("Choose method")
        self.method_temporal.addItems(['mean', 'median', 'fit_background_peak'])
        # fill box:
        temporal_correction_layout.addRow("method:", self.method_temporal)

        # box 2: spatial correction. Box is composed from partly dynamic, partly fixed entry fields
        self.frame_spatial_correction = Group_Box(title="Spatial", highlight=False)
        self.frame_spatial_correction.setCheckable(True)
        self.frame_spatial_correction.setChecked(False)

        self.frame_spatial_correction.setToolTip('"see help')
        form_spatial_background = QFormLayout(self.frame_spatial_correction)
        # --- Method selector spatial background---
        self.method_selector_spatial_background = QComboBox()
        self.method_selector_spatial_background.setToolTip("Choose peak_find_method")
        self.method_selector_spatial_background.currentTextChanged.connect(self._update_method_panel_spatial_background)
        form_spatial_background.addRow("Method:", self.method_selector_spatial_background)
        # --- Dynamic options container, depends on method chosen ---
        self.stack_spatial_background = QWidget()
        self.stack_spatial_background_layout = QVBoxLayout(self.stack_spatial_background)
        self.stack_spatial_background_layout.setContentsMargins(0, 0, 0, 0)
        form_spatial_background.addRow("Options:", self.stack_spatial_background)
        # frame range:
        self.button_spatial_frame_range = QLineEdit()
        self.button_spatial_frame_range.setText("[0, 20]")
        form_spatial_background.addRow("frame range", self.button_spatial_frame_range)

        # movie corrections box 3: general ---------------------------------
        self.frame_general_correction = Group_Box(title="General", highlight=False)
        self.frame_general_correction.setCheckable(True)
        self.frame_general_correction.setChecked(False)

        self.frame_general_correction.setToolTip('"see help')
        general_correction_layout = QFormLayout(self.frame_general_correction)
        # method:
        self.method_general = QComboBox()
        self.method_general.setToolTip("Choose method")
        self.method_general.addItems(['mean', 'median', 'fit_background_peak'])
        general_correction_layout.addRow("method:", self.method_general)
        # frame range:
        self.button_general_frame_range = QLineEdit()
        self.button_general_frame_range.setText("[0, 20]")
        general_correction_layout.addRow("frame range", self.button_general_frame_range)


        this_tab_layout = QHBoxLayout()
        this_tab_layout.setAlignment(Qt.AlignLeft)
        this_tab_layout.addWidget(self.frame_temporal_correction)
        this_tab_layout.addWidget(self.frame_spatial_correction)
        this_tab_layout.addWidget(self.frame_general_correction)


        # main action:
        start_help_button = build_control_layouts([
            make_push_button('Apply', self.apply_corrections, "Apply correction(s)"),
            make_push_button('Help', self.show_main_help, None)])



        start_tab_layout = QVBoxLayout()
        start_tab_layout.addLayout(this_tab_layout)
        start_tab_layout.addStretch()
        start_tab_layout.addWidget(start_help_button)
        self.setLayout(start_tab_layout)

        self.file = None
        self.experiment = None

        # TODO: building
        # collect spatial filter methods for building flexible GUI forms
        #'median_filter', 'gaussian_filter', 'minimum_filter'
        self.register_method_for_spatial_background('median_filter', median_filter)
        self.register_method_for_spatial_background('minimum_filter', minimum_filter)
        self.register_method_for_spatial_background('gaussian_filter', gaussian_filter)

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        self._file = file

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        if experiment is not None:
            self.update_button_settings()

    def apply_corrections(self):
        file = self.parent.experiment.selectedFiles[0]
        # file=self.file

        if self.frame_temporal_correction.isChecked(): #not skip
            mth_t=get_button_value(self.method_temporal)
            file.movie.determine_temporal_background_correction(method=mth_t)
            print('temporal correction done')
        if self.frame_spatial_correction.isChecked():
            # spatial background (Gui_box 2):
            method_name_spatial_background = self.method_selector_spatial_background.currentText()
            _, inputs_spatial_background = self.method_forms_spatial_background[method_name_spatial_background]
            # kwargs for peak finding
            frs_s = get_button_value(self.button_spatial_frame_range)
            spatial_background_kwargs = build_parameters_input(method_name_spatial_background, inputs_spatial_background)
            file.movie.determine_spatial_background_correction(frame_range=frs_s, **spatial_background_kwargs)
            print('spatial correction done')
        if self.frame_general_correction.isChecked():
            mth_g = get_button_value(self.method_general)
            frs_g = get_button_value(self.button_general_frame_range)
            print(frs_g)
            file.movie.determine_general_background_correction(method=mth_g, frame_range=frs_g)
            print('fixed-value correction done')

    def register_method_for_spatial_background(self, name, func):
        """Register a peak finding method, introspect arguments,
        and build forms for spot_detection"""
        #skips and defaults:
        skip_inputs=['input', 'output','footprint', 'origin', 'cval', 'reflect', 'truncate', 'mode']
        defaults = {"size": "15", 'sigma': "10"}
        form_widget_spatial_background, inputs_spatial_background = build_form(func,skip_inputs, defaults)
        self.methods_spatial_background[name] = func
        self.method_forms_spatial_background[name] = (form_widget_spatial_background, inputs_spatial_background)
        self.method_selector_spatial_background.addItem(name) #add options to appropriate selector box
        if self.method_selector_spatial_background.count() == 1: # First registered method becomes default
            self._update_method_panel_spatial_background(name)


    def _update_method_panel_spatial_background(self, name):
        # Clear the old form
        for i in reversed(range(self.stack_spatial_background_layout.count())):
            widget = self.stack_spatial_background_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        # Add new form
        if name in self.method_forms_spatial_background:
            form_widget, _ = self.method_forms_spatial_background[name]
            self.stack_spatial_background_layout.addWidget(form_widget)

    def show_main_help(self):
        help_text = """
                <html>
                  <body style="font-family: sans-serif; font-size: 10pt;">

                    <h2>Background corrections</h2>

                    <p>
                    There are three types of background subtraction, to be performed in this order: 
                    </p>
                    
                    <p>
                    <ol> 
                        <li> Temporal background subtraction 
                        <li> Spatial background correction
                        <li> Single-value background correction
                    </ol>    
                    </p>
                    
                    <h3>Temporal background subtraction</h3>
                    <p> 
                    A temporal background subtraction that corrects variations in background over time (but not in x and y), such as the effect of bleaching.
                    Option 'median' will use the median of every frame to do so, analogous for other options.
                    </p>
                    
                    <h3>Spatial background subtraction</h3>
                    <p> 
                    Spatial background filtering uses local averaging by chosen method, with a window size as specified, over range of frames as specified
                    </p>
                    
                    <h3>General background subtraction</h3>
                    <p> 
                    Use a single value to subtract for all frames. 
                    Option 'median' will use the median of selected frames to do so, analogous for other options.
                    </p>
                    
                    
                    <p>
                      For background, see
                      <a href="https://papylio.readthedocs.io/en/stable/user_guide/background_subtraction.html">
                        Background Subtraction</a>
                      </a>.
                    </p>
                  </body>
                </html>
                """
        self.help_dialog = HelpDialog(self, help_text)
        # dialog.exec_()  # modal
        self.help_dialog.show()
