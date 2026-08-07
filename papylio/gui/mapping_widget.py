from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
    QComboBox, QSpinBox, QLabel, QFormLayout
from PySide2.QtCore import Qt, Signal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from papylio import File
from papylio.gui.common_layouts import (Expander, HelpDialog,
                                        build_control_layouts,make_push_button,
                                        build_form,build_parameters_input, Group_Box)
from papylio.peak_finding import (find_peaks_absolute_threshold,
                                  find_peaks_adaptive_threshold,
                                  find_peaks_local_maximum,
                                  find_peaks_local_maximum_auto,
                                  find_peaks_relative_local_maximum)
from papylio.gui.image_widget import ImageCanvas
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

class MappingWidget(QWidget):
    request_top_tab_change = Signal(int)  # send index of tab to activate
    def __init__(self, parent=None):
        super(MappingWidget, self).__init__(parent)
        self.parent = parent
        self.methods_donor = {}
        self.method_forms_donor = {}  # method_name -> (widget, inputs)
        self.methods_acceptor = {}
        self.method_forms_acceptor = {}  #

        # imagery
        # #--------------------------------------------
        self.fig1 = Figure(figsize=(4, 3))
        self.map_overlay_image_canvas = FigureCanvas(self.fig1)
        map_overlay_image_layout = QVBoxLayout()
        map_overlay_image_layout.addWidget(self.map_overlay_image_canvas)
        self.map_overlay_image = QWidget()
        self.map_overlay_image.setLayout(map_overlay_image_layout)
        self.map_overlay_image.setToolTip("Overlay image shows result of mapping")
        self.map_image_canvas = ImageCanvas(self, width=4, height=3, dpi=100)
        self.map_image_canvas.setToolTip("Shows raw image (1st of multiple selection)")
        #map_image_toolbar = NavigationToolbar(self.map_image_canvas, self)
        map_image_layout = QVBoxLayout()

        # Create a placeholder widget to hold our toolbar and canvas.
        self.map_image = QWidget()
        self.map_image.setLayout(map_image_layout)

        #build a flexible form for the donor channel:-------------------------
        frame_donor = Group_Box(title="Donor Spot Detection")
        frame_donor.setToolTip('Settings source: function defaults of selected method')
        form_donor = QFormLayout(frame_donor)
        # --- Method selector donor---
        self.method_selector_donor = QComboBox()
        self.method_selector_donor.setToolTip("Choose peak_find_method")
        self.method_selector_donor.currentTextChanged.connect(self._update_method_panel_donor)
        form_donor.addRow("Method:", self.method_selector_donor)
        # --- Dynamic options container donor ---
        self.stack_donor = QWidget()
        self.stack_donor_layout = QVBoxLayout(self.stack_donor)
        self.stack_donor_layout.setContentsMargins(0, 0, 0, 0)
        form_donor.addRow("Options:", self.stack_donor)

        # build a flexible form for the acceptor channel:-------------------------
        frame_acceptor= Group_Box(title="Acceptor Spot Detection")
        frame_acceptor.setToolTip('Settings source: function defaults of selected method')
        form_acceptor = QFormLayout(frame_acceptor)
        # --- Method selector acceptor---
        self.method_selector_acceptor = QComboBox()
        self.method_selector_acceptor.setToolTip("Choose peak_find_method")
        self.method_selector_acceptor.currentTextChanged.connect(self._update_method_panel_acceptor)
        form_acceptor.addRow("Method:", self.method_selector_acceptor)
        # --- Dynamic options container -acceptor ---
        self.stack_acceptor = QWidget()
        self.stack_acceptor_layout = QVBoxLayout(self.stack_acceptor)
        self.stack_acceptor_layout.setContentsMargins(0, 0, 0, 0)
        form_acceptor.addRow("Options:", self.stack_acceptor)


        #donor_acceptor block
        donor_acceptor_layout = QHBoxLayout()
        donor_acceptor_layout.addWidget(frame_donor)
        donor_acceptor_layout.addWidget(frame_acceptor)

        self.button_map_label = QLabel("method:")
        self.button_map_method_combobox = QComboBox()
        self.button_map_options = ['icp', 'nn']
        self.button_map_method_combobox.addItems(self.button_map_options)
        self.button_map_dist_treshold = QSpinBox()
        self.button_map_dist_treshold.setValue(3)
        self.button_transformation = QComboBox()
        self.button_transformation_options = ['polynomial', 'linear', 'nonlinear']
        self.button_transformation.addItems(self.button_transformation_options)
        self.button_initial_translation = QComboBox()
        self.button_initial_translation_options = ['width/2' , [1024,0]]
        self.button_initial_translation.addItems(self.button_initial_translation_options)

        # advanced mapping grid layout:
        frame = Group_Box(title="Common")
        frame.setToolTip('settings source: hardwired here in GUI')
        map_advanced_layout = QHBoxLayout()
        map_advanced_layout.setAlignment(Qt.AlignLeft)
        map_advanced2_layout = QFormLayout(frame)
        map_advanced2_layout.addRow("Method", self.button_map_method_combobox)
        map_advanced2_layout.addRow("Distance treshold:",self.button_map_dist_treshold)
        map_advanced2_layout.addRow("Transformation_type:", self.button_transformation)
        map_advanced2_layout.addRow("Initial_translation:", self.button_initial_translation)
        map_advanced_layout.addWidget(frame)
        map_advanced_layout.addLayout(donor_acceptor_layout)

        #build panel layout:
        #map_advanced = Expander("Advanced")
        map_advanced = QWidget()
        map_advanced.setLayout(map_advanced_layout)

        #main action:
        map_controls = build_control_layouts(
            [make_push_button('Map', self.perform_mapping,"Map selected file(s)"),
             make_push_button('Help', self.show_help, None)])

        #collect:
        map_controls_layout = QVBoxLayout()
        map_controls_layout.setAlignment(Qt.AlignTop)
        map_controls_layout.addWidget(map_advanced)


        #pack in widget:
        self.map_controls = QWidget()
        self.map_controls.setLayout(map_controls_layout)
        self.map_controls.setMinimumWidth(150)

        #add all to tab:
        mapping_sequence_layout = QHBoxLayout()
        mapping_sequence_layout.addWidget(self.map_controls)
        mapping_sequence_layout.addWidget(self.map_image)
        mapping_sequence_layout.addWidget(self.map_overlay_image)

        #one more
        mapping_tab_layout=QVBoxLayout()
        mapping_tab_layout.addLayout(mapping_sequence_layout, stretch=1)
        mapping_tab_layout.addWidget(map_controls)

        self.setLayout(mapping_tab_layout)

        #collect peak finding methods for building flexible GUI forms
        self.register_method('local-maximum-auto', find_peaks_local_maximum_auto)  #default  in GUI
        self.register_method('absolute-threshold', find_peaks_absolute_threshold)
        self.register_method('adaptive-threshold', find_peaks_adaptive_threshold)
        self.register_method('local-maximum', find_peaks_local_maximum)
        self.register_method('relative-local-maximum', find_peaks_relative_local_maximum)

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
            self.map_image_canvas.file = file

    # -------------------------------------------------------------------------
    # Register methods dynamically and create their forms
    # -------------------------------------------------------------------------
    def register_method(self, name, func):
        """Register a peak finding method, introspect arguments,
        and build forms for donor and acceptor channels"""

        #donor-------------------------:
        form_widget_donor, inputs_donor = build_form(func)
        self.methods_donor[name] = func
        self.method_forms_donor[name] = (form_widget_donor, inputs_donor)
        self.method_selector_donor.addItem(name) #add options to appropriate selector box
        if self.method_selector_donor.count() == 1: # First registered method becomes default
            self._update_method_panel_donor(name)

        # acceptor-------------------------:
        form_widget_acceptor, inputs_acceptor = build_form(func)
        self.methods_acceptor[name] = func
        self.method_forms_acceptor[name] = (form_widget_acceptor, inputs_acceptor)
        self.method_selector_acceptor.addItem(name)  # add options to appropriate selector box
        if self.method_selector_acceptor.count() == 1:  # First registered method becomes default
            self._update_method_panel_acceptor(name)

    def _update_method_panel_donor(self, name):
        # Clear the old form
        for i in reversed(range(self.stack_donor_layout.count())):
            widget = self.stack_donor_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        # Add new form
        if name in self.method_forms_donor:
            form_widget, _ = self.method_forms_donor[name]
            self.stack_donor_layout.addWidget(form_widget)

    def _update_method_panel_acceptor(self, name):
        # Clear the old form
        for i in reversed(range(self.stack_acceptor_layout.count())):
            widget = self.stack_acceptor_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        # Add new form
        if name in self.method_forms_acceptor:
            form_widget, _ = self.method_forms_acceptor[name]
            self.stack_acceptor_layout.addWidget(form_widget)

    def perform_mapping(self, t):
        self.request_top_tab_change.emit(0)
        print(t)
        self.fig1.clear()
        ax1 = self.fig1.add_subplot(111)

        selected_files = self.parent.experiment.selectedFiles

        #get methods and corresponding parameters
        #donor:
        method_name_donor = self.method_selector_donor.currentText()
        _, inputs_donor = self.method_forms_donor[method_name_donor]
        method_name_acceptor = self.method_selector_acceptor.currentText()
        _, inputs_acceptor = self.method_forms_acceptor[method_name_acceptor]

        # reform the chosen method and its args into a kwarg list
        donor_kwargs = build_parameters_input(method_name_donor, inputs_donor)
        acceptor_kwargs = build_parameters_input(method_name_acceptor, inputs_acceptor)

        #jk-read in a default configuration and allocate GUI values:
        mapping_config = self.parent.experiment.configuration['mapping']
        mapping_config['method']= self.button_map_method_combobox.currentText()
        mapping_config['distance_threshold'] = float(self.button_map_dist_treshold.text())
        mapping_config['peak_finding']['donor'] = donor_kwargs
        mapping_config['peak_finding']['acceptor'] = acceptor_kwargs

        self.map_overlay_image_canvas.draw_idle()
        if selected_files:
            plot_file = selected_files[0]
            selected_files.serial.perform_mapping(**mapping_config)
            self.parent.update_plots()
            self.map_image_canvas.refresh()
            plot_file.mapping.show_mapping_transformation(axis=ax1)
            plt.show()

    def show_help(self):
        help_text = """
            <html>
              <body style="font-family: sans-serif; font-size: 10pt;">

                <h2>Mapping</h2>

                <p>
                  Find corresponding XY positions in donor and acceptor channels.
                </p>

                <ol>
                  <li> check if a .MAPPING file already exists. If not, or renew:</li>
                  <li> select a movie suitable for mapping</li>
                  <li> hover over group boxes to inspect settings source </li>
                  <li> adapt settings if wished </li>
                  <li> press 'Map' </li>
                </ul>

                <h3>More help</h3>
                <p>
                See the 
                    <a href="https://papylio.readthedocs.io/en/stable/user_guide/channel_mapping.html">
                mapping with Papylio
                    </a>.
                </p>
                   

                
                </body>
                </html>
                """

        self.help_dialog = HelpDialog(self, help_text)
        # dialog.exec_()  # modal
        self.help_dialog.show()


