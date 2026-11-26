#import sys
#import json
from PySide2.QtWidgets import QHBoxLayout,  \
    QPushButton, QTabWidget, QComboBox, QFormLayout, QWidget, QLineEdit, QVBoxLayout, QSpinBox
from PySide2.QtCore import Qt

#from papylio import File
from papylio.gui.common_layouts import (HelpDialog, Group_Box, get_button_value,
                                        build_control_layouts, make_push_button)
from papylio.plotting import wysiwyg_export
from matplotlib.figure import Figure
#import numpy as np

from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

class KineticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # imagery
        self.fig_dwell_times = Figure(figsize=(14, 3))
        self.dwell_kinetics_canvas = FigureCanvas(self.fig_dwell_times)

        #tuning:
        frame_dwell_options = Group_Box(title="Dwell-time analysis options", highlight=False)
        frame_dwell_options_layout = QFormLayout(frame_dwell_options)


        #method:
        self.dwell_method_combobox = QComboBox()
        dwell_method_items = ['histogram_fit', 'cdf_fit', 'maximum_likelihood_estimation','.....']
        self.dwell_method_combobox.setToolTip('choose method to find dwell times')
        self.dwell_method_combobox.addItems(dwell_method_items)

        frame_dwell_options_layout.addRow("method:",
                                          self.dwell_method_combobox)
        #exponentials:
        self.button_multiple_exponents = QSpinBox()
        self.button_multiple_exponents.setToolTip('choose number of exponents for multiple-exponent-fit (>1)')
        self.button_multiple_exponents.setValue(2)
        frame_dwell_options_layout.addRow("multiple exponents:", self.button_multiple_exponents)

        # plot_range:
        self.button_plot_range = QLineEdit()
        self.button_plot_range.setToolTip('choose plot ranges (s)')
        self.button_plot_range.setText("[(0,2), (0,4)]")
        frame_dwell_options_layout.addRow("plot_range:", self.button_plot_range)


        # variable to be used for level averaging:
        self.dwell_variable_combobox = QComboBox()
        self.dwell_variable_combobox.setToolTip('use averaged value of variable dwells to allocate to states')
        dwell_variable_items = ['FRET', 'intensity', '.....']
        self.dwell_variable_combobox.addItems(dwell_variable_items)
        frame_dwell_options_layout.addRow("variable_to_average:",
                                          self.dwell_variable_combobox)

        #dwell_options_layout=QVBoxLayout()
        #dwell_options_layout.addWidget(dwell_variable_label)
        #dwell_options_layout.addWidget(dwell_variable_combobox)

        #dwell_options = QWidget()
        #dwell_options.setLayout(dwell_options_layout)

        # main control buttons
        dwell_controls=build_control_layouts(
            [make_push_button("Get Dwells", self.perform_dwell_times_sequence,"fit and show dwell times "),
            make_push_button('Export',self.export_kinetics, "export plot contents"),
            make_push_button('Help',self.show_dwell_help, None)])

        #tune panel on the right

        dwell_panel_layout=QVBoxLayout()
        dwell_panel_layout.setAlignment(Qt.AlignRight)
        dwell_panel_layout.addWidget(frame_dwell_options)
        dwell_panel_layout.addWidget(dwell_controls)

        #tab layout
        dwell_times_tab_layout = QHBoxLayout()
        dwell_times_tab_layout.addWidget(self.dwell_kinetics_canvas)
        dwell_times_tab_layout.addLayout(dwell_panel_layout)



        #other
        other_graph_layout = QHBoxLayout()
        #other_graph_layout.addWidget(box2)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)

        tab1 = QWidget(self)
        tab1.setLayout(dwell_times_tab_layout)
        tabs.addTab(tab1, 'Dwell Time')
        tab2 = QWidget(self)
        tab2.setLayout(other_graph_layout)
        #tabs.addTab(tab2, 'Other')


        self.kinetics_widget = QWidget()
        kinetics_layout = QHBoxLayout()
        kinetics_layout.addWidget(tabs)
        #self.kinetics_widget.setLayout(kinetics_layout)
        self.setLayout(kinetics_layout)

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

    def export_kinetics(self, file):
        fpath = self.parent.experiment.analysis_path / 'Dwell time analysis'
        wysiwyg_export(self.fig_dwell_times, filepath=fpath,  filename=self.file.name +"_kinetics_export", filetype="txt")

    def perform_dwell_times_sequence(self):
        self.fig_dwell_times.clear()
        axes = self.fig_dwell_times.subplots(1, 2)#, sharex=False)

        selected_files = self.parent.experiment.selectedFiles

        self.dwell_kinetics_canvas.draw_idle()
        #here it would be good to pass button choices: variable, method
        variable=get_button_value(self.dwell_variable_combobox)
        method=get_button_value(self.dwell_method_combobox)
        N_exponents=get_button_value(self.button_multiple_exponents)
        plot_range=get_button_value(self.button_plot_range)
        if selected_files:
            selected_files.serial.determine_dwells_from_classification(variable=variable, selected=True, inactivate_start_and_end_states=True)
            selected_files.serial.analyze_dwells(method=method, number_of_exponentials=[1, N_exponents])
            selected_files.serial.plot_dwell_analysis(plot_range=plot_range, axes=axes, log=False, save_path=None)

    def show_dwell_help(self):

        help_text = """
                <html>
                  <body style="font-family: sans-serif; font-size: 10pt;">
                
                    <h2>Dwell Times</h2>
                
                    <p>
                      Choose which variable and method should be used for the dwell time analysis.
                    </p>
                
                    <ul>
                      <li>Select the variable for dwell time extraction</li>
                      <li>Select the analysis method</li>
                      <li>'Export' saves as-seen data from the graph panels to .csv and .png
                    </ul>
                
                    <p>
                      For more help, see the
                      <a href="https://papylio.readthedocs.io/en/stable/user_guide/dwell_time_analysis/index.html">
                        dwell time analysis documentation
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


