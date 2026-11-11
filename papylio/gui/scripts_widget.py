from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QPushButton, QPlainTextEdit, QLabel, QComboBox
)
from PySide2.QtCore import Qt
import sys
import io
import textwrap


from papylio.gui.common_layouts import HelpDialog
from papylio.gui.scripts_examples import example_scripts


class ScriptBox(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        self._file = None

        # ------------------------------------------------------------
        # Script editor
        # ------------------------------------------------------------

        self.script_edit = QPlainTextEdit()
        self.script_edit.setPlaceholderText(
            "Paste Python/Papylio code here..."

        )

        # ------------------------------------------------------------
        # Output window
        # ------------------------------------------------------------

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)


        self.combo_category = QComboBox()
        self.combo_category.setToolTip("Choose a script category")

        self.combo_script = QComboBox()
        self.combo_script.setToolTip("Choose an example script")

        self.combo_category.addItems(list(example_scripts.keys()))



        self.combo_category.currentTextChanged.connect(
            self.update_script_list
        )

        self.combo_script.currentTextChanged.connect(
            self.load_example
        )

        self.update_script_list()



        #buttons:

        #self.load_button = QPushButton("Load")
        #self.load_button.clicked.connect(self.load_example)
        #self.load_button.setToolTip('Load pre-set script')


        self.run_button = QPushButton("Run")
        self.run_button.setToolTip('Run current script')
        self.run_button.clicked.connect(self.run_script)

        self.clear_button = QPushButton("Clear output")
        self.clear_button.setToolTip('Clear current output')
        self.clear_button.clicked.connect(self.clear_output)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_script_help)

        button_layout = QHBoxLayout()

        button_layout.addWidget(QLabel("Theme:"))
        button_layout.addWidget(self.combo_category)
        button_layout.addWidget(QLabel("Examples:"))
        button_layout.addWidget(self.combo_script)
        button_layout.addWidget(QLabel("Actions:"))
        #button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.help_button)

        # ------------------------------------------------------------
        # Main layout
        # ------------------------------------------------------------

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Python script:"))
        layout.addWidget(self.script_edit)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output)
        layout.addLayout(button_layout)

        self.setDisabled(True)


    # ================================================================
    # Current file
    # ================================================================

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


    # ================================================================
    # Run script
    # ================================================================

    def run_script(self):

        code = self.script_edit.toPlainText()

        if not code.strip():
            return

        self.output.clear()
        self.output.appendPlainText("Running script...\n")

        # Objects available to the script
        namespace = {
            "self": self.parent,
            "file": self.file,
            "experiment": self.parent.experiment,
        }

        # Capture print() output
        stdout = io.StringIO()
        stderr = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = stdout
        sys.stderr = stderr

        try:
            exec(code, namespace)

        except Exception as e:
            # Put the error into our captured output
            print(f"{type(e).__name__}: {e}")

        finally:
            # VERY IMPORTANT: restore normal Python output
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Show captured output in the GUI
        output_text = stdout.getvalue()
        error_text = stderr.getvalue()

        if output_text:
            self.output.appendPlainText(output_text)

        if error_text:
            self.output.appendPlainText(error_text)

        self.output.appendPlainText("\nScript finished.")

    # ================================================================
    # Clear output
    # ================================================================

    def clear_output(self):
        self.output.clear()

    def load_example(self):

        name = self.example_combo.currentText()

        if name == "My Script":

            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Select Python script",
                "",
                "Python files (*.py);;All files (*)"
            )

            if filename:
                with open(filename, "r", encoding="utf-8") as f:
                    script = f.read()


                self.script_box.setPlainText(textwrap.dedent(script).strip())

            return

        # normal example script
        script = example_scripts[name]
        self.script_edit.setPlainText(script)

    def update_script_list(self):
        category = self.combo_category.currentText()

        self.combo_script.clear()

        self.combo_script.addItems(
            list(example_scripts[category].keys())
        )

    def load_example(self):

        category = self.combo_category.currentText()
        name = self.combo_script.currentText()

        # User's own script
        if category == "My Script":

            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Select Python script",
                "",
                "Python files (*.py);;All files (*)"
            )

            if filename:
                with open(filename, "r", encoding="utf-8") as f:
                    script = f.read()

                self.script_edit.setPlainText(script)

            return

        # Normal example script
        if category in example_scripts and name in example_scripts[category]:
            script = example_scripts[category][name]
            self.script_edit.setPlainText(script)


    # ================================================================
    # Help
    # ================================================================

    def show_script_help(self):

        help_text = """
        <html>
        <body style="font-family: sans-serif; font-size: 10pt;">

        <h2>Custom Scripts</h2>

        <p>
        Execute custom Python/Papylio scripts. 
        Any script can be pasted. 
        Additionally, pre-set scripts can be loaded.
        </p>

        <p>
        The following objects are available:
        </p>

        <ul>
          <li><b>file</b> - currently selected file</li>
          <li><b>experiment</b> - current experiment</li>
          <li><b>self</b> - the MainWindow</li>
        </ul>
        
        <h3>Usage</h3>
        
        <p>
        <ul>
          <li>Hover over the buttons to see their functions </li>
          <li>Choose Theme and Example and Load to load pre-set example scripts </li>
          <li>Before running, edit at wish </li>
        </p>

        <h3>Example</h3>
        <pre>
        #show current mapping
        mapping_file = experiment.files[0]
        figure, axis = mapping_file.show_image()
        mapping_file.mapping.show(axis=axis, show_source=True)
        figure.show()
        </pre>
        
        <h3>Special pre_sets</h3>
        <p>
        <ul>
          <li> Theme 'pre-process' writes any processing to a variable 'processed', 
          to be used for quick checking within the histogram panel </li>
          <li>Choose Theme and Example and Load to load pre-set example scripts </li>
          <li>Before running, edit at wish </li>
        </p>
        

        <h3>Note</h3>
        <p>
         The user is assumed to have a proper insight in use of Papylio scripts.
         Custom code may overwrite earlier GUI pipeline results.
         It may lead to mismatches and/or errors in the stored data or in GUI use
         
        </p>
        
        

        </body>
        </html>
        """

        self.help_dialog = HelpDialog(self, help_text)
        self.help_dialog.show()