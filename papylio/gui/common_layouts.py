#this widget contains layouts that are used in multiple tabs.
#They are stored here to avoid circular imports -jk

import sys

import matplotlib as mpl
import inspect
import ast


from matplotlib.backends.backend_qtagg import (
    FigureCanvas)

from PySide2.QtWidgets import (QFormLayout, QDoubleSpinBox,QSpinBox, QComboBox,
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,QToolButton, QTextBrowser,
    QLabel, QSizePolicy, QGroupBox, QGridLayout, QDialog, QPushButton, QTextEdit, QDialog, QLineEdit
)


from PySide2.QtCore import Qt
import sys



class Expander(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")

        self.content = QWidget()
        self.content.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

        self.toggle_button.toggled.connect(self.on_toggled)

    def on_toggled(self, checked):
        self.content.setVisible(checked)
        self.toggle_button.setArrowType(
            Qt.DownArrow if checked else Qt.RightArrow
        )

    def setContentLayout(self, content_layout):
        self.content.setLayout(content_layout)

class HelpDialog(QDialog):
    def __init__(self, parent=None, help_text=""):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(900, 800)

        layout = QVBoxLayout(self)

        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setHtml(help_text)

        layout.addWidget(self.text)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)

        layout.addWidget(self.text)
        layout.addWidget(close_button)

class Group_Box(QGroupBox):
    def __init__(self, parent=None, title="title", highlight=False):
        super().__init__(parent)
        self.setTitle(title)

        base_style = """
        QGroupBox {
            border: 1px solid gray;
            border-radius: 3px;
            margin-top: 10px;  /* space for title */
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        """

        highlight_style = """
        QGroupBox::title {
            color: blue;
            font-weight: bold;
        }
        """

        if highlight:
            self.setStyleSheet(base_style + highlight_style)
        else:
            self.setStyleSheet(base_style)

def make_push_button(text, method, tooltip):
    #build a standardized push button
    btn = QPushButton(text)
    #btn.setFixedHeight(40)  # small height
    #btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    #btn.setStyleSheet("QPushButton { padding: 2px 6px; font-size: 16px; }")
    btn.clicked.connect(method)
    if tooltip is not None:
        btn.setToolTip(tooltip)
    return btn

def build_control_layouts(button_list):
    controls_layout = QHBoxLayout()

    # Push everything to the right
    controls_layout.addStretch()

    for b in button_list:
        b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(b)

    controls = QWidget()
    controls.setLayout(controls_layout)
    return controls

def deep_get_config(d, keys, default=None):
    #this function is used for linking a saved nested configurations to gui front
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

#below a series of functions to link GUI elements to Papylio methods in generic fashion
def get_input_type(annotation, default):
    # Pick appropriate input type
    if annotation == int or isinstance(default, int):
        widget = QSpinBox()
        widget.setRange(-1_000_000, 1_000_000)
        if default is not None:
            widget.setValue(default)
    elif annotation == float or isinstance(default, float):
        widget = QDoubleSpinBox()
        widget.setRange(-1e9, 1e9)
        widget.setDecimals(6)
        if default is not None:
            widget.setValue(default)
    else:
        widget = QLineEdit()
        if default not in (None, inspect.Parameter.empty):
            widget.setText(str(default))
    return widget

def build_form(func):
    # --- build the form for a function ---
    form_widget = QWidget()
    form = QFormLayout(form_widget)
    inputs = {}
    sig = inspect.signature(func)
    for param_name, param in sig.parameters.items():
        if param_name in ['image']:
            continue
        default = param.default if param.default is not inspect.Parameter.empty else None
        annotation = param.annotation
        widget = get_input_type(annotation=annotation, default=default)
        form.addRow(f"{param_name}:", widget)
        inputs[param_name] = widget

    return form_widget, inputs


def build_parameters_input(method_name, inputs):
    #build input for chosen method
    kwargs = {'method': method_name}
    for pname, widget in inputs.items():
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            val = widget.value()
        else:
            val = widget.text()
            try:
                val = float(val) if "." in val else int(val)
            except ValueError:
                pass
        kwargs[pname] = val

    return kwargs

def get_button_value(widget):
    #return proper conversion depending on type of entry field
    if isinstance(widget, QComboBox):
        button_val = widget.currentText()

    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        button_val = widget.value()

    elif isinstance(widget, QLineEdit):
        text = widget.text().strip()

        # Try list / literal parsing first
        try:
            if text.startswith("[") or text.startswith("(") or text.startswith("{"):
                button_val = ast.literal_eval(text)
            else:
                # Try numeric conversion
                try:
                    button_val = int(text)
                except ValueError:
                    try:
                        button_val = float(text)
                    except ValueError:
                        button_val = text
        except (ValueError, SyntaxError):
            button_val = text

    return button_val