import sys

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


class SliderPlotWidget(QWidget):
    """A QSlider that controls a line plot embedded in a matplotlib canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Matplotlib figure/canvas ---
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.figure.add_subplot(111)

        self.x = np.linspace(0, 2 * np.pi, 500)
        (self.line,) = self.ax.plot(self.x, np.sin(self.x))
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("sin(f · x)")
        # self.figure.tight_layout()

        # --- Slider ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 20)  # frequency
        self.slider.setValue(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        self.label = QLabel()
        self.label.setMinimumWidth(80)
    #
        # --- Layout ---
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frequency:"))
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(self.label)
        #
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(slider_row)

        # --- Signals ---
        self.slider.valueChanged.connect(self.update_plot)
        self.update_plot(self.slider.value())

    def update_plot(self, value: int):
        self.line.set_ydata(np.sin(value * self.x))
        self.label.setText(f"f = {value}")
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = SliderPlotWidget()
    widget.setWindowTitle("PySide6 + Matplotlib")
    widget.resize(700, 500)
    widget.show()
    sys.exit(app.exec())