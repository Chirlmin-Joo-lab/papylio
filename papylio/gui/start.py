"""Entry point for launching the Papylio GUI.

Provides a convenience function to start the GUI application with default options.
"""

from PySide2.QtWidgets import QApplication
import sys
import traceback

from multiprocessing import Process, freeze_support

# Necessary when using pythonw, since it has no console, otherwise there is no way to output the text.
# import sys
# sys.stdout = open("C:/temp/stdout.log", "w")
# sys.stderr = open("C:/temp/stderr.log", "w")

def start_gui():
    """Starts the Papylio GUI application."""
    freeze_support()
    app = QApplication(sys.argv)
    from papylio.gui.main import MainWindow
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    try:
        start_gui()
    except Exception:
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")
