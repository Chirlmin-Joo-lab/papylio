from PySide2.QtWidgets import QApplication
import sys

from multiprocessing import Process, freeze_support

# Necessary when using pythonw, since it has no console, otherwise there is no way to output the text.
# import sys
# sys.stdout = open("C:/temp/stdout.log", "w")
# sys.stderr = open("C:/temp/stderr.log", "w")

def start_gui():
    freeze_support()
    app = QApplication(sys.argv)
    from papylio.gui.main import MainWindow
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    start_gui()