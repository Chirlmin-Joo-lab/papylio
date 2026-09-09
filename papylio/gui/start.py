"""Entry point for launching the Papylio GUI.

Provides a convenience function to start the GUI application with default options.
"""
import os
import subprocess

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
    app = QApplication(sys.argv)
    from papylio.gui.main import MainWindow
    window = MainWindow()
    window.show()
    app.exec_()

# def start_jupyter(directory):
#     install_kernelspec()
#
#     from jupyter_client.kernelspec import KernelSpecManager
#     ksm = KernelSpecManager()
#     spec = ksm.get_kernel_spec("papylio")
#
#     print("KERNEL SPEC:")
#     print(spec.argv)
#
#     from jupyterlab.labapp import LabApp
#     LabApp.launch_instance(argv=["--notebook-dir", str(directory), "--MappingKernelManager.allowed_kernelspecs=Papylio", "--MappingKernelManager.default_kernel_name=Papylio"])
#
# def install_kernelspec():
#     import sys, os, json, tempfile
#     from jupyter_client.kernelspec import KernelSpecManager
#
#     if getattr(sys, "frozen", False):
#         argv = [sys.executable, "--ipykernel", "-f", "{connection_file}"]
#     else:
#         argv = [sys.executable, sys.argv[0], "--ipykernel", "-f", "{connection_file}"]
#
#     spec = {"argv": argv, "display_name": "Papylio", "language": "python"}
#
#     print("Installing kernelspec:")
#     print(json.dumps(spec, indent=2))
#
#     ksm = KernelSpecManager()
#
#     with tempfile.TemporaryDirectory() as staging_dir:
#         with open(os.path.join(staging_dir, "kernel.json"), "w") as f:
#             json.dump(spec, f)
#         ksm.install_kernel_spec(staging_dir, "papylio", user=True, replace=True)

def start_marimo(directory):
    from marimo._cli.cli import edit

    edit.main(
        args=[directory],
        standalone_mode=False,
    )

if __name__ == '__main__':
    freeze_support()
    try:
        if "--marimo" in sys.argv:
            index = sys.argv.index("--marimo")

            if index + 1 >= len(sys.argv):
                raise ValueError("--marimo requires a notebook directory")

            notebooks_dir = sys.argv[index + 1]
            start_marimo(notebooks_dir)
        # elif "--jupyter" in sys.argv:
        #     index = sys.argv.index("--jupyter")
        #
        #     if index + 1 >= len(sys.argv):
        #         raise ValueError("--jupyter requires a notebook directory")
        #
        #     notebooks_dir = sys.argv[index + 1]
        #     start_jupyter(notebooks_dir)
        # elif "--ipykernel" in sys.argv:
        #     import os
        #
        #     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        #
        #     print("1: entering kernel branch", flush=True)
        #
        #     # sys.argv = sys.argv[1:]
        #     sys.argv = [
        #         sys.argv[0],
        #         *sys.argv[sys.argv.index("--ipykernel") + 1:]
        #     ]
        #
        #     print("2: importing zmq", flush=True)
        #     import zmq
        #
        #     print(f"3: zmq imported: {zmq.__version__}", flush=True)
        #
        #     print("4: creating ZMQ context", flush=True)
        #     ctx = zmq.Context()
        #     print("5: ZMQ context created", flush=True)
        #
        #     print("6: creating test socket", flush=True)
        #     socket = ctx.socket(zmq.PAIR)
        #     print("7: test socket created", flush=True)
        #     socket.close()
        #     ctx.term()
        #
        #     print("8: importing ipykernel", flush=True)
        #     from ipykernel import kernelapp as app
        #
        #     print("9: ipykernel imported", flush=True)
        #
        #     print("10: launching kernel", flush=True)
        #     app.launch_new_instance()
        #
        #     print("11: kernel exited", flush=True)
        #     sys.exit(0)
        else:
            start_gui()
    except Exception:
        traceback.print_exc()
        if len(sys.argv) <= 1:
            input("\nPress Enter to exit...")


