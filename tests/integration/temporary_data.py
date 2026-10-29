from pathlib import Path
import shutil
import os
import tempfile
from contextlib import contextmanager

@contextmanager
def temporary_data(relative_path, root_path=None):
    if root_path is None:
        root_path = Path(__file__).parent / 'data'
    relative_path = Path(relative_path)
    source = root_path / relative_path
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='papylio_test_') as temporary_directory:
        destination = Path(temporary_directory) / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(temporary_directory)
        shutil.copy(source, destination)

        try:
            yield destination
        finally:
            os.chdir(cwd) #