"""Execute the end-to-end notebook with this interpreter and save outputs as it runs."""
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'notebooks/initial_design.ipynb'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/notebook/.matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(ROOT / 'artifacts/notebook/.ipython'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')


def main():
    Path(os.environ['IPYTHONDIR']).mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(PATH, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix='claims-kernel-') as runtime:
        manager = AsyncKernelManager(connection_file=str(Path(runtime) / 'kernel.json'))
        manager._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                          display_name='Claims triage', language='python')
        def progress(cell, cell_index, **kwargs):
            print(f'Executing cell {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:100]}', flush=True)
        client = NotebookClient(notebook, km=manager, timeout=None,
                                resources={'metadata': {'path': str(ROOT)}},
                                on_cell_execute=progress)
        try:
            client.execute(cleanup_kc=True)
        finally:
            nbformat.write(notebook, PATH)
    print(f'Saved executed notebook: {PATH}', flush=True)


if __name__ == '__main__':
    main()
