"""Deterministic JSON writing and file hashing for the evidence trail."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
import tempfile

import numpy as np


@contextmanager
def atomic_output(path):
    """Publish a completed artifact without exposing a partially written file."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=f'.{path.stem}.',suffix=path.suffix,dir=path.parent)
    os.close(fd)
    temporary=Path(name)
    try:
        yield temporary
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def savefig_atomic(figure,path,**kwargs):
    """Keep the previous complete figure readable during a Matplotlib export."""
    with atomic_output(path) as temporary:
        figure.savefig(temporary,**kwargs)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=convert,
                                     allow_nan=False)+'\n')
