"""Locations and curated geometry of the read-only basis repository.

Nothing here is written to. The raw time-series archive and the curated
transducer geometry live in the basis project; by default that is the sibling
directory ``DT_mockup_2026``, overridden by ``FETUNE_BASIS``.
"""
import json
import os
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
BASIS = Path(os.environ.get('FETUNE_BASIS', str(PROJECT.parent/'DT_mockup_2026')))
DATA = BASIS/'data'


def drive_points():
    """Identifier and position of each calibrated force location."""
    d = json.loads((DATA/'curated/drive_points.json').read_text())
    return {p['id']: np.array(p['xyz'], float) for p in d['points']}


def accelerometers():
    """One-based channel numbers and coordinates of the 40 accelerometers."""
    g = json.loads((DATA/'curated/geometry.json').read_text())
    ch = np.array([c['channel'] for c in g['acceleration_channels']])
    xyz = np.array([c['xyz'] for c in g['acceleration_channels']], float)
    return ch, xyz
