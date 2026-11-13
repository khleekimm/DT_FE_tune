import importlib.util
from pathlib import Path
import unittest

import numpy as np
from fetune.holdout_assessment import response_levels, rank_trial


class HoldoutAssessmentTests(unittest.TestCase):
    def test_power_to_amplitude_and_symmetric_level_ranking(self):
        f = np.array([360., 500., 800., 1000.])
        measured = np.ones(4)
        high = response_levels(f, measured, 4 * measured)
        low = response_levels(f, measured, measured / 4)
        self.assertAlmostEqual(high['rms_ratio'], 2.)
        self.assertAlmostEqual(high['measured_rms'], np.sqrt(640))
        self.assertAlmostEqual(high['rms_level_db'], 20 * np.log10(2.))
        high['complex_error'] = low['complex_error'] = 1.
        np.testing.assert_allclose(rank_trial([high]), rank_trial([low]), atol=1e-14)

    def test_no_display_floor_or_invalid_grid(self):
        for f, measured in [(np.arange(4), np.array([1., 0., 1., 1.])),
                            (np.arange(4)[::-1], np.ones(4))]:
            with self.assertRaises(ValueError):
                response_levels(f, measured, np.ones(4))

    def test_partition_rejects_initialization_leakage(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/search_spatial_holdouts.py'
        spec = importlib.util.spec_from_file_location('search_spatial_holdouts', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        channels = np.array([8, 33, 34, 35, 36, 37, 38, 39, 40])
        seed = [33, 34, 36, 37, 39]
        fit, held = module.partition(channels, [8, 35, 38, 40], seed)
        self.assertEqual(channels[fit].tolist(), seed)
        self.assertEqual(channels[held].tolist(), [8, 35, 38, 40])
        with self.assertRaises(ValueError):
            module.partition(channels, [33, 40], seed)
        with self.assertRaises(ValueError):
            module.partition(channels, [13, 40], seed)
        with self.assertRaises(ValueError):
            module.partition(channels, [40, 40], seed)
