"""Controles independentes para os diagnósticos numéricos de áudio."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import librosa
import numpy as np

from audio_legacy_diagnostics import LegacyNumpy, load_functions
from diagnose_full_audio import PROFILES, extract_profile, historical_zcr


class HistoricalZcrTests(unittest.TestCase):
    def test_known_crossings_and_edge_padding(self):
        result = historical_zcr(np.array([1., -1., 0., -1.]), frame_length=4, hop_length=2)
        np.testing.assert_array_equal(result, [[0.25, 0.75, 0.25]])

    def test_threshold_treats_tiny_negative_value_as_zero(self):
        result = historical_zcr(np.array([-1., -1e-12, 0., 1.]), frame_length=4, hop_length=2)
        np.testing.assert_array_equal(result, [[0.25, 0.25, 0.]])

    def test_silence_has_no_crossings(self):
        np.testing.assert_array_equal(historical_zcr(np.zeros(4096)), np.zeros((1, 9)))

    def test_does_not_modify_input(self):
        y = np.array([-1., -1e-12, 0., 1.])
        original = y.copy()
        historical_zcr(y, frame_length=4, hop_length=2)
        np.testing.assert_array_equal(y, original)

    def test_invalid_audio_is_rejected(self):
        for signal in ([], [[1.]], [np.inf], [np.nan]):
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                historical_zcr(signal)


class DiagnosticIsolationTests(unittest.TestCase):
    def test_library_functions_are_restored_after_failure(self):
        originals = (librosa.load, librosa.stft, librosa.cqt, librosa.feature.rms,
                     librosa.feature.chroma_cqt, librosa.feature.chroma_cens, librosa.filters.chroma)
        with patch('diagnose_full_audio.extract_features', side_effect=RuntimeError('probe')):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                extract_profile(Path('unused.mp3'), np.ones(4096), 44100, PROFILES['chroma_float64'])
        restored = (librosa.load, librosa.stft, librosa.cqt, librosa.feature.rms,
                    librosa.feature.chroma_cqt, librosa.feature.chroma_cens, librosa.filters.chroma)
        for before, after in zip(originals, restored):
            self.assertIs(before, after)

    def test_changed_historical_source_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'constantq.py'
            source.write_text('raise RuntimeError("must not execute")', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Fonte histórica diferente'):
                load_functions(source, ['cqt'], {})

    def test_scalar_scaling_preserves_float32_like_numpy_1(self):
        y = np.array([1., -2., 0.25], dtype=np.float32)
        result = LegacyNumpy().sqrt(2) * y
        self.assertEqual(result.dtype, np.dtype('float32'))
        np.testing.assert_allclose(result, [np.sqrt(2), -2 * np.sqrt(2), np.sqrt(2) / 4], rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
