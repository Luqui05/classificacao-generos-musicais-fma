"""Controles analíticos para distinguir as normalizações espectrais."""
import unittest
import numpy as np
import librosa
from diagnose_audio_rms import compare_waveforms, historical_rms


class RmsNormalizationTests(unittest.TestCase):
    def test_flat_spectrum(self):
        spectrum = np.ones((1025, 3), dtype=np.float32)
        np.testing.assert_allclose(historical_rms(spectrum), 1)
        np.testing.assert_allclose(librosa.feature.rms(S=spectrum), 1 / np.sqrt(2048))

    def test_dc_only_has_different_scaling(self):
        spectrum = np.zeros((1025, 1), dtype=np.float32)
        spectrum[0] = 1
        np.testing.assert_allclose(historical_rms(spectrum), 1 / np.sqrt(1025))
        np.testing.assert_allclose(librosa.feature.rms(S=spectrum), 1 / 2048)


class WaveformComparisonTests(unittest.TestCase):
    def setUp(self):
        self.signal = np.random.default_rng(42).normal(size=4096)

    def test_identical_samples(self):
        result = compare_waveforms(self.signal, self.signal)
        self.assertTrue(result['exactly_equal'])
        self.assertEqual(result['lag_samples'], 0)
        self.assertEqual(result['aligned_rmse'], 0)

    def test_delayed_candidate_is_aligned(self):
        candidate = np.pad(self.signal, (37, 0))
        result = compare_waveforms(self.signal, candidate)
        self.assertEqual(result['sample_count_difference'], 37)
        self.assertEqual(result['lag_samples'], -37)
        self.assertEqual(result['aligned_rmse'], 0)
        self.assertGreater(result['raw_rmse'], 0)

    def test_delayed_reference_is_aligned(self):
        reference = np.pad(self.signal, (29, 0))
        result = compare_waveforms(reference, self.signal)
        self.assertEqual(result['lag_samples'], 29)
        self.assertEqual(result['aligned_rmse'], 0)

    def test_amplitude_change_is_not_removed(self):
        result = compare_waveforms(self.signal, 0.5 * self.signal)
        self.assertEqual(result['lag_samples'], 0)
        self.assertAlmostEqual(result['aligned_rmse'], 0.5 * np.sqrt(np.mean(self.signal ** 2)))

    def test_empty_or_nonfinite_audio_is_rejected(self):
        for candidate in ([], [np.nan], [np.inf]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare_waveforms(self.signal, candidate)


if __name__ == '__main__':
    unittest.main()
