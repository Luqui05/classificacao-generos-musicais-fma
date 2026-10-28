"""Controles analíticos para distinguir as normalizações espectrais."""
import unittest
import numpy as np
import librosa
from diagnose_audio_rms import historical_rms


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


if __name__ == '__main__':
    unittest.main()
