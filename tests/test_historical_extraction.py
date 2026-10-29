"""Regressões reais opt-in; referências oficiais, não saídas do diagnóstico."""
import hashlib
import json
import os
from pathlib import Path
import unittest
import numpy as np
from src.audio_features import extract_features, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]

class ProfileContractTests(unittest.TestCase):
    def test_historical_profile_requires_explicit_decoder(self):
        with self.assertRaisesRegex(ValueError, 'ffmpeg_path'):
            extract_features(__file__, profile='fma-2017')

    def test_unknown_profile_rejected(self):
        with self.assertRaisesRegex(ValueError, 'perfil|Perfil'):
            extract_features(__file__, profile='unknown')

@unittest.skipUnless(os.environ.get('FMA_AUDIO_REGRESSION') == '1',
                     'Ative FMA_AUDIO_REGRESSION=1 com os MP3s completos oficiais disponíveis')
class OfficialAudioRegressionTests(unittest.TestCase):
    def test_both_complete_tracks_against_official_csv(self):
        fixtures = json.loads((ROOT/'tests/fixtures/fma_complete_references.json').read_text())
        ffmpeg = os.environ.get('FMA_FFMPEG', str(ROOT/'validation/audio_features/runs/tools/imageio_ffmpeg_0_6_0/ffmpeg-win-x86_64-v7.1.exe'))
        for track in fixtures:
            with self.subTest(track_id=track['track_id']):
                audio = Path(os.environ.get('FMA_AUDIO_'+str(track['track_id']), str(ROOT/track['local_path'])))
                self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), track['sha256'])
                actual = extract_features(audio, profile='fma-2017', ffmpeg_path=ffmpeg)
                self.assertEqual(tuple(actual.index), FEATURE_NAMES)
                expected = np.array([track['official'][name] for name in FEATURE_NAMES],dtype=np.float64)
                errors = np.abs(actual.to_numpy(dtype=np.float64)-expected)
                failed = errors > 1e-6+1e-4*np.abs(expected)
                self.assertFalse(failed.any(), str(np.array(FEATURE_NAMES)[failed].tolist()))
