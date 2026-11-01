import tempfile
from pathlib import Path
import unittest
import json
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from src.audio_features import FEATURE_NAMES
from src.inference import infer_mp3, predict_features, load_models
from src.models import build_models

class InferenceTests(unittest.TestCase):
    def test_corrupted_artifact_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'bad.joblib').write_bytes(b'corrupted')
            manifest=dict(status='complete',features=list(FEATURE_NAMES),versions={},
                          models={name:dict(file='bad.joblib',sha256='0'*64) for name in build_models()})
            (root/'manifest.json').write_text(json.dumps(manifest))
            with patch('src.inference.joblib.load') as load:
                with self.assertRaisesRegex(ValueError,'Artefato'):
                    load_models(root)
                load.assert_not_called()

    def test_incomplete_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'manifest.json').write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError,'incompleto'):
                load_models(directory)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                infer_mp3('absent.mp3','absent','absent',directory)

    def test_invalid_vector_never_reaches_model(self):
        model=Mock()
        with self.assertRaises(ValueError):
            predict_features(pd.Series(np.full(518,np.nan),index=FEATURE_NAMES),{'test':model})
        model.predict.assert_not_called()

    def test_column_order_and_single_row_are_preserved(self):
        model=Mock(); model.predict.return_value=['Rock']
        self.assertEqual(predict_features(pd.Series(np.arange(518),index=FEATURE_NAMES),{'test':model}),{'test':'Rock'})
        frame=model.predict.call_args.args[0]
        self.assertEqual(frame.shape,(1,518))
        self.assertEqual(list(frame.columns),list(FEATURE_NAMES))
