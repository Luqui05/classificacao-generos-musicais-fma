from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.training.train_final_models import train, check_environment, assert_same_state
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class TrainingSafetyTests(unittest.TestCase):
    def test_tree_roundtrip_and_changed_state_detection(self):
        model = DecisionTreeClassifier(random_state=42).fit([[0],[1],[2],[3]], [0,0,1,1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tree.joblib'
            joblib.dump(model,path)
            restored = joblib.load(path)
            assert_same_state(model,restored)
            restored.tree_.threshold[0] += 1
            with self.assertRaises(AssertionError):
                assert_same_state(model,restored)

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/'model.joblib'
            marker.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                train('absent.csv', directory)
            self.assertEqual(marker.read_bytes(), b'existing')

    def test_environment_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory)/'lock.txt'
            lock.write_text('numpy==1.0.0\n')
            with patch('scripts.training.train_final_models.version', return_value='2.0.0'):
                with self.assertRaisesRegex(RuntimeError, 'numpy'):
                    check_environment(lock)
