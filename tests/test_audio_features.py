"""Testes de contrato; os mocks não validam DSP nem decodificação de MP3."""

import csv
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd

from src import audio_features as audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def librosa_module(module):
    """Substitui apenas Librosa, preservando módulos nativos importados no teste."""
    missing = object()
    original = sys.modules.get("librosa", missing)
    sys.modules["librosa"] = module
    try:
        yield
    finally:
        if original is missing:
            sys.modules.pop("librosa", None)
        else:
            sys.modules["librosa"] = original


def valid_vector():
    return pd.Series(np.ones(518), index=audio.FEATURE_NAMES, name="faixa")


class SchemaAndStatisticsTests(unittest.TestCase):
    def test_schema_matches_training_csv_including_order(self):
        with (PROJECT_ROOT / "data/fma_small_features.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            columns = next(csv.reader(stream))
        expected = tuple(c for c in columns if c not in {"track_id", "genre"})
        self.assertEqual(len(expected), 518)
        self.assertEqual(len(set(audio.FEATURE_NAMES)), 518)
        self.assertEqual(audio.FEATURE_NAMES, expected)

    def test_statistics_against_hand_calculated_values(self):
        values = np.array([[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.]])
        expected = {
            "mean": [3., 6.], "std": [np.sqrt(2), np.sqrt(8)],
            "skew": [0., 0.], "kurtosis": [-1.3, -1.3],
            "median": [3., 6.], "min": [1., 2.], "max": [5., 10.],
        }
        actual = audio._feature_statistics(values)
        self.assertEqual(set(actual), set(expected))
        for statistic, result in expected.items():
            with self.subTest(statistic=statistic):
                np.testing.assert_allclose(actual[statistic], result, atol=1e-12)

    def test_module_can_import_without_librosa(self):
        spec = importlib.util.spec_from_file_location("audio_without_librosa", audio.__file__)
        module = importlib.util.module_from_spec(spec)
        with librosa_module(None):
            spec.loader.exec_module(module)
        self.assertEqual(module.FEATURE_NAMES, audio.FEATURE_NAMES)


class ValidationTests(unittest.TestCase):
    def test_valid_vector_is_float32_and_input_is_unchanged(self):
        original = valid_vector()
        before = original.copy(deep=True)
        result = audio.validate_features(original)
        self.assertEqual(result.dtype, np.dtype("float32"))
        self.assertEqual(result.name, "faixa")
        pd.testing.assert_series_equal(original, before)
        np.testing.assert_array_equal(result.to_numpy(), before.to_numpy())

    def test_rejects_other_container_types(self):
        for value in [np.ones(518), [1.] * 518, valid_vector().to_frame()]:
            with self.subTest(container=type(value)), self.assertRaises(TypeError):
                audio.validate_features(value)

    def test_rejects_wrong_length(self):
        for value in [valid_vector().iloc[:-1], pd.concat([valid_vector(), pd.Series([1.])])]:
            with self.subTest(length=len(value)), self.assertRaisesRegex(ValueError, "518"):
                audio.validate_features(value)

    def test_rejects_missing_extra_duplicate_or_reordered_labels(self):
        for mode in ["unknown", "duplicate", "reordered"]:
            value = valid_vector()
            labels = list(value.index)
            if mode == "unknown":
                labels[0] = "atributo_desconhecido"
            elif mode == "duplicate":
                labels[0] = labels[1]
            else:
                labels[0], labels[1] = labels[1], labels[0]
            value.index = labels
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "nomes"):
                audio.validate_features(value)

    def test_rejects_non_numeric_values(self):
        value = valid_vector().astype(object)
        value.iloc[0] = "inválido"
        with self.assertRaisesRegex(TypeError, "numéricas"):
            audio.validate_features(value)

    def test_rejects_non_finite_values(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            value = valid_vector()
            value.iloc[0] = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(
                ValueError, audio.FEATURE_NAMES[0]
            ):
                audio.validate_features(value)

    def test_rejects_float32_overflow_without_runtime_warning(self):
        for invalid in [1e100, -1e100]:
            value = valid_vector()
            value.iloc[0] = invalid
            with self.subTest(value=invalid), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "float32"):
                    audio.validate_features(value)


def fake_librosa():
    """Saídas controladas com cinco frames e bandas de estatísticas conhecidas."""
    module = ModuleType("librosa")
    module.load = Mock(return_value=(np.arange(2048, dtype=np.float32), 44100))
    module.cqt = Mock(return_value=np.full((84, 5), 1 + 2j))
    module.stft = Mock(return_value=np.full((1025, 5), 3 + 4j))
    module.power_to_db = Mock(return_value=np.full((128, 5), -10.))
    dimensions = {
        "zero_crossing_rate": 1, "chroma_cqt": 12, "chroma_cens": 12,
        "tonnetz": 6, "chroma_stft": 12, "rms": 1,
        "spectral_centroid": 1, "spectral_bandwidth": 1,
        "spectral_contrast": 7, "spectral_rolloff": 1, "mfcc": 20,
    }
    operations = {}
    for scale, (name, bins) in enumerate(dimensions.items(), start=1):
        frames = scale * np.arange(1, bins + 1)[:, None] * np.arange(1., 6.)
        operations[name] = Mock(return_value=frames)
    operations["melspectrogram"] = Mock(return_value=np.full((128, 5), 2.))
    module.feature = SimpleNamespace(**operations)
    return module


class ExtractionContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "faixa.mp3"
        # A existência do arquivo é real; nenhum decodificador é executado.
        self.path.touch()
        self.librosa = fake_librosa()
        self.enterContext(librosa_module(self.librosa))

    def test_all_518_statistics_are_mapped_to_the_correct_labels(self):
        result = audio.extract_features(str(self.path))
        self.assertEqual(tuple(result.index), audio.FEATURE_NAMES)
        self.assertEqual(result.dtype, np.dtype("float32"))
        self.assertEqual(result.name, "faixa")
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        # Para [a, 2a, 3a, 4a, 5a], os sete momentos têm valores analíticos.
        operations = [
            ("zcr", 1), ("chroma_cqt", 12), ("chroma_cens", 12),
            ("tonnetz", 6), ("chroma_stft", 12), ("rmse", 1),
            ("spectral_centroid", 1), ("spectral_bandwidth", 1),
            ("spectral_contrast", 7), ("spectral_rolloff", 1), ("mfcc", 20),
        ]
        for scale, (family, bins) in enumerate(operations, start=1):
            for band in range(1, bins + 1):
                a = scale * band
                expected = {"mean": 3*a, "std": np.sqrt(2)*a, "skew": 0.,
                            "kurtosis": -1.3, "median": 3*a, "min": a, "max": 5*a}
                for statistic, value in expected.items():
                    label = f"{family}_{statistic}_{band:02d}"
                    with self.subTest(label=label):
                        np.testing.assert_allclose(result[label], value, rtol=1e-6, atol=1e-6)

    def test_load_preserves_native_sample_rate_and_uses_mono(self):
        audio.extract_features(self.path)
        self.librosa.load.assert_called_once_with(self.path, sr=None, mono=True)
        cqt_args = self.librosa.cqt.call_args
        self.assertEqual(cqt_args.kwargs, {
            "sr": 44100, "hop_length": 512, "bins_per_octave": 12,
            "n_bins": 84, "tuning": None,
        })
        self.assertEqual(self.librosa.stft.call_args.kwargs, {"n_fft": 2048, "hop_length": 512})

    def test_descriptor_routing_preserves_magnitude_power_and_cens(self):
        audio.extract_features(self.path)
        feature = self.librosa.feature
        for operation in [feature.chroma_cqt, feature.chroma_cens]:
            np.testing.assert_allclose(operation.call_args.kwargs["C"], np.sqrt(5))
        self.assertIs(feature.tonnetz.call_args.kwargs["chroma"], feature.chroma_cens.return_value)
        np.testing.assert_allclose(feature.chroma_stft.call_args.kwargs["S"], 25.)
        for name in ["rms", "spectral_centroid", "spectral_bandwidth",
                     "spectral_contrast", "spectral_rolloff"]:
            np.testing.assert_allclose(getattr(feature, name).call_args.kwargs["S"], 5.)
        self.assertEqual(feature.spectral_contrast.call_args.kwargs["n_bands"], 6)
        mel_args = feature.melspectrogram.call_args.kwargs
        self.assertEqual(mel_args["sr"], 44100)
        np.testing.assert_allclose(mel_args["S"], 25.)
        self.assertIs(self.librosa.power_to_db.call_args.args[0], feature.melspectrogram.return_value)
        self.assertIs(feature.mfcc.call_args.kwargs["S"], self.librosa.power_to_db.return_value)
        self.assertEqual(feature.mfcc.call_args.kwargs["n_mfcc"], 20)

    def test_legacy_rmse_entry_point_is_supported(self):
        self.librosa.feature.rmse = self.librosa.feature.rms
        del self.librosa.feature.rms
        result = audio.extract_features(self.path)
        self.assertEqual(len(result), 518)
        self.librosa.feature.rmse.assert_called_once()

    def test_missing_file_or_directory_is_rejected_before_decoding(self):
        for path in [self.path.with_name("ausente.mp3"), self.path.parent]:
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                audio.extract_features(path)
        self.librosa.load.assert_not_called()

    def test_missing_librosa_has_actionable_error(self):
        with librosa_module(None):
            with self.assertRaisesRegex(RuntimeError, "audio_features.md") as raised:
                audio.extract_features(self.path)
        self.assertIsInstance(raised.exception.__cause__, ImportError)

    def test_empty_audio_is_rejected(self):
        self.librosa.load.return_value = (np.array([], dtype=np.float32), 44100)
        with self.assertRaisesRegex(ValueError, "não contém amostras"):
            audio.extract_features(self.path)
        self.librosa.cqt.assert_not_called()

    def test_decoder_failure_preserves_original_cause(self):
        error = OSError("decodificador indisponível")
        self.librosa.load.side_effect = error
        with self.assertRaisesRegex(ValueError, "faixa.mp3") as raised:
            audio.extract_features(self.path)
        self.assertIs(raised.exception.__cause__, error)

    def test_invalid_descriptor_shapes_are_rejected(self):
        for values in [np.ones(5), np.ones((1, 0)), np.ones((2, 5))]:
            self.librosa.feature.zero_crossing_rate.return_value = values
            with self.subTest(shape=values.shape), self.assertRaisesRegex(ValueError, "A saída de zcr"):
                audio.extract_features(self.path)

    def test_invalid_transform_shapes_are_rejected(self):
        for operation, shape, message in [
            ("cqt", (83, 5), "CQT inesperado"),
            ("cqt", (84, 3), "quadros CQT"),
            ("cqt", (84, 6), "quadros CQT"),
            ("stft", (1024, 5), "STFT inesperada"),
            ("stft", (1025, 3), "quadros STFT"),
            ("stft", (1025, 6), "quadros STFT"),
        ]:
            with self.subTest(operation=operation, shape=shape):
                with patch.object(self.librosa, operation, Mock(return_value=np.ones(shape))):
                    with self.assertRaisesRegex(ValueError, message):
                        audio.extract_features(self.path)

    def test_non_finite_statistics_are_rejected_instead_of_returning_partial_vector(self):
        self.librosa.feature.zero_crossing_rate.return_value = np.zeros((1, 5))
        with self.assertRaisesRegex(ValueError, "não finitas"):
            audio.extract_features(self.path)


if __name__ == "__main__":
    unittest.main()
