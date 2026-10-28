"""Testa a comparação com valores controlados; não estabelece equivalência FMA."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import audio_validation as validation
from src.audio_features import FEATURE_NAMES


def write_reference(path, rows):
    parts = [name.rsplit("_", 2) for name in FEATURE_NAMES]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for level, title in enumerate(["feature", "statistics", "number"]):
            writer.writerow([title, *[p[level] for p in parts]])
        writer.writerow(["track_id", *[""] * 518])
        for track_id, values in rows:
            writer.writerow([track_id, *values])


class VectorComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.Series(np.ones(518), index=FEATURE_NAMES)
        self.extracted = self.reference.astype(np.float32)

    def test_identical_vectors_have_zero_error(self):
        result = validation.compare_vectors(self.reference, self.extracted, rtol=0, atol=0)
        self.assertTrue(result.within_tolerance.all())
        self.assertEqual(result.absolute_error.sum(), 0)
        self.assertEqual(result.iloc[0]["family"], "chroma_cens")
        self.assertEqual(result.iloc[0]["statistic"], "kurtosis")
        self.assertEqual(result.iloc[0]["number"], "01")

    def test_absolute_relative_and_zero_reference_errors(self):
        self.reference.iloc[:3] = [0., -2., 1.]
        self.extracted.iloc[:3] = [.25, -2.5, 1.25]
        result = validation.compare_vectors(self.reference, self.extracted, rtol=.125, atol=.125)
        np.testing.assert_array_equal(result.absolute_error.iloc[:3], [.25, .5, .25])
        self.assertTrue(np.isnan(result.relative_error.iloc[0]))
        np.testing.assert_array_equal(result.relative_error.iloc[1:3], [.25, .25])
        self.assertEqual(result.within_tolerance.iloc[:3].tolist(), [False, False, True])
        summary = validation.summarize_differences(result, ["family", "statistic"])
        self.assertEqual(summary.comparisons.sum(), 518)
        self.assertEqual(summary.within_tolerance.sum(), 516)
        self.assertEqual(summary.zero_references.sum(), 1)

    def test_zero_reference_exact_match_has_undefined_relative_error_and_passes(self):
        self.reference.iloc[0] = self.extracted.iloc[0] = 0.
        result = validation.compare_vectors(self.reference, self.extracted, rtol=0, atol=0)
        self.assertTrue(result.within_tolerance.iloc[0])
        self.assertTrue(np.isnan(result.relative_error.iloc[0]))

    def test_invalid_tolerances_and_vectors_are_rejected(self):
        for tolerance in [-1., np.nan, np.inf]:
            for name in ["rtol", "atol"]:
                with self.subTest(name=name, tolerance=tolerance), self.assertRaises(ValueError):
                    validation.compare_vectors(self.reference, self.extracted, **{name: tolerance})
        with self.assertRaises(ValueError):
            validation.compare_vectors(self.reference.iloc[::-1], self.extracted)
        for vector in [self.reference, self.extracted]:
            vector.iloc[0] = np.nan
            with self.assertRaises(ValueError):
                validation.compare_vectors(self.reference, self.extracted)
            vector.iloc[0] = 1.


class ReferenceAndRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "features.csv"
        write_reference(self.reference, [(2, np.ones(518)), (10, np.full(518, 2.))])
        self.sample = self.root / "sample"
        self.sample.mkdir()
        manifest = pd.DataFrame({"track_id": [2, 10], "genre": ["Hip-Hop", "Pop"],
                                 "relative_path": ["000/000002.mp3", "000/000010.mp3"]})
        manifest.to_csv(self.sample / "manifest.csv", index=False)
        validation.write_json(self.sample / "selection.json", {
            "manifest_sha256": validation.sha256(self.sample / "manifest.csv"),
            "official_features_sha256": validation.sha256(self.reference),
        })
        self.audio_root = self.root / "audio"
        (self.audio_root / "000").mkdir(parents=True)
        self.output = self.root / "report"

    def test_reads_requested_official_rows_in_requested_order(self):
        frame = validation.read_official_features(self.reference, [10, 2])
        self.assertEqual(frame.index.tolist(), [10, 2])
        self.assertEqual(tuple(frame.columns), FEATURE_NAMES)
        np.testing.assert_array_equal(frame.iloc[:, 0], [2., 1.])

    def test_rejects_missing_duplicate_or_non_finite_reference_rows(self):
        with self.assertRaisesRegex(ValueError, "ausentes"):
            validation.read_official_features(self.reference, [99])
        with self.assertRaises(ValueError):
            validation.read_official_features(self.reference, [2, 2])
        for rows in [[(2, np.ones(518)), (2, np.ones(518))], [(2, np.full(518, np.nan))]]:
            write_reference(self.reference, rows)
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                validation.read_official_features(self.reference, [2])

    def test_rejects_changed_official_schema(self):
        text = self.reference.read_text(encoding="utf-8")
        self.reference.write_text(text.replace("chroma_cens", "unknown", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "esquema"):
            validation.read_official_features(self.reference, [2])

    def test_sample_selection_is_reproducible_and_excludes_known_short_tracks(self):
        ids = pd.Series([2, 3, 4, 10, 11, 12, 98565])
        genres = pd.Series(["Hip-Hop"] * 3 + ["Pop"] * 3 + ["Hip-Hop"])
        values = np.arange(1., 8.)[:, None] * np.ones((7, 518))
        bundle = SimpleNamespace(track_ids=ids, y=genres,
                                 X=pd.DataFrame(values, columns=FEATURE_NAMES),
                                 diagnostics={"redundant_samples": 16})
        dataset_path = self.root / "dataset.csv"
        dataset_path.write_text("source fixture", encoding="utf-8")
        write_reference(self.reference, list(zip(ids, values)))
        with patch.object(validation, "load_dataset", return_value=bundle):
            first = validation.prepare_sample(dataset_path, self.reference, self.root / "first")
            second = validation.prepare_sample(dataset_path, self.reference, self.root / "second")
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.groupby("genre").size().to_dict(), {"Hip-Hop": 2, "Pop": 2})
        self.assertFalse(set(first.track_id).intersection(validation.KNOWN_SHORT_SMALL))
        self.assertEqual(first.track_id.nunique(), 4)

    def test_changed_official_file_requires_new_selection(self):
        with self.reference.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "features.csv alterado"):
            self.compare()

    def make_audio_files(self):
        for name in ["000002.mp3", "000010.mp3"]:
            (self.audio_root / "000" / name).write_bytes(b"mock audio; not a real MP3")

    def compare(self):
        return validation.run_comparison(self.sample, self.reference, self.audio_root, self.output)

    def test_missing_audio_is_incomplete_and_never_reported_as_success(self):
        with patch.object(validation, "extract_features") as extract:
            self.assertEqual(self.compare(), 2)
        extract.assert_not_called()
        report = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status_counts"], {"missing": 2})
        self.assertEqual(report["conclusion"], "incomplete")
        self.assertFalse((self.output / "differences.csv").exists())

    def test_successful_run_records_all_values_and_provenance(self):
        self.make_audio_files()
        def extracted(path):
            value = 1. if path.stem == "000002" else 2.
            return pd.Series(np.full(518, value, dtype=np.float32), index=FEATURE_NAMES)
        with patch.object(validation, "inspect_audio", return_value={"samplerate": 44100}), \
             patch.object(validation, "extract_features", side_effect=extracted):
            self.assertEqual(self.compare(), 0)
        details = pd.read_csv(self.output / "differences.csv")
        self.assertEqual(len(details), 2 * 518)
        self.assertTrue(details.within_tolerance.all())
        report = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status_counts"], {"within_tolerance": 2})
        self.assertEqual(len(report["audio"][0]["sha256"]), 64)
        self.assertEqual(len(report["extractor_sha256"]), 64)
        self.assertTrue(report["packages"])
        self.assertTrue((self.output / "family_statistic_summary.csv").is_file())

    def test_divergent_run_returns_one_and_keeps_differences(self):
        self.make_audio_files()
        vector = pd.Series(np.zeros(518, dtype=np.float32), index=FEATURE_NAMES)
        with patch.object(validation, "inspect_audio", return_value={}), \
             patch.object(validation, "extract_features", return_value=vector):
            self.assertEqual(self.compare(), 1)
        tracks = pd.read_csv(self.output / "tracks.csv")
        self.assertEqual(tracks.status.tolist(), ["divergent", "divergent"])
        self.assertTrue((self.output / "differences.csv").is_file())

    def test_extraction_error_is_recorded_and_does_not_stop_other_tracks(self):
        self.make_audio_files()
        with patch.object(validation, "inspect_audio", return_value={}), \
             patch.object(validation, "extract_features", side_effect=ValueError("decode failed")) as extract:
            self.assertEqual(self.compare(), 2)
        self.assertEqual(extract.call_count, 2)
        tracks = pd.read_csv(self.output / "tracks.csv")
        self.assertTrue(tracks.error.str.contains("decode failed").all())

    def test_modified_sources_and_existing_output_are_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.compare()
        self.output.rmdir()
        manifest_path = self.sample / "manifest.csv"
        manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Manifesto alterado"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
