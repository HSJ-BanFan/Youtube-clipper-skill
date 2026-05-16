from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from translation.segmentation import (
    SegmentBoundaryType,
    SegmentationOptions,
    SegmentationValidationError,
    SubtitleSegmentationSource,
    segment_subtitles,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class SegmentationLibraryTests(unittest.TestCase):
    def test_full_vtt_window_extracts_processing_window_and_marks_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:01.000\n"
                "before\n\n"
                "00:00:01.000 --> 00:00:03.000\n"
                "alpha beta\n\n"
                "00:00:03.000 --> 00:00:05.000\n"
                "gamma delta\n\n"
                "00:00:05.000 --> 00:00:07.000\n"
                "epsilon zeta\n\n",
                encoding="utf-8",
            )
            source = SubtitleSegmentationSource(
                mode="full_vtt_window",
                full_vtt_path=subtitle_path,
                clip_start_ms=2000,
                clip_end_ms=6000,
                padding_before_ms=1000,
                padding_after_ms=1000,
            )
            options = SegmentationOptions(max_source_cues=1)

            result = segment_subtitles(source, options)

        self.assertEqual(result.processing_start_ms, 1000)
        self.assertEqual(result.processing_end_ms, 7000)
        self.assertEqual(result.stats.extracted_cue_count, 3)
        self.assertEqual(len(result.units), 3)
        self.assertEqual(result.units[0].boundary_type, SegmentBoundaryType.CROSSES_CLIP_START)
        self.assertEqual(result.units[1].boundary_type, SegmentBoundaryType.INSIDE_CLIP)
        self.assertEqual(result.units[2].boundary_type, SegmentBoundaryType.CROSSES_CLIP_END)

    def test_single_file_srt_uses_cue_proportional_timing_and_reports_degraded_precision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n",
                encoding="utf-8",
            )
            source = SubtitleSegmentationSource(mode="single_file", subtitle_path=subtitle_path)

            result = segment_subtitles(source, SegmentationOptions())

        self.assertTrue(result.warnings)
        self.assertTrue(any(warning.code == "timing_precision_degraded" for warning in result.warnings))
        self.assertEqual({token.timing_source for token in result.active_tokens}, {"cue_proportional"})
        self.assertTrue(all(unit.boundary_type == SegmentBoundaryType.INSIDE_CLIP for unit in result.units))

    def test_vtt_inline_timing_is_preferred_and_tags_are_stripped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:03.000\n"
                "<00:00:00.000><c>hello</c> <00:00:01.500><c>world</c>\n\n",
                encoding="utf-8",
            )
            source = SubtitleSegmentationSource(mode="single_file", subtitle_path=subtitle_path)

            result = segment_subtitles(source, SegmentationOptions(max_source_cues=1))

        self.assertEqual([token.text for token in result.active_tokens], ["hello", "world"])
        self.assertEqual(result.active_tokens[0].start_ms, 0)
        self.assertEqual(result.active_tokens[0].end_ms, 1500)
        self.assertEqual(result.active_tokens[1].start_ms, 1500)
        self.assertEqual(result.active_tokens[1].end_ms, 3000)
        self.assertEqual({token.timing_source for token in result.active_tokens}, {"vtt_inline"})

    def test_duplicate_cue_cleanup_is_local_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:01.000\n"
                "hello there\n\n"
                "00:00:01.020 --> 00:00:02.000\n"
                "hello there\n\n"
                "00:00:03.000 --> 00:00:04.000\n"
                "bridge text\n\n"
                "00:00:04.000 --> 00:00:05.000\n"
                "hello there\n\n",
                encoding="utf-8",
            )
            result = segment_subtitles(
                SubtitleSegmentationSource(mode="single_file", subtitle_path=subtitle_path),
                SegmentationOptions(max_source_cues=1),
            )

        self.assertEqual(result.stats.removed_duplicate_cue_count, 1)
        self.assertEqual([unit.source_text for unit in result.units], ["hello there", "bridge text", "hello there"])

    def test_rolling_overlap_cleanup_removes_prefix_overlap_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:02.000\n"
                "move on to do a number\n\n"
                "00:00:02.000 --> 00:00:04.000\n"
                "to do a number guesser\n\n",
                encoding="utf-8",
            )
            result = segment_subtitles(
                SubtitleSegmentationSource(mode="single_file", subtitle_path=subtitle_path),
                SegmentationOptions(max_unit_chars=200),
            )

        self.assertEqual([token.text for token in result.active_tokens], ["move", "on", "to", "do", "a", "number", "guesser"])
        self.assertEqual(result.stats.removed_rolling_overlap_token_count, 4)

    def test_strong_boundary_phrase_match_splits_across_cue_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:02.000\n"
                "keeping track of how many times with the\n\n"
                "00:00:02.000 --> 00:00:04.000\n"
                "user and the computer wins now we're\n\n"
                "00:00:04.000 --> 00:00:06.000\n"
                "going to move on to the next project\n\n",
                encoding="utf-8",
            )
            result = segment_subtitles(
                SubtitleSegmentationSource(mode="single_file", subtitle_path=subtitle_path),
                SegmentationOptions(max_unit_chars=200, max_source_cues=8),
            )

        self.assertGreaterEqual(len(result.units), 2)
        self.assertNotIn("now we're going to move on", result.units[0].source_text)
        self.assertTrue(result.units[1].source_text.startswith("now we're going to move on"))

    def test_zero_duration_token_raises_segmentation_validation_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:01,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )

            with self.assertRaises(SegmentationValidationError):
                segment_subtitles(
                    SubtitleSegmentationSource(mode="single_file", subtitle_path=subtitle_path),
                    SegmentationOptions(),
                )

    def test_payload_renderers_include_metadata_and_authoritative_source_spans(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n",
                encoding="utf-8",
            )
            result = segment_subtitles(
                SubtitleSegmentationSource(mode="single_file", subtitle_path=subtitle_path),
                SegmentationOptions(max_source_cues=1),
            )

        units_payload = result.to_translation_units_payload()
        cue_map_payload = result.to_cue_map_payload()
        srt_text = result.to_segmented_srt_text()
        report_text = result.to_report_markdown()

        self.assertEqual(units_payload["schema_version"], "segmentation.v1")
        self.assertIn("segmentation_strategy_version", units_payload)
        self.assertIn("timing_strategy_version", units_payload)
        self.assertIn("source_spans", units_payload["units"][0])
        self.assertNotIn("translation", units_payload["units"][0])
        self.assertEqual(cue_map_payload["schema_version"], "segmentation.v1")
        self.assertIn(result.units[0].unit_id, cue_map_payload["units"])
        self.assertIn("hello world", srt_text)
        self.assertIn("coverage is based on cleaned active token stream", report_text)


class SegmentationCliTests(unittest.TestCase):
    def test_cli_writes_four_artifacts_and_prints_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n",
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "out"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "translation.segment_subtitles",
                    "--mode",
                    "single_file",
                    "--subtitle",
                    str(subtitle_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

            artifact_names = sorted(path.name for path in output_dir.iterdir())
            translation_units = json.loads((output_dir / "translation_units.json").read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            artifact_names,
            [
                "cue_map.json",
                "segmentation_report.md",
                "segmented_source.srt",
                "translation_units.json",
            ],
        )
        self.assertIn("mode: single_file", result.stdout)
        self.assertIn("units:", result.stdout)
        self.assertEqual(translation_units["schema_version"], "segmentation.v1")

    def test_cli_avoids_partial_outputs_on_fatal_validation_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:01,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "out"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "translation.segment_subtitles",
                    "--mode",
                    "single_file",
                    "--subtitle",
                    str(subtitle_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("SegmentationValidationError", result.stderr)
        self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
