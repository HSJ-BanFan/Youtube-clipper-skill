import tempfile
import unittest
from pathlib import Path

from translation.models import Cue
from translation.subtitles import (
    detect_subtitle_format,
    parse_srt,
    parse_subtitle_file,
    parse_vtt,
    validate_cues,
    validate_translations,
    write_bilingual_srt,
    write_translated_srt,
)


class SubtitleParserTests(unittest.TestCase):
    def test_detect_subtitle_format_accepts_srt_and_vtt_only(self):
        self.assertEqual(detect_subtitle_format(Path("sample.srt")), "srt")
        self.assertEqual(detect_subtitle_format(Path("sample.vtt")), "vtt")

        with self.assertRaisesRegex(ValueError, "subtitle_path must point to .srt or .vtt"):
            detect_subtitle_format(Path("sample.ass"))

    def test_parse_subtitle_file_reads_utf8_sig_and_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "﻿1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )

            cues = parse_subtitle_file(subtitle_path)

        self.assertEqual(cues[0].source, "hello")

    def test_parse_ordinary_srt(self):
        cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")

        self.assertEqual(
            cues,
            [
                Cue(
                    id="1",
                    index=1,
                    start="00:00:00,000",
                    end="00:00:01,000",
                    source="hello",
                    raw_timing="00:00:00,000 --> 00:00:01,000",
                )
            ],
        )

    def test_parse_multiline_srt_source(self):
        cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\nhello\nworld\n\n")

        self.assertEqual(cues[0].source, "hello\nworld")

    def test_parse_srt_without_trailing_blank_line(self):
        cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\nhello")

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].source, "hello")

    def test_parse_srt_with_crlf_newlines(self):
        cues = parse_srt("1\r\n00:00:00,000 --> 00:00:01,000\r\nhello\r\n\r\n")

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].source, "hello")

    def test_parse_srt_keeps_digit_only_source_line(self):
        cues = parse_srt(
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "42\n"
            "meaning\n\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,000\n"
            "done\n"
        )

        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].source, "42\nmeaning")

    def test_parse_srt_without_sequence_uses_internal_index_as_id(self):
        cues = parse_srt("00:00:00,000 --> 00:00:01,000\nhello\n\n")

        self.assertEqual(cues[0].id, "1")
        self.assertEqual(cues[0].index, 1)

    def test_parse_srt_non_numeric_identifier_line_uses_internal_index(self):
        cues = parse_srt("subtitle-abc\n00:00:00,000 --> 00:00:01,000\nhello\n\n")

        self.assertEqual(cues[0].id, "1")
        self.assertEqual(cues[0].source, "hello")

    def test_parse_empty_srt_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "subtitle file contains no cues"):
            parse_srt("")

    def test_parse_srt_with_empty_source_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "source is empty"):
            parse_srt("1\n00:00:00,000 --> 00:00:01,000\n\n")

    def test_parse_srt_rejects_malformed_block_instead_of_skipping_it(self):
        with self.assertRaisesRegex(ValueError, "invalid subtitle cue block near cue index 1: not a valid cue"):
            parse_srt(
                "not a valid cue\nmissing timing\n\n"
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"
            )

    def test_parse_srt_accepts_hourless_timestamps(self):
        cues = parse_srt("1\n01:23,456 --> 01:25,000\nhello\n\n")

        self.assertEqual(cues[0].start, "00:01:23,456")
        self.assertEqual(cues[0].end, "00:01:25,000")

    def test_parse_vtt_with_header_and_ordinary_cue(self):
        cues = parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n\n")

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].id, "1")
        self.assertEqual(cues[0].start, "00:00:00,000")
        self.assertEqual(cues[0].end, "00:00:01,000")
        self.assertEqual(cues[0].source, "hello")

    def test_parse_vtt_with_cue_identifier(self):
        cues = parse_vtt("WEBVTT\n\nintro-cue\n00:00:00.000 --> 00:00:01.000\nhello\n\n")

        self.assertEqual(cues[0].id, "intro-cue")

    def test_parse_vtt_with_timing_settings(self):
        cues = parse_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.000 align:start position:0%\n"
            "hello\n\n"
        )

        self.assertEqual(cues[0].raw_timing, "00:00:00.000 --> 00:00:01.000 align:start position:0%")
        self.assertEqual(cues[0].start, "00:00:00,000")
        self.assertEqual(cues[0].end, "00:00:01,000")

    def test_parse_vtt_accepts_hourless_timestamps(self):
        cues = parse_vtt("WEBVTT\n\n01:23.456 --> 01:25.000\nhello\n\n")

        self.assertEqual(cues[0].start, "00:01:23,456")
        self.assertEqual(cues[0].end, "00:01:25,000")

    def test_parse_vtt_rejects_malformed_cue_block_instead_of_skipping_it(self):
        with self.assertRaisesRegex(ValueError, "invalid subtitle cue block near cue index 1: bad cue"):
            parse_vtt(
                "WEBVTT\n\n"
                "bad cue\nmissing timing\n\n"
                "00:00:00.000 --> 00:00:01.000\nhello\n\n"
            )

    def test_parse_vtt_skips_note_style_and_region_blocks(self):
        cues = parse_vtt(
            "WEBVTT\n\n"
            "NOTE this is metadata\nignored\n\n"
            "STYLE\n::cue { color: white }\n\n"
            "REGION\nid:fred\n\n"
            "00:00:00.000 --> 00:00:01.000\nhello\n\n"
        )

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].source, "hello")

    def test_validate_cues_rejects_duplicate_ids(self):
        cues = [
            Cue(id="same", index=1, start="00:00:00,000", end="00:00:01,000", source="a"),
            Cue(id="same", index=2, start="00:00:02,000", end="00:00:03,000", source="b"),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate cue id: same"):
            validate_cues(cues)

    def test_validate_cues_rejects_missing_fields(self):
        cues = [Cue(id="1", index=1, start="00:00:00,000", end="", source="hello")]

        with self.assertRaisesRegex(ValueError, "cue id 1 end is empty"):
            validate_cues(cues)

    def test_validate_cues_rejects_non_contiguous_indexes(self):
        cues = [Cue(id="1", index=2, start="00:00:00,000", end="00:00:01,000", source="hello")]

        with self.assertRaisesRegex(ValueError, "cue index must be 1, got 2"):
            validate_cues(cues)


class SubtitleWriterTests(unittest.TestCase):
    def setUp(self):
        self.cues = [
            Cue(id="intro", index=1, start="00:00:00,000", end="00:00:01,000", source="hello"),
            Cue(id="2", index=2, start="00:00:02,000", end="00:00:03,000", source="world"),
        ]
        self.translations = {"intro": "你好", "2": "世界"}

    def test_validate_translations_requires_same_count(self):
        with self.assertRaisesRegex(ValueError, "translation count 1 does not match cue count 2"):
            validate_translations(self.cues, {"intro": "你好"})

    def test_validate_translations_reports_missing_cue_id(self):
        with self.assertRaisesRegex(ValueError, "missing translation for cue id 2"):
            validate_translations(self.cues, {"intro": "你好", "other": "别的"})

    def test_validate_translations_reports_blank_translation(self):
        with self.assertRaisesRegex(ValueError, "translation for cue id intro is empty"):
            validate_translations(self.cues, {"intro": "  ", "2": "世界"})

    def test_write_translated_srt_writes_only_translation_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "translated.srt"

            write_translated_srt(self.cues, self.translations, output_path)

            content = output_path.read_text(encoding="utf-8")

        self.assertEqual(
            content,
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\n世界\n\n",
        )

    def test_write_bilingual_srt_defaults_translation_before_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bilingual.srt"

            write_bilingual_srt(self.cues, self.translations, output_path)

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("你好\nhello", content)
        self.assertIn("世界\nworld", content)

    def test_write_bilingual_srt_can_write_source_before_translation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bilingual.srt"

            write_bilingual_srt(self.cues, self.translations, output_path, source_first=True)

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("hello\n你好", content)
        self.assertIn("world\n世界", content)


if __name__ == "__main__":
    unittest.main()
