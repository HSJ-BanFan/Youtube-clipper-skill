import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translation.config import TranslationConfig
from translation.pipeline import run_translation_pipeline


def _write_auto_sub_srt(root: Path) -> Path:
    subtitle_path = root / "auto.srt"
    subtitle_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "hello world this is a test\n\n"
        "2\n"
        "00:00:02,000 --> 00:00:04,000\n"
        "of the semantic segmentation pipeline\n\n"
        "3\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "which should produce refined units\n\n",
        encoding="utf-8",
    )
    return subtitle_path


def _make_semantic_response(token_count: int) -> str:
    return json.dumps({"segments": [{"start_token": 0, "end_token": token_count}]})


def _make_split_semantic_response(token_count: int) -> str:
    """Return two-unit semantic response so semantic result differs from single rule unit."""
    mid = token_count // 2 or 1
    return json.dumps({"segments": [
        {"start_token": 0, "end_token": mid},
        {"start_token": mid, "end_token": token_count},
    ]})


class SemanticPipelineDisabledTests(unittest.TestCase):
    def test_semantic_disabled_does_not_call_provider_segment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_auto_sub_srt(root)
            config = TranslationConfig(
                preprocess_auto_subs=True,
                semantic_segmentation_enabled=False,
                dry_run=True,
            )

            result = run_translation_pipeline(subtitle_path, config)

        self.assertFalse(result.provider_called)
        self.assertTrue(result.cue_count > 0)

    def test_semantic_disabled_produces_no_semantic_stats_in_report(self):
        class FakeProvider:
            def __init__(self, config):
                pass

            def translate_batch(self, prompt):
                return json.dumps([{"id": cue_id, "translation": "翻译"} for cue_id in _extract_ids(prompt)])

            def review_suspicious(self, prompt):
                return json.dumps([])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_auto_sub_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                preprocess_auto_subs=True,
                semantic_segmentation_enabled=False,
                qa_mode="none",
                output_dir=str(output_dir),
                batch_size=80,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertNotIn("Semantic Segmentation", report)


class SemanticPipelineEnabledTests(unittest.TestCase):
    def test_semantic_enabled_calls_segment_semantically_and_writes_report_section(self):
        segment_calls: list[str] = []

        class FakeProvider:
            def __init__(self, config):
                pass

            def translate_batch(self, prompt):
                return json.dumps([{"id": cue_id, "translation": "翻译"} for cue_id in _extract_ids(prompt)])

            def review_suspicious(self, prompt):
                return json.dumps([])

            def segment_semantically(self, prompt, model=None):
                segment_calls.append(prompt)
                token_count = prompt.count('"token_index"')
                return _make_semantic_response(token_count)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_auto_sub_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                preprocess_auto_subs=True,
                semantic_segmentation_enabled=True,
                semantic_segmentation_mode="hybrid",
                qa_mode="none",
                output_dir=str(output_dir),
                batch_size=80,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(len(segment_calls), 1)
        self.assertIn("## Semantic Segmentation", report)
        self.assertIn("- enabled: True", report)
        self.assertIn("- attempted: True", report)
        self.assertIn("- semantic_fallback_used: False", report)

    def test_semantic_provider_failure_falls_back_to_rules(self):
        class FakeProvider:
            def __init__(self, config):
                pass

            def translate_batch(self, prompt):
                return json.dumps([{"id": cue_id, "translation": "翻译"} for cue_id in _extract_ids(prompt)])

            def review_suspicious(self, prompt):
                return json.dumps([])

            def segment_semantically(self, prompt, model=None):
                raise RuntimeError("LLM unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_auto_sub_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                preprocess_auto_subs=True,
                semantic_segmentation_enabled=True,
                semantic_segmentation_fallback_to_rules=True,
                qa_mode="none",
                output_dir=str(output_dir),
                batch_size=80,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertIn("## Semantic Segmentation", report)
        self.assertIn("- semantic_fallback_used: True", report)
        self.assertIn("provider_error", report)

    def test_semantic_enabled_writes_rule_artifacts_when_semantic_result_differs(self):
        """Semantic response splits rule single unit into two; pipeline must write rule_* audit artifacts."""

        class FakeProvider:
            def __init__(self, config):
                pass

            def translate_batch(self, prompt):
                return json.dumps([{"id": cue_id, "translation": "翻译"} for cue_id in _extract_ids(prompt)])

            def review_suspicious(self, prompt):
                return json.dumps([])

            def segment_semantically(self, prompt, model=None):
                token_count = prompt.count('"token_index"')
                return _make_split_semantic_response(token_count)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_auto_sub_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                preprocess_auto_subs=True,
                semantic_segmentation_enabled=True,
                qa_mode="none",
                output_dir=str(output_dir),
                batch_size=80,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            rule_report = output_dir / "rule_segmentation_report.md"
            rule_units = output_dir / "rule_translation_units.json"
            rule_cue_map = output_dir / "rule_cue_map.json"
            rule_source = output_dir / "rule_segmented_source.srt"
            semantic_source = output_dir / "segmented_source.srt"
            semantic_source_exists = semantic_source.exists()
            rule_report_exists = rule_report.exists()
            rule_units_exists = rule_units.exists()
            rule_cue_map_exists = rule_cue_map.exists()
            rule_source_exists = rule_source.exists()

        self.assertTrue(semantic_source_exists)
        self.assertTrue(rule_report_exists)
        self.assertTrue(rule_units_exists)
        self.assertTrue(rule_cue_map_exists)
        self.assertTrue(rule_source_exists)

    def test_semantic_too_many_tokens_falls_back_without_provider_call(self):
        segment_calls: list[str] = []

        class FakeProvider:
            def __init__(self, config):
                pass

            def translate_batch(self, prompt):
                return json.dumps([{"id": cue_id, "translation": "翻译"} for cue_id in _extract_ids(prompt)])

            def review_suspicious(self, prompt):
                return json.dumps([])

            def segment_semantically(self, prompt, model=None):
                segment_calls.append(prompt)
                return "{}"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_auto_sub_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                preprocess_auto_subs=True,
                semantic_segmentation_enabled=True,
                semantic_segmentation_fallback_to_rules=True,
                semantic_segmentation_max_tokens_per_request=1,
                qa_mode="none",
                output_dir=str(output_dir),
                batch_size=80,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(len(segment_calls), 0)
        self.assertIn("- semantic_fallback_used: True", report)
        self.assertIn("too_many_tokens", report)


def _extract_ids(prompt: str) -> list[str]:
    import re

    section = prompt
    if "Current cues to translate:" in prompt and "After context:" in prompt:
        section = prompt.split("Current cues to translate:", 1)[1].split("After context:", 1)[0]
    return re.findall(r'"id":\s*"([^"]+)"', section)


if __name__ == "__main__":
    unittest.main()
