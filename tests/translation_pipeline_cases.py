import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translation.config import TranslationConfig
from translation.models import Cue
from translation.pipeline import parse_translation_response, run_translation_pipeline


ONE_CUE = [Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")]


class TranslationResponseParserTests(unittest.TestCase):
    def test_parse_translation_response_accepts_json_array(self):
        result = parse_translation_response('[{"id":"1","translation":"你好"}]', ONE_CUE, batch_id=7)

        self.assertEqual(result, {"1": "你好"})

    def test_parse_translation_response_accepts_fenced_json_array(self):
        result = parse_translation_response('```json\n[{"id":"1","translation":"你好"}]\n```', ONE_CUE, batch_id=7)

        self.assertEqual(result, {"1": "你好"})

    def test_parse_translation_response_rejects_explanation_around_json(self):
        with self.assertRaisesRegex(ValueError, "batch_id 7"):
            parse_translation_response('Here is JSON:\n[{"id":"1","translation":"你好"}]', ONE_CUE, batch_id=7)

    def test_parse_translation_response_requires_expected_count(self):
        with self.assertRaisesRegex(ValueError, "batch_id 7"):
            parse_translation_response("[]", ONE_CUE, batch_id=7)

    def test_parse_translation_response_requires_matching_id(self):
        with self.assertRaisesRegex(ValueError, "cue id 1"):
            parse_translation_response('[{"id":"2","translation":"你好"}]', ONE_CUE, batch_id=7)

    def test_parse_translation_response_rejects_empty_translation(self):
        with self.assertRaisesRegex(ValueError, "cue id 1"):
            parse_translation_response('[{"id":"1","translation":"  "}]', ONE_CUE, batch_id=7)


class TranslationPipelineExecutionTests(unittest.TestCase):
    def test_non_dry_run_writes_translated_and_bilingual_srt_with_fake_provider(self):
        class FakeProvider:
            called = False

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.called = True
                return json.dumps(
                    [
                        {"id": "1", "translation": "你好"},
                        {"id": "2", "translation": "世界"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            config = TranslationConfig(api_key="test-secret", output_dir=str(output_dir), batch_size=2)

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                result = run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")

        self.assertTrue(FakeProvider.called)
        self.assertTrue(result.provider_called)
        self.assertIn("你好", translated)
        self.assertIn("世界", translated)
        self.assertIn("你好\nhello", bilingual)
        self.assertIn("世界\nworld", bilingual)

    def test_dry_run_does_not_call_provider_or_write_outputs(self):
        class FailingProvider:
            def __init__(self, config):
                raise AssertionError("provider should not be constructed")

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            config = TranslationConfig(api_key="test-secret", output_dir=str(output_dir), dry_run=True)

            with patch("translation.pipeline.OpenAICompatibleProvider", FailingProvider):
                result = run_translation_pipeline(subtitle_path, config)

            self.assertFalse((output_dir / "translated.zh-CN.srt").exists())
            self.assertFalse((output_dir / "bilingual.srt").exists())

        self.assertFalse(result.provider_called)

    def test_malformed_json_retries_then_succeeds(self):
        class FlakyProvider:
            attempts = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FlakyProvider.attempts += 1
                if FlakyProvider.attempts == 1:
                    return "not json"
                return json.dumps(
                    [
                        {"id": "1", "translation": "你好"},
                        {"id": "2", "translation": "世界"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=2,
                max_retries=1,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FlakyProvider):
                run_translation_pipeline(subtitle_path, config)

        self.assertEqual(FlakyProvider.attempts, 2)

    def test_final_batch_failure_reports_batch_id_and_attempt_count(self):
        class BrokenProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return "not json"

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=2,
                max_retries=1,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", BrokenProvider):
                with self.assertRaisesRegex(RuntimeError, r"(?s)batch_id 1.*2 attempts.*not valid JSON"):
                    run_translation_pipeline(subtitle_path, config)

    def test_existing_outputs_require_overwrite_before_provider_call(self):
        class FailingProvider:
            def __init__(self, config):
                raise AssertionError("provider should not be constructed")

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            (output_dir / "translated.zh-CN.srt").write_text("existing", encoding="utf-8")
            config = TranslationConfig(api_key="test-secret", output_dir=str(output_dir))

            with patch("translation.pipeline.OpenAICompatibleProvider", FailingProvider):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    run_translation_pipeline(subtitle_path, config)


def _write_two_cue_srt(temp_dir: Path) -> Path:
    subtitle_path = temp_dir / "sample.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nworld\n\n",
        encoding="utf-8",
    )
    return subtitle_path


if __name__ == "__main__":
    unittest.main()
