import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translation.config import TranslationConfig
from translation.models import Cue, TranslationBatch
from translation.pipeline import (
    _build_batch_source_hash,
    parse_qa_response,
    parse_translation_response,
    run_translation_pipeline,
)
from translation.prompts import build_translation_prompt


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


class QAResponseParserTests(unittest.TestCase):
    def test_parse_qa_response_applies_fix_and_ignores_keep(self):
        candidates = [
            Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello"),
            Cue(id="2", index=2, start="00:00:02,000", end="00:00:03,000", source="world"),
        ]
        response = json.dumps(
            [
                {"id": "1", "action": "fix", "translation": "你好", "reason": "empty"},
                {"id": "2", "action": "keep", "translation": "world", "reason": "ok"},
            ]
        )

        fixes = parse_qa_response(response, candidates)

        self.assertEqual(fixes, {"1": "你好"})

    def test_parse_qa_response_accepts_fenced_json(self):
        candidates = [Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")]

        fixes = parse_qa_response('```json\n[{"id":"1","action":"fix","translation":"你好","reason":"empty"}]\n```', candidates)

        self.assertEqual(fixes, {"1": "你好"})

    def test_parse_qa_response_rejects_natural_language_wrapper(self):
        candidates = [Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")]

        with self.assertRaisesRegex(RuntimeError, "QA response is not valid JSON"):
            parse_qa_response('Here is JSON:\n[{"id":"1","action":"fix","translation":"你好","reason":"empty"}]', candidates)

    def test_parse_qa_response_requires_matching_ids_actions_and_non_empty_translation(self):
        candidates = [Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")]

        bad_responses = [
            '[{"id":"2","action":"fix","translation":"你好","reason":"empty"}]',
            '[{"id":"1","action":"rewrite","translation":"你好","reason":"empty"}]',
            '[{"id":"1","action":"fix","translation":"  ","reason":"empty"}]',
        ]

        for response in bad_responses:
            with self.subTest(response=response):
                with self.assertRaisesRegex(RuntimeError, "QA response"):
                    parse_qa_response(response, candidates)


class TranslationPipelineExecutionTests(unittest.TestCase):
    def test_non_dry_run_writes_outputs_cache_report_and_context_with_fake_provider(self):
        class FakeProvider:
            calls = 0
            prompts: list[str] = []

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls += 1
                FakeProvider.prompts.append(prompt)
                return json.dumps([{"id": "1", "translation": "你好"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            glossary_path = _write_glossary(root)
            output_dir = root / "out"
            cache_path = root / "translation-cache.sqlite3"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                cache_path=str(cache_path),
                glossary_path=str(glossary_path),
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                result = run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")
            context = (output_dir / "global_context.md").read_text(encoding="utf-8")
            cache_exists = cache_path.exists()
            cache_rows = _read_cache_rows(cache_path)
            prompt = FakeProvider.prompts[0]

        self.assertTrue(result.provider_called)
        self.assertEqual(FakeProvider.calls, 1)
        self.assertIn("你好", translated)
        self.assertIn("你好\nhello", bilingual)
        self.assertTrue(cache_exists)
        self.assertEqual(len(cache_rows), 1)
        self.assertEqual(json.loads(cache_rows[0]), [{"id": "1", "translation": "你好"}])
        self.assertNotIn("test-secret", cache_rows[0])
        self.assertIn("provider_calls: 1", report)
        self.assertIn("cache_misses: 1", report)
        self.assertIn("cache_hits: 0", report)
        self.assertNotIn("test-secret", report)
        self.assertIn("# Translation Global Context", context)
        self.assertNotIn("test-secret", context)
        self.assertIn("Glossary:", prompt)
        self.assertIn("hello => 你好", prompt)
        self.assertIn("Global Context:", prompt)
        self.assertIn("Translation Global Context", prompt)
        self.assertNotIn("test-secret", prompt)

    def test_second_run_with_same_cache_uses_cache_without_provider_call(self):
        class FakeProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls += 1
                return json.dumps([{"id": "1", "translation": "你好"}])

        class FailingProvider:
            def __init__(self, config):
                raise AssertionError("provider should not be constructed on full cache hit")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            cache_path = root / "translation-cache.sqlite3"
            first_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out1"),
                batch_size=1,
                cache_path=str(cache_path),
            )
            second_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out2"),
                batch_size=1,
                cache_path=str(cache_path),
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, first_config)
            with patch("translation.pipeline.OpenAICompatibleProvider", FailingProvider):
                result = run_translation_pipeline(subtitle_path, second_config)

            report = (root / "out2" / "translation_report.md").read_text(encoding="utf-8")
            translated = (root / "out2" / "translated.zh-CN.srt").read_text(encoding="utf-8")

        self.assertEqual(FakeProvider.calls, 1)
        self.assertFalse(result.provider_called)
        self.assertIn("你好", translated)
        self.assertIn("cache_hits: 1", report)
        self.assertIn("provider_calls: 0", report)
        self.assertIn("cache_misses: 0", report)

    def test_batch_source_hash_changes_when_prompt_context_changes(self):
        current_cue = Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")
        context_cue = Cue(id="2", index=2, start="00:00:02,000", end="00:00:03,000", source="world")
        without_context = TranslationBatch(
            batch_id=1,
            cues=(current_cue,),
            context_before=(),
            context_after=(),
        )
        with_context = TranslationBatch(
            batch_id=1,
            cues=(current_cue,),
            context_before=(),
            context_after=(context_cue,),
        )
        prompt_without_context = build_translation_prompt(without_context, "zh-CN")
        prompt_with_context = build_translation_prompt(with_context, "zh-CN")

        self.assertNotEqual(
            _build_batch_source_hash(prompt_without_context),
            _build_batch_source_hash(prompt_with_context),
        )

    def test_cache_disabled_calls_provider_every_run(self):
        class FakeProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls += 1
                return json.dumps([{"id": "1", "translation": f"你好{FakeProvider.calls}"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            cache_path = root / "translation-cache.sqlite3"
            first_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out1"),
                batch_size=1,
                cache_enabled=False,
                cache_path=str(cache_path),
            )
            second_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out2"),
                batch_size=1,
                cache_enabled=False,
                cache_path=str(cache_path),
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, first_config)
                run_translation_pipeline(subtitle_path, second_config)

            report = (root / "out2" / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(FakeProvider.calls, 2)
        self.assertFalse(cache_path.exists())
        self.assertIn("provider_calls: 1", report)
        self.assertIn("cache_hits: 0", report)
        self.assertIn("cache_misses: 0", report)

    def test_malformed_cached_json_falls_back_to_provider_and_overwrites_cache(self):
        class SeedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "初始"}])

        class FakeProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls += 1
                return json.dumps([{"id": "1", "translation": "修复后"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            cache_path = root / "translation-cache.sqlite3"
            seed_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "seed"),
                batch_size=1,
                cache_path=str(cache_path),
            )
            recovery_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out"),
                batch_size=1,
                cache_path=str(cache_path),
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", SeedProvider):
                run_translation_pipeline(subtitle_path, seed_config)
            _overwrite_only_cache_result(cache_path, "not json")

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, recovery_config)

            report = (root / "out" / "translation_report.md").read_text(encoding="utf-8")
            cache_rows = _read_cache_rows(cache_path)

        self.assertEqual(FakeProvider.calls, 1)
        self.assertIn("cache_hits: 0", report)
        self.assertIn("cache_misses: 1", report)
        self.assertIn("provider_calls: 1", report)
        self.assertEqual(json.loads(cache_rows[0]), [{"id": "1", "translation": "修复后"}])
        self.assertNotIn("not json", cache_rows[0])

    def test_existing_report_or_context_requires_overwrite_before_provider_construction(self):
        class FailingProvider:
            def __init__(self, config):
                raise AssertionError("provider should not be constructed")

        for existing_name in ("translation_report.md", "global_context.md"):
            with self.subTest(existing_name=existing_name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    subtitle_path = _write_one_cue_srt(root)
                    output_dir = root / "out"
                    output_dir.mkdir()
                    (output_dir / existing_name).write_text("existing", encoding="utf-8")
                    config = TranslationConfig(api_key="test-secret", output_dir=str(output_dir))

                    with patch("translation.pipeline.OpenAICompatibleProvider", FailingProvider):
                        with self.assertRaisesRegex(FileExistsError, "already exists"):
                            run_translation_pipeline(subtitle_path, config)

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
            self.assertFalse((output_dir / "translation_report.md").exists())
            self.assertFalse((output_dir / "global_context.md").exists())

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
                cache_enabled=False,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FlakyProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(FlakyProvider.attempts, 2)
        self.assertIn("provider_calls: 2", report)
        self.assertIn("retries: 1", report)

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
                cache_enabled=False,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", BrokenProvider):
                with self.assertRaisesRegex(RuntimeError, r"(?s)batch_id 1.*2 attempts.*not valid JSON"):
                    run_translation_pipeline(subtitle_path, config)

    def test_provider_failure_writes_no_context_report_or_srt_outputs(self):
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
                cache_enabled=False,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", BrokenProvider):
                with self.assertRaisesRegex(RuntimeError, r"(?s)batch_id 1.*2 attempts.*not valid JSON"):
                    run_translation_pipeline(subtitle_path, config)

            self.assertFalse((output_dir / "translated.zh-CN.srt").exists())
            self.assertFalse((output_dir / "bilingual.srt").exists())
            self.assertFalse((output_dir / "global_context.md").exists())
            self.assertFalse((output_dir / "translation_report.md").exists())

    def test_suspicious_only_without_candidates_does_not_call_review_suspicious(self):
        class FakeProvider:
            review_calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "你好"}])

            def review_suspicious(self, prompt):
                FakeProvider.review_calls += 1
                raise AssertionError("review_suspicious should not be called")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                cache_enabled=False,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(FakeProvider.review_calls, 0)
        self.assertIn("qa_candidates: 0", report)
        self.assertIn("qa_provider_calls: 0", report)

    def test_suspicious_only_applies_fix_and_keep_to_outputs_and_report(self):
        class FakeProvider:
            review_calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps(
                    [
                        {"id": "1", "translation": "As an AI, I cannot help."},
                        {"id": "2", "translation": "world"},
                    ]
                )

            def review_suspicious(self, prompt):
                FakeProvider.review_calls += 1
                return json.dumps(
                    [
                        {"id": "1", "action": "fix", "translation": "你好", "reason": "fixed refusal"},
                        {"id": "2", "action": "keep", "translation": "world", "reason": "not obvious"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_suspicious_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=2,
                cache_enabled=False,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(FakeProvider.review_calls, 1)
        self.assertIn("你好", translated)
        self.assertIn("world", translated)
        self.assertNotIn("As an AI", translated)
        self.assertIn("你好\nhello", bilingual)
        self.assertIn("world\nopen https://example.test/docs and read it", bilingual)
        self.assertIn("qa_candidates: 2", report)
        self.assertIn("qa_provider_calls: 1", report)
        self.assertIn("qa_fixed: 1", report)
        self.assertIn("qa_kept: 1", report)
        self.assertNotIn("test-secret", report)

    def test_malformed_qa_json_raises_clear_runtime_error_and_writes_no_outputs(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "As an AI, I cannot help."}])

            def review_suspicious(self, prompt):
                return "not json"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                cache_enabled=False,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                with self.assertRaisesRegex(RuntimeError, "QA response is not valid JSON"):
                    run_translation_pipeline(subtitle_path, config)

            self.assertFalse((output_dir / "translated.zh-CN.srt").exists())
            self.assertFalse((output_dir / "bilingual.srt").exists())
            self.assertFalse((output_dir / "translation_report.md").exists())

    def test_qa_none_skips_review_suspicious(self):
        self._assert_qa_disabled_skips_review_suspicious("none")

    def test_qa_off_skips_review_suspicious(self):
        self._assert_qa_disabled_skips_review_suspicious("off")

    def _assert_qa_disabled_skips_review_suspicious(self, qa_mode):
        class FakeProvider:
            review_calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "As an AI, I cannot help."}])

            def review_suspicious(self, prompt):
                FakeProvider.review_calls += 1
                raise AssertionError("review_suspicious should not be called")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                cache_enabled=False,
                qa_mode=qa_mode,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

        self.assertEqual(FakeProvider.review_calls, 0)

    def test_existing_translated_or_bilingual_outputs_require_overwrite_before_provider_call(self):
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


def _write_one_cue_srt(temp_dir: Path) -> Path:
    subtitle_path = temp_dir / "sample.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
        encoding="utf-8",
    )
    return subtitle_path


def _write_two_cue_srt(temp_dir: Path) -> Path:
    subtitle_path = temp_dir / "sample.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nworld\n\n",
        encoding="utf-8",
    )
    return subtitle_path


def _write_two_suspicious_cue_srt(temp_dir: Path) -> Path:
    subtitle_path = temp_dir / "sample.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nopen https://example.test/docs and read it\n\n",
        encoding="utf-8",
    )
    return subtitle_path


def _write_glossary(temp_dir: Path) -> Path:
    glossary_path = temp_dir / "glossary.md"
    glossary_path.write_text("hello => 你好", encoding="utf-8")
    return glossary_path


def _read_cache_rows(cache_path: Path) -> list[str]:
    connection = sqlite3.connect(cache_path)
    try:
        rows = connection.execute("SELECT result_json FROM translation_cache").fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def _overwrite_only_cache_result(cache_path: Path, result_json: str) -> None:
    connection = sqlite3.connect(cache_path)
    try:
        connection.execute("UPDATE translation_cache SET result_json = ?", (result_json,))
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
