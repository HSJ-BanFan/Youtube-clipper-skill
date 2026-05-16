import json
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from translation.config import TranslationConfig
from translation.models import BatchState, Cue, ErrorType, MinimalBatchReportEntry, TranslationBatch
from translation.pipeline import (
    BatchExecutionResult,
    BatchStatsDelta,
    _build_batch_source_hash,
    _build_structured_batch_record,
    _classify_structured_cue_id,
    _run_translation_batches,
    parse_qa_response,
    parse_translation_response,
    run_translation_pipeline,
)
from translation.prompts import build_translation_prompt
from translation.provider import OpenAICompatibleProvider, ProviderError


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

    def test_parse_translation_response_accepts_minimal_structured_cue_id_array(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        before_cue = Cue(id="before-1", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        batch = TranslationBatch(
            batch_id=7,
            cues=(target_cue,),
            context_before=(before_cue,),
            context_after=(),
        )
        batch_record = _build_structured_batch_record(batch)

        result = parse_translation_response(
            '[{"cue_id":"target-2","translation":"你好"}]',
            batch.cues,
            batch_id=7,
            translation_id_key="cue_id",
            batch_record=batch_record,
        )

        self.assertEqual(result, {"target-2": "你好"})

    def test_parse_translation_response_rejects_missing_structured_cue_id_key(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        batch = TranslationBatch(batch_id=7, cues=(target_cue,), context_before=(), context_after=())
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                '[{"translation":"你好"}]',
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.SCHEMA_MISMATCH)

    def test_parse_translation_response_rejects_invalid_json_on_structured_path(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        batch = TranslationBatch(batch_id=7, cues=(target_cue,), context_before=(), context_after=())
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                "not json",
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.INVALID_JSON)

    def test_parse_translation_response_rejects_schema_mismatch_on_structured_path(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        batch = TranslationBatch(batch_id=7, cues=(target_cue,), context_before=(), context_after=())
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                '{"cue_id":"target-2","translation":"你好"}',
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.SCHEMA_MISMATCH)

    def test_parse_translation_response_rejects_extra_structured_key(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        batch = TranslationBatch(batch_id=7, cues=(target_cue,), context_before=(), context_after=())
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                '[{"cue_id":"target-2","translation":"你好","start":"00:00:02,000"}]',
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.SCHEMA_MISMATCH)

    def test_parse_translation_response_rejects_empty_translation_on_structured_path(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        batch = TranslationBatch(batch_id=7, cues=(target_cue,), context_before=(), context_after=())
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                '[{"cue_id":"target-2","translation":"  "}]',
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.EMPTY_TRANSLATION)

    def test_parse_translation_response_rejects_duplicate_cue_id(self):
        first = Cue(id="target-1", index=1, start="00:00:00,000", end="00:00:01,000", source="first")
        second = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="second")
        batch = TranslationBatch(batch_id=7, cues=(first, second), context_before=(), context_after=())
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                '[{"cue_id":"target-1","translation":"一"},{"cue_id":"target-1","translation":"二"}]',
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.DUPLICATE_CUE_ID)

    def test_parse_translation_response_classifies_context_cue_output_violation(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        before_cue = Cue(id="before-1", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        batch = TranslationBatch(
            batch_id=7,
            cues=(target_cue,),
            context_before=(before_cue,),
            context_after=(),
        )
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                '[{"cue_id":"before-1","translation":"上下文"}]',
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.CONTEXT_CUE_OUTPUT_VIOLATION)

    def test_parse_translation_response_classifies_invalid_cue_id_for_unknown_non_context_id(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        before_cue = Cue(id="before-1", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        batch = TranslationBatch(
            batch_id=7,
            cues=(target_cue,),
            context_before=(before_cue,),
            context_after=(),
        )
        batch_record = _build_structured_batch_record(batch)

        with self.assertRaises(ValueError) as raised:
            parse_translation_response(
                '[{"cue_id":"missing-3","translation":"陌生"}]',
                batch.cues,
                batch_id=7,
                translation_id_key="cue_id",
                batch_record=batch_record,
            )

        self.assertEqual(raised.exception.error_type, ErrorType.INVALID_CUE_ID)

    def test_parse_translation_response_reconciles_fallback_cue_id_back_to_original_target_id(self):
        target_cue = Cue(id="shared-id", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        before_cue = Cue(id="shared-id", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        batch = TranslationBatch(
            batch_id=7,
            cues=(target_cue,),
            context_before=(before_cue,),
            context_after=(),
        )
        batch_record = _build_structured_batch_record(batch)

        result = parse_translation_response(
            '[{"cue_id":"2","translation":"你好"}]',
            batch.cues,
            batch_id=7,
            translation_id_key="cue_id",
            batch_record=batch_record,
        )

        self.assertEqual(result, {"shared-id": "你好"})

    def test_parse_translation_response_rebuilds_batch_result_in_original_target_order(self):
        first_target = Cue(id="shared-a", index=2, start="00:00:02,000", end="00:00:03,000", source="first")
        second_target = Cue(id="shared-b", index=3, start="00:00:03,000", end="00:00:04,000", source="second")
        before_cue = Cue(id="shared-a", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        batch = TranslationBatch(
            batch_id=7,
            cues=(first_target, second_target),
            context_before=(before_cue,),
            context_after=(),
        )
        batch_record = _build_structured_batch_record(batch)

        result = parse_translation_response(
            '[{"cue_id":"shared-b","translation":"第二"},{"cue_id":"2","translation":"第一"}]',
            batch.cues,
            batch_id=7,
            translation_id_key="cue_id",
            batch_record=batch_record,
        )

        self.assertEqual(list(result.items()), [("shared-a", "第一"), ("shared-b", "第二")])


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
            '[{"id":"1","action":"fix","translation":"你好","reason":"  "}]',
            '[{"id":"1","action":"fix","translation":"你好"}]',
        ]

        for response in bad_responses:
            with self.subTest(response=response):
                with self.assertRaisesRegex(RuntimeError, "QA response"):
                    parse_qa_response(response, candidates)

    def test_parse_qa_response_rejects_duplicate_ids_wrong_item_count_and_reordered_ids(self):
        candidates = [
            Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello"),
            Cue(id="2", index=2, start="00:00:01,000", end="00:00:02,000", source="world"),
        ]

        with self.assertRaisesRegex(RuntimeError, "duplicate id"):
            parse_qa_response(
                '[{"id":"1","action":"fix","translation":"你好","reason":"fix"},{"id":"1","action":"keep","translation":"world","reason":"keep"}]',
                candidates,
            )

        with self.assertRaisesRegex(RuntimeError, "item count"):
            parse_qa_response('[{"id":"1","action":"fix","translation":"你好","reason":"fix"}]', candidates)

        with self.assertRaisesRegex(RuntimeError, "order"):
            parse_qa_response(
                '[{"id":"2","action":"keep","translation":"world","reason":"keep"},{"id":"1","action":"fix","translation":"你好","reason":"fix"}]',
                candidates,
            )


class TranslationProviderErrorTests(unittest.TestCase):
    def test_provider_timeout_is_classified(self):
        config = TranslationConfig(api_key="test-secret", base_url="https://example.test/v1")
        provider = OpenAICompatibleProvider(config)

        with patch("translation.provider.request.urlopen", side_effect=TimeoutError):
            with self.assertRaises(ProviderError) as raised:
                provider.translate_batch("hello")

        self.assertEqual(raised.exception.error_type, ErrorType.PROVIDER_TIMEOUT)

    def test_provider_http_5xx_is_classified(self):
        config = TranslationConfig(api_key="test-secret", base_url="https://example.test/v1")
        provider = OpenAICompatibleProvider(config)
        error = HTTPError(
            url="https://example.test/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

        with patch("translation.provider.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderError) as raised:
                provider.translate_batch("hello")

        self.assertEqual(raised.exception.error_type, ErrorType.PROVIDER_HTTP_5XX)

    def test_provider_request_failure_is_classified(self):
        config = TranslationConfig(api_key="test-secret", base_url="https://example.test/v1")
        provider = OpenAICompatibleProvider(config)

        with patch("translation.provider.request.urlopen", side_effect=URLError("boom")):
            with self.assertRaises(ProviderError) as raised:
                provider.translate_batch("hello")

        self.assertEqual(raised.exception.error_type, ErrorType.PROVIDER_REQUEST_FAILED)

    def test_provider_missing_choices_is_classified(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return b'{"choices": []}'

        config = TranslationConfig(api_key="test-secret", base_url="https://example.test/v1")
        provider = OpenAICompatibleProvider(config)

        with patch("translation.provider.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(ProviderError) as raised:
                provider.translate_batch("hello")

        self.assertEqual(raised.exception.error_type, ErrorType.PROVIDER_MISSING_CHOICES)


class TranslationPipelineExecutionTests(unittest.TestCase):
    def test_non_dry_run_writes_outputs_cache_report_and_context_with_fake_provider(self):
        captured_prompts: list[str] = []

        class FakeProvider:
            calls = 0

            def __init__(self, config):
                self.config = config
                self.prompts: list[str] = []

            def translate_batch(self, prompt):
                FakeProvider.calls += 1
                self.prompts.append(prompt)
                captured_prompts.append(prompt)
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
            prompt = captured_prompts[0]

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

    def test_concurrency_one_two_and_four_keep_same_output_and_batch_report_order(self):
        class SlowProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if '"id": "1",\n    "source": "hello"' in prompt:
                    time.sleep(0.05)
                    return json.dumps([{"id": "1", "translation": "你好"}])
                if '"id": "2",\n    "source": "world"' in prompt:
                    return json.dumps([{"id": "2", "translation": "世界"}])
                raise AssertionError(f"unexpected prompt: {prompt}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            outputs: dict[int, tuple[str, str]] = {}

            for concurrency in (1, 2, 4):
                output_dir = root / f"out-{concurrency}"
                config = TranslationConfig(
                    api_key="test-secret",
                    output_dir=str(output_dir),
                    batch_size=1,
                    context_before=0,
                    context_after=0,
                    cache_enabled=False,
                    qa_mode="none",
                    concurrency=concurrency,
                )

                with patch("translation.pipeline.OpenAICompatibleProvider", SlowProvider):
                    run_translation_pipeline(subtitle_path, config)

                outputs[concurrency] = (
                    (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8"),
                    (output_dir / "translation_report.md").read_text(encoding="utf-8"),
                )

        translated_one, report_one = outputs[1]
        translated_two, report_two = outputs[2]
        translated_four, report_four = outputs[4]

        self.assertEqual(translated_one, translated_two)
        self.assertEqual(translated_one, translated_four)
        self.assertLess(
            report_two.find("batch_id: 1 | cue_range: 1-1"),
            report_two.find("batch_id: 2 | cue_range: 2-2"),
        )
        self.assertLess(
            report_four.find("batch_id: 1 | cue_range: 1-1"),
            report_four.find("batch_id: 2 | cue_range: 2-2"),
        )
        self.assertIn("provider_calls: 2", report_one)
        self.assertIn("provider_calls: 2", report_two)
        self.assertIn("provider_calls: 2", report_four)

    def test_concurrent_batches_use_distinct_provider_instances_and_run_once_each(self):
        class IsolatedProvider:
            instances_by_batch: list[tuple[str, int]] = []
            lock = threading.Lock()

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if '"id": "1",\n    "source": "hello"' in prompt:
                    cue_id = "1"
                    translation = "你好"
                elif '"id": "2",\n    "source": "world"' in prompt:
                    cue_id = "2"
                    translation = "世界"
                else:
                    raise AssertionError(f"unexpected prompt: {prompt}")
                with IsolatedProvider.lock:
                    IsolatedProvider.instances_by_batch.append((cue_id, id(self)))
                return json.dumps([{"id": cue_id, "translation": translation}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                cache_enabled=False,
                qa_mode="none",
                concurrency=4,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", IsolatedProvider):
                run_translation_pipeline(subtitle_path, config)

        self.assertEqual(sorted(cue_id for cue_id, _ in IsolatedProvider.instances_by_batch), ["1", "2"])
        self.assertEqual(
            len({instance_id for _, instance_id in IsolatedProvider.instances_by_batch}),
            2,
        )

    def test_concurrent_partial_cache_hit_keeps_provider_calls_and_stats_correct(self):
        from translation.batching import create_batches
        from translation.cache import CacheEntry, TranslationCache, build_batch_cache_key
        from translation.context import build_global_context
        from translation.glossary import load_glossary
        from translation.prompts import PROMPT_VERSION
        from translation.subtitles import parse_subtitle_file

        class PartialCacheProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if '"id": "2",\n    "source": "world"' not in prompt:
                    raise AssertionError("cached batch should not call provider")
                PartialCacheProvider.calls += 1
                return json.dumps([{"id": "2", "translation": "世界"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            glossary_path = _write_glossary(root)
            cache_path = root / "translation-cache.sqlite3"
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                cache_path=str(cache_path),
                glossary_path=str(glossary_path),
                qa_mode="none",
                concurrency=2,
            )
            cues = parse_subtitle_file(subtitle_path)
            glossary = load_glossary(glossary_path)
            global_context = build_global_context(cues, subtitle_path, config)
            cached_batch = create_batches(cues, config.batch_size, config.context_before, config.context_after)[0]
            prompt = build_translation_prompt(
                cached_batch,
                config.target_lang,
                glossary.text,
                global_context.text,
            )
            batch_source_hash = _build_batch_source_hash(prompt)
            cache_key = build_batch_cache_key(
                config.engine_version,
                config.structured_output,
                config.provider,
                config.base_url,
                config.model,
                config.main_model_alias,
                config.target_lang,
                PROMPT_VERSION,
                config.output_schema_version,
                config.batching_strategy_version,
                glossary.hash,
                global_context.hash,
                batch_source_hash,
            )
            with TranslationCache(cache_path) as cache:
                cache.set(
                    CacheEntry(
                        cache_key=cache_key,
                        engine_version=config.engine_version,
                        structured_output=config.structured_output,
                        provider=config.provider,
                        base_url=config.base_url,
                        model=config.model,
                        main_model_alias=config.main_model_alias,
                        target_lang=config.target_lang,
                        prompt_version=PROMPT_VERSION,
                        output_schema_version=config.output_schema_version,
                        batching_strategy_version=config.batching_strategy_version,
                        glossary_hash=glossary.hash,
                        context_hash=global_context.hash,
                        batch_source_hash=batch_source_hash,
                        result_json=json.dumps([{"id": "1", "translation": "缓存你好"}]),
                    )
                )

            with patch("translation.pipeline.OpenAICompatibleProvider", PartialCacheProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(PartialCacheProvider.calls, 1)
        self.assertIn("缓存你好", translated)
        self.assertIn("世界", translated)
        self.assertIn("cache_hits: 1", report)
        self.assertIn("cache_misses: 1", report)
        self.assertIn("provider_calls: 1", report)

    def test_concurrent_cache_access_serializes_cache_critical_sections(self):
        class GuardedCache:
            active_sections = 0
            max_active_sections = 0
            guard_lock = threading.Lock()
            entries: dict[str, str] = {}

            def __init__(self, path):
                self.path = path
                self._enter_section()
                try:
                    time.sleep(0.03)
                finally:
                    self._leave_section()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.close()

            def get(self, cache_key):
                self._enter_section()
                try:
                    return GuardedCache.entries.get(cache_key)
                finally:
                    self._leave_section()

            def set(self, entry):
                self._enter_section()
                try:
                    time.sleep(0.03)
                    GuardedCache.entries[entry.cache_key] = entry.result_json
                finally:
                    self._leave_section()

            def close(self):
                return None

            @classmethod
            def _enter_section(cls):
                with cls.guard_lock:
                    if cls.active_sections != 0:
                        raise AssertionError("cache critical section overlapped")
                    cls.active_sections = 1
                    cls.max_active_sections = max(cls.max_active_sections, cls.active_sections)

            @classmethod
            def _leave_section(cls):
                with cls.guard_lock:
                    cls.active_sections = 0

        class CacheProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if '"id": "1",\n    "source": "hello"' in prompt:
                    return json.dumps([{"id": "1", "translation": "你好"}])
                if '"id": "2",\n    "source": "world"' in prompt:
                    return json.dumps([{"id": "2", "translation": "世界"}])
                raise AssertionError(f"unexpected prompt: {prompt}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                cache_enabled=True,
                cache_path=str(root / "translation-cache.sqlite3"),
                qa_mode="none",
                concurrency=2,
            )

            with patch("translation.pipeline.TranslationCache", GuardedCache), patch(
                "translation.pipeline.OpenAICompatibleProvider",
                CacheProvider,
            ):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")

        self.assertEqual(GuardedCache.max_active_sections, 1)
        self.assertIn("你好", translated)
        self.assertIn("世界", translated)

    def test_concurrency_only_applies_to_batch_translation_and_qa_runs_after_merge(self):
        class QAAfterMergeProvider:
            review_calls = 0
            translate_calls = 0
            lock = threading.Lock()

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if '"id": "1",\n    "source": "hello"' in prompt:
                    time.sleep(0.05)
                    result = json.dumps([{"id": "1", "translation": "As an AI, I cannot help."}])
                elif '"id": "2",\n    "source": "open https://example.test/docs and read it"' in prompt:
                    result = json.dumps([{"id": "2", "translation": "open https://example.test/docs and read it"}])
                else:
                    raise AssertionError(f"unexpected prompt: {prompt}")
                with QAAfterMergeProvider.lock:
                    QAAfterMergeProvider.translate_calls += 1
                return result

            def review_suspicious(self, prompt):
                with QAAfterMergeProvider.lock:
                    if QAAfterMergeProvider.translate_calls != 2:
                        raise AssertionError("QA started before all batch translations completed")
                    QAAfterMergeProvider.review_calls += 1
                return json.dumps(
                    [
                        {"id": "1", "action": "fix", "translation": "你好", "reason": "fixed refusal"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_suspicious_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                cache_enabled=False,
                concurrency=2,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", QAAfterMergeProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(QAAfterMergeProvider.review_calls, 1)
        self.assertIn("你好", translated)
        self.assertIn("qa_provider_calls: 1", report)

    def test_serial_and_concurrent_batch_failure_raise_same_runtime_error_shape(self):
        class BrokenProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return "not json"

        observed: list[tuple[type[BaseException], str, BaseException | None]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_one_cue_srt(Path(temp_dir))

            for concurrency in (1, 2):
                output_dir = Path(temp_dir) / f"out-{concurrency}"
                config = TranslationConfig(
                    api_key="test-secret",
                    output_dir=str(output_dir),
                    batch_size=1,
                    context_before=0,
                    context_after=0,
                    max_retries=0,
                    cache_enabled=False,
                    qa_mode="none",
                    concurrency=concurrency,
                )

                with patch("translation.pipeline.OpenAICompatibleProvider", BrokenProvider):
                    with self.assertRaisesRegex(RuntimeError, r"(?s)batch_id 1.*1 attempts.*not valid JSON") as raised:
                        run_translation_pipeline(subtitle_path, config)

                observed.append((type(raised.exception), str(raised.exception), raised.exception.__cause__))

        self.assertEqual([error_type for error_type, _, _ in observed], [RuntimeError, RuntimeError])
        self.assertEqual(observed[0][1], observed[1][1])
        self.assertIsInstance(observed[0][2], RuntimeError)
        self.assertIsInstance(observed[1][2], RuntimeError)
        self.assertEqual(str(observed[0][2]), str(observed[1][2]))

    def test_batch_stats_delta_no_longer_tracks_failed_batches(self):
        self.assertNotIn("failed_batches", BatchStatsDelta.__dataclass_fields__)

    def test_concurrent_batch_failure_keeps_old_pipeline_failure_behavior(self):
        class MixedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if '"id": "1",\n    "source": "hello"' in prompt:
                    time.sleep(0.05)
                    return json.dumps([{"id": "1", "translation": "你好"}])
                if '"id": "2",\n    "source": "world"' in prompt:
                    return "not json"
                raise AssertionError(f"unexpected prompt: {prompt}")

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                max_retries=0,
                cache_enabled=False,
                qa_mode="none",
                concurrency=2,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", MixedProvider):
                with self.assertRaisesRegex(RuntimeError, r"(?s)batch_id 2.*1 attempts.*not valid JSON"):
                    run_translation_pipeline(subtitle_path, config)

            self.assertFalse((output_dir / "translated.zh-CN.srt").exists())
            self.assertFalse((output_dir / "bilingual.srt").exists())
            self.assertFalse((output_dir / "global_context.md").exists())
            self.assertFalse((output_dir / "translation_report.md").exists())

    def test_concurrent_batch_failure_does_not_wait_for_earlier_slow_batch(self):
        class EarlyFailProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if '"id": "1",\n    "source": "hello"' in prompt:
                    time.sleep(0.35)
                    return json.dumps([{"id": "1", "translation": "你好"}])
                if '"id": "2",\n    "source": "world"' in prompt:
                    raise RuntimeError("provider boom")
                raise AssertionError(f"unexpected prompt: {prompt}")

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                max_retries=0,
                cache_enabled=False,
                qa_mode="none",
                concurrency=2,
            )

            started_at = time.perf_counter()
            with patch("translation.pipeline.OpenAICompatibleProvider", EarlyFailProvider):
                with self.assertRaisesRegex(RuntimeError, r"batch_id 2 failed after 1 attempts: provider boom"):
                    run_translation_pipeline(subtitle_path, config)
            elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.2)
        self.assertFalse((output_dir / "translated.zh-CN.srt").exists())
        self.assertFalse((output_dir / "bilingual.srt").exists())
        self.assertFalse((output_dir / "global_context.md").exists())
        self.assertFalse((output_dir / "translation_report.md").exists())

    def test_non_runtime_worker_error_also_fails_fast_without_waiting_for_slow_batch(self):
        class SlowStructuredProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                time.sleep(0.35)
                return json.dumps([{"cue_id": "valid-1", "translation": "你好"}])

        stable_cue = Cue(id="2", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        slow_target_cue = Cue(id="valid-1", index=10, start="00:00:02,000", end="00:00:03,000", source="hello")
        colliding_target_cue = Cue(id="same", index=2, start="00:00:04,000", end="00:00:05,000", source="world")
        colliding_after_cue = Cue(id="same", index=3, start="00:00:06,000", end="00:00:07,000", source="after")
        valid_batch = TranslationBatch(
            batch_id=1,
            cues=(slow_target_cue,),
            context_before=(),
            context_after=(),
        )
        invalid_batch = TranslationBatch(
            batch_id=2,
            cues=(colliding_target_cue,),
            context_before=(stable_cue,),
            context_after=(colliding_after_cue,),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = _write_two_cue_srt(Path(temp_dir))
            output_dir = Path(temp_dir) / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                cache_enabled=False,
                qa_mode="none",
                concurrency=2,
                engine_version="v2",
                structured_output=True,
            )

            started_at = time.perf_counter()
            with patch("translation.pipeline.parse_subtitle_file", return_value=[slow_target_cue, colliding_target_cue]), patch(
                "translation.pipeline.create_batches",
                return_value=(valid_batch, invalid_batch),
            ), patch("translation.pipeline.OpenAICompatibleProvider", SlowStructuredProvider):
                with self.assertRaisesRegex(ValueError, "generated non-unique structured cue_ids"):
                    run_translation_pipeline(subtitle_path, config)
            elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.2)
        self.assertFalse((output_dir / "translated.zh-CN.srt").exists())
        self.assertFalse((output_dir / "bilingual.srt").exists())

    def test_concurrent_batches_cap_executor_workers_to_batch_count(self):
        batch_one = TranslationBatch(cues=(ONE_CUE[0],), context_before=(), context_after=(), batch_id=1)
        batch_two = TranslationBatch(
            cues=(Cue(id="2", index=2, start="00:00:01,000", end="00:00:02,000", source="world"),),
            context_before=(),
            context_after=(),
            batch_id=2,
        )
        captured_max_workers: list[int] = []

        class CapturingExecutor:
            def __init__(self, *, max_workers):
                captured_max_workers.append(max_workers)

            def submit(self, fn, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as error:
                    future.set_exception(error)
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        def fake_result(batch, *_args):
            return BatchExecutionResult(
                translations=((batch.cues[0].id, "ok"),),
                stats_delta=BatchStatsDelta(),
                batch_entry=MinimalBatchReportEntry(
                    batch_id=batch.batch_id,
                    state=BatchState.SUCCESS,
                    cue_count=1,
                    attempts=1,
                    cache_hit=False,
                    cue_range=(batch.cues[0].index, batch.cues[0].index),
                    attempt=1,
                    error_type=None,
                    duration_ms=1,
                ),
            )

        config = TranslationConfig(api_key="test-secret", cache_enabled=False, qa_mode="none", concurrency=8)

        with patch("translation.pipeline.ThreadPoolExecutor", CapturingExecutor), patch(
            "translation.pipeline._execute_translation_batch",
            side_effect=fake_result,
        ):
            results = _run_translation_batches((batch_one, batch_two), config, "", "", "", "")

        self.assertEqual(captured_max_workers, [2])
        self.assertEqual(len(results), 2)

    def test_adaptive_concurrency_disabled_keeps_fixed_concurrency_level(self):
        class TrackingProvider:
            active = 0
            max_active = 0
            lock = threading.Lock()

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                with TrackingProvider.lock:
                    TrackingProvider.active += 1
                    TrackingProvider.max_active = max(TrackingProvider.max_active, TrackingProvider.active)
                try:
                    time.sleep(0.05)
                    current_cues = prompt.split("Current cues to translate:", 1)[1].split("After context:", 1)[0]
                    cue_id = "1" if '"id": "1"' in current_cues else "2" if '"id": "2"' in current_cues else "3" if '"id": "3"' in current_cues else "4"
                    return json.dumps([{"id": cue_id, "translation": f"ok-{cue_id}"}])
                finally:
                    with TrackingProvider.lock:
                        TrackingProvider.active -= 1

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_four_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                cache_enabled=False,
                qa_mode="none",
                concurrency=3,
                adaptive_concurrency_enabled=False,
                adaptive_concurrency_min=1,
                adaptive_concurrency_max=1,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", TrackingProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(TrackingProvider.max_active, 3)
        self.assertIn("adaptive_concurrency_enabled: False", report)

    def test_adaptive_concurrency_caps_executor_workers_to_effective_max(self):
        batch_one = TranslationBatch(cues=(ONE_CUE[0],), context_before=(), context_after=(), batch_id=1)
        batch_two = TranslationBatch(
            cues=(Cue(id="2", index=2, start="00:00:01,000", end="00:00:02,000", source="world"),),
            context_before=(),
            context_after=(),
            batch_id=2,
        )
        batch_three = TranslationBatch(
            cues=(Cue(id="3", index=3, start="00:00:02,000", end="00:00:03,000", source="again"),),
            context_before=(),
            context_after=(),
            batch_id=3,
        )
        captured_max_workers: list[int] = []

        class CapturingExecutor:
            def __init__(self, *, max_workers):
                captured_max_workers.append(max_workers)

            def submit(self, fn, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as error:
                    future.set_exception(error)
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        def fake_result(batch, *_args):
            return BatchExecutionResult(
                translations=((batch.cues[0].id, "ok"),),
                stats_delta=BatchStatsDelta(),
                batch_entry=MinimalBatchReportEntry(
                    batch_id=batch.batch_id,
                    state=BatchState.SUCCESS,
                    cue_count=1,
                    attempts=1,
                    cache_hit=False,
                    cue_range=(batch.cues[0].index, batch.cues[0].index),
                    attempt=1,
                    error_type=None,
                    duration_ms=1,
                ),
            )

        config = TranslationConfig(
            api_key="test-secret",
            cache_enabled=False,
            qa_mode="none",
            concurrency=8,
            adaptive_concurrency_enabled=True,
            adaptive_concurrency_min=1,
            adaptive_concurrency_max=2,
        )

        with patch("translation.pipeline.ThreadPoolExecutor", CapturingExecutor), patch(
            "translation.pipeline._execute_translation_batch",
            side_effect=fake_result,
        ):
            results = _run_translation_batches((batch_one, batch_two, batch_three), config, "", "", "", "")

        self.assertEqual(captured_max_workers, [2])
        self.assertEqual(len(results), 3)

    def test_adaptive_concurrency_reduces_on_provider_pressure_signals(self):
        batches = tuple(
            TranslationBatch(
                cues=(Cue(id=str(index), index=index, start="00:00:00,000", end="00:00:01,000", source=f"cue-{index}"),),
                context_before=(),
                context_after=(),
                batch_id=index,
            )
            for index in range(1, 5)
        )
        late_phase_active = 0
        late_phase_max_active = 0
        lock = threading.Lock()

        def fake_result(batch, *_args):
            nonlocal late_phase_active, late_phase_max_active
            if batch.batch_id >= 3:
                with lock:
                    late_phase_active += 1
                    late_phase_max_active = max(late_phase_max_active, late_phase_active)
            try:
                time.sleep(0.05)
                error_type = ErrorType.PROVIDER_TIMEOUT if batch.batch_id <= 2 else None
                return BatchExecutionResult(
                    translations=((batch.cues[0].id, f"ok-{batch.batch_id}"),),
                    stats_delta=BatchStatsDelta(),
                    batch_entry=MinimalBatchReportEntry(
                        batch_id=batch.batch_id,
                        state=BatchState.SUCCESS,
                        cue_count=1,
                        attempts=1,
                        cache_hit=False,
                        cue_range=(batch.cues[0].index, batch.cues[0].index),
                        attempt=1,
                        error_type=error_type,
                        duration_ms=1,
                        final_route_label="fallback" if error_type else "main",
                    ),
                )
            finally:
                if batch.batch_id >= 3:
                    with lock:
                        late_phase_active -= 1

        config = TranslationConfig(
            api_key="test-secret",
            cache_enabled=False,
            qa_mode="none",
            concurrency=2,
            adaptive_concurrency_enabled=True,
            adaptive_concurrency_min=1,
            adaptive_concurrency_max=2,
        )

        with patch("translation.pipeline._execute_translation_batch", side_effect=fake_result):
            results = _run_translation_batches(batches, config, "", "", "", "")

        self.assertEqual(len(results), 4)
        self.assertEqual(late_phase_max_active, 1)

    def test_adaptive_concurrency_recovers_gradually_and_respects_maximum(self):
        batches = tuple(
            TranslationBatch(
                cues=(Cue(id=str(index), index=index, start="00:00:00,000", end="00:00:01,000", source=f"cue-{index}"),),
                context_before=(),
                context_after=(),
                batch_id=index,
            )
            for index in range(1, 9)
        )
        mid_phase_active = 0
        mid_phase_max_active = 0
        late_phase_active = 0
        late_phase_max_active = 0
        lock = threading.Lock()

        def fake_result(batch, *_args):
            nonlocal mid_phase_active, mid_phase_max_active, late_phase_active, late_phase_max_active
            if 3 <= batch.batch_id <= 4:
                with lock:
                    mid_phase_active += 1
                    mid_phase_max_active = max(mid_phase_max_active, mid_phase_active)
            if batch.batch_id >= 7:
                with lock:
                    late_phase_active += 1
                    late_phase_max_active = max(late_phase_max_active, late_phase_active)
            try:
                time.sleep(0.05)
                error_type = ErrorType.PROVIDER_TIMEOUT if batch.batch_id <= 2 else None
                return BatchExecutionResult(
                    translations=((batch.cues[0].id, f"ok-{batch.batch_id}"),),
                    stats_delta=BatchStatsDelta(),
                    batch_entry=MinimalBatchReportEntry(
                        batch_id=batch.batch_id,
                        state=BatchState.SUCCESS,
                        cue_count=1,
                        attempts=1,
                        cache_hit=False,
                        cue_range=(batch.cues[0].index, batch.cues[0].index),
                        attempt=1,
                        error_type=error_type,
                        duration_ms=1,
                        final_route_label="fallback" if error_type else "main",
                    ),
                )
            finally:
                if 3 <= batch.batch_id <= 4:
                    with lock:
                        mid_phase_active -= 1
                if batch.batch_id >= 7:
                    with lock:
                        late_phase_active -= 1

        config = TranslationConfig(
            api_key="test-secret",
            cache_enabled=False,
            qa_mode="none",
            concurrency=2,
            adaptive_concurrency_enabled=True,
            adaptive_concurrency_min=1,
            adaptive_concurrency_max=2,
        )

        with patch("translation.pipeline._execute_translation_batch", side_effect=fake_result):
            results = _run_translation_batches(batches, config, "", "", "", "")

        self.assertEqual(len(results), 8)
        self.assertEqual(mid_phase_max_active, 1)
        self.assertEqual(late_phase_max_active, 2)

    def test_structured_cache_hit_batch_report_marks_cache_hit_true_and_attempt_zero(self):
        class SeedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"cue_id": "1", "translation": "初始"}])

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
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )
            second_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out2"),
                batch_size=1,
                cache_path=str(cache_path),
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", SeedProvider):
                run_translation_pipeline(subtitle_path, first_config)
            with patch("translation.pipeline.OpenAICompatibleProvider", FailingProvider):
                run_translation_pipeline(subtitle_path, second_config)

            report = (root / "out2" / "translation_report.md").read_text(encoding="utf-8")

        self.assertIn("## Batch Results", report)
        self.assertIn("batch_id: 1 | cue_range: 1-1 | status: success | attempt: 0 | error_type: none | cache_hit: True", report)
        self.assertRegex(report, r"duration_ms: \d+")

    def test_v2_structured_cache_does_not_reuse_v1_entry(self):
        class SeedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "初始v1"}])

        class StructuredProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                StructuredProvider.calls += 1
                return json.dumps([{"cue_id": "1", "translation": "结构化v2"}])

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
            v2_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out"),
                batch_size=1,
                cache_path=str(cache_path),
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", SeedProvider):
                run_translation_pipeline(subtitle_path, seed_config)
            with patch("translation.pipeline.OpenAICompatibleProvider", StructuredProvider):
                run_translation_pipeline(subtitle_path, v2_config)

            report = (root / "out" / "translation_report.md").read_text(encoding="utf-8")
            translated = (root / "out" / "translated.zh-CN.srt").read_text(encoding="utf-8")

        self.assertEqual(StructuredProvider.calls, 1)
        self.assertIn("结构化v2", translated)
        self.assertIn("cache_hits: 0", report)
        self.assertIn("cache_misses: 1", report)
        self.assertIn("provider_calls: 1", report)

    def test_structured_output_toggle_does_not_reuse_cache_entry(self):
        class SeedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "非结构化v2"}])

        class StructuredProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                StructuredProvider.calls += 1
                return json.dumps([{"cue_id": "1", "translation": "结构化v2"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            cache_path = root / "translation-cache.sqlite3"
            seed_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "seed"),
                batch_size=1,
                cache_path=str(cache_path),
                qa_mode="none",
                engine_version="v2",
                structured_output=False,
            )
            structured_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out"),
                batch_size=1,
                cache_path=str(cache_path),
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", SeedProvider):
                run_translation_pipeline(subtitle_path, seed_config)
            with patch("translation.pipeline.OpenAICompatibleProvider", StructuredProvider):
                run_translation_pipeline(subtitle_path, structured_config)

            report = (root / "out" / "translation_report.md").read_text(encoding="utf-8")
            translated = (root / "out" / "translated.zh-CN.srt").read_text(encoding="utf-8")

        self.assertEqual(StructuredProvider.calls, 1)
        self.assertIn("结构化v2", translated)
        self.assertIn("cache_hits: 0", report)
        self.assertIn("cache_misses: 1", report)
        self.assertIn("provider_calls: 1", report)

    def test_base_url_change_does_not_reuse_cache_entry(self):
        class SeedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "主网关"}])

        class AlternateProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                AlternateProvider.calls += 1
                return json.dumps([{"id": "1", "translation": "次网关"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            cache_path = root / "translation-cache.sqlite3"
            seed_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "seed"),
                batch_size=1,
                cache_path=str(cache_path),
                base_url="https://gateway-a.example.test/v1",
            )
            second_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out"),
                batch_size=1,
                cache_path=str(cache_path),
                base_url="https://gateway-b.example.test/v1",
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", SeedProvider):
                run_translation_pipeline(subtitle_path, seed_config)
            with patch("translation.pipeline.OpenAICompatibleProvider", AlternateProvider):
                run_translation_pipeline(subtitle_path, second_config)

            report = (root / "out" / "translation_report.md").read_text(encoding="utf-8")
            translated = (root / "out" / "translated.zh-CN.srt").read_text(encoding="utf-8")

        self.assertEqual(AlternateProvider.calls, 1)
        self.assertIn("次网关", translated)
        self.assertIn("cache_hits: 0", report)
        self.assertIn("cache_misses: 1", report)
        self.assertIn("provider_calls: 1", report)

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

    def test_build_structured_batch_record_reuses_stable_cue_ids_across_target_and_context(self):
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        before_cue = Cue(id="before-1", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        after_cue = Cue(id="after-3", index=3, start="00:00:04,000", end="00:00:05,000", source="after")
        batch = TranslationBatch(
            batch_id=7,
            cues=(target_cue,),
            context_before=(before_cue,),
            context_after=(after_cue,),
        )

        record = _build_structured_batch_record(batch)

        self.assertEqual(tuple(cue.cue_id for cue in record.target_cues), ("target-2",))
        self.assertEqual(tuple(cue.cue_id for cue in record.context_before), ("before-1",))
        self.assertEqual(tuple(cue.cue_id for cue in record.context_after), ("after-3",))
        self.assertEqual(record.target_cues[0].original_index, 2)
        self.assertEqual(record.context_before[0].original_index, 1)
        self.assertEqual(record.context_after[0].original_index, 3)

    def test_build_structured_batch_record_falls_back_to_original_index_for_unstable_ids(self):
        before_cue = Cue(id="same", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        target_cue = Cue(id="same", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        batch = TranslationBatch(
            batch_id=8,
            cues=(target_cue,),
            context_before=(before_cue,),
            context_after=(),
        )

        record = _build_structured_batch_record(batch)

        self.assertEqual(tuple(cue.cue_id for cue in record.context_before), ("1",))
        self.assertEqual(tuple(cue.cue_id for cue in record.target_cues), ("2",))

    def test_classify_structured_cue_id_distinguishes_context_from_unknown(self):
        before_cue = Cue(id="before-1", index=1, start="00:00:00,000", end="00:00:01,000", source="before")
        target_cue = Cue(id="target-2", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        batch = TranslationBatch(
            batch_id=9,
            cues=(target_cue,),
            context_before=(before_cue,),
            context_after=(),
        )

        record = _build_structured_batch_record(batch)

        self.assertIsNone(_classify_structured_cue_id(record, "target-2"))
        self.assertEqual(
            _classify_structured_cue_id(record, "before-1"),
            ErrorType.CONTEXT_CUE_OUTPUT_VIOLATION,
        )
        self.assertEqual(_classify_structured_cue_id(record, "missing-3"), ErrorType.INVALID_CUE_ID)

    def test_build_structured_batch_record_rejects_generated_cue_id_collision(self):
        stable_cue = Cue(id="2", index=1, start="00:00:00,000", end="00:00:01,000", source="stable")
        target_cue = Cue(id="same", index=2, start="00:00:02,000", end="00:00:03,000", source="target")
        after_cue = Cue(id="same", index=3, start="00:00:04,000", end="00:00:05,000", source="after")
        batch = TranslationBatch(
            batch_id=10,
            cues=(target_cue,),
            context_before=(stable_cue,),
            context_after=(after_cue,),
        )

        with self.assertRaisesRegex(ValueError, "generated non-unique structured cue_ids"):
            _build_structured_batch_record(batch)

    def test_default_v1_path_keeps_current_prompt_and_parser_behavior(self):
        prompt = self._capture_translation_prompt(TranslationConfig(api_key="test-secret", cache_enabled=False))

        self.assertIn("Each item must have id and translation fields.", prompt)
        self.assertIn('[{"id": "1", "translation": "..."}]', prompt)
        self.assertNotIn("cue_id", prompt)

    def test_v2_false_keeps_current_prompt_and_parser_behavior(self):
        prompt = self._capture_translation_prompt(
            TranslationConfig(
                api_key="test-secret",
                cache_enabled=False,
                engine_version="v2",
                structured_output=False,
            )
        )

        self.assertIn("Each item must have id and translation fields.", prompt)
        self.assertIn('[{"id": "1", "translation": "..."}]', prompt)
        self.assertNotIn("cue_id", prompt)

    def test_structured_prompt_activates_only_on_v2_true(self):
        prompt = self._capture_translation_prompt(
            TranslationConfig(
                api_key="test-secret",
                cache_enabled=False,
                engine_version="v2",
                structured_output=True,
            )
        )

        self.assertIn("Each item must have cue_id and translation fields.", prompt)
        self.assertIn('[{"cue_id": "1", "translation": "..."}]', prompt)
        self.assertIn('Current cues to translate:\n[\n  {\n    "cue_id": "1",', prompt)
        self.assertNotIn("Each item must have id and translation fields.", prompt)

    def test_v2_true_accepts_structured_cue_id_response(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"cue_id": "1", "translation": "你好"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")

        self.assertIn("你好", translated)

    def test_v2_true_reconciles_fallback_cue_id_before_final_validation(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"cue_id": "1", "translation": "你好"}])

        target_cue = Cue(id="shared-id", index=1, start="00:00:02,000", end="00:00:03,000", source="target")
        context_cue = Cue(id="shared-id", index=99, start="00:00:00,000", end="00:00:01,000", source="context")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.parse_subtitle_file", return_value=[target_cue]), patch(
                "translation.pipeline.create_batches",
                return_value=(
                    TranslationBatch(
                        batch_id=1,
                        cues=(target_cue,),
                        context_before=(context_cue,),
                        context_after=(),
                    ),
                ),
            ), patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")

        self.assertIn("你好", translated)

    def test_structured_prompt_uses_fallback_cue_ids_when_original_ids_are_unstable(self):
        captured_prompts: list[str] = []

        class FakeProvider:
            def __init__(self, config):
                self.config = config
                self.prompts: list[str] = []

            def translate_batch(self, prompt):
                self.prompts.append(prompt)
                captured_prompts.append(prompt)
                return json.dumps([{"cue_id": "1", "translation": "你好"}])

        target_cue = Cue(id="shared-id", index=1, start="00:00:02,000", end="00:00:03,000", source="target")
        context_cue = Cue(id="shared-id", index=99, start="00:00:00,000", end="00:00:01,000", source="context")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.parse_subtitle_file", return_value=[target_cue]), patch(
                "translation.pipeline.create_batches",
                return_value=(
                    TranslationBatch(
                        batch_id=1,
                        cues=(target_cue,),
                        context_before=(context_cue,),
                        context_after=(),
                    ),
                ),
            ), patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

        self.assertIn('Current cues to translate:\n[\n  {\n    "cue_id": "1"', captured_prompts[0])
        self.assertIn('Before context:\n[\n  {\n    "cue_id": "99"', captured_prompts[0])
        self.assertNotIn('"cue_id": "shared-id"', captured_prompts[0])

    def _capture_translation_prompt(self, config):
        captured_prompts: list[str] = []

        class FakeProvider:
            def __init__(self, provider_config):
                self.config = provider_config
                self.prompts: list[str] = []

            def translate_batch(self, prompt):
                self.prompts.append(prompt)
                captured_prompts.append(prompt)
                if self.config.engine_version == "v2" and self.config.structured_output:
                    return json.dumps([{"cue_id": "1", "translation": "你好"}])
                return json.dumps([{"id": "1", "translation": "你好"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config_values = {**config.__dict__, "output_dir": str(output_dir), "batch_size": 1}
            config = TranslationConfig(**config_values)

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

        return captured_prompts[0]

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

    def test_invalid_structured_cache_entry_falls_back_to_provider_and_overwrites_cache(self):
        class SeedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"cue_id": "1", "translation": "初始"}])

        class FakeProvider:
            calls = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls += 1
                return json.dumps([{"cue_id": "1", "translation": "修复后"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            cache_path = root / "translation-cache.sqlite3"
            seed_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "seed"),
                batch_size=1,
                cache_path=str(cache_path),
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )
            recovery_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "out"),
                batch_size=1,
                cache_path=str(cache_path),
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", SeedProvider):
                run_translation_pipeline(subtitle_path, seed_config)
            _overwrite_only_cache_result(cache_path, json.dumps([{"cue_id": "missing", "translation": "坏缓存"}]))

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, recovery_config)

            report = (root / "out" / "translation_report.md").read_text(encoding="utf-8")
            cache_rows = _read_cache_rows(cache_path)

        self.assertEqual(FakeProvider.calls, 1)
        self.assertIn("cache_hits: 0", report)
        self.assertIn("cache_misses: 1", report)
        self.assertIn("provider_calls: 1", report)
        self.assertEqual(json.loads(cache_rows[0]), [{"cue_id": "1", "translation": "修复后"}])
        self.assertNotIn("坏缓存", cache_rows[0])

    def test_invalid_structured_provider_response_does_not_write_cache_entry(self):
        class BrokenProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"cue_id": "missing", "translation": "坏响应"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            cache_path = root / "translation-cache.sqlite3"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                max_retries=0,
                cache_path=str(cache_path),
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", BrokenProvider):
                with self.assertRaisesRegex(RuntimeError, r"batch_id 1 failed after 1 attempts.*unexpected cue_id missing"):
                    run_translation_pipeline(subtitle_path, config)

            cache_rows = _read_cache_rows(cache_path)

        self.assertEqual(cache_rows, [])

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

    def test_provider_transport_errors_retry_then_fallback_without_cache_write(self):
        error_types = (
            ErrorType.PROVIDER_TIMEOUT,
            ErrorType.PROVIDER_HTTP_5XX,
            ErrorType.PROVIDER_REQUEST_FAILED,
            ErrorType.PROVIDER_MISSING_CHOICES,
        )

        for error_type in error_types:
            with self.subTest(error_type=error_type.value):
                class FakeProvider:
                    calls_by_model: list[str] = []

                    def __init__(self, config):
                        self.config = config
                        self.error_type = error_type

                    def translate_batch(self, prompt):
                        FakeProvider.calls_by_model.append(self.config.model)
                        if self.config.model == "main-model":
                            raise ProviderError(self.error_type, f"main failed: {self.error_type.value}")
                        return json.dumps(
                            [
                                {"id": "1", "translation": "你好"},
                                {"id": "2", "translation": "世界"},
                            ]
                        )

                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    subtitle_path = _write_two_cue_srt(root)
                    output_dir = root / "out"
                    cache_path = root / "translation-cache.sqlite3"
                    config = TranslationConfig(
                        api_key="test-secret",
                        model="main-model",
                        fallback_model="fallback-model",
                        output_dir=str(output_dir),
                        batch_size=2,
                        max_retries=1,
                        cache_path=str(cache_path),
                        qa_mode="none",
                    )

                    with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                        run_translation_pipeline(subtitle_path, config)

                    report = (output_dir / "translation_report.md").read_text(encoding="utf-8")
                    translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
                    cache_rows = _read_cache_rows(cache_path) if cache_path.exists() else []

                self.assertEqual(FakeProvider.calls_by_model, ["main-model", "main-model", "fallback-model"])
                self.assertIn("你好", translated)
                self.assertIn("世界", translated)
                self.assertIn("provider_calls: 3", report)
                self.assertIn("fallback_provider_calls: 1", report)
                self.assertIn("final_route_label: fallback", report)
                self.assertEqual(cache_rows, [])

    def test_invalid_json_retries_then_falls_back_without_cache_write(self):
        class FakeProvider:
            calls_by_model: list[str] = []

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls_by_model.append(self.config.model)
                if self.config.model == "main-model":
                    return "not json"
                return json.dumps(
                    [
                        {"id": "1", "translation": "你好"},
                        {"id": "2", "translation": "世界"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            cache_path = root / "translation-cache.sqlite3"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                fallback_model="fallback-model",
                output_dir=str(output_dir),
                batch_size=2,
                max_retries=1,
                cache_path=str(cache_path),
                qa_mode="none",
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")
            cache_rows = _read_cache_rows(cache_path) if cache_path.exists() else []

        self.assertEqual(FakeProvider.calls_by_model, ["main-model", "main-model", "fallback-model"])
        self.assertIn("provider_calls: 3", report)
        self.assertIn("fallback_provider_calls: 1", report)
        self.assertIn("final_route_label: fallback", report)
        self.assertEqual(cache_rows, [])

    def test_schema_mismatch_retries_on_main_and_does_not_fallback(self):
        class FakeProvider:
            calls_by_model: list[str] = []

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls_by_model.append(self.config.model)
                return json.dumps({"cue_id": "1", "translation": "你好"})

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                fallback_model="fallback-model",
                output_dir=str(output_dir),
                batch_size=1,
                max_retries=1,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                with self.assertRaisesRegex(RuntimeError, r"batch_id 1.*2 attempts"):
                    run_translation_pipeline(subtitle_path, config)

        self.assertEqual(FakeProvider.calls_by_model, ["main-model", "main-model"])

    def test_fallback_failure_still_fails_pipeline(self):
        class FakeProvider:
            calls_by_model: list[str] = []

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FakeProvider.calls_by_model.append(self.config.model)
                if self.config.model == "main-model":
                    raise ProviderError(ErrorType.PROVIDER_TIMEOUT, "main timed out")
                raise ProviderError(ErrorType.PROVIDER_HTTP_5XX, "fallback failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            cache_path = root / "translation-cache.sqlite3"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                fallback_model="fallback-model",
                output_dir=str(output_dir),
                batch_size=1,
                max_retries=0,
                cache_path=str(cache_path),
                qa_mode="none",
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                with self.assertRaisesRegex(RuntimeError, r"batch_id 1.*fallback failed"):
                    run_translation_pipeline(subtitle_path, config)

            cache_rows = _read_cache_rows(cache_path) if cache_path.exists() else []

        self.assertEqual(FakeProvider.calls_by_model, ["main-model", "fallback-model"])
        self.assertEqual(cache_rows, [])
        self.assertFalse((output_dir / "translated.zh-CN.srt").exists())

    def test_concurrency_two_keeps_output_order_and_fallback_stats_stable(self):
        class FakeProvider:
            calls_by_model: list[tuple[str, str]] = []
            lock = threading.Lock()

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                current_cues = prompt.split("Current cues to translate:", 1)[1].split("After context:", 1)[0]
                cue_id = "1" if '"id": "1"' in current_cues else "2"
                with FakeProvider.lock:
                    FakeProvider.calls_by_model.append((self.config.model, cue_id))
                if self.config.model == "main-model":
                    raise ProviderError(ErrorType.PROVIDER_REQUEST_FAILED, f"main failed for {cue_id}")
                if cue_id == "1":
                    time.sleep(0.05)
                    return json.dumps([{"id": "1", "translation": "你好"}])
                return json.dumps([{"id": "2", "translation": "世界"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                fallback_model="fallback-model",
                output_dir=str(output_dir),
                batch_size=1,
                context_before=0,
                context_after=0,
                max_retries=0,
                cache_enabled=False,
                qa_mode="none",
                concurrency=2,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertIn("你好", translated)
        self.assertIn("世界", translated)
        self.assertLess(translated.find("你好"), translated.find("世界"))
        self.assertIn("provider_calls: 4", report)
        self.assertIn("fallback_provider_calls: 2", report)
        self.assertLess(
            report.find("batch_id: 1 | cue_range: 1-1"),
            report.find("batch_id: 2 | cue_range: 2-2"),
        )
        self.assertIn("final_route_label: fallback", report)

    def test_structure_error_can_shrink_after_main_and_fallback_exhaust(self):
        class ShrinkThenSucceedProvider:
            calls_by_model: list[str] = []

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                ShrinkThenSucceedProvider.calls_by_model.append(self.config.model)
                if len(ShrinkThenSucceedProvider.calls_by_model) <= 2:
                    return "not json"
                if '"cue_id": "1"' in prompt and '"cue_id": "2"' in prompt:
                    return json.dumps(
                        [
                            {"cue_id": "1", "translation": "一"},
                            {"cue_id": "2", "translation": "二"},
                        ]
                    )
                return json.dumps(
                    [
                        {"cue_id": "3", "translation": "三"},
                        {"cue_id": "4", "translation": "四"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_four_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                fallback_model="fallback-model",
                output_dir=str(output_dir),
                batch_size=4,
                context_before=0,
                context_after=0,
                max_retries=0,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", ShrinkThenSucceedProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(ShrinkThenSucceedProvider.calls_by_model, ["main-model", "fallback-model", "main-model", "main-model"])
        self.assertIn("一", translated)
        self.assertIn("二", translated)
        self.assertIn("三", translated)
        self.assertIn("四", translated)
        self.assertIn("provider_calls: 4", report)
        self.assertIn("fallback_provider_calls: 1", report)

    def test_provider_timeout_does_not_trigger_shrink_batch(self):
        class TimeoutOnlyProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                raise ProviderError(ErrorType.PROVIDER_TIMEOUT, "timed out")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                fallback_model="fallback-model",
                output_dir=str(output_dir),
                batch_size=2,
                context_before=0,
                context_after=0,
                max_retries=0,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.split_batch") as split_batch_mock, patch(
                "translation.pipeline.OpenAICompatibleProvider", TimeoutOnlyProvider
            ):
                with self.assertRaisesRegex(RuntimeError, r"batch_id 1.*timed out"):
                    run_translation_pipeline(subtitle_path, config)

        self.assertEqual(split_batch_mock.call_count, 0)

    def test_run_translation_batches_merges_shrunk_child_results_in_parent_order(self):
        class ShrinkThenSucceedProvider:
            calls_by_model: list[str] = []

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                ShrinkThenSucceedProvider.calls_by_model.append(self.config.model)
                if len(ShrinkThenSucceedProvider.calls_by_model) <= 2:
                    return "not json"
                if '"cue_id": "1"' in prompt and '"cue_id": "2"' in prompt:
                    return json.dumps(
                        [
                            {"cue_id": "1", "translation": "一"},
                            {"cue_id": "2", "translation": "二"},
                        ]
                    )
                return json.dumps(
                    [
                        {"cue_id": "3", "translation": "三"},
                        {"cue_id": "4", "translation": "四"},
                    ]
                )

        batch = TranslationBatch(
            batch_id=1,
            cues=(
                Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="one"),
                Cue(id="2", index=2, start="00:00:01,000", end="00:00:02,000", source="two"),
                Cue(id="3", index=3, start="00:00:02,000", end="00:00:03,000", source="three"),
                Cue(id="4", index=4, start="00:00:03,000", end="00:00:04,000", source="four"),
            ),
            context_before=(),
            context_after=(),
        )
        config = TranslationConfig(
            api_key="test-secret",
            model="main-model",
            fallback_model="fallback-model",
            batch_size=4,
            context_before=0,
            context_after=0,
            max_retries=0,
            cache_enabled=False,
            qa_mode="none",
            engine_version="v2",
            structured_output=True,
        )

        with patch("translation.pipeline.OpenAICompatibleProvider", ShrinkThenSucceedProvider):
            batch_results = _run_translation_batches((batch,), config, "", "", "", "")

        self.assertEqual(batch_results[0].translations, (("1", "一"), ("2", "二"), ("3", "三"), ("4", "四")))

    def test_shrink_batch_stops_after_max_split_attempts(self):
        from translation.batching import split_batch as real_split_batch

        class AlwaysInvalidProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return "not json"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_eight_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                output_dir=str(output_dir),
                batch_size=8,
                context_before=0,
                context_after=0,
                max_retries=0,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", AlwaysInvalidProvider), patch(
                "translation.pipeline.split_batch", wraps=real_split_batch
            ) as split_batch_mock:
                with self.assertRaisesRegex(RuntimeError, "not valid JSON"):
                    run_translation_pipeline(subtitle_path, config)

        self.assertEqual([call.args[0].batch_id for call in split_batch_mock.call_args_list], [1, 2])

    def test_concurrency_two_keeps_root_batch_order_when_first_batch_shrinks(self):
        class ConcurrentShrinkProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                current_cues = prompt.split("Current cues to translate:\n", 1)[1].split("\nAfter context:", 1)[0]
                if '"cue_id": "5"' in current_cues and '"cue_id": "8"' in current_cues:
                    return json.dumps(
                        [
                            {"cue_id": "5", "translation": "五"},
                            {"cue_id": "6", "translation": "六"},
                            {"cue_id": "7", "translation": "七"},
                            {"cue_id": "8", "translation": "八"},
                        ]
                    )
                if '"cue_id": "1"' in current_cues and '"cue_id": "4"' in current_cues:
                    time.sleep(0.05)
                    return "not json"
                if '"cue_id": "1"' in current_cues and '"cue_id": "2"' in current_cues:
                    time.sleep(0.05)
                    return json.dumps(
                        [
                            {"cue_id": "1", "translation": "一"},
                            {"cue_id": "2", "translation": "二"},
                        ]
                    )
                if '"cue_id": "3"' in current_cues and '"cue_id": "4"' in current_cues:
                    return json.dumps(
                        [
                            {"cue_id": "3", "translation": "三"},
                            {"cue_id": "4", "translation": "四"},
                        ]
                    )
                raise AssertionError(f"unexpected prompt: {prompt}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_eight_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                model="main-model",
                output_dir=str(output_dir),
                batch_size=4,
                context_before=0,
                context_after=0,
                max_retries=0,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
                concurrency=2,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", ConcurrentShrinkProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        positions = [translated.find(text) for text in ("一", "二", "三", "四", "五", "六", "七", "八")]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(report.find("batch_id: 1 | cue_range: 1-4"), report.find("batch_id: 2 | cue_range: 5-8"))
        self.assertIn("provider_calls: 4", report)
        self.assertIn("child_batch_ids: 3,4", report)

    def test_structured_context_cue_output_violation_retries_then_succeeds_without_retry_count_change(self):
        class FlakyProvider:
            attempts = 0

            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                FlakyProvider.attempts += 1
                if FlakyProvider.attempts == 1:
                    return json.dumps([{"cue_id": "context-99", "translation": "上下文"}])
                return json.dumps([{"cue_id": "1", "translation": "目标"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_one_cue_srt(root)
            output_dir = root / "out"
            context_cue = Cue(
                id="context-99",
                index=99,
                start="00:00:09,000",
                end="00:00:10,000",
                source="context",
            )
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=1,
                max_retries=1,
                cache_enabled=False,
                qa_mode="none",
                engine_version="v2",
                structured_output=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FlakyProvider), patch(
                "translation.pipeline.create_batches",
                return_value=(
                    TranslationBatch(
                        batch_id=1,
                        cues=(ONE_CUE[0],),
                        context_before=(context_cue,),
                        context_after=(),
                    ),
                ),
            ):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(FlakyProvider.attempts, 2)
        self.assertIn("provider_calls: 2", report)
        self.assertIn("retries: 1", report)
        self.assertIn("## Batch Results", report)
        self.assertIn(
            "batch_id: 1 | cue_range: 1-1 | status: success | attempt: 2 | error_type: context_cue_output_violation | cache_hit: False",
            report,
        )
        self.assertRegex(report, r"duration_ms: \d+")

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

    def test_malformed_qa_json_keeps_translation_stage_outputs_and_records_failure(self):
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
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertIn("As an AI, I cannot help.", translated)
        self.assertIn("As an AI, I cannot help.\nhello", bilingual)
        self.assertIn("qa_kept: 0", report)
        self.assertIn("qa_skipped: 1", report)
        self.assertIn("qa_parser_failures: 1", report)
        self.assertIn("qa_provider_failures: 0", report)

    def test_qa_provider_failure_keeps_translation_stage_outputs_and_records_failure(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "As an AI, I cannot help."}])

            def review_suspicious(self, prompt):
                raise RuntimeError("qa provider down")

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

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")
            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertIn("As an AI, I cannot help.", translated)
        self.assertIn("As an AI, I cannot help.\nhello", bilingual)
        self.assertIn("qa_kept: 0", report)
        self.assertIn("qa_skipped: 1", report)
        self.assertIn("qa_parser_failures: 0", report)
        self.assertIn("qa_provider_failures: 1", report)

    def test_qa_fix_remains_post_processing_and_does_not_overwrite_batch_cache(self):
        class SeedProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "1", "translation": "As an AI, I cannot help."}])

            def review_suspicious(self, prompt):
                return json.dumps(
                    [
                        {"id": "1", "action": "fix", "translation": "你好", "reason": "fixed refusal"},
                    ]
                )

        class CachedTranslationProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                raise AssertionError("translation-stage cache hit should skip provider")

            def review_suspicious(self, prompt):
                return json.dumps(
                    [
                        {"id": "1", "action": "fix", "translation": "再次修复", "reason": "fixed refusal"},
                    ]
                )

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
            rerun_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(root / "rerun"),
                batch_size=1,
                cache_path=str(cache_path),
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", SeedProvider):
                run_translation_pipeline(subtitle_path, seed_config)

            cache_rows_after_seed = _read_cache_rows(cache_path)
            seed_translated = (root / "seed" / "translated.zh-CN.srt").read_text(encoding="utf-8")
            seed_report = (root / "seed" / "translation_report.md").read_text(encoding="utf-8")

            with patch("translation.pipeline.OpenAICompatibleProvider", CachedTranslationProvider):
                run_translation_pipeline(subtitle_path, rerun_config)

            cache_rows_after_rerun = _read_cache_rows(cache_path)
            rerun_translated = (root / "rerun" / "translated.zh-CN.srt").read_text(encoding="utf-8")
            rerun_report = (root / "rerun" / "translation_report.md").read_text(encoding="utf-8")

        self.assertEqual(json.loads(cache_rows_after_seed[0]), [{"id": "1", "translation": "As an AI, I cannot help."}])
        self.assertEqual(cache_rows_after_rerun, cache_rows_after_seed)
        self.assertIn("你好", seed_translated)
        self.assertIn("再次修复", rerun_translated)
        self.assertIn("cache_misses: 1", seed_report)
        self.assertIn("cache_hits: 1", rerun_report)
        self.assertIn("provider_calls: 0", rerun_report)
        self.assertIn("qa_fixed: 1", rerun_report)

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


class TranslationSegmentationPipelineIntegrationTests(unittest.TestCase):
    def test_default_off_path_skips_segmentation_and_writes_no_segmentation_artifacts(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps(
                    [
                        {"id": "1", "translation": "你好"},
                        {"id": "2", "translation": "世界"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=2,
                qa_mode="none",
                cache_enabled=False,
            )

            with patch("translation.pipeline.segment_subtitles", side_effect=AssertionError("segmenter should not be called")):
                with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                    run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")

        self.assertEqual(translated.count("-->"), 2)
        self.assertIn("你好", translated)
        self.assertIn("世界", translated)
        self.assertIn("你好\nhello", bilingual)
        self.assertIn("世界\nworld", bilingual)
        self.assertFalse((output_dir / "segmented_source.srt").exists())
        self.assertFalse((output_dir / "translation_units.json").exists())
        self.assertFalse((output_dir / "cue_map.json").exists())
        self.assertFalse((output_dir / "segmentation_report.md").exists())

    def test_enabled_single_file_segmentation_path_uses_segment_units_and_writes_artifacts(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "u001", "translation": "你好世界"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=2,
                qa_mode="none",
                cache_enabled=False,
                preprocess_auto_subs=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")
            segmented_source = (output_dir / "segmented_source.srt").read_text(encoding="utf-8")
            translation_units_payload = json.loads((output_dir / "translation_units.json").read_text(encoding="utf-8"))
            cue_map_payload = json.loads((output_dir / "cue_map.json").read_text(encoding="utf-8"))
            segmentation_report = (output_dir / "segmentation_report.md").read_text(encoding="utf-8")

        unit_payload = translation_units_payload["units"][0]
        self.assertEqual(translated.count("-->"), 1)
        self.assertIn("你好世界", translated)
        self.assertIn("你好世界\nhello world", bilingual)
        self.assertIn("hello world", segmented_source)
        self.assertEqual(unit_payload["unit_id"], "u001")
        self.assertEqual(unit_payload["source_cue_ids"], ["1", "2"])
        self.assertNotEqual(unit_payload["unit_id"], unit_payload["source_cue_ids"][0])
        self.assertNotEqual(unit_payload["unit_id"], unit_payload["source_cue_ids"][1])
        self.assertEqual(cue_map_payload["units"]["u001"]["source_cue_ids"], ["1", "2"])
        self.assertEqual([span["cue_id"] for span in unit_payload["source_spans"]], ["1", "2"])
        self.assertIn("# Segmentation Report", segmentation_report)

    def test_enabled_segmentation_report_includes_auto_sub_section_and_artifact_links(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "u001", "translation": "你好世界"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=2,
                qa_mode="none",
                cache_enabled=False,
                preprocess_auto_subs=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            report = (output_dir / "translation_report.md").read_text(encoding="utf-8")

        self.assertIn("## Auto-sub Segmentation", report)
        self.assertIn("- enabled: true", report)
        self.assertIn("- source_mode: single_file", report)
        self.assertIn("- segmentation_strategy_version: cycle1-rules-v1", report)
        self.assertIn("- timing_strategy_version: cycle1-timing-v1", report)
        self.assertIn("- segmentation_report: segmentation_report.md", report)
        self.assertIn("- translation_units: translation_units.json", report)
        self.assertIn("- cue_map: cue_map.json", report)
        self.assertIn("- segmented_source: segmented_source.srt", report)

    def test_padding_only_units_are_excluded_from_outputs_but_retained_in_artifacts(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                return json.dumps([{"id": "u002", "translation": "片段"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_full_vtt_padding_case(root)
            output_dir = root / "out"
            config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(output_dir),
                batch_size=2,
                qa_mode="none",
                cache_enabled=False,
                preprocess_auto_subs=True,
                auto_sub_source_mode="full_vtt_window",
                auto_sub_full_vtt_path=str(subtitle_path),
                auto_sub_clip_start_ms=1000,
                auto_sub_clip_end_ms=3000,
                auto_sub_padding_before_ms=1000,
                auto_sub_padding_after_ms=1000,
                segment_max_source_cues=1,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, config)

            translated = (output_dir / "translated.zh-CN.srt").read_text(encoding="utf-8")
            bilingual = (output_dir / "bilingual.srt").read_text(encoding="utf-8")
            translation_units = (output_dir / "translation_units.json").read_text(encoding="utf-8")

        self.assertEqual(translated.count("-->"), 1)
        self.assertIn("片段", translated)
        self.assertIn("片段\ninside clip", bilingual)
        self.assertIn('"boundary_type": "padding_only"', translation_units)
        self.assertNotIn("pad before", bilingual)
        self.assertNotIn("pad after", bilingual)

    def test_preprocess_auto_subs_changes_effective_cache_identity_for_same_input(self):
        class FakeProvider:
            def __init__(self, config):
                self.config = config

            def translate_batch(self, prompt):
                if 'u001' in prompt:
                    return json.dumps([{"id": "u001", "translation": "你好世界"}])
                return json.dumps(
                    [
                        {"id": "1", "translation": "你好"},
                        {"id": "2", "translation": "世界"},
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)
            cache_path = root / "translation-cache.sqlite3"
            disabled_output_dir = root / "disabled"
            enabled_output_dir = root / "enabled"
            disabled_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(disabled_output_dir),
                cache_path=str(cache_path),
                batch_size=2,
                qa_mode="none",
            )
            enabled_config = TranslationConfig(
                api_key="test-secret",
                output_dir=str(enabled_output_dir),
                cache_path=str(cache_path),
                batch_size=2,
                qa_mode="none",
                preprocess_auto_subs=True,
            )

            with patch("translation.pipeline.OpenAICompatibleProvider", FakeProvider):
                run_translation_pipeline(subtitle_path, disabled_config)
                disabled_keys = _read_cache_keys(cache_path)
                run_translation_pipeline(subtitle_path, enabled_config)
                enabled_keys = _read_cache_keys(cache_path)

        self.assertEqual(len(disabled_keys), 1)
        self.assertEqual(len(enabled_keys), 2)
        self.assertIn(disabled_keys[0], enabled_keys)
        self.assertEqual(len(set(enabled_keys)), 2)

    def test_full_vtt_window_missing_requirements_fail_before_provider_construction(self):
        class FailingProvider:
            def __init__(self, config):
                raise AssertionError("provider should not be constructed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subtitle_path = _write_two_cue_srt(root)

            config_factories = {
                "missing_full_vtt_path": lambda: TranslationConfig(
                    api_key="test-secret",
                    output_dir=str(root / "out1"),
                    preprocess_auto_subs=True,
                    auto_sub_source_mode="full_vtt_window",
                    auto_sub_clip_start_ms=1000,
                    auto_sub_clip_end_ms=2000,
                ),
                "missing_clip_start": lambda: TranslationConfig(
                    api_key="test-secret",
                    output_dir=str(root / "out2"),
                    preprocess_auto_subs=True,
                    auto_sub_source_mode="full_vtt_window",
                    auto_sub_full_vtt_path=str(subtitle_path),
                    auto_sub_clip_end_ms=2000,
                ),
                "missing_clip_end": lambda: TranslationConfig(
                    api_key="test-secret",
                    output_dir=str(root / "out3"),
                    preprocess_auto_subs=True,
                    auto_sub_source_mode="full_vtt_window",
                    auto_sub_full_vtt_path=str(subtitle_path),
                    auto_sub_clip_start_ms=1000,
                ),
            }

            for case_name, config_factory in config_factories.items():
                with self.subTest(case_name=case_name):
                    with patch("translation.pipeline.OpenAICompatibleProvider", FailingProvider):
                        with self.assertRaises(ValueError):
                            run_translation_pipeline(subtitle_path, config_factory())


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


def _write_four_cue_srt(temp_dir: Path) -> Path:
    subtitle_path = temp_dir / "sample.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\none\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\ntwo\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\nthree\n\n"
        "4\n00:00:03,000 --> 00:00:04,000\nfour\n\n",
        encoding="utf-8",
    )
    return subtitle_path


def _write_eight_cue_srt(temp_dir: Path) -> Path:
    subtitle_path = temp_dir / "sample.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\none\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\ntwo\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\nthree\n\n"
        "4\n00:00:03,000 --> 00:00:04,000\nfour\n\n"
        "5\n00:00:04,000 --> 00:00:05,000\nfive\n\n"
        "6\n00:00:05,000 --> 00:00:06,000\nsix\n\n"
        "7\n00:00:06,000 --> 00:00:07,000\nseven\n\n"
        "8\n00:00:07,000 --> 00:00:08,000\neight\n\n",
        encoding="utf-8",
    )
    return subtitle_path


def _write_full_vtt_padding_case(temp_dir: Path) -> Path:
    subtitle_path = temp_dir / "sample.vtt"
    subtitle_path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:00.500\n"
        "pad before\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "inside clip\n\n"
        "00:00:03.500 --> 00:00:03.900\n"
        "pad after\n\n",
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


def _read_cache_keys(cache_path: Path) -> list[str]:
    connection = sqlite3.connect(cache_path)
    try:
        rows = connection.execute("SELECT cache_key FROM translation_cache ORDER BY updated_at, cache_key").fetchall()
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
