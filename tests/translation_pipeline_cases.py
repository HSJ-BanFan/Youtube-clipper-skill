import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from translation.config import TranslationConfig
from translation.models import Cue, ErrorType, TranslationBatch
from translation.pipeline import (
    _build_batch_source_hash,
    _build_structured_batch_record,
    _classify_structured_cue_id,
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
        ]

        for response in bad_responses:
            with self.subTest(response=response):
                with self.assertRaisesRegex(RuntimeError, "QA response"):
                    parse_qa_response(response, candidates)


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
        self.assertFalse((output_dir / "global_context.md").exists())
        self.assertFalse((output_dir / "translation_report.md").exists())

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
