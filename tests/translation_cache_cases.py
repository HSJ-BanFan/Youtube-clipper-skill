import sqlite3
import tempfile
import unittest
from pathlib import Path

from translation.cache import CacheEntry, TranslationCache, build_batch_cache_key
from translation.config import TranslationConfig


class TranslationCacheTests(unittest.TestCase):
    def _identity_values(self, **overrides):
        values = {
            "engine_version": "v1",
            "structured_output": False,
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "gpt-test",
            "main_model_alias": "main",
            "target_lang": "zh-CN",
            "prompt_version": "prompt-v1",
            "output_schema_version": "v1",
            "batching_strategy_version": "v1",
            "glossary_hash": "glossary-a",
            "context_hash": "context-a",
            "batch_source_hash": "batch-a",
        }
        values.update(overrides)
        return values

    def _cache_key(self, **overrides):
        values = self._identity_values(**overrides)
        return build_batch_cache_key(
            values["engine_version"],
            values["structured_output"],
            values["provider"],
            values["base_url"],
            values["model"],
            values["main_model_alias"],
            values["target_lang"],
            values["prompt_version"],
            values["output_schema_version"],
            values["batching_strategy_version"],
            values["glossary_hash"],
            values["context_hash"],
            values["batch_source_hash"],
        )

    def _entry(self, **overrides):
        values = {
            **self._identity_values(),
            "result_json": '{"items": [{"text": "你好"}]}',
        }
        values.update(overrides)
        cache_key = self._cache_key(**values)
        return CacheEntry(cache_key=cache_key, **values)

    def _cache_key_from_config(self, config: TranslationConfig, **overrides):
        return self._cache_key(
            engine_version=config.engine_version,
            structured_output=config.structured_output,
            provider=config.provider,
            base_url=config.base_url,
            model=config.model,
            main_model_alias=config.main_model_alias,
            target_lang=config.target_lang,
            output_schema_version=config.output_schema_version,
            batching_strategy_version=config.batching_strategy_version,
            **overrides,
        )

    def test_set_get_returns_cached_result_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with TranslationCache(Path(temp_dir) / "nested" / "cache.sqlite3") as cache:
                entry = self._entry()

                cache.set(entry)
                result = cache.get(entry.cache_key)

        self.assertEqual(result, entry.result_json)

    def test_missing_key_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with TranslationCache(Path(temp_dir) / "cache.sqlite3") as cache:
                result = cache.get("missing-key")

        self.assertIsNone(result)

    def test_cache_key_changes_when_identity_fields_change(self):
        base_key = self._cache_key()

        changed_keys = {
            "engine_version": self._cache_key(engine_version="v2"),
            "structured_output": self._cache_key(structured_output=True),
            "base_url": self._cache_key(base_url="https://gateway.example.test/v1"),
            "model": self._cache_key(model="gpt-other"),
            "main_model_alias": self._cache_key(main_model_alias="primary"),
            "target_lang": self._cache_key(target_lang="ja"),
            "prompt_version": self._cache_key(prompt_version="prompt-v2"),
            "output_schema_version": self._cache_key(output_schema_version="v2"),
            "batching_strategy_version": self._cache_key(batching_strategy_version="v2"),
            "glossary_hash": self._cache_key(glossary_hash="glossary-b"),
            "context_hash": self._cache_key(context_hash="context-b"),
            "batch_source_hash": self._cache_key(batch_source_hash="batch-b"),
        }

        for field_name, changed_key in changed_keys.items():
            with self.subTest(field_name=field_name):
                self.assertNotEqual(base_key, changed_key)

    def test_cache_key_ignores_qa_mode_failure_mode_and_output_settings(self):
        base_config = TranslationConfig(api_key="test-secret")
        base_key = self._cache_key_from_config(base_config)

        variants = {
            "qa_mode": TranslationConfig(api_key="test-secret", qa_mode="none"),
            "failure_mode": TranslationConfig(api_key="test-secret", failure_mode="partial"),
            "output_dir": TranslationConfig(api_key="test-secret", output_dir="out"),
            "output_path": TranslationConfig(api_key="test-secret", output_path="single.srt"),
            "dry_run": TranslationConfig(api_key="test-secret", dry_run=True),
            "overwrite": TranslationConfig(api_key="test-secret", overwrite=True),
        }

        for field_name, config in variants.items():
            with self.subTest(field_name=field_name):
                self.assertEqual(base_key, self._cache_key_from_config(config))

    def test_existing_phase1_cache_schema_migrates_additively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache.sqlite3"
            connection = sqlite3.connect(cache_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE translation_cache (
                        cache_key TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        target_lang TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        glossary_hash TEXT NOT NULL,
                        context_hash TEXT NOT NULL,
                        batch_source_hash TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
            finally:
                connection.close()

            with TranslationCache(cache_path) as cache:
                entry = self._entry()
                cache.set(entry)
                result = cache.get(entry.cache_key)

            connection = sqlite3.connect(cache_path)
            try:
                schema_rows = connection.execute("PRAGMA table_info(translation_cache)").fetchall()
            finally:
                connection.close()
            column_names = [str(row[1]) for row in schema_rows]

        self.assertEqual(result, entry.result_json)
        self.assertIn("engine_version", column_names)
        self.assertIn("structured_output", column_names)
        self.assertIn("base_url", column_names)
        self.assertIn("main_model_alias", column_names)
        self.assertIn("output_schema_version", column_names)
        self.assertIn("batching_strategy_version", column_names)

    def test_no_api_key_column_exists_in_sqlite_schema(self):
        secret_api_key = "sk-secret-value-that-must-not-be-stored"
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache.sqlite3"
            with TranslationCache(cache_path) as cache:
                entry = self._entry(result_json='{"translated": "safe content"}')

                cache.set(entry)

            connection = sqlite3.connect(cache_path)
            try:
                schema_rows = connection.execute("PRAGMA table_info(translation_cache)").fetchall()
                content_rows = connection.execute(
                    "SELECT cache_key, provider, model, target_lang, prompt_version, "
                    "glossary_hash, context_hash, batch_source_hash, result_json FROM translation_cache"
                ).fetchall()
            finally:
                connection.close()
            column_names = [str(row[1]) for row in schema_rows]
            raw_bytes = cache_path.read_bytes()

        self.assertNotIn("api_key", column_names)
        self.assertNotIn(secret_api_key, str(content_rows))
        self.assertNotIn(secret_api_key.encode("utf-8"), raw_bytes)


if __name__ == "__main__":
    unittest.main()
