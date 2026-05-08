import sqlite3
import tempfile
import unittest
from pathlib import Path

from translation.cache import CacheEntry, TranslationCache, build_batch_cache_key


class TranslationCacheTests(unittest.TestCase):
    def _entry(self, **overrides):
        values = {
            "provider": "openai-compatible",
            "model": "gpt-test",
            "target_lang": "zh-CN",
            "prompt_version": "prompt-v1",
            "glossary_hash": "glossary-a",
            "context_hash": "context-a",
            "batch_source_hash": "batch-a",
            "result_json": '{"items": [{"text": "你好"}]}',
        }
        values.update(overrides)
        cache_key = build_batch_cache_key(
            values["provider"],
            values["model"],
            values["target_lang"],
            values["prompt_version"],
            values["glossary_hash"],
            values["context_hash"],
            values["batch_source_hash"],
        )
        return CacheEntry(cache_key=cache_key, **values)

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

    def test_cache_key_changes_when_prompt_glossary_or_context_changes(self):
        base = self._entry()

        prompt_changed = self._entry(prompt_version="prompt-v2")
        glossary_changed = self._entry(glossary_hash="glossary-b")
        context_changed = self._entry(context_hash="context-b")

        self.assertNotEqual(base.cache_key, prompt_changed.cache_key)
        self.assertNotEqual(base.cache_key, glossary_changed.cache_key)
        self.assertNotEqual(base.cache_key, context_changed.cache_key)

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
