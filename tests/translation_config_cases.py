import tempfile
import unittest
from pathlib import Path

from translation.config import TranslationConfig, load_config


class TranslationConfigTests(unittest.TestCase):
    def test_env_values_load_and_cli_overrides_take_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TRANSLATION_PROVIDER=openai-compatible",
                        "TRANSLATION_BASE_URL=http://env.example/v1",
                        "TRANSLATION_API_KEY=env-secret",
                        "TRANSLATION_MODEL=env-model",
                        "TRANSLATION_TARGET_LANG=ja-JP",
                        "TRANSLATION_BATCH_SIZE=42",
                        "TRANSLATION_CONTEXT_BEFORE=4",
                        "TRANSLATION_CONTEXT_AFTER=5",
                        "TRANSLATION_TEMPERATURE=0.7",
                        "TRANSLATION_MAX_RETRIES=2",
                        "TRANSLATION_CACHE=false",
                        "TRANSLATION_CACHE_PATH=env-cache.sqlite3",
                        "TRANSLATION_GLOSSARY_PATH=env-glossary.md",
                        "TRANSLATION_QA=suspicious-only",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(
                cli_args={
                    "model": "cli-model",
                    "target_lang": "zh-CN",
                    "batch_size": 80,
                    "cache_enabled": True,
                    "qa_mode": "none",
                },
                env_path=env_path,
                environ={},
            )

        self.assertEqual(config.base_url, "http://env.example/v1")
        self.assertEqual(config.api_key, "env-secret")
        self.assertEqual(config.model, "cli-model")
        self.assertEqual(config.review_model, None)
        self.assertEqual(config.effective_review_model, "cli-model")
        self.assertEqual(config.target_lang, "zh-CN")
        self.assertEqual(config.batch_size, 80)
        self.assertTrue(config.cache_enabled)
        self.assertEqual(config.qa_mode, "none")

    def test_unsupported_provider_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "only openai-compatible is supported"):
            TranslationConfig(provider="gemini")

    def test_invalid_numeric_values_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be greater than 0"):
            TranslationConfig(batch_size=0)

        with self.assertRaisesRegex(ValueError, "temperature must be between 0 and 2"):
            TranslationConfig(temperature=3.0)

    def test_env_api_key_loads_without_leaking_in_safe_outputs(self):
        config = load_config(
            cli_args={},
            env_path=None,
            environ={"TRANSLATION_API_KEY": "env-secret"},
        )

        self.assertEqual(config.api_key, "env-secret")
        safe = config.to_safe_dict()
        self.assertNotIn("api_key", safe)
        self.assertNotIn("env-secret", str(safe))
        self.assertNotIn("env-secret", repr(config))


if __name__ == "__main__":
    unittest.main()
