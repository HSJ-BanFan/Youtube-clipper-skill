import tempfile
import unittest
from pathlib import Path

from translation.config import TranslationConfig, load_config
from translation.pipeline import run_translation_pipeline


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

    def test_qa_mode_accepts_off_as_skip_alias(self):
        config = TranslationConfig(qa_mode="off")

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

    def test_system_environment_values_override_env_file_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("TRANSLATION_MODEL=file-model\n", encoding="utf-8")

            config = load_config(
                cli_args={},
                env_path=env_path,
                environ={"TRANSLATION_MODEL": "system-model"},
            )

        self.assertEqual(config.model, "system-model")

    def test_safe_config_redacts_credentials_when_password_contains_at_symbol(self):
        config = TranslationConfig(base_url="https://user:p@ssword@example.test/v1")

        safe = config.to_safe_dict()

        self.assertEqual(safe["base_url"], "https://<redacted>@example.test/v1")
        self.assertNotIn("p@ssword", str(safe))

    def test_invalid_env_numeric_value_names_environment_variable(self):
        with self.assertRaisesRegex(ValueError, "TRANSLATION_BATCH_SIZE"):
            load_config(
                cli_args={},
                env_path=None,
                environ={"TRANSLATION_BATCH_SIZE": "abc"},
            )

    def test_invalid_mode_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            TranslationConfig(mode="invalid")

    def test_target_lang_allows_letters_numbers_and_hyphens(self):
        config = TranslationConfig(target_lang="zh-CN")

        self.assertEqual(config.target_lang, "zh-CN")

    def test_target_lang_rejects_path_traversal_segments(self):
        with self.assertRaisesRegex(ValueError, "target_lang must contain only letters, numbers, and hyphens"):
            TranslationConfig(target_lang="../zh")

    def test_target_lang_rejects_forward_slashes(self):
        with self.assertRaisesRegex(ValueError, "target_lang must contain only letters, numbers, and hyphens"):
            TranslationConfig(target_lang="zh/CN")

    def test_target_lang_rejects_backslashes(self):
        with self.assertRaisesRegex(ValueError, "target_lang must contain only letters, numbers, and hyphens"):
            TranslationConfig(target_lang=r"zh\CN")

    def test_srt_cue_count_ignores_digit_only_text_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "42\n\n"
                "2\n"
                "00:00:02.000 --> 00:00:03.000\n"
                "done\n\n",
                encoding="utf-8",
            )

            result = run_translation_pipeline(subtitle_path, TranslationConfig(dry_run=True))

        self.assertEqual(result.cue_count, 2)

    def test_dry_run_uses_parser_and_returns_cue_previews(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "hello\nworld\n\n"
                "2\n"
                "00:00:02,000 --> 00:00:03,000\n"
                "done\n\n",
                encoding="utf-8",
            )

            result = run_translation_pipeline(subtitle_path, TranslationConfig(dry_run=True))

        self.assertEqual(result.cue_count, 2)
        self.assertEqual(result.first_cue_preview, "hello world")
        self.assertEqual(result.last_cue_preview, "done")

    def test_dry_run_preview_escapes_terminal_control_characters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello \x1b[2J world\n\n",
                encoding="utf-8",
            )

            result = run_translation_pipeline(subtitle_path, TranslationConfig(dry_run=True))

        self.assertNotIn("\x1b", result.first_cue_preview)
        self.assertEqual(result.first_cue_preview, "hello [2J world")

    def test_dry_run_preview_preserves_unicode_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n你好 world\n\n",
                encoding="utf-8",
            )

            result = run_translation_pipeline(subtitle_path, TranslationConfig(dry_run=True))

        self.assertEqual(result.first_cue_preview, "你好 world")

    def test_non_dry_run_requires_api_key_after_parser_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "TRANSLATION_API_KEY is required. Set it as an environment variable or provide it via --env-file.",
            ) as error:
                run_translation_pipeline(subtitle_path, TranslationConfig(dry_run=False))

        self.assertNotIn("--api-key", str(error.exception))


if __name__ == "__main__":
    unittest.main()
