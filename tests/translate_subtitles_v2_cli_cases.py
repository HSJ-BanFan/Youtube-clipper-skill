import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "translate_subtitles_v2.py"


class TranslateSubtitlesV2CliTests(unittest.TestCase):
    def test_help_displays_expected_arguments(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("subtitle_path", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--no-cache", result.stdout)
        self.assertIn("--no-qa", result.stdout)
        self.assertNotIn("--api-key", result.stdout)

    def test_missing_subtitle_path_returns_usage_error(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("subtitle_path", result.stderr)

    def test_dry_run_prints_safe_config_and_does_not_write_secret(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TRANSLATION_API_KEY=env-secret\nTRANSLATION_MODEL=env-model\n",
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "out"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(subtitle_path),
                    "--env-file",
                    str(env_path),
                    "--output-dir",
                    str(output_dir),
                    "--model",
                    "cli-model",
                    "--target-lang",
                    "zh-CN",
                    "--no-cache",
                    "--no-qa",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Dry run", result.stdout)
        self.assertIn("cli-model", result.stdout)
        self.assertIn("cache_enabled: False", result.stdout)
        self.assertIn("qa_mode: none", result.stdout)
        self.assertIn("translated.zh-CN.srt", result.stdout)
        self.assertNotIn("env-secret", result.stdout)
        self.assertNotIn("TRANSLATION_API_KEY", result.stdout)

    def test_existing_outputs_require_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            (output_dir / "translated.zh-CN.srt").write_text("existing", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(subtitle_path),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertIn("--overwrite", result.stderr)

    def test_overwrite_allows_dry_run_when_output_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            (output_dir / "translated.zh-CN.srt").write_text("existing", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(subtitle_path),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    "--overwrite",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("overwrite: True", result.stdout)

    def test_vtt_dry_run_reports_vtt_input_and_srt_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(subtitle_path), "--dry-run"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("input_format: vtt", result.stdout)
        self.assertIn("output_format: srt", result.stdout)
        self.assertIn("cue_count: 1", result.stdout)

    def test_non_dry_run_without_api_key_fails_before_network_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "out"
            empty_env = Path(temp_dir) / ".env"
            empty_env.write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(subtitle_path),
                    "--env-file",
                    str(empty_env),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("TRANSLATION_API_KEY", result.stderr)
        self.assertNotIn("Authorization", result.stderr)


if __name__ == "__main__":
    unittest.main()
