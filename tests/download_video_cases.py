import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "download_video.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_video


class DownloadVideoCliTests(unittest.TestCase):
    def test_help_displays_cookie_arguments(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("youtube_url", result.stdout)
        self.assertIn("--cookies-from-browser", result.stdout)
        self.assertIn("--cookies-file", result.stdout)
        self.assertIn("--proxy", result.stdout)
        self.assertIn("--rate-limit", result.stdout)
        self.assertIn("--env-file", result.stdout)


class DownloadVideoSettingsTests(unittest.TestCase):
    def test_cookies_from_browser_sets_ydl_opt(self):
        settings = {
            "cookies_from_browser": "firefox",
            "cookies_file": None,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "1080",
        }

        options = download_video.build_ydl_opts(Path("out"), settings)

        self.assertEqual(options["cookiesfrombrowser"], ("firefox",))

    def test_cookies_file_sets_ydl_opt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_path = Path(temp_dir) / "cookies.txt"
            cookie_path.write_text("cookie-data", encoding="utf-8")
            env_path = Path(temp_dir) / "empty.env"
            env_path.write_text("", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--cookies-file",
                    str(cookie_path),
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = download_video.resolve_download_settings(args)

        options = download_video.build_ydl_opts(Path("out"), settings)
        self.assertEqual(options["cookiefile"], str(cookie_path.resolve()))

    def test_cookies_mutual_exclusion_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_path = Path(temp_dir) / "cookies.txt"
            cookie_path.write_text("cookie-data", encoding="utf-8")
            env_path = Path(temp_dir) / "empty.env"
            env_path.write_text("", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--cookies-from-browser",
                    "firefox",
                    "--cookies-file",
                    str(cookie_path),
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    download_video.resolve_download_settings(args)

    def test_cookies_file_missing_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.txt"
            env_path = Path(temp_dir) / "empty.env"
            env_path.write_text("", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--cookies-file",
                    str(missing),
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(FileNotFoundError, "Cookies file not found"):
                    download_video.resolve_download_settings(args)

    def test_proxy_from_cli_overrides_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "download.env"
            env_path.write_text("YT_DLP_PROXY=http://env-proxy:8080\n", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--proxy",
                    "http://cli-proxy:8080",
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = download_video.resolve_download_settings(args)

        self.assertEqual(settings["proxy"], "http://cli-proxy:8080")

    def test_rate_limit_from_env_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "download.env"
            env_path.write_text("YT_DLP_RATE_LIMIT=50K\n", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = download_video.resolve_download_settings(args)

        self.assertEqual(settings["rate_limit"], "50K")

    def test_env_file_flag_loads_specified_dotenv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "download.env"
            env_path.write_text("YT_DLP_COOKIES_FROM_BROWSER=firefox\n", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = download_video.resolve_download_settings(args)

        self.assertEqual(settings["cookies_from_browser"], "firefox")

    def test_none_values_omitted_from_ydl_opts(self):
        settings = {
            "cookies_from_browser": None,
            "cookies_file": None,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "1080",
        }

        options = download_video.build_ydl_opts(Path("out"), settings)

        self.assertNotIn("cookiesfrombrowser", options)
        self.assertNotIn("cookiefile", options)
        self.assertNotIn("proxy", options)
        self.assertNotIn("ratelimit", options)

    def test_max_video_height_affects_format(self):
        settings = {
            "cookies_from_browser": None,
            "cookies_file": None,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "2160",
        }

        options = download_video.build_ydl_opts(Path("out"), settings)

        self.assertIn("height<=2160", options["format"])


class DownloadVideoExecutionTests(unittest.TestCase):
    def test_sanitize_error_message_redacts_proxy_credentials(self):
        message = "proxy failed: http://user:secret@proxy.example:8080 timeout"

        sanitized = download_video._sanitize_error_message(message, "http://user:secret@proxy.example:8080")

        self.assertNotIn("secret", sanitized)
        self.assertIn("<redacted-proxy>", sanitized)

    def test_main_returns_error_for_conflicting_cookie_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_path = Path(temp_dir) / "cookies.txt"
            cookie_path.write_text("cookie-data", encoding="utf-8")
            env_path = Path(temp_dir) / "download.env"
            env_path.write_text("YT_DLP_PROXY=http://user:secret@proxy.example:8080\n", encoding="utf-8")
            stderr = io.StringIO()

            with patch.dict(os.environ, {}, clear=True):
                with patch("sys.stderr", stderr):
                    exit_code = download_video.main(
                        [
                            "https://youtube.com/watch?v=Ckt1cj0xjRM",
                            "--cookies-from-browser",
                            "firefox",
                            "--cookies-file",
                            str(cookie_path),
                            "--env-file",
                            str(env_path),
                        ]
                    )

        self.assertEqual(exit_code, 1)
        self.assertIn("mutually exclusive", stderr.getvalue())
        self.assertNotIn("secret", stderr.getvalue())

    def test_download_video_returns_dict_contract(self):
        class FakeYoutubeDL:
            last_options = None

            def __init__(self, options):
                FakeYoutubeDL.last_options = options
                self.options = options
                self.output_path = Path(options["outtmpl"].replace("%(id)s", "abc123").replace("%(ext)s", "mp4"))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                if download:
                    self.output_path.write_bytes(b"video")
                    self.output_path.with_name("abc123.en.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
                return {
                    "id": "abc123",
                    "title": "Example Video",
                    "duration": 123,
                }

            def prepare_filename(self, info):
                return str(self.output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                result = download_video.download_video(
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    temp_dir,
                    cookies_from_browser="firefox",
                    proxy="http://proxy:8080",
                    rate_limit="50K",
                    max_video_height="2160",
                )

        self.assertEqual(set(result.keys()), {"video_path", "subtitle_path", "title", "duration", "file_size", "video_id"})
        self.assertEqual(result["title"], "Example Video")
        self.assertEqual(result["video_id"], "abc123")
        self.assertEqual(FakeYoutubeDL.last_options["cookiesfrombrowser"], ("firefox",))
        self.assertEqual(FakeYoutubeDL.last_options["proxy"], "http://proxy:8080")
        self.assertEqual(FakeYoutubeDL.last_options["ratelimit"], "50K")
        self.assertIn("height<=2160", FakeYoutubeDL.last_options["format"])


if __name__ == "__main__":
    unittest.main()
