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
        self.assertIn("--fresh-firefox-cookies", result.stdout)
        self.assertIn("--fresh-firefox-profile", result.stdout)
        self.assertIn("--keep-temp-cookies", result.stdout)
        self.assertIn("--proxy", result.stdout)
        self.assertIn("--rate-limit", result.stdout)
        self.assertIn("--env-file", result.stdout)


class DownloadVideoSettingsTests(unittest.TestCase):
    def test_cookies_from_browser_sets_ydl_opt(self):
        settings = {
            "cookies_from_browser": "firefox",
            "cookies_file": None,
            "fresh_firefox_cookies": False,
            "fresh_firefox_profile": None,
            "fresh_firefox_cookiefile": None,
            "keep_temp_cookies": False,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "1080",
            "output_dir": "out",
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

    def test_fresh_firefox_cookies_conflicts_with_cookies_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_path = Path(temp_dir) / "cookies.txt"
            cookie_path.write_text("cookie-data", encoding="utf-8")
            env_path = Path(temp_dir) / "empty.env"
            env_path.write_text("", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--fresh-firefox-cookies",
                    "--cookies-file",
                    str(cookie_path),
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    download_video.resolve_download_settings(args)

    def test_fresh_firefox_cookies_conflicts_with_cookies_from_browser(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "empty.env"
            env_path.write_text("", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--fresh-firefox-cookies",
                    "--cookies-from-browser",
                    "firefox",
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    download_video.resolve_download_settings(args)

    def test_fresh_firefox_profile_conflicts_with_cookies_from_browser(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "empty.env"
            env_path.write_text("", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--fresh-firefox-profile",
                    "work",
                    "--cookies-from-browser",
                    "firefox",
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

    def test_empty_max_video_height_uses_default(self):
        with patch.dict(os.environ, {"MAX_VIDEO_HEIGHT": ""}, clear=True):
            self.assertEqual(download_video._resolve_max_video_height(), "1080")

    def test_max_video_height_strips_whitespace_before_validation(self):
        with patch.dict(os.environ, {"MAX_VIDEO_HEIGHT": " 720 "}, clear=True):
            self.assertEqual(download_video._resolve_max_video_height(), "720")

    def test_non_numeric_max_video_height_error_includes_value(self):
        with patch.dict(os.environ, {"MAX_VIDEO_HEIGHT": "large"}, clear=True):
            with self.assertRaisesRegex(ValueError, "large"):
                download_video._resolve_max_video_height()

    def test_cli_output_dir_overrides_env_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cli_output_dir = Path(temp_dir) / "cli-out"
            env_output_dir = Path(temp_dir) / "env-out"
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    str(cli_output_dir),
                    "--env-file",
                    str(Path(temp_dir) / "missing.env"),
                ]
            )

            with patch.dict(os.environ, {"OUTPUT_DIR": str(env_output_dir)}, clear=True):
                settings = download_video.resolve_download_settings(args)

        self.assertEqual(settings["output_dir"], str(cli_output_dir))

    def test_env_output_dir_adds_timestamp_child(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_output_dir = Path(temp_dir) / "env-out"
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--env-file",
                    str(Path(temp_dir) / "missing.env"),
                ]
            )

            with patch.dict(os.environ, {"OUTPUT_DIR": str(env_output_dir)}, clear=True):
                settings = download_video.resolve_download_settings(args)

        output_dir = Path(settings["output_dir"])
        self.assertEqual(output_dir.parent, env_output_dir)
        self.assertRegex(output_dir.name, r"^\d{8}_\d{6}$")

    def test_env_output_dir_expands_home_directory(self):
        home_dir = str(Path.home())
        args = download_video.parse_args(
            [
                "https://youtube.com/watch?v=Ckt1cj0xjRM",
                "--env-file",
                str(Path.cwd() / "missing.env"),
            ]
        )

        with patch.dict(os.environ, {"OUTPUT_DIR": "~/Downloads", "USERPROFILE": home_dir, "HOME": home_dir}, clear=True):
            settings = download_video.resolve_download_settings(args)

        output_dir = Path(settings["output_dir"])
        self.assertEqual(output_dir.parent, Path.home() / "Downloads")
        self.assertRegex(output_dir.name, r"^\d{8}_\d{6}$")

    def test_cli_output_dir_expands_home_directory(self):
        home_dir = str(Path.home())
        with tempfile.TemporaryDirectory() as temp_dir:
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "~/Downloads",
                    "--env-file",
                    str(Path(temp_dir) / "missing.env"),
                ]
            )

            with patch.dict(os.environ, {"USERPROFILE": home_dir, "HOME": home_dir}, clear=True):
                settings = download_video.resolve_download_settings(args)

        self.assertEqual(Path(settings["output_dir"]), Path.home() / "Downloads")

    def test_default_output_dir_uses_youtube_clips_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--env-file",
                    str(Path(temp_dir) / "missing.env"),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = download_video.resolve_download_settings(args)

        output_dir = Path(settings["output_dir"])
        self.assertEqual(output_dir.parent.name, "youtube-clips")
        self.assertRegex(output_dir.name, r"^\d{8}_\d{6}$")
        self.assertNotEqual(output_dir, Path.cwd())

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

    def test_fresh_firefox_env_flags_resolve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "download.env"
            env_path.write_text(
                "YT_DLP_FRESH_FIREFOX_COOKIES=true\n"
                "YT_DLP_FRESH_FIREFOX_PROFILE=work\n"
                "YT_DLP_KEEP_TEMP_COOKIES=1\n",
                encoding="utf-8",
            )
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = download_video.resolve_download_settings(args)

        self.assertTrue(settings["fresh_firefox_cookies"])
        self.assertEqual(settings["fresh_firefox_profile"], "work")
        self.assertTrue(settings["keep_temp_cookies"])
        self.assertIsNone(settings["fresh_firefox_cookiefile"])

    def test_fresh_firefox_profile_enables_fresh_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "download.env"
            env_path.write_text("", encoding="utf-8")
            args = download_video.parse_args(
                [
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    "--fresh-firefox-profile",
                    "work",
                    "--env-file",
                    str(env_path),
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = download_video.resolve_download_settings(args)

        self.assertTrue(settings["fresh_firefox_cookies"])
        self.assertEqual(settings["fresh_firefox_profile"], "work")
        self.assertIsNone(settings["fresh_firefox_cookiefile"])

    def test_fresh_firefox_cookies_sets_ydl_opts(self):
        settings = {
            "cookies_from_browser": None,
            "cookies_file": None,
            "fresh_firefox_cookies": True,
            "fresh_firefox_profile": None,
            "fresh_firefox_cookiefile": str(Path(tempfile.gettempdir()) / "fresh-firefox-cookies.txt"),
            "keep_temp_cookies": False,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "1080",
            "output_dir": "out",
        }

        options = download_video.build_ydl_opts(Path("out"), settings)

        self.assertEqual(options["cookiesfrombrowser"], ("firefox",))
        self.assertEqual(options["cookiefile"], str(Path(tempfile.gettempdir()) / "fresh-firefox-cookies.txt"))

    def test_fresh_firefox_profile_sets_ydl_opts(self):
        settings = {
            "cookies_from_browser": None,
            "cookies_file": None,
            "fresh_firefox_cookies": True,
            "fresh_firefox_profile": "work",
            "fresh_firefox_cookiefile": str(Path(tempfile.gettempdir()) / "fresh-firefox-cookies.txt"),
            "keep_temp_cookies": False,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "1080",
            "output_dir": "out",
        }

        options = download_video.build_ydl_opts(Path("out"), settings)

        self.assertEqual(options["cookiesfrombrowser"], ("firefox", "work"))
        self.assertEqual(options["cookiefile"], str(Path(tempfile.gettempdir()) / "fresh-firefox-cookies.txt"))

    def test_none_values_omitted_from_ydl_opts(self):
        settings = {
            "cookies_from_browser": None,
            "cookies_file": None,
            "fresh_firefox_cookies": False,
            "fresh_firefox_profile": None,
            "fresh_firefox_cookiefile": None,
            "keep_temp_cookies": False,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "1080",
            "output_dir": "out",
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
            "fresh_firefox_cookies": False,
            "fresh_firefox_profile": None,
            "fresh_firefox_cookiefile": None,
            "keep_temp_cookies": False,
            "proxy": None,
            "rate_limit": None,
            "max_video_height": "2160",
            "output_dir": "out",
        }

        options = download_video.build_ydl_opts(Path("out"), settings)

        self.assertIn("height<=2160", options["format"])


class DownloadVideoExecutionTests(unittest.TestCase):
    def test_sanitize_error_message_redacts_exact_proxy_credentials(self):
        message = "proxy failed: http://user:secret@proxy.example:8080 timeout"

        sanitized = download_video._sanitize_error_message(message, "http://user:secret@proxy.example:8080")

        self.assertNotIn("secret", sanitized)
        self.assertIn("<redacted-proxy>", sanitized)

    def test_sanitize_error_message_redacts_proxy_with_trailing_slash(self):
        message = "proxy failed: http://user:secret@proxy.example:8080/ timeout"

        sanitized = download_video._sanitize_error_message(message, "http://user:secret@proxy.example:8080")

        self.assertNotIn("secret", sanitized)
        self.assertIn("<redacted-proxy>/", sanitized)

    def test_sanitize_error_message_redacts_proxy_userinfo_without_scheme(self):
        message = "proxy failed: user:secret@proxy.example:8080 timeout"

        sanitized = download_video._sanitize_error_message(message, "http://user:secret@proxy.example:8080")

        self.assertNotIn("secret", sanitized)
        self.assertIn("<redacted-userinfo>@proxy.example:8080", sanitized)

    def test_sanitize_error_message_redacts_password_containing_at_symbol(self):
        message = "proxy failed: http://user:p@ss@proxy.example:8080/ timeout"

        sanitized = download_video._sanitize_error_message(message, None)

        self.assertNotIn("p@ss", sanitized)
        self.assertNotIn("ss@proxy", sanitized)
        self.assertIn("http://<redacted-proxy>@proxy.example:8080/", sanitized)

    def test_sanitize_error_message_redacts_fallback_url_credentials(self):
        message = "proxy failed: http://user:secret@proxy.example:8080 timeout"

        sanitized = download_video._sanitize_error_message(message, "http://other:pass@other.example:8080")

        self.assertNotIn("secret", sanitized)
        self.assertIn("proxy.example:8080", sanitized)
        self.assertIn("<redacted-proxy>", sanitized)

    def test_download_video_error_does_not_print_error_to_stdout(self):
        class FailingYoutubeDL:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                raise RuntimeError("proxy failed: http://user:p@ss@proxy.example:8080/ timeout")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FailingYoutubeDL)):
                with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                    with self.assertRaises(RuntimeError):
                        download_video.download_video(
                            "https://youtube.com/watch?v=Ckt1cj0xjRM",
                            temp_dir,
                            proxy="http://user:p@ss@proxy.example:8080",
                        )

        self.assertNotIn("[ERROR] 下载失败", stdout.getvalue())
        self.assertNotIn("p@ss", stdout.getvalue())
        self.assertNotIn("p@ss", stderr.getvalue())

    def test_main_download_failure_redacts_proxy_password_from_output(self):
        class FailingYoutubeDL:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                raise RuntimeError("proxy failed: http://user:p@ss@proxy.example:8080/ timeout")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FailingYoutubeDL)):
                with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                    exit_code = download_video.main(
                        [
                            "https://youtube.com/watch?v=Ckt1cj0xjRM",
                            temp_dir,
                            "--proxy",
                            "http://user:p@ss@proxy.example:8080",
                            "--env-file",
                            str(Path(temp_dir) / "missing.env"),
                        ]
                    )

        self.assertEqual(exit_code, 1)
        self.assertNotIn("p@ss", stdout.getvalue())
        self.assertNotIn("p@ss", stderr.getvalue())
        self.assertNotIn("ss@proxy", stdout.getvalue())
        self.assertNotIn("ss@proxy", stderr.getvalue())

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

    def test_download_video_rejects_conflicting_cookie_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_path = Path(temp_dir) / "cookies.txt"
            cookie_path.write_text("cookie-data", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                download_video.download_video(
                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                    temp_dir,
                    fresh_firefox_cookies=True,
                    cookies_file=str(cookie_path),
                )

    def test_download_video_raises_sanitized_error_for_direct_call(self):
        class FakeYoutubeDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                raise RuntimeError("proxy failed: http://user:secret@proxy.example:8080 timeout")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                with self.assertRaisesRegex(RuntimeError, "<redacted-proxy>") as context:
                    download_video.download_video(
                        "https://youtube.com/watch?v=Ckt1cj0xjRM",
                        temp_dir,
                        proxy="http://user:secret@proxy.example:8080",
                    )

        self.assertNotIn("secret", str(context.exception))

    def test_download_video_defaults_to_timestamped_youtube_clips_dir(self):
        class FakeYoutubeDL:
            last_options = None

            def __init__(self, options):
                FakeYoutubeDL.last_options = options
                self.output_path = Path(options["outtmpl"].replace("%(id)s", "abc123").replace("%(ext)s", "mp4"))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                if download:
                    self.output_path.write_bytes(b"video")
                return {"id": "abc123", "title": "Example Video", "duration": 123}

            def prepare_filename(self, info):
                return str(self.output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            default_output_dir = Path(temp_dir) / "youtube-clips" / "20260510_120000"
            with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                with patch.object(download_video, "_timestamped_output_dir", return_value=str(default_output_dir)):
                    download_video.download_video("https://youtube.com/watch?v=Ckt1cj0xjRM")

        self.assertEqual(Path(FakeYoutubeDL.last_options["outtmpl"]).parent, default_output_dir)

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

    def test_download_video_does_not_delete_caller_supplied_cookiefile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_path = Path(temp_dir) / "caller-cookies.txt"
            cookie_path.write_text("cookie-data", encoding="utf-8")

            class FakeYoutubeDL:
                def __init__(self, options):
                    self.output_path = Path(options["outtmpl"].replace("%(id)s", "abc123").replace("%(ext)s", "mp4"))
                    self.cookiefile = Path(options["cookiefile"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, url, download=False):
                    if not self.cookiefile.exists():
                        raise AssertionError("caller cookiefile should exist during yt-dlp execution")
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

            with patch.object(download_video, "_create_temp_cookiefile", side_effect=AssertionError("should not create temp cookiefile")):
                with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                    result = download_video.download_video(
                        "https://youtube.com/watch?v=Ckt1cj0xjRM",
                        temp_dir,
                        fresh_firefox_cookies=True,
                        fresh_firefox_cookiefile=str(cookie_path),
                    )

            self.assertEqual(result["video_id"], "abc123")
            self.assertTrue(cookie_path.exists())

    def test_download_video_cleans_up_temp_cookiefile_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cookie_path = Path(temp_dir) / "fresh-firefox-cookies.txt"
            temp_cookie_path.write_text("", encoding="utf-8")

            class FakeYoutubeDL:
                last_options = None

                def __init__(self, options):
                    FakeYoutubeDL.last_options = options
                    self.output_path = Path(options["outtmpl"].replace("%(id)s", "abc123").replace("%(ext)s", "mp4"))
                    self.cookiefile = Path(options["cookiefile"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, url, download=False):
                    if not self.cookiefile.exists():
                        raise AssertionError("temp cookiefile should exist during yt-dlp execution")
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

            with patch.object(download_video, "_create_temp_cookiefile", return_value=temp_cookie_path):
                with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                    result = download_video.download_video(
                        "https://youtube.com/watch?v=Ckt1cj0xjRM",
                        temp_dir,
                        fresh_firefox_cookies=True,
                    )

            self.assertEqual(result["video_id"], "abc123")
            self.assertEqual(FakeYoutubeDL.last_options["cookiesfrombrowser"], ("firefox",))
            self.assertFalse(temp_cookie_path.exists())

    def test_download_video_keeps_temp_cookiefile_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cookie_path = Path(temp_dir) / "fresh-firefox-cookies.txt"
            temp_cookie_path.write_text("", encoding="utf-8")

            class FakeYoutubeDL:
                last_options = None

                def __init__(self, options):
                    FakeYoutubeDL.last_options = options
                    self.output_path = Path(options["outtmpl"].replace("%(id)s", "abc123").replace("%(ext)s", "mp4"))
                    self.cookiefile = Path(options["cookiefile"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, url, download=False):
                    if not self.cookiefile.exists():
                        raise AssertionError("temp cookiefile should exist during yt-dlp execution")
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

            with patch.object(download_video, "_create_temp_cookiefile", return_value=temp_cookie_path):
                with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                    with patch("builtins.print") as mock_print:
                        result = download_video.download_video(
                            "https://youtube.com/watch?v=Ckt1cj0xjRM",
                            temp_dir,
                            fresh_firefox_cookies=True,
                            fresh_firefox_profile="work",
                            keep_temp_cookies=True,
                        )

            warning_output = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
            self.assertIn("保留临时 cookies 文件", warning_output)
            self.assertTrue(temp_cookie_path.exists())
            temp_cookie_path.unlink()

        self.assertEqual(result["video_id"], "abc123")
        self.assertEqual(FakeYoutubeDL.last_options["cookiesfrombrowser"], ("firefox", "work"))

    def test_cleanup_temp_cookiefile_unlink_error_does_not_mask_download_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cookie_path = Path(temp_dir) / "fresh-firefox-cookies.txt"
            temp_cookie_path.write_text("", encoding="utf-8")
            stderr = io.StringIO()

            class FakeYoutubeDL:
                def __init__(self, options):
                    self.cookiefile = Path(options["cookiefile"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, url, download=False):
                    if not self.cookiefile.exists():
                        raise AssertionError("temp cookiefile should exist during yt-dlp execution")
                    raise RuntimeError("boom")

            with patch.object(download_video, "_create_temp_cookiefile", return_value=temp_cookie_path):
                with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                    with patch.object(Path, "unlink", side_effect=PermissionError("access denied")):
                        with patch("sys.stderr", stderr):
                            with self.assertRaisesRegex(RuntimeError, "boom"):
                                download_video.download_video(
                                    "https://youtube.com/watch?v=Ckt1cj0xjRM",
                                    temp_dir,
                                    fresh_firefox_cookies=True,
                                )

            self.assertIn("清理临时 cookies 文件失败", stderr.getvalue())

    def test_download_video_cleans_up_temp_cookiefile_when_download_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cookie_path = Path(temp_dir) / "fresh-firefox-cookies.txt"
            temp_cookie_path.write_text("", encoding="utf-8")

            class FakeYoutubeDL:
                def __init__(self, options):
                    self.cookiefile = Path(options["cookiefile"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, url, download=False):
                    if not self.cookiefile.exists():
                        raise AssertionError("temp cookiefile should exist during yt-dlp execution")
                    raise RuntimeError("boom")

            with patch.object(download_video, "_create_temp_cookiefile", return_value=temp_cookie_path):
                with patch.object(download_video, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL)):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        download_video.download_video(
                            "https://youtube.com/watch?v=Ckt1cj0xjRM",
                            temp_dir,
                            fresh_firefox_cookies=True,
                        )

            self.assertFalse(temp_cookie_path.exists())


if __name__ == "__main__":
    unittest.main()
