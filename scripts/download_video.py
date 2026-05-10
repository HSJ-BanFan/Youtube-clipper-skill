#!/usr/bin/env python3
"""
下载 YouTube 视频和字幕
使用 yt-dlp 下载视频和英文字幕
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import dotenv_values

try:
    import yt_dlp
except ImportError:
    print("❌ Error: yt-dlp not installed")
    print("Please install: pip install yt-dlp")
    sys.exit(1)

from utils import (
    ensure_directory,
    format_file_size,
    get_video_duration_display,
    validate_url,
)


class DownloadSettings(TypedDict):
    cookies_from_browser: str | None
    cookies_file: str | None
    fresh_firefox_cookies: bool
    fresh_firefox_profile: str | None
    fresh_firefox_cookiefile: str | None
    keep_temp_cookies: bool
    proxy: str | None
    rate_limit: str | None
    max_video_height: str
    output_dir: str


class DownloadResult(TypedDict):
    video_path: str
    subtitle_path: str | None
    title: str
    duration: int
    file_size: int
    video_id: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YouTube video, subtitles, and metadata.")
    parser.add_argument("youtube_url", help="YouTube URL")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("--cookies-from-browser", dest="cookies_from_browser", help="Browser source for yt-dlp cookies, e.g. firefox")
    parser.add_argument("--cookies-file", dest="cookies_file", help="Netscape cookies.txt file path for yt-dlp")
    parser.add_argument("--fresh-firefox-cookies", action="store_true", help="Export fresh Firefox cookies into a temporary cookies.txt for this download")
    parser.add_argument("--fresh-firefox-profile", dest="fresh_firefox_profile", help="Firefox profile for fresh cookie export")
    parser.add_argument("--keep-temp-cookies", action="store_true", help="Keep temporary fresh Firefox cookies.txt after download")
    parser.add_argument("--proxy", help="Proxy URL for yt-dlp")
    parser.add_argument("--rate-limit", dest="rate_limit", help="Download rate limit for yt-dlp, e.g. 50K or 4.2M")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    return parser.parse_args(argv)


def _load_env_values(env_file: str) -> dict[str, str]:
    return {key: value for key, value in dotenv_values(env_file).items() if value is not None}


def _resolve_cli_or_env(cli_value: str | None, env_name: str, env_values: dict[str, str]) -> str | None:
    if cli_value:
        return cli_value
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    file_value = env_values.get(env_name)
    return file_value or None


def _resolve_bool_cli_or_env(cli_value: bool, env_name: str, env_values: dict[str, str]) -> bool:
    if cli_value:
        return True
    env_value = os.getenv(env_name)
    if env_value is None:
        env_value = env_values.get(env_name)
    if env_value is None:
        return False
    return env_value.strip().lower() in {"1", "true", "yes", "on"}


def _create_temp_cookiefile() -> Path:
    descriptor, cookiefile_path = tempfile.mkstemp(suffix=".txt", prefix="yt-dlp-fresh-cookies-")
    os.close(descriptor)
    try:
        os.chmod(cookiefile_path, 0o600)
    except OSError:
        pass
    return Path(cookiefile_path)


def _resolve_max_video_height(env_values: dict[str, str] | None = None) -> str:
    if env_values is None:
        env_values = {}
    height = os.getenv("MAX_VIDEO_HEIGHT")
    if height is None:
        height = env_values.get("MAX_VIDEO_HEIGHT")
    if height is None or not height.strip():
        return "1080"
    height = height.strip()
    if not height.isdigit():
        raise ValueError(f"MAX_VIDEO_HEIGHT must be an integer: {height}")
    return height


def _parse_browser_spec(spec: str) -> tuple[str, ...]:
    parts = tuple(part for part in spec.split(":") if part)
    if not parts:
        raise ValueError("cookies-from-browser must not be empty")
    return parts


def _normalize_output_dir(path_value: str | Path) -> str:
    return str(Path(str(path_value).strip()).expanduser())


def _timestamped_output_dir(base_dir: str | Path) -> str:
    return str(Path(_normalize_output_dir(base_dir)) / datetime.now().strftime("%Y%m%d_%H%M%S"))


def _resolve_output_dir(cli_output_dir: str | None, env_values: dict[str, str] | None = None) -> str:
    if cli_output_dir:
        return _normalize_output_dir(cli_output_dir)
    if env_values is None:
        env_values = {}
    env_output_dir = os.getenv("OUTPUT_DIR")
    if env_output_dir is None:
        env_output_dir = env_values.get("OUTPUT_DIR")
    if env_output_dir and env_output_dir.strip():
        return _timestamped_output_dir(env_output_dir)
    return _timestamped_output_dir(Path("youtube-clips"))


def _cleanup_temp_cookiefile(cookiefile: Path | None, keep_temp_cookies: bool) -> None:
    if cookiefile is None:
        return
    if keep_temp_cookies:
        print("[WARN] 保留临时 cookies 文件", file=sys.stderr)
        return
    try:
        cookiefile.unlink(missing_ok=True)
    except OSError:
        print("[WARN] 清理临时 cookies 文件失败", file=sys.stderr)


def _validate_cookie_sources(
    cookies_from_browser: str | None,
    cookies_file: str | None,
    fresh_firefox_cookies: bool,
) -> None:
    if cookies_from_browser and cookies_file:
        raise ValueError("--cookies-from-browser and --cookies-file are mutually exclusive")
    if fresh_firefox_cookies and (cookies_from_browser or cookies_file):
        raise ValueError("--fresh-firefox-cookies is mutually exclusive with --cookies-from-browser and --cookies-file")


def resolve_download_settings(
    args: argparse.Namespace,
    env_values: dict[str, str] | None = None,
) -> DownloadSettings:
    if env_values is None:
        env_values = _load_env_values(args.env_file)

    cookies_from_browser = _resolve_cli_or_env(args.cookies_from_browser, "YT_DLP_COOKIES_FROM_BROWSER", env_values)
    cookies_file = _resolve_cli_or_env(args.cookies_file, "YT_DLP_COOKIES_FILE", env_values)
    fresh_firefox_cookies = _resolve_bool_cli_or_env(args.fresh_firefox_cookies, "YT_DLP_FRESH_FIREFOX_COOKIES", env_values)
    fresh_firefox_profile = _resolve_cli_or_env(args.fresh_firefox_profile, "YT_DLP_FRESH_FIREFOX_PROFILE", env_values)
    keep_temp_cookies = _resolve_bool_cli_or_env(args.keep_temp_cookies, "YT_DLP_KEEP_TEMP_COOKIES", env_values)
    proxy = _resolve_cli_or_env(args.proxy, "YT_DLP_PROXY", env_values)
    rate_limit = _resolve_cli_or_env(args.rate_limit, "YT_DLP_RATE_LIMIT", env_values)
    max_video_height = _resolve_max_video_height(env_values)
    output_dir = _resolve_output_dir(args.output_dir, env_values)

    if fresh_firefox_profile and not fresh_firefox_cookies:
        fresh_firefox_cookies = True
    _validate_cookie_sources(cookies_from_browser, cookies_file, fresh_firefox_cookies)

    resolved_cookie_file = None
    if cookies_file:
        cookie_path = Path(cookies_file).expanduser().resolve()
        if not cookie_path.is_file():
            raise FileNotFoundError(f"Cookies file not found: {cookie_path}")
        resolved_cookie_file = str(cookie_path)

    return {
        "cookies_from_browser": cookies_from_browser,
        "cookies_file": resolved_cookie_file,
        "fresh_firefox_cookies": fresh_firefox_cookies,
        "fresh_firefox_profile": fresh_firefox_profile,
        "fresh_firefox_cookiefile": None,
        "keep_temp_cookies": keep_temp_cookies,
        "proxy": proxy,
        "rate_limit": rate_limit,
        "max_video_height": max_video_height,
        "output_dir": output_dir,
    }


def _sanitize_error_message(message: str, proxy: str | None) -> str:
    sanitized = message
    if proxy:
        sanitized = sanitized.replace(proxy, "<redacted-proxy>")
    sanitized = re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^/\s]+@", r"\1<redacted-proxy>@", sanitized)
    return re.sub(r"(?<![\w/:])[^\s/@:]+:[^\s/]+@", "<redacted-userinfo>@", sanitized)


def build_ydl_opts(output_dir: Path, settings: DownloadSettings) -> dict[str, object]:
    uses_fresh_firefox_cookies = settings["fresh_firefox_cookies"] or settings["fresh_firefox_profile"] is not None
    _validate_cookie_sources(settings["cookies_from_browser"], settings["cookies_file"], uses_fresh_firefox_cookies)
    max_video_height = settings["max_video_height"]
    ydl_opts = {
        "format": f"bestvideo[height<={max_video_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_video_height}][ext=mp4]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "writethumbnail": False,
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [_progress_hook],
    }

    cookies_from_browser = settings["cookies_from_browser"]
    cookies_file = settings["cookies_file"]
    fresh_firefox_cookies = settings["fresh_firefox_cookies"]
    fresh_firefox_profile = settings["fresh_firefox_profile"]
    fresh_firefox_cookiefile = settings["fresh_firefox_cookiefile"]
    proxy = settings["proxy"]
    rate_limit = settings["rate_limit"]

    if fresh_firefox_cookies:
        browser_spec = ("firefox", fresh_firefox_profile) if fresh_firefox_profile else ("firefox",)
        ydl_opts["cookiesfrombrowser"] = browser_spec
        if fresh_firefox_cookiefile:
            ydl_opts["cookiefile"] = fresh_firefox_cookiefile
    else:
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = _parse_browser_spec(cookies_from_browser)
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file

    if proxy:
        ydl_opts["proxy"] = proxy
    if rate_limit:
        ydl_opts["ratelimit"] = rate_limit

    return ydl_opts


def download_video(
    url: str,
    output_dir: str | None = None,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    fresh_firefox_cookies: bool = False,
    fresh_firefox_profile: str | None = None,
    fresh_firefox_cookiefile: str | None = None,
    keep_temp_cookies: bool = False,
    proxy: str | None = None,
    rate_limit: str | None = None,
    max_video_height: str = "1080",
) -> DownloadResult:
    if not validate_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")

    uses_fresh_firefox_cookies = fresh_firefox_cookies or fresh_firefox_profile is not None
    _validate_cookie_sources(cookies_from_browser, cookies_file, uses_fresh_firefox_cookies)

    if output_dir is None:
        resolved_output_dir = Path(_resolve_output_dir(None))
    else:
        resolved_output_dir = Path(_normalize_output_dir(output_dir))

    resolved_output_dir = ensure_directory(resolved_output_dir)

    print("[INFO] 开始下载视频...")
    print(f"   URL: {url}")
    print(f"   输出目录: {resolved_output_dir}")

    resolved_fresh_cookiefile = fresh_firefox_cookiefile
    created_temp_cookiefile: Path | None = None
    if uses_fresh_firefox_cookies and resolved_fresh_cookiefile is None:
        created_temp_cookiefile = _create_temp_cookiefile()
        resolved_fresh_cookiefile = str(created_temp_cookiefile)
    settings: DownloadSettings = {
        "cookies_from_browser": cookies_from_browser,
        "cookies_file": cookies_file,
        "fresh_firefox_cookies": uses_fresh_firefox_cookies,
        "fresh_firefox_profile": fresh_firefox_profile,
        "fresh_firefox_cookiefile": resolved_fresh_cookiefile,
        "keep_temp_cookies": keep_temp_cookies,
        "proxy": proxy,
        "rate_limit": rate_limit,
        "max_video_height": max_video_height,
        "output_dir": str(resolved_output_dir),
    }
    ydl_opts = build_ydl_opts(resolved_output_dir, settings)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("\n[INFO] 获取视频信息...")
            info = ydl.extract_info(url, download=False)

            title = info.get("title", "Unknown")
            duration = info.get("duration", 0)
            video_id = info.get("id", "unknown")

            print(f"   标题: {title}")
            print(f"   时长: {get_video_duration_display(duration)}")
            print(f"   视频ID: {video_id}")

            print("\n[INFO] 开始下载...")
            info = ydl.extract_info(url, download=True)

            video_filename = ydl.prepare_filename(info)
            video_path = Path(video_filename)

            subtitle_path = None
            for potential_sub in (video_path.with_suffix(".en.vtt"), video_path.with_suffix(".vtt")):
                if potential_sub.exists():
                    subtitle_path = potential_sub
                    break

            file_size = video_path.stat().st_size if video_path.exists() else 0

            if not video_path.exists():
                raise Exception("Video file not found after download")

            print(f"\n[OK] 视频下载完成: {video_path.name}")
            print(f"   大小: {format_file_size(file_size)}")

            if subtitle_path and subtitle_path.exists():
                print(f"[OK] 字幕下载完成: {subtitle_path.name}")
            else:
                print("[WARN] 未找到英文字幕")
                print("   提示：某些视频可能没有字幕或需要自动生成")

            return {
                "video_path": str(video_path),
                "subtitle_path": str(subtitle_path) if subtitle_path else None,
                "title": title,
                "duration": duration,
                "file_size": file_size,
                "video_id": video_id,
            }
    except Exception as exc:
        raise RuntimeError(_sanitize_error_message(str(exc), proxy)) from exc
    finally:
        _cleanup_temp_cookiefile(created_temp_cookiefile, keep_temp_cookies)


def _progress_hook(progress: dict) -> None:
    if progress["status"] == "downloading":
        if "downloaded_bytes" in progress and "total_bytes" in progress and progress["total_bytes"]:
            percent = progress["downloaded_bytes"] / progress["total_bytes"] * 100
            downloaded = format_file_size(progress["downloaded_bytes"])
            total = format_file_size(progress["total_bytes"])
            speed = progress.get("speed", 0)
            speed_str = format_file_size(speed) + "/s" if speed else "N/A"

            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = "#" * filled + "-" * (bar_length - filled)

            print(f"\r   [{bar}] {percent:.1f}% - {downloaded}/{total} - {speed_str}", end="", flush=True)
        elif "downloaded_bytes" in progress:
            downloaded = format_file_size(progress["downloaded_bytes"])
            speed = progress.get("speed", 0)
            speed_str = format_file_size(speed) + "/s" if speed else "N/A"
            print(f"\r   下载中... {downloaded} - {speed_str}", end="", flush=True)
    elif progress["status"] == "finished":
        print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_values = _load_env_values(args.env_file)
    settings: DownloadSettings | None = None

    try:
        settings = resolve_download_settings(args, env_values)
        result = download_video(
            args.youtube_url,
            settings["output_dir"],
            cookies_from_browser=settings["cookies_from_browser"],
            cookies_file=settings["cookies_file"],
            fresh_firefox_cookies=settings["fresh_firefox_cookies"],
            fresh_firefox_profile=settings["fresh_firefox_profile"],
            fresh_firefox_cookiefile=settings["fresh_firefox_cookiefile"],
            keep_temp_cookies=settings["keep_temp_cookies"],
            proxy=settings["proxy"],
            rate_limit=settings["rate_limit"],
            max_video_height=settings["max_video_height"],
        )
        print("\n" + "=" * 60)
        print("下载结果 (JSON):")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        resolved_proxy = settings["proxy"] if settings is not None else _resolve_cli_or_env(args.proxy, "YT_DLP_PROXY", env_values)
        sanitized_message = _sanitize_error_message(str(exc), resolved_proxy)
        print(f"\n[ERROR] 错误: {sanitized_message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
