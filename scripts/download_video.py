#!/usr/bin/env python3
"""
下载 YouTube 视频和字幕
使用 yt-dlp 下载视频和英文字幕
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YouTube video, subtitles, and metadata.")
    parser.add_argument("youtube_url", help="YouTube URL")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("--cookies-from-browser", dest="cookies_from_browser", help="Browser source for yt-dlp cookies, e.g. firefox")
    parser.add_argument("--cookies-file", dest="cookies_file", help="Netscape cookies.txt file path for yt-dlp")
    parser.add_argument("--proxy", help="Proxy URL for yt-dlp")
    parser.add_argument("--rate-limit", dest="rate_limit", help="Download rate limit for yt-dlp, e.g. 50K or 4.2M")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    return parser.parse_args(argv)


def _resolve_cli_or_env(cli_value: str | None, env_name: str) -> str | None:
    if cli_value:
        return cli_value
    env_value = os.getenv(env_name)
    return env_value or None


def _resolve_max_video_height() -> str:
    height = os.getenv("MAX_VIDEO_HEIGHT", "1080")
    if not height.isdigit():
        raise ValueError("MAX_VIDEO_HEIGHT must be an integer")
    return height


def _parse_browser_spec(spec: str) -> tuple[str, ...]:
    parts = tuple(part for part in spec.split(":") if part)
    if not parts:
        raise ValueError("cookies-from-browser must not be empty")
    return parts


def resolve_download_settings(args: argparse.Namespace) -> dict[str, str | None]:
    load_dotenv(args.env_file, override=False)

    cookies_from_browser = _resolve_cli_or_env(args.cookies_from_browser, "YT_DLP_COOKIES_FROM_BROWSER")
    cookies_file = _resolve_cli_or_env(args.cookies_file, "YT_DLP_COOKIES_FILE")
    proxy = _resolve_cli_or_env(args.proxy, "YT_DLP_PROXY")
    rate_limit = _resolve_cli_or_env(args.rate_limit, "YT_DLP_RATE_LIMIT")
    max_video_height = _resolve_max_video_height()

    if cookies_from_browser and cookies_file:
        raise ValueError("--cookies-from-browser and --cookies-file are mutually exclusive")

    resolved_cookie_file = None
    if cookies_file:
        cookie_path = Path(cookies_file).expanduser().resolve()
        if not cookie_path.is_file():
            raise FileNotFoundError(f"Cookies file not found: {cookie_path}")
        resolved_cookie_file = str(cookie_path)

    return {
        "cookies_from_browser": cookies_from_browser,
        "cookies_file": resolved_cookie_file,
        "proxy": proxy,
        "rate_limit": rate_limit,
        "max_video_height": max_video_height,
    }


def _sanitize_error_message(message: str, proxy: str | None) -> str:
    if not proxy:
        return message
    return message.replace(proxy, "<redacted-proxy>")


def build_ydl_opts(output_dir: Path, settings: dict[str, str | None]) -> dict:
    max_video_height = settings["max_video_height"] or "1080"
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

    cookies_from_browser = settings.get("cookies_from_browser")
    cookies_file = settings.get("cookies_file")
    proxy = settings.get("proxy")
    rate_limit = settings.get("rate_limit")

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
    proxy: str | None = None,
    rate_limit: str | None = None,
    max_video_height: str = "1080",
) -> dict:
    if not validate_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")

    if output_dir is None:
        resolved_output_dir = Path.cwd()
    else:
        resolved_output_dir = Path(output_dir)

    resolved_output_dir = ensure_directory(resolved_output_dir)

    print("[INFO] 开始下载视频...")
    print(f"   URL: {url}")
    print(f"   输出目录: {resolved_output_dir}")

    settings = {
        "cookies_from_browser": cookies_from_browser,
        "cookies_file": cookies_file,
        "proxy": proxy,
        "rate_limit": rate_limit,
        "max_video_height": max_video_height,
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
            subtitle_exts = [".en.vtt", ".vtt"]
            for ext in subtitle_exts:
                potential_sub = video_path.with_suffix(ext)
                if not potential_sub.exists():
                    stem = video_path.stem
                    potential_sub = video_path.parent / f"{stem}.en.vtt"

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
        sanitized_message = _sanitize_error_message(str(exc), proxy)
        print(f"\n[ERROR] 下载失败: {sanitized_message}")
        raise


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

    try:
        settings = resolve_download_settings(args)
        result = download_video(
            args.youtube_url,
            args.output_dir,
            cookies_from_browser=settings["cookies_from_browser"],
            cookies_file=settings["cookies_file"],
            proxy=settings["proxy"],
            rate_limit=settings["rate_limit"],
            max_video_height=settings["max_video_height"] or "1080",
        )
        print("\n" + "=" * 60)
        print("下载结果 (JSON):")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        sanitized_message = _sanitize_error_message(str(exc), getattr(args, "proxy", None) or os.getenv("YT_DLP_PROXY"))
        print(f"\n[ERROR] 错误: {sanitized_message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
