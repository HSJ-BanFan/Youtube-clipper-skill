#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translation.models import Cue  # noqa: E402
from translation.subtitles import crop_cues, parse_subtitle, parse_timestamp, write_srt  # noqa: E402


def extract_subtitle_clip(
    subtitle_file: str,
    start_time: str,
    end_time: str,
    output_file: str,
) -> list[Cue]:
    start_seconds = parse_timestamp(start_time)
    end_seconds = parse_timestamp(end_time)

    print("Extracting subtitle clip...")
    print(f"   input: {subtitle_file}")
    print(f"   range: {start_time} - {end_time}")
    print(f"   seconds: {start_seconds:.1f}s - {end_seconds:.1f}s")

    cues = parse_subtitle(subtitle_file)
    cropped = crop_cues(cues, start_seconds, end_seconds)
    write_srt(cropped, output_file)

    print("Subtitle clip complete")
    print(f"   output: {output_file}")
    print(f"   cue_count: {len(cropped)}")
    return cropped


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        print("Usage: python extract_subtitle_clip.py <vtt_or_srt_file> <start_time> <end_time> <output_srt>")
        print("Example: python extract_subtitle_clip.py input.vtt 00:05:47 00:09:19 output.srt")
        return 1

    try:
        extract_subtitle_clip(args[0], args[1], args[2], args[3])
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
