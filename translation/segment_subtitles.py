from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from translation.segmentation import (
    SegmentationOptions,
    SegmentationResult,
    SegmentationValidationError,
    SubtitleSegmentationSource,
    segment_subtitles,
)


ARTIFACT_NAMES = {
    "segmented_source": "segmented_source.srt",
    "translation_units": "translation_units.json",
    "cue_map": "cue_map.json",
    "report": "segmentation_report.md",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cycle 1 subtitle segmentation and write debug artifacts.")
    parser.add_argument("--mode", required=True, choices=["full_vtt_window", "single_file"])
    parser.add_argument("--full-vtt", dest="full_vtt_path", help="Path to preserved full raw VTT")
    parser.add_argument("--subtitle", dest="subtitle_path", help="Path to SRT/VTT subtitle file")
    parser.add_argument("--clip-start-ms", type=int, help="Clip start in milliseconds")
    parser.add_argument("--clip-end-ms", type=int, help="Clip end in milliseconds")
    parser.add_argument("--padding-before-ms", type=int, default=0)
    parser.add_argument("--padding-after-ms", type=int, default=0)
    parser.add_argument("--output-dir", help="Directory for segmentation artifacts")
    parser.add_argument("--max-unit-chars", type=int, default=180)
    parser.add_argument("--max-unit-duration-ms", type=int, default=7000)
    parser.add_argument("--max-source-cues", type=int, default=4)
    parser.add_argument("--max-sentences", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source = SubtitleSegmentationSource(
            mode=args.mode,
            full_vtt_path=Path(args.full_vtt_path) if args.full_vtt_path else None,
            subtitle_path=Path(args.subtitle_path) if args.subtitle_path else None,
            clip_start_ms=args.clip_start_ms,
            clip_end_ms=args.clip_end_ms,
            padding_before_ms=args.padding_before_ms,
            padding_after_ms=args.padding_after_ms,
        )
        options = SegmentationOptions(
            max_unit_chars=args.max_unit_chars,
            max_unit_duration_ms=args.max_unit_duration_ms,
            max_source_cues=args.max_source_cues,
            max_sentences=args.max_sentences,
        )
        result = segment_subtitles(source, options)
        output_dir = _resolve_output_dir(args.output_dir)
    except (FileNotFoundError, ValueError, SegmentationValidationError) as exc:
        print(f"SegmentationValidationError: {exc}" if isinstance(exc, SegmentationValidationError) else f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ARTIFACT_NAMES["segmented_source"]).write_text(result.to_segmented_srt_text(), encoding="utf-8")
    (output_dir / ARTIFACT_NAMES["translation_units"]).write_text(
        json.dumps(result.to_translation_units_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / ARTIFACT_NAMES["cue_map"]).write_text(
        json.dumps(result.to_cue_map_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / ARTIFACT_NAMES["report"]).write_text(result.to_report_markdown(), encoding="utf-8")
    _print_summary(result, output_dir)
    return 0


def _resolve_output_dir(explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir)
    env_output_dir = os.getenv("YOUTUBE_CLIPS_OUTPUT_DIR") or None
    if env_output_dir:
        return Path(env_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("youtube-clips") / f"segmentation-{timestamp}"


def _print_summary(result: SegmentationResult, output_dir: Path) -> None:
    print(f"mode: {result.source.mode}")
    print(f"input_format: {result.input_format}")
    print(f"cues: {result.stats.extracted_cue_count}")
    print(f"units: {result.stats.translation_unit_count}")
    print("outputs:")
    for name in ARTIFACT_NAMES.values():
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    raise SystemExit(main())
