from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import SupportsFloat

from translation.models import Cue

_TIMESTAMP_RE = r"(?:\d{2}:)?\d{2}:\d{2}[,.]\d{3}"
_TIMING_RE = re.compile(
    rf"^(?P<start>{_TIMESTAMP_RE})\s+-->\s+"
    rf"(?P<end>{_TIMESTAMP_RE})(?P<settings>.*)$"
)
_SKIP_VTT_BLOCK_PREFIXES = ("NOTE", "STYLE", "REGION")


def parse_timestamp(value: str | SupportsFloat) -> float:
    """Parse SRT/VTT timestamps or numeric seconds into seconds."""
    if not isinstance(value, str):
        return float(value)

    normalized = value.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return float(normalized)


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    total_millis = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def detect_subtitle_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return "srt"
    if suffix == ".vtt":
        return "vtt"
    raise ValueError("subtitle_path must point to .srt or .vtt")


def parse_subtitle(path: str | Path) -> list[Cue]:
    """Parse an SRT or VTT file into normalized cues."""
    subtitle_path = Path(path)
    subtitle_format = detect_subtitle_format(subtitle_path)
    content = subtitle_path.read_text(encoding="utf-8-sig")
    if subtitle_format == "srt":
        return parse_srt(content)
    return parse_vtt(content)


def parse_subtitle_file(path: Path) -> list[Cue]:
    return parse_subtitle(path)


def parse_srt(content: str) -> list[Cue]:
    cues = _parse_blocks(content, is_vtt=False)
    validate_cues(cues)
    return cues


def parse_vtt(content: str) -> list[Cue]:
    lines = _normalize_lines(content)
    if lines and lines[0].lstrip("﻿").startswith("WEBVTT"):
        lines = lines[1:]
    lines = _strip_vtt_header_metadata(lines)
    lines = _collapse_blank_after_timing(lines)
    cues = _parse_blocks("\n".join(lines), is_vtt=True)
    validate_cues(cues)
    return cues


def validate_cues(cues: list[Cue]) -> None:
    if not cues:
        raise ValueError("subtitle file contains no cues")

    seen_ids: set[str] = set()
    for expected_index, cue in enumerate(cues, start=1):
        if cue.index != expected_index:
            raise ValueError(f"cue index must be {expected_index}, got {cue.index}")
        if not cue.id.strip():
            raise ValueError(f"cue index {cue.index} id is empty")
        if cue.id in seen_ids:
            raise ValueError(f"duplicate cue id: {cue.id}")
        seen_ids.add(cue.id)
        if not cue.start.strip():
            raise ValueError(f"cue id {cue.id} start is empty")
        if not cue.end.strip():
            raise ValueError(f"cue id {cue.id} end is empty")
        if not cue.source.strip():
            raise ValueError(f"cue id {cue.id} source is empty")


def validate_translations(cues: list[Cue], translations: Mapping[str, str]) -> None:
    validate_cues(cues)
    if len(translations) != len(cues):
        raise ValueError(f"translation count {len(translations)} does not match cue count {len(cues)}")

    for cue in cues:
        if cue.id not in translations:
            raise ValueError(f"missing translation for cue id {cue.id}")
        if not translations[cue.id].strip():
            raise ValueError(f"translation for cue id {cue.id} is empty")


def crop_cues(
    cues: list[Cue],
    start: str | SupportsFloat,
    end: str | SupportsFloat,
    mode: str = "overlap",
    rebase: bool = True,
) -> list[Cue]:
    """Keep overlapping cues, clamp to clip bounds, and optionally rebase to zero."""
    if mode != "overlap":
        raise ValueError("crop mode must be overlap")

    clip_start = parse_timestamp(start)
    clip_end = parse_timestamp(end)
    if clip_start >= clip_end:
        raise ValueError("clip start must be before clip end")

    cropped: list[Cue] = []
    for cue in cues:
        cue_start = parse_timestamp(cue.start)
        cue_end = parse_timestamp(cue.end)
        if cue_end <= clip_start or cue_start >= clip_end:
            continue

        new_start = max(cue_start, clip_start)
        new_end = min(cue_end, clip_end)
        if rebase:
            new_start -= clip_start
            new_end -= clip_start

        index = len(cropped) + 1
        cropped.append(
            Cue(
                id=str(index),
                index=index,
                start=format_timestamp(new_start),
                end=format_timestamp(new_end),
                source=cue.source,
                raw_timing=f"{format_timestamp(new_start)} --> {format_timestamp(new_end)}",
                note=cue.note,
            )
        )
    return cropped


def write_srt(cues: list[Cue], path: str | Path) -> None:
    """Write cues as source-text SRT blocks."""
    _write_srt_blocks(cues, Path(path), lambda cue: cue.source)


def write_translated_srt(cues: list[Cue], translations: Mapping[str, str], path: Path) -> None:
    validate_translations(cues, translations)
    _write_srt_blocks(cues, path, lambda cue: translations[cue.id])


def write_bilingual_srt(
    cues: list[Cue],
    translations: Mapping[str, str],
    path: Path,
    source_first: bool = False,
) -> None:
    validate_translations(cues, translations)

    def text_for(cue: Cue) -> str:
        if source_first:
            return f"{cue.source}\n{translations[cue.id]}"
        return f"{translations[cue.id]}\n{cue.source}"

    _write_srt_blocks(cues, path, text_for)


def _parse_blocks(content: str, is_vtt: bool) -> list[Cue]:
    blocks = _split_blocks(content)
    cues: list[Cue] = []
    for block in blocks:
        if is_vtt and _is_skipped_vtt_block(block):
            continue
        cue = _parse_block(block, index=len(cues) + 1, is_vtt=is_vtt)
        if cue is None:
            snippet = _format_block_snippet(block)
            raise ValueError(f"invalid subtitle cue block near cue index {len(cues) + 1}: {snippet}")
        cues.append(cue)
    return cues


def _split_blocks(content: str) -> list[list[str]]:
    lines = _normalize_lines(content)
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _normalize_lines(content: str) -> list[str]:
    return content.replace("\r\n", "\n").replace("\r", "\n").split("\n")


_VTT_HEADER_METADATA_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*\s*:")


def _strip_vtt_header_metadata(lines: list[str]) -> list[str]:
    """Drop VTT header metadata lines (Kind:, Language:, etc.) before the first cue.

    Per the WebVTT spec, the optional header section after WEBVTT contains key:value
    metadata terminated by a blank line. YouTube auto-sub VTT always emits Kind:
    and Language: which the cue parser cannot interpret.
    """
    result: list[str] = []
    in_header = True
    for line in lines:
        if in_header:
            stripped = line.strip()
            if stripped == "":
                in_header = False
                result.append(line)
                continue
            if _TIMING_RE.match(stripped):
                in_header = False
                result.append(line)
                continue
            if _VTT_HEADER_METADATA_RE.match(stripped):
                continue
            in_header = False
            result.append(line)
            continue
        result.append(line)
    return result


def _collapse_blank_after_timing(lines: list[str]) -> list[str]:
    """Drop whitespace-only lines that immediately follow a VTT timing line.

    YouTube auto-sub VTT places a blank/space-only line between the timing line
    and the cue body to act as a "previous text" placeholder. The block splitter
    treats this as a block boundary, so we collapse it before splitting.
    """
    result: list[str] = []
    just_saw_timing = False
    for line in lines:
        if just_saw_timing and line.strip() == "":
            continue
        result.append(line)
        just_saw_timing = bool(_TIMING_RE.match(line.strip()))
    return result


def _format_block_snippet(block: list[str]) -> str:
    return " | ".join(line.strip() for line in block[:3])


def _is_skipped_vtt_block(block: list[str]) -> bool:
    first = block[0].strip()
    return any(first == prefix or first.startswith(f"{prefix} ") for prefix in _SKIP_VTT_BLOCK_PREFIXES)


def _parse_block(block: list[str], index: int, is_vtt: bool) -> Cue | None:
    timing_line_index = _find_timing_line_index(block)
    if timing_line_index is None:
        return None

    cue_id = _cue_id_for_block(block, timing_line_index, index, is_vtt)
    raw_timing = block[timing_line_index].strip()
    match = _TIMING_RE.match(raw_timing)
    if match is None:
        return None

    source_lines = block[timing_line_index + 1 :]
    source = "\n".join(source_lines).strip()
    return Cue(
        id=cue_id,
        index=index,
        start=_normalize_timestamp(match.group("start")),
        end=_normalize_timestamp(match.group("end")),
        source=source,
        raw_timing=raw_timing,
    )


def _find_timing_line_index(block: list[str]) -> int | None:
    for index, line in enumerate(block):
        if _TIMING_RE.match(line.strip()):
            return index
    return None


def _cue_id_for_block(block: list[str], timing_line_index: int, index: int, is_vtt: bool) -> str:
    if timing_line_index == 0:
        return str(index)
    candidate = block[timing_line_index - 1].strip()
    if not candidate:
        return str(index)
    if is_vtt:
        return candidate
    if timing_line_index == 1 and candidate.isdigit():
        return candidate
    return str(index)


def _normalize_timestamp(value: str) -> str:
    return format_timestamp(parse_timestamp(value))


def _write_srt_blocks(cues: list[Cue], path: Path, text_for: Callable[[Cue], str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for cue in cues:
        blocks.append(f"{cue.index}\n{cue.start} --> {cue.end}\n{text_for(cue)}\n\n")
    path.write_text("".join(blocks), encoding="utf-8")
