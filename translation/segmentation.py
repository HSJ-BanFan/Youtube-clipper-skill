from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from translation.models import Cue
from translation.subtitles import detect_subtitle_format, format_timestamp, parse_subtitle_file, parse_timestamp


SCHEMA_VERSION = "segmentation.v1"
LOCAL_DUPLICATE_THRESHOLD_MS = 150
MAX_ROLLING_OVERLAP_TOKENS = 20
DANGLING_CONTINUATION_CAP_TOKENS = 4
DEFAULT_STRONG_BOUNDARY_PHRASES = (
    "now we're going to move on",
    "next project",
    "all right",
    "okay",
    "so now",
    "in this project",
    "let's create",
    "let's move on",
    "this next project",
)
DEFAULT_DANGLING_END_TOKENS = (
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "with",
    "and",
    "or",
    "but",
    "so",
    "because",
    "that",
    "which",
    "who",
    "where",
    "when",
    "if",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "we're",
    "you're",
    "it's",
    "i'm",
)
TIMESTAMP_TAG_RE = re.compile(r"^(?:\d{2}:)?\d{2}:\d{2}\.\d{3}$")
ANGLE_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\S+")

SubtitleSegmentationMode = Literal["full_vtt_window", "single_file"]
TimingSource = Literal["vtt_inline", "cue_proportional", "unknown"]
WarningSeverity = Literal["info", "warning"]


class SegmentBoundaryType(str, Enum):
    INSIDE_CLIP = "inside_clip"
    CROSSES_CLIP_START = "crosses_clip_start"
    CROSSES_CLIP_END = "crosses_clip_end"
    PADDING_ONLY = "padding_only"


class SegmentationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SubtitleSegmentationSource:
    mode: SubtitleSegmentationMode
    full_vtt_path: Path | None = None
    subtitle_path: Path | None = None
    clip_start_ms: int | None = None
    clip_end_ms: int | None = None
    padding_before_ms: int = 0
    padding_after_ms: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"full_vtt_window", "single_file"}:
            raise ValueError("mode must be full_vtt_window or single_file")
        if self.padding_before_ms < 0 or self.padding_after_ms < 0:
            raise ValueError("padding must be >= 0")
        if self.mode == "full_vtt_window":
            if self.full_vtt_path is None:
                raise ValueError("full_vtt_path is required for full_vtt_window")
            if self.clip_start_ms is None:
                raise ValueError("clip_start_ms is required for full_vtt_window")
            if self.clip_end_ms is None:
                raise ValueError("clip_end_ms is required for full_vtt_window")
        if self.mode == "single_file" and self.subtitle_path is None:
            raise ValueError("subtitle_path is required for single_file")
        if self.clip_start_ms is not None and self.clip_end_ms is not None and self.clip_start_ms >= self.clip_end_ms:
            raise ValueError("clip_start_ms must be before clip_end_ms")


@dataclass(frozen=True)
class SegmentationOptions:
    max_unit_chars: int = 180
    max_unit_duration_ms: int = 7_000
    max_source_cues: int = 4
    max_sentences: int = 2
    strong_boundary_phrases: tuple[str, ...] = DEFAULT_STRONG_BOUNDARY_PHRASES
    dangling_end_tokens: tuple[str, ...] = DEFAULT_DANGLING_END_TOKENS
    drop_duplicate_cues: bool = True
    remove_rolling_overlap: bool = True
    segmentation_strategy_version: str = "cycle1-rules-v1"
    timing_strategy_version: str = "cycle1-timing-v1"

    def __post_init__(self) -> None:
        if self.max_unit_chars <= 0:
            raise ValueError("max_unit_chars must be > 0")
        if self.max_unit_duration_ms <= 0:
            raise ValueError("max_unit_duration_ms must be > 0")
        if self.max_source_cues <= 0:
            raise ValueError("max_source_cues must be > 0")
        if self.max_sentences <= 0:
            raise ValueError("max_sentences must be > 0")


@dataclass(frozen=True)
class SourceCue:
    cue_id: str
    index: int
    start_ms: int
    end_ms: int
    text: str
    raw_timing: str | None
    source_kind: str
    intersects_clip: bool
    is_padding_only: bool
    crosses_clip_start: bool
    crosses_clip_end: bool


@dataclass(frozen=True)
class SourceToken:
    token_id: int
    cue_id: str
    cue_index: int
    token_index_in_cue: int
    text: str
    normalized_text: str
    start_ms: int
    end_ms: int
    timing_source: TimingSource
    intersects_clip: bool
    is_padding_only: bool
    is_removed: bool = False
    removal_reason: str | None = None


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    cue_id: str
    cue_token_start: int
    cue_token_end: int


@dataclass(frozen=True)
class SegmentUnit:
    unit_id: str
    source_text: str
    start_ms: int
    end_ms: int
    source_spans: tuple[SourceSpan, ...]
    boundary_type: SegmentBoundaryType
    is_padding_only: bool
    crosses_clip_start: bool
    crosses_clip_end: bool
    intersects_clip: bool

    @property
    def source_cue_ids(self) -> tuple[str, ...]:
        return _derive_source_cue_ids(self.source_spans)


@dataclass(frozen=True)
class SegmentationWarning:
    code: str
    message: str
    severity: WarningSeverity
    unit_id: str | None = None
    cue_id: str | None = None
    token_id: int | None = None


@dataclass(frozen=True)
class SegmentationStats:
    input_format: str
    source_mode: str
    original_cue_count: int
    extracted_cue_count: int
    cleaned_active_token_count: int
    removed_duplicate_cue_count: int
    removed_duplicate_token_count: int
    removed_rolling_overlap_token_count: int
    translation_unit_count: int
    inline_timing_token_count: int
    proportional_timing_token_count: int
    unknown_timing_token_count: int
    forced_split_count: int
    padding_only_unit_count: int
    warning_count: int


@dataclass(frozen=True)
class SegmentationResult:
    source: SubtitleSegmentationSource
    options: SegmentationOptions
    input_format: str
    processing_start_ms: int | None
    processing_end_ms: int | None
    clip_start_ms: int | None
    clip_end_ms: int | None
    source_cues: tuple[SourceCue, ...]
    active_tokens: tuple[SourceToken, ...]
    removed_tokens: tuple[SourceToken, ...]
    units: tuple[SegmentUnit, ...]
    warnings: tuple[SegmentationWarning, ...]
    stats: SegmentationStats

    def to_translation_units_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "segmentation_strategy_version": self.options.segmentation_strategy_version,
            "timing_strategy_version": self.options.timing_strategy_version,
            "mode": self.source.mode,
            "input_format": self.input_format,
            "processing_start_ms": self.processing_start_ms,
            "processing_end_ms": self.processing_end_ms,
            "clip_start_ms": self.clip_start_ms,
            "clip_end_ms": self.clip_end_ms,
            "units": [self._unit_payload(unit) for unit in self.units],
            "warnings": [self._warning_payload(warning) for warning in self.warnings],
            "stats": self._stats_payload(),
        }

    def to_cue_map_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "segmentation_strategy_version": self.options.segmentation_strategy_version,
            "timing_strategy_version": self.options.timing_strategy_version,
            "mode": self.source.mode,
            "units": {unit.unit_id: self._unit_payload(unit) for unit in self.units},
        }

    def to_segmented_srt_text(self) -> str:
        blocks: list[str] = []
        for index, unit in enumerate(self.units, start=1):
            blocks.append(
                f"{index}\n"
                f"{format_timestamp(unit.start_ms / 1000)} --> {format_timestamp(unit.end_ms / 1000)}\n"
                f"{unit.source_text}\n\n"
            )
        return "".join(blocks)

    def to_report_markdown(self) -> str:
        lines = [
            "# Segmentation Report",
            "",
            "## Summary",
            f"- mode: {self.source.mode}",
            f"- input_format: {self.input_format}",
            f"- original_cue_count: {self.stats.original_cue_count}",
            f"- extracted_cue_count: {self.stats.extracted_cue_count}",
            f"- active_token_count: {self.stats.cleaned_active_token_count}",
            f"- translation_unit_count: {self.stats.translation_unit_count}",
            "",
            "## Detection / timing",
            f"- inline_timing_token_count: {self.stats.inline_timing_token_count}",
            f"- proportional_timing_token_count: {self.stats.proportional_timing_token_count}",
            f"- unknown_timing_token_count: {self.stats.unknown_timing_token_count}",
            "",
            "## Cleanup stats",
            f"- removed_duplicate_cue_count: {self.stats.removed_duplicate_cue_count}",
            f"- removed_duplicate_token_count: {self.stats.removed_duplicate_token_count}",
            f"- removed_rolling_overlap_token_count: {self.stats.removed_rolling_overlap_token_count}",
            "",
            "## Validation",
            "- fatal_errors: none",
            "",
            "## Warnings",
        ]
        if not self.warnings:
            lines.append("- none")
        else:
            for warning in self.warnings:
                lines.append(f"- {warning.code}: {warning.message}")
        lines.extend(
            [
                "",
                "## Coverage domain note",
                "- coverage is based on cleaned active token stream, not raw removed tokens.",
            ]
        )
        return "\n".join(lines)

    def _unit_payload(self, unit: SegmentUnit) -> dict[str, object]:
        return {
            "unit_id": unit.unit_id,
            "source_text": unit.source_text,
            "start_ms": unit.start_ms,
            "end_ms": unit.end_ms,
            "start": format_timestamp(unit.start_ms / 1000),
            "end": format_timestamp(unit.end_ms / 1000),
            "source_spans": [self._span_payload(span) for span in unit.source_spans],
            "source_cue_ids": list(unit.source_cue_ids),
            "boundary_type": unit.boundary_type.value,
            "is_padding_only": unit.is_padding_only,
            "crosses_clip_start": unit.crosses_clip_start,
            "crosses_clip_end": unit.crosses_clip_end,
            "intersects_clip": unit.intersects_clip,
        }

    @staticmethod
    def _span_payload(span: SourceSpan) -> dict[str, object]:
        return {
            "start": span.start,
            "end": span.end,
            "cue_id": span.cue_id,
            "cue_token_start": span.cue_token_start,
            "cue_token_end": span.cue_token_end,
        }

    @staticmethod
    def _warning_payload(warning: SegmentationWarning) -> dict[str, object]:
        return {
            "code": warning.code,
            "message": warning.message,
            "severity": warning.severity,
            "unit_id": warning.unit_id,
            "cue_id": warning.cue_id,
            "token_id": warning.token_id,
        }

    def _stats_payload(self) -> dict[str, object]:
        return {
            "input_format": self.stats.input_format,
            "source_mode": self.stats.source_mode,
            "original_cue_count": self.stats.original_cue_count,
            "extracted_cue_count": self.stats.extracted_cue_count,
            "cleaned_active_token_count": self.stats.cleaned_active_token_count,
            "removed_duplicate_cue_count": self.stats.removed_duplicate_cue_count,
            "removed_duplicate_token_count": self.stats.removed_duplicate_token_count,
            "removed_rolling_overlap_token_count": self.stats.removed_rolling_overlap_token_count,
            "translation_unit_count": self.stats.translation_unit_count,
            "inline_timing_token_count": self.stats.inline_timing_token_count,
            "proportional_timing_token_count": self.stats.proportional_timing_token_count,
            "unknown_timing_token_count": self.stats.unknown_timing_token_count,
            "forced_split_count": self.stats.forced_split_count,
            "padding_only_unit_count": self.stats.padding_only_unit_count,
            "warning_count": self.stats.warning_count,
        }


def segment_subtitles(source: SubtitleSegmentationSource, options: SegmentationOptions) -> SegmentationResult:
    original_cue_count, cues, input_format, processing_start_ms, processing_end_ms = _load_source_cues(source)
    raw_tokens, warnings = _build_raw_tokens(cues, input_format, source)
    active_tokens, removed_tokens, cleanup_warnings, cleanup_stats = _cleanup_tokens(cues, raw_tokens, options)
    units, segmentation_warnings, forced_split_count = _segment_tokens(active_tokens, source, options)
    all_warnings = tuple(warnings + cleanup_warnings + segmentation_warnings)
    _validate_active_tokens(active_tokens)
    _validate_units(units, active_tokens)
    stats = SegmentationStats(
        input_format=input_format,
        source_mode=source.mode,
        original_cue_count=original_cue_count,
        extracted_cue_count=len(cues),
        cleaned_active_token_count=len(active_tokens),
        removed_duplicate_cue_count=cleanup_stats.removed_duplicate_cue_count,
        removed_duplicate_token_count=cleanup_stats.removed_duplicate_token_count,
        removed_rolling_overlap_token_count=cleanup_stats.removed_rolling_overlap_token_count,
        translation_unit_count=len(units),
        inline_timing_token_count=sum(1 for token in active_tokens if token.timing_source == "vtt_inline"),
        proportional_timing_token_count=sum(1 for token in active_tokens if token.timing_source == "cue_proportional"),
        unknown_timing_token_count=sum(1 for token in active_tokens if token.timing_source == "unknown"),
        forced_split_count=forced_split_count,
        padding_only_unit_count=sum(1 for unit in units if unit.is_padding_only),
        warning_count=sum(1 for w in all_warnings if w.severity == "warning"),
    )
    return SegmentationResult(
        source=source,
        options=options,
        input_format=input_format,
        processing_start_ms=processing_start_ms,
        processing_end_ms=processing_end_ms,
        clip_start_ms=source.clip_start_ms,
        clip_end_ms=source.clip_end_ms,
        source_cues=tuple(cues),
        active_tokens=tuple(active_tokens),
        removed_tokens=tuple(removed_tokens),
        units=tuple(units),
        warnings=all_warnings,
        stats=stats,
    )


@dataclass(frozen=True)
class _CleanupStats:
    removed_duplicate_cue_count: int = 0
    removed_duplicate_token_count: int = 0
    removed_rolling_overlap_token_count: int = 0


def _load_source_cues(
    source: SubtitleSegmentationSource,
) -> tuple[int, list[SourceCue], str, int | None, int | None]:
    path = source.full_vtt_path if source.mode == "full_vtt_window" else source.subtitle_path
    if path is None:
        raise ValueError("source path is missing")
    subtitle_path = Path(path)
    input_format = detect_subtitle_format(subtitle_path)
    raw_cues = parse_subtitle_file(subtitle_path)
    original_cue_count = len(raw_cues)
    clip_start_ms = source.clip_start_ms
    clip_end_ms = source.clip_end_ms
    processing_start_ms: int | None = None
    processing_end_ms: int | None = None
    if source.mode == "full_vtt_window":
        processing_start_ms = max(0, clip_start_ms - source.padding_before_ms) if clip_start_ms is not None else None
        processing_end_ms = clip_end_ms + source.padding_after_ms if clip_end_ms is not None else None
    selected: list[SourceCue] = []
    for cue in raw_cues:
        cue_start_ms = int(round(parse_timestamp(cue.start) * 1000))
        cue_end_ms = int(round(parse_timestamp(cue.end) * 1000))
        if processing_start_ms is not None and processing_end_ms is not None:
            if cue_end_ms <= processing_start_ms or cue_start_ms >= processing_end_ms:
                continue
        intersects_clip, is_padding_only, crosses_clip_start, crosses_clip_end = _boundary_flags(
            cue_start_ms,
            cue_end_ms,
            clip_start_ms,
            clip_end_ms,
        )
        selected.append(
            SourceCue(
                cue_id=cue.id,
                index=cue.index,
                start_ms=cue_start_ms,
                end_ms=cue_end_ms,
                text=cue.source,
                raw_timing=cue.raw_timing,
                source_kind=input_format,
                intersects_clip=intersects_clip,
                is_padding_only=is_padding_only,
                crosses_clip_start=crosses_clip_start,
                crosses_clip_end=crosses_clip_end,
            )
        )
    if not selected:
        raise SegmentationValidationError("No cues intersect requested processing window")
    return original_cue_count, selected, input_format, processing_start_ms, processing_end_ms


def _build_raw_tokens(
    cues: list[SourceCue],
    input_format: str,
    source: SubtitleSegmentationSource,
) -> tuple[list[SourceToken], list[SegmentationWarning]]:
    tokens: list[SourceToken] = []
    warnings: list[SegmentationWarning] = []
    next_token_id = 0
    for cue in cues:
        cue_tokens, cue_warnings = _tokenize_cue(cue, input_format)
        warnings.extend(cue_warnings)
        if source.mode == "single_file" and cue_tokens and input_format != "vtt":
            warnings.append(
                SegmentationWarning(
                    code="timing_precision_degraded",
                    message="single_file path uses cue-proportional timing without raw VTT inline precision",
                    severity="warning",
                    cue_id=cue.cue_id,
                )
            )
        for token_index, token_data in enumerate(cue_tokens):
            tokens.append(
                SourceToken(
                    token_id=next_token_id,
                    cue_id=cue.cue_id,
                    cue_index=cue.index,
                    token_index_in_cue=token_index,
                    text=token_data[0],
                    normalized_text=_normalize_token(token_data[0]),
                    start_ms=token_data[1],
                    end_ms=token_data[2],
                    timing_source=token_data[3],
                    intersects_clip=cue.intersects_clip,
                    is_padding_only=cue.is_padding_only,
                )
            )
            next_token_id += 1
    if not tokens:
        raise SegmentationValidationError("No tokens could be extracted from selected cues")
    return tokens, warnings


def _tokenize_cue(
    cue: SourceCue,
    input_format: str,
) -> tuple[list[tuple[str, int, int, TimingSource]], list[SegmentationWarning]]:
    warnings: list[SegmentationWarning] = []
    if cue.end_ms <= cue.start_ms:
        return [], [
            SegmentationWarning(
                code="malformed_cue_skipped",
                message="cue has zero or negative duration; skipped token generation",
                severity="warning",
                cue_id=cue.cue_id,
            )
        ]
    if input_format == "vtt":
        inline_tokens, inline_warning = _parse_vtt_inline_tokens(cue)
        if inline_tokens is not None:
            return inline_tokens, warnings
        if inline_warning is not None:
            warnings.append(inline_warning)
    proportional_tokens = _build_proportional_tokens(cue)
    if input_format == "vtt":
        warnings.append(
            SegmentationWarning(
                code="proportional_timing_fallback",
                message="inline timing unavailable or invalid; used cue-proportional timing",
                severity="info",
                cue_id=cue.cue_id,
            )
        )
    return proportional_tokens, warnings


def _parse_vtt_inline_tokens(
    cue: SourceCue,
) -> tuple[list[tuple[str, int, int, TimingSource]] | None, SegmentationWarning | None]:
    saw_inline = False
    current_ms: int | None = None
    token_starts: list[tuple[str, int | None]] = []
    parts = re.split(r"(<[^>]+>)", cue.text)
    for part in parts:
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            content = part[1:-1].strip()
            if TIMESTAMP_TAG_RE.match(content):
                saw_inline = True
                current_ms = int(round(parse_timestamp(content) * 1000))
            continue
        for match in WORD_RE.finditer(part):
            token_starts.append((match.group(0), current_ms))
    if not token_starts:
        return None, None
    if not saw_inline:
        return None, None
    # YouTube auto-sub VTT prefixes each cue with the previous cue's plain text
    # (no inline timing) followed by the new tokens with inline timing. Assign
    # synthetic proportional timing within [cue.start_ms, first_inline_ms] so the
    # leading prefix can still participate in rolling-overlap dedup against the
    # prior cue while the inline-timed tokens keep their authoritative timing.
    first_timed_index = next(
        (idx for idx, (_, start) in enumerate(token_starts) if start is not None),
        None,
    )
    if first_timed_index is None:
        return None, None
    if first_timed_index > 0:
        first_inline_ms = token_starts[first_timed_index][1]
        assert first_inline_ms is not None
        if first_inline_ms > cue.start_ms:
            span_ms = first_inline_ms - cue.start_ms
            for i in range(first_timed_index):
                text, _ = token_starts[i]
                synthetic = cue.start_ms + (span_ms * i) // first_timed_index
                token_starts[i] = (text, synthetic)
        else:
            token_starts = token_starts[first_timed_index:]
    if any(start is None for _, start in token_starts):
        return None, SegmentationWarning(
            code="invalid_inline_timing",
            message="inline timing missing before some tokens; falling back to cue-proportional timing",
            severity="warning",
            cue_id=cue.cue_id,
        )
    starts = [start for _, start in token_starts if start is not None]
    if starts != sorted(starts):
        return None, SegmentationWarning(
            code="invalid_inline_timing",
            message="inline timing is non-monotonic; falling back to cue-proportional timing",
            severity="warning",
            cue_id=cue.cue_id,
        )
    tokens: list[tuple[str, int, int, TimingSource]] = []
    for index, (text, start_ms) in enumerate(token_starts):
        if start_ms is None:
            return None, SegmentationWarning(
                code="invalid_inline_timing",
                message="inline timing missing before some tokens; falling back to cue-proportional timing",
                severity="warning",
                cue_id=cue.cue_id,
            )
        end_ms = cue.end_ms if index == len(token_starts) - 1 else token_starts[index + 1][1]
        if end_ms is None or start_ms >= end_ms:
            return None, SegmentationWarning(
                code="invalid_inline_timing",
                message="inline timing produced zero or negative duration token; falling back to cue-proportional timing",
                severity="warning",
                cue_id=cue.cue_id,
            )
        tokens.append((text, start_ms, end_ms, "vtt_inline"))
    return tokens, None


def _build_proportional_tokens(cue: SourceCue) -> list[tuple[str, int, int, TimingSource]]:
    stripped = SPACE_RE.sub(" ", ANGLE_TAG_RE.sub(" ", cue.text)).strip()
    words = stripped.split()
    if not words:
        return []
    duration_ms = cue.end_ms - cue.start_ms
    if duration_ms <= 0:
        return []
    tokens: list[tuple[str, int, int, TimingSource]] = []
    for index, word in enumerate(words):
        start_ms = cue.start_ms + round(duration_ms * index / len(words))
        end_ms = cue.start_ms + round(duration_ms * (index + 1) / len(words))
        tokens.append((word, start_ms, end_ms, "cue_proportional"))
    return tokens


def _cleanup_tokens(
    cues: list[SourceCue],
    raw_tokens: list[SourceToken],
    options: SegmentationOptions,
) -> tuple[list[SourceToken], list[SourceToken], list[SegmentationWarning], _CleanupStats]:
    tokens_by_cue: dict[str, list[SourceToken]] = {cue.cue_id: [] for cue in cues}
    for token in raw_tokens:
        tokens_by_cue[token.cue_id].append(token)
    warnings: list[SegmentationWarning] = []
    removed: list[SourceToken] = []
    active: list[SourceToken] = []
    kept_previous_cue: SourceCue | None = None
    kept_previous_tokens: list[SourceToken] = []
    kept_previous_text = ""
    stats = _CleanupStats()
    removed_duplicate_cue_count = 0
    removed_duplicate_token_count = 0
    removed_rolling_overlap_token_count = 0
    for cue in cues:
        cue_tokens = list(tokens_by_cue[cue.cue_id])
        cue_text = _normalize_text_for_compare(ANGLE_TAG_RE.sub(" ", cue.text))
        if (
            options.drop_duplicate_cues
            and kept_previous_cue is not None
            and cue_text
            and cue_text == kept_previous_text
            and abs(cue.start_ms - kept_previous_cue.end_ms) <= LOCAL_DUPLICATE_THRESHOLD_MS
        ):
            removed_duplicate_cue_count += 1
            removed_duplicate_token_count += len(cue_tokens)
            removed.extend(_mark_removed(cue_tokens, "duplicate_cue"))
            continue
        if options.remove_rolling_overlap and kept_previous_tokens and cue_tokens:
            overlap = _rolling_overlap_size(kept_previous_tokens, cue_tokens)
            if overlap > 0:
                removed_rolling_overlap_token_count += overlap
                removed.extend(_mark_removed(cue_tokens[:overlap], "rolling_overlap"))
                cue_tokens = cue_tokens[overlap:]
                warnings.append(
                    SegmentationWarning(
                        code="rolling_overlap_removed",
                        message=f"removed {overlap} rolling overlap tokens",
                        severity="info",
                        cue_id=cue.cue_id,
                    )
                )
        active.extend(cue_tokens)
        if cue_tokens:
            kept_previous_cue = cue
            kept_previous_tokens = cue_tokens
            kept_previous_text = cue_text
    stats = _CleanupStats(
        removed_duplicate_cue_count=removed_duplicate_cue_count,
        removed_duplicate_token_count=removed_duplicate_token_count,
        removed_rolling_overlap_token_count=removed_rolling_overlap_token_count,
    )
    return active, removed, warnings, stats


def _mark_removed(tokens: list[SourceToken], reason: str) -> list[SourceToken]:
    return [
        SourceToken(
            token_id=token.token_id,
            cue_id=token.cue_id,
            cue_index=token.cue_index,
            token_index_in_cue=token.token_index_in_cue,
            text=token.text,
            normalized_text=token.normalized_text,
            start_ms=token.start_ms,
            end_ms=token.end_ms,
            timing_source=token.timing_source,
            intersects_clip=token.intersects_clip,
            is_padding_only=token.is_padding_only,
            is_removed=True,
            removal_reason=reason,
        )
        for token in tokens
    ]


def _rolling_overlap_size(previous_tokens: list[SourceToken], current_tokens: list[SourceToken]) -> int:
    max_size = min(len(previous_tokens), len(current_tokens), MAX_ROLLING_OVERLAP_TOKENS)
    for size in range(max_size, 0, -1):
        previous_suffix = [token.normalized_text for token in previous_tokens[-size:]]
        current_prefix = [token.normalized_text for token in current_tokens[:size]]
        if previous_suffix == current_prefix:
            return size
    return 0


def _segment_tokens(
    active_tokens: list[SourceToken],
    source: SubtitleSegmentationSource,
    options: SegmentationOptions,
) -> tuple[list[SegmentUnit], list[SegmentationWarning], int]:
    if not active_tokens:
        raise SegmentationValidationError("No active tokens remain after cleanup")
    units: list[SegmentUnit] = []
    warnings: list[SegmentationWarning] = []
    index = 0
    forced_split_count = 0
    while index < len(active_tokens):
        end_index, forced_split = _choose_split_end(active_tokens, index, options)
        if forced_split:
            forced_split_count += 1
        unit_tokens = active_tokens[index:end_index]
        spans = _build_source_spans(active_tokens, index, end_index)
        source_text = " ".join(token.text for token in unit_tokens)
        boundary = _unit_boundary(unit_tokens, source.clip_start_ms, source.clip_end_ms)
        unit_id = f"u{len(units) + 1:03d}"
        unit = SegmentUnit(
            unit_id=unit_id,
            source_text=source_text,
            start_ms=unit_tokens[0].start_ms,
            end_ms=unit_tokens[-1].end_ms,
            source_spans=tuple(spans),
            boundary_type=boundary[0],
            is_padding_only=boundary[1],
            crosses_clip_start=boundary[2],
            crosses_clip_end=boundary[3],
            intersects_clip=boundary[4],
        )
        units.append(unit)
        if forced_split:
            warnings.append(
                SegmentationWarning(
                    code="forced_split",
                    message="hard constraints forced fallback split selection",
                    severity="warning",
                    unit_id=unit.unit_id,
                )
            )
        index = end_index
    return units, warnings, forced_split_count


def _choose_split_end(tokens: list[SourceToken], start: int, options: SegmentationOptions) -> tuple[int, bool]:
    candidate_non_dangling: int | None = None
    candidate_cue_boundary: int | None = None
    char_count = 0
    cue_ids: set[str] = set()
    unit_start_ms = tokens[start].start_ms
    for index in range(start, len(tokens)):
        if index > start and _matches_strong_boundary(tokens, index, options.strong_boundary_phrases):
            return index, False
        token = tokens[index]
        next_char_count = char_count + (len(token.text) if index == start else len(token.text) + 1)
        next_cue_count = len(cue_ids | {token.cue_id})
        next_duration_ms = token.end_ms - unit_start_ms
        violates = (
            next_char_count > options.max_unit_chars
            or next_duration_ms > options.max_unit_duration_ms
            or next_cue_count > options.max_source_cues
        )
        if violates:
            fallback = _forced_split_end(start, index, candidate_non_dangling, candidate_cue_boundary)
            return fallback, True
        char_count = next_char_count
        cue_ids.add(token.cue_id)
        if not _is_dangling(token.normalized_text, options.dangling_end_tokens):
            candidate_non_dangling = index + 1
        if index + 1 < len(tokens) and tokens[index + 1].cue_id != token.cue_id:
            candidate_cue_boundary = index + 1
    return len(tokens), False


def _forced_split_end(
    start: int,
    violating_index: int,
    candidate_non_dangling: int | None,
    candidate_cue_boundary: int | None,
) -> int:
    if candidate_non_dangling is not None and candidate_non_dangling > start:
        return candidate_non_dangling
    if candidate_cue_boundary is not None and candidate_cue_boundary > start:
        return candidate_cue_boundary
    return max(start + 1, violating_index)


def _matches_strong_boundary(tokens: list[SourceToken], start: int, phrases: tuple[str, ...]) -> bool:
    normalized = [token.normalized_text for token in tokens]
    for phrase in phrases:
        parts = phrase.split()
        end = start + len(parts)
        if end <= len(tokens) and normalized[start:end] == parts:
            return True
    return False


def _is_dangling(token: str, dangling_tokens: tuple[str, ...]) -> bool:
    return token in dangling_tokens


def _build_source_spans(active_tokens: list[SourceToken], start: int, end: int) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    current_cue_id = active_tokens[start].cue_id
    current_start = start
    cue_token_start = active_tokens[start].token_index_in_cue
    last_token_index = active_tokens[start].token_index_in_cue
    for position in range(start + 1, end):
        token = active_tokens[position]
        if token.cue_id != current_cue_id or token.token_index_in_cue != last_token_index + 1:
            spans.append(
                SourceSpan(
                    start=current_start,
                    end=position,
                    cue_id=current_cue_id,
                    cue_token_start=cue_token_start,
                    cue_token_end=last_token_index + 1,
                )
            )
            current_cue_id = token.cue_id
            current_start = position
            cue_token_start = token.token_index_in_cue
        last_token_index = token.token_index_in_cue
    spans.append(
        SourceSpan(
            start=current_start,
            end=end,
            cue_id=current_cue_id,
            cue_token_start=cue_token_start,
            cue_token_end=last_token_index + 1,
        )
    )
    return spans


def _validate_active_tokens(tokens: list[SourceToken]) -> None:
    for token in tokens:
        if token.start_ms >= token.end_ms:
            raise SegmentationValidationError(
                f"active token {token.token_id} has invalid timing: start_ms >= end_ms"
            )


def _validate_units(units: list[SegmentUnit], active_tokens: list[SourceToken]) -> None:
    if not units:
        raise SegmentationValidationError("No segment units were produced")
    coverage = [0] * len(active_tokens)
    previous_start_ms = -1
    previous_end_ms = -1
    for unit in units:
        if unit.start_ms >= unit.end_ms:
            raise SegmentationValidationError(f"unit {unit.unit_id} has invalid timing: start_ms >= end_ms")
        if unit.start_ms < previous_start_ms or unit.end_ms < previous_end_ms:
            raise SegmentationValidationError("non-monotonic unit order")
        previous_start_ms = unit.start_ms
        previous_end_ms = unit.end_ms
        if tuple(unit.source_cue_ids) != _derive_source_cue_ids(unit.source_spans):
            raise SegmentationValidationError(f"source_cue_ids cannot be derived for unit {unit.unit_id}")
        for span in unit.source_spans:
            if span.start < 0 or span.end > len(active_tokens) or span.start >= span.end:
                raise SegmentationValidationError(f"unit {unit.unit_id} has invalid [start, end) span")
            for position in range(span.start, span.end):
                coverage[position] += 1
                if active_tokens[position].cue_id != span.cue_id:
                    raise SegmentationValidationError(f"unit {unit.unit_id} span cue_id does not match active token")
    if any(value == 0 for value in coverage):
        raise SegmentationValidationError("missing active token coverage")
    if any(value > 1 for value in coverage):
        raise SegmentationValidationError("duplicate active token assignment")



def _derive_source_cue_ids(spans: tuple[SourceSpan, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for span in spans:
        if span.cue_id not in seen:
            seen.append(span.cue_id)
    return tuple(seen)


def _unit_boundary(
    tokens: list[SourceToken],
    clip_start_ms: int | None,
    clip_end_ms: int | None,
) -> tuple[SegmentBoundaryType, bool, bool, bool, bool]:
    intersects_clip = any(token.intersects_clip for token in tokens)
    is_padding_only = not intersects_clip
    starts = [token.start_ms for token in tokens]
    ends = [token.end_ms for token in tokens]
    crosses_clip_start = False
    crosses_clip_end = False
    if clip_start_ms is not None:
        crosses_clip_start = min(starts) < clip_start_ms < max(ends)
    if clip_end_ms is not None:
        crosses_clip_end = min(starts) < clip_end_ms < max(ends)
    if is_padding_only:
        boundary_type = SegmentBoundaryType.PADDING_ONLY
    elif crosses_clip_start:
        boundary_type = SegmentBoundaryType.CROSSES_CLIP_START
    elif crosses_clip_end:
        boundary_type = SegmentBoundaryType.CROSSES_CLIP_END
    else:
        boundary_type = SegmentBoundaryType.INSIDE_CLIP
    return boundary_type, is_padding_only, crosses_clip_start, crosses_clip_end, intersects_clip


def _boundary_flags(
    start_ms: int,
    end_ms: int,
    clip_start_ms: int | None,
    clip_end_ms: int | None,
) -> tuple[bool, bool, bool, bool]:
    if clip_start_ms is None or clip_end_ms is None:
        return True, False, False, False
    intersects_clip = end_ms > clip_start_ms and start_ms < clip_end_ms
    is_padding_only = not intersects_clip
    crosses_clip_start = start_ms < clip_start_ms < end_ms
    crosses_clip_end = start_ms < clip_end_ms < end_ms
    return intersects_clip, is_padding_only, crosses_clip_start, crosses_clip_end


def _normalize_text_for_compare(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip().lower()


def _normalize_token(text: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", text.lower()) or text.lower()


__all__ = [
    "SCHEMA_VERSION",
    "SegmentBoundaryType",
    "SegmentUnit",
    "SegmentationOptions",
    "SegmentationResult",
    "SegmentationStats",
    "SegmentationValidationError",
    "SegmentationWarning",
    "SourceCue",
    "SourceSpan",
    "SourceToken",
    "SubtitleSegmentationSource",
    "segment_subtitles",
]
