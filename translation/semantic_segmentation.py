from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from translation.segmentation import (
    SegmentBoundaryType,
    SegmentationWarning,
    SegmentUnit,
    SourceSpan,
    SourceToken,
    _derive_source_cue_ids,
    _unit_boundary,
)


FORBIDDEN_TEXT_FIELDS = frozenset({"source_text", "text", "tokens", "rewritten_text"})
SEMANTIC_HARD_LIMIT_MULTIPLIER = 1.5


class SemanticSegmentationError(Exception):
    pass


@dataclass(frozen=True)
class SemanticSegmenterOptions:
    enabled: bool = False
    mode: str = "off"
    max_unit_chars: int = 220
    max_unit_duration_ms: int = 9000
    min_unit_duration_ms: int = 800
    max_tokens_per_request: int = 350
    fallback_to_rules: bool = True
    model: str | None = None
    prompt_version: str = "cycle3a-semantic-v1"


@dataclass(frozen=True)
class SemanticBoundary:
    start_token: int
    end_token: int
    reason: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    segments: tuple[SemanticBoundary, ...] | None
    reason: str | None
    warnings: tuple[SegmentationWarning, ...]


@dataclass(frozen=True)
class SemanticSegmentationResult:
    units: tuple[SegmentUnit, ...]
    mode: str
    fallback_used: bool
    fallback_reason: str | None
    warnings: tuple[SegmentationWarning, ...]


def parse_semantic_boundaries(payload: str | dict[str, Any]) -> tuple[SemanticBoundary, ...]:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SemanticSegmentationError(f"parse_error: {exc}") from exc
    else:
        data = payload

    if not isinstance(data, dict):
        raise SemanticSegmentationError("parse_error: payload is not a JSON object")

    if "segments" not in data:
        raise SemanticSegmentationError("parse_error: missing 'segments' key")

    segments = data["segments"]
    if not isinstance(segments, list):
        raise SemanticSegmentationError("parse_error: 'segments' is not an array")

    boundaries: list[SemanticBoundary] = []
    for seg in segments:
        if not isinstance(seg, dict):
            raise SemanticSegmentationError("parse_error: segment entry is not an object")
        if "start_token" not in seg or "end_token" not in seg:
            raise SemanticSegmentationError("parse_error: segment missing start_token or end_token")
        try:
            start_token = int(seg["start_token"])
            end_token = int(seg["end_token"])
        except (TypeError, ValueError) as exc:
            raise SemanticSegmentationError(
                f"parse_error: invalid start_token/end_token value in segment {seg}"
            ) from exc
        boundaries.append(
            SemanticBoundary(
                start_token=start_token,
                end_token=end_token,
                reason=seg.get("reason"),
            )
        )
    return tuple(boundaries)


def validate_semantic_boundaries(
    boundaries: tuple[SemanticBoundary, ...],
    eligible_tokens: tuple[SourceToken, ...],
    options: SemanticSegmenterOptions,
    raw_payload: dict[str, Any] | None = None,
) -> ValidationResult:
    warnings: list[SegmentationWarning] = []
    n = len(eligible_tokens)

    if not boundaries:
        return ValidationResult(ok=False, segments=None, reason="empty_segments", warnings=())

    if raw_payload is not None:
        for seg in raw_payload.get("segments", []):
            if isinstance(seg, dict):
                text_fields = FORBIDDEN_TEXT_FIELDS & set(seg.keys())
                if text_fields:
                    return ValidationResult(
                        ok=False, segments=None,
                        reason=f"forbidden_text_fields: {sorted(text_fields)}",
                        warnings=(),
                    )

    for i, b in enumerate(boundaries):
        if not isinstance(b.start_token, int) or b.start_token < 0:
            return ValidationResult(ok=False, segments=None, reason=f"V4: invalid start_token at segment {i}", warnings=())
        if not isinstance(b.end_token, int) or b.end_token <= b.start_token:
            return ValidationResult(ok=False, segments=None, reason=f"V5: end_token <= start_token at segment {i}", warnings=())
        if b.start_token >= n or b.end_token > n:
            return ValidationResult(ok=False, segments=None, reason=f"V11: out_of_range at segment {i}", warnings=())

    if boundaries[0].start_token != 0:
        return ValidationResult(ok=False, segments=None, reason="V6: first segment does not start at 0", warnings=())

    if boundaries[-1].end_token != n:
        return ValidationResult(ok=False, segments=None, reason=f"V7: last segment end_token != {n}", warnings=())

    for i in range(len(boundaries) - 1):
        if boundaries[i].end_token > boundaries[i + 1].start_token:
            return ValidationResult(ok=False, segments=None, reason=f"V9: overlap at segment {i}/{i+1}", warnings=())
        if boundaries[i].end_token < boundaries[i + 1].start_token:
            return ValidationResult(ok=False, segments=None, reason=f"V10: gap at segment {i}/{i+1}", warnings=())

    hard_max_chars = int(options.max_unit_chars * SEMANTIC_HARD_LIMIT_MULTIPLIER)
    hard_max_duration = int(options.max_unit_duration_ms * SEMANTIC_HARD_LIMIT_MULTIPLIER)

    for i, b in enumerate(boundaries):
        seg_tokens = eligible_tokens[b.start_token:b.end_token]
        source_text = " ".join(t.text for t in seg_tokens)
        start_ms = seg_tokens[0].start_ms
        end_ms = seg_tokens[-1].end_ms
        duration_ms = end_ms - start_ms

        if len(source_text) > hard_max_chars:
            return ValidationResult(ok=False, segments=None, reason=f"V14: segment {i} exceeds hard max chars ({len(source_text)} > {hard_max_chars})", warnings=())

        if duration_ms > hard_max_duration:
            return ValidationResult(ok=False, segments=None, reason=f"V15: segment {i} exceeds hard max duration ({duration_ms} > {hard_max_duration})", warnings=())

        if duration_ms < options.min_unit_duration_ms:
            warnings.append(SegmentationWarning(
                code="semantic_short_segment",
                message=f"segment {i} duration {duration_ms}ms < min {options.min_unit_duration_ms}ms (advisory)",
                severity="info",
                unit_id=f"s{i+1:03d}",
            ))

    return ValidationResult(ok=True, segments=boundaries, reason=None, warnings=tuple(warnings))


def _derive_source_spans_from_token_range(
    eligible_tokens: tuple[SourceToken, ...],
    start_token: int,
    end_token: int,
    original_indices: tuple[int, ...] | None = None,
) -> tuple[SourceSpan, ...]:
    tokens = eligible_tokens[start_token:end_token]
    if not tokens:
        return ()

    if original_indices is not None:
        idx_slice = original_indices[start_token:end_token]
    else:
        idx_slice = tuple(range(start_token, end_token))

    spans: list[SourceSpan] = []
    current_cue_id = tokens[0].cue_id
    current_start_orig = idx_slice[0]
    cue_token_start = tokens[0].token_index_in_cue

    for i, token in enumerate(tokens):
        if token.cue_id != current_cue_id:
            spans.append(SourceSpan(
                start=current_start_orig,
                end=idx_slice[i],
                cue_id=current_cue_id,
                cue_token_start=cue_token_start,
                cue_token_end=tokens[i - 1].token_index_in_cue + 1,
            ))
            current_cue_id = token.cue_id
            current_start_orig = idx_slice[i]
            cue_token_start = token.token_index_in_cue

    spans.append(SourceSpan(
        start=current_start_orig,
        end=idx_slice[-1] + 1,
        cue_id=current_cue_id,
        cue_token_start=cue_token_start,
        cue_token_end=tokens[-1].token_index_in_cue + 1,
    ))
    return tuple(spans)


def rebuild_units_from_boundaries(
    boundaries: tuple[SemanticBoundary, ...],
    eligible_tokens: tuple[SourceToken, ...],
    preserved_padding_units: tuple[SegmentUnit, ...],
    clip_start_ms: int | None,
    clip_end_ms: int | None,
    original_indices: tuple[int, ...] | None = None,
) -> tuple[SegmentUnit, ...]:
    units: list[SegmentUnit] = []

    for idx, b in enumerate(boundaries):
        seg_tokens = eligible_tokens[b.start_token:b.end_token]
        source_text = " ".join(t.text for t in seg_tokens)
        start_ms = seg_tokens[0].start_ms
        end_ms = seg_tokens[-1].end_ms

        source_spans = _derive_source_spans_from_token_range(
            eligible_tokens, b.start_token, b.end_token, original_indices=original_indices
        )

        boundary_type, is_padding_only, crosses_clip_start, crosses_clip_end, intersects_clip = _unit_boundary(
            list(seg_tokens), clip_start_ms, clip_end_ms
        )

        units.append(SegmentUnit(
            unit_id=f"s{idx + 1:03d}",
            source_text=source_text,
            start_ms=start_ms,
            end_ms=end_ms,
            source_spans=source_spans,
            boundary_type=boundary_type,
            is_padding_only=is_padding_only,
            crosses_clip_start=crosses_clip_start,
            crosses_clip_end=crosses_clip_end,
            intersects_clip=intersects_clip,
        ))

    combined = list(units) + list(preserved_padding_units)
    combined.sort(key=lambda u: (u.start_ms, u.end_ms, u.unit_id))
    return tuple(combined)


def extract_translation_eligible_tokens(
    active_tokens: tuple[SourceToken, ...],
    rule_units: tuple[SegmentUnit, ...],
) -> tuple[SourceToken, ...]:
    eligible, _ = _extract_eligible_with_indices(active_tokens, rule_units)
    return eligible


def _extract_eligible_with_indices(
    active_tokens: tuple[SourceToken, ...],
    rule_units: tuple[SegmentUnit, ...],
) -> tuple[tuple[SourceToken, ...], tuple[int, ...]]:
    padding_only_token_ids: set[int] = set()
    for unit in rule_units:
        if unit.is_padding_only:
            for span in unit.source_spans:
                for pos in range(span.start, span.end):
                    padding_only_token_ids.add(pos)

    tokens: list[SourceToken] = []
    indices: list[int] = []
    for idx, token in enumerate(active_tokens):
        if idx not in padding_only_token_ids:
            tokens.append(token)
            indices.append(idx)
    return tuple(tokens), tuple(indices)


def extract_padding_only_units(rule_units: tuple[SegmentUnit, ...]) -> tuple[SegmentUnit, ...]:
    return tuple(unit for unit in rule_units if unit.is_padding_only)


def fallback_to_rule_units(
    rule_units: tuple[SegmentUnit, ...],
    reason: str,
    extra_warnings: tuple[SegmentationWarning, ...] = (),
) -> SemanticSegmentationResult:
    return SemanticSegmentationResult(
        units=rule_units,
        mode="fallback",
        fallback_used=True,
        fallback_reason=reason,
        warnings=extra_warnings + (SegmentationWarning(
            code="semantic_segmentation_fallback",
            message=f"LLM semantic segmentation failed ({reason}); reverted to Cycle 2 rule units",
            severity="warning",
        ),),
    )


def refine_units_from_semantic_boundaries(
    active_tokens: tuple[SourceToken, ...],
    rule_units: tuple[SegmentUnit, ...],
    raw_payload: str | dict[str, Any],
    options: SemanticSegmenterOptions,
    clip_start_ms: int | None,
    clip_end_ms: int | None,
) -> SemanticSegmentationResult:
    eligible_tokens, original_indices = _extract_eligible_with_indices(active_tokens, rule_units)

    if len(eligible_tokens) > options.max_tokens_per_request:
        if options.fallback_to_rules:
            return SemanticSegmentationResult(
                units=rule_units,
                mode="fallback",
                fallback_used=True,
                fallback_reason="too_many_tokens",
                warnings=(SegmentationWarning(
                    code="semantic_segmentation_skipped_too_many_tokens",
                    message="Semantic segmentation skipped; eligible token count exceeded MAX_TOKENS_PER_REQUEST",
                    severity="info",
                ),),
            )
        raise SemanticSegmentationError("eligible token count exceeded MAX_TOKENS_PER_REQUEST")

    try:
        boundaries = parse_semantic_boundaries(raw_payload)
    except SemanticSegmentationError as exc:
        reason = str(exc)
        if options.fallback_to_rules:
            return fallback_to_rule_units(rule_units, reason)
        raise

    parsed_data = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
    validation = validate_semantic_boundaries(boundaries, eligible_tokens, options, raw_payload=parsed_data)

    if not validation.ok:
        if options.fallback_to_rules:
            return fallback_to_rule_units(rule_units, validation.reason or "unknown", validation.warnings)
        raise SemanticSegmentationError(validation.reason or "validation_failed")

    preserved_padding = extract_padding_only_units(rule_units)
    semantic_units = rebuild_units_from_boundaries(
        boundaries=validation.segments,
        eligible_tokens=eligible_tokens,
        preserved_padding_units=preserved_padding,
        clip_start_ms=clip_start_ms,
        clip_end_ms=clip_end_ms,
        original_indices=original_indices,
    )

    return SemanticSegmentationResult(
        units=semantic_units,
        mode=options.mode,
        fallback_used=False,
        fallback_reason=None,
        warnings=validation.warnings,
    )
