from __future__ import annotations

import json
import unittest

from translation.segmentation import (
    SegmentBoundaryType,
    SegmentUnit,
    SegmentationWarning,
    SourceSpan,
    SourceToken,
)
from translation.semantic_segmentation import (
    SemanticBoundary,
    SemanticSegmentationError,
    SemanticSegmenterOptions,
    extract_padding_only_units,
    extract_translation_eligible_tokens,
    fallback_to_rule_units,
    parse_semantic_boundaries,
    rebuild_units_from_boundaries,
    refine_units_from_semantic_boundaries,
    validate_semantic_boundaries,
)


def _token(token_id: int, cue_id: str, cue_index: int, token_index_in_cue: int, text: str, start_ms: int, end_ms: int, intersects_clip: bool = True, is_padding_only: bool = False) -> SourceToken:
    return SourceToken(
        token_id=token_id,
        cue_id=cue_id,
        cue_index=cue_index,
        token_index_in_cue=token_index_in_cue,
        text=text,
        normalized_text=text.lower(),
        start_ms=start_ms,
        end_ms=end_ms,
        timing_source="vtt_inline",
        intersects_clip=intersects_clip,
        is_padding_only=is_padding_only,
    )


def _eligible_token_stream() -> tuple[SourceToken, ...]:
    return (
        _token(0, "1", 0, 0, "this", 0, 200),
        _token(1, "1", 0, 1, "video", 200, 400),
        _token(2, "1", 0, 2, "shows", 400, 600),
        _token(3, "1", 0, 3, "you", 600, 800),
        _token(4, "1", 0, 4, "cool", 800, 1000),
        _token(5, "2", 1, 0, "things", 1000, 2000),
        _token(6, "2", 1, 1, "today", 2000, 2500),
        _token(7, "3", 2, 0, "and", 2500, 2700),
        _token(8, "3", 2, 1, "tomorrow", 2700, 3500),
        _token(9, "3", 2, 2, "as", 3500, 3700),
        _token(10, "3", 2, 3, "well", 3700, 4000),
    )


def _padding_unit() -> SegmentUnit:
    return SegmentUnit(
        unit_id="u002",
        source_text="extra padding tokens",
        start_ms=10_000,
        end_ms=12_000,
        source_spans=(SourceSpan(start=11, end=14, cue_id="9", cue_token_start=0, cue_token_end=3),),
        boundary_type=SegmentBoundaryType.PADDING_ONLY,
        is_padding_only=True,
        crosses_clip_start=False,
        crosses_clip_end=False,
        intersects_clip=False,
    )


def _rule_unit_translatable() -> SegmentUnit:
    return SegmentUnit(
        unit_id="u001",
        source_text="this video shows you cool things today and tomorrow as well",
        start_ms=0,
        end_ms=4000,
        source_spans=(SourceSpan(start=0, end=11, cue_id="1", cue_token_start=0, cue_token_end=11),),
        boundary_type=SegmentBoundaryType.INSIDE_CLIP,
        is_padding_only=False,
        crosses_clip_start=False,
        crosses_clip_end=False,
        intersects_clip=True,
    )


class ParseSemanticBoundariesTests(unittest.TestCase):
    def test_parses_valid_payload(self):
        payload = {"segments": [{"start_token": 0, "end_token": 5, "reason": "intro"}, {"start_token": 5, "end_token": 11}]}
        boundaries = parse_semantic_boundaries(payload)
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(boundaries[0].start_token, 0)
        self.assertEqual(boundaries[0].end_token, 5)
        self.assertEqual(boundaries[0].reason, "intro")
        self.assertIsNone(boundaries[1].reason)

    def test_parses_string_payload(self):
        payload = json.dumps({"segments": [{"start_token": 0, "end_token": 11}]})
        boundaries = parse_semantic_boundaries(payload)
        self.assertEqual(len(boundaries), 1)

    def test_invalid_json_raises(self):
        with self.assertRaises(SemanticSegmentationError) as ctx:
            parse_semantic_boundaries("not json{")
        self.assertIn("parse_error", str(ctx.exception))

    def test_missing_segments_key_raises(self):
        with self.assertRaises(SemanticSegmentationError):
            parse_semantic_boundaries({"foo": "bar"})

    def test_segments_not_array_raises(self):
        with self.assertRaises(SemanticSegmentationError):
            parse_semantic_boundaries({"segments": "nope"})

    def test_invalid_start_token_value_raises_parse_error(self):
        with self.assertRaises(SemanticSegmentationError) as ctx:
            parse_semantic_boundaries({"segments": [{"start_token": "abc", "end_token": 1}]})
        self.assertIn("parse_error", str(ctx.exception))

    def test_invalid_end_token_value_raises_parse_error(self):
        with self.assertRaises(SemanticSegmentationError) as ctx:
            parse_semantic_boundaries({"segments": [{"start_token": 0, "end_token": None}]})
        self.assertIn("parse_error", str(ctx.exception))


class ValidateSemanticBoundariesTests(unittest.TestCase):
    def setUp(self):
        self.tokens = _eligible_token_stream()
        self.options = SemanticSegmenterOptions(
            enabled=True,
            max_unit_chars=220,
            max_unit_duration_ms=9000,
            min_unit_duration_ms=500,
            max_tokens_per_request=350,
        )

    def test_valid_contiguous_coverage_passes(self):
        boundaries = (SemanticBoundary(0, 5), SemanticBoundary(5, 11))
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options)
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, None)

    def test_first_segment_must_start_at_zero(self):
        boundaries = (SemanticBoundary(1, 5), SemanticBoundary(5, 11))
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options)
        self.assertFalse(result.ok)
        self.assertIn("V6", result.reason)

    def test_last_segment_must_end_at_n(self):
        boundaries = (SemanticBoundary(0, 5), SemanticBoundary(5, 10))
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options)
        self.assertFalse(result.ok)
        self.assertIn("V7", result.reason)

    def test_overlap_fails_with_v9(self):
        boundaries = (SemanticBoundary(0, 6), SemanticBoundary(5, 11))
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options)
        self.assertFalse(result.ok)
        self.assertIn("V9", result.reason)

    def test_gap_fails_with_v10(self):
        boundaries = (SemanticBoundary(0, 4), SemanticBoundary(5, 11))
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options)
        self.assertFalse(result.ok)
        self.assertIn("V10", result.reason)

    def test_out_of_range_fails(self):
        boundaries = (SemanticBoundary(0, 5), SemanticBoundary(5, 99))
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options)
        self.assertFalse(result.ok)
        self.assertIn("V11", result.reason)

    def test_end_le_start_fails(self):
        boundaries = (SemanticBoundary(0, 0),)
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options)
        self.assertFalse(result.ok)
        self.assertIn("V5", result.reason)

    def test_text_bearing_fields_rejected(self):
        boundaries = (SemanticBoundary(0, 11),)
        raw_payload = {"segments": [{"start_token": 0, "end_token": 11, "source_text": "rewritten"}]}
        result = validate_semantic_boundaries(boundaries, self.tokens, self.options, raw_payload=raw_payload)
        self.assertFalse(result.ok)
        self.assertIn("forbidden_text_fields", result.reason)

    def test_short_segment_emits_advisory_warning_not_failure(self):
        options = SemanticSegmenterOptions(
            enabled=True,
            min_unit_duration_ms=10_000,
            max_unit_chars=220,
            max_unit_duration_ms=9000,
        )
        boundaries = (SemanticBoundary(0, 11),)
        result = validate_semantic_boundaries(boundaries, self.tokens, options)
        self.assertTrue(result.ok)
        self.assertTrue(any(w.code == "semantic_short_segment" for w in result.warnings))
        self.assertTrue(all(w.severity == "info" for w in result.warnings))

    def test_hard_max_chars_fails(self):
        options = SemanticSegmenterOptions(
            enabled=True,
            max_unit_chars=5,
            max_unit_duration_ms=9000,
            min_unit_duration_ms=100,
        )
        boundaries = (SemanticBoundary(0, 11),)
        result = validate_semantic_boundaries(boundaries, self.tokens, options)
        self.assertFalse(result.ok)
        self.assertIn("V14", result.reason)

    def test_hard_max_duration_fails(self):
        options = SemanticSegmenterOptions(
            enabled=True,
            max_unit_chars=500,
            max_unit_duration_ms=100,
            min_unit_duration_ms=10,
        )
        boundaries = (SemanticBoundary(0, 11),)
        result = validate_semantic_boundaries(boundaries, self.tokens, options)
        self.assertFalse(result.ok)
        self.assertIn("V15", result.reason)


class RebuildUnitsTests(unittest.TestCase):
    def test_rebuild_creates_semantic_unit_ids(self):
        tokens = _eligible_token_stream()
        boundaries = (SemanticBoundary(0, 5), SemanticBoundary(5, 11))
        units = rebuild_units_from_boundaries(
            boundaries=boundaries,
            eligible_tokens=tokens,
            preserved_padding_units=(),
            clip_start_ms=0,
            clip_end_ms=10_000,
        )
        self.assertEqual(units[0].unit_id, "s001")
        self.assertEqual(units[1].unit_id, "s002")

    def test_rebuild_source_text_joins_tokens(self):
        tokens = _eligible_token_stream()
        boundaries = (SemanticBoundary(0, 5),)
        units = rebuild_units_from_boundaries(
            boundaries=boundaries,
            eligible_tokens=tokens,
            preserved_padding_units=(),
            clip_start_ms=0,
            clip_end_ms=10_000,
        )
        self.assertEqual(units[0].source_text, "this video shows you cool")

    def test_rebuild_preserves_first_seen_cue_order(self):
        repeated = (
            _token(0, "5", 0, 0, "alpha", 0, 100),
            _token(1, "5", 0, 1, "beta", 100, 200),
            _token(2, "7", 1, 0, "gamma", 200, 300),
            _token(3, "5", 0, 2, "delta", 300, 400),
            _token(4, "9", 2, 0, "epsilon", 400, 500),
        )
        boundaries = (SemanticBoundary(0, 5),)
        units = rebuild_units_from_boundaries(
            boundaries=boundaries,
            eligible_tokens=repeated,
            preserved_padding_units=(),
            clip_start_ms=0,
            clip_end_ms=1000,
        )
        cue_ids = [span.cue_id for span in units[0].source_spans]
        self.assertEqual(cue_ids, ["5", "7", "5", "9"])
        self.assertEqual(units[0].source_cue_ids, ("5", "7", "9"))

    def test_padding_units_appended_unchanged(self):
        tokens = _eligible_token_stream()
        padding = _padding_unit()
        boundaries = (SemanticBoundary(0, 11),)
        units = rebuild_units_from_boundaries(
            boundaries=boundaries,
            eligible_tokens=tokens,
            preserved_padding_units=(padding,),
            clip_start_ms=0,
            clip_end_ms=5000,
        )
        self.assertEqual(len(units), 2)
        self.assertIs(units[1], padding)
        self.assertTrue(units[1].is_padding_only)

    def test_padding_units_keep_chronological_order(self):
        tokens = _eligible_token_stream()
        leading_padding = SegmentUnit(
            unit_id="u000",
            source_text="leading padding",
            start_ms=-500,
            end_ms=-100,
            source_spans=(SourceSpan(start=20, end=22, cue_id="0", cue_token_start=0, cue_token_end=2),),
            boundary_type=SegmentBoundaryType.PADDING_ONLY,
            is_padding_only=True,
            crosses_clip_start=False,
            crosses_clip_end=False,
            intersects_clip=False,
        )
        boundaries = (SemanticBoundary(0, 11),)
        units = rebuild_units_from_boundaries(
            boundaries=boundaries,
            eligible_tokens=tokens,
            preserved_padding_units=(leading_padding,),
            clip_start_ms=0,
            clip_end_ms=5000,
        )
        self.assertEqual([unit.unit_id for unit in units], ["u000", "s001"])

    def test_refine_preserves_original_active_token_indices_in_source_spans(self):
        active = (
            _token(0, "0", 0, 0, "pad", -200, -100, intersects_clip=False, is_padding_only=True),
            _token(1, "1", 1, 0, "alpha", 0, 200),
            _token(2, "1", 1, 1, "beta", 200, 400),
            _token(3, "9", 2, 0, "midpad", 400, 500, intersects_clip=False, is_padding_only=True),
            _token(4, "2", 3, 0, "gamma", 500, 700),
            _token(5, "2", 3, 1, "delta", 700, 900),
        )
        rule_units = (
            SegmentUnit(
                unit_id="u000",
                source_text="pad",
                start_ms=-200,
                end_ms=-100,
                source_spans=(SourceSpan(start=0, end=1, cue_id="0", cue_token_start=0, cue_token_end=1),),
                boundary_type=SegmentBoundaryType.PADDING_ONLY,
                is_padding_only=True,
                crosses_clip_start=False,
                crosses_clip_end=False,
                intersects_clip=False,
            ),
            SegmentUnit(
                unit_id="u001",
                source_text="alpha beta gamma delta",
                start_ms=0,
                end_ms=900,
                source_spans=(
                    SourceSpan(start=1, end=3, cue_id="1", cue_token_start=0, cue_token_end=2),
                    SourceSpan(start=4, end=6, cue_id="2", cue_token_start=0, cue_token_end=2),
                ),
                boundary_type=SegmentBoundaryType.INSIDE_CLIP,
                is_padding_only=False,
                crosses_clip_start=False,
                crosses_clip_end=False,
                intersects_clip=True,
            ),
            SegmentUnit(
                unit_id="u002",
                source_text="midpad",
                start_ms=400,
                end_ms=500,
                source_spans=(SourceSpan(start=3, end=4, cue_id="9", cue_token_start=0, cue_token_end=1),),
                boundary_type=SegmentBoundaryType.PADDING_ONLY,
                is_padding_only=True,
                crosses_clip_start=False,
                crosses_clip_end=False,
                intersects_clip=False,
            ),
        )
        payload = {"segments": [{"start_token": 0, "end_token": 4}]}
        options = SemanticSegmenterOptions(
            enabled=True,
            mode="hybrid",
            max_unit_chars=220,
            max_unit_duration_ms=9000,
            min_unit_duration_ms=100,
            max_tokens_per_request=350,
            fallback_to_rules=True,
        )
        result = refine_units_from_semantic_boundaries(
            active_tokens=active,
            rule_units=rule_units,
            raw_payload=payload,
            options=options,
            clip_start_ms=0,
            clip_end_ms=1000,
        )
        semantic_unit = next(unit for unit in result.units if unit.unit_id == "s001")
        self.assertEqual(
            semantic_unit.source_spans,
            (
                SourceSpan(start=1, end=4, cue_id="1", cue_token_start=0, cue_token_end=2),
                SourceSpan(start=4, end=6, cue_id="2", cue_token_start=0, cue_token_end=2),
            ),
        )

    def test_crosses_clip_end_boundary_type(self):
        tokens = _eligible_token_stream()
        boundaries = (SemanticBoundary(0, 11),)
        units = rebuild_units_from_boundaries(
            boundaries=boundaries,
            eligible_tokens=tokens,
            preserved_padding_units=(),
            clip_start_ms=0,
            clip_end_ms=2200,
        )
        self.assertEqual(units[0].boundary_type, SegmentBoundaryType.CROSSES_CLIP_END)
        self.assertTrue(units[0].crosses_clip_end)


class EligibleTokenExtractionTests(unittest.TestCase):
    def test_excludes_padding_only_token_indices(self):
        active = _eligible_token_stream() + (
            _token(11, "9", 3, 0, "padding", 10_000, 11_000, intersects_clip=False, is_padding_only=True),
            _token(12, "9", 3, 1, "stuff", 11_000, 12_000, intersects_clip=False, is_padding_only=True),
            _token(13, "9", 3, 2, "here", 12_000, 12_500, intersects_clip=False, is_padding_only=True),
        )
        rule_units = (_rule_unit_translatable(), _padding_unit())
        eligible = extract_translation_eligible_tokens(active, rule_units)
        self.assertEqual(len(eligible), 11)
        self.assertNotIn("padding", [t.text for t in eligible])

    def test_extract_padding_only_units_filters(self):
        rule_units = (_rule_unit_translatable(), _padding_unit())
        padding = extract_padding_only_units(rule_units)
        self.assertEqual(len(padding), 1)
        self.assertEqual(padding[0].unit_id, "u002")


class FallbackHelperTests(unittest.TestCase):
    def test_fallback_returns_rule_units_with_warning(self):
        rule_units = (_rule_unit_translatable(),)
        result = fallback_to_rule_units(rule_units, reason="invalid_coverage")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "invalid_coverage")
        self.assertEqual(result.units, rule_units)
        self.assertEqual(result.mode, "fallback")
        self.assertTrue(any(w.code == "semantic_segmentation_fallback" for w in result.warnings))

    def test_fallback_carries_extra_warnings(self):
        rule_units = (_rule_unit_translatable(),)
        extra = (SegmentationWarning(code="semantic_short_segment", message="short", severity="info"),)
        result = fallback_to_rule_units(rule_units, reason="V14", extra_warnings=extra)
        codes = {w.code for w in result.warnings}
        self.assertIn("semantic_short_segment", codes)
        self.assertIn("semantic_segmentation_fallback", codes)


class RefineEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.active_tokens = _eligible_token_stream()
        self.rule_units = (_rule_unit_translatable(),)
        self.options = SemanticSegmenterOptions(
            enabled=True,
            mode="hybrid",
            max_unit_chars=220,
            max_unit_duration_ms=9000,
            min_unit_duration_ms=200,
            max_tokens_per_request=350,
            fallback_to_rules=True,
        )

    def test_valid_payload_produces_semantic_units(self):
        payload = {"segments": [{"start_token": 0, "end_token": 5}, {"start_token": 5, "end_token": 11}]}
        result = refine_units_from_semantic_boundaries(
            active_tokens=self.active_tokens,
            rule_units=self.rule_units,
            raw_payload=payload,
            options=self.options,
            clip_start_ms=0,
            clip_end_ms=10_000,
        )
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.mode, "hybrid")
        self.assertEqual(len(result.units), 2)
        self.assertEqual(result.units[0].unit_id, "s001")

    def test_invalid_json_falls_back_when_enabled(self):
        result = refine_units_from_semantic_boundaries(
            active_tokens=self.active_tokens,
            rule_units=self.rule_units,
            raw_payload="not json{",
            options=self.options,
            clip_start_ms=0,
            clip_end_ms=10_000,
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.units, self.rule_units)

    def test_invalid_json_raises_when_fallback_disabled(self):
        options = SemanticSegmenterOptions(
            enabled=True,
            mode="llm",
            fallback_to_rules=False,
            max_unit_chars=220,
            max_unit_duration_ms=9000,
        )
        with self.assertRaises(SemanticSegmentationError):
            refine_units_from_semantic_boundaries(
                active_tokens=self.active_tokens,
                rule_units=self.rule_units,
                raw_payload="not json{",
                options=options,
                clip_start_ms=0,
                clip_end_ms=10_000,
            )

    def test_invalid_coverage_falls_back(self):
        payload = {"segments": [{"start_token": 0, "end_token": 5}]}
        result = refine_units_from_semantic_boundaries(
            active_tokens=self.active_tokens,
            rule_units=self.rule_units,
            raw_payload=payload,
            options=self.options,
            clip_start_ms=0,
            clip_end_ms=10_000,
        )
        self.assertTrue(result.fallback_used)
        self.assertIn("V7", result.fallback_reason)

    def test_text_bearing_field_falls_back(self):
        payload = {"segments": [{"start_token": 0, "end_token": 11, "rewritten_text": "x"}]}
        result = refine_units_from_semantic_boundaries(
            active_tokens=self.active_tokens,
            rule_units=self.rule_units,
            raw_payload=payload,
            options=self.options,
            clip_start_ms=0,
            clip_end_ms=10_000,
        )
        self.assertTrue(result.fallback_used)
        self.assertIn("forbidden_text_fields", result.fallback_reason)

    def test_too_many_tokens_skips_with_info_warning(self):
        options = SemanticSegmenterOptions(
            enabled=True,
            mode="hybrid",
            max_tokens_per_request=5,
            fallback_to_rules=True,
            max_unit_chars=220,
            max_unit_duration_ms=9000,
        )
        result = refine_units_from_semantic_boundaries(
            active_tokens=self.active_tokens,
            rule_units=self.rule_units,
            raw_payload={"segments": [{"start_token": 0, "end_token": 11}]},
            options=options,
            clip_start_ms=0,
            clip_end_ms=10_000,
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "too_many_tokens")
        codes = {w.code for w in result.warnings}
        self.assertIn("semantic_segmentation_skipped_too_many_tokens", codes)
        self.assertEqual(next(w for w in result.warnings if w.code == "semantic_segmentation_skipped_too_many_tokens").severity, "info")

    def test_padding_preserved_after_refine(self):
        active = self.active_tokens + (
            _token(11, "9", 3, 0, "padding", 10_000, 11_000, intersects_clip=False, is_padding_only=True),
            _token(12, "9", 3, 1, "stuff", 11_000, 12_000, intersects_clip=False, is_padding_only=True),
            _token(13, "9", 3, 2, "here", 12_000, 12_500, intersects_clip=False, is_padding_only=True),
        )
        rule_units = (_rule_unit_translatable(), _padding_unit())
        payload = {"segments": [{"start_token": 0, "end_token": 11}]}
        result = refine_units_from_semantic_boundaries(
            active_tokens=active,
            rule_units=rule_units,
            raw_payload=payload,
            options=self.options,
            clip_start_ms=0,
            clip_end_ms=5000,
        )
        self.assertFalse(result.fallback_used)
        self.assertEqual(len(result.units), 2)
        self.assertTrue(result.units[-1].is_padding_only)
        self.assertEqual(result.units[-1].unit_id, "u002")

    def test_dangling_connector_split_can_avoid_in_terms(self):
        tokens = (
            _token(0, "1", 0, 0, "anything", 0, 200),
            _token(1, "1", 0, 1, "you", 200, 400),
            _token(2, "1", 0, 2, "can", 400, 600),
            _token(3, "1", 0, 3, "imagine", 600, 800),
            _token(4, "1", 0, 4, "is", 800, 1000),
            _token(5, "2", 1, 0, "in", 1000, 1100),
            _token(6, "2", 1, 1, "terms", 1100, 1300),
            _token(7, "2", 1, 2, "of", 1300, 1400),
            _token(8, "2", 1, 3, "navigating", 1400, 2000),
            _token(9, "2", 1, 4, "here", 2000, 2400),
        )
        rule_unit = SegmentUnit(
            unit_id="u001",
            source_text="anything you can imagine is in terms of navigating here",
            start_ms=0,
            end_ms=2400,
            source_spans=(SourceSpan(start=0, end=10, cue_id="1", cue_token_start=0, cue_token_end=10),),
            boundary_type=SegmentBoundaryType.INSIDE_CLIP,
            is_padding_only=False,
            crosses_clip_start=False,
            crosses_clip_end=False,
            intersects_clip=True,
        )
        payload = {"segments": [{"start_token": 0, "end_token": 4, "reason": "complete clause"}, {"start_token": 4, "end_token": 10, "reason": "transition"}]}
        result = refine_units_from_semantic_boundaries(
            active_tokens=tokens,
            rule_units=(rule_unit,),
            raw_payload=payload,
            options=self.options,
            clip_start_ms=0,
            clip_end_ms=3000,
        )
        self.assertFalse(result.fallback_used)
        last_words = [u.source_text.split()[-1].rstrip(".,!?") for u in result.units]
        for word in last_words:
            self.assertNotIn(word, {"in", "of", "to", "and", "or", "but", "for"})


if __name__ == "__main__":
    unittest.main()
