# Translation V2 Phase 0-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver strict, low-risk Phase 0 and Phase 1 rollout for Translation V2 that preserves current runtime behavior by default and only enables `cue_id` structured output behind explicit feature gate.

**Architecture:** Phase 0 adds passive data/config/report scaffolding only. Phase 1 adds minimal array-based structured parsing, `cue_id` reconciliation, and minimal per-batch reporting, but only when `TRANSLATION_ENGINE_VERSION=v2` and `TRANSLATION_STRUCTURED_OUTPUT=true`. Existing provider behavior, retry count, QA flow, cache key format, and serial runtime flow remain unchanged.

**Tech Stack:** Python, dataclasses, existing translation modules under `translation/`, unittest/pytest-style command execution, markdown report output.

---

## Implementation Overview

### Scope baseline

Current runtime baseline lives in:

- `translation/pipeline.py`
- `translation/prompts.py`
- `translation/report.py`
- `translation/cache.py`
- `translation/provider.py`
- `scripts/translate_subtitles_v2.py`

### Phase 1 feature gate

Phase 1 behavior must be gated exactly as follows:

- Default path: `TRANSLATION_ENGINE_VERSION=v1`
  - Keep current prompt behavior.
  - Keep current parser behavior.
  - Keep current runtime flow.
  - Keep current output contract.
- Structured-output path: enable only when both are true:
  - `TRANSLATION_ENGINE_VERSION=v2`
  - `TRANSLATION_STRUCTURED_OUTPUT=true`
- Any other combination, including `v2 + false`, must stay on current non-structured path.

### Hard constraints

- Phase 0 must not change runtime translation behavior.
- Phase 1 only adds:
  - minimal array-based structured output parsing
  - `cue_id` reconciliation
  - minimal per-batch report entries
- Phase 1 must not add:
  - cache key redesign
  - retry policy changes
  - `retry.py`
  - fallback routing
  - repair routing
  - shrink-batch compensation
  - concurrency
  - provider rewrite
- Model output owns translation text only.
- Model output must not own `start`, `end`, `original_index`, or subtitle structure.
- Final merge must always use `cue_id` mapping plus original cue order.
- Repair and fallback remain separate concepts, but neither is introduced in Phase 1.

### Phase 0 and Phase 1 Done Criteria summary

| Phase | Scope | Done Criteria |
|---|---|---|
| 0 | Passive data/config scaffolding only | New types and config fields exist, defaults stay on current runtime path, existing translation behavior is unchanged, existing tests still pass, new tests prove no behavior drift. |
| 1 | Structured output + `cue_id` reconciliation + minimal report | Structured parser only activates behind `v2 + true` gate, minimal schema is enforced, invalid structured results do not enter existing final cache, final merge is deterministic by original cue order, retry count is unchanged, no new routing systems appear. |

---

## Module-by-module Change Plan

### `translation/models.py`

**Modify in Phase 0**

Purpose:

- Add canonical V2 record names and enums without replacing current runtime dataclasses.

Required additions:

- `CueRecord`
- `BatchRecord`
- `AttemptRecord`
- `BatchState`
- `FailureMode`
- `ErrorType`
- `MinimalBatchReportEntry`

Required rules:

- Keep existing `Cue`, `TranslatedCue`, `TranslationBatch`, `TranslationOutputPaths`, and `PipelineResult` available.
- Do not force runtime migration to new records in Phase 0.
- New records must support Phase 1 parser/report work without changing current `v1` runtime flow.

### `translation/config.py`

**Modify in Phase 0**

Purpose:

- Add passive config surface and exact Phase 1 gate.

Required additions:

- `engine_version`
- `structured_output`
- `failure_mode`
- `main_model_alias`
- `repair_model_alias`
- `fallback_model_alias`
- `batch_max_chars`
- `batch_max_cues`
- `output_schema_version`
- `batching_strategy_version`

Required env support:

- `TRANSLATION_ENGINE_VERSION`
- `TRANSLATION_STRUCTURED_OUTPUT`
- `TRANSLATION_FAILURE_MODE`
- `TRANSLATION_MAIN_MODEL_ALIAS`
- `TRANSLATION_REPAIR_MODEL_ALIAS`
- `TRANSLATION_FALLBACK_MODEL_ALIAS`
- `TRANSLATION_BATCH_MAX_CHARS`
- `TRANSLATION_BATCH_MAX_CUES`
- `TRANSLATION_OUTPUT_SCHEMA_VERSION`
- `TRANSLATION_BATCHING_STRATEGY_VERSION`

Required rules:

- Default `TRANSLATION_ENGINE_VERSION` must be `v1`.
- Default `TRANSLATION_STRUCTURED_OUTPUT` must be `false`.
- Config parsing must not by itself change runtime behavior.
- Safe config output may expose these values, but no secrets.

### `translation/prompts.py`

**Modify in Phase 1**

Purpose:

- Add minimal structured output prompt contract only for gated Phase 1 path.

Structured-output path requirements:

- Model output schema must be exactly:
  - `[{"cue_id": "string", "translation": "string"}]`
- Do not add schema wrapper.
- Do not require `schema_version` in model output.
- `schema_version` remains config/report/cache-identity metadata only.
- Prompt must explicitly forbid output of:
  - `start`
  - `end`
  - `original_index`
  - context cue translations

Non-structured path requirements:

- Existing prompt text and current parser contract remain unchanged.

### `translation/pipeline.py`

**Modify in Phase 1**

Purpose:

- Add gated structured parser path, `cue_id` reconciliation, and minimal report collection.

Required behavior split:

- If `engine_version != v2` or `structured_output != true`:
  - use current prompt path
  - use current parser path
  - use current runtime path
- If `engine_version == v2` and `structured_output == true`:
  - use minimal `cue_id` array prompt
  - parse structured array response
  - classify parser/report `error_type`
  - reconcile by `cue_id`
  - write minimal per-batch report entries

Required `cue_id` rules:

- Prefer reusing current `Cue.id` when stable.
- Otherwise derive `cue_id` from `original_index` as string.
- `target_cues`, `context_before`, and `context_after` share one `cue_id` namespace per subtitle document.
- If returned `cue_id` belongs to context cue set, classify as `context_cue_output_violation`.
- If returned `cue_id` is unknown and not in context cue set, classify as `invalid_cue_id`.

Required parser/report-only error handling in Phase 1:

- `invalid_json`
- `schema_mismatch`
- `missing_required_cue_id`
- `duplicate_cue_id`
- `invalid_cue_id`
- `context_cue_output_violation`
- `empty_translation`

Required limits:

- Phase 1 `error_type` is for parser/report classification only.
- Do not change retry count.
- Do not change retry loop structure.
- Do not add routing by `error_type`.
- Do not add `retry.py`.
- Do not add fallback.
- Do not add repair.
- Do not add shrink-batch.

Required merge rules:

- Build map by `cue_id`.
- Validate exact key-set match against target cue IDs.
- Rebuild batch result by iterating original target cue order.
- Merge chapter result by iterating original subtitle cue order.

### `translation/report.py`

**Modify in Phase 1**

Purpose:

- Add minimal per-batch structured-output observability without full routing/report redesign.

Required additions:

- `TranslationStats.batch_entries`
- minimal `## Batch Results` section with fields:
  - `batch_id`
  - `cue_range`
  - `status`
  - `attempt`
  - `error_type`
  - `cache_hit`
  - `duration_ms`

Required limits:

- Keep current aggregate report fields.
- Keep current QA section.
- Do not add full Phase 7 report fields yet.
- Do not introduce fallback or repair counters in Phase 1.

### `translation/cache.py`

**Do not redesign in Phase 1**

Required Phase 1 rule:

- Keep current cache key design unchanged.
- Keep current cache read path unchanged.
- Keep current cache storage format unchanged.
- But when structured-output path is active, any response that structured parser marks invalid must not be written into existing final cache.

Explicit anti-scope rule:

- No cache key redesign until later phase.

### `translation/subtitles.py`

**No Phase 0/1 behavior change required**

Required invariant:

- Output writing continues to use source cue order and source timing only.
- Model-owned fields never include timing or original index.

### `translation/provider.py`

**No Phase 0/1 change required**

Required invariant:

- No provider rewrite.
- No provider-specific schema transport changes required for Phase 1.

### `scripts/translate_subtitles_v2.py`

**Light Phase 0/1 updates only if needed**

Required limits:

- Keep CLI thin.
- Do not move orchestration logic here.
- If dry-run exposes new config flags, expose them safely only.
- Default CLI execution must still behave as current path because default engine is `v1`.

---

## Data Model Spec

### `CueRecord`

Canonical translation unit.

Fields:

- `cue_id: str`
- `original_index: int`
- `start: str`
- `end: str`
- `source_text: str`
- `raw_timing: str | None`
- `note: str | None`

Rules:

- Prefer `Cue.id` as `cue_id` when current parser output is stable.
- Otherwise derive `cue_id = str(original_index)`.
- `cue_id` namespace is shared across target and context cues in one subtitle document.
- `start`, `end`, and `original_index` are source-owned fields only.

### `BatchRecord`

Fields:

- `batch_id: str`
- `target_cues: tuple[CueRecord, ...]`
- `context_before: tuple[CueRecord, ...]`
- `context_after: tuple[CueRecord, ...]`
- `status: BatchState`
- `parent_batch_id: str | None`
- `child_batch_ids: tuple[str, ...]`
- `split_reason: str | None`
- `split_attempt: int`
- `original_target_cue_range: tuple[int, int] | None`

Rules:

- `target_cues` define exact allowed output key set.
- `context_before` and `context_after` are read-only reference sets.
- Context cue IDs appearing in model output are invalid and must classify as `context_cue_output_violation`.

### `AttemptRecord`

Fields:

- `attempt_index: int`
- `batch_id: str`
- `model_alias: str`
- `error_type: ErrorType | None`
- `duration_ms: int | None`
- `cache_hit: bool`
- `result_state: BatchState`

Phase 1 rule:

- Used for observability/reporting only.
- Not yet used for routing policy.

### `ErrorType`

Phase 1 required parser/report set:

- `invalid_json`
- `schema_mismatch`
- `missing_required_cue_id`
- `duplicate_cue_id`
- `invalid_cue_id`
- `context_cue_output_violation`
- `empty_translation`

Phase 1 rule:

- Do not route actions from these values yet.
- Routing begins in Phase 4/5.

---

## JSON Output Schema

Phase 1 structured-output schema must be minimal array schema only.

```json
[
  {"cue_id": "string", "translation": "string"}
]
```

Validation requirements when gated path active:

1. Root must be JSON array.
2. Each item must be object.
3. Each item must contain exactly:
   - `cue_id`
   - `translation`
4. `cue_id` must be non-empty string.
5. `translation` must be non-empty string after trimming.
6. Item count must equal target cue count.
7. Returned `cue_id` set must equal target cue ID set exactly.
8. Duplicate `cue_id` is `duplicate_cue_id`.
9. Missing expected `cue_id` is `missing_required_cue_id`.
10. Context cue ID in output is `context_cue_output_violation`.
11. Unknown non-context cue ID is `invalid_cue_id`.
12. Final merge ignores model item order and rebuilds by original target cue order.

Phase 1 non-goals:

- no wrapper object
- no output `schema_version`
- no output `start`
- no output `end`
- no output `original_index`

---

## Error Decision Table

Phase 0/1 scope is classification only. Real routing starts later.

| error_type | Meaning in Phase 1 | Phase 1 action | Not allowed in Phase 1 | Routing phase |
|---|---|---|---|---|
| `invalid_json` | response not parseable as JSON array | fail parse, record in report, preserve current retry count | no custom retry policy, no fallback | 4 |
| `schema_mismatch` | root/item shape wrong | fail parse, record in report, preserve current retry count | no routing changes | 4 |
| `missing_required_cue_id` | expected target cue missing | fail parse, record in report, preserve current retry count | no shrink-batch | 4/5 |
| `duplicate_cue_id` | duplicated target key | fail parse, record in report, preserve current retry count | no routing changes | 4 |
| `invalid_cue_id` | unknown returned cue ID that is not context cue | fail parse, record in report, preserve current retry count | no fallback | 4 |
| `context_cue_output_violation` | returned cue ID belongs to context set | fail parse, record in report, preserve current retry count | no repair/fallback | 4 |
| `empty_translation` | blank translation text | fail parse, record in report, preserve current retry count | no custom routing | 4 |

Phase 1 rules:

- Error classification does not change retry count.
- Error classification does not change retry loop shape.
- Error classification does not introduce new routing.
- Real `error_type -> action` routing begins in Phase 4/5.

---

## Cache Identity Spec

### Phase 1 cache rule

- Do not redesign cache key in Phase 1.
- Do not add new cache identity fields in Phase 1 key computation.
- Keep current cache storage contract.
- But if structured-output path is active and parser marks response invalid, do not write that result into existing final cache.

### Future cache identity note

`schema_version` remains valid metadata for future config/report/cache identity work, but:

- it is not part of model output
- it does not require Phase 1 cache key redesign

---

## Phase 0 Tasks and Tests

**Status:** Complete on 2026-05-13.

**Closure evidence:**
- Default runtime stays on `v1 + false` path.
- Safe config output exposes new passive fields without leaking secrets.
- CLI compatibility preserved for existing arguments and dry-run flow.
- Verification suite passed: `tests/translation_pipeline_cases.py`, `tests/translation_models_cases.py`, `tests/translation_report_cases.py`, `tests/translation_config_cases.py`, `tests/translate_subtitles_v2_cli_cases.py`.


### Task 1: Add passive V2 records and enums

**Files:**
- Modify: `translation/models.py`
- Test: `tests/translation_pipeline_cases.py`
- Test: `tests/translation_report_cases.py`
- Test: `tests/translation_config_cases.py`

- [ ] Add canonical V2 data types and enums without removing current runtime dataclasses.
- [ ] Confirm current pipeline imports still resolve without runtime migration.
- [ ] Add tests that instantiate new records and confirm no current runtime path depends on them yet.

### Task 2: Add passive config surface and exact feature gate defaults

**Files:**
- Modify: `translation/config.py`
- Test: `tests/translation_config_cases.py`
- Test: `tests/translate_subtitles_v2_cli_cases.py`

- [ ] Add new env/config fields for `engine_version`, `structured_output`, and later-phase metadata.
- [ ] Set defaults to `v1` and `false`.
- [ ] Add validation for accepted values only.
- [ ] Add tests proving default config stays on current runtime path.
- [ ] Add tests proving `v2 + false` still stays on current runtime path.

### Task 3: Expose passive metadata safely only

**Files:**
- Modify: `translation/config.py`
- Modify: `translation/report.py` if needed
- Modify: `scripts/translate_subtitles_v2.py` if needed
- Test: `tests/translation_report_cases.py`
- Test: `tests/translate_subtitles_v2_cli_cases.py`

- [ ] Expose new flags in safe config output only if needed.
- [ ] Confirm no secret leakage.
- [ ] Confirm dry-run output remains current by default.

### Phase 0 Tests

Must keep passing:

- `python -m pytest tests/translation_config_cases.py -q`
- `python -m pytest tests/translation_pipeline_cases.py -q`
- `python -m pytest tests/translation_report_cases.py -q`
- `python -m pytest tests/translate_subtitles_v2_cli_cases.py -q`

Must add:

- `test_default_engine_version_is_v1`
- `test_default_structured_output_is_false`
- `test_v2_false_does_not_enable_structured_parser_path`
- `test_new_config_fields_do_not_change_default_runtime_behavior`
- `test_safe_config_output_does_not_leak_api_key_with_new_fields`

### Phase 0 Done Criteria

All must be true:

- Current default runtime path still uses current prompt contract.
- Current default runtime path still uses current parser contract.
- Current default runtime path still uses current retry count and loop shape.
- Current default runtime path still uses current cache behavior.
- Current default runtime path still uses current QA behavior.
- New config fields parse and validate cleanly.
- `v2 + false` does not activate structured path.
- Existing tests pass.
- New Phase 0 regression tests pass.

---

## Phase 1 Tasks and Tests

**Status:** Complete on 2026-05-13.

**Closure evidence:**
- Structured output contract activates only on `TRANSLATION_ENGINE_VERSION=v2` plus `TRANSLATION_STRUCTURED_OUTPUT=true`.
- `cue_id` prompt emission, schema validation, deterministic reconciliation, cache write guard, and minimal batch report are covered by regression tests.
- Default `v1` path and `v2 + false` path remain behaviorally unchanged.
- Verification suite passed: `tests/translation_pipeline_cases.py`, `tests/translation_models_cases.py`, `tests/translation_report_cases.py`, `tests/translation_config_cases.py`, `tests/translate_subtitles_v2_cli_cases.py`.


### Task 1: Add gated minimal structured-output prompt path

**Files:**
- Modify: `translation/prompts.py`
- Modify: `translation/pipeline.py`
- Test: `tests/translation_pipeline_cases.py`

- [ ] Add alternate prompt path that requires minimal `[{"cue_id", "translation"}]` array.
- [ ] Keep current prompt path untouched for non-gated path.
- [ ] Add tests proving structured prompt only activates on `v2 + true`.
- [ ] Add tests proving default `v1` path keeps current prompt text/contract.

### Task 2: Define stable `cue_id` generation and namespace rules

**Files:**
- Modify: `translation/pipeline.py`
- Modify: `translation/models.py` if helper names need alignment
- Test: `tests/translation_pipeline_cases.py`
- Test: `tests/subtitles_cases.py` if cue identity behavior is surfaced there

- [ ] Reuse current `Cue.id` when stable.
- [ ] Otherwise derive `cue_id = str(original_index)`.
- [ ] Ensure target and context cues use one shared namespace.
- [ ] Add tests for context cue vs unknown cue classification split.

### Task 3: Add gated structured parser and exact classification

**Files:**
- Modify: `translation/pipeline.py`
- Test: `tests/translation_pipeline_cases.py`

- [ ] Parse minimal array schema only on gated path.
- [ ] Classify `context_cue_output_violation` separately from `invalid_cue_id`.
- [ ] Keep `error_type` usage limited to parser/report classification.
- [ ] Confirm current retry count remains unchanged after failures.

### Task 4: Add deterministic `cue_id` reconciliation

**Files:**
- Modify: `translation/pipeline.py`
- Modify: `translation/subtitles.py` only if helper signatures need no-behavior-change adjustments
- Test: `tests/translation_pipeline_cases.py`

- [ ] Build translation map by `cue_id`.
- [ ] Rebuild batch output by original target cue order.
- [ ] Merge final chapter output by original subtitle order.
- [ ] Add regression test where model returns valid items in reversed order and output still matches source cue order.

### Task 5: Protect existing final cache from invalid structured results

**Files:**
- Modify: `translation/pipeline.py`
- Modify: `translation/cache.py` only if write guard placement needs helper extraction
- Test: `tests/translation_pipeline_cases.py`
- Test: `tests/translation_cache_cases.py`

- [ ] Keep current cache key logic unchanged.
- [ ] When structured parser path is active and parse fails, skip final cache write.
- [ ] Add regression test proving invalid structured payload never overwrites existing final cache.

### Task 6: Add minimal per-batch report section only

**Files:**
- Modify: `translation/report.py`
- Modify: `translation/pipeline.py`
- Test: `tests/translation_report_cases.py`
- Test: `tests/translation_pipeline_cases.py`

- [ ] Add minimal batch entries with exact required fields.
- [ ] Keep current aggregate and QA sections intact.
- [ ] Do not add fallback, repair, or shrink-batch fields.

### Phase 1 Tests

Must keep passing:

- `python -m pytest tests/translation_pipeline_cases.py -q`
- `python -m pytest tests/translation_report_cases.py -q`
- `python -m pytest tests/translation_config_cases.py -q`
- `python -m pytest tests/translate_subtitles_v2_cli_cases.py -q`

Must add:

- `test_structured_prompt_activates_only_on_v2_true`
- `test_default_v1_path_keeps_current_prompt_and_parser_behavior`
- `test_v2_false_keeps_current_prompt_and_parser_behavior`
- `test_parse_structured_response_accepts_minimal_cue_id_array`
- `test_parse_structured_response_rejects_missing_required_cue_id`
- `test_parse_structured_response_rejects_duplicate_cue_id`
- `test_parse_structured_response_classifies_context_cue_output_violation`
- `test_parse_structured_response_classifies_invalid_cue_id_for_unknown_non_context_id`
- `test_structured_merge_rebuilds_output_in_original_order`
- `test_invalid_structured_result_is_not_written_to_final_cache`
- `test_minimal_report_contains_batch_results_section`
- `test_phase1_error_type_does_not_change_retry_count`
- `test_phase1_does_not_add_fallback_or_repair_behavior`

### Phase 1 Done Criteria

All must be true:

- Structured-output path activates only on `TRANSLATION_ENGINE_VERSION=v2` plus `TRANSLATION_STRUCTURED_OUTPUT=true`.
- Default `v1` path is behaviorally unchanged.
- `v2 + false` path is behaviorally unchanged.
- Phase 1 parser accepts only minimal array schema.
- No schema wrapper appears in model output contract.
- No `schema_version` is required in model output.
- `cue_id` generation rule is deterministic and documented in tests.
- Context cue IDs in output classify as `context_cue_output_violation`.
- Unknown non-context cue IDs classify as `invalid_cue_id`.
- Final merge is deterministic by original cue order.
- Invalid structured results are not written to existing final cache.
- Retry count and retry loop shape are unchanged.
- No `retry.py`, fallback, repair, shrink-batch, or cache key redesign appears.
- Existing tests pass.
- New Phase 1 regression tests pass.

---

## Phase Gate / Anti-scope-creep Checklist

### Global

- [ ] No provider rewrite.
- [ ] No scheduler or concurrency work before Phase 3.
- [ ] No `retry.py` before Phase 4.
- [ ] No `error_type -> action` routing before Phase 4/5.
- [ ] No fallback before Phase 4.
- [ ] No shrink-batch before Phase 5.
- [ ] No repair path before Phase 6.
- [ ] No full report before Phase 7.
- [ ] No adaptive concurrency before Phase 8.

### Phase 0

- [ ] No prompt text change.
- [ ] No parser behavior change.
- [ ] No runtime translation flow change.
- [ ] No retry count change.
- [ ] No cache behavior change.
- [ ] No QA behavior change.
- [ ] Default config remains `v1 + false`.
- [ ] `v2 + false` remains on current path.

### Phase 1

- [ ] Structured path only behind `v2 + true`.
- [ ] Minimal array schema only.
- [ ] No schema wrapper.
- [ ] No model-owned `schema_version`.
- [ ] No model-owned `start`, `end`, or `original_index`.
- [ ] `error_type` used only for parser/report classification.
- [ ] No retry count change.
- [ ] No retry loop shape change.
- [ ] No `retry.py`.
- [ ] No fallback.
- [ ] No repair.
- [ ] No shrink-batch.
- [ ] No cache key redesign.
- [ ] Invalid structured results never enter existing final cache.
- [ ] Final merge uses `cue_id` mapping plus original cue order only.

### Merge and identity safety

- [ ] Prefer current stable `Cue.id` for `cue_id`.
- [ ] Otherwise derive `cue_id` from `original_index` string.
- [ ] Target and context cues share one namespace.
- [ ] Context cue output classifies as `context_cue_output_violation`.
- [ ] Unknown non-context cue output classifies as `invalid_cue_id`.
- [ ] Output writers continue using source cue timing and order.

---

## Execution Notes

This plan intentionally covers only Phase 0 and Phase 1. Do not pull in Phase 2+ work while implementing this file.
