# Translation V2 Phase 2 Cache Closure

**Phase 2 Complete**

## Landed change

- Commit: `2431da1`
- Subject: `feat: harden translation cache identity`
- Branch state at closure check: `main` ahead of `origin/main` by 22 commits, working tree clean
- Integration target for this repo remains fork-first workflow per repo guidance

## Actual completed scope

### Code changes shipped

- `translation/cache.py`
  - Expanded batch final cache identity to include:
    - `engine_version`
    - `structured_output`
    - `provider`
    - `base_url`
    - `model`
    - `main_model_alias`
    - `target_lang`
    - `prompt_version`
    - `output_schema_version`
    - `batching_strategy_version`
    - `glossary_hash`
    - `context_hash`
    - `batch_source_hash`
  - Stored Phase 2 identity metadata in sqlite rows.
  - Added additive-only schema migration for older cache files with missing Phase 2 columns.

- `translation/pipeline.py`
  - Wired live runtime config into final batch cache key construction.
  - Wired same identity fields into `CacheEntry` on successful final cache write.
  - Kept parser-based cache re-validation on read.
  - Kept fail-closed write behavior: only valid provider output reaches final cache.

### Test changes shipped

- `tests/translation_cache_cases.py`
  - Added direct key-identity assertions for all required included fields.
  - Added assertions that excluded knobs do not affect key shape.
  - Added additive migration coverage from Phase 1 cache schema.
  - Kept no-secret sqlite assertions.

- `tests/translation_pipeline_cases.py`
  - Added runtime isolation checks so cache does not cross-reuse between:
    - v1 and v2 structured paths
    - structured_output false and true
    - different `base_url` values
  - Kept malformed-cache and invalid-provider fail-closed coverage passing.

## Explicitly not included

Phase 2 closure does **not** include:

- bounded or adaptive concurrency
- retry router
- fallback or repair routing
- suspicious-only QA expansion
- shrink-batch compensation
- full report redesign
- provider rewrite
- prompt semantic rewrite
- default v1 behavior changes
- any new Phase 3 runtime code

## Verification run used for closure

### Git state

Commands:

```powershell
git status --short --branch
git log --oneline -5
```

Observed:

```text
## main...origin/main [ahead 22]
2431da1 feat: harden translation cache identity
61f8ad4 Merge pull request #10 from HSJ-BanFan/feat/translation-v2-phase0-1
9b81bd0 fix: reject structured cue id collisions
706b0ce fix: tighten translation v2 schema validation
abc1cab docs: mark translation v2 phase 0 and 1 complete
```

### Test evidence from implementation pass

Commands that passed during Phase 2 implementation:

```powershell
python -m pytest -q tests/translation_cache_cases.py
python -m pytest -q tests/translation_pipeline_cases.py
python -m pytest -q
```

Recorded results from implementation pass:

```text
148 passed, 23 subtests passed in 4.05s
46 passed in 4.26s
```

## Minimal smoke test plan

Run from repo root after setting translation env vars.

### 1. Default v1 path

Goal: confirm default path still behaves as pre-Phase-2 baseline.

- Use default `TRANSLATION_ENGINE_VERSION=v1` behavior.
- Run `scripts/translate_subtitles_v2.py` on small fixture subtitle file.
- Verify translated output generates successfully.
- Verify no structured-only assumptions appear in output or report.

### 2. V2 structured path

Goal: confirm structured gate still works with widened cache identity.

- Set `TRANSLATION_ENGINE_VERSION=v2`.
- Set `TRANSLATION_STRUCTURED_OUTPUT=true`.
- Run same subtitle fixture.
- Verify structured output parses and final output file is produced.

### 3. Cache hit

Goal: confirm second identical run hits final cache.

- Run same command twice with unchanged env and subtitle input.
- Verify second run reports cache hit behavior.
- Verify provider call count does not increase on second run.

### 4. Cache miss on identity change

Goal: confirm widened identity invalidates stale entries.

- Seed cache with one successful run.
- Change `TRANSLATION_BASE_URL` and rerun.
- Restore base URL, change `output_schema_version`, and rerun.
- Verify both changes cause cache miss and fresh provider call.

## Closure status

Phase 2 cache hardening is complete at commit `2431da1`. Next allowed work is Phase 3 design or later implementation scoped strictly to bounded concurrency only.