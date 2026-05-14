# Translation V2 Phase 3 Bounded Concurrency Plan

**Design only. No Phase 3 code in this step.**

## Prerequisite baseline

- Phase 2 final cache identity hardening landed at `2431da1`.
- Phase 3 starts from that baseline and must preserve all Phase 2 cache identity rules.

## Goal

Add bounded concurrency to Translation Engine V2 batch processing with deterministic output ordering and safe cache behavior, while keeping scope tightly limited.

## Allowed Phase 3 scope

### 1. New config surface

- Add `TRANSLATION_CONCURRENCY`.
- Treat it as fixed worker-pool size for Phase 3.
- Keep default behavior compatible with existing single-thread or sequential baseline unless explicitly configured otherwise.

### 2. Fixed bounded worker pool

- Use a fixed worker count.
- Workers consume whole batches, not partial batch fragments.
- Each batch is owned by exactly one worker for one provider call path.

### 3. Batch-exclusive consumption

- No two workers may process same batch concurrently.
- Batch-level ownership must remain explicit and simple.
- No adaptive resizing, speculative retry redistribution, or batch splitting in Phase 3.

### 4. Deterministic aggregation

- Aggregate completed batch outputs deterministically.
- Preserve final subtitle ordering by original cue order.
- For structured mode, preserve deterministic aggregation keyed by `cue_id` while still emitting final output in original cue order.

### 5. Cache safety

- Final cache semantics stay unchanged:
  - cache read still re-validates through normal parse path
  - invalid cache payload still fails closed
  - only valid provider output writes final cache
- Concurrency must not introduce duplicate final writes for same batch input during one run.
- Cache identity logic from Phase 2 must remain unchanged unless a Phase 3 spec explicitly broadens it.

### 6. Minimal report compatibility

- Keep existing report shape unless narrow additions are required for concurrent provider-call accounting.
- Any report change must remain minimal and backward-friendly for current tests.

## Explicit non-goals

Phase 3 does **not** include:

- adaptive concurrency
- retry router
- fallback or repair routing
- suspicious-only QA expansion
- shrink-batch compensation
- full report redesign
- provider rewrite
- prompt semantic rewrite
- default v1 behavior changes

## Implementation constraints for future Phase 3 work

- Keep `translate_subtitles_v2.py` thin CLI glue.
- Prefer changes inside `translation/` package only.
- Do not widen scope into unrelated provider, prompt, or reporting work.
- Preserve default v1 behavior.
- Preserve structured/unstructured gating rules already present.

## Required future test plan for Phase 3 implementation

### Unit-level

- config parsing for `TRANSLATION_CONCURRENCY`
- worker-pool sizing behavior
- deterministic aggregation order
- batch ownership invariants

### Pipeline-level

- sequential baseline still passes when concurrency is default or `1`
- concurrent run with `TRANSLATION_CONCURRENCY > 1` preserves output ordering
- cache hit path still avoids provider call
- malformed cached payload still falls back cleanly under concurrent execution
- structured mode preserves `cue_id` mapping and final order

### Negative coverage

- no duplicate batch processing
- no duplicate final cache writes per batch in one run
- no report regression beyond minimal allowed deltas

## Readiness gate

Phase 3 can start only as **bounded concurrency implementation** under this document. Any request to add adaptive concurrency, retry routing, fallback or repair, suspicious-only QA expansion, shrink-batch compensation, or broader report redesign requires a new spec before code work starts.