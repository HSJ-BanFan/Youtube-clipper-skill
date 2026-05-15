from __future__ import annotations

import hashlib
import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock
from time import perf_counter

from translation.batching import allocate_child_batch_ids, create_batches, split_batch
from translation.cache import CacheEntry, TranslationCache, build_batch_cache_key
from translation.config import TranslationConfig
from translation.context import build_global_context, write_global_context
from translation.glossary import load_glossary
from translation.models import BatchRecord, BatchState, Cue, CueRecord, ErrorType, MinimalBatchReportEntry, PipelineResult, TranslationBatch, TranslationOutputPaths
from translation.prompts import (
    QA_PROMPT_VERSION,
    PROMPT_VERSION,
    build_structured_translation_prompt,
    build_suspicious_qa_prompt,
    build_translation_prompt,
)
from translation.provider import OpenAICompatibleProvider, TranslationProvider
from translation.qa import QACandidate, find_suspicious_translations
from translation.report import QAStats, TranslationStats, write_translation_report
from translation.subtitles import (
    detect_subtitle_format,
    parse_subtitle_file,
    validate_translations,
    write_bilingual_srt,
    write_translated_srt,
)


@dataclass(frozen=True)
class BatchStatsDelta:
    provider_calls: int = 0
    fallback_provider_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0


@dataclass(frozen=True)
class BatchExecutionResult:
    translations: tuple[tuple[str, str], ...]
    stats_delta: BatchStatsDelta
    batch_entry: MinimalBatchReportEntry


class BatchExecutionFailed(RuntimeError):
    def __init__(self, error: RuntimeError) -> None:
        super().__init__(str(error))
        self.error = error


class BatchRoutingFailed(RuntimeError):
    def __init__(self, batch_id: int, error: Exception | None, error_type: ErrorType | None, attempts_used: int) -> None:
        detail = f": {error}" if error is not None else ""
        super().__init__(f"batch_id {batch_id} failed after {attempts_used} attempts{detail}")
        self.error = error
        self.error_type = error_type
        self.attempts_used = attempts_used


class ChildBatchIdAllocator:
    def __init__(self, next_child_batch_id: int) -> None:
        self._next_child_batch_id = next_child_batch_id
        self._lock = Lock()

    def allocate_pair(self) -> tuple[int, int]:
        with self._lock:
            left_child_id, right_child_id, next_child_batch_id = allocate_child_batch_ids(self._next_child_batch_id)
            self._next_child_batch_id = next_child_batch_id
            return left_child_id, right_child_id


_CACHE_ACCESS_LOCK = Lock()
_INVALID_JSON_MESSAGE = "translation response is not valid JSON"
SHRINK_BATCH_MAX_SPLIT_ATTEMPTS = 2
SHRINK_BATCH_STRATEGY_VERSION = "v1"
SHRINK_ELIGIBLE_ERROR_TYPES = {
    ErrorType.INVALID_JSON,
    ErrorType.SCHEMA_MISMATCH,
    ErrorType.MISSING_REQUIRED_CUE_ID,
    ErrorType.DUPLICATE_CUE_ID,
    ErrorType.INVALID_CUE_ID,
    ErrorType.CONTEXT_CUE_OUTPUT_VIOLATION,
    ErrorType.EMPTY_TRANSLATION,
}


def run_translation_pipeline(subtitle_path: str | Path, config: TranslationConfig) -> PipelineResult:
    input_path = Path(subtitle_path)
    if not input_path.exists():
        raise FileNotFoundError(f"subtitle_path not found: {input_path}")

    input_format = detect_subtitle_format(input_path)
    output_paths = build_output_paths(input_path, config)
    _ensure_outputs_do_not_exist(output_paths, config.overwrite)
    cues = parse_subtitle_file(input_path)

    if config.dry_run:
        return PipelineResult(
            input_path=input_path,
            input_format=input_format,
            output_format="srt",
            output_paths=output_paths,
            dry_run=True,
            cue_count=len(cues),
            provider_called=False,
            first_cue_preview=_preview_cue(cues[0]),
            last_cue_preview=_preview_cue(cues[-1]),
        )

    if not config.api_key:
        raise ValueError("TRANSLATION_API_KEY is required. Set it as an environment variable or provide it via --env-file.")

    glossary = load_glossary(config.glossary_path)
    global_context = build_global_context(cues, input_path, config)

    batches = create_batches(cues, config.batch_size, config.context_before, config.context_after)
    stats = TranslationStats(total_batches=len(batches))
    all_translations: dict[str, str] = {}

    batch_results = _run_translation_batches(batches, config, glossary.text, glossary.hash, global_context.text, global_context.hash)
    for batch_result in batch_results:
        _merge_batch_execution_result(stats, all_translations, batch_result)

    final_translations = _run_suspicious_qa(cues, all_translations, config, glossary.text, global_context.text, None, stats)
    validate_translations(cues, final_translations)
    write_translated_srt(cues, final_translations, output_paths.translated_srt)
    write_bilingual_srt(cues, final_translations, output_paths.bilingual_srt)
    write_global_context(global_context, output_paths.global_context)

    result = PipelineResult(
        input_path=input_path,
        input_format=input_format,
        output_format="srt",
        output_paths=output_paths,
        dry_run=False,
        cue_count=len(cues),
        provider_called=stats.provider_calls > 0,
        first_cue_preview=_preview_cue(cues[0]),
        last_cue_preview=_preview_cue(cues[-1]),
    )
    write_translation_report(output_paths.translation_report, result, config, stats, glossary, global_context)
    return result


def _run_translation_batches(
    batches: list[TranslationBatch] | tuple[TranslationBatch, ...],
    config: TranslationConfig,
    glossary_text: str,
    glossary_hash: str,
    global_context_text: str,
    global_context_hash: str,
) -> list[BatchExecutionResult]:
    child_batch_id_allocator = _create_child_batch_id_allocator(batches)
    if config.concurrency == 1:
        ordered_results: list[BatchExecutionResult] = []
        for batch in batches:
            try:
                ordered_results.append(
                    _execute_translation_batch(
                        batch,
                        config,
                        glossary_text,
                        glossary_hash,
                        global_context_text,
                        global_context_hash,
                        child_batch_id_allocator,
                    )
                )
            except BatchExecutionFailed as failure:
                _raise_batch_runtime_error(failure.error)
        return ordered_results

    cancellation_event = Event()
    executor = ThreadPoolExecutor(max_workers=min(config.concurrency, len(batches)))
    wait_for_shutdown = True
    try:
        futures = {
            executor.submit(
                _execute_translation_batch,
                batch,
                config,
                glossary_text,
                glossary_hash,
                global_context_text,
                global_context_hash,
                child_batch_id_allocator,
                cancellation_event,
            ): batch_index
            for batch_index, batch in enumerate(batches)
        }
        ordered_results: list[BatchExecutionResult | None] = [None] * len(futures)
        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                ordered_results[batch_index] = future.result()
            except BatchExecutionFailed as failure:
                cancellation_event.set()
                wait_for_shutdown = False
                for pending_future in futures:
                    if pending_future is not future:
                        pending_future.cancel()
                _raise_batch_runtime_error(failure.error)
            except Exception:
                cancellation_event.set()
                wait_for_shutdown = False
                for pending_future in futures:
                    if pending_future is not future:
                        pending_future.cancel()
                raise
        if any(result is None for result in ordered_results):
            raise RuntimeError("batch result was None -- this should not happen")
        return [result for result in ordered_results if result is not None]
    finally:
        executor.shutdown(wait=wait_for_shutdown, cancel_futures=not wait_for_shutdown)


def _execute_translation_batch(
    batch: TranslationBatch,
    config: TranslationConfig,
    glossary_text: str,
    glossary_hash: str,
    global_context_text: str,
    global_context_hash: str,
    child_batch_id_allocator: ChildBatchIdAllocator,
    cancellation_event: Event | None = None,
    split_attempt: int = 0,
) -> BatchExecutionResult:
    batch_started_at = perf_counter()
    translation_id_key = "id"
    structured_batch_record: BatchRecord | None = None
    if config.engine_version == "v2" and config.structured_output:
        translation_id_key = "cue_id"
        structured_batch_record = _build_structured_batch_record(batch)
        prompt = build_structured_translation_prompt(
            batch,
            config.target_lang,
            glossary_text,
            global_context_text,
            batch_record=structured_batch_record,
        )
    else:
        prompt = build_translation_prompt(
            batch,
            config.target_lang,
            glossary_text,
            global_context_text,
        )
    batch_source_hash = _build_batch_source_hash(prompt)
    cache_key = build_batch_cache_key(
        config.engine_version,
        config.structured_output,
        config.provider,
        config.base_url,
        config.model,
        config.main_model_alias,
        config.target_lang,
        PROMPT_VERSION,
        config.output_schema_version,
        config.batching_strategy_version,
        glossary_hash,
        global_context_hash,
        batch_source_hash,
    )
    local_stats = TranslationStats()

    cached_json = _read_cached_response(config, cache_key)
    if cached_json is not None:
        try:
            batch_translations = parse_translation_response(
                cached_json,
                batch.cues,
                batch.batch_id,
                translation_id_key=translation_id_key,
                batch_record=structured_batch_record,
            )
        except ValueError:
            local_stats.cache_misses += 1
        else:
            local_stats.cache_hits += 1
            ordered_translations = tuple((cue.id, batch_translations[cue.id]) for cue in batch.cues)
            return BatchExecutionResult(
                translations=ordered_translations,
                stats_delta=_stats_delta_from_stats(local_stats),
                batch_entry=MinimalBatchReportEntry(
                    batch_id=batch.batch_id,
                    state=BatchState.SUCCESS,
                    cue_count=len(batch.cues),
                    attempts=0,
                    cache_hit=True,
                    cue_range=(batch.cues[0].index, batch.cues[-1].index),
                    attempt=0,
                    error_type=None,
                    duration_ms=_duration_ms(batch_started_at),
                    final_route_label="main",
                ),
            )
    elif config.cache_enabled:
        local_stats.cache_misses += 1

    provider = OpenAICompatibleProvider(config)
    try:
        response_text, batch_translations, batch_error_type, final_route_label, attempts_used = _translate_batch_with_retries(
            provider,
            prompt,
            batch,
            config,
            local_stats,
            translation_id_key=translation_id_key,
            batch_record=structured_batch_record,
        )
    except BatchRoutingFailed as error:
        if config.engine_version == "v2" and config.structured_output and _should_shrink_batch(batch, error.error_type, split_attempt):
            return _execute_shrunk_batch(
                batch,
                config,
                glossary_text,
                glossary_hash,
                global_context_text,
                global_context_hash,
                child_batch_id_allocator,
                local_stats,
                batch_started_at,
                error.error_type,
                split_attempt,
                cancellation_event,
            )
        raise BatchExecutionFailed(error) from error
    except RuntimeError as error:
        raise BatchExecutionFailed(error) from error

    if final_route_label == "main" and (cancellation_event is None or not cancellation_event.is_set()):
        _write_cached_response(
            config,
            CacheEntry(
                cache_key=cache_key,
                engine_version=config.engine_version,
                structured_output=config.structured_output,
                provider=config.provider,
                base_url=config.base_url,
                model=config.model,
                main_model_alias=config.main_model_alias,
                target_lang=config.target_lang,
                prompt_version=PROMPT_VERSION,
                output_schema_version=config.output_schema_version,
                batching_strategy_version=config.batching_strategy_version,
                glossary_hash=glossary_hash,
                context_hash=global_context_hash,
                batch_source_hash=batch_source_hash,
                result_json=response_text,
            ),
        )

    ordered_translations = tuple((cue.id, batch_translations[cue.id]) for cue in batch.cues)
    return BatchExecutionResult(
        translations=ordered_translations,
        stats_delta=_stats_delta_from_stats(local_stats),
        batch_entry=MinimalBatchReportEntry(
            batch_id=batch.batch_id,
            state=BatchState.SUCCESS,
            cue_count=len(batch.cues),
            attempts=attempts_used,
            cache_hit=False,
            cue_range=(batch.cues[0].index, batch.cues[-1].index),
            attempt=attempts_used,
            error_type=batch_error_type,
            duration_ms=_duration_ms(batch_started_at),
            final_route_label=final_route_label,
        ),
    )


def _read_cached_response(config: TranslationConfig, cache_key: str) -> str | None:
    if not config.cache_enabled:
        return None
    with _CACHE_ACCESS_LOCK:
        with TranslationCache(config.cache_path) as cache:
            return cache.get(cache_key)


def _write_cached_response(config: TranslationConfig, entry: CacheEntry) -> None:
    if not config.cache_enabled:
        return
    with _CACHE_ACCESS_LOCK:
        with TranslationCache(config.cache_path) as cache:
            cache.set(entry)


def _execute_shrunk_batch(
    batch: TranslationBatch,
    config: TranslationConfig,
    glossary_text: str,
    glossary_hash: str,
    global_context_text: str,
    global_context_hash: str,
    child_batch_id_allocator: ChildBatchIdAllocator,
    parent_stats: TranslationStats,
    batch_started_at: float,
    split_error_type: ErrorType | None,
    split_attempt: int,
    cancellation_event: Event | None,
) -> BatchExecutionResult:
    left_child_id, right_child_id = child_batch_id_allocator.allocate_pair()
    left_child, right_child = split_batch(batch, left_child_id=left_child_id, right_child_id=right_child_id)
    child_results = (
        _execute_translation_batch(
            left_child,
            config,
            glossary_text,
            glossary_hash,
            global_context_text,
            global_context_hash,
            child_batch_id_allocator,
            cancellation_event,
            split_attempt + 1,
        ),
        _execute_translation_batch(
            right_child,
            config,
            glossary_text,
            glossary_hash,
            global_context_text,
            global_context_hash,
            child_batch_id_allocator,
            cancellation_event,
            split_attempt + 1,
        ),
    )
    parent_attempts = parent_stats.provider_calls
    total_attempts = parent_attempts + sum(result.batch_entry.attempts for result in child_results)
    cue_range = (batch.cues[0].index, batch.cues[-1].index)
    return BatchExecutionResult(
        translations=_merge_child_translations_in_parent_order(batch, child_results),
        stats_delta=_combine_stats_deltas(
            _stats_delta_from_stats(parent_stats),
            *(result.stats_delta for result in child_results),
        ),
        batch_entry=MinimalBatchReportEntry(
            batch_id=batch.batch_id,
            state=BatchState.SUCCESS,
            cue_count=len(batch.cues),
            attempts=total_attempts,
            cache_hit=False,
            cue_range=cue_range,
            attempt=total_attempts,
            error_type=split_error_type,
            duration_ms=_duration_ms(batch_started_at),
            final_route_label="shrink",
            child_batch_ids=(left_child_id, right_child_id),
            split_reason=split_error_type.value if split_error_type is not None else "unknown",
            split_attempt=split_attempt + 1,
            split_strategy_version=SHRINK_BATCH_STRATEGY_VERSION,
            original_target_cue_range=cue_range,
        ),
    )


def _merge_batch_execution_result(
    stats: TranslationStats,
    all_translations: dict[str, str],
    batch_result: BatchExecutionResult,
) -> None:
    _merge_stats_delta(stats, batch_result.stats_delta)
    all_translations.update(batch_result.translations)
    stats.batch_entries.append(batch_result.batch_entry)


def _merge_stats_delta(stats: TranslationStats, stats_delta: BatchStatsDelta) -> None:
    stats.provider_calls += stats_delta.provider_calls
    stats.fallback_provider_calls += stats_delta.fallback_provider_calls
    stats.cache_hits += stats_delta.cache_hits
    stats.cache_misses += stats_delta.cache_misses
    stats.retries += stats_delta.retries


def _stats_delta_from_stats(stats: TranslationStats) -> BatchStatsDelta:
    return BatchStatsDelta(
        provider_calls=stats.provider_calls,
        fallback_provider_calls=stats.fallback_provider_calls,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        retries=stats.retries,
    )


def _combine_stats_deltas(*stats_deltas: BatchStatsDelta) -> BatchStatsDelta:
    return BatchStatsDelta(
        provider_calls=sum(stats_delta.provider_calls for stats_delta in stats_deltas),
        fallback_provider_calls=sum(stats_delta.fallback_provider_calls for stats_delta in stats_deltas),
        cache_hits=sum(stats_delta.cache_hits for stats_delta in stats_deltas),
        cache_misses=sum(stats_delta.cache_misses for stats_delta in stats_deltas),
        retries=sum(stats_delta.retries for stats_delta in stats_deltas),
    )


def _create_child_batch_id_allocator(
    batches: list[TranslationBatch] | tuple[TranslationBatch, ...],
) -> ChildBatchIdAllocator:
    next_child_batch_id = max((batch.batch_id for batch in batches), default=0) + 1
    return ChildBatchIdAllocator(next_child_batch_id)


def _merge_child_translations_in_parent_order(
    parent_batch: TranslationBatch,
    child_results: tuple[BatchExecutionResult, BatchExecutionResult],
) -> tuple[tuple[str, str], ...]:
    merged_translations: dict[str, str] = {}
    for child_result in child_results:
        merged_translations.update(child_result.translations)
    return tuple((cue.id, merged_translations[cue.id]) for cue in parent_batch.cues)


def _should_shrink_batch(
    batch: TranslationBatch,
    error_type: ErrorType | None,
    split_attempt: int,
) -> bool:
    if len(batch.cues) < 2:
        return False
    if split_attempt >= SHRINK_BATCH_MAX_SPLIT_ATTEMPTS:
        return False
    return error_type in SHRINK_ELIGIBLE_ERROR_TYPES


def _raise_batch_runtime_error(error: RuntimeError) -> None:
    raise RuntimeError(str(error)) from error


class StructuredTranslationError(ValueError):
    def __init__(self, batch_id: int, error_type: ErrorType, detail: str) -> None:
        super().__init__(f"batch_id {batch_id} {detail}")
        self.error_type = error_type



def parse_translation_response(
    response_text: str,
    expected_cues: list[Cue] | tuple[Cue, ...],
    batch_id: int,
    translation_id_key: str = "id",
    batch_record: BatchRecord | None = None,
) -> dict[str, str]:
    stripped = _strip_json_fence(response_text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        if batch_record is not None and translation_id_key == "cue_id":
            raise _structured_translation_error(batch_id, ErrorType.INVALID_JSON, _INVALID_JSON_MESSAGE) from exc
        raise ValueError(f"batch_id {batch_id} {_INVALID_JSON_MESSAGE}") from exc

    if batch_record is not None and translation_id_key == "cue_id":
        structured_translations = _parse_structured_translation_response(payload, batch_record, batch_id)
        return _reconcile_structured_translations(expected_cues, batch_record, structured_translations)

    if not isinstance(payload, list):
        raise ValueError(f"batch_id {batch_id} translation response must be a JSON array")
    if len(payload) != len(expected_cues):
        raise ValueError(
            f"batch_id {batch_id} translation count {len(payload)} does not match cue count {len(expected_cues)}"
        )

    translations: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"batch_id {batch_id} translation item must be an object")
        cue_id = item.get(translation_id_key)
        translation = item.get("translation")
        if not isinstance(cue_id, str):
            raise ValueError(f"batch_id {batch_id} translation item {translation_id_key} must be a string")
        if not isinstance(translation, str):
            raise ValueError(f"batch_id {batch_id} cue id {cue_id} translation must be a string")
        translations[cue_id] = translation

    try:
        _validate_batch_translations(expected_cues, translations)
    except ValueError as exc:
        raise ValueError(f"batch_id {batch_id} {exc}") from exc
    return translations


def _validate_batch_translations(cues: list[Cue] | tuple[Cue, ...], translations: dict[str, str]) -> None:
    if len(translations) != len(cues):
        raise ValueError(f"translation count {len(translations)} does not match cue count {len(cues)}")

    seen_ids: set[str] = set()
    for cue in cues:
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
        if cue.id not in translations:
            raise ValueError(f"missing translation for cue id {cue.id}")
        if not translations[cue.id].strip():
            raise ValueError(f"translation for cue id {cue.id} is empty")


def _reconcile_structured_translations(
    expected_cues: list[Cue] | tuple[Cue, ...],
    batch_record: BatchRecord,
    translations: dict[str, str],
) -> dict[str, str]:
    reconciled: dict[str, str] = {}
    for cue, cue_record in zip(expected_cues, batch_record.target_cues, strict=True):
        reconciled[cue.id] = translations[cue_record.cue_id]

    try:
        _validate_batch_translations(expected_cues, reconciled)
    except ValueError as exc:
        raise ValueError(f"batch_id {batch_record.batch_id} {exc}") from exc
    return reconciled



def _parse_structured_translation_response(
    payload: object,
    batch_record: BatchRecord,
    batch_id: int,
) -> dict[str, str]:
    if not isinstance(payload, list):
        raise _structured_translation_error(batch_id, ErrorType.SCHEMA_MISMATCH, "translation response must be a JSON array")
    if len(payload) != len(batch_record.target_cues):
        raise _structured_translation_error(
            batch_id,
            ErrorType.SCHEMA_MISMATCH,
            f"translation count {len(payload)} does not match cue count {len(batch_record.target_cues)}",
        )

    expected_item_keys = {"cue_id", "translation"}
    target_cue_ids = {cue.cue_id for cue in batch_record.target_cues}
    context_cue_ids = {cue.cue_id for cue in batch_record.context_before} | {
        cue.cue_id for cue in batch_record.context_after
    }
    translations: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise _structured_translation_error(batch_id, ErrorType.SCHEMA_MISMATCH, "translation item must be an object")
        if set(item) != expected_item_keys:
            raise _structured_translation_error(
                batch_id,
                ErrorType.SCHEMA_MISMATCH,
                "translation item keys must exactly match cue_id and translation",
            )
        cue_id = item["cue_id"]
        translation = item["translation"]
        if not isinstance(cue_id, str) or not cue_id.strip():
            raise _structured_translation_error(batch_id, ErrorType.MISSING_REQUIRED_CUE_ID, "translation item cue_id must be a non-empty string")
        if cue_id in translations:
            raise _structured_translation_error(batch_id, ErrorType.DUPLICATE_CUE_ID, f"duplicate cue_id {cue_id}")
        classification = _classify_structured_cue_id(
            batch_record,
            cue_id,
            target_cue_ids=target_cue_ids,
            context_cue_ids=context_cue_ids,
        )
        if classification is not None:
            raise _structured_translation_error(batch_id, classification, f"unexpected cue_id {cue_id}")
        if not isinstance(translation, str):
            raise _structured_translation_error(batch_id, ErrorType.SCHEMA_MISMATCH, f"cue_id {cue_id} translation must be a string")
        if not translation.strip():
            raise _structured_translation_error(batch_id, ErrorType.EMPTY_TRANSLATION, f"translation for cue_id {cue_id} is empty")
        translations[cue_id] = translation

    returned_cue_ids = set(translations)
    if returned_cue_ids != target_cue_ids:
        missing_cue_ids = sorted(target_cue_ids - returned_cue_ids)
        missing = missing_cue_ids[0] if missing_cue_ids else "unknown"
        raise _structured_translation_error(batch_id, ErrorType.MISSING_REQUIRED_CUE_ID, f"missing translation for cue_id {missing}")
    return translations


def _structured_translation_error(batch_id: int, error_type: ErrorType, detail: str) -> StructuredTranslationError:
    return StructuredTranslationError(batch_id, error_type, detail)


def parse_qa_response(
    response_text: str,
    candidates: list[QACandidate] | tuple[QACandidate, ...] | list[Cue] | tuple[Cue, ...],
) -> dict[str, str]:
    expected_ids = [candidate.cue.id if isinstance(candidate, QACandidate) else candidate.id for candidate in candidates]
    stripped = _strip_json_fence(response_text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError("QA response is not valid JSON") from exc

    if not isinstance(payload, list):
        raise RuntimeError("QA response must be a JSON array")
    if len(payload) != len(expected_ids):
        raise RuntimeError("QA response item count does not match candidate count")

    fixes: dict[str, str] = {}
    seen_ids: set[str] = set()
    for expected_id, item in zip(expected_ids, payload, strict=True):
        if not isinstance(item, dict):
            raise RuntimeError("QA response item must be an object")
        cue_id = item.get("id")
        action = item.get("action")
        translation = item.get("translation")
        reason = item.get("reason")
        if not isinstance(cue_id, str) or cue_id not in expected_ids:
            raise RuntimeError("QA response id does not match candidates")
        if cue_id in seen_ids:
            raise RuntimeError("QA response contains duplicate id")
        if cue_id != expected_id:
            raise RuntimeError("QA response id order does not match candidates")
        seen_ids.add(cue_id)
        if action not in {"keep", "fix"}:
            raise RuntimeError("QA response action must be keep or fix")
        if not isinstance(translation, str) or not translation.strip():
            raise RuntimeError("QA response translation must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError("QA response reason must be a non-empty string")
        if action == "fix":
            fixes[cue_id] = translation

    if seen_ids != set(expected_ids):
        raise RuntimeError("QA response ids do not match candidates")
    return fixes


def build_output_paths(input_path: Path, config: TranslationConfig) -> TranslationOutputPaths:
    if config.output_path:
        bilingual_srt = Path(config.output_path)
        output_dir = bilingual_srt.parent
    else:
        output_dir = Path(config.output_dir) if config.output_dir else input_path.parent / "translated"
        bilingual_srt = output_dir / "bilingual.srt"

    return TranslationOutputPaths(
        output_dir=output_dir,
        translated_srt=output_dir / f"translated.{config.target_lang}.srt",
        bilingual_srt=bilingual_srt,
        translation_report=output_dir / "translation_report.md",
        global_context=output_dir / "global_context.md",
    )


def _duration_ms(started_at: float) -> int:
    return max(int((perf_counter() - started_at) * 1000), 0)



def _run_suspicious_qa(
    cues: list[Cue],
    translations: dict[str, str],
    config: TranslationConfig,
    glossary_text: str,
    global_context_text: str,
    provider: TranslationProvider | None,
    stats: TranslationStats,
) -> dict[str, str]:
    qa_stats = QAStats(qa_mode=config.qa_mode)
    stats.qa = qa_stats
    if config.qa_mode != "suspicious-only":
        return translations

    candidates = find_suspicious_translations(cues, translations, config.target_lang)
    qa_stats.qa_candidates = len(candidates)
    qa_stats.issues = tuple(issue for candidate in candidates for issue in candidate.issues)
    qa_stats.qa_prompt_version = QA_PROMPT_VERSION
    if not candidates:
        return translations

    reviewer = provider or OpenAICompatibleProvider(config)
    prompt = build_suspicious_qa_prompt(candidates, config.target_lang, glossary_text, global_context_text)
    qa_stats.qa_reviewed = len(candidates)
    qa_stats.qa_provider_calls += 1
    try:
        response_text = reviewer.review_suspicious(prompt)
    except RuntimeError:
        qa_stats.qa_failed += 1
        qa_stats.qa_provider_failures += 1
        qa_stats.qa_skipped = len(candidates)
        return translations

    try:
        fixes = parse_qa_response(response_text, candidates)
    except RuntimeError:
        qa_stats.qa_failed += 1
        qa_stats.qa_parser_failures += 1
        qa_stats.qa_skipped = len(candidates)
        return translations
    qa_stats.qa_fixed = len(fixes)
    qa_stats.qa_kept = len(candidates) - len(fixes)
    return translations | fixes


def _translate_batch_with_retries(
    provider: TranslationProvider,
    prompt: str,
    batch: TranslationBatch,
    config: TranslationConfig,
    stats: TranslationStats,
    translation_id_key: str = "id",
    batch_record: BatchRecord | None = None,
) -> tuple[str, dict[str, str], ErrorType | None, str, int]:
    def _attempt_route(active_provider: TranslationProvider) -> tuple[str, dict[str, str]]:
        response_text = active_provider.translate_batch(prompt)
        translations = parse_translation_response(
            response_text,
            batch.cues,
            batch.batch_id,
            translation_id_key=translation_id_key,
            batch_record=batch_record,
        )
        return response_text, translations

    main_attempts = _attempt_count(config)
    total_attempts = 0
    last_error: Exception | None = None
    last_error_type: ErrorType | None = None
    result: tuple[str, dict[str, str], ErrorType | None, str, int] | None = None

    for _ in range(main_attempts):
        total_attempts += 1
        try:
            stats.provider_calls += 1
            response_text, translations = _attempt_route(provider)
            result = (response_text, translations, last_error_type, "main", total_attempts)
            break
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            last_error_type = _classify_translation_error(exc)

    if result is None and _should_use_fallback_route(last_error_type, config):
        fallback_provider = OpenAICompatibleProvider(replace(config, model=config.fallback_model))
        total_attempts += 1
        try:
            stats.provider_calls += 1
            stats.fallback_provider_calls += 1
            response_text, translations = _attempt_route(fallback_provider)
            result = (response_text, translations, last_error_type, "fallback", total_attempts)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if last_error_type is None:
                last_error_type = _classify_translation_error(exc)

    stats.retries += max(total_attempts - 1, 0)
    if result is not None:
        return result

    raise BatchRoutingFailed(batch.batch_id, last_error, last_error_type, total_attempts) from last_error


def _classify_translation_error(error: RuntimeError | ValueError) -> ErrorType | None:
    error_type = getattr(error, "error_type", None)
    if isinstance(error_type, ErrorType):
        return error_type
    if str(error).endswith(_INVALID_JSON_MESSAGE):
        return ErrorType.INVALID_JSON
    return None


PROVIDER_FALLBACK_ERROR_TYPES = {
    ErrorType.PROVIDER_TIMEOUT,
    ErrorType.PROVIDER_HTTP_5XX,
    ErrorType.PROVIDER_REQUEST_FAILED,
    ErrorType.PROVIDER_MISSING_CHOICES,
}


FALLBACK_ELIGIBLE_ERROR_TYPES = PROVIDER_FALLBACK_ERROR_TYPES | {ErrorType.INVALID_JSON}


def _should_use_fallback_route(error_type: ErrorType | None, config: TranslationConfig) -> bool:
    if not config.fallback_model:
        return False
    return error_type in FALLBACK_ELIGIBLE_ERROR_TYPES


def _build_structured_batch_record(batch: TranslationBatch) -> BatchRecord:
    cues_in_namespace = batch.context_before + batch.cues + batch.context_after
    stable_id_counts: dict[str, int] = {}
    for cue in cues_in_namespace:
        candidate = cue.id.strip()
        if candidate:
            stable_id_counts[candidate] = stable_id_counts.get(candidate, 0) + 1

    cue_records: dict[Cue, CueRecord] = {}
    for cue in cues_in_namespace:
        candidate = cue.id.strip()
        cue_id = candidate if candidate and stable_id_counts[candidate] == 1 else str(cue.index)
        cue_records[cue] = CueRecord(
            cue_id=cue_id,
            original_index=cue.index,
            start=cue.start,
            end=cue.end,
            source_text=cue.source,
            raw_timing=cue.raw_timing,
            note=cue.note,
        )

    generated_cue_ids = {record.cue_id for record in cue_records.values()}
    if len(generated_cue_ids) != len(cue_records):
        raise ValueError(f"batch_id {batch.batch_id} generated non-unique structured cue_ids")

    return BatchRecord(
        batch_id=batch.batch_id,
        target_cues=tuple(cue_records[cue] for cue in batch.cues),
        context_before=tuple(cue_records[cue] for cue in batch.context_before),
        context_after=tuple(cue_records[cue] for cue in batch.context_after),
        status=BatchState.PENDING,
    )


def _classify_structured_cue_id(
    batch_record: BatchRecord,
    cue_id: str,
    target_cue_ids: set[str] | None = None,
    context_cue_ids: set[str] | None = None,
) -> ErrorType | None:
    resolved_target_cue_ids = target_cue_ids or {cue.cue_id for cue in batch_record.target_cues}
    if cue_id in resolved_target_cue_ids:
        return None

    resolved_context_cue_ids = context_cue_ids or ({cue.cue_id for cue in batch_record.context_before} | {
        cue.cue_id for cue in batch_record.context_after
    })
    if cue_id in resolved_context_cue_ids:
        return ErrorType.CONTEXT_CUE_OUTPUT_VIOLATION
    return ErrorType.INVALID_CUE_ID


def _build_batch_source_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _attempt_count(config: TranslationConfig) -> int:
    return config.max_retries + 1


def _strip_json_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines[0] in {"```", "```json"} and lines[-1] == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _ensure_outputs_do_not_exist(paths: TranslationOutputPaths, overwrite: bool) -> None:
    if overwrite:
        return
    candidates = [paths.translated_srt, paths.bilingual_srt, paths.translation_report, paths.global_context]
    existing = [path for path in candidates if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output file already exists: {joined}. Add --overwrite to replace outputs.")


def _preview_cue(cue: Cue) -> str:
    safe_source = "".join(
        character
        for character in cue.source
        if character.isspace() or not unicodedata.category(character).startswith("C")
    )
    preview = " ".join(safe_source.split())
    if len(preview) <= 80:
        return preview
    return f"{preview[:77]}..."
